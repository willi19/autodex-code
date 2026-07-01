#!/usr/bin/env python3
"""Interactive (B)-planning viewer: "does seeded planning succeed at an
arbitrary object placement?"

Sliders place the object (radius / yaw / azimuth) and pick a grasp. IK runs
LIVE on every slider move (it's cheap) to show reachability. The heavy step —
mechanism (B): nearest grid cell's CACHED canonical trajectory -> adjust_seed
-> plan_with_seed (trajopt seeded by the adjusted traj) -> play the approach —
runs only when you press the Plan button.

Azimuth is absorbed by the endpoint IK (joint-0), yaw/radius by the seed
adjustment, so the cached grid can be coarse. This tool is the eyeball test of
how far that holds.

    conda activate mingi
    python src/visualization/seed_plan_viz.py --obj pepsi --hand inspire_left --pose 2 --port 8080
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
from pathlib import Path

import numpy as np
import torch
import trimesh
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, "/home/mingi/AutoDex")
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "paradex"))

import src.visualization.exp as exp
from autodex.planner import GraspPlanner
from autodex.planner.planner import _to_curobo_world, _to_curobo_pose, _snap_joint6
from autodex.utils.conversion import se32cart
from autodex.utils.path import project_dir, obj_path
from src.grasp_generation.order.compute_order import load_grasp_data
import src.visualization.seed_cache as seed_cache_mod
from paradex.visualization.visualizer.viser import ViserViewer

URDF = exp.URDF_BY_HAND
YAW4 = np.linspace(0, 2 * np.pi, 4, endpoint=False)        # reduced yaw grid (90)
C_OK, C_FAIL, C_READY = (0.18, 0.78, 0.28), (0.85, 0.18, 0.16), (0.9, 0.7, 0.15)


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def wrap(d):
    return (d + np.pi) % (2 * np.pi) - np.pi


def adjust_seed(seed, q_start, q_goal):
    """JOINT-space endpoint re-anchor: pin seed[0]->q_start and seed[-1]->q_goal,
    ramp the corrections linearly. Cheap, but interior waypoints are NOT re-IK'd
    so the end-effector path can dip through the floor at large deltas."""
    H = len(seed)
    d0, d1 = wrap(q_start - seed[0]), wrap(q_goal - seed[-1])
    a = np.linspace(0, 1, H)[:, None]
    return seed + (1 - a) * d0 + a * d1


def _skew(w):
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def se3_log(T):
    w = Rot.from_matrix(T[:3, :3]).as_rotvec()
    th = np.linalg.norm(w)
    p = T[:3, 3]
    if th < 1e-8:
        return p.copy(), w
    W = _skew(w)
    Vinv = (np.eye(3) - 0.5 * W
            + (1.0 / th ** 2) * (1 - (th * np.sin(th)) / (2 * (1 - np.cos(th)))) * (W @ W))
    return Vinv @ p, w


def se3_exp(v, w):
    th = np.linalg.norm(w)
    T = np.eye(4)
    if th < 1e-8:
        T[:3, 3] = v
        return T
    W = _skew(w)
    T[:3, :3] = np.eye(3) + (np.sin(th) / th) * W + ((1 - np.cos(th)) / th ** 2) * (W @ W)
    V = np.eye(3) + ((1 - np.cos(th)) / th ** 2) * W + ((th - np.sin(th)) / th ** 3) * (W @ W)
    T[:3, 3] = V @ v
    return T


def se3_interp(T, a):
    """Screw interpolation: identity at a=0, T at a=1 (exp(a*log T))."""
    v, w = se3_log(T)
    return se3_exp(a * v, a * w)


def adjust_seed_task(planner, to_pose, seed, T_cell_obj, T_target_obj, q_start, q_goal,
                     fingers, n=48, gate=0.5):
    """TASK-space adjust: FK each canonical waypoint to its wrist pose, apply the
    FULL object-pose delta (cell->target) in Cartesian from the START (not ramped
    — so the whole approach is shaped for the ACTUAL object), then RE-IK every
    waypoint SEQUENTIALLY, warm-started from the previous solution so it stays on
    the same branch.

    NO fallback: a waypoint is solved by warm-start (num_seeds=1 from prev); if that
    misses, ONE random retry (the solver's full seed set, biased to prev). If both
    fail OR the step exceeds `gate` (a genuine branch discontinuity), the WHOLE
    config is declared a task-fail and (None, None) is returned. A hybrid
    task/joint-blend seed is never produced.

    Returns (out, goal_qpos) on success, else (None, None). goal_qpos is the seed's
    own last config (a valid IK of the target grasp), used as plan_with_seed's
    goal_state so there is no endpoint seam between the seed and the goal."""
    dev = planner._tensor_args.device
    lo, hi = exp._arm_limits(planner)
    H0 = len(seed)
    idx = np.linspace(0, H0 - 1, min(n, H0)).astype(int)
    sub = seed[idx]                                          # (n, dof)
    n = len(sub)

    st = planner._ik_solver.fk(torch.tensor(sub, dtype=torch.float32, device=dev))
    pos, quat = st.ee_position.cpu().numpy(), st.ee_quaternion.cpu().numpy()  # wxyz
    Wc = np.tile(np.eye(4), (n, 1, 1))
    Wc[:, :3, 3] = pos
    Wc[:, :3, :3] = Rot.from_quat(quat[:, [1, 2, 3, 0]]).as_matrix()

    Tco = T_target_obj @ np.linalg.inv(T_cell_obj)           # object pose delta (cell -> target)
    A = np.linspace(0.0, 1.0, n)                             # RAMP the delta: 0 at start, full at grasp
    fing = np.asarray(fingers, dtype=np.float32)

    def _solve(Wt, prev):
        # ONE solve: seed 0 warm-started at prev, the other 31 random (num_seeds=32).
        # return_seeds=8 gives the top solutions; pick the one CLOSEST to prev within
        # the gate (NOT the lowest-cost best, which may be a different branch). This
        # keeps the same IK branch when one exists but still explores 31 random seeds,
        # so a reachable waypoint rarely fails -> far fewer spurious None returns.
        pc = torch.tensor(np.concatenate([prev, fing]), dtype=torch.float32, device=dev)
        res = planner._ik_solver.solve_single(
            to_pose(Wt[None], dev), retract_config=pc[None],
            seed_config=pc[None, None], num_seeds=32, return_seeds=8)
        sols = res.solution.detach().cpu().numpy().reshape(-1, res.solution.shape[-1])
        succ = res.success.detach().cpu().numpy().reshape(-1)
        best, bd = None, gate
        for i in range(len(sols)):
            if not succ[i]:
                continue
            c = exp._unwrap_arm(sols[i][:6], prev, lo, hi)
            d = float(np.abs(c - prev).max())
            if d < bd:                                       # closest-to-prev within the gate
                bd, best = d, c
        return best

    out = np.zeros((n, len(q_start)), dtype=np.float32)
    prev = sub[0, :6].astype(np.float32)                     # (A) START AT THE SEED'S OWN START (j0 rotated
    for t in range(n):                                       #     by the build's dd), NOT at INIT. The robot
        Wt = se3_interp(Tco, A[t]) @ Wc[t]                   #     pre-rotates its base to here at runtime, so
        cand = _solve(Wt, prev)                              #     the seed's first wrist pose matches prev ->
        if cand is None:                                     #     NO t=0 branch jump (the old INIT-pin caused it).
            return None, None                                # genuine fail -> NO substitution
        out[t, :6] = cand
        prev = cand.astype(np.float32)
    out[:, 6:] = fing
    out[0, :6] = sub[0, :6]                                  # pin start to the seed's rotated start
    return out, out[-1].copy()


def _prepend_base_move(traj, q_init, n_pre=24):
    """Prepend a linear move from q_init (the real robot start = INIT) to traj[0]
    (the seed's rotated start). The delta is a pure base(+wrist) rotation of the
    retracted arm -- collision-free -- so the returned trajectory begins at the real
    INIT and the executor runs it unchanged. No-op if traj already starts at q_init."""
    traj = np.asarray(traj)
    if np.abs(traj[0] - q_init).max() < 1e-3:
        return traj
    a = np.linspace(0.0, 1.0, n_pre)[:, None]
    pre = q_init[None] * (1 - a) + traj[0][None] * a
    return np.concatenate([pre[:-1], traj], axis=0)


def adjust_and_plan(planner, to_pose, seed, T_cell_obj, T_target_obj, q_start, q_goal, fingers):
    """Deployed (B)-1 strategy = joint-first, task-fallback, scratch last-resort.

    1) JOINT adjust (~50ms): linear-ramp the cached seed start->goal, seed trajopt.
       Clean and cheap; handles ~81% of cells (the seed's own IK branch).
    2) TASK adjust: SE3-ramp + per-waypoint re-IK (warm-started, closest-to-prev
       within the continuity gate). Recovers cells whose branch shifts only
       SLIGHTLY off-grid (~near a branch boundary).
    3) SCRATCH (graph search): plan_single_js from INIT to the goal config with NO
       external seed. This is the ONLY step that can cross IK branches -- the
       seed-tracking steps (1,2) follow ONE branch by construction, so a cell that
       needs a >gate branch SWITCH mid-approach can only be solved here.

    Measured on pepsi/pose2 branch-jump cells: of 81 joint-fails, task recovers 18,
    scratch recovers 57 more, 6 (7%) genuinely unplannable -> caller drops the grasp.
    Cell-has-seed still == IK-reachable, so reachability stays deterministic; this
    just maximises how many reachable cells actually get a trajectory.

    Returns (ok, traj, start_state, which) with which in
    {'joint','task','scratch','fail'}. Each branch yields a WHOLE clean trajectory,
    never a hybrid joint-blend."""
    aj = adjust_seed(seed, q_start, q_goal)
    ok, traj, st = planner.plan_with_seed(q_goal, aj)
    if ok:
        return True, traj, st, "joint"
    at, goal_t = adjust_seed_task(planner, to_pose, seed, T_cell_obj, T_target_obj,
                                  q_start, q_goal, fingers)
    if at is not None:
        ok, traj, st = planner.plan_with_seed(goal_t, at, start_qpos=at[0])  # (A) plan from seed's rotated start
        if ok:
            traj = _prepend_base_move(np.asarray(traj), q_start)            # (A) deploy: begin at real INIT
            return True, traj, st, "task"
    # 3) scratch: graph search can switch IK branch where seed-tracking cannot.
    ok, traj = planner._refine_fingers(q_start, q_goal)
    return (True, traj, q_start, "scratch") if ok else (False, None, None, "fail")


def plan_lift(planner, to_pose, grasp_wrist_world, grasp_qpos, lift_h=0.10):
    """Plan the grasp->lift segment as a proper TRAJOPT (never a cartesian raise —
    that dies on joint-limit / singularity). Lift goal = grasp wrist raised by
    `lift_h` in world +z, fingers held at grasp_qpos[6:]; seed = straight joint
    interp (a vertical lift is collision-free so the interp is a fine seed). The
    CALLER must already have set the lift world (held object STRIPPED, table only).
    Returns (ok, traj) where traj starts at grasp_qpos."""
    dev = planner._tensor_args.device
    lo, hi = exp._arm_limits(planner)
    lift_w = np.asarray(grasp_wrist_world, dtype=np.float64).copy()
    lift_w[2, 3] += lift_h
    gq = torch.tensor(grasp_qpos, dtype=torch.float32, device=dev)
    r = planner._ik_solver.solve_batch(to_pose(lift_w[None], dev), retract_config=gq.unsqueeze(0))
    if not bool(r.success.view(-1)[0]):
        return False, None                                   # lift pose unreachable (joint limit)
    q_lift = np.asarray(grasp_qpos, dtype=np.float32).copy()
    q_lift[:6] = exp._unwrap_arm(r.solution.cpu().numpy().reshape(-1)[:6], grasp_qpos[:6], lo, hi)
    H = planner._motion_gen.js_trajopt_solver.action_horizon
    a = np.linspace(0, 1, H)[:, None]
    seed = (1 - a) * np.asarray(grasp_qpos, np.float32) + a * q_lift  # straight joint interp
    ok, traj, _ = planner.plan_with_seed(q_lift, seed, start_qpos=grasp_qpos)
    return ok, traj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="pepsi")
    ap.add_argument("--hand", default="inspire_left")
    ap.add_argument("--pose", type=int, default=2)
    ap.add_argument("--n_yaw", type=int, default=4, help="yaw grid cells (must match the dumped failures)")
    ap.add_argument("--failures", default=None, help="JSON of failing configs to step through")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    global YAW4
    YAW4 = np.linspace(0, 2 * np.pi, args.n_yaw, endpoint=False)   # match the sweep grid

    fails = []
    if args.failures:
        import json as _json
        fails = _json.load(open(args.failures))["failures"]
        print(f"loaded {len(fails)} failing configs from {args.failures}")

    planner = GraspPlanner(hand=args.hand)
    cand_root = str(Path(project_dir) / "candidates" / args.hand / exp.GRASP_VERSION)
    info, wrist_obj, preg = load_grasp_data(cand_root, args.obj)
    gmeta = [(g[1], g[2], g[3], exp.scene_pose_idx(args.obj, g[1], g[2])) for g in info]
    tab = exp.load_tabletop_poses(args.obj)

    # (B)-1: load the FROZEN seed cache — viz never re-plans the canonical seed.
    cpath = seed_cache_mod.cache_path(args.obj, args.hand, args.pose, args.n_yaw)
    if not cpath.exists():
        raise SystemExit(f"no seed cache at {cpath}\nbuild it first:\n  python src/visualization/"
                         f"seed_cache.py --obj {args.obj} --hand {args.hand} --pose {args.pose} "
                         f"--n_yaw {args.n_yaw}")
    SEEDS, cmeta = seed_cache_mod.load(cpath)
    g_for_pose = cmeta["grasps"]                       # only cached grasps are selectable
    print(f"[{args.obj}/{args.hand}] pose {args.pose}: cached grasps={g_for_pose} "
          f"({sum(v is not None for v in SEEDS.values())}/{len(SEEDS)} seeds)")

    X_GRID = np.arange(0.30, 0.71, 0.05)    # must match seed_cache.X_GRID
    z0 = float(tab[args.pose][2, 3]) + exp.TABLE_SURFACE_Z

    def place_T(radius, yaw, azim):
        T = np.eye(4)
        T[:3, :3] = rotz(azim) @ rotz(yaw) @ tab[args.pose][:3, :3]
        T[:3, 3] = rotz(azim) @ np.array([radius, 0.0, z0])
        return T

    mesh_file = str(Path(obj_path) / args.obj / "processed_data" / "mesh" / "simplified.obj")
    if not Path(mesh_file).exists():
        mesh_file = str(Path(obj_path) / args.obj / "raw_mesh" / f"{args.obj}.obj")

    def world_at(T_obj):
        # trajopt gets the object mesh so the approach AVOIDS the object (like the
        # real planner.plan); IK stays table-only (hand IS near the object).
        return _to_curobo_world({
            "mesh": {"target": {"pose": se32cart(T_obj), "file_path": mesh_file}},
            "cuboid": {"table": exp.TABLE_CUBOID}})

    def set_obj_world(T_obj):
        w = world_at(T_obj)
        if planner._world_structure_changed(w):
            planner._update_world(w)
        else:
            planner._update_target_pose_only(w)
        planner._cached_world = w

    # viser fires slider callbacks on worker threads -> disable CUDA graphs so
    # cuRobo is safe off the main thread (a lock still serializes the calls).
    planner._init_motion_gen(world_at(place_T(0.45, 0.0, 0.0)), use_cuda_graph=False)
    planner._cached_world = world_at(place_T(0.45, 0.0, 0.0))
    planner._init_ik_solver(
        _to_curobo_world({"mesh": {}, "cuboid": {"table": exp.TABLE_CUBOID}}), use_cuda_graph=False)
    dev = planner._tensor_args.device

    init_t = torch.tensor(planner._init_state, dtype=torch.float32, device=dev)
    _lo, _hi = exp._arm_limits(planner)

    def ik(Twrist, gi):
        # random IK (32 seeds) for robustness, then UNWRAP all 6 arm joints toward
        # INIT -> short rotation (wraps removed). aligned with cache (built the same).
        r = planner._ik_solver.solve_batch(_to_curobo_pose(Twrist[None], dev),
                                           retract_config=init_t.unsqueeze(0))
        if not bool(r.success.view(-1)[0]):
            return None
        full = planner._init_state.copy()
        full[:6] = exp._unwrap_arm(r.solution.cpu().numpy().reshape(-1)[:6],
                                   planner._init_state[:6], _lo, _hi)
        full[6:] = preg[gi]                       # PREGRASP (validated clear; object mesh in world)
        return full

    def canonical_seed(gi, xi, yi):
        return SEEDS.get((gi, xi, yi))                # frozen cache lookup — never re-plan

    # ---------------- viser ----------------
    vis = ViserViewer(port_number=args.port)
    vis.add_robot("xarm", str(URDF[args.hand]))
    vis.robot_dict["xarm"].update_cfg(planner._init_state)
    vis.add_floor(0.0)
    tbl = trimesh.creation.box(extents=exp.TABLE_CUBOID["dims"])
    tpose = np.eye(4); tpose[:3, 3] = exp.TABLE_CUBOID["pose"][:3]
    vis.add_object("table", tbl, tpose)
    vis.change_color("table", (0.9, 0.9, 0.92, 0.5))

    mesh_path = Path(obj_path) / args.obj / "raw_mesh" / f"{args.obj}.obj"
    obj_mesh = trimesh.load(str(mesh_path), force="mesh", process=False)
    vis.add_object("obj", obj_mesh, place_T(0.45, 0.0, 0.0))

    g = vis.server.gui
    sl_g = g.add_slider("grasp #", min=0, max=len(g_for_pose) - 1, step=1, initial_value=0)
    sl_r = g.add_slider("radius (m)", min=0.30, max=0.58, step=0.005, initial_value=0.45)
    sl_y = g.add_slider("yaw (deg)", min=0, max=359, step=1, initial_value=0)
    sl_a = g.add_slider("azimuth (deg)", min=-60, max=60, step=1, initial_value=0)
    btn = g.add_button("▶ Plan (B)")
    btn_seed = g.add_button("show seed (B input)")
    cb_task = g.add_checkbox("task-space seed (re-IK; experimental, branch-jumpy)",
                             initial_value=False)   # joint-space jblend is the clean default
    status = g.add_markdown("```\nslide to place; IK shown live; press Plan\n```")

    cur = {"gi": None, "qp": None, "T": None, "xi": 0, "yi": 0, "adj": None}
    lock = threading.Lock()
    gen = {"v": 0}

    def set_obj_pose(T):
        fr = vis.frame_nodes["obj"]
        fr.position = T[:3, 3].astype(np.float32)
        fr.wxyz = Rot.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]].astype(np.float32)

    def head():
        return (f"grasp {int(sl_g.value)}  r={float(sl_r.value):.3f} "
                f"yaw={sl_y.value:.0f} az={sl_a.value:.0f}")

    nav_busy = {"v": False}

    def preview(_=None, force=False):
        """LIVE on slider: place object + IK (cheap) to show reachability."""
        if nav_busy["v"] and not force:              # ignore slider echoes during nav
            return
        gen["v"] += 1
        my = gen["v"]
        with lock:
            if my != gen["v"]:                       # superseded by a newer drag
                return
            gi = g_for_pose[int(sl_g.value)]
            r, yaw, az = float(sl_r.value), np.radians(sl_y.value), np.radians(sl_a.value)
            T = place_T(r, yaw, az)
            set_obj_pose(T)
            vis.clear_traj()
            qp = ik(T @ wrist_obj[gi], gi)
            cur.update(gi=gi, T=T, qp=qp, adj=None,
                       xi=int(np.argmin(np.abs(X_GRID - r))),
                       yi=int(np.argmin(np.abs(wrap(YAW4 - yaw)))))
            if qp is None:
                vis.robot_dict["xarm"].update_cfg(planner._init_state)
                vis.change_color("obj", C_FAIL)
                status.content = f"```\n{head()}\nIK UNREACHABLE — move sliders\n```"
            else:
                vis.robot_dict["xarm"].update_cfg(qp)        # preview the goal grasp config
                vis.change_color("obj", C_READY)
                status.content = f"```\n{head()}\nIK ok  ->  press PLAN\n```"

    def do_plan(_=None):
        """Plan button: the heavy (B) — seed cache + adjust + trajopt-with-seed."""
        with lock:
            gi, qp, T, xi, yi = cur["gi"], cur["qp"], cur["T"], cur["xi"], cur["yi"]
            if qp is None:
                status.content = f"```\n{head()}\nIK unreachable — nothing to plan\n```"
                return
            seed = canonical_seed(gi, xi, yi)
            if seed is None:
                vis.change_color("obj", C_FAIL)
                status.content = (f"```\n{head()}\nno canonical seed at cell "
                                  f"(x={X_GRID[xi]:.2f}, yaw={np.degrees(YAW4[yi]):.0f})\n```")
                return
            if cb_task.value:
                T_cell = place_T(float(X_GRID[xi]), float(YAW4[yi]), 0.0)
                adj, goal_q = adjust_seed_task(planner, _to_curobo_pose, seed, T_cell, T,
                                               planner._init_state, qp, preg[gi])
                mode = "task"
                if adj is None:                              # waypoint IK couldn't stay on-branch
                    vis.change_color("obj", C_FAIL)
                    vis.robot_dict["xarm"].update_cfg(qp)
                    status.content = f"```\n{head()}\nTASK adjust failed (waypoint IK off-branch)\n```"
                    return
            else:
                adj = adjust_seed(seed, planner._init_state, qp)
                goal_q = qp
                mode = "joint"
            cur["adj"] = adj                                 # remember the (B) input for "show seed"
            set_obj_world(T)                                 # object in trajopt world -> avoid it
            ok, traj, st = planner.plan_with_seed(goal_q, adj)
            if ok:
                tr = traj[:: max(len(traj) // 120, 1)]      # planner already trimmed to valid length
                vis.add_traj("plan", {"xarm": tr}, {"obj": np.tile(T[None], (len(tr), 1, 1))})
                vis.robot_dict["xarm"].update_cfg(traj[-1])
                vis.change_color("obj", C_OK)
                if hasattr(vis, "gui_playing"):
                    vis.gui_playing.value = True             # auto-play the approach
                res = f"PLAN OK  ({len(traj)} steps)"
            else:
                vis.robot_dict["xarm"].update_cfg(qp)
                vis.change_color("obj", C_FAIL)
                res = "PLAN FAIL"
            r, yaw = float(sl_r.value), np.radians(sl_y.value)
            res = f"{res}  [{mode}]"
            status.content = (f"```\n{head()}\n{res}   trajopt {st:.3f}s\n"
                              f"cell (x={X_GRID[xi]:.2f}, yaw={np.degrees(YAW4[yi]):.0f})  "
                              f"dr={r - X_GRID[xi]:+.3f} dyaw={np.degrees(wrap(yaw - YAW4[yi])):+.0f}\n```")

    def show_seed(_=None):
        """Play the adjusted seed that was fed to trajopt — i.e. exactly WHAT
        went in as the (B) seed (run Plan first to populate it)."""
        with lock:
            adj, T = cur["adj"], cur["T"]
            if adj is None:
                status.content = f"```\n{head()}\npress PLAN first to build the seed\n```"
                return
            vis.clear_traj()
            tr = adj[:: max(len(adj) // 120, 1)]
            vis.add_traj("seed", {"xarm": tr}, {"obj": np.tile(T[None], (len(tr), 1, 1))})
            vis.robot_dict["xarm"].update_cfg(adj[-1])
            vis.change_color("obj", (0.25, 0.5, 0.95))      # blue = seed view
            if hasattr(vis, "gui_playing"):
                vis.gui_playing.value = True
            status.content = (f"```\n{head()}\nSHOWING SEED (B input)  {len(adj)} steps\n"
                              f"blue = adjusted seed fed to trajopt\n```")

    for s in (sl_g, sl_r, sl_y, sl_a):
        s.on_update(preview)
    btn.on_click(do_plan)
    btn_seed.on_click(show_seed)

    # ---- failure stepper ----
    if fails:
        btn_prev = g.add_button("◀ prev fail")
        btn_next = g.add_button("next fail ▶")
        nav = g.add_markdown(f"```\n{len(fails)} failures loaded\n```")
        fidx = {"i": -1}

        def goto_fail(step):
            fidx["i"] = (fidx["i"] + step) % len(fails)
            f = fails[fidx["i"]]
            nav.content = (f"```\nfail {fidx['i']+1}/{len(fails)}  gi={f['gi']}\n"
                           f"r={f['r']} yaw={f['yaw_deg']}  "
                           f"dr={f.get('dr')} dyaw={f.get('dyaw')}\n```")
            nav_busy["v"] = True
            try:
                if f["gi"] in g_for_pose:
                    sl_g.value = g_for_pose.index(f["gi"])
                sl_r.value = float(f["r"])
                sl_y.value = int(f["yaw_deg"])
                sl_a.value = 0
                preview(force=True)
                do_plan()
            finally:
                nav_busy["v"] = False

        btn_prev.on_click(lambda _: goto_fail(-1))
        btn_next.on_click(lambda _: goto_fail(1))

    preview()
    print(f"[seed_plan_viz] http://localhost:{args.port}")
    vis.start_viewer()


if __name__ == "__main__":
    main()
