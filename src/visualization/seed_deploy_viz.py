#!/usr/bin/env python3
"""Deployed-strategy visualizer for the (B)-1 seed cache.

Standalone (no sweep pkl needed): loads a seed cache, samples OFF-GRID placements,
and for each runs the DEPLOYED strategy LIVE --
    adjust_and_plan = joint -> task -> scratch fallback   (+ plan_lift)
then serves a viser viewer. Per config you see:
  * the object mesh at the actual (off-grid) pose + the table,
  * the SEED (trajopt input) vs the final PLAN, toggled,
  * the wrist Cartesian PATH as a colored spline -- so a clean straight approach vs
    an arc is obvious at a glance,
  * the robot + path COLORED BY WHICH TIER solved it
    (green=joint, blue=task, orange=scratch, red=fail),
  * the lift segment appended (toggle), so the full INIT->grasp->lift plays.

    conda activate mingi
    python src/visualization/seed_deploy_viz.py --obj pepsi --hand inspire_left \
        --pose 2 --n_yaw 36 --n_configs 160 --port 8081
"""
import argparse, os, sys, threading, pickle
from pathlib import Path
import numpy as np, torch, trimesh
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, "/home/mingi/AutoDex")
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "paradex"))

import src.visualization.exp as exp
import src.visualization.seed_cache as sc
from autodex.planner import GraspPlanner
from autodex.planner.planner import _to_curobo_world, _to_curobo_pose
from autodex.utils.path import obj_path, project_dir
from autodex.utils.conversion import se32cart
from src.grasp_generation.order.compute_order import load_grasp_data
from src.visualization.seed_plan_viz import adjust_seed, adjust_and_plan, plan_lift
from paradex.visualization.visualizer.viser import ViserViewer

TIER_RGB = {"joint": (40, 200, 70), "task": (60, 130, 240),
            "scratch": (240, 150, 30), "fail": (220, 40, 40)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="pepsi")
    ap.add_argument("--hand", default="inspire_left")
    ap.add_argument("--pose", type=int, default=2)
    ap.add_argument("--n_yaw", type=int, default=36)
    ap.add_argument("--n_configs", type=int, default=160, help="off-grid placements to precompute")
    ap.add_argument("--lift_h", type=float, default=0.10)
    ap.add_argument("--port", type=int, default=8081)
    args = ap.parse_args()

    # ---- planner + cache ------------------------------------------------------
    info, wrist_obj, preg = load_grasp_data(
        str(Path(project_dir) / "candidates" / args.hand / exp.GRASP_VERSION), args.obj)
    tab = exp.load_tabletop_poses(args.obj)
    z0 = float(tab[args.pose][2, 3]) + exp.TABLE_SURFACE_Z
    tabR = tab[args.pose][:3, :3]
    cpath = sc.cache_path(args.obj, args.hand, args.pose, args.n_yaw)
    cache = pickle.load(open(cpath, "rb"))["cache"]
    grasps = sorted({k[0] for k in cache if cache[k] is not None})
    print(f"loaded cache {cpath.name}: {sum(v is not None for v in cache.values())}/{len(cache)} seeds, "
          f"{len(grasps)} grasps")

    pl = GraspPlanner(hand=args.hand)
    INIT = pl._init_state
    dev = pl._tensor_args.device
    mesh_path = str(Path(obj_path) / args.obj / "processed_data" / "mesh" / "simplified.obj")
    if not Path(mesh_path).exists():
        mesh_path = str(Path(obj_path) / args.obj / "raw_mesh" / f"{args.obj}.obj")

    def world_at(T):
        return _to_curobo_world({"mesh": {"target": {"pose": se32cart(np.asarray(T)),
                                                      "file_path": mesh_path}},
                                 "cuboid": {"table": exp.TABLE_CUBOID}})
    pl._init_motion_gen(world_at(np.eye(4)), use_cuda_graph=False)
    pl._init_ik_solver(_to_curobo_world({"mesh": {}, "cuboid": {"table": exp.TABLE_CUBOID}}),
                       use_cuda_graph=False)
    lo, hi = exp._arm_limits(pl)
    _it = torch.tensor(INIT, dtype=torch.float32, device=dev)
    X, YAW = sc.X_GRID, sc.yaw_grid(args.n_yaw)

    def wrap(a):
        return (a + np.pi) % (2 * np.pi) - np.pi

    def place(r, yaw):
        return sc.place_T(tabR, z0, float(r), float(yaw))

    def ik(Tw, gi):
        r = pl._ik_solver.solve_batch(_to_curobo_pose(Tw[None], dev), retract_config=_it.unsqueeze(0))
        if not bool(r.success.view(-1)[0]):
            return None
        f = INIT.copy()
        f[:6] = exp._unwrap_arm(r.solution.cpu().numpy().reshape(-1)[:6], INIT[:6], lo, hi)
        f[6:] = preg[gi]
        return f

    def set_obj_world(T):
        w = world_at(T)
        (pl._update_world if pl._world_structure_changed(w) else pl._update_target_pose_only)(w)
        pl._cached_world = w

    def set_lift_world():
        w = _to_curobo_world({"mesh": {}, "cuboid": {"table": exp.TABLE_CUBOID}})
        (pl._update_world if pl._world_structure_changed(w) else pl._update_target_pose_only)(w)
        pl._cached_world = w

    def wrist_xyz(q_traj):                                  # FK arm -> wrist Cartesian path (N,3)
        t = torch.tensor(np.asarray(q_traj), dtype=torch.float32, device=dev)
        return pl._ik_solver.fk(t).ee_position.cpu().numpy()

    # ---- precompute off-grid configs through the deployed strategy ------------
    rng = np.random.default_rng(0)
    radii = np.arange(float(X[0]) + 0.01, float(X[-1]), 0.035)     # off the radius cells
    yaws = np.arange(7, 360, 13)                                   # off the 10deg yaw cells
    combos = [(gi, r, yd) for gi in grasps for r in radii for yd in yaws]
    rng.shuffle(combos)
    results = []
    print(f"precomputing up to {args.n_configs} configs through joint->task->scratch + lift ...")
    for gi, r, yd in combos:
        if len(results) >= args.n_configs:
            break
        yaw = np.radians(yd)
        xi = int(np.argmin(np.abs(X - r)))
        yi = int(np.argmin(np.abs(wrap(YAW - yaw))))
        seed = cache.get((gi, xi, yi))
        if seed is None:
            continue
        T = place(r, yaw)
        qp = ik(T @ wrist_obj[gi], gi)
        if qp is None:
            continue
        set_obj_world(T)
        Tcell = place(X[xi], YAW[yi])
        ok, traj, _st, which = adjust_and_plan(pl, _to_curobo_pose, np.asarray(seed),
                                               Tcell, T, INIT, qp, preg[gi])
        lift = None
        if ok:
            set_lift_world()
            okl, ltraj = plan_lift(pl, _to_curobo_pose, T @ wrist_obj[gi], qp, args.lift_h)
            lift = np.asarray(ltraj, dtype=np.float32) if okl else None
            set_obj_world(T)
        aj = np.asarray(adjust_seed(np.asarray(seed), INIT, qp), dtype=np.float32)
        results.append({
            "gi": int(gi), "r": round(float(r), 3), "yaw_deg": int(yd),
            "dr": round(float(r - X[xi]), 3),
            "dyaw": round(float(np.degrees(wrap(yaw - YAW[yi]))), 1),
            "T": T.astype(np.float32), "qp": qp.astype(np.float32),
            "which": which if ok else "fail",
            "seed": aj, "traj": (None if not ok else np.asarray(traj, dtype=np.float32)),
            "lift": lift})
        if len(results) % 20 == 0:
            print(f"  {len(results)}/{args.n_configs}")
    cnt = {k: sum(1 for e in results if e["which"] == k) for k in ("joint", "task", "scratch", "fail")}
    print(f"done: {len(results)} configs  {cnt}")

    # ---- viewer ---------------------------------------------------------------
    URDF = exp.URDF_BY_HAND
    import yourdfpy
    _aj = yourdfpy.URDF.load(str(URDF[args.hand])).actuated_joints[:6]
    ARM_LIMS = [(float(j.limit.lower), float(j.limit.upper)) for j in _aj]
    obj_mesh = trimesh.load(mesh_path, force="mesh", process=False)

    vis = ViserViewer(port_number=args.port)
    vis.add_robot("xarm", str(URDF[args.hand]))
    vis.add_floor(0.0)
    tbl = trimesh.creation.box(extents=exp.TABLE_CUBOID["dims"])
    tp = np.eye(4); tp[:3, 3] = exp.TABLE_CUBOID["pose"][:3]
    vis.add_object("table", tbl, tp)
    vis.change_color("table", (0.9, 0.9, 0.92, 0.4))
    vis.add_object("obj", obj_mesh, np.eye(4))

    g = vis.server.gui
    dd = g.add_dropdown("filter", ("all", "joint", "task", "scratch", "fail"), initial_value="all")
    btn_p = g.add_button("◀ prev"); btn_n = g.add_button("next ▶")
    cb_seed = g.add_checkbox("show SEED (trajopt input) not the plan", initial_value=False)
    cb_lift = g.add_checkbox("append LIFT segment", initial_value=True)
    status = g.add_markdown("```\n...\n```")
    frame_sl = g.add_slider("frame (scrub)", 0, 1, 1, 0)
    joint_sl = [g.add_slider(f"j{k} [{np.degrees(l):.0f}..{np.degrees(h):.0f}]",
                             float(l), float(h), 0.001, 0.0) for k, (l, h) in enumerate(ARM_LIMS)]
    state = {"idx": 0, "list": [], "q": None}
    lock = threading.Lock()
    _spline = {"node": None}

    def set_obj_pose(T):
        fr = vis.frame_nodes["obj"]
        fr.position = T[:3, 3].astype(np.float32)
        fr.wxyz = Rot.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]].astype(np.float32)

    def refilter():
        f = dd.value
        state["list"] = [i for i, e in enumerate(results) if f == "all" or e["which"] == f]
        state["idx"] = 0

    def cur_qtraj(e):
        """Full joint trajectory to scrub = (seed | plan)[ + lift]."""
        if cb_seed.value:
            base = e["seed"]
        else:
            base = e["traj"] if e["traj"] is not None else e["seed"]
        base = np.asarray(base)
        if cb_lift.value and (not cb_seed.value) and e["lift"] is not None:
            base = np.concatenate([base, np.asarray(e["lift"])], axis=0)
        return base

    def show():
        with lock:
            if not state["list"]:
                status.content = f"```\n(no {dd.value})\n```"; return
            e = results[state["list"][state["idx"]]]
            set_obj_pose(np.asarray(e["T"]))
            rgb = TIER_RGB[e["which"]]
            vis.change_color("xarm", rgb)                  # robot colored by tier
            vis.change_color("obj", tuple(c / 255 for c in rgb))
            q = cur_qtraj(e); state["q"] = q
            # wrist path spline (same tier color), shows arc vs straight at a glance
            if _spline["node"] is not None:
                _spline["node"].remove(); _spline["node"] = None
            if q is not None and len(q) >= 2:
                pts = wrist_xyz(q).astype(np.float32)
                _spline["node"] = vis.server.scene.add_spline_catmull_rom(
                    "/wrist_path", pts, color=rgb, line_width=3.0)
            frame_sl.max = max(len(q) - 1, 1)
            frame_sl.value = len(q) // 2
            set_frame(len(q) // 2)
            zr = 0.0
            if q is not None:
                w = wrist_xyz(q)[:, 2]; zr = 100 * (w.max() - max(w[0], w[-1]))
            status.content = (
                f"```\nfilter={dd.value}   {state['idx']+1}/{len(state['list'])}\n"
                f"gi={e['gi']} r={e['r']} yaw={e['yaw_deg']}  dr={e['dr']} dyaw={e['dyaw']}deg\n"
                f"TIER: {e['which'].upper()}   "
                f"{'SEED' if cb_seed.value else 'PLAN'}{' +LIFT' if (cb_lift.value and not cb_seed.value and e['lift'] is not None) else ''}\n"
                f"wrist z-rise={zr:.1f}cm   lift={'planned' if e['lift'] is not None else 'n/a'}\n"
                f"green=joint  blue=task  orange=scratch  red=fail\n```")

    def set_frame(f):
        q = state["q"]
        if q is None:
            return
        f = int(max(0, min(f, len(q) - 1)))
        qn = np.asarray(q[f])
        vis.robot_dict["xarm"].update_cfg(qn)
        for k in range(6):
            l, h = ARM_LIMS[k]
            joint_sl[k].value = float(min(max(qn[k], l), h))

    frame_sl.on_update(lambda _: set_frame(int(frame_sl.value)))

    def nav(s):
        if state["list"]:
            state["idx"] = (state["idx"] + s) % len(state["list"]); show()

    btn_p.on_click(lambda _: nav(-1))
    btn_n.on_click(lambda _: nav(1))
    dd.on_update(lambda _: (refilter(), show()))
    cb_seed.on_update(lambda _: show())
    cb_lift.on_update(lambda _: show())

    refilter(); show()
    print(f"[seed_deploy_viz] http://localhost:{args.port}   "
          f"tiers: {cnt}")
    vis.start_viewer()


if __name__ == "__main__":
    main()
