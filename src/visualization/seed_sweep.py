#!/usr/bin/env python3
"""Headless off-grid success sweep for mechanism (B).

Same path as seed_plan_viz: for each off-grid (radius, yaw, azimuth) placement,
IK the grasp -> nearest (radius,yaw) grid cell's cached canonical seed ->
adjust_seed -> plan_with_seed. Reports planning success among IK-reachable
placements (averaged over several grasps).

    python src/visualization/seed_sweep.py --obj pepsi --hand inspire_left --pose 2 --n_grasps 8
"""
import argparse, sys
from pathlib import Path
import numpy as np, torch

sys.path.insert(0, "/home/mingi/AutoDex")
import src.visualization.exp as exp
from autodex.planner import GraspPlanner
from autodex.planner.planner import _to_curobo_world, _to_curobo_pose, _snap_joint6
from autodex.utils.path import project_dir, obj_path
from autodex.utils.conversion import se32cart
from pathlib import Path
from src.grasp_generation.order.compute_order import load_grasp_data
from src.visualization.seed_plan_viz import adjust_seed_task, adjust_and_plan, plan_lift

YAW4 = np.linspace(0, 2 * np.pi, 4, endpoint=False)
def rotz(a):
    c, s = np.cos(a), np.sin(a); return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
def wrap(d): return (d + np.pi) % (2*np.pi) - np.pi
def adjust_seed(seed, q0, q1):
    H = len(seed); a = np.linspace(0,1,H)[:,None]
    return seed + (1-a)*wrap(q0-seed[0]) + a*wrap(q1-seed[-1])

ap = argparse.ArgumentParser()
ap.add_argument("--obj", default="pepsi"); ap.add_argument("--hand", default="inspire_left")
ap.add_argument("--pose", type=int, default=2); ap.add_argument("--n_grasps", type=int, default=8)
ap.add_argument("--n_yaw", type=int, default=8, help="yaw grid cells (denser = smaller dyaw)")
ap.add_argument("--seed", type=int, default=0, help="RNG seed for random grasp sample")
ap.add_argument("--dump_failures", default=None, help="write joint-space-fail configs to this JSON")
ap.add_argument("--save_results", default=None, help="save per-config {ok,T,qp,seed,traj} pkl for replay viz")
ap.add_argument("--task", action="store_true", help="also measure task-space re-IK adjust")
ap.add_argument("--max_grasps", type=int, default=0, help="subsample first N cached grasps (0=all) for a fast check")
ap.add_argument("--lift", action="store_true", help="also plan the grasp->lift segment (deployed joint->task strategy + trajopt lift)")
ap.add_argument("--lift_h", type=float, default=0.10, help="lift height (m)")
args = ap.parse_args()
YAW4 = np.linspace(0, 2 * np.pi, args.n_yaw, endpoint=False)   # denser yaw grid

planner = GraspPlanner(hand=args.hand)
cand_root = str(Path(project_dir) / "candidates" / args.hand / exp.GRASP_VERSION)
info, wrist_obj, preg = load_grasp_data(cand_root, args.obj)
gmeta = [(g[1], g[2], g[3], exp.scene_pose_idx(args.obj, g[1], g[2])) for g in info]
tab = exp.load_tabletop_poses(args.obj)
# (B)-1: use the FROZEN seed cache (same grasps + same seeds as viz/probe)
import src.visualization.seed_cache as seed_cache_mod
cpath = seed_cache_mod.cache_path(args.obj, args.hand, args.pose, args.n_yaw)
if not cpath.exists():
    raise SystemExit(f"no seed cache at {cpath} -- build it with seed_cache.py first")
SEEDS, cmeta = seed_cache_mod.load(cpath)
g_for_pose = cmeta["grasps"]
if args.max_grasps:
    g_for_pose = g_for_pose[:args.max_grasps]                 # fast check on a subset
print(f"loaded cache: grasps={g_for_pose}  ({sum(v is not None for v in SEEDS.values())}/{len(SEEDS)} seeds)")
X_GRID = np.arange(0.30, 0.71, 0.05)        # must match seed_cache.X_GRID
z0 = float(tab[args.pose][2,3]) + exp.TABLE_SURFACE_Z

_mesh_file = str(Path(obj_path) / args.obj / "processed_data" / "mesh" / "simplified.obj")
if not Path(_mesh_file).exists():
    _mesh_file = str(Path(obj_path) / args.obj / "raw_mesh" / f"{args.obj}.obj")

def world_at(T_obj):                                   # trajopt sees object -> avoids it
    return _to_curobo_world({"mesh": {"target": {"pose": se32cart(T_obj), "file_path": _mesh_file}},
                             "cuboid": {"table": exp.TABLE_CUBOID}})

def set_obj_world(T_obj):
    w = world_at(T_obj)
    if planner._world_structure_changed(w): planner._update_world(w)
    else: planner._update_target_pose_only(w)
    planner._cached_world = w

_world_table_only = _to_curobo_world({"mesh": {}, "cuboid": {"table": exp.TABLE_CUBOID}})
def set_lift_world():                                  # object is HELD -> strip it, table only
    planner._update_world(_world_table_only)
    planner._cached_world = _world_table_only

planner._init_motion_gen(world_at(np.eye(4)), use_cuda_graph=False)  # match viz; manual_seed works
planner._cached_world = world_at(np.eye(4))
planner._init_ik_solver(_to_curobo_world({"mesh": {}, "cuboid": {"table": exp.TABLE_CUBOID}}),
                        use_cuda_graph=False)
dev = planner._tensor_args.device

def place_T(r, yaw, az):
    T = np.eye(4); T[:3,:3] = rotz(az) @ rotz(yaw) @ tab[args.pose][:3,:3]
    T[:3,3] = rotz(az) @ np.array([r,0.,z0]); return T

_init_t = torch.tensor(planner._init_state, dtype=torch.float32, device=dev)
_lo, _hi = exp._arm_limits(planner)
def ik(Tw, gi):
    # random IK (32 seeds) + unwrap all 6 joints toward INIT (short rotation).
    r = planner._ik_solver.solve_batch(_to_curobo_pose(Tw[None], dev), retract_config=_init_t.unsqueeze(0))
    if not bool(r.success.view(-1)[0]): return None
    full = planner._init_state.copy()
    full[:6] = exp._unwrap_arm(r.solution.cpu().numpy().reshape(-1)[:6], planner._init_state[:6], _lo, _hi)
    full[6:] = preg[gi]                           # PREGRASP (validated clear; object mesh in world)
    return full

def canonical_seed(gi, xi, yi):
    return SEEDS.get((gi, xi, yi))                # frozen cache lookup — never re-plan

# azimuth is NOT swept: it is a pure joint-0 offset (base-z rotation, table is
# symmetric about base-z) -> feasibility-invariant, corrected post-hoc. Only the
# radius / yaw off-grid deltas actually need the seed adjustment.
from itertools import product
from tqdm import tqdm
radii = np.arange(0.36, 0.55, 0.03)
yaws  = np.arange(0, 360, 20)
tot = dict(ik_fail=0, no_seed=0, reach=0, ok_joint=0, ok_task=0, only_task=0,
           ok_app=0, ok_app_lift=0, lift_fail=0)
fails = []
saved = []                                            # per-config result for the replay viz
def _ds(a, n=120):                                    # downsample a (T,dof) traj for storage
    return a[:: max(len(a) // n, 1)].astype(np.float32)
_combos = list(product(g_for_pose, radii, yaws))
_bar = tqdm(_combos, ncols=90)
for gi, r, yd in _bar:
        if True:
            yaw = np.radians(yd)
            T_tgt = place_T(r, yaw, 0.0)
            xi = int(np.argmin(np.abs(X_GRID - r)))
            yi = int(np.argmin(np.abs(wrap(YAW4 - yaw))))
            seed = canonical_seed(gi, xi, yi)
            if seed is None: tot["no_seed"] += 1; continue
            qp = ik(T_tgt @ wrist_obj[gi], gi)
            if qp is None: tot["ik_fail"] += 1; continue
            tot["reach"] += 1
            set_obj_world(T_tgt)                      # object in trajopt world -> avoid it
            aj = adjust_seed(seed, planner._init_state, qp)
            okj, traj, _ = planner.plan_with_seed(qp, aj)
            tot["ok_joint"] += int(okj)
            if args.save_results:                     # store EXACTLY what trajopt saw -> viz replays
                saved.append({"gi": int(gi), "r": round(float(r), 3), "yaw_deg": int(yd),
                              "dyaw": round(float(np.degrees(wrap(yaw - YAW4[yi]))), 1),
                              "dr": round(float(r - X_GRID[xi]), 3), "ok": bool(okj),
                              "T": T_tgt.astype(np.float32), "qp": qp.astype(np.float32),
                              "adj": _ds(aj), "traj": (None if not okj else _ds(traj))})
            if not okj:
                fails.append({"gi": int(gi), "r": round(float(r), 3), "yaw_deg": int(yd),
                              "dyaw": round(float(np.degrees(wrap(yaw - YAW4[yi]))), 1),
                              "dr": round(float(r - X_GRID[xi]), 3)})
            if args.task:
                T_cell = place_T(float(X_GRID[xi]), float(YAW4[yi]), 0.0)
                at, goal_t = adjust_seed_task(planner, _to_curobo_pose, seed, T_cell, T_tgt,
                                              planner._init_state, qp, preg[gi])
                okt = False if at is None else planner.plan_with_seed(goal_t, at)[0]  # None -> task-fail
                tot["ok_task"] += int(okt)
                tot["only_task"] += int(okt and not okj)
            if args.lift:
                # deployed path: joint->task fallback approach, THEN trajopt lift (object held -> stripped).
                T_cell = place_T(float(X_GRID[xi]), float(YAW4[yi]), 0.0)
                ok_app, _, _, _which = adjust_and_plan(planner, _to_curobo_pose, seed, T_cell, T_tgt,
                                                       planner._init_state, qp, preg[gi])
                tot["ok_app"] += int(ok_app)
                if ok_app:
                    set_lift_world()                              # held object stripped, table only
                    ok_lift, _ = plan_lift(planner, _to_curobo_pose, T_tgt @ wrist_obj[gi], qp, args.lift_h)
                    tot["ok_app_lift"] += int(ok_lift)
                    tot["lift_fail"] += int(not ok_lift)
                    set_obj_world(T_tgt)                          # restore object world for next iter

R = tot["reach"]
print(f"\n=== (B) off-grid sweep  {args.obj}/pose{args.pose}  grasps={len(g_for_pose)} "
      f"radius x yaw = {len(radii)}x{len(yaws)}  (yaw grid n={args.n_yaw}) ===")
print(f"IK-unreachable : {tot['ik_fail']}")
print(f"no canon seed  : {tot['no_seed']}")
print(f"IK-reachable & seeded : {R}")
if R:
    print(f"\n>>> joint-space adjust : {tot['ok_joint']}/{R} = {100*tot['ok_joint']/R:.1f}%")
    if args.task:
        print(f">>> task-space  adjust : {tot['ok_task']}/{R} = {100*tot['ok_task']/R:.1f}%")
        print(f">>> recovered by task (task ok & joint fail): {tot['only_task']}")
    if args.lift:
        print(f"\n>>> approach (joint->task strategy)   : {tot['ok_app']}/{R} = {100*tot['ok_app']/R:.1f}%")
        print(f">>> approach + lift (full deployed)   : {tot['ok_app_lift']}/{R} = {100*tot['ok_app_lift']/R:.1f}%")
        print(f">>> lift fails among approach-success : {tot['lift_fail']}/{tot['ok_app']}")
if args.dump_failures:
    import json
    json.dump({"obj": args.obj, "hand": args.hand, "pose": args.pose,
               "n_yaw": args.n_yaw, "failures": fails}, open(args.dump_failures, "w"), indent=2)
    print(f"\nwrote {len(fails)} joint-space failures -> {args.dump_failures}")
if args.save_results:
    import pickle
    pickle.dump({"obj": args.obj, "hand": args.hand, "pose": args.pose, "n_yaw": args.n_yaw,
                 "results": saved}, open(args.save_results, "wb"))
    print(f"\nsaved {len(saved)} configs (for replay viz) -> {args.save_results}")
