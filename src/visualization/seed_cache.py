#!/usr/bin/env python3
"""(B)-1 persistent seed cache.

The whole premise of mechanism (B): the per-(grasp, placement-cell) canonical
trajectory is built ONCE offline (with retries — it's stochastic) and FROZEN to
disk. At runtime nothing is re-planned from scratch; the fixed seed is loaded,
adjusted to the actual off-grid pose, and fed to trajopt. viz / sweep / probe
all load the SAME file, so the seed is identical and reproducible everywhere.

Grid: radius X_GRID x yaw (n_yaw cells), at azimuth 0 (azimuth is a pure joint-0
offset, not cached).

Build:
    python src/visualization/seed_cache.py --obj pepsi --hand inspire_left \
        --pose 2 --n_yaw 12 --n_grasps 10 --seed 0
"""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/home/mingi/AutoDex")
import src.visualization.exp as exp
from autodex.planner import GraspPlanner
from autodex.planner.planner import _to_curobo_world, _to_curobo_pose, _snap_joint6
from autodex.utils.path import repo_dir, project_dir, obj_path
from autodex.utils.conversion import se32cart
from src.grasp_generation.order.compute_order import load_grasp_data

X_GRID = np.arange(0.30, 0.71, 0.05)        # radius cells 0.30..0.70


def yaw_grid(n_yaw):
    return np.linspace(0, 2 * np.pi, n_yaw, endpoint=False)


def cache_path(obj, hand, pose, n_yaw):
    d = Path(repo_dir) / "order" / hand / exp.GRASP_VERSION / obj / "seed_cache"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"pose{pose}_y{n_yaw}.pkl"


def place_T(tabR, z0, r, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    T = np.eye(4)
    T[:3, :3] = Rz @ tabR
    T[:3, 3] = [r, 0.0, z0]
    return T


def _ik(planner, preg_gi, Tw, lo, hi):
    # random IK (32 seeds) ONCE — no retry (re-rolling random 32 seeds doesn't help;
    # if 32 fail the pose is unreachable). Then UNWRAP all 6 arm joints toward INIT
    # so wrapped joints take the short equivalent -> minimal rotation.
    dev = planner._tensor_args.device
    retract = torch.tensor(planner._init_state, dtype=torch.float32, device=dev).unsqueeze(0)
    r = planner._ik_solver.solve_batch(_to_curobo_pose(Tw[None], dev), retract_config=retract)
    if not bool(r.success.view(-1)[0]):
        return None
    f = planner._init_state.copy()
    f[:6] = exp._unwrap_arm(r.solution.cpu().numpy().reshape(-1)[:6], planner._init_state[:6], lo, hi)
    f[6:] = preg_gi
    return f


J0_OFFSETS = [0.0, -np.pi / 6, np.pi / 6, -np.pi / 3, np.pi / 3,
              -np.pi / 2, np.pi / 2]                               # 0, ±30°, ±60°, ±90° base angle
CART_THRESH = 0.15     # angle 0 already clean (Cartesian waste <= 0.15) -> stop, no sweep.
                       # IMPORTANT: skip uses _cart, NOT _redundancy — a redundancy-0 trajectory can
                       # still loop the wrist (e.g. 2.2 m / 341 deg); using redundancy here wrongly
                       # short-circuited the worst cells to a single bad candidate.
W_Z = 5.0              # weight on wrist z-rise (m) added to joint redundancy (rad)


def _cost(planner, tr):
    # legacy selection objective (joint redundancy + weighted z-rise); _cart is the better one
    return _redundancy(tr) + W_Z * _zrise(planner, tr)


def _rotz(a):
    c, s = np.cos(a), np.sin(a)
    R = np.eye(4)
    R[:2, :2] = [[c, -s], [s, c]]
    return R


def _zrise(planner, tr):
    z = planner._ik_solver.fk(
        torch.tensor(np.asarray(tr), dtype=torch.float32, device=planner._tensor_args.device)
    ).ee_position.cpu().numpy()[:, 2]
    return float(z.max() - max(z[0], z[-1]))           # wrist Cartesian rise above endpoints (m)


def _redundancy(tr):
    """Redundant motion = how much each ARM joint goes OUTSIDE its own
    [start, end] interval (rad), summed over the 6 arm joints. A clean direct
    trajectory keeps every joint within [j_init, j_end] so this is ~0. Catches
    wrist lift, sideways reach AND long-way (e.g. 350°) rotations in one number,
    no FK needed."""
    a = np.asarray(tr)[:, :6]
    s, e = a[0], a[-1]
    lo_, hi_ = np.minimum(s, e), np.maximum(s, e)
    over = np.maximum(a.max(0) - hi_, 0.0) + np.maximum(lo_ - a.min(0), 0.0)
    return float(over.sum())


W_ROT = 0.10           # rotation-excess weight (rad) relative to position-excess (m)


def _cart(planner, tr):
    """WRIST Cartesian path waste = how much LONGER (m) the wrist position path is
    than the straight start->end line + W_ROT * extra ORIENTATION rotation (rad)
    beyond the direct start->end rotation. Catches wrist lift, sideways reach,
    far-from-base AND redundant hand rotation in ONE number — the joint-space
    `_redundancy` misses these (a monotonic-joint trajectory can still loop the wrist
    wildly). Needs FK (position + quaternion)."""
    st = planner._ik_solver.fk(
        torch.tensor(np.asarray(tr), dtype=torch.float32, device=planner._tensor_args.device))
    pos = st.ee_position.cpu().numpy()
    q = st.ee_quaternion.cpu().numpy()
    pexc = float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum() - np.linalg.norm(pos[-1] - pos[0]))

    def _ang(a, b):
        return 2.0 * np.arccos(min(abs(float((a * b).sum())), 1.0))
    rexc = float(sum(_ang(q[i], q[i + 1]) for i in range(len(q) - 1)) - _ang(q[0], q[-1]))
    return pexc + W_ROT * rexc


Z_FALLBACK = 0.15      # m; a fixed-INIT 1-plan whose wrist z-rise exceeds this ARCS
                       # -> fall back to the azimuth sweep. Measured: matched-start does
                       # NOT beat fixed-INIT (slightly worse z-rise), and a single plan
                       # (any start) leaves ~10% arc outliers -- only the sweep's diverse
                       # GOALS remove them. So: cheap 1-plan default, sweep only for arcs.


def build_one(planner, wrist_obj_gi, preg_gi, T_obj_cell, world_at, lo, hi, n_retry=5):
    """HYBRID build of the INIT->grasp seed candidates for one cell.

    1) ONE plan from the fixed INIT. If its wrist z-rise <= Z_FALLBACK keep it and stop
       (most cells -- 1 plan).
    2) Only if it ARCS (or fails) run the full azimuth SWEEP {±30,±60,±90}: rotate the
       object by dd, RE-IK the grasp (a NEW goal config), plan INIT->it, rotate the
       seed's j0 back by -dd. Diverse GOALS are what remove arcs (same-start retries
       just arc together). dd=0 == the step-1 INIT plan, so it is not repeated.

    SELECTION (min metric among candidates) happens LATER and is free. Returns [] if
    the grasp is unreachable at this cell."""
    # set world to the cell + IK the grasp
    w = world_at(T_obj_cell)
    if planner._world_structure_changed(w):
        planner._update_world(w)
    else:
        planner._update_target_pose_only(w)
    planner._cached_world = w
    q0 = _ik(planner, preg_gi, T_obj_cell @ wrist_obj_gi, lo, hi)
    if q0 is None:
        return []                                       # unreachable cell
    # 1) single plan from fixed INIT (== sweep dd=0)
    ok, tr = False, None
    for _ in range(n_retry):
        ok, tr = planner._refine_fingers(planner._init_state, q0)
        if ok:
            break
    if not ok:
        return []                                       # plan FAILED -> skip this grasp,
                                                        # don't sweep to rescue a failure
    tr = tr.astype(np.float32)
    if _zrise(planner, tr) <= Z_FALLBACK:
        return [tr]                                     # clean -> done, ONE plan
    # 2) SUCCEEDED but ARCS (z-up) -> azimuth sweep for a cleaner approach. The sweep
    #    is only worth it on a working-but-ugly plan; a failed plan is just skipped.
    cands = [tr]
    for dd in J0_OFFSETS:
        if dd == 0.0:
            continue                                    # dd=0 is the INIT plan above
        T_obj = _rotz(dd) @ T_obj_cell                  # object at base angle dd
        w = world_at(T_obj)
        if planner._world_structure_changed(w):
            planner._update_world(w)
        else:
            planner._update_target_pose_only(w)
        planner._cached_world = w
        q = _ik(planner, preg_gi, T_obj @ wrist_obj_gi, lo, hi)
        if q is None:
            continue
        ok2, tr2 = False, None
        for _ in range(n_retry):
            ok2, tr2 = planner._refine_fingers(planner._init_state, q)
            if ok2:
                break
        if not ok2:
            continue
        tr2 = tr2.astype(np.float32).copy()
        tr2[:, 0] -= dd                                 # rotate seed j0 back -> azimuth-0 grasp
        cands.append(tr2)
        if _zrise(planner, tr2) <= Z_FALLBACK:
            break                                       # found a clean (non-arcing) one -> stop sweeping
    return cands


def select(cands, planner, metric="cart"):
    """Pick the best candidate (min metric) from a cell's candidate list. FREE — no
    planning. metric: 'cart' (wrist Cartesian waste — the good one), 'zrise' (wrist
    height only), 'red' (joint redundancy only), 'cost' (redundancy + W_Z*z-rise).
    Returns the chosen traj or None."""
    if not cands:
        return None
    fn = {"cart": lambda t: _cart(planner, t),
          "red": _redundancy,
          "zrise": lambda t: _zrise(planner, t),
          "cost": lambda t: _cost(planner, t)}[metric]
    return min(cands, key=fn)


def build_all(planner, obj, hand, pose, grasps, wrist_obj, preg, tab, n_yaw, n_retry=5):
    Y = yaw_grid(n_yaw)
    z0 = float(tab[pose][2, 3]) + exp.TABLE_SURFACE_Z
    tabR = tab[pose][:3, :3]
    mesh_path = str(Path(obj_path) / obj / "processed_data" / "mesh" / "simplified.obj")
    if not Path(mesh_path).exists():
        mesh_path = str(Path(obj_path) / obj / "raw_mesh" / f"{obj}.obj")

    def world_at(T_obj):
        # motion_gen (trajopt) gets the object mesh so it AVOIDS the object during
        # approach — exactly like the real planner.plan (planner.py:1492).
        return _to_curobo_world({
            "mesh": {"target": {"pose": se32cart(T_obj), "file_path": mesh_path}},
            "cuboid": {"table": exp.TABLE_CUBOID}})

    T0 = place_T(tabR, z0, float(X_GRID[0]), float(Y[0]))
    if planner._motion_gen is None:                          # init once, reuse across poses
        planner._init_motion_gen(world_at(T0))
        planner._init_ik_solver(                              # IK: table-only (hand IS near object)
            _to_curobo_world({"mesh": {}, "cuboid": {"table": exp.TABLE_CUBOID}}), use_cuda_graph=False)
    planner._cached_world = world_at(T0)
    lo, hi = exp._arm_limits(planner)                        # compute once, reuse per cell

    from itertools import product as _product
    from tqdm import tqdm as _tqdm
    cand_cache, built, miss = {}, 0, 0                       # store ALL candidates per cell
    _bar = _tqdm(list(_product(grasps, enumerate(X_GRID), enumerate(Y))), ncols=90)
    for gi, (xi, x), (yi, yaw) in _bar:
            T_obj_cell = place_T(tabR, z0, float(x), float(yaw))    # object pose at THIS cell
            cands = build_one(planner, wrist_obj[gi], preg[gi], T_obj_cell,
                              world_at, lo, hi, n_retry)             # sweeps base angle (world set inside)
            cand_cache[(int(gi), xi, yi)] = cands                    # list of candidate seeds
            if cands:
                built += 1
            else:
                miss += 1
            _bar.set_postfix(built=built, miss=miss)
    meta = {"obj": obj, "hand": hand, "pose": pose, "n_yaw": n_yaw,
            "X": X_GRID.tolist(), "grasps": [int(g) for g in grasps]}
    cache = {k: select(v, planner, "cart") for k, v in cand_cache.items()}  # default = Cartesian
    return cache, cand_cache, meta


def load(path):
    d = pickle.load(open(path, "rb"))
    return d["cache"], d["meta"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="pepsi")
    ap.add_argument("--hand", default="inspire_left")
    ap.add_argument("--pose", default="all", help="tabletop pose index, or 'all'")
    ap.add_argument("--n_yaw", type=int, default=12)
    ap.add_argument("--n_grasps", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_retry", type=int, default=5)
    ap.add_argument("--reselect", default=None, choices=["cart", "cost", "red", "zrise"],
                    help="skip building: load *_cands.pkl and re-pick the cache with this metric (FREE)")
    ap.add_argument("--version", default=None,
                    help="grasp source under candidates/{hand}/ (default exp.GRASP_VERSION=v7; "
                         "e.g. 'table_only' for tabletop/reposition grasps). Also routes the "
                         "cache path order/{hand}/{version}/...")
    args = ap.parse_args()
    if args.version:
        exp.GRASP_VERSION = args.version                       # route candidate root + cache path

    planner = GraspPlanner(hand=args.hand)

    if args.reselect:                                          # re-select from stored candidates, no build
        poses = ([int(args.pose)] if args.pose != "all" else
                 [int(p.stem.split("_")[0][4:]) for p in
                  cache_path(args.obj, args.hand, 0, args.n_yaw).parent.glob("pose*_cands.pkl")])
        for pose in poses:
            cpath = cache_path(args.obj, args.hand, pose, args.n_yaw).with_name(
                f"pose{pose}_y{args.n_yaw}_cands.pkl")
            d = pickle.load(open(cpath, "rb"))
            cache = {k: select(v, planner, args.reselect) for k, v in d["cands"].items()}
            path = cache_path(args.obj, args.hand, pose, args.n_yaw)
            pickle.dump({"meta": d["meta"], "cache": cache}, open(path, "wb"))
            n_ok = sum(1 for v in cache.values() if v is not None)
            print(f"pose {pose}: re-selected ({args.reselect}) {n_ok} seeds -> {path}")
        return
    cand_root = str(Path(project_dir) / "candidates" / args.hand / exp.GRASP_VERSION)
    info, wrist_obj, preg = load_grasp_data(cand_root, args.obj)
    gmeta = [(g[1], g[2], g[3], exp.scene_pose_idx(args.obj, g[1], g[2])) for g in info]
    tab = exp.load_tabletop_poses(args.obj)

    if args.pose == "all":
        poses = sorted({m[3] for m in gmeta if m[3] is not None})
    else:
        poses = [int(args.pose)]
    print(f"[{args.obj}/{args.hand}] poses to build: {poses}  grid={len(X_GRID)}x{args.n_yaw}")

    for pose in poses:
        all_g = [i for i, m in enumerate(gmeta) if m[3] == pose]
        if not all_g:
            print(f"  pose {pose}: no grasps, skip"); continue
        rng = np.random.default_rng(args.seed)
        grasps = sorted(rng.choice(all_g, size=min(args.n_grasps, len(all_g)),
                                   replace=False).tolist())
        print(f"\n=== pose {pose}  grasps={grasps} ===")
        cache, cand_cache, meta = build_all(planner, args.obj, args.hand, pose, grasps,
                                            wrist_obj, preg, tab, args.n_yaw, args.n_retry)
        path = cache_path(args.obj, args.hand, pose, args.n_yaw)
        pickle.dump({"meta": meta, "cache": cache}, open(path, "wb"))
        cpath = path.with_name(path.stem + "_cands.pkl")        # ALL candidates -> re-select for free
        pickle.dump({"meta": meta, "cands": cand_cache}, open(cpath, "wb"))
        n_ok = sum(1 for v in cache.values() if v is not None)
        ncand = sum(len(v) for v in cand_cache.values())
        print(f"pose {pose}: wrote {n_ok}/{len(cache)} seeds -> {path}")
        print(f"pose {pose}: wrote {ncand} candidates ({ncand/max(n_ok,1):.1f}/cell) -> {cpath}")


if __name__ == "__main__":
    main()
