#!/usr/bin/env python3
"""Characterize the run-to-run stochasticity of (B) planning.

For each off-grid placement, run the FULL pipeline (live IK -> adjust -> object-
aware trajopt) K times and count successes. NO qp quantization here — we want the
RAW behaviour. Buckets:
    K/K   = deterministic success
    0/K   = deterministic fail
    1..K-1 = STOCHASTIC (flips run to run)

    python src/visualization/seed_stochastic.py --obj pepsi --hand inspire_left \
        --pose 2 --n_yaw 12 --n_grasps 3 --K 5
"""
import argparse, sys
from pathlib import Path
from itertools import product
from collections import Counter
import numpy as np, torch
from tqdm import tqdm

sys.path.insert(0, "/home/mingi/AutoDex")
import src.visualization.exp as exp
import src.visualization.seed_cache as seed_cache_mod
from autodex.planner import GraspPlanner
from autodex.planner.planner import _to_curobo_world, _to_curobo_pose, _snap_joint6
from autodex.utils.path import project_dir, obj_path
from autodex.utils.conversion import se32cart
from src.grasp_generation.order.compute_order import load_grasp_data
from src.visualization.seed_plan_viz import adjust_seed

ap = argparse.ArgumentParser()
ap.add_argument("--obj", default="pepsi"); ap.add_argument("--hand", default="inspire_left")
ap.add_argument("--pose", type=int, default=2); ap.add_argument("--n_yaw", type=int, default=12)
ap.add_argument("--n_grasps", type=int, default=3); ap.add_argument("--K", type=int, default=5)
ap.add_argument("--mode", choices=["plan", "full"], default="plan",
                help="plan = fix qp once, vary only trajopt (pure floating); full = re-IK each run")
ap.add_argument("--out", default="/tmp/stoch.json")
ap.add_argument("--det_ik", action="store_true", help="use deterministic seeded IK (the fix)")
args = ap.parse_args()

YAW = np.linspace(0, 2*np.pi, args.n_yaw, endpoint=False)
X_GRID = np.arange(0.30, 0.71, 0.05)        # must match seed_cache.X_GRID
def rotz(a): c,s=np.cos(a),np.sin(a); return np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
def wrap(d): return (d+np.pi)%(2*np.pi)-np.pi

pl = GraspPlanner(hand=args.hand)
info, wo, preg = load_grasp_data(str(Path(project_dir)/"candidates"/args.hand/exp.GRASP_VERSION), args.obj)
gm = [(g[1],g[2],g[3],exp.scene_pose_idx(args.obj,g[1],g[2])) for g in info]
tab = exp.load_tabletop_poses(args.obj); z0=float(tab[args.pose][2,3])+exp.TABLE_SURFACE_Z; tabR=tab[args.pose][:3,:3]
SEEDS, cmeta = seed_cache_mod.load(seed_cache_mod.cache_path(args.obj,args.hand,args.pose,args.n_yaw))
grasps = cmeta["grasps"][:args.n_grasps]
mesh = str(Path(obj_path)/args.obj/"processed_data"/"mesh"/"simplified.obj")
def placeT(r,yaw): T=np.eye(4); T[:3,:3]=rotz(yaw)@tabR; T[:3,3]=[r,0,z0]; return T
def worldat(T): return _to_curobo_world({"mesh":{"target":{"pose":se32cart(T),"file_path":mesh}},"cuboid":{"table":exp.TABLE_CUBOID}})
pl._init_motion_gen(worldat(np.eye(4)), use_cuda_graph=False); pl._cached_world=worldat(np.eye(4))
pl._init_ik_solver(_to_curobo_world({"mesh":{},"cuboid":{"table":exp.TABLE_CUBOID}}), use_cuda_graph=False)
dev = pl._tensor_args.device

_INIT_T = torch.tensor(pl._init_state, dtype=torch.float32, device=dev)
def ik(Tw, gi):
    if args.det_ik:   # seed_config=INIT + num_seeds=1: deterministic, nearest branch
        g = pl._ik_solver.solve_single(_to_curobo_pose(Tw[None],dev), retract_config=_INIT_T.unsqueeze(0),
                                        seed_config=_INIT_T.view(1,1,-1), num_seeds=1)
    else:             # raw: 32 random seeds (the current default everywhere before the fix)
        g = pl._ik_solver.solve_batch(_to_curobo_pose(Tw[None],dev), retract_config=_INIT_T.unsqueeze(0))
    if not bool(g.success.view(-1)[0]): return None
    q = g.solution.cpu().numpy().reshape(-1)[:6].copy(); q[5]=_snap_joint6(q[5],pl._init_state[5])
    f = pl._init_state.copy(); f[:6]=q; f[6:]=preg[gi]; return f
def setw(T):
    w=worldat(T); (pl._update_world if pl._world_structure_changed(w) else pl._update_target_pose_only)(w); pl._cached_world=w

radii = np.arange(0.36, 0.55, 0.03); yaws = np.arange(0,360,20)
buckets = Counter(); results = []
for gi, r, yd in tqdm(list(product(grasps, radii, yaws)), ncols=90):
    yaw = np.radians(yd); Tt = placeT(r, yaw)
    xi=int(np.argmin(np.abs(X_GRID-r))); yi=int(np.argmin(np.abs(wrap(YAW-yaw))))
    seed = SEEDS.get((gi,xi,yi))
    if seed is None: continue
    if args.mode == "plan":
        # isolate trajopt floating: ONE qp + ONE adjusted seed, run plan K times
        qp = ik(Tt@wo[gi], gi)
        if qp is None: continue
        adj = adjust_seed(seed, pl._init_state, qp); setw(Tt)
        succ = sum(int(pl.plan_with_seed(qp, adj)[0]) for _ in range(args.K))
    else:
        # combined: re-IK each run (qp varies)
        oks, reach = [], 0
        for _ in range(args.K):
            qp = ik(Tt@wo[gi], gi)
            if qp is None: continue
            reach += 1; setw(Tt)
            oks.append(int(pl.plan_with_seed(qp, adjust_seed(seed, pl._init_state, qp))[0]))
        if reach < args.K: continue
        succ = sum(oks)
    buckets[succ] += 1
    results.append({"gi": int(gi), "r": round(float(r),3), "yaw": int(yd), "succ": int(succ)})

tot = sum(buckets.values())
import json
json.dump({"mode": args.mode, "K": args.K, "grasps": [int(g) for g in grasps],
           "buckets": {int(k): int(v) for k,v in buckets.items()}, "results": results},
          open(args.out, "w"), indent=2)
print(f"\n=== stochasticity [{args.mode}]  {args.obj}/pose{args.pose}  grasps={grasps}  K={args.K} ===")
print(f"placements: {tot}   (saved -> {args.out})")
for k in range(args.K, -1, -1):
    if buckets[k]:
        tag = "det-SUCCESS" if k==args.K else ("det-FAIL" if k==0 else "** STOCHASTIC **")
        print(f"  {k}/{args.K} succ: {buckets[k]:4d}  {tag}")
stoch = sum(v for k,v in buckets.items() if 0<k<args.K)
print(f"\n>>> stochastic fraction: {stoch}/{tot} = {100*stoch/max(tot,1):.1f}%")
print(f">>> det-success {buckets[args.K]}  det-fail {buckets[0]}")
_fl = [r for r in results if 0 < r["succ"] < args.K]
if _fl: print("sample flippy (gi,r,yaw,succ/K):", [(r["gi"], r["r"], r["yaw"], r["succ"]) for r in _fl[:12]])
