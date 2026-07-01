#!/usr/bin/env python3
"""Visualize the IK reach table over the (x, yaw) grid for a scene's grasps.

For each covering grasp of a scene, solve IK at every (X_GRID, YAW_GRID) cell
TWICE -- with the table cuboid and with NO collision world -- and plot both as
heatmaps. Where "table" is empty but "free" is full, the table is what kills the
reach (e.g. the table cuboid swallowing the robot base/lower arm).

    python src/visualization/reach_viz.py --obj pepsi --scene shelf/9
    python src/visualization/reach_viz.py --obj pepsi --scene shelf/9 --attempts 8
"""
import argparse, sys
import numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

sys.path.insert(0, "/home/mingi/AutoDex")
import src.visualization.exp as exp
from autodex.planner import GraspPlanner
from autodex.planner.planner import _to_curobo_world, _to_curobo_pose


def reach_grid(pl, wo_g, pose_T, world, attempts=1):
    """reachable[xi, yi] over the (X_GRID, YAW_GRID) grid for one grasp."""
    pl._ik_solver = None
    pl._init_ik_solver(world, use_cuda_graph=False)
    dev, BS = pl._tensor_args.device, pl.BATCH_SIZE
    INIT = torch.tensor(pl._init_state, dtype=torch.float32, device=dev)
    X, YAW = exp.X_GRID, exp.YAW_GRID
    R = np.zeros((len(X), len(YAW)), bool)
    for xi, x in enumerate(X):
        goals = np.asarray([exp._place_T(pose_T, float(x), float(yw)) @ wo_g
                            for yw in YAW], np.float32)
        hit = np.zeros(len(YAW), bool)
        for _ in range(attempts):
            for cs in range(0, len(goals), BS):
                ch = goals[cs:cs + BS]; B = len(ch)
                if B < BS:
                    ch = np.concatenate([ch, np.tile(ch[:1], (BS - B, 1, 1))])
                r = pl._ik_solver.solve_batch(_to_curobo_pose(ch, dev),
                        retract_config=INIT.unsqueeze(0).repeat(BS, 1))
                s = r.success.cpu().numpy().reshape(-1)[:B]
                hit[cs:cs + B] |= s
        R[xi] = hit
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="pepsi")
    ap.add_argument("--hand", default="inspire_left")
    ap.add_argument("--scene", default="shelf/9", help="scene_type/scene_id")
    ap.add_argument("--attempts", type=int, default=1, help="IK tries per cell")
    ap.add_argument("--out", default="outputs/seed_sweep/reach_table.png")
    args = ap.parse_args()
    st, sid = args.scene.split("/")

    pl = GraspPlanner(hand=args.hand)
    d = exp.compute_coverage_data(pl, args.obj, args.hand)
    va, gm, sm, wo = d["valid_array"], d["grasp_meta"], d["scene_meta"], d["wrist_obj"]
    tab = exp.load_tabletop_poses(args.obj)
    si = [i for i, s in enumerate(sm) if s[0] == st and str(s[1]) == sid][0]
    pose = sm[si][2]
    cov = list(np.where(va[si])[0])
    print(f"{args.scene} (pose {pose}): {len(cov)} covering grasps", flush=True)

    W_table = _to_curobo_world({"mesh": {}, "cuboid": {"table": exp.TABLE_CUBOID}})
    W_free = _to_curobo_world({"mesh": {}, "cuboid": {}})
    X, YAW = exp.X_GRID, exp.YAW_GRID

    n = len(cov)
    fig, axes = plt.subplots(n, 2, figsize=(11, 2.2 * n), squeeze=False)
    for row, g in enumerate(cov):
        Rt = reach_grid(pl, wo[g], tab[pose], W_table, args.attempts)
        Rf = reach_grid(pl, wo[g], tab[pose], W_free, args.attempts)
        for col, (R, ttl) in enumerate([(Rt, "with TABLE"), (Rf, "NO collision")]):
            ax = axes[row][col]
            ax.imshow(R, aspect="auto", cmap="Greens", vmin=0, vmax=1,
                      extent=[0, 360, X[-1], X[0]])
            ax.set_title(f"{gm[g][2]}  {ttl}  ({int(R.sum())} cells)", fontsize=8)
            ax.set_xlabel("yaw (deg)"); ax.set_ylabel("x (m)")
        print(f"  {gm[g][2]}: table={int(Rt.sum())}  free={int(Rf.sum())}", flush=True)
    fig.suptitle(f"{args.obj} {args.scene} reach table (attempts={args.attempts}) "
                 f"— green=IK reachable", y=1.0)
    plt.tight_layout()
    plt.savefig(args.out, dpi=110, bbox_inches="tight")
    print("saved", args.out, flush=True)


if __name__ == "__main__":
    main()
