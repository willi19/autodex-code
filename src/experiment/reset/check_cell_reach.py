#!/usr/bin/env python3
"""Check whether a reorient cell's grasp candidates are IK-reachable at any
obj position. Useful to answer "does this cell ever succeed?".

Loads ``candidates/{hand}/reset/{obj}/reorient_{h_cm}/{i}_{j}/`` grasps,
grid-searches obj xy (with z from canonical tabletop pose) and yaw, runs IK,
reports the count + best examples.

Usage:
    python src/experiment/reset/check_cell_reach.py \\
        --obj attached_container --cell 0_1
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from autodex.utils.path import project_dir
from autodex.planner import GraspPlanner


TABLE_SURFACE_Z = 0.039
SCENE_TABLE_ROOT = Path.home() / "shared_data/AutoDex/object/paradex"
TABLETOP_ROOT = Path.home() / "shared_data/AutoDex/object/paradex"


def _load_canonical_tabletop(obj: str, i_int: int) -> np.ndarray:
    """Return 4x4 tabletop pose ``i_int`` in robot frame (xy=0, z=table)."""
    p = (TABLETOP_ROOT / obj / "processed_data" / "info" / "tabletop"
         / f"{i_int:03d}.npy")
    if not p.exists():
        sys.exit(f"tabletop pose not found: {p}")
    T = np.load(p)
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True)
    ap.add_argument("--hand", default="inspire_left")
    ap.add_argument("--cell", required=True,
                    help="e.g. 0_1  (i_int_j_int under reorient_{h_cm})")
    ap.add_argument("--h_cm", type=int, default=0)
    ap.add_argument("--x_grid", nargs=3, type=float,
                    default=[0.30, 0.65, 0.05],
                    metavar=("MIN", "MAX", "STEP"))
    ap.add_argument("--y_grid", nargs=3, type=float,
                    default=[-0.25, 0.25, 0.05],
                    metavar=("MIN", "MAX", "STEP"))
    ap.add_argument("--yaw_grid", nargs=3, type=float,
                    default=[0.0, 360.0, 30.0],
                    metavar=("MIN", "MAX", "STEP"))
    args = ap.parse_args()

    i_int, j_int = (int(x) for x in args.cell.split("_"))
    cell_dir = (Path(project_dir) / "candidates" / args.hand / "reset"
                / args.obj / f"reorient_{args.h_cm}" / args.cell)
    if not cell_dir.exists():
        sys.exit(f"cell dir missing: {cell_dir}")

    grasp_dirs = sorted([p for p in cell_dir.iterdir() if p.is_dir()],
                        key=lambda p: int(p.name))
    if not grasp_dirs:
        sys.exit(f"no grasp dirs in {cell_dir}")
    print(f"[check] {len(grasp_dirs)} grasps in cell {args.cell}")

    wrist_obj = np.stack([np.load(g / "wrist_se3.npy") for g in grasp_dirs])
    print(f"[check] wrist_obj shape={wrist_obj.shape}")

    T_canonical = _load_canonical_tabletop(args.obj, i_int)
    print(f"[check] canonical tabletop {i_int:03d} pose:\n{T_canonical.round(3)}")
    obj_z = float(T_canonical[2, 3]) + TABLE_SURFACE_Z
    R_canonical = T_canonical[:3, :3].copy()

    # Build scene_cfg with placeholder obj (table only world for IK).
    from autodex.utils.conversion import se32cart
    scene_cfg = {
        "mesh": {},
        "cuboid": {
            "table": {
                "dims": [2, 3, 0.2],
                "pose": [1.1, 0, -0.1 + 0.037, 1, 0, 0, 0],
            },
        },
    }

    print(f"[check] init planner ({args.hand}) ...")
    planner = GraspPlanner(hand=args.hand, use_cuda_graph=False)
    # Init IK solver via dummy plan-style call: use planner.solve_ik path —
    # but the simplest is to call planner._init_ik_solver directly with
    # the table-only world.
    from autodex.planner.planner import _to_curobo_world
    world_cfg = _to_curobo_world(scene_cfg)
    planner._init_ik_solver(world_cfg)

    xs = np.arange(*args.x_grid)
    ys = np.arange(*args.y_grid)
    yaws_deg = np.arange(*args.yaw_grid)
    print(f"[check] grid: |x|={len(xs)} |y|={len(ys)} "
          f"|yaw|={len(yaws_deg)} → {len(xs)*len(ys)*len(yaws_deg)} obj poses")

    n_g = len(wrist_obj)
    # Build all (x, y, yaw) × grasp wrist poses in one big array → one IK
    # batch (~17 chunks of BATCH_SIZE=50) instead of 840 separate calls.
    grid_xyz = [(float(x), float(y), float(yd))
                for x in xs for y in ys for yd in yaws_deg]
    n_grid = len(grid_xyz)
    all_wrists = np.zeros((n_grid * n_g, 4, 4))
    for k, (x, y, yd) in enumerate(grid_xyz):
        yaw = np.deg2rad(yd)
        c, s = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        T_obj = np.eye(4)
        T_obj[:3, :3] = Rz @ R_canonical
        T_obj[:3, 3] = [x, y, obj_z]
        all_wrists[k * n_g:(k + 1) * n_g] = T_obj[None] @ wrist_obj

    print(f"[check] running batched IK on {len(all_wrists)} poses ...")
    succ_flat = planner.ik_pose_batch(all_wrists).reshape(n_grid, n_g)

    hits = []
    best = (-1, None, None, None)
    for k, (x, y, yd) in enumerate(grid_xyz):
        n_ok = int(succ_flat[k].sum())
        if n_ok > 0:
            hits.append((n_ok, x, y, yd))
            if n_ok > best[0]:
                best = (n_ok, x, y, yd)

    print(f"\n[result] {len(hits)} / {n_grid} obj "
          f"poses had >=1 IK-feasible grasp")
    if hits:
        print(f"[result] best: {best[0]}/{n_g} feasible at "
              f"x={best[1]:.2f} y={best[2]:.2f} yaw={best[3]:.0f}°")
        print(f"[result] first 5 hits:")
        for h in hits[:5]:
            print(f"    n_ok={h[0]}  x={h[1]:.2f} y={h[2]:.2f} yaw={h[3]:.0f}°")
    else:
        print(f"[result] NO obj position in grid makes cell {args.cell} "
              f"IK-feasible. Cell is likely unreachable by design.")


if __name__ == "__main__":
    main()
