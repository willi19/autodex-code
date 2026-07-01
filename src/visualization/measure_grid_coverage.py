#!/usr/bin/env python3
"""Measure scene coverage as a function of the (radius, yaw) reachability grid.

Motivation: azimuth is handled at runtime by a pure joint-0 offset and object
yaw by an incremental wrist adjustment, so the cached placement grid can be
COARSE in yaw/radius. This script quantifies, BEFORE modeling any adjustment,
how much raw scene coverage a reduced grid keeps vs the full grid — i.e. the
"worst case" the adjustment must recover.

Coverage (per the campaign's logic, grouped by tabletop pose p):
    scene s (at pose p) is COVERED  iff
        ∃ grasp g with gpose[g]==p  AND
          valid_array[s,g]  (g collision-free with scene s)  AND
          reach[g].any()    (g IK-reachable at SOME cell of the grid)

Reuses exp.py internals; computes reachability fresh per grid (does NOT touch
the cached reach_current.npy / coverage_current.* used by the campaign).

    conda activate mingi
    python src/visualization/measure_grid_coverage.py --obj pepsi --hand inspire_left
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import src.visualization.exp as exp
from autodex.planner import GraspPlanner
from autodex.planner.planner import _to_curobo_world
from autodex.utils.path import project_dir, obj_path
from autodex.utils.conversion import cart2se3


def build_valid_array(planner, obj, hand, wrist_obj, preg):
    """(valid_array (S,G), scene_meta) — collision-free coverage over current
    deployment scenes only. Replicates compute_coverage_data's scene loop so we
    don't load the grid-dependent reach cache."""
    scene_root = Path(obj_path) / obj / "scene"
    rows, scene_meta = [], []
    for st in exp.CURRENT_SCENE_TYPES:
        std = scene_root / st
        if not std.is_dir():
            continue
        for sf in sorted(std.glob("*.json")):
            scfg = json.load(open(sf))["scene"]
            obj_se3 = cart2se3(scfg["mesh"]["target"]["pose"])
            wrist_world = np.einsum("ij,ajk->aik", obj_se3, wrist_obj)
            coll = planner._check_collision(_to_curobo_world(scfg), wrist_world, preg)
            if (~coll).any():
                rows.append(~coll)
                scene_meta.append((st, sf.stem, exp.scene_pose_idx(obj, st, sf.stem)))
    return np.array(rows), scene_meta


def coverage(valid_array, scene_meta, grasp_meta, reach):
    """Per-pose + overall scene coverage under a given reach map."""
    scene_pose = np.array([m[2] for m in scene_meta])
    gpose = np.array([g[3] for g in grasp_meta])
    reach_any = reach.any(axis=(1, 2))                 # (G,) reachable somewhere
    S = len(scene_meta)
    covered = np.zeros(S, dtype=bool)
    per_pose = {}
    for p in sorted(set(scene_pose)):
        s_idx = np.where(scene_pose == p)[0]
        cand = np.where(gpose == p)[0]
        cand_ok = cand[reach_any[cand]]                # grasps reachable in grid
        n_cov = 0
        for s in s_idx:
            if cand_ok.size and valid_array[s, cand_ok].any():
                covered[s] = True
                n_cov += 1
        per_pose[p] = (n_cov, len(s_idx), int(reach_any[cand].sum()), len(cand))
    return covered.sum(), S, per_pose


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="pepsi")
    ap.add_argument("--hand", default="inspire_left")
    args = ap.parse_args()

    from src.grasp_generation.order.compute_order import load_grasp_data
    cand_root = str(Path(project_dir) / "candidates" / args.hand / exp.GRASP_VERSION)
    info, wrist_obj, preg = load_grasp_data(cand_root, args.obj)
    grasp_meta = [(g[1], g[2], g[3], exp.scene_pose_idx(args.obj, g[1], g[2])) for g in info]
    print(f"[{args.obj}/{args.hand}] grasps={len(info)}")

    planner = GraspPlanner(hand=args.hand)
    valid_array, scene_meta = build_valid_array(planner, args.obj, args.hand, wrist_obj, preg)
    print(f"scenes (>=1 collision-free grasp) = {len(scene_meta)}  "
          f"(poses: {sorted(set(m[2] for m in scene_meta))})\n")

    grids = {
        "full   4x8 (45)": (np.arange(0.35, 0.55, 0.05), np.linspace(0, 2*np.pi, 8, endpoint=False)),
        "reduced 4x4 (90)": (np.arange(0.35, 0.55, 0.05), np.linspace(0, 2*np.pi, 4, endpoint=False)),
    }
    for name, (xg, yg) in grids.items():
        exp.X_GRID, exp.YAW_GRID = xg, yg
        reach = exp._compute_reachability(planner, args.obj, wrist_obj, grasp_meta)
        cov, S, per_pose = coverage(valid_array, scene_meta, grasp_meta, reach)
        print(f"=== {name}  cells={len(xg)*len(yg)} ===")
        print(f"  COVERAGE: {cov}/{S} = {100*cov/S:.1f}%")
        for p, (nc, ns, rg, ng) in sorted(per_pose.items()):
            print(f"    pose {p}: scenes {nc}/{ns} covered | grasps reachable-in-grid {rg}/{ng}")
        print()


if __name__ == "__main__":
    main()
