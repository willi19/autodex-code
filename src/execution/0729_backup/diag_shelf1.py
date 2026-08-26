#!/usr/bin/env python3
"""Diagnostic — visualize the 5 untried shelf/1 candidates for
servingbowl_small (v7) with cyl_yaw expansion, IK-check each, show in viser.

No robot / no perception. Loads the most-recent run_auto trial's pose_world
(or --pose_world override) to get the object pose, then renders the candidates
colored by IK status (green=valid, yellow=ik_fail, red=filtered).

Usage:
    python src/execution/diag_shelf1.py
    python src/execution/diag_shelf1.py --port 8081
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from autodex.utils.path import obj_path, get_candidate_path
from autodex.utils.conversion import cart2se3, se32cart
from autodex.utils.symmetry import get_cyl_axis_local
from autodex.planner import GraspPlanner
from autodex.planner.obstacles import add_obstacles
from autodex.planner.visualizer import ScenePlanVisualizer
from paradex.calibration.utils import load_c2r

from src.execution.scene_cfg import pose_world_to_scene_cfg


HAND = "inspire_left"
OBJ = "servingbowl_small"
VERSION = "v7"
SCENE_TYPE = "shelf"
SCENE_ID = "1"


def _latest_pose_world():
    """Find pose_world.npy from the most-recent run_auto trial for this obj."""
    root = (Path.home() / "shared_data" / "AutoDex" / "experiment" / VERSION
            / HAND / OBJ)
    if not root.is_dir():
        return None, None
    trials = sorted(p for p in root.iterdir() if p.is_dir()
                    and (p / "pose_world.npy").exists())
    if not trials:
        return None, None
    t = trials[-1]
    return np.load(t / "pose_world.npy"), str(t)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pose_world", default=None,
                        help="Explicit pose_world.npy path. Default: latest "
                             "trial dir under v7/inspire_left/servingbowl_small/.")
    parser.add_argument("--c2r_dir", default=None,
                        help="Dir containing C2R.npy. Default: same as pose_world "
                             "(latest trial dir).")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--all", action="store_true",
                        help="Show ALL 9 shelf/1 candidates (not just the 5 untried).")
    args = parser.parse_args()

    if args.pose_world:
        pose_world = np.load(args.pose_world)
        c2r_dir = args.c2r_dir or os.path.dirname(args.pose_world)
    else:
        pose_world, trial_dir = _latest_pose_world()
        if pose_world is None:
            sys.exit("no latest trial pose_world found; pass --pose_world")
        print(f"[diag] using pose_world from {trial_dir}")
        c2r_dir = args.c2r_dir or trial_dir
    c2r = load_c2r(c2r_dir)
    obj_pose_robot = np.linalg.inv(c2r) @ pose_world
    obj_pose = obj_pose_robot

    # Manually load the 5 (or 9 with --all) shelf/1 candidates from disk.
    base = Path(get_candidate_path(HAND)) / VERSION / OBJ / SCENE_TYPE / SCENE_ID
    grasp_dirs = sorted([p for p in base.iterdir() if p.is_dir()])
    if not args.all:
        # Drop the ones that already have result.json (the 4 fails).
        grasp_dirs = [g for g in grasp_dirs if not (g / "result.json").exists()]
    print(f"[diag] {len(grasp_dirs)} shelf/1 candidates loaded:")
    for g in grasp_dirs:
        flag = "FAIL_recorded" if (g / "result.json").exists() else "UNTRIED"
        print(f"    {g.name:18s} [{flag}]")

    wrist_obj = np.stack([np.load(g / "wrist_se3.npy") for g in grasp_dirs])
    pregrasp = np.stack([np.load(g / "pregrasp_pose.npy") for g in grasp_dirs])
    grasp = np.stack([np.load(g / "grasp_pose.npy") for g in grasp_dirs])
    wrist_se3 = obj_pose @ wrist_obj   # to world (robot) frame

    # cyl expansion (8x around symmetry axis)
    cyl_axis = get_cyl_axis_local(OBJ)
    cyl_grid = (np.linspace(0, 2*np.pi, 8, endpoint=False)
                if cyl_axis is not None else None)
    if cyl_grid is not None:
        from scipy.spatial.transform import Rotation
        axis = cyl_axis / (np.linalg.norm(cyl_axis) + 1e-12)
        obj_inv = np.linalg.inv(obj_pose)
        new_w, new_p, new_g = [], [], []
        for i in range(len(wrist_se3)):
            for theta in cyl_grid:
                R_cyl = Rotation.from_rotvec(axis * float(theta)).as_matrix()
                R4 = np.eye(4); R4[:3,:3] = R_cyl
                new_w.append(obj_pose @ R4 @ obj_inv @ wrist_se3[i])
                new_p.append(pregrasp[i])
                new_g.append(grasp[i])
        wrist_se3 = np.array(new_w)
        pregrasp = np.array(new_p)
        grasp = np.array(new_g)
        print(f"[diag] expanded x8 cyl → {len(wrist_se3)} candidates")

    # Build scene_cfg for the planner (table-only obstacles).
    scene_cfg = pose_world_to_scene_cfg(pose_world, c2r, OBJ)
    scene_cfg = add_obstacles(scene_cfg, "table")

    # Run planner's IK + lift IK check to mark each as VALID / IK_FAIL / FILTERED.
    print("[diag] warming up planner...")
    planner = GraspPlanner(hand=HAND)

    world_cfg_no_target = {"mesh": {}, "cuboid": dict(scene_cfg["cuboid"])}
    from autodex.planner.planner import _to_curobo_world, _to_curobo_pose
    from curobo.geom.types import WorldConfig
    import torch
    wc = _to_curobo_world(world_cfg_no_target)
    if planner._ik_solver is None:
        planner._init_ik_solver(wc)
    else:
        planner._ik_solver.update_world(WorldConfig.from_dict(wc))

    def _run_ik(poses):
        out = np.zeros(len(poses), dtype=bool)
        for cs in range(0, len(poses), planner.BATCH_SIZE):
            chunk = poses[cs:cs+planner.BATCH_SIZE]
            B = len(chunk)
            if B < planner.BATCH_SIZE:
                pad = planner.BATCH_SIZE - B
                chunk = np.concatenate([chunk, np.tile(chunk[:1], (pad,1,1))], axis=0)
            goal = _to_curobo_pose(chunk, planner._tensor_args.device)
            retract = torch.tensor(planner._init_state, dtype=torch.float32,
                                    device=planner._tensor_args.device
                                    ).unsqueeze(0).repeat(planner.BATCH_SIZE, 1)
            r = planner._ik_solver.solve_batch(goal, retract_config=retract)
            out[cs:cs+B] = r.success.cpu().numpy()[:B]
        return out

    print(f"[diag] running grasp IK on {len(wrist_se3)} candidates...")
    grasp_succ = _run_ik(wrist_se3)
    print(f"    grasp IK: {grasp_succ.sum()}/{len(wrist_se3)}")
    lift_poses = wrist_se3.copy(); lift_poses[:, 2, 3] += 0.05
    lift_succ = _run_ik(lift_poses)
    print(f"    lift  IK: {lift_succ.sum()}/{len(wrist_se3)} (z+5cm)")
    ok = grasp_succ & lift_succ
    ik_failed = ~ok
    filtered = np.zeros(len(wrist_se3), dtype=bool)

    # Per-candidate report (group by original grasp_dir name + cyl_yaw).
    n_per = len(cyl_grid) if cyl_grid is not None else 1
    print(f"\n[diag] per-candidate (cyl_yaw → grasp_ok / lift_ok):")
    for i, gd in enumerate(grasp_dirs):
        marks = []
        for c in range(n_per):
            k = i * n_per + c
            tag = ("OK" if ok[k] else
                   "g✗" if not grasp_succ[k] else "L✗")
            marks.append(tag)
        print(f"    {gd.name:18s}  {' '.join(marks)}")

    # Viser
    vis = ScenePlanVisualizer(scene_cfg, None, port=args.port, hand=HAND)
    vis.add_candidates(wrist_se3, grasp, filtered, ik_failed=ik_failed)
    vis.start_viewer(use_thread=True)
    print(f"\n[diag] viser http://localhost:{args.port}  "
          f"(green=IK ok, yellow=IK fail, slider 0..{len(wrist_se3)-1})")
    try:
        import time
        while True: time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[diag] bye")


if __name__ == "__main__":
    main()
