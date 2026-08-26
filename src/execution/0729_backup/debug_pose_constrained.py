#!/usr/bin/env python3
"""Offline reproducer for ``planner.plan_pose_constrained``.

Snapshots are auto-saved by ``planner.plan_pose_constrained`` when called
with ``debug_dump_dir="/tmp/pose_constrained_debug"`` (the default in
``run_auto.py``).

Each snapshot is two files::

    {ms_timestamp}.npz         # start_full_qpos, target_wrist_pose, hold_vec_weight, include_obj_obstacle
    {ms_timestamp}_scene.json  # scene_cfg dict

Usage::

    # List snapshots
    ls /tmp/pose_constrained_debug

    # Replay the most recent snapshot
    python src/execution/debug_pose_constrained.py --latest

    # Replay a specific one
    python src/execution/debug_pose_constrained.py --stem 1717000000123

    # Try a different hand
    python src/execution/debug_pose_constrained.py --latest --hand inspire_left
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from autodex.planner import GraspPlanner


DEFAULT_DUMP_DIR = "/tmp/pose_constrained_debug"


def _latest_stem(dump_dir: str) -> str:
    npzs = sorted(glob.glob(os.path.join(dump_dir, "*.npz")))
    if not npzs:
        sys.exit(f"no snapshots under {dump_dir}")
    return Path(npzs[-1]).stem


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=DEFAULT_DUMP_DIR,
                   help="snapshot directory")
    p.add_argument("--stem", default=None,
                   help="snapshot stem (millisecond timestamp). Default: latest")
    p.add_argument("--latest", action="store_true",
                   help="use most recent snapshot (default if --stem omitted)")
    p.add_argument("--hand", default="inspire_left")
    p.add_argument("--include_obj_obstacle", action="store_true",
                   help="override snapshot to include obj mesh as obstacle")
    p.add_argument("--exclude_table", action="store_true",
                   help="also drop the table cuboid from the world")
    args = p.parse_args()

    stem = args.stem or _latest_stem(args.dir)
    npz_path = os.path.join(args.dir, f"{stem}.npz")
    scene_path = os.path.join(args.dir, f"{stem}_scene.json")
    if not os.path.exists(npz_path):
        sys.exit(f"snapshot npz not found: {npz_path}")
    print(f"[debug] loading {npz_path}")
    data = np.load(npz_path)
    start_full_qpos = data["start_full_qpos"]
    # Snapshots saved post-fix use key "target_wrist_pose"; older snapshots
    # used "target_link6_pose" (now-misnamed — content was always wrist-
    # frame because cuRobo's ee_link = wrist). Accept both for back-compat.
    if "target_wrist_pose" in data.files:
        target_wrist_pose = data["target_wrist_pose"]
    else:
        target_wrist_pose = data["target_link6_pose"]
    hold_vec_weight = list(map(float, data["hold_vec_weight"]))
    include_obj_obstacle = bool(data["include_obj_obstacle"])
    if args.include_obj_obstacle:
        include_obj_obstacle = True

    scene_cfg = None
    if os.path.exists(scene_path):
        with open(scene_path) as f:
            scene_cfg = json.load(f)
        print(f"[debug] loaded scene_cfg ({len(scene_cfg.get('cuboid', {}))} cuboids, "
              f"{len(scene_cfg.get('mesh', {}))} meshes)")
        if args.exclude_table:
            if "cuboid" in scene_cfg:
                scene_cfg = dict(scene_cfg)
                scene_cfg["cuboid"] = {}
                print("[debug] removed all cuboids (incl. table)")
    else:
        print(f"[debug] no scene_cfg ({scene_path} missing)")

    print(f"[debug] start_full_qpos shape={start_full_qpos.shape}")
    print(f"[debug] target_wrist_pose=\n{np.array2string(target_wrist_pose, precision=4)}")
    print(f"[debug] hold_vec_weight={hold_vec_weight}  include_obj={include_obj_obstacle}")

    print(f"[debug] initializing planner (hand={args.hand}, no cuda_graph) ...")
    planner = GraspPlanner(hand=args.hand, use_cuda_graph=False)

    # 1) Constraint plan (the actual flow used at runtime).
    print("\n[debug] === try 1: WITH constraint (hold_vec_weight) ===")
    traj = planner.plan_pose_constrained(
        start_full_qpos, target_wrist_pose,
        hold_vec_weight=hold_vec_weight,
        scene_cfg=scene_cfg,
        include_obj_obstacle=include_obj_obstacle,
    )
    print(f"[debug] result: {'OK' if traj is not None else 'None'} "
          f"{'shape='+str(traj.shape) if traj is not None else ''}")

    # 2) Plain IK to the goal pose, no constraint, no plan. Tells us whether
    #    the goal pose itself is reachable (decoupled from constraint metric).
    print("\n[debug] === try 2: plain IK to goal pose (no constraint) ===")
    import torch
    from curobo.types.math import Pose
    from scipy.spatial.transform import Rotation as R
    print(f"[debug] before init: _ik_solver is None? {planner._ik_solver is None}; "
          f"scene_cfg None? {scene_cfg is None}")
    if planner._ik_solver is None and scene_cfg is not None:
        try:
            from autodex.planner.planner import _to_curobo_world
            world_no_obj = _to_curobo_world(scene_cfg)
            world_no_obj["mesh"] = {}
            planner._init_ik_solver(world_no_obj)
            print("[debug] (lazy init ik_solver with no-obj world)")
        except Exception as _e:
            import traceback
            traceback.print_exc()
            print(f"[debug] init ik_solver FAILED: {_e!r}")
    if planner._ik_solver is None:
        print("[debug] _ik_solver still None — skipping")
    else:
        dev = planner._tensor_args.device
        q_xyzw = R.from_matrix(target_wrist_pose[:3, :3]).as_quat()
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]],
                          dtype=np.float32)
        pos = torch.tensor(target_wrist_pose[:3, 3], dtype=torch.float32,
                           device=dev).unsqueeze(0).repeat(planner.BATCH_SIZE, 1)
        quat = torch.tensor(q_wxyz, dtype=torch.float32,
                            device=dev).unsqueeze(0).repeat(planner.BATCH_SIZE, 1)
        ik_goal = Pose(position=pos, quaternion=quat)
        retract = torch.tensor(planner._init_state, dtype=torch.float32,
                                device=dev).unsqueeze(0).repeat(planner.BATCH_SIZE, 1)
        ik_res = planner._ik_solver.solve_batch(ik_goal, retract_config=retract)
        succ = bool(ik_res.success.cpu().numpy().reshape(-1)[0])
        print(f"[debug] IK reachable: {succ}")
        if succ:
            q_sol = ik_res.solution.cpu().numpy()
            if q_sol.ndim == 3:
                q_sol = q_sol[:, 0, :]
            print(f"[debug] IK qpos[:6] = {q_sol[0, :6].round(3)}")
            print(f"[debug] start qpos[:6] = {np.asarray(start_full_qpos)[:6].round(3)}")
            delta = q_sol[0, :6] - np.asarray(start_full_qpos)[:6]
            print(f"[debug] |delta|        = {np.linalg.norm(delta):.4f}")

        # 3) IK with retract = start_qpos (bias near current config so the
        #    trajectory under constraint can connect).
        print("\n[debug] === try 3: IK with retract_config=start_qpos ===")
        start_arm = np.asarray(start_full_qpos)[:planner._init_state.shape[0]]
        retract_start = torch.tensor(
            start_arm, dtype=torch.float32, device=dev
        ).unsqueeze(0).repeat(planner.BATCH_SIZE, 1)
        ik_res2 = planner._ik_solver.solve_batch(ik_goal, retract_config=retract_start)
        succ2 = bool(ik_res2.success.cpu().numpy().reshape(-1)[0])
        print(f"[debug] IK reachable (start-biased): {succ2}")
        if succ2:
            q_sol2 = ik_res2.solution.cpu().numpy()
            if q_sol2.ndim == 3:
                q_sol2 = q_sol2[:, 0, :]
            delta2 = q_sol2[0, :6] - np.asarray(start_full_qpos)[:6]
            print(f"[debug] IK qpos[:6] = {q_sol2[0, :6].round(3)}")
            print(f"[debug] |delta|        = {np.linalg.norm(delta2):.4f}")


if __name__ == "__main__":
    main()
