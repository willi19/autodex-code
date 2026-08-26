#!/usr/bin/env python3
"""Offline run_auto rehearsal — loads a saved trial's pose_world + C2R, runs
planner.plan + lift/repose/place pre-compute + ScenePlanVisualizer with full
phase obj animation. No camera, no robot, no daemons.

Use to iterate on viz / planning / target-pose code without paying the
~1-minute hardware recovery per cycle.

Usage:
    # latest trial under v7/inspire_left/attached_container
    python src/execution/mock_run_auto.py

    # specific trial
    python src/execution/mock_run_auto.py \\
        --trial_dir ~/shared_data/AutoDex/experiment/v7/inspire_left/attached_container/20260603_150845

    # different obj / version / port
    python src/execution/mock_run_auto.py --obj banana --port 8082
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from autodex.utils.conversion import cart2se3
from autodex.utils.coverage import load_v7_coverage_order
from autodex.utils.path import project_dir
from autodex.utils.robot_config import INSPIRE_LEFT_LINK6_TO_WRIST
from autodex.utils.symmetry import get_cyl_axis_local, get_cyl_yaw_grid
from autodex.planner import GraspPlanner
from autodex.planner.obstacles import add_obstacles
from autodex.planner.visualizer import ScenePlanVisualizer
from paradex.calibration.utils import load_c2r

from src.execution.scene_cfg import pose_world_to_scene_cfg
from src.experiment.reset.tabletop_pose import classify_tabletop_pose


# Hand-specific link6→wrist constant (matches RealExecutor._link6_to_wrist).
LINK6_TO_WRIST = {
    "inspire_left": INSPIRE_LEFT_LINK6_TO_WRIST,
}


def _latest_trial(obj: str, hand: str, version: str) -> Path | None:
    root = Path(project_dir) / "experiment" / version / hand / obj
    if not root.is_dir():
        return None
    trials = sorted(p for p in root.iterdir()
                    if p.is_dir() and (p / "pose_world.npy").exists())
    return trials[-1] if trials else None


def _fk_wrist_factory(planner: GraspPlanner):
    """Closure returning FK at ee_link (= base_link = wrist) for a full qpos."""
    device = planner._tensor_args.device

    def _fk(qpos: np.ndarray) -> np.ndarray:
        kin = planner._motion_gen.kinematics.get_state(
            torch.tensor(qpos, dtype=torch.float32, device=device).unsqueeze(0)
        )
        pos = kin.ee_position[0].detach().cpu().numpy()
        quat = kin.ee_quaternion[0].detach().cpu().numpy()    # wxyz
        Rmat = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
        T = np.eye(4)
        T[:3, :3] = Rmat
        T[:3, 3] = pos
        return T
    return _fk


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obj", default="attached_container")
    p.add_argument("--hand", default="inspire_left")
    p.add_argument("--version", default="v7")
    p.add_argument("--scene", default="table",
                   choices=["table", "wall", "shelf", "cluttered"])
    p.add_argument("--trial_dir", default=None,
                   help="Trial dir with pose_world.npy + cam_param/C2R.npy. "
                        "Default: latest trial for --obj.")
    p.add_argument("--pose_world", default=None,
                   help="Explicit pose_world.npy path (overrides --trial_dir).")
    p.add_argument("--c2r_dir", default=None,
                   help="Dir with C2R.npy. Default: same as trial_dir.")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--R_PLACE", type=float, default=0.55)
    args = p.parse_args()

    # 1. Resolve pose_world + c2r ---------------------------------------------
    if args.pose_world:
        pose_world = np.load(args.pose_world)
        c2r_dir = args.c2r_dir or os.path.dirname(args.pose_world)
    else:
        td = (Path(args.trial_dir).expanduser() if args.trial_dir
              else _latest_trial(args.obj, args.hand, args.version))
        if td is None or not td.is_dir():
            sys.exit(f"no trial dir found for {args.obj}; pass --trial_dir or --pose_world")
        if not (td / "pose_world.npy").exists():
            sys.exit(f"{td}/pose_world.npy missing")
        pose_world = np.load(td / "pose_world.npy")
        c2r_dir = args.c2r_dir or str(td)
        print(f"[mock] using trial {td}")
    c2r = load_c2r(c2r_dir)
    print(f"[mock] pose_world loaded, c2r from {c2r_dir}")

    # 2. scene_cfg ------------------------------------------------------------
    scene_cfg = pose_world_to_scene_cfg(pose_world, c2r, args.obj)
    scene_cfg = add_obstacles(scene_cfg, args.scene)
    pose_robot = np.linalg.inv(c2r) @ pose_world
    tb = classify_tabletop_pose(pose_robot, args.obj)
    pose_stem = tb["filename"].replace(".npy", "") if tb else None
    print(f"[mock] tabletop: {tb}")

    # 3. Planner --------------------------------------------------------------
    print(f"[mock] initializing planner (no cuda_graph)...")
    planner = GraspPlanner(hand=args.hand, use_cuda_graph=False)
    _fk_wrist = _fk_wrist_factory(planner)

    # 4. Candidate ordering (v7 = priority by coverage count) ----------------
    if args.version == "v7":
        from autodex.utils.coverage import load_v7_coverage_map
        priority_map = load_v7_coverage_map(args.obj, tabletop_pose_stem=pose_stem)
        cov_order = None
        scene_type_filter = (args.scene
                             if args.scene in ("wall", "shelf", "box")
                             else None)
        tabletop_filter = pose_stem
        scene_id = None
    else:
        priority_map = None
        cov_order = None
        scene_type_filter = None
        tabletop_filter = None
        scene_id = str(tb["idx"]) if tb else None

    cyl_axis = get_cyl_axis_local(args.obj)
    cyl_grid = get_cyl_yaw_grid(args.obj)
    print(f"[mock] running planner.plan (skip_done=True, mirroring run_auto) ...")
    result = planner.plan(
        scene_cfg, args.obj, args.version,
        skip_done=True, success_only=False, hand=args.hand,
        scene_id=scene_id,
        scene_type_filter=scene_type_filter,
        skip_scenes_with_success=True,
        openpose_pose_stem=pose_stem,
        cyl_axis_local=cyl_axis,
        cyl_yaw_grid=cyl_grid,
        tabletop_pose_stem=tabletop_filter,
        candidate_order=cov_order,
        priority_map=priority_map,
    )
    if not result.success:
        sys.exit(f"[mock] planner.plan FAILED: {result.timing}")
    print(f"[mock] plan ok, scene_info={result.scene_info}, "
          f"traj.shape={result.traj.shape}")

    # Debug print — pose chain
    np.set_printoptions(precision=4, suppress=True)
    T_obj_robot = cart2se3(scene_cfg["mesh"]["target"]["pose"])
    print(f"\n[mock] === pose debug ===")
    print(f"  obj pose (robot frame): pos={T_obj_robot[:3, 3]}")
    print(f"  result.wrist_se3 pos:   {result.wrist_se3[:3, 3]}")

    # 5. Viz: scene + grasp + lift + repose + place
    sv = ScenePlanVisualizer(scene_cfg, result, port=args.port, hand=args.hand)
    grasp_end_qpos = np.asarray(result.traj[-1], dtype=np.float32)
    grasp_end_arm = grasp_end_qpos[:6]
    T_wrist_grasp_end = _fk_wrist(grasp_end_qpos)
    print(f"  T_wrist_grasp_end pos:  {T_wrist_grasp_end[:3, 3]}")
    # Use FK-derived wrist (not planner's result.wrist_se3) for T_obj_in_wrist
    # so the obj viz at grasp end exactly matches scene_cfg obj pose (no jump).
    T_obj_in_wrist = np.linalg.inv(T_wrist_grasp_end) @ T_obj_robot

    def obj_traj_along(robot_traj: np.ndarray) -> np.ndarray:
        out = np.zeros((len(robot_traj), 4, 4))
        for i, q in enumerate(robot_traj):
            Tw = _fk_wrist(np.asarray(q, dtype=np.float32))
            out[i] = Tw @ T_obj_in_wrist
        return out

    # Lift
    lift_wrist = T_wrist_grasp_end.copy()
    lift_wrist[2, 3] += 0.10
    grasp_full = np.concatenate([
        grasp_end_arm, np.asarray(result.grasp_pose, dtype=np.float32)
    ])
    print(f"[mock] computing lift ...")
    lift_traj = planner.plan_pose_constrained(
        grasp_full, lift_wrist,
        hold_vec_weight=[1, 1, 1, 1, 1, 0],
        scene_cfg=scene_cfg, include_obj_obstacle=False,
    )
    if lift_traj is not None:
        sv.add_traj("lift", {"traj_robot": lift_traj},
                    obj_traj={"mesh_target": obj_traj_along(lift_traj)})

        # Repose (v7)
        if args.version == "v7":
            lift_end_qpos = np.asarray(lift_traj[-1], dtype=np.float32)
            lift_end_arm = lift_end_qpos[:6]
            T_wrist_lift_end = _fk_wrist(lift_end_qpos)
            T_obj_lift_end = T_wrist_lift_end @ T_obj_in_wrist
            obj_z_now = float(T_obj_lift_end[2, 3])
            R_obj_canonical = T_obj_robot[:3, :3]
            T_obj_repo = np.eye(4)
            T_obj_repo[:3, :3] = R_obj_canonical
            T_obj_repo[:3, 3] = [args.R_PLACE, 0.0, obj_z_now]
            T_wrist_repo = T_obj_repo @ np.linalg.inv(T_obj_in_wrist)
            T_wrist_repo[2, 3] = T_wrist_lift_end[2, 3]
            lift_full = np.concatenate([
                lift_end_arm, np.asarray(result.grasp_pose, dtype=np.float32)
            ])
            print(f"[mock] computing repose ...")
            repo_traj = planner.plan_pose_constrained(
                lift_full, T_wrist_repo,
                hold_vec_weight=[0, 0, 0, 0, 0, 1],
                scene_cfg=scene_cfg, include_obj_obstacle=False,
            )
            if repo_traj is not None:
                sv.add_traj("repose", {"traj_robot": repo_traj},
                            obj_traj={"mesh_target": obj_traj_along(repo_traj)})

                # Place
                repo_end_qpos = np.asarray(repo_traj[-1], dtype=np.float32)
                repo_end_arm = repo_end_qpos[:6]
                T_wrist_repo_end = _fk_wrist(repo_end_qpos)
                place_wrist = T_wrist_repo_end.copy()
                place_wrist[2, 3] -= 0.10
                repo_full = np.concatenate([
                    repo_end_arm, np.asarray(result.grasp_pose, dtype=np.float32)
                ])
                print(f"[mock] computing place ...")
                place_traj = planner.plan_pose_constrained(
                    repo_full, place_wrist,
                    hold_vec_weight=[1, 1, 1, 1, 1, 0],
                    scene_cfg=scene_cfg, include_obj_obstacle=False,
                )
                if place_traj is not None:
                    sv.add_traj("place", {"traj_robot": place_traj},
                                obj_traj={"mesh_target": obj_traj_along(place_traj)})

    print(f"[mock] viewer http://localhost:{args.port}  Ctrl-C to quit")
    sv.start_viewer(use_thread=True)
    try:
        while True: time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[mock] bye")


if __name__ == "__main__":
    main()
