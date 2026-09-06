#!/usr/bin/env python3
"""Rotate an object on the table around world-z by a specified yaw angle.

Standalone (no run_auto): perception → grasp via v8 candidate →
lift → repose to (current_obj_xy, current_obj_z, yaw=θ) → place → retract.

Use this as a pre-step when reorient.py reports "rotate obj by Xdeg" — runs
one targeted yaw-rotation cycle then exits.

Usage:
    bash scripts/init_daemons.sh start

    python src/execution/rotate_obj_yaw.py --obj attached_container \\
        --target_yaw_deg 60

Prereqs:
    - init_daemons running on capture1-3, 5, 6
    - v8 candidate pool for ``--obj``.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from paradex.io.robot_controller import get_arm, get_hand  # noqa: F401
from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.utils.system import network_info, get_pc_ip, get_camera_list
from paradex.calibration.utils import save_current_C2R, save_current_camparam, load_c2r

from autodex.utils.path import project_dir
from autodex.utils.conversion import cart2se3, se32cart
from autodex.utils.robot_config import CHARUCO_BOARD_11_CENTER_XY
from autodex.utils.symmetry import get_cyl_axis_local, get_cyl_yaw_grid
from autodex.planner import GraspPlanner
from autodex.planner.obstacles import add_obstacles
from autodex.utils.path import get_obj_root
from autodex.perception.init_orchestrator import InitOrchestrator

from src.execution.scene_cfg import pose_world_to_scene_cfg
from src.experiment.reset.tabletop_pose import classify_tabletop_pose
from src.execution.franka_executor import PLACE_VERTICAL_TRAVEL_M


def _rcc_start(rcc, mode, sync_mode, save_path=None, fps=30):
    """Start a capture, translating the retired 'stream'/'video' modes.

    paradex's camera API dropped both: a capture arms in 'acquire' and its
    outputs are toggled as SINKS. The capture PCs reject the old names, and the
    rejection LATCHES an error on every camera that only a daemon-side reload
    clears -- so one call from a stale script poisons the next run too.
    """
    if mode == "stream":
        rcc.arm(syncMode=sync_mode, fps=fps)
        rcc.set_stream(True)
    elif mode == "full":
        # "full" was video AVI + SHM stream at once (snapshot_daemon reads the
        # stream while the AVI records). Both are just sinks now.
        rcc.arm(syncMode=sync_mode, fps=fps)
        rcc.set_record(save_path=save_path, on=True)
        rcc.set_stream(True)
    elif mode == "video":
        rcc.arm(syncMode=sync_mode, fps=fps)
        rcc.set_record(save_path=save_path, on=True)
    else:                       # 'image' is still a real capture mode
        rcc.start(mode, sync_mode, save_path, fps=fps)



DEFAULT_PC_LIST = ["capture1", "capture2", "capture3", "capture5", "capture6"]
ASSETS_BASE = Path.home() / "shared_data/AutoDex/foundpose_assets"
MESH_BASE = Path.home() / "shared_data/AutoDex/object/paradex"
CAM_PARAM_ROOT = Path.home() / "shared_data/cam_param"
# Carry height after grasp. The distinct placement stroke is fixed below.
TRANSFER_LIFT_HEIGHT = 0.10

# The centroid is the maximum-clearance placement for the board itself.  Its
# measurement provenance is documented with CHARUCO_BOARD_11_CENTER_XY.
CHARUCO_BOARD_CENTER_X, CHARUCO_BOARD_CENTER_Y = CHARUCO_BOARD_11_CENTER_XY

def _planner_robot(arm: str, hand: str) -> str:
    """Return the cuRobo config for the selected physical arm/hand pair."""
    if arm == "xarm":
        return hand
    if hand != "inspire":
        raise SystemExit("FR3 rotation currently supports only --hand inspire")
    return "fr3_inspire"


def _load_calib(calib_dir):
    with open(calib_dir / "intrinsics.json") as f:
        intr_raw = json.load(f)
    with open(calib_dir / "extrinsics.json") as f:
        extr_raw = json.load(f)
    intrinsics_full, extrinsics_full = {}, {}
    for s, d in intr_raw.items():
        intrinsics_full[s] = {
            "K_orig": np.asarray(d["original_intrinsics"], dtype=np.float64).reshape(3, 3),
            "K_undist": np.asarray(d["intrinsics_undistort"], dtype=np.float64).reshape(3, 3),
            "dist_params": np.asarray(d["dist_params"], dtype=np.float64).reshape(-1),
            "width": int(d["width"]), "height": int(d["height"]),
        }
    for s, ext in extr_raw.items():
        a = np.asarray(ext, dtype=np.float64).reshape(-1)
        a = (np.vstack([a.reshape(3, 4), [0, 0, 0, 1]]) if a.size == 12 else a.reshape(4, 4))
        extrinsics_full[s] = a
    first = next(iter(intrinsics_full.values()))
    return intrinsics_full, extrinsics_full, int(first["height"]), int(first["width"])


def _rotation_wrist_targets(
    wrist_grasp: np.ndarray,
    obj_grasp: np.ndarray,
    target_x: float,
    target_y: float,
    target_yaw_rad: float,
    lift_height: float = TRANSFER_LIFT_HEIGHT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build lift, transfer, pre-place, and 10 cm descent wrist goals.

    A carry height is not a release height. The final two goals are therefore
    the table-height release wrist +10 cm and the release wrist itself, so the
    preflight validates the same perpendicular placement stroke as the live
    Franka executor.
    """
    wrist_grasp = np.asarray(wrist_grasp, dtype=np.float64)
    obj_grasp = np.asarray(obj_grasp, dtype=np.float64)
    obj_in_wrist = np.linalg.inv(wrist_grasp) @ obj_grasp

    wrist_lift = wrist_grasp.copy()
    wrist_lift[2, 3] += float(lift_height)
    obj_z_lifted = float((wrist_lift @ obj_in_wrist)[2, 3])

    c, s = np.cos(target_yaw_rad), np.sin(target_yaw_rad)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    obj_target = np.eye(4)
    obj_target[:3, :3] = Rz @ obj_grasp[:3, :3]
    obj_target[:3, 3] = [float(target_x), float(target_y), obj_z_lifted]

    wrist_transfer = obj_target @ np.linalg.inv(obj_in_wrist)
    # ``plan_pose_constrained(... hold z)`` preserves the lift wrist's z.
    # Assign it explicitly to avoid sub-millimetre matrix-chain drift.
    wrist_transfer[2, 3] = wrist_lift[2, 3]

    obj_release = obj_target.copy()
    obj_release[2, 3] = obj_grasp[2, 3]
    wrist_descend = obj_release @ np.linalg.inv(obj_in_wrist)
    wrist_preplace = wrist_descend.copy()
    wrist_preplace[2, 3] += PLACE_VERTICAL_TRAVEL_M
    return wrist_lift, wrist_transfer, wrist_preplace, wrist_descend


def _preflight_rotation_motion(
    planner: GraspPlanner,
    result,
    scene_cfg: dict,
    target_x: float,
    target_y: float,
    target_yaw_rad: float,
    arm_dof: int,
    retract_goal_arm_qpos: np.ndarray | None = None,
) -> tuple[bool, str]:
    """Dry-run lift → transfer → descend from a selected grasp trajectory.

    Endpoint IK filtering is cheap enough to apply to every candidate. This
    stronger check is run only for candidates that passed that filter, and
    catches a joint-space trajectory failure before the physical grasp.
    """
    obj_grasp = cart2se3(scene_cfg["mesh"]["target"]["pose"])
    wrist_lift, wrist_transfer, wrist_preplace, wrist_descend = _rotation_wrist_targets(
        result.wrist_se3, obj_grasp, target_x, target_y, target_yaw_rad)

    def _full(qpos: np.ndarray) -> np.ndarray:
        return np.concatenate([
            np.asarray(qpos[:arm_dof], dtype=np.float32),
            np.asarray(result.grasp_pose, dtype=np.float32),
        ])

    stages = (
        ("lift", wrist_lift, [1, 1, 1, 1, 1, 0]),
        ("repose", wrist_transfer, [0, 0, 0, 0, 0, 1]),
        ("preplace", wrist_preplace, [0, 0, 0, 0, 0, 0]),
        ("descend_10cm", wrist_descend, [1, 1, 1, 1, 1, 0]),
    )
    qpos = np.asarray(result.traj[-1], dtype=np.float32)
    for name, wrist_goal, hold_vec_weight in stages:
        traj = planner.plan_pose_constrained(
            _full(qpos), wrist_goal, hold_vec_weight=hold_vec_weight,
            scene_cfg=scene_cfg, include_obj_obstacle=False,
        )
        if traj is None:
            return False, name
        qpos = np.asarray(traj[-1], dtype=np.float32)

    # Preflight the planned +10cm departure against the placed object too.
    obj_in_wrist = np.linalg.inv(result.wrist_se3) @ obj_grasp
    released = wrist_descend @ obj_in_wrist
    placed_scene = dict(scene_cfg)
    placed_scene["mesh"] = dict(scene_cfg.get("mesh", {}))
    placed_scene["mesh"]["target"] = dict(scene_cfg["mesh"]["target"])
    placed_scene["mesh"]["target"]["pose"] = se32cart(released).tolist()
    post_release_high = wrist_descend.copy()
    post_release_high[2, 3] += PLACE_VERTICAL_TRAVEL_M
    post_start = np.concatenate([
        qpos[:arm_dof], np.asarray(result.pregrasp_pose, dtype=np.float32)])
    post_lift = planner.plan_pose_constrained(
        post_start, post_release_high,
        hold_vec_weight=[1, 1, 1, 1, 1, 0],
        scene_cfg=placed_scene, include_obj_obstacle=True)
    if post_lift is None:
        return False, "post_release_lift_10cm"
    retract = planner.plan_js_to_init(
        placed_scene, post_lift[-1, :arm_dof],
        start_hand_qpos=np.asarray(result.pregrasp_pose, dtype=np.float32),
        goal_arm_qpos=retract_goal_arm_qpos,
    )
    if retract is None:
        return False, "post_release_retract"
    return True, ""


def rotate_from_live_scene(
    *,
    obj: str,
    hand: str,
    arm: str,
    grasp_version: str,
    planner: GraspPlanner,
    executor,
    scene_cfg: dict,
    target_x: float,
    target_yaw_deg: float,
    target_y: float = CHARUCO_BOARD_CENTER_Y,
    tabletop_pose_stem: str | None = None,
    candidate_order: list | None = None,
    priority_map: dict | None = None,
    scene_type_filter: str | None = None,
    scene_id: str | None = None,
    success_only: bool = False,
    skip_done: bool = False,
    skip_scenes_with_success: bool = False,
    cyl_axis_local: np.ndarray | None = None,
    cyl_yaw_grid: np.ndarray | None = None,
    held_speed_scale: float = 0.25,
    rcc=None,
) -> dict:
    """Rotate/reposition using a scene that has already been perceived.

    This is the in-process core of the standalone CLI below.  It deliberately
    accepts the live planner, executor, camera controller, and ``scene_cfg``:
    a recovery pipeline can move the object without creating a second camera
    controller, FoundPose orchestrator, CUDA planner, or robot connection.
    The caller owns lifecycle/stream restart and should run perception again
    after a successful rotation before starting its next task trial.
    """
    if held_speed_scale <= 0:
        raise ValueError("held_speed_scale must be positive")
    target_yaw_rad = np.deg2rad(target_yaw_deg)
    adof = getattr(executor, "arm_dof", 6)

    # Filter the exact grasp variants by all three endpoint IK checks first.
    # This avoids closing the hand before discovering that the final descend
    # has no solution.
    wse, preg, grasp, filt, ikf, scene_infos = planner.get_candidates(
        scene_cfg, obj, grasp_version,
        success_only=success_only, skip_done=skip_done, hand=hand,
        scene_id=scene_id, scene_type_filter=scene_type_filter,
        cyl_axis_local=cyl_axis_local, cyl_yaw_grid=cyl_yaw_grid,
        skip_scenes_with_success=skip_scenes_with_success,
        tabletop_pose_stem=tabletop_pose_stem,
        candidate_order=candidate_order,
        run_ik=True, return_scene_info=True,
    )
    base_ok = ~(filt | ikf)
    endpoint_ok = np.zeros(len(wse), dtype=bool)
    endpoint_stage_ok = np.zeros((4, len(wse)), dtype=bool)
    obj_grasp = cart2se3(scene_cfg["mesh"]["target"]["pose"])
    base_idx = np.flatnonzero(base_ok)
    if len(base_idx) > 0:
        endpoint_groups = [[], [], [], []]  # lift, transfer, pre-place, release
        for idx in base_idx:
            targets = _rotation_wrist_targets(
                wse[idx], obj_grasp, target_x, target_y, target_yaw_rad)
            for group, target in zip(endpoint_groups, targets):
                group.append(target)
        endpoint_batch = np.concatenate(
            [np.asarray(group, dtype=np.float64) for group in endpoint_groups],
            axis=0,
        )
        endpoint_values = planner.ik_pose_batch(endpoint_batch).reshape(4, -1)
        endpoint_stage_ok[:, base_idx] = endpoint_values
        endpoint_ok = endpoint_stage_ok.all(axis=0)

    print(f"  [rotate pre-flight] grasp+short-lift={int(base_ok.sum())}/{len(wse)}  "
          f"lift(10cm)={int(endpoint_stage_ok[0].sum())}  "
          f"repose={int(endpoint_stage_ok[1].sum())}  "
          f"preplace(+10cm)={int(endpoint_stage_ok[2].sum())}  "
          f"descend(-10cm)={int(endpoint_stage_ok[3].sum())}  "
          f"all-endpoints={int(endpoint_ok.sum())}")

    candidate_indices = [int(i) for i in np.flatnonzero(endpoint_ok)]
    if priority_map is not None:
        candidate_indices.sort(
            key=lambda i: -priority_map.get(
                tuple(str(x) for x in scene_infos[i]), 0))

    if tabletop_pose_stem is not None:
        from autodex.utils.path import load_openpose_for_candidates
        openpose_list = load_openpose_for_candidates(
            obj, scene_infos, hand, grasp_version, tabletop_pose_stem)
    else:
        openpose_list = [None] * len(scene_infos)

    # Plan the whole held-object motion before moving the real arm. If a
    # candidate's constrained path fails, omit only that variant and continue.
    result = None
    preflight_rejections = []
    remaining = candidate_indices.copy()
    while remaining:
        idx = np.asarray(remaining, dtype=np.int64)
        attempt_priority = {
            tuple(str(x) for x in scene_infos[orig_idx]): len(idx) - rank
            for rank, orig_idx in enumerate(idx)
        }
        attempt = planner.plan(
            scene_cfg, obj, grasp_version,
            skip_done=skip_done, success_only=success_only, hand=hand,
            scene_id=scene_id, scene_type_filter=scene_type_filter,
            skip_scenes_with_success=skip_scenes_with_success,
            openpose_pose_stem=None,
            tabletop_pose_stem=tabletop_pose_stem,
            priority_map=attempt_priority,
            candidate_override=(
                wse[idx], preg[idx], grasp[idx],
                [scene_infos[i] for i in idx],
                [openpose_list[i] for i in idx],
            ),
        )
        if not attempt.success:
            print("  [rotate pre-flight] no remaining endpoint-feasible "
                  "candidate has an approach trajectory")
            break

        selected_local_idx = int(attempt.timing.get("candidate_idx", 0))
        if not 0 <= selected_local_idx < len(remaining):
            raise RuntimeError(
                "planner returned an invalid candidate index during rotation "
                f"pre-flight: {selected_local_idx} not in [0, {len(remaining)})")
        full_motion_ok, failed_stage = _preflight_rotation_motion(
            planner, attempt, scene_cfg, target_x, target_y, target_yaw_rad, adof,
            retract_goal_arm_qpos=getattr(executor, "_clear_view", None))
        if full_motion_ok:
            result = attempt
            print(f"  [rotate pre-flight] selected {attempt.scene_info}: "
                  "lift → repose → preplace → descend → post-release lift → retract feasible")
            break

        preflight_rejections.append((attempt.scene_info, failed_stage))
        print(f"  [rotate pre-flight] reject {attempt.scene_info}: "
              f"{failed_stage} trajectory infeasible; trying next candidate")
        del remaining[selected_local_idx]

    if result is None:
        return {
            "success": False,
            "reason": "rotation_preflight_failed",
            "preflight_rejections": preflight_rejections,
            "n_candidates": int(len(wse)),
            "n_endpoint_feasible": int(endpoint_ok.sum()),
        }

    if rcc is not None:
        try:
            rcc.stop()
        except Exception as exc:
            print(f"[rotate] rcc.stop before motion failed: {exc!r}")

    try:
        execute_kwargs = {}
        if arm == "franka":
            execute_kwargs["held_speed_scale"] = held_speed_scale
        s_hand = executor.execute(result, planner=planner, scene_cfg=scene_cfg,
                                  **execute_kwargs)
    except Exception as exc:
        print(f"[rotate] execute failed: {exc!r}")
        try:
            executor.reset_fallback(result, planner=planner, scene_cfg=scene_cfg)
        except Exception as recovery_exc:
            print(f"[rotate] execute recovery failed: {recovery_exc!r}")
        return {"success": False, "reason": "rotation_execute_failed",
                "exception": repr(exc), "result": result}

    # Build the same world-z yaw target used by the standalone program.
    T_wrist_now = executor.arm.get_data()["position"] @ executor._link6_to_wrist
    obj_in_wrist = np.linalg.inv(result.wrist_se3) @ obj_grasp
    obj_now = T_wrist_now @ obj_in_wrist
    c, s = np.cos(target_yaw_rad), np.sin(target_yaw_rad)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    obj_target = np.eye(4)
    obj_target[:3, :3] = Rz @ obj_grasp[:3, :3]
    obj_target[:3, 3] = [target_x, target_y, float(obj_now[2, 3])]
    wrist_target = obj_target @ np.linalg.inv(obj_in_wrist)
    wrist_target[2, 3] = T_wrist_now[2, 3]

    print(f"[rotate] transfer to (x={target_x:.3f}, y={target_y:.3f}, "
          f"yaw={target_yaw_deg:.1f}°) ...")
    start_full = np.concatenate([
        np.asarray(executor.arm.get_data()["qpos"][:adof], dtype=np.float32),
        np.asarray(result.grasp_pose, dtype=np.float32),
    ])
    traj_repose = planner.plan_pose_constrained(
        start_full, wrist_target,
        hold_vec_weight=[0, 0, 0, 0, 0, 1],
        scene_cfg=scene_cfg, include_obj_obstacle=False,
    )
    repose_ok = traj_repose is not None
    if repose_ok:
        hold = np.tile(s_hand, (len(traj_repose), 1))
        move_kwargs = {"speed": held_speed_scale} if arm == "franka" else {}
        if arm == "franka":
            print(f"[franka] held-object speed scale: {held_speed_scale:.2f} "
                  "(rotate repose)")
        executor._move_joints(traj_repose[:, :adof], hold, **move_kwargs)
    else:
        print("[rotate] transfer plan failed after grasp — placing at current pose")

    place_kwargs = {}
    if arm == "franka":
        # Place at the original tabletop height, not ``current_z - 10cm``:
        # the carry/rotate height is independent of the mandatory 10 cm
        # perpendicular placement stroke. FrankaExecutor.place() first moves
        # to this wrist +10cm, then descends vertically by exactly 10cm.
        obj_place_target = obj_target.copy()
        obj_place_target[2, 3] = obj_grasp[2, 3]
        place_kwargs["grasp_wrist"] = (
            obj_place_target @ np.linalg.inv(obj_in_wrist))
    place_info = executor.place(result, planner=planner, scene_cfg=scene_cfg,
                                **place_kwargs)
    if arm != "franka":
        executor.release(result)
    try:
        executor.reset(result, planner, scene_cfg)
    except Exception as exc:
        print(f"[rotate] reset failed: {exc!r}; trying reset_fallback")
        try:
            executor.reset_fallback(result, planner=planner, scene_cfg=scene_cfg)
        except Exception as recovery_exc:
            print(f"[rotate] reset fallback failed: {recovery_exc!r}")

    # A generated descent path is not enough: only restart the normal pipeline
    # when the object actually reached its intended table-height window. A
    # contact stop significantly above the target is a failed placement, not a
    # safe new tabletop state to perceive from.
    descended = float(place_info.get("descended", 0.0))
    descend_target = float(place_info.get("target", 0.0))
    placed = (place_info.get("mode") != "plan_failed"
              and descend_target - descended <= 0.005)
    success = bool(repose_ok and placed)
    return {
        "success": success,
        "reason": None if success else "rotation_repose_or_place_failed",
        "result": result,
        "scene_info": result.scene_info,
        "place": place_info,
        "target": {"x": target_x, "y": target_y,
                   "yaw_deg": target_yaw_deg},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obj", required=True)
    p.add_argument("--hand", default="inspire_left")
    p.add_argument("--arm", choices=["xarm", "franka"], default="xarm",
                   help="Physical arm used for the grasp/repose cycle.")
    p.add_argument("--target_yaw_deg", type=float, required=True,
                   help="Rotate obj by this angle around world z (degrees).")
    p.add_argument("--target_x", type=float, default=CHARUCO_BOARD_CENTER_X,
                   help="Target obj x in robot frame. Default is the measured "
                        "center of floor Charuco board 11 (0.608 m).")
    p.add_argument("--target_y", type=float, default=CHARUCO_BOARD_CENTER_Y,
                   help="Target obj y in robot frame. Default is the measured "
                        "center of floor Charuco board 11 (0.153 m).")
    p.add_argument("--grasp_version", default="v8",
                   help="v8 candidate/tabletop asset contract")
    p.add_argument("--pc_list", nargs="+", default=DEFAULT_PC_LIST)
    p.add_argument("--port_mask", type=int, default=5006)
    p.add_argument("--port_pose", type=int, default=5007)
    p.add_argument("--port_cmd", type=int, default=6893)
    p.add_argument("--prompt", default="object on the checkerboard")
    p.add_argument("--sil_iters", type=int, default=100)
    p.add_argument("--sil_lr", type=float, default=0.002)
    p.add_argument("--init_timeout_s", type=float, default=120.0)
    p.add_argument("--stream_fps", type=int, default=10)
    p.add_argument("--stream_warmup_s", type=float, default=2.0)
    p.add_argument("--held_speed_scale", type=float, default=0.25,
                   help="Franka trajectory-speed scale from grasp closure through "
                        "release (lift, yaw transfer, and descend).")
    args = p.parse_args()
    if args.grasp_version != "v8":
        p.error("rotate_obj_yaw supports only --grasp_version v8")
    if args.held_speed_scale <= 0:
        p.error("--held_speed_scale must be positive")
    planner_robot = _planner_robot(args.arm, args.hand)

    print(f"[rotate] target yaw = {args.target_yaw_deg:.1f}° around world z")
    if args.arm == "franka":
        print(f"[rotate] held-object speed scale = {args.held_speed_scale:.2f}x")

    mesh_path = MESH_BASE / args.obj / "raw_mesh" / f"{args.obj}.obj"
    assets_root = ASSETS_BASE / args.obj
    if not mesh_path.exists():
        sys.exit(f"mesh not found: {mesh_path}")
    if not (assets_root / "object_repre/v1" / args.obj / "1/repre.pth").exists():
        sys.exit(f"repre.pth missing for {args.obj}")

    calib_dir = sorted(CAM_PARAM_ROOT.iterdir())[-1]
    print(f"calib: {calib_dir.name}")
    intrinsics_full, extrinsics_full, H, W = _load_calib(calib_dir)
    pc_ips = [get_pc_ip(pc) for pc in args.pc_list]
    pc_serials = {pc: get_camera_list(pc) for pc in args.pc_list}
    active = {s for pc in args.pc_list for s in pc_serials[pc]}
    intrinsics_full = {s: v for s, v in intrinsics_full.items() if s in active}
    extrinsics_full = {s: v for s, v in extrinsics_full.items() if s in active}
    print(f"  {len(intrinsics_full)} cams active")

    rcc = remote_camera_controller("rotate_yaw", pc_list=args.pc_list)
    print(f"[stream] start...")
    _rcc_start(rcc, "stream", False, fps=args.stream_fps)
    time.sleep(args.stream_warmup_s)

    print(f"[orch] init for {args.obj}...")
    orch = InitOrchestrator(
        pc_list=args.pc_list, capture_ips=pc_ips,
        port_mask=args.port_mask, port_pose=args.port_pose, port_cmd=args.port_cmd,
    )
    orch.init_object(
        obj_name=args.obj,
        mesh_path=str(mesh_path), assets_root=str(assets_root),
        intrinsics_full=intrinsics_full, extrinsics_full=extrinsics_full,
        image_hw=(H, W), mode="live", pc_serials=pc_serials,
    )

    print(f"[planner] warmup ({planner_robot})...")
    planner = GraspPlanner(hand=planner_robot)
    print("[executor] connect...")
    if args.arm == "franka":
        from src.execution.franka_executor import FrankaExecutor
        executor = FrankaExecutor(hand_name=args.hand)
        executor.set_speed_profile_planner(planner)
        # Match run_auto: keep the FR3 outside the camera views before the
        # perception snapshot that establishes the object pose.
        executor.home(clear_view=True)
    else:
        from autodex.executor.real import RealExecutor
        executor = RealExecutor(hand_name=args.hand)
    dir_idx = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(project_dir) / "experiment" / "rotate_obj_yaw" / args.obj / dir_idx
    out_dir.mkdir(parents=True, exist_ok=True)
    save_current_C2R(str(out_dir))
    save_current_camparam(str(out_dir))

    # 1. Perception
    print(f"[1/4] perception...")
    save_capture_dir = str(out_dir / "init_capture")
    pose_world, perc_t = orch.trigger_init(
        prompt=args.prompt,
        save_capture_dir=save_capture_dir,
        sil_iters=args.sil_iters, sil_lr=args.sil_lr,
        timeout_s=args.init_timeout_s,
    )
    if pose_world is None:
        sys.exit(f"[rotate] perception failed: {perc_t}")
    np.save(out_dir / "pose_world.npy", pose_world)
    c2r = load_c2r(str(out_dir))
    pose_robot = np.linalg.inv(c2r) @ pose_world
    print(f"  obj pos (robot): {pose_robot[:3, 3].round(3)}")

    # 2. Plan grasp
    print(f"[2/4] plan grasp (version={args.grasp_version}) ...")
    # v8 candidates were generated against object_processing; planning with
    # the paradex mesh instead makes every candidate read as colliding.
    obj_root = get_obj_root(args.grasp_version)
    scene_cfg = pose_world_to_scene_cfg(pose_world, c2r, args.obj, obj_root)
    scene_cfg = add_obstacles(scene_cfg, "table")
    tb = classify_tabletop_pose(pose_robot, args.obj, obj_root)
    pose_stem = tb["filename"].replace(".npy", "") if tb else None
    cyl_axis = get_cyl_axis_local(args.obj)
    cyl_grid = get_cyl_yaw_grid(args.obj)
    # Rank candidates by (past success count desc, remaining coverage count
    # desc).  The priority map sorts post-IK but never hides a valid candidate.
    from autodex.utils.coverage import load_coverage_map, _disk_success_keys
    succ_keys = _disk_success_keys(
        args.obj, args.hand, args.grasp_version, arm=args.arm)
    cov_map = load_coverage_map(
        args.obj, tabletop_pose_stem=pose_stem,
        hand=args.hand, version=args.grasp_version, arm=args.arm) or {}
    priority_map = {k: (1000 if k in succ_keys else 0) + cov_map.get(k, 0)
                    for k in set(cov_map) | set(succ_keys)}
    cand_order = None
    scene_type_filter = None
    if succ_keys:
        print(f"  [order] {len(succ_keys)} prior-success grasps "
              f"boosted to top of priority_map")
    rotate_info = rotate_from_live_scene(
        obj=args.obj, hand=args.hand, arm=args.arm,
        grasp_version=args.grasp_version,
        planner=planner, executor=executor, scene_cfg=scene_cfg,
        target_x=args.target_x, target_y=args.target_y,
        target_yaw_deg=args.target_yaw_deg,
        tabletop_pose_stem=pose_stem, candidate_order=cand_order,
        priority_map=priority_map, scene_type_filter=scene_type_filter,
        cyl_axis_local=cyl_axis, cyl_yaw_grid=cyl_grid,
        held_speed_scale=args.held_speed_scale, rcc=rcc,
    )
    if not rotate_info["success"]:
        print(f"[rotate] FAILED: {rotate_info['reason']}")
        sys.exit(1)
    print(f"[done] obj rotated by {args.target_yaw_deg:.1f}°. "
          f"Output dir: {out_dir}")

    try: executor.shutdown()
    except Exception: pass
    try: orch.close()
    except Exception: pass
    for fn in (rcc.stop, rcc.end):
        try: fn()
        except Exception: pass


if __name__ == "__main__":
    main()
