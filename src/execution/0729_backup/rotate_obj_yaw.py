#!/usr/bin/env python3
"""Rotate an object on the table around world-z by a specified yaw angle.

Standalone (no run_auto): perception → grasp via table_only candidate →
lift → repose to (current_obj_xy, current_obj_z, yaw=θ) → place → retract.

Use this as a pre-step when reorient.py reports "rotate obj by Xdeg" — runs
one targeted yaw-rotation cycle then exits.

Usage:
    bash scripts/init_daemons.sh start

    python src/execution/rotate_obj_yaw.py --obj attached_container \\
        --target_yaw_deg 60

Prereqs:
    - init_daemons running on capture1-3, 5, 6
    - table_only candidate pool for ``--obj`` (with stats.json updated by
      run_auto reposition mode).
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
from autodex.utils.conversion import cart2se3
from autodex.utils.coverage import table_only_grasp_order_by_stats
from autodex.utils.symmetry import get_cyl_axis_local, get_cyl_yaw_grid
from autodex.planner import GraspPlanner
from autodex.planner.obstacles import add_obstacles
from autodex.executor.real import RealExecutor
from autodex.perception.init_orchestrator import InitOrchestrator

from src.execution.scene_cfg import pose_world_to_scene_cfg
from src.experiment.reset.tabletop_pose import classify_tabletop_pose


DEFAULT_PC_LIST = ["capture1", "capture2", "capture3", "capture5", "capture6"]
ASSETS_BASE = Path.home() / "shared_data/AutoDex/foundpose_assets"
MESH_BASE = Path.home() / "shared_data/AutoDex/object/paradex"
CAM_PARAM_ROOT = Path.home() / "shared_data/cam_param"


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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obj", required=True)
    p.add_argument("--hand", default="inspire_left")
    p.add_argument("--target_yaw_deg", type=float, required=True,
                   help="Rotate obj by this angle around world z (degrees).")
    p.add_argument("--target_x", type=float, default=0.50,
                   help="Target obj x in robot frame (y is fixed = 0 since "
                        "xarm6 joint 0 covers angle around z). Default 0.55 "
                        "≈ charuco board center x.")
    p.add_argument("--grasp_version", default="table_only",
                   help="Candidate pool to grasp from. ``table_only`` filters "
                        "by current tabletop pose (small pool); ``v7`` uses "
                        "the full v7 pool (much larger, more reach).")
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
    args = p.parse_args()

    target_yaw_rad = np.deg2rad(args.target_yaw_deg)
    print(f"[rotate] target yaw = {args.target_yaw_deg:.1f}° around world z")

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
    rcc.start("stream", False, fps=args.stream_fps)
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

    print("[planner] warmup...")
    planner = GraspPlanner(hand=args.hand)
    print("[executor] connect...")
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
    scene_cfg = pose_world_to_scene_cfg(pose_world, c2r, args.obj)
    scene_cfg = add_obstacles(scene_cfg, "table")
    tb = classify_tabletop_pose(pose_robot, args.obj)
    pose_stem = tb["filename"].replace(".npy", "") if tb else None
    cyl_axis = get_cyl_axis_local(args.obj)
    cyl_grid = get_cyl_yaw_grid(args.obj)
    if args.grasp_version == "table_only":
        cand_order = table_only_grasp_order_by_stats(args.obj, hand=args.hand)
        scene_type_filter = "table"
        priority_map = None
    else:
        # v7 (or other): rank candidates by (past success count desc,
        # remaining coverage count desc). priority_map (used post-IK in
        # planner.plan) sorts but DOES NOT filter — every IK-valid
        # candidate is still considered, but proven-good ones go first.
        from autodex.utils.coverage import (
            load_v7_coverage_map, _disk_success_keys,
        )
        succ_keys = _disk_success_keys(
            args.obj, args.hand, args.grasp_version)
        cov_map = load_v7_coverage_map(
            args.obj, tabletop_pose_stem=pose_stem,
            hand=args.hand, version=args.grasp_version) or {}
        # Boost successful keys by +1000 so they always outrank cov-only.
        priority_map = {k: (1000 if k in succ_keys else 0) + cov_map.get(k, 0)
                        for k in set(cov_map) | set(succ_keys)}
        cand_order = None
        scene_type_filter = None
        if succ_keys:
            print(f"  [order] {len(succ_keys)} prior-success grasps "
                  f"boosted to top of priority_map")
    result = planner.plan(
        scene_cfg, args.obj, args.grasp_version,
        skip_done=False, success_only=False, hand=args.hand,
        scene_id=None, scene_type_filter=scene_type_filter,
        skip_scenes_with_success=False,
        openpose_pose_stem=pose_stem,
        cyl_axis_local=cyl_axis, cyl_yaw_grid=cyl_grid,
        tabletop_pose_stem=pose_stem,
        candidate_order=cand_order,
        priority_map=priority_map,
    )
    # Pre-flight: simulate lift+repose using FK from planned grasp state.
    # If repose to (R_PLACE, 0, lifted_z) with target_yaw is infeasible,
    # bail BEFORE we grasp/lift — otherwise we'd pick the obj up only to
    # discover at L4 that we can't put it down where we promised.
    if result.success:
        R_PLACE = args.target_x
        T_obj_grasp_world = cart2se3(scene_cfg["mesh"]["target"]["pose"])
        T_obj_in_wrist = (np.linalg.inv(result.wrist_se3) @ T_obj_grasp_world)
        # End of execute(): wrist sits at grasp pose lifted by +z (rigid).
        T_wrist_lift = result.wrist_se3.copy()
        T_wrist_lift[2, 3] += 0.10
        obj_z_lifted = float((T_wrist_lift @ T_obj_in_wrist)[2, 3])
        c0, s0 = np.cos(target_yaw_rad), np.sin(target_yaw_rad)
        Rz0 = np.array([[c0, -s0, 0], [s0, c0, 0], [0, 0, 1]])
        T_obj_target_pre = np.eye(4)
        T_obj_target_pre[:3, :3] = Rz0 @ T_obj_grasp_world[:3, :3]
        T_obj_target_pre[:3, 3] = [R_PLACE, 0.0, obj_z_lifted]
        T_wrist_target_pre = (T_obj_target_pre
                               @ np.linalg.inv(T_obj_in_wrist))
        T_wrist_target_pre[2, 3] = T_wrist_lift[2, 3]
        # cuRobo's ee_link = wrist, so ik_pose_batch (despite the
        # historical "link6" name in its arg) actually targets wrist —
        # pass T_wrist_target_pre directly.
        if not bool(planner.ik_pose_batch(T_wrist_target_pre[None])[0]):
            print(f"[rotate] pre-flight: repose target (x={R_PLACE}, "
                  f"y=0, yaw={args.target_yaw_deg:.0f}°) is IK-infeasible. "
                  f"Refusing to grasp.")
            sys.exit(2)
        print(f"  [pre-flight] repose target IK feasible")

    if not result.success:
        print(f"[rotate] grasp plan failed: {result.timing}")
        # Viz: launch ScenePlanVisualizer so user can inspect which candidates
        # were IK/collision-failed at this obj pose.
        try:
            from autodex.planner.visualizer import ScenePlanVisualizer
            wse, preg, _g, filt, ikf = planner.get_candidates(
                scene_cfg, args.obj, args.grasp_version,
                hand=args.hand, scene_type_filter=scene_type_filter,
                cyl_axis_local=cyl_axis, cyl_yaw_grid=cyl_grid,
                tabletop_pose_stem=pose_stem,
                candidate_order=cand_order,
                run_ik=True,
            )
            fv = ScenePlanVisualizer(scene_cfg, None, port=8080, hand=args.hand)
            fv.add_candidates(wse, preg, filt, ik_failed=ikf)
            fv.start_viewer(use_thread=True)
            print(f"  [viz] http://localhost:8080  "
                  f"(yellow=IK fail, red=filtered, slider 0..{len(wse)-1})")
            input(f"  Press Enter to quit: ")
        except Exception as _ve:
            print(f"  [viz] launch failed: {_ve!r}")
        sys.exit(1)
    print(f"  scene_info={result.scene_info}")

    # 3. Execute grasp + lift
    print(f"[3/4] grasp + lift...")
    try: rcc.stop()
    except Exception: pass
    s_hand = executor.execute(result, planner=planner, scene_cfg=scene_cfg)

    # 4. Repose to (R_PLACE, 0, current z) with yaw=θ around world z
    #    (matches run_auto.py reposition mode).
    R_PLACE = args.target_x
    print(f"[4/4] repose to (x={R_PLACE}, y=0, yaw={args.target_yaw_deg:.1f}°) ...")
    T_wrist_now = executor.arm.get_data()["position"] @ executor._link6_to_wrist
    T_obj_grasp_world = cart2se3(scene_cfg["mesh"]["target"]["pose"])
    T_obj_in_wrist = np.linalg.inv(result.wrist_se3) @ T_obj_grasp_world
    T_obj_now = T_wrist_now @ T_obj_in_wrist
    obj_z = float(T_obj_now[2, 3])
    # IMPORTANT: use perception-time obj orientation (not drifted lift orientation)
    # so target tabletop stays canonical.
    R_obj_canonical = T_obj_grasp_world[:3, :3]

    c, s_ = np.cos(target_yaw_rad), np.sin(target_yaw_rad)
    Rz = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1]])
    T_obj_target = np.eye(4)
    T_obj_target[:3, :3] = Rz @ R_obj_canonical
    T_obj_target[:3, 3] = [R_PLACE, 0.0, obj_z]

    np.set_printoptions(precision=3, suppress=True)
    print(f"  current obj pose (robot):  pos={T_obj_now[:3, 3]}")
    print(f"                              R=\n{T_obj_now[:3, :3]}")
    print(f"  target  obj pose (robot):  pos={T_obj_target[:3, 3]}")
    print(f"                              R=\n{T_obj_target[:3, :3]}")
    print(f"  Δ obj yaw (world z) = {args.target_yaw_deg:+.1f}°  "
          f"(xy=({R_PLACE}, 0), z preserved)")

    T_wrist_target = T_obj_target @ np.linalg.inv(T_obj_in_wrist)
    # Force goal wrist z = current wrist z (avoid floating-point drift
    # tripping cuRobo's INVALID_PARTIAL_POSE_COST_METRIC check).
    T_wrist_now_world = (executor.arm.get_data()["position"]
                          @ executor._link6_to_wrist)
    T_wrist_target[2, 3] = T_wrist_now_world[2, 3]

    start_full = np.concatenate([
        np.asarray(executor.arm.get_data()["qpos"][:6], dtype=np.float32),
        np.asarray(result.grasp_pose, dtype=np.float32),
    ])
    traj_repose = planner.plan_pose_constrained(
        start_full, T_wrist_target,
        hold_vec_weight=[0, 0, 0, 0, 0, 1],   # hold z only
        scene_cfg=scene_cfg, include_obj_obstacle=False,
    )
    if traj_repose is not None:
        arm_repose = traj_repose[:, :6]
        hand_repose = np.tile(s_hand, (len(traj_repose), 1))
        executor._move_joints(arm_repose, hand_repose)
        print(f"  repose OK")
    else:
        print(f"  repose plan failed — place at current pose without rotating")

    # Place + release + retract
    place_info = executor.place(result)
    print(f"  place: {place_info}")
    executor.release(result)
    try:
        executor.reset(result, planner, scene_cfg)
    except Exception as e:
        print(f"  reset failed: {e!r}, trying reset_hybrid")
        try:
            executor.reset_hybrid(result, planner, scene_cfg)
        except Exception as e2:
            print(f"  reset_hybrid also failed: {e2!r}, "
                  f"trying reset_fallback")
            try:
                executor.reset_fallback(result)
            except Exception as e3:
                print(f"  reset_fallback also failed: {e3!r}")

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
