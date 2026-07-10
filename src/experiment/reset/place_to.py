#!/usr/bin/env python3
"""Place an object at a user-specified (r, 0, z) position with a target yaw,
preserving the current orientation otherwise.

Per cycle:
    perception -> classify start tabletop -> pick a table_only grasp feasible
    at CURRENT object pose -> execute grasp + lift -> reorient wrist mid-air
    so the held object sits at (r, 0, z_release) with R = Rz(target_yaw) @
    R_current_obj -> release -> reset_hybrid retract

Prerequisites:
    bash scripts/init_daemons.sh start

Usage:
    python src/experiment/reset/place_to.py --obj pepsi --r 0.50 --z 0.05
    python src/experiment/reset/place_to.py --obj servingbowl_small \\
        --r 0.45 --z 0.04 --yaw 90 --auto
"""
from __future__ import annotations

import argparse
import atexit
import datetime
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

import chime
import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.io.camera_system.signal_generator import UTGE900
from paradex.io.camera_system.timestamp_monitor import TimestampMonitor
from paradex.utils.system import network_info, get_pc_ip, get_camera_list
from paradex.calibration.utils import save_current_camparam, save_current_C2R, load_c2r

from autodex.utils.path import project_dir, load_openpose_for_candidates
from autodex.utils.conversion import cart2se3
from autodex.utils.symmetry import get_cyl_axis_local
from autodex.planner import GraspPlanner
from autodex.planner.planner import PlanResult, _to_curobo_world
from autodex.planner.obstacles import TABLE_CUBOID, add_obstacles
from autodex.executor.real import RealExecutor
from autodex.perception.init_orchestrator import InitOrchestrator
from autodex.perception.snapshot_orchestrator import SnapshotOrchestrator

from src.execution.scene_cfg import pose_world_to_scene_cfg
from src.execution.run_auto import (
    DEFAULT_PC_LIST, ASSETS_BASE, MESH_BASE, CAM_PARAM_ROOT, _load_calib,
)
from src.execution.label import auto_label_charuco
from src.experiment.reset.tabletop_pose import classify_tabletop_pose

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logging.getLogger("curobo").setLevel(logging.WARNING)


GRASP_VERSION = "table_only"
LIFT_HEIGHT_M = 0.25
RELEASE_HEIGHT_M = 0.15
TABLE_SURFACE_Z = TABLE_CUBOID["pose"][2] + TABLE_CUBOID["dims"][2] / 2
EXP_NAME = "place_to"
SCENE = "table"
PC_LIST = DEFAULT_PC_LIST
PORT_MASK = 5006
PORT_POSE = 5007
PORT_CMD = 6893
PORT_SNAP = 5009
PORT_SNAP_CMD = 6894
PROMPT = "object on the checkerboard"
SIL_ITERS = 100
SIL_LR = 0.002
INIT_TIMEOUT_S = 120.0
STREAM_FPS = 30
STREAM_WARMUP_S = 2.0
VIDEO_FPS = 30
CHARUCO_BOARD = "1"


def _now() -> str:
    return datetime.datetime.now().isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--hand", type=str, default="inspire_left",
                        choices=["allegro", "inspire", "inspire_left"])
    parser.add_argument("--r", type=float, required=True,
                        help="Target radial x (target xyz=(r, 0, z), robot frame).")
    parser.add_argument("--z", type=float, required=True,
                        help="Target object center z (robot frame).")
    parser.add_argument("--yaw", type=float, default=0.0,
                        help="Target world-z yaw added to current orientation, in degrees.")
    parser.add_argument("--auto", action="store_true",
                        help="Skip per-cycle Enter prompt + charuco auto-label.")
    args = parser.parse_args()

    target_yaw_rad = np.deg2rad(args.yaw)
    print(f"[place_to] target = (r={args.r}, y=0, z={args.z})  "
          f"yaw={args.yaw}°  obj={args.obj}  hand={args.hand}")

    mesh_path = MESH_BASE / args.obj / "raw_mesh" / f"{args.obj}.obj"
    assets_root = ASSETS_BASE / args.obj
    if not mesh_path.exists():
        sys.exit(f"mesh not found: {mesh_path}")
    if not (assets_root / "object_repre/v1" / args.obj / "1/repre.pth").exists():
        sys.exit(f"repre.pth missing for {args.obj}")

    calib_dir = sorted(CAM_PARAM_ROOT.iterdir())[-1]
    print(f"calib: {calib_dir.name}")
    intrinsics_full, extrinsics_full, H, W = _load_calib(calib_dir)

    pc_ips = [get_pc_ip(p) for p in PC_LIST]
    pc_serials = {p: get_camera_list(p) for p in PC_LIST}
    active = {s for pc in PC_LIST for s in pc_serials[pc]}
    intrinsics_full = {s: v for s, v in intrinsics_full.items() if s in active}
    extrinsics_full = {s: v for s, v in extrinsics_full.items() if s in active}

    client_name = f"place_to_{os.getpid()}"
    rcc = remote_camera_controller(client_name, pc_list=PC_LIST)
    rcc.start("stream", False, fps=STREAM_FPS)
    time.sleep(STREAM_WARMUP_S)
    sync_generator = UTGE900(**network_info["signal_generator"]["param"])
    timestamp_monitor = TimestampMonitor(**network_info["timestamp"]["param"])

    obj_root = Path(project_dir) / "experiment" / EXP_NAME / args.hand / args.obj
    obj_root.mkdir(parents=True, exist_ok=True)

    print(f"[orch] init for {args.obj}...")
    orch = InitOrchestrator(
        pc_list=PC_LIST, capture_ips=pc_ips,
        port_mask=PORT_MASK, port_pose=PORT_POSE, port_cmd=PORT_CMD,
    )
    snap_orch = SnapshotOrchestrator(
        pc_list=PC_LIST, capture_ips=pc_ips,
        port_snap=PORT_SNAP, port_cmd=PORT_SNAP_CMD,
    )
    n_cams_total = sum(len(pc_serials[p]) for p in PC_LIST)
    orch.init_object(
        obj_name=args.obj,
        mesh_path=str(mesh_path), assets_root=str(assets_root),
        intrinsics_full=intrinsics_full, extrinsics_full=extrinsics_full,
        image_hw=(H, W), mode="live", pc_serials=pc_serials,
    )

    print("[planner] warming up curobo...")
    planner = GraspPlanner(hand=args.hand)
    from curobo.util.logger import setup_curobo_logger
    setup_curobo_logger("warning")
    executor = RealExecutor(hand_name=args.hand)

    cycle = 0

    _cleanup_done = [False]
    def _cleanup():
        if _cleanup_done[0]: return
        _cleanup_done[0] = True
        for fn in (executor.stop_recording, executor.shutdown, orch.close,
                   rcc.stop, sync_generator.stop, timestamp_monitor.stop,
                   sync_generator.end, timestamp_monitor.end, rcc.end):
            try: fn()
            except Exception: pass
        os._exit(0)
    atexit.register(_cleanup)
    signal.signal(signal.SIGTERM, lambda *a: _cleanup())
    signal.signal(signal.SIGINT,  lambda *a: _cleanup())

    while True:
        cycle += 1
        print(f"\n{'#'*60}\n# Cycle {cycle}\n{'#'*60}")
        chime.info()
        if not args.auto:
            try:
                cmd = input(f"[cycle {cycle}] Enter=start, q=quit: ").strip().lower()
            except KeyboardInterrupt:
                break
            if cmd == "q":
                break

        trial_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        cdir = obj_root / trial_ts
        (cdir / "plan").mkdir(parents=True, exist_ok=True)
        save_current_C2R(str(cdir))
        save_current_camparam(str(cdir))

        rec = {"cycle": cycle, "trial_ts": trial_ts, "start": _now(),
               "obj": args.obj, "hand": args.hand,
               "target": {"r": args.r, "z": args.z, "yaw_deg": args.yaw},
               "status": "started", "progress": {}, "timing": {}}

        try:
            # 1. Perception
            t0 = time.time()
            pose_world, perc_timing = orch.trigger_init(
                prompt=PROMPT,
                save_capture_dir=str(cdir / "init_capture"),
                sil_iters=SIL_ITERS, sil_lr=SIL_LR,
                timeout_s=INIT_TIMEOUT_S,
            )
            rec["timing"]["perception_s"] = round(time.time() - t0, 2)
            if pose_world is None:
                rec["status"] = "perception_failed"
                print(f"    perception FAILED")
                continue
            np.save(cdir / "pose_world.npy", pose_world)

            # 2. Tabletop classify (start) — only for scene_id filtering of
            # candidates. We don't change the object's orientation; just add
            # yaw to it for the placement target.
            c2r = load_c2r(str(cdir))
            pose_robot = np.linalg.inv(c2r) @ pose_world
            tb_before = classify_tabletop_pose(pose_robot, args.obj)
            scene_id = (str(tb_before["idx"]) if tb_before is not None else None)
            pose_stem = (tb_before["filename"].replace(".npy", "")
                          if tb_before is not None else None)
            rec["tabletop_before"] = tb_before

            # 3. Plan — solve_ik on table_only candidates for current pose.
            scene_cfg = pose_world_to_scene_cfg(pose_world, c2r, args.obj)
            scene_cfg = add_obstacles(scene_cfg, SCENE)
            t0 = time.time()
            if planner._motion_gen is not None:
                planner._motion_gen.clear_world_cache()
                planner._motion_gen.reset_seed()

            # Symmetry-axis enumerate (revolute → 8 / discrete → order-N).
            from autodex.utils.symmetry import get_cyl_yaw_grid as _gyg
            cyl_axis = get_cyl_axis_local(args.obj)
            cyl_grid = _gyg(args.obj)

            ik_res = planner.solve_ik(
                scene_cfg, args.obj, GRASP_VERSION,
                hand=args.hand, scene_id=scene_id,
                cyl_axis_local=cyl_axis, cyl_yaw_grid=cyl_grid)
            ik_ok = list(np.where(ik_res["ik_success"])[0])
            np.random.shuffle(ik_ok)
            n_total = ik_res["n_total"]
            planner._ik_solver = None
            print(f"    grasp IK: {len(ik_ok)}/{n_total} feasible")
            if not ik_ok:
                rec["status"] = "no_feasible_grasp"
                continue

            # Load openpose for each candidate (matched to start scene_id).
            if pose_stem is not None:
                openpose_list = load_openpose_for_candidates(
                    args.obj, ik_res["scene_info"], args.hand,
                    GRASP_VERSION, pose_stem)
            else:
                openpose_list = [None] * n_total

            # Target rotation: world-z yaw added to CURRENT object orientation.
            R_current = pose_robot[:3, :3]
            cy, sy = np.cos(target_yaw_rad), np.sin(target_yaw_rad)
            Rz_target = np.array([[cy, -sy, 0],
                                   [sy,  cy, 0],
                                   [0,   0,  1]])
            R_target_obj_world = Rz_target @ R_current
            # Target obj position
            obj_target_xyz_lift    = np.array([args.r, 0.0, args.z + LIFT_HEIGHT_M])
            obj_target_xyz_release = np.array([args.r, 0.0, args.z + RELEASE_HEIGHT_M])

            T_obj_grasp_world = cart2se3(scene_cfg["mesh"]["target"]["pose"])
            scene_lift_pre = {"mesh": {},
                              "cuboid": dict(scene_cfg["cuboid"])}

            world_approach = _to_curobo_world(scene_cfg)
            planner._init_motion_gen(world_approach)
            planner._cached_world = world_approach

            # Candidate loop
            chosen = None
            for cand_idx in ik_ok:
                cand_idx = int(cand_idx)
                wrist_grasp_cand = ik_res["wrist_se3"][cand_idx]
                grasp_q = ik_res["grasp"][cand_idx]
                pregrasp_q = ik_res["pregrasp"][cand_idx]
                T_obj_in_wrist = (np.linalg.inv(wrist_grasp_cand)
                                  @ T_obj_grasp_world)

                # (a) approach with openpose finger config if available
                if planner._world_structure_changed(world_approach):
                    planner._update_world(world_approach)
                    planner._cached_world = world_approach
                approach_goal = ik_res["ik_qpos"][cand_idx].copy()
                if openpose_list[cand_idx] is not None:
                    approach_goal[6:] = openpose_list[cand_idx]
                ok_ap, approach_traj = planner._refine_fingers(
                    planner._init_state, approach_goal)
                if not ok_ap:
                    continue

                # (b) lift
                T_wrist_lift = wrist_grasp_cand.copy()
                T_wrist_lift[2, 3] += LIFT_HEIGHT_M
                cur_qpos_lift = np.concatenate(
                    [approach_traj[-1, :6], pregrasp_q])
                lift_traj, lift_info = planner.plan_wrist_reorient(
                    scene_lift_pre, cur_qpos_lift, T_wrist_lift,
                    hold_hand_qpos=pregrasp_q, n_yaw=8)
                if lift_traj is None:
                    continue

                # (c) reorient to target pose (single-point grid)
                cur_qpos_reorient = lift_traj[-1].copy()
                reor_traj, reor_info = planner.plan_obj_placement(
                    scene_lift_pre, cur_qpos_reorient, T_obj_in_wrist,
                    R_target_obj_world, obj_target_xyz_lift,
                    hold_hand_qpos=pregrasp_q,
                    x_grid=np.array([args.r]),
                    yaw_grid=np.array([0.0]),
                    cyl_yaw_grid=None,
                    skip_plan=False)
                if reor_traj is None:
                    continue

                # (d) descent to release z
                cur_qpos_descent = reor_traj[-1].copy()
                desc_traj, _ = planner.plan_obj_placement(
                    scene_lift_pre, cur_qpos_descent, T_obj_in_wrist,
                    R_target_obj_world, obj_target_xyz_release,
                    hold_hand_qpos=pregrasp_q,
                    x_grid=np.array([args.r]),
                    yaw_grid=np.array([0.0]),
                    cyl_yaw_grid=None,
                    skip_plan=False)
                if desc_traj is None:
                    continue

                # Commit
                chosen = {
                    "cand_idx": cand_idx,
                    "approach": approach_traj, "lift": lift_traj,
                    "reorient": reor_traj, "descent": desc_traj,
                    "wrist_grasp": wrist_grasp_cand,
                    "grasp": grasp_q, "pregrasp": pregrasp_q,
                    "openpose": openpose_list[cand_idx],
                    "T_obj_in_wrist": T_obj_in_wrist,
                    "scene_info": ik_res["scene_info"][cand_idx],
                }
                break

            rec["timing"]["plan_s"] = round(time.time() - t0, 2)
            if chosen is None:
                rec["status"] = "no_grasp_full_chain"
                print(f"    No candidate passed full chain")
                continue

            print(f"    plan: {rec['timing']['plan_s']}s  cand#{chosen['cand_idx']}")
            np.save(cdir / "plan" / "approach_traj.npy", chosen["approach"])
            np.save(cdir / "plan" / "lift_traj.npy",     chosen["lift"])
            np.save(cdir / "plan" / "reorient_traj.npy", chosen["reorient"])
            np.save(cdir / "plan" / "descent_traj.npy",  chosen["descent"])
            np.save(cdir / "plan" / "wrist_se3.npy",     chosen["wrist_grasp"])
            np.save(cdir / "plan" / "T_obj_in_wrist.npy", chosen["T_obj_in_wrist"])

            # 4. Recording / execute (grasp + squeeze, no lift in execute).
            raw_dir = str(cdir / "raw")
            rcc.stop()
            video_rel = os.path.join("AutoDex", "experiment", EXP_NAME,
                                      args.hand, args.obj, trial_ts, "raw")
            rcc.start("full", True, video_rel)
            timestamp_monitor.start(os.path.join(raw_dir, "timestamps"))
            sync_generator.start(fps=VIDEO_FPS)
            executor.start_recording(raw_dir)

            result = PlanResult(
                success=True, traj=chosen["approach"],
                wrist_se3=chosen["wrist_grasp"],
                pregrasp_pose=chosen["pregrasp"],
                grasp_pose=chosen["grasp"],
                scene_info=chosen["scene_info"],
                openpose_pose=chosen["openpose"],
            )
            t0 = time.time()
            try:
                s_hand = executor.execute(result, skip_lift=True)
            except Exception as e:
                rec["status"] = f"execute_failed: {e!r}"
                print(f"    execute FAILED: {e!r}")
                try: executor.stop_recording()
                except Exception: pass
                continue
            rec["timing"]["execute_s"] = round(time.time() - t0, 2)

            # 5. Lift (joint-space replay)
            t1 = time.time()
            executor._log_state("lift")
            lift = chosen["lift"]
            executor._move_joints(lift[:, :6],
                                  np.tile(s_hand[None], (len(lift), 1)))
            rec["timing"]["lift_s"] = round(time.time() - t1, 2)

            # 6. Charuco lift-check (snapshot, --auto only labels here)
            if args.auto:
                label_abs = str(cdir / "label_at_lift" / "raw" / "images")
                _, snap_timing = snap_orch.snap(
                    n_expected=n_cams_total, timeout_s=3.0,
                    save_dir_local=label_abs)
                auto_succ, label_info = auto_label_charuco(
                    label_abs, required_board=CHARUCO_BOARD)
                rec["charuco"] = label_info
                rec["charuco_success"] = bool(auto_succ)
                if not auto_succ:
                    print(f"    [charuco] FAIL — aborting (no place)")
                    try:
                        fb = executor.reset_hybrid(result, planner, scene_cfg)
                        rec["retract"] = fb
                    except Exception as fe:
                        rec["retract_error"] = repr(fe)
                    rec["status"] = "charuco_fail"
                    continue

            # 7. Reorient + descent
            t1 = time.time()
            executor._log_state("reorient")
            reor = chosen["reorient"]
            executor._move_joints(reor[:, :6],
                                  np.tile(s_hand[None], (len(reor), 1)))
            rec["timing"]["reorient_s"] = round(time.time() - t1, 2)

            t1 = time.time()
            executor._log_state("descent")
            desc = chosen["descent"]
            executor._move_joints(desc[:, :6],
                                  np.tile(s_hand[None], (len(desc), 1)))
            rec["timing"]["descent_s"] = round(time.time() - t1, 2)

            # 8. Release + openpose + retract
            t1 = time.time()
            try:
                executor.release(result)
            except Exception as re_e:
                rec["release_error"] = repr(re_e)
            # reset_hybrid handles slow pregrasp→openpose interp internally.
            rec["timing"]["release_s"] = round(time.time() - t1, 2)

            t1 = time.time()
            try:
                fb = executor.reset_hybrid(result, planner, scene_cfg)
                rec["retract"] = fb
            except Exception as rt_e:
                rec["retract_error"] = repr(rt_e)
            rec["timing"]["retract_s"] = round(time.time() - t1, 2)

            executor.stop_recording()
            try: timestamp_monitor.stop()
            except Exception: pass
            try: sync_generator.stop()
            except Exception: pass
            rcc.start("stream", False, fps=STREAM_FPS)
            rec["status"] = "ok"
            print(f"    placed at (r={args.r}, z={args.z}, yaw={args.yaw}°)")
        finally:
            with open(cdir / "result.json", "w") as f:
                json.dump(rec, f, indent=2, default=str)


if __name__ == "__main__":
    main()
