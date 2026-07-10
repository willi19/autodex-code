#!/usr/bin/env python3
"""Naive drop policy: grasp + lift, then release to pregrasp, open hand to
zeros in mid-air, and sequentially retract the arm.

Per-cycle:
    perception -> classify tabletop pose (before) -> plan
    -> execute (grasp + lift) -> release (reverse squeeze to pregrasp)
    -> reset_fallback (hand -> INSPIRE_INIT all zeros, then sequential
       arm retract to clear_view)
    -> post-drop perception -> classify tabletop pose (after)

Lift failure (any exception during execute) is logged; release / post-perception
are skipped and only ``reset_fallback`` runs before the next cycle.

Prerequisites:
    bash scripts/init_daemons.sh start

Usage:
    python src/experiment/reset/naive_drop.py --obj brown_ramen --auto
    python src/experiment/reset/naive_drop.py --obj brown_ramen --auto --video
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.io.camera_system.signal_generator import UTGE900
from paradex.io.camera_system.timestamp_monitor import TimestampMonitor
from paradex.utils.system import network_info, get_pc_ip, get_camera_list
from paradex.calibration.utils import save_current_camparam, save_current_C2R, load_c2r

from autodex.utils.path import project_dir
from autodex.utils.conversion import cart2se3
from autodex.planner import GraspPlanner
from autodex.planner.obstacles import add_obstacles
from autodex.planner.visualizer import ScenePlanVisualizer
from autodex.executor.real import RealExecutor
from autodex.perception.init_orchestrator import InitOrchestrator

from src.execution.scene_cfg import pose_world_to_scene_cfg
from src.execution.run_auto import (
    DEFAULT_PC_LIST, ASSETS_BASE, MESH_BASE, CAM_PARAM_ROOT, _load_calib,
)
from src.experiment.reset.tabletop_pose import classify_tabletop_pose

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logging.getLogger("curobo").setLevel(logging.WARNING)


# ── Hardcoded defaults (rarely changed across runs) ──────────────────────────
GRASP_VERSION = "table_only"
LIFT_HEIGHT_M = 0.15         # +15cm above grasp pose (matches reorient_drop)
EXP_NAME = "reset_test/naive_drop"
SCENE = "table"
SUCCESS_ONLY = False
PC_LIST = DEFAULT_PC_LIST
PORT_MASK = 5006
PORT_POSE = 5007
PORT_CMD = 6893
PROMPT = "object on the checkerboard"
SIL_ITERS = 100
SIL_LR = 0.002
INIT_TIMEOUT_S = 120.0
POST_INIT_TIMEOUT_S = 60.0
STREAM_FPS = 30
STREAM_WARMUP_S = 2.0
VIDEO_FPS = 30
CYCLE_SLEEP_S = 2.0
POST_DROP_SETTLE_S = 1.0


class _SoftSkip(Exception):
    """Perception/plan/lift failure: log the cycle and continue to the next."""


def _now() -> str:
    return datetime.datetime.now().isoformat()


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _stop_video(rcc, sync_generator, timestamp_monitor):
    """Stop video recording. Order matters:

    1) rcc.stop() FIRST so camera_server_daemon receives "leave video mode".
    2) Brief sleep so the daemon can complete its in-flight frame grab and
       actually exit WaitForNextImage — sync_generator MUST keep firing during
       this transition, otherwise the daemon hangs.
    3) sync_generator.stop() and timestamp_monitor.stop() — safe now that
       cameras are no longer in externally-triggered mode.
    """
    rcc.stop()
    time.sleep(0.3)
    sync_generator.stop()
    timestamp_monitor.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--hand", type=str, default="inspire_left",
                        choices=["allegro", "inspire", "inspire_left"])
    parser.add_argument("--auto", action="store_true",
                        help="Skip per-cycle Enter prompt, fully autonomous.")
    parser.add_argument("--viz", action="store_true",
                        help="Launch viser viewer of plan; kept alive until "
                             "the next cycle's plan replaces it.")
    parser.add_argument("--port_viser", type=int, default=8080)
    args = parser.parse_args()

    # Asset sanity.
    mesh_path = MESH_BASE / args.obj / "raw_mesh" / f"{args.obj}.obj"
    assets_root = ASSETS_BASE / args.obj
    if not mesh_path.exists():
        sys.exit(f"mesh not found: {mesh_path}")
    if not (assets_root / "object_repre/v1" / args.obj / "1/repre.pth").exists():
        sys.exit(f"repre.pth missing for {args.obj}")

    # Calibration.
    calib_dir = sorted(CAM_PARAM_ROOT.iterdir())[-1]
    print(f"calib: {calib_dir.name}")
    intrinsics_full, extrinsics_full, H, W = _load_calib(calib_dir)

    pc_ips = [get_pc_ip(p) for p in PC_LIST]
    pc_serials = {p: get_camera_list(p) for p in PC_LIST}
    active = {s for pc in PC_LIST for s in pc_serials[pc]}
    intrinsics_full = {s: v for s, v in intrinsics_full.items() if s in active}
    extrinsics_full = {s: v for s, v in extrinsics_full.items() if s in active}
    print(f"  {len(intrinsics_full)} cams active across {len(PC_LIST)} PCs ({H}x{W})")

    # Hardware stream. Unique client name per run so a previous crashed run's
    # stale state on the capture PCs doesn't block reconnection.
    client_name = f"naive_drop_{os.getpid()}"
    rcc = remote_camera_controller(client_name, pc_list=PC_LIST)
    print(f"[stream] starting on {len(PC_LIST)} PCs @ {STREAM_FPS} FPS "
          f"(client={client_name})...")
    rcc.start("stream", False, fps=STREAM_FPS)
    time.sleep(STREAM_WARMUP_S)

    sync_generator = UTGE900(**network_info["signal_generator"]["param"])
    timestamp_monitor = TimestampMonitor(**network_info["timestamp"]["param"])
    print(f"[video] @ {VIDEO_FPS} FPS (always on during drop motion)")

    # Obj-root dir: experiment/{EXP_NAME}/{hand}/{obj}/
    # Each trial gets its own timestamped subdir; summary.json is appended at
    # the obj-root across trials in this run (execution_prev pattern).
    sub = args.hand
    obj_root = Path(project_dir) / "experiment" / EXP_NAME / sub / args.obj
    obj_root.mkdir(parents=True, exist_ok=True)

    # Init orchestrator (1x).
    print(f"[orch] init for {args.obj}...")
    orch = InitOrchestrator(
        pc_list=PC_LIST, capture_ips=pc_ips,
        port_mask=PORT_MASK, port_pose=PORT_POSE, port_cmd=PORT_CMD,
    )
    orch.init_object(
        obj_name=args.obj,
        mesh_path=str(mesh_path), assets_root=str(assets_root),
        intrinsics_full=intrinsics_full, extrinsics_full=extrinsics_full,
        image_hw=(H, W), mode="live", pc_serials=pc_serials,
    )

    print("[planner] warming up curobo...")
    planner = GraspPlanner(hand=args.hand)
    print("[executor] connecting to robot...")
    executor = RealExecutor(hand_name=args.hand)

    trials: list = []
    summary_path = obj_root / "summary.json"
    cycle = 0
    vis = None   # ScenePlanVisualizer (kept alive between cycles when --viz)

    # ── Guaranteed cleanup (atexit + signal handlers) ────────────────────
    # Order matters: rcc.stop FIRST so cameras leave video mode before sync
    # generator is silenced. Otherwise cameras wait for triggers forever and
    # daemons get wedged for the next run.
    _cleanup_done = [False]

    def _do_cleanup():
        if _cleanup_done[0]:
            return
        _cleanup_done[0] = True
        print("\n[cleanup] tearing down resources...")
        nonlocal vis

        import threading
        def _call_with_timeout(label, fn, timeout=5):
            done = threading.Event()
            err_holder: list = []
            def _wrap():
                try:
                    fn()
                except Exception as ce:
                    err_holder.append(ce)
                finally:
                    done.set()
            t = threading.Thread(target=_wrap, daemon=True)
            t.start()
            done.wait(timeout=timeout)
            if not done.is_set():
                print(f"[cleanup] {label} TIMED OUT after {timeout}s — skipping")
                return False
            if err_holder:
                print(f"[cleanup] {label} failed: {err_holder[0]!r}")
            return True

        for label, fn in (
            ("vis.stop_viewer", (lambda: vis.stop_viewer()) if vis else None),
            ("executor.stop_recording", executor.stop_recording),
            ("executor.shutdown", executor.shutdown),
            ("orch.close", orch.close),
            ("rcc.stop", rcc.stop),
            ("sync_generator.stop", sync_generator.stop),
            ("timestamp_monitor.stop", timestamp_monitor.stop),
            ("sync_generator.end", sync_generator.end),
            ("timestamp_monitor.end", timestamp_monitor.end),
            ("rcc.end", rcc.end),
        ):
            if fn is None:
                continue
            _call_with_timeout(label, fn, timeout=5)
        print("[cleanup] done — forcing exit")
        os._exit(0)

    def _signal_handler(signum, frame):
        print(f"\n[signal] received {signal.Signals(signum).name}, cleaning up...")
        _do_cleanup()
        # Re-raise default behavior so the process exits with the right code.
        sys.exit(128 + signum)

    atexit.register(_do_cleanup)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)
    # NOTE: SIGKILL (-9) cannot be caught. Avoid `kill -9` on this process —
    # use Ctrl+C or `kill <pid>` (SIGTERM) for clean shutdown.

    try:
        while True:
            print(f"\n{'#'*60}\n# Cycle {cycle}\n{'#'*60}")
            chime.info()
            if not args.auto:
                try:
                    cmd = input(f"[cycle {cycle}] Enter=start, q=quit: ").strip().lower()
                except KeyboardInterrupt:
                    print("\n[loop] KeyboardInterrupt, stopping.")
                    break
                if cmd == "q":
                    break

            trial_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            cdir = obj_root / trial_ts
            (cdir / "plan").mkdir(parents=True, exist_ok=True)
            save_current_C2R(str(cdir))
            save_current_camparam(str(cdir))

            rec: dict = {
                "cycle": cycle,
                "trial_ts": trial_ts,
                "start": _now(),
                "obj": args.obj, "hand": args.hand, "scene": SCENE,
                "status": "started",
                "progress": {
                    "perception": None, "tabletop_before": None, "plan": None,
                    "execute": None, "release": None, "retract": None,
                    "post_perception": None, "tabletop_after": None,
                },
                "timing": {},
                "tabletop_before": None,
                "tabletop_after": None,
                "tabletop_changed": None,
                "drop_quality": None,
                "files": {
                    "pose_world": "pose_world.npy",
                    "pose_world_after_drop": "pose_world_after_drop.npy",
                    "perception_images": "init_capture/",
                    "post_drop_images": "post_drop_capture/",
                    "plan_dir": "plan/",
                    "actual_robot": "raw/",
                    "cam_param": "cam_param/",
                },
            }
            rec["files"]["raw_video"] = "raw/"
            rec["files"]["timestamps"] = "raw/timestamps"

            result = None
            scene_cfg = None
            planned_obj_pose = None
            video_started = False

            try:
                # 1. Perception (always — no lock_pose).
                print(f"[cycle {cycle}] Perception...")
                t0 = time.time()
                pose_world, perc_timing = orch.trigger_init(
                    prompt=PROMPT,
                    save_capture_dir=str(cdir / "init_capture"),
                    sil_iters=SIL_ITERS, sil_lr=SIL_LR,
                    timeout_s=INIT_TIMEOUT_S,
                )
                rec["timing"]["perception_s"] = round(time.time() - t0, 2)
                if perc_timing:
                    rec["timing"]["perception_detail"] = perc_timing
                if pose_world is None:
                    reason = (perc_timing or {}).get("reason", "perception_failed")
                    rec["progress"]["perception"] = f"failed: {reason}"
                    rec["status"] = "perception_failed"
                    rec["reason"] = reason
                    print(f"    perception FAILED ({reason}) — skipping cycle")
                    raise _SoftSkip
                rec["progress"]["perception"] = "ok"
                np.save(cdir / "pose_world.npy", pose_world)

                # 2. Tabletop classification (before).
                c2r = load_c2r(str(cdir))
                pose_robot_before = np.linalg.inv(c2r) @ pose_world
                tb_before = classify_tabletop_pose(pose_robot_before, args.obj)
                rec["tabletop_before"] = tb_before
                if tb_before is None:
                    rec["progress"]["tabletop_before"] = "no_tabletop_data"
                    rec["status"] = "no_tabletop_data"
                    print("    no tabletop data — skipping cycle")
                    raise _SoftSkip
                rec["progress"]["tabletop_before"] = (
                    f"idx={tb_before['idx']} ({tb_before['rot_err_deg']:.1f}deg)"
                )
                print(f"    [tabletop before] idx={tb_before['idx']} "
                      f"({tb_before['filename']}) err={tb_before['rot_err_deg']:.1f}deg")

                # 3. Plan — restrict to grasps for this tabletop idx.
                scene_id = str(tb_before["idx"])
                print(f"[cycle {cycle}] Planning (scene={SCENE}, tabletop_idx={scene_id})...")
                t0 = time.time()
                scene_cfg = pose_world_to_scene_cfg(pose_world, c2r, args.obj)
                scene_cfg = add_obstacles(scene_cfg, SCENE)
                pose_stem = tb_before["filename"].replace(".npy", "")
                # Cylinder freedom (continuous-revolute objects only) — expand
                # candidates by N_cyl rotations around the symmetry axis if
                # the axis is not (nearly) world-vertical at the current pose.
                from autodex.utils.symmetry import (
                    get_cyl_axis_local, get_cyl_yaw_grid,
                )
                _cyl_axis = get_cyl_axis_local(args.obj)
                _cyl_grid = get_cyl_yaw_grid(args.obj)
                result = planner.plan(
                    scene_cfg, args.obj, GRASP_VERSION,
                    skip_done=False,
                    success_only=SUCCESS_ONLY, hand=args.hand,
                    scene_id=scene_id,
                    openpose_pose_stem=pose_stem,
                    cyl_axis_local=_cyl_axis,
                    cyl_yaw_grid=_cyl_grid,
                )
                rec["timing"]["plan_s"] = round(time.time() - t0, 2)
                print(f"    plan: {rec['timing']['plan_s']}s  success={result.success}")
                _write_json(cdir / "scene_cfg.json", scene_cfg)

                # 3.5 Viser preview (if enabled). Always show candidate slider
                #     (red=filtered, green=valid) so collision-failed grasps
                #     are inspectable; on plan success, the planned trajectory
                #     is shown too. Closes the previous cycle's viewer first.
                if args.viz:
                    if vis is not None:
                        try:
                            vis.stop_viewer()
                        except Exception as ve:
                            print(f"[viz] previous viewer stop failed: {ve!r}")
                    try:
                        cand_wrist, _, cand_grasp, cand_filtered = (
                            planner.get_candidates(
                                scene_cfg, args.obj, GRASP_VERSION,
                                success_only=SUCCESS_ONLY,
                                skip_done=False, hand=args.hand,
                                scene_id=scene_id,
                            )
                        )
                        vis = ScenePlanVisualizer(
                            scene_cfg,
                            result if result.success else None,
                            port=args.port_viser, hand=args.hand,
                        )
                        vis.add_candidates(cand_wrist, cand_grasp, cand_filtered)
                        vis.start_viewer(use_thread=True)
                        print(f"[viz] http://localhost:{args.port_viser} "
                              f"(red=filtered, green=valid)")
                    except Exception as cve:
                        print(f"[viz] viewer setup failed: {cve!r}")

                if not result.success:
                    rec["progress"]["plan"] = "failed"
                    rec["status"] = "plan_failed"
                    raise _SoftSkip
                rec["progress"]["plan"] = "ok"

                np.save(cdir / "plan" / "traj.npy", result.traj)
                np.save(cdir / "plan" / "wrist_se3.npy", result.wrist_se3)
                np.save(cdir / "plan" / "pregrasp_pose.npy", result.pregrasp_pose)
                np.save(cdir / "plan" / "grasp_pose.npy", result.grasp_pose)
                if result.timing:
                    _write_json(cdir / "plan" / "timing.json", result.timing)
                rec["scene_info"] = result.scene_info
                rec["candidate_idx"] = (result.timing.get("candidate_idx")
                                        if result.timing else None)

                # 4. Recording start (arm/hand + optional video).
                raw_dir = str(cdir / "raw")
                # Stream stop → video start (synchronized 30 fps).
                rcc.stop()
                video_rel = os.path.join(
                    "AutoDex", "experiment", EXP_NAME,
                    sub, args.obj, trial_ts, "raw",
                )
                rcc.start("video", True, video_rel)
                timestamp_monitor.start(os.path.join(raw_dir, "timestamps"))
                sync_generator.start(fps=VIDEO_FPS)
                video_started = True
                executor.start_recording(raw_dir)

                # 5. Execute (grasp + lift).
                print(f"[cycle {cycle}] Execute (grasp + lift)...")
                t0 = time.time()
                try:
                    s_hand = executor.execute(result, lift_height=LIFT_HEIGHT_M)
                except Exception as e:
                    rec["timing"]["execute_s"] = round(time.time() - t0, 2)
                    # Determine which semantic phase failed:
                    # executor.state_timestamps gets a _log_state at each phase
                    # transition (init / approach / pregrasp / grasp / squeeze
                    # / lift / lift_done), so the last entry tells us where we
                    # were when the exception fired.
                    states = getattr(executor, "state_timestamps", []) or []
                    phase = states[-1]["state"] if states else "unknown"
                    rec["progress"]["execute"] = f"{phase}_failed: {e!r}"
                    rec["progress"]["release"] = "skipped"
                    rec["progress"]["post_perception"] = "skipped"
                    rec["status"] = f"{phase}_failed"
                    rec["error"] = repr(e)
                    rec["fail_phase"] = phase
                    rec["fail_primitive"] = getattr(e, "where", None)
                    print(f"[cycle {cycle}] {phase.upper()} FAILED: {e!r} "
                          f"— reset_fallback only")

                    # Cleanup recording / video so reset_fallback runs cleanly.
                    # Errors here are logged but don't mask the original lift fail.
                    cleanup_errs: list = []
                    try:
                        executor.stop_recording()
                    except Exception as ce:
                        cleanup_errs.append(f"stop_recording: {ce!r}")
                    if video_started:
                        try:
                            _stop_video(rcc, sync_generator, timestamp_monitor)
                            video_started = False
                            rcc.start("stream", False, fps=STREAM_FPS)
                        except Exception as ce:
                            cleanup_errs.append(f"video_stop_or_stream: {ce!r}")
                    if cleanup_errs:
                        rec["cleanup_errors"] = cleanup_errs

                    try:
                        fb_log = executor.reset_fallback(result)
                        rec["reset"] = fb_log
                        rec["progress"]["retract"] = "fallback_after_lift_fail"
                    except Exception as fe:
                        rec["fallback_error"] = repr(fe)
                        rec["progress"]["retract"] = f"fallback_failed: {fe!r}"
                    raise _SoftSkip
                rec["timing"]["execute_s"] = round(time.time() - t0, 2)
                rec["progress"]["execute"] = "ok"
                if s_hand is not None:
                    np.save(cdir / "squeeze_hand.npy", s_hand)

                # Snapshot object pose at the moment of release.
                T_obj_grasp = cart2se3(scene_cfg["mesh"]["target"]["pose"])
                T_obj_in_wrist = np.linalg.inv(result.wrist_se3) @ T_obj_grasp
                T_wrist_now = (executor.arm.get_data()["position"]
                               @ executor._link6_to_wrist)
                planned_obj_pose = T_wrist_now @ T_obj_in_wrist

                # 6. Drop: release(pregrasp) -> openpose -> hand=0 + sequential retract.
                print(f"[cycle {cycle}] Drop: release -> hand=0 -> sequential retract...")
                t0 = time.time()
                try:
                    executor.release(result)
                except Exception as re_e:
                    rec["progress"]["release"] = f"exception: {re_e!r}"
                    rec["release_error"] = repr(re_e)
                    print(f"    release FAILED: {re_e!r}")
                else:
                    rec["timing"]["release_s"] = round(time.time() - t0, 2)
                    rec["progress"]["release"] = "ok"

                # reset_hybrid now slow-interps pregrasp→openpose internally.
                t1 = time.time()
                try:
                    fb_log = executor.reset_hybrid(result, planner, scene_cfg)
                    rec["timing"]["retract_s"] = round(time.time() - t1, 2)
                    rec["reset"] = fb_log
                    rec["progress"]["retract"] = "hybrid"
                    rec["states"] = executor.state_timestamps
                    print(f"    release={rec['timing'].get('release_s', '?')}s  "
                          f"retract={rec['timing']['retract_s']}s  "
                          f"final_qpos_err={fb_log.get('final_qpos_err'):.4f}")
                except Exception as fb_e:
                    rec["timing"]["retract_s"] = round(time.time() - t1, 2)
                    rec["progress"]["retract"] = f"fallback_failed: {fb_e!r}"
                    rec["fallback_error"] = repr(fb_e)
                    rec["status"] = "reset_failed"
                    rec["progress"]["post_perception"] = "skipped"
                    print(f"    reset_fallback FAILED: {fb_e!r} — skipping post-drop perception")
                    raise _SoftSkip

                # 7. Stop recordings → video stop → image snapshot → stream.
                executor.stop_recording()
                _stop_video(rcc, sync_generator, timestamp_monitor)
                video_started = False
                time.sleep(0.5)

                # Single-shot image right after the drop, for visual record.
                image_rel = os.path.join(
                    "shared_data", "AutoDex", "experiment", EXP_NAME,
                    sub, args.obj, trial_ts, "post_drop_snap",
                )
                rcc.start("image", False, image_rel)
                rcc.stop()
                time.sleep(0.5)   # let capture PCs flush PNGs to NFS

                # Stream back on for post-drop perception (init_daemon needs SHM).
                rcc.start("stream", False, fps=STREAM_FPS)

                if POST_DROP_SETTLE_S > 0:
                    time.sleep(POST_DROP_SETTLE_S)

                # 8. Post-drop perception + tabletop_after + drop_quality.
                print(f"[cycle {cycle}] Post-drop perception...")
                t0 = time.time()
                pose_world_after, post_timing = orch.trigger_init(
                    prompt=PROMPT,
                    save_capture_dir=str(cdir / "post_drop_capture"),
                    sil_iters=SIL_ITERS, sil_lr=SIL_LR,
                    timeout_s=POST_INIT_TIMEOUT_S,
                )
                rec["timing"]["post_perception_s"] = round(time.time() - t0, 2)
                if post_timing:
                    rec["timing"]["post_perception_detail"] = post_timing

                if pose_world_after is None:
                    reason = (post_timing or {}).get("reason", "perception_failed")
                    rec["progress"]["post_perception"] = f"failed: {reason}"
                    rec["status"] = "ok_post_perception_fail"
                else:
                    np.save(cdir / "pose_world_after_drop.npy", pose_world_after)
                    rec["progress"]["post_perception"] = "ok"
                    rec["status"] = "ok"

                    pose_robot_after = np.linalg.inv(c2r) @ pose_world_after
                    tb_after = classify_tabletop_pose(pose_robot_after, args.obj)
                    rec["tabletop_after"] = tb_after
                    rec["progress"]["tabletop_after"] = (
                        f"idx={tb_after['idx']} ({tb_after['rot_err_deg']:.1f}deg)"
                        if tb_after else "no_tabletop_data"
                    )
                    if tb_after and tb_before:
                        rec["tabletop_changed"] = bool(
                            tb_after["idx"] != tb_before["idx"]
                        )
                        print(f"    [tabletop after]  idx={tb_after['idx']} "
                              f"({tb_after['filename']}) err={tb_after['rot_err_deg']:.1f}deg "
                              f"changed={rec['tabletop_changed']}")

                    R_a = planned_obj_pose[:3, :3]
                    R_b = pose_robot_after[:3, :3]
                    cos = (np.trace(R_a.T @ R_b) - 1.0) / 2.0
                    cos = float(np.clip(cos, -1.0, 1.0))
                    rot_err = float(np.degrees(np.arccos(cos)))
                    trans_err = float(np.linalg.norm(
                        planned_obj_pose[:3, 3] - pose_robot_after[:3, 3]
                    ))
                    z_drop = float(planned_obj_pose[2, 3] - pose_robot_after[2, 3])
                    rec["drop_quality"] = {
                        "planned_obj_pose_robot": planned_obj_pose.tolist(),
                        "post_drop_obj_pose_robot": pose_robot_after.tolist(),
                        "trans_err_m": trans_err,
                        "rot_err_deg": rot_err,
                        "z_drop_m": z_drop,
                    }
                    print(f"    drop_quality: trans={trans_err*1000:.1f}mm  "
                          f"rot={rot_err:.1f}deg  z={z_drop*1000:.1f}mm")

            except _SoftSkip:
                pass
            except Exception as e:
                rec["status"] = "aborted"
                rec["error"] = repr(e)
                # Best-effort cleanup so next cycle has a clean slate. Failures
                # here are recorded, not silently swallowed.
                cleanup_errs: list = []
                try:
                    executor.stop_recording()
                except Exception as ce:
                    cleanup_errs.append(f"stop_recording: {ce!r}")
                if video_started:
                    try:
                        _stop_video(rcc, sync_generator, timestamp_monitor)
                        video_started = False
                    except Exception as ce:
                        cleanup_errs.append(f"_stop_video: {ce!r}")
                if cleanup_errs:
                    rec["cleanup_errors"] = cleanup_errs
                rec["end"] = _now()
                _write_json(cdir / "result.json", rec)
                trials.append(rec)
                _write_json(summary_path, trials)
                print(f"[cycle {cycle}] ABORTED: {e!r}")
                raise

            # End-of-cycle safety cleanup. Only runs if state suggests it's
            # needed (e.g. _SoftSkip path before normal cleanup). Errors are
            # recorded, not swallowed.
            cycle_cleanup_errs: list = []
            if video_started:
                try:
                    _stop_video(rcc, sync_generator, timestamp_monitor)
                    video_started = False
                    rcc.start("stream", False, fps=STREAM_FPS)
                except Exception as ce:
                    cycle_cleanup_errs.append(f"video_cleanup: {ce!r}")
            if cycle_cleanup_errs:
                rec["cleanup_errors"] = (rec.get("cleanup_errors") or []) + cycle_cleanup_errs

            arm_data = executor.arm.get_data()
            rec["final_qpos"] = arm_data["qpos"].tolist()
            rec["final_arm_pose"] = arm_data["position"].tolist()

            rec["end"] = _now()
            _write_json(cdir / "result.json", rec)
            trials.append(rec)
            _write_json(summary_path, trials)

            n_ok = sum(1 for c in trials if c.get("status") == "ok")
            n_lift_fail = sum(1 for c in trials if c.get("status") == "execute_failed")
            n_change = sum(1 for c in trials if c.get("tabletop_changed") is True)
            print(f"    cycle {cycle} done — ok: {n_ok}/{len(trials)}  "
                  f"lift_fail: {n_lift_fail}  tabletop_changed: {n_change}")
            cycle += 1
            time.sleep(CYCLE_SLEEP_S)

    finally:
        print(f"\n{'='*60}\nSUMMARY: {args.obj} × {len(trials)} trials")
        for c in trials:
            tag = c.get("status", "?")
            extra = ""
            plan_s = (c.get("timing") or {}).get("plan_s")
            if plan_s is not None:
                extra += f"  plan={plan_s}s"
            dq = c.get("drop_quality")
            if dq:
                extra += f"  drop t={dq['trans_err_m']*1000:.0f}mm r={dq['rot_err_deg']:.0f}d"
            if c.get("tabletop_changed") is not None:
                extra += f"  ttchg={c['tabletop_changed']}"
            print(f"  {c.get('trial_ts', '?')}: {tag}{extra}")
        _write_json(summary_path, trials)
        print(f"  summary -> {summary_path}")

        # All hardware cleanup is now in _do_cleanup() — called automatically
        # via atexit / SIGINT / SIGTERM handlers regardless of how we exit
        # (normal end, KeyboardInterrupt, RuntimeError, kill).
        _do_cleanup()


if __name__ == "__main__":
    main()
