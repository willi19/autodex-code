#!/usr/bin/env python3
"""Repeat-loop stress test for the full release+reset sequence.

Per cycle: FoundPose init -> plan -> execute -> place -> release -> reset.
``reset`` opens the hand to ``hand_init`` and retracts the arm to ``XARM_INIT``
(see ``RealExecutor.reset``). Loop semantics mirror ``run_auto.py``: planner /
orchestrator / executor are instantiated once outside the loop so curobo's GPU
JIT and the init pipeline only pay their warmup cost on cycle 0.

Prerequisites:
    bash scripts/init_daemons.sh start

Usage:
    python src/execution/reset_test.py --obj attached_container --max_cycles 20 --auto
    python src/execution/reset_test.py --obj brown_ramen --lock_pose --auto
    python src/execution/reset_test.py --obj brown_ramen --hand inspire_left
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import chime
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.io.camera_system.signal_generator import UTGE900
from paradex.io.camera_system.timestamp_monitor import TimestampMonitor
from paradex.utils.system import network_info, get_pc_ip, get_camera_list
from paradex.calibration.utils import save_current_camparam, save_current_C2R, load_c2r

from autodex.utils.path import project_dir
from autodex.planner import GraspPlanner
from autodex.planner.obstacles import add_obstacles
from autodex.planner.visualizer import ScenePlanVisualizer
from autodex.executor.real import RealExecutor, ContactDetected
from autodex.perception.init_orchestrator import InitOrchestrator

from src.execution.scene_cfg import pose_world_to_scene_cfg
from src.execution.label import auto_label_charuco
from src.execution.run_auto import (
    DEFAULT_PC_LIST, ASSETS_BASE, MESH_BASE, CAM_PARAM_ROOT, _load_calib,
)

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logging.getLogger("curobo").setLevel(logging.WARNING)


class _SoftSkip(Exception):
    """Perception/plan failure: log the cycle and continue to the next."""


def _now() -> str:
    return datetime.datetime.now().isoformat()


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--grasp_version", type=str, default="selected_100")
    parser.add_argument("--exp_name", type=str, default="reset_test")
    parser.add_argument("--hand", type=str, default="inspire_left",
                        choices=["allegro", "inspire", "inspire_left"])
    parser.add_argument("--scene", type=str, default="table",
                        choices=["table", "wall", "shelf", "cluttered"])
    parser.add_argument("--success_only", action="store_true")
    parser.add_argument("--viz", action="store_true",
                        help="Launch non-blocking viser preview per cycle")

    # scene-specific args
    parser.add_argument("--wall_gap", type=float, default=0.04)
    parser.add_argument("--wall_angle", type=float, default=0.0)
    parser.add_argument("--clutter_seed", type=int, default=42)
    parser.add_argument("--clutter_min_dist", type=float, default=0.12)
    parser.add_argument("--clutter_max_dist", type=float, default=0.20)
    parser.add_argument("--clutter_n", type=int, default=4)
    parser.add_argument("--shelf_width", type=float, default=0.30)
    parser.add_argument("--shelf_depth", type=float, default=0.30)
    parser.add_argument("--shelf_height", type=float, default=0.30)
    parser.add_argument("--shelf_gap", type=float, default=0.02)
    parser.add_argument("--no_shelf_back", action="store_true")
    parser.add_argument("--no_shelf_sides", action="store_true")
    parser.add_argument("--no_shelf_top", action="store_true")

    # init pipeline args
    parser.add_argument("--pc_list", type=str, nargs="+", default=DEFAULT_PC_LIST)
    parser.add_argument("--port_mask", type=int, default=5006)
    parser.add_argument("--port_pose", type=int, default=5007)
    parser.add_argument("--port_cmd", type=int, default=6893)
    parser.add_argument("--prompt", type=str, default="object on the checkerboard")
    parser.add_argument("--sil_iters", type=int, default=100)
    parser.add_argument("--sil_lr", type=float, default=0.002)
    parser.add_argument("--init_timeout_s", type=float, default=120.0)
    parser.add_argument("--calib_dir", type=str, default=None)
    parser.add_argument("--stream_fps", type=int, default=10)
    parser.add_argument("--stream_warmup_s", type=float, default=2.0)
    parser.add_argument("--port_viser", type=int, default=8080)

    # loop control
    parser.add_argument("--max_cycles", type=int, default=0,
                        help="0 = unlimited")
    parser.add_argument("--cycle_sleep_s", type=float, default=2.0)
    parser.add_argument("--lock_pose", action="store_true",
                        help="Reuse first cycle's pose_world for all subsequent "
                             "cycles (pure mechanical reset stress).")
    parser.add_argument("--auto", action="store_true",
                        help="Skip per-cycle Enter prompt, fully autonomous.")
    parser.add_argument("--video", action="store_true",
                        help="Record synchronized 30fps video on each cycle "
                             "(execute → reset). Requires sync_generator + "
                             "timestamp_monitor — pattern from run_auto.py.")
    parser.add_argument("--video_fps", type=int, default=30)
    args = parser.parse_args()

    # Asset sanity.
    mesh_path = MESH_BASE / args.obj / "raw_mesh" / f"{args.obj}.obj"
    assets_root = ASSETS_BASE / args.obj
    if not mesh_path.exists():
        sys.exit(f"mesh not found: {mesh_path}")
    if not (assets_root / "object_repre/v1" / args.obj / "1/repre.pth").exists():
        sys.exit(f"repre.pth missing for {args.obj}")

    # Calibration.
    calib_dir = (Path(args.calib_dir).expanduser() if args.calib_dir
                 else sorted(CAM_PARAM_ROOT.iterdir())[-1])
    print(f"calib: {calib_dir.name}")
    intrinsics_full, extrinsics_full, H, W = _load_calib(calib_dir)

    pc_ips = [get_pc_ip(p) for p in args.pc_list]
    pc_serials = {p: get_camera_list(p) for p in args.pc_list}
    active = {s for pc in args.pc_list for s in pc_serials[pc]}
    intrinsics_full = {s: v for s, v in intrinsics_full.items() if s in active}
    extrinsics_full = {s: v for s, v in extrinsics_full.items() if s in active}
    print(f"  {len(intrinsics_full)} cams active across {len(args.pc_list)} PCs ({H}x{W})")

    # Hardware stream.
    rcc = remote_camera_controller("reset_test", pc_list=args.pc_list)
    print(f"[stream] starting on {len(args.pc_list)} PCs @ {args.stream_fps} FPS...")
    rcc.start("stream", False, fps=args.stream_fps)
    if args.stream_warmup_s > 0:
        time.sleep(args.stream_warmup_s)

    # Optional video recording (run_auto.py pattern — sync + timestamps).
    sync_generator = None
    timestamp_monitor = None
    if args.video:
        sync_generator = UTGE900(**network_info["signal_generator"]["param"])
        timestamp_monitor = TimestampMonitor(**network_info["timestamp"]["param"])
        print(f"[video] enabled @ {args.video_fps} FPS")

    # Run-root dir.
    scene_prefix = args.scene if args.scene != "table" else ""
    if args.success_only:
        scene_prefix = f"{scene_prefix}_success_only" if scene_prefix else "success_only"
    sub = f"{scene_prefix}/{args.hand}" if scene_prefix else args.hand
    run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(project_dir) / "experiment" / args.exp_name / sub / args.obj / run_ts
    run_root.mkdir(parents=True, exist_ok=True)

    # ── Dump full run configuration so the experiment is self-describing. ────
    import subprocess
    def _git_sha():
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stderr=subprocess.DEVNULL,
            ).decode().strip()
        except Exception:
            return None

    def _git_dirty():
        try:
            out = subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            return bool(out)
        except Exception:
            return None

    run_config = {
        "run_ts": run_ts,
        "argv": sys.argv,
        "args": vars(args),
        "calib_dir": str(calib_dir),
        "mesh_path": str(mesh_path),
        "assets_root": str(assets_root),
        "image_hw": [H, W],
        "active_serials": sorted(intrinsics_full.keys()),
        "pc_serials": pc_serials,
        "pc_ips": pc_ips,
        "git_sha": _git_sha(),
        "git_dirty": _git_dirty(),
        "tau_model_path": str(Path.home() / "shared_data" / "AutoDex"
                              / "weights" / "tau_model" / "inspire_left.pt"),
        "project_dir": str(project_dir),
        "python_version": sys.version,
    }
    _write_json(run_root / "run_config.json", run_config)
    print(f"[run] config -> {run_root}/run_config.json")

    # Init orchestrator (1x).
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

    # Planner + executor (1x each — curobo JIT only warms up on the first plan).
    print("[planner] warming up curobo...")
    planner = GraspPlanner(hand=args.hand)
    print("[executor] connecting to robot...")
    executor = RealExecutor(hand_name=args.hand)

    summary = {
        "obj": args.obj, "hand": args.hand, "scene": args.scene,
        "exp_name": args.exp_name, "run_ts": run_ts,
        "lock_pose": args.lock_pose, "auto": args.auto,
        "cycles": [],
    }
    locked_pose: Optional[np.ndarray] = None
    cycle = 0

    try:
        while True:
            if args.max_cycles and cycle >= args.max_cycles:
                print(f"\n[loop] reached --max_cycles {args.max_cycles}")
                break

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

            cdir = run_root / f"cycle_{cycle:03d}"
            (cdir / "plan").mkdir(parents=True, exist_ok=True)
            save_current_C2R(str(cdir))
            save_current_camparam(str(cdir))

            # Per-trial result dict — populated as the cycle progresses so even
            # on early abort the partial state is persisted to result.json.
            result_json: dict = {
                "cycle": cycle,
                "start": _now(),
                "obj": args.obj,
                "hand": args.hand,
                "scene": args.scene,
                "status": "started",     # overwritten by terminal status below
                "progress": {
                    "perception": None, "plan": None, "execute": None,
                    "charuco": None, "place": None, "release": None, "reset": None,
                },
                "timing": {},
                "files": {
                    "pose_world": "pose_world.npy",
                    "perception_images": "init_capture/",
                    "plan_dir": "plan/",
                    "charuco_images": "label_at_lift/raw/images/",
                    "place_log": "place_mcc_log.csv",
                    "actual_robot": "raw/",       # arm + hand qpos recordings
                    "cam_param": "cam_param/",
                },
            }
            rec = result_json     # keep `rec` name used downstream

            try:
                # 1. Perception.
                if args.lock_pose and locked_pose is not None:
                    pose_world = locked_pose
                    rec["progress"]["perception"] = "locked"
                else:
                    print(f"[cycle {cycle}] Perception...")
                    t0 = time.time()
                    pose_world, perc_timing = orch.trigger_init(
                        prompt=args.prompt,
                        save_capture_dir=str(cdir / "init_capture"),
                        sil_iters=args.sil_iters, sil_lr=args.sil_lr,
                        timeout_s=args.init_timeout_s,
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
                    if args.lock_pose:
                        locked_pose = pose_world
                np.save(cdir / "pose_world.npy", pose_world)

                # 2. Plan.
                print(f"[cycle {cycle}] Planning (scene={args.scene})...")
                t0 = time.time()
                c2r = load_c2r(str(cdir))
                scene_cfg = pose_world_to_scene_cfg(pose_world, c2r, args.obj)
                scene_cfg = add_obstacles(
                    scene_cfg, args.scene,
                    wall_gap=args.wall_gap, wall_angle=args.wall_angle,
                    seed=args.clutter_seed,
                    clutter_min_dist=args.clutter_min_dist,
                    clutter_max_dist=args.clutter_max_dist,
                    clutter_n=args.clutter_n,
                    shelf_width=args.shelf_width, shelf_depth=args.shelf_depth,
                    shelf_height=args.shelf_height, shelf_gap=args.shelf_gap,
                    shelf_back=not args.no_shelf_back,
                    shelf_sides=not args.no_shelf_sides,
                    shelf_top=not args.no_shelf_top,
                )
                result = planner.plan(
                    scene_cfg, args.obj, args.grasp_version,
                    skip_done=False,
                    success_only=args.success_only, hand=args.hand,
                )
                rec["timing"]["plan_s"] = round(time.time() - t0, 2)
                print(f"    plan: {rec['timing']['plan_s']}s  success={result.success}")
                # Save the exact scene_cfg used (after obstacle injection) so
                # planning is reproducible offline.
                _write_json(cdir / "scene_cfg.json", scene_cfg)
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
                rec["grasp_idx"] = (
                    result.scene_info[2] if result.scene_info else None
                )
                if result.timing and "candidate_idx" in result.timing:
                    rec["candidate_idx"] = result.timing["candidate_idx"]

                if args.viz:
                    vis = ScenePlanVisualizer(scene_cfg, result,
                                              port=args.port_viser, hand=args.hand)
                    vis.start_viewer(use_thread=True)

                # 3. Execute (grasp + lift). Record arm/hand throughout.
                raw_dir = str(cdir / "raw")
                # Optional video: pause stream → start video + sync + timestamps.
                if args.video:
                    try:
                        rcc.stop()
                    except Exception:
                        pass
                    video_rel = os.path.join(
                        "AutoDex", "experiment", args.exp_name,
                        sub, args.obj, run_ts, f"cycle_{cycle:03d}", "raw",
                    )
                    rcc.start("video", True, video_rel)
                    timestamp_monitor.start(os.path.join(raw_dir, "timestamps"))
                    sync_generator.start(fps=args.video_fps)
                executor.start_recording(raw_dir)
                print(f"[cycle {cycle}] Execute (grasp + lift)...")
                t0 = time.time()
                s_hand = executor.execute(result)
                rec["progress"]["execute"] = "ok"

                # 3.5 Auto-label probe via charuco board 1 (table-clear check).
                #     stream/video → image mode → restore after.
                # Pause video if running.
                if args.video:
                    try:
                        sync_generator.stop()
                    except Exception:
                        pass
                    try:
                        timestamp_monitor.stop()
                    except Exception:
                        pass
                label_rel = os.path.join(
                    "shared_data", "AutoDex", "experiment", args.exp_name,
                    sub, args.obj, run_ts, f"cycle_{cycle:03d}",
                    "label_at_lift", "raw",
                )
                label_abs = str(cdir / "label_at_lift" / "raw" / "images")
                try:
                    rcc.stop()
                except Exception:
                    pass
                rcc.start("image", False, label_rel)
                rcc.stop()
                time.sleep(0.5)
                auto_succ, label_info = auto_label_charuco(label_abs, required_board="1")
                if label_info.get("reason"):
                    print(f"    [auto-label] FAILED ({label_info['reason']})")
                    rec["progress"]["charuco"] = f"failed: {label_info['reason']}"
                else:
                    print(f"    [auto-label] success={auto_succ}  "
                          f"covered {label_info['covered']}/{label_info['expected']}")
                    rec["progress"]["charuco"] = (
                        "pass" if auto_succ else
                        f"fail (covered {label_info['covered']}/{label_info['expected']})"
                    )
                rec["charuco"] = label_info
                rec["charuco_success"] = bool(auto_succ)
                # Resume video for place→reset segment, else just stream.
                if args.video:
                    video_rel2 = os.path.join(
                        "AutoDex", "experiment", args.exp_name,
                        sub, args.obj, run_ts, f"cycle_{cycle:03d}", "raw_post",
                    )
                    rcc.start("video", True, video_rel2)
                    timestamp_monitor.start(os.path.join(raw_dir, "timestamps_post"))
                    sync_generator.start(fps=args.video_fps)
                else:
                    rcc.start("stream", False, fps=args.stream_fps)

                # 3.6 Place → Release → Reset.
                place_info = executor.place(
                    result, log_path=str(cdir / "place_mcc_log.csv"),
                )
                rec["place"] = place_info
                rec["progress"]["place"] = (
                    "contact_stop" if place_info.get("stopped_on_contact")
                    else "reached_target"
                )
                executor.release(result)
                rec["progress"]["release"] = "ok"
                if auto_succ:
                    reset_log = executor.reset(result, planner=planner, scene_cfg=scene_cfg)
                    rec["progress"]["reset"] = reset_log.get("retract_mode", "replanned")
                else:
                    print(f"    [reset] label failed → fallback (sequential)")
                    reset_log = executor.reset_fallback(result)
                    rec["progress"]["reset"] = "fallback_sequential"
                rec["timing"]["execute_total_s"] = round(time.time() - t0, 2)
                rec["status"] = "ok" if auto_succ else "ok_charuco_fail"
                rec["reset"] = reset_log
                rec["states"] = executor.state_timestamps
                if s_hand is not None:
                    np.save(cdir / "squeeze_hand.npy", s_hand)
                print(f"    reset: {reset_log.get('total_s')}s  "
                      f"final_qpos_err={reset_log.get('final_qpos_err'):.4f}")

            except _SoftSkip:
                pass
            except ContactDetected as e:
                rec["status"] = "approach_contact_stop"
                rec["progress"]["execute"] = f"contact_stop ({e.where})"
                rec["error"] = repr(e)
                rec["contact_where"] = e.where
                rec["contact_tau_dev"] = e.tau_dev.tolist()
                rec["contact_ratio"] = e.ratio.tolist()
                print(f"[cycle {cycle}] CONTACT during execute — trial aborted, "
                      f"moving to next cycle ({e})")
                # Video may still be running from before execute() — stop it
                # before fallback so charuco / fallback have a clean handle.
                if args.video:
                    for fn in (
                        lambda: sync_generator.stop(),
                        lambda: timestamp_monitor.stop(),
                    ):
                        try:
                            fn()
                        except Exception:
                            pass
                try:
                    fb_log = executor.reset_fallback(result)
                    rec["fallback_after_contact"] = fb_log
                    rec["progress"]["reset"] = "fallback_after_contact"
                except Exception as fe:
                    rec["fallback_error"] = repr(fe)
                    rec["progress"]["reset"] = f"fallback_failed: {fe!r}"
                    print(f"    fallback after contact also failed: {fe!r}")
            except Exception as e:
                rec["status"] = "aborted"
                rec["error"] = repr(e)
                try:
                    executor.stop_recording()
                except Exception:
                    pass
                rec["end"] = _now()
                _write_json(cdir / "result.json", rec)
                summary["cycles"].append(rec)
                _write_json(run_root / "summary.json", summary)
                print(f"[cycle {cycle}] ABORTED: {e!r}")
                raise

            # Stop arm/hand + video recording (all started before execute()).
            try:
                executor.stop_recording()
            except Exception:
                pass
            if args.video:
                for fn in (
                    lambda: sync_generator.stop(),
                    lambda: timestamp_monitor.stop(),
                    lambda: rcc.stop(),
                ):
                    try:
                        fn()
                    except Exception:
                        pass
                # Resume stream so the next cycle's init pipeline has live SHM.
                try:
                    rcc.start("stream", False, fps=args.stream_fps)
                except Exception:
                    pass

            # Snapshot final arm pose + qpos so we know where the cycle
            # actually ended up (clear_view target vs reality).
            try:
                arm_data = executor.arm.get_data()
                rec["final_qpos"] = arm_data["qpos"].tolist()
                rec["final_arm_pose"] = arm_data["position"].tolist()
            except Exception as _e:
                rec["final_qpos_error"] = repr(_e)

            rec["end"] = _now()
            _write_json(cdir / "result.json", rec)
            summary["cycles"].append(rec)
            _write_json(run_root / "summary.json", summary)

            n_ok = sum(1 for c in summary["cycles"] if c.get("status") == "ok")
            print(f"    cycle {cycle} done — running OK: {n_ok}/{len(summary['cycles'])}")
            cycle += 1
            time.sleep(args.cycle_sleep_s)

    finally:
        # Final perception capture so the LAST cycle's post-reset object pose
        # is recorded (other cycles get this for free via the next cycle's
        # initial perception). Run only if orchestrator is still alive and at
        # least one cycle attempted a reset.
        try:
            if summary["cycles"]:
                last_cdir = run_root / f"cycle_{summary['cycles'][-1]['cycle']:03d}"
                final_dir = last_cdir / "pose_after_reset"
                final_dir.mkdir(parents=True, exist_ok=True)
                print(f"\n[final] capturing post-reset object pose...")
                fp_world, fp_timing = orch.trigger_init(
                    prompt=args.prompt,
                    save_capture_dir=str(final_dir / "capture"),
                    sil_iters=args.sil_iters, sil_lr=args.sil_lr,
                    timeout_s=args.init_timeout_s,
                )
                if fp_world is not None:
                    np.save(last_cdir / "pose_world_after_reset.npy", fp_world)
                    summary["cycles"][-1]["pose_after_reset"] = {
                        "ok": True, "timing": fp_timing,
                        "file": "pose_world_after_reset.npy",
                    }
                    print(f"  saved -> {last_cdir}/pose_world_after_reset.npy")
                else:
                    reason = (fp_timing or {}).get("reason", "perception_failed")
                    summary["cycles"][-1]["pose_after_reset"] = {
                        "ok": False, "reason": reason,
                    }
                    print(f"  final perception FAILED ({reason})")
                # Re-write the last cycle's result.json with the new field.
                _write_json(last_cdir / "result.json", summary["cycles"][-1])
        except Exception as e:
            print(f"[final] post-reset perception skipped: {e!r}")

        print(f"\n{'='*60}\nSUMMARY: {args.obj} × {len(summary['cycles'])} cycles")
        for c in summary["cycles"]:
            tag = c.get("status", "?")
            extra = ""
            plan_s = (c.get("timing") or {}).get("plan_s")
            if plan_s is not None:
                extra += f"  plan={plan_s}s"
            if isinstance(c.get("reset"), dict) and c["reset"].get("total_s") is not None:
                extra += f"  reset={c['reset']['total_s']}s  err={c['reset'].get('final_qpos_err'):.3f}"
            print(f"  cycle_{c['cycle']:03d}: {tag}{extra}")
        _write_json(run_root / "summary.json", summary)
        print(f"  summary -> {run_root}/summary.json")

        try:
            executor.shutdown()
        except Exception:
            pass
        try:
            orch.close()
        except Exception:
            pass
        if args.video:
            for fn in (
                lambda: sync_generator.end(),
                lambda: timestamp_monitor.end(),
            ):
                try:
                    fn()
                except Exception:
                    pass
        for fn in (rcc.stop, rcc.end):
            try:
                fn()
            except Exception:
                pass


if __name__ == "__main__":
    main()
