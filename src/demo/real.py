"""
Demo: perceive object, plan grasp, and execute on real robot.

Modes:
    --mode auto  (default) Autonomous execution with velocity-limited trajectory
    --mode gui   Interactive Tkinter GUI for step-by-step control

Usage:
    python src/demo/real.py --obj attached_container --ref_idx 000
    python src/demo/real.py --obj attached_container --ref_idx 000 --mode gui
"""
import datetime
import json
import os
import time
import argparse

import numpy as np

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.io.camera_system.signal_generator import UTGE900
from paradex.io.camera_system.timestamp_monitor import TimestampMonitor
from paradex.calibration.utils import save_current_camparam, save_current_C2R
from paradex.utils.system import network_info

from autodex.utils.path import project_dir, obj_path
from autodex.utils.scene import get_scene_image_dict_template
from autodex.utils.conversion import se32cart
from autodex.planner import GraspPlanner
from autodex.executor import RealExecutor


def _rcc_start(rcc, mode, sync_mode, save_path=None, fps=30):
    """Start a capture, translating the retired 'stream'/'video' modes.

    paradex's camera API dropped both: a capture arms in 'acquire' and its
    outputs are toggled as SINKS (set_stream / set_record). The capture PCs
    reject the old names outright, and the rejection LATCHES an error on every
    camera that only a daemon-side reload clears -- so a single call from one
    stale script leaves the next run with no frames at all.
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


parser = argparse.ArgumentParser()
parser.add_argument("--obj", type=str, required=True, help="Object name")
parser.add_argument("--ref_idx", type=str, required=True, help="Reference index for pose estimation")
parser.add_argument("--mode", type=str, default="auto", choices=["auto", "gui"],
                    help="Execution mode: 'auto' (default) or 'gui'")
parser.add_argument("--grasp_version", type=str, default="selected_100",
                    help="Candidate version")
parser.add_argument("--exp_name", type=str, default="demo",
                    help="Experiment name for data saving")
parser.add_argument("--lift_height", type=float, default=0.12,
                    help="Lift height in meters (default 0.12)")
args = parser.parse_args()


# ── Camera system ────────────────────────────────────────────────────────────

rcc = remote_camera_controller("test_lookup_obstacle")
sync_generator = UTGE900(**network_info["signal_generator"]["param"])
timestamp_monitor = TimestampMonitor(**network_info["timestamp"]["param"])


def capture_scene(exp_name, obj_name, dir_idx, ref_idx):
    """Capture images, estimate 6D object pose, return scene_cfg."""
    exp_dir = os.path.join(project_dir, "experiment", exp_name, obj_name, dir_idx)
    os.makedirs(exp_dir, exist_ok=True)

    # Capture single frame from all cameras
    rcc.start("image", False,
              os.path.join("shared_data/AutoDex", "experiment", exp_name, obj_name, dir_idx, "raw"))
    rcc.stop()

    # Save calibration
    save_current_C2R(exp_dir)
    save_current_camparam(exp_dir)

    # 6D pose estimation via template matching
    ref_dir = os.path.join(project_dir, "..", "shared_data", "AutoDex",
                           "object_pose_template", obj_name, ref_idx)
    scene_cfg = get_scene_image_dict_template(exp_dir, ref_dir, obj_name)
    if scene_cfg is None:
        return None

    scene_cfg["cuboid"]["table"] = {
        "dims": [2, 3, 0.2],
        "pose": [1.1, 0, -0.1 + 0.037, 1, 0, 0, 0],
    }
    return scene_cfg


def capture_label(exp_name, obj_name, dir_idx):
    """Capture a label image and ask operator for success/failure."""
    rcc.start("image", False,
              os.path.join("shared_data/AutoDex", "experiment", exp_name, obj_name, dir_idx, "label", "raw"))
    rcc.stop()
    while True:
        answer = input("Press 'y' if the grasp was successful, 'n' if not: ").lower()
        if answer == "y":
            return 1
        if answer == "n":
            return 0


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    dir_idx = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = os.path.join(project_dir, "experiment", args.exp_name, args.obj, dir_idx)

    # 1. Perceive scene
    t0 = time.time()
    scene_cfg = capture_scene(args.exp_name, args.obj, dir_idx, args.ref_idx)
    if scene_cfg is None:
        print("Object 6D pose not found, exiting.")
        rcc.end()
        sync_generator.end()
        timestamp_monitor.end()
        exit(0)
    print(f"Scene captured ({time.time() - t0:.2f}s)")

    # 2. Plan grasp
    t0 = time.time()
    planner = GraspPlanner()
    result = planner.plan(scene_cfg, args.obj, args.grasp_version)
    print(f"Planning done ({time.time() - t0:.2f}s) — {'success' if result.success else 'FAILED'}")

    if not result.success:
        print("No feasible grasp found.")
        rcc.end()
        sync_generator.end()
        timestamp_monitor.end()
        exit(0)

    print(f"Executing grasp (mode={args.mode}) from scene: {result.scene_info}")

    # 3. Execute
    executor = RealExecutor(mode=args.mode)

    # Start recording
    _rcc_start(rcc, "video", True, os.path.join("shared_data/AutoDex", "experiment", args.exp_name, args.obj, dir_idx, "raw"))
    timestamp_monitor.start(os.path.join(exp_dir, "raw", "timestamps"))
    executor.start_recording(os.path.join(exp_dir, "raw"))
    sync_generator.start(fps=30)

    squeeze_hand = executor.execute(result, lift_height=args.lift_height)

    # Stop recording
    sync_generator.stop()
    rcc.stop()
    timestamp_monitor.stop()

    # 4. Label
    succ = capture_label(args.exp_name, args.obj, dir_idx)
    json.dump(
        {"scene_info": result.scene_info, "success": bool(succ)},
        open(os.path.join(exp_dir, "result.json"), "w"),
    )
    if squeeze_hand is not None:
        np.save(os.path.join(exp_dir, "squeeze_hand.npy"), squeeze_hand)

    # 5. Release and return
    executor.release(result)
    executor.shutdown()

    # Cleanup
    timestamp_monitor.end()
    sync_generator.end()
    rcc.end()

    print(f"Done. Result: {'SUCCESS' if succ else 'FAIL'}")
    print(f"Data saved to: {exp_dir}")