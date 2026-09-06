#!/usr/bin/env python3
"""Run perception + planning pipeline — real robot execution loop.

Usage:
    python src/execution/run_perception.py --depth da3

    # Interactive:
    #   1. Enter object name (e.g. attached_container)
    #   2. Press Enter to capture + run perception + planning
    #   3. Enter new object name to switch, or 'q' to quit
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.execution_prev.daemon.perception_pipeline import PerceptionPipeline

logging.basicConfig(level=logging.INFO, format='[%(name)s] %(message)s')

MESH_ROOT = Path.home() / "shared_data/object_6d/data/mesh"

# Daemons are NAS-path workers, not per-camera services: any host can process
# any serial, so this list is a compute budget rather than a camera mapping.
SAM3_HOSTS = [
    ("192.168.0.101", 5001),   # capture1
    ("192.168.0.102", 5001),   # capture2
    ("192.168.0.103", 5001),   # capture3
]
# capture4 (192.168.0.104) is gone: it answers no ssh and paradex's system
# config raises KeyError('capture4'). Leaving it here costs a 30 s ZMQ timeout
# on every chunk that lands on it.
FPOSE_HOSTS = [
    ("192.168.0.105", 5003),   # capture5
    ("192.168.0.106", 5003),   # capture6
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--depth", type=str, default="da3", choices=["da3", "stereo"])
    parser.add_argument("--prompt", type=str, default="object on the checkerboard")
    parser.add_argument("--sil_iters", type=int, default=100)
    parser.add_argument("--sil_lr", type=float, default=0.002)
    args = parser.parse_args()

    print(f"Depth: {args.depth}")
    print("Enter object name to start, 'q' to quit.\n")

    pipeline = None
    current_obj = None

    while True:
        try:
            if current_obj is None:
                user_input = input("Object name: ").strip()
            else:
                user_input = input(f"[{current_obj}] Enter to run, object name to switch, 'q' to quit: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.lower() == 'q':
            break

        # New object name
        if user_input and "/" not in user_input:
            if not (MESH_ROOT / user_input).exists():
                print(f"Object not found: {user_input}")
                continue
            current_obj = user_input
            if pipeline is None:
                pipeline = PerceptionPipeline(
                    sam3_hosts=SAM3_HOSTS,
                    fpose_hosts=FPOSE_HOSTS,
                    obj_name=current_obj,
                    depth_method=args.depth,
                )
            else:
                pipeline.change_object(current_obj)
            print(f"Object: {current_obj}")
            continue

        # Empty input or path — run perception
        if current_obj is None:
            print("Set object first")
            continue

        if user_input:
            # User gave a capture_dir path
            capture_dir = Path(user_input).expanduser()
        else:
            # TODO: capture from cameras
            # capture_dir = capture_from_cameras(current_obj)
            print("Camera capture not implemented yet. Enter capture_dir path:")
            path_input = input("> ").strip()
            if not path_input:
                continue
            capture_dir = Path(path_input).expanduser()

        if not capture_dir.exists():
            print(f"Not found: {capture_dir}")
            continue

        # Run perception
        pose_world, timing = pipeline.run(
            capture_dir=str(capture_dir),
            prompt=args.prompt,
            sil_iters=args.sil_iters,
            sil_lr=args.sil_lr,
        )

        if pose_world is not None:
            np.save(str(capture_dir / "pose_world.npy"), pose_world)
            with open(capture_dir / "timing.json", "w") as f:
                json.dump(timing, f, indent=2)
            print(f"\nPose saved. Timing: total={timing['total']:.1f}s "
                  f"(SAM3={timing['sam3']:.1f} Depth={timing['depth']:.1f} "
                  f"FPose={timing['fpose']:.1f} Select={timing['select']:.1f} "
                  f"Sil={timing['sil']:.1f})")
        else:
            print("FAILED: no pose estimated")

    if pipeline:
        pipeline.close()
    print("Done.")


if __name__ == "__main__":
    main()
