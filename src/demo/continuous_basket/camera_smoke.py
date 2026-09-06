#!/usr/bin/env python3
"""Verify live ParaDex cameras for the continuous basket demo without robot motion.

The check claims the currently idle camera daemons, arms a free-running stream,
proves every configured camera's frame id advances, writes one non-disruptive
snapshot, and releases the camera controller.  It never connects to an arm or
hand controller.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
import time
from pathlib import Path
from typing import Iterable, List, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
for _paradex_root in (os.environ.get("AUTODEX_PARADEX_ROOT"), str(Path.home() / "paradex")):
    if _paradex_root and (Path(_paradex_root).expanduser() / "paradex").is_dir():
        sys.path.insert(0, str(Path(_paradex_root).expanduser()))
        break

from src.demo.continuous_basket.camera import capture_catalog_snapshot


DEFAULT_PC_LIST = ["capture1", "capture2", "capture3", "capture5", "capture6"]


def advancing_frame_errors(
    before: Mapping[str, object], after: Mapping[str, object], pc_list: Iterable[str],
) -> List[str]:
    """Return health errors when any configured camera failed to advance."""
    errors: List[str] = []
    if bool(after.get("error")):
        errors.append(f"controller error: {after.get('stalled') or after.get('interrupt_msg')}")
    before_pc = before.get("pc") or {}
    after_pc = after.get("pc") or {}
    for pc in pc_list:
        previous = before_pc.get(pc) if isinstance(before_pc, Mapping) else None
        current = after_pc.get(pc) if isinstance(after_pc, Mapping) else None
        if not isinstance(previous, Mapping) or not isinstance(current, Mapping):
            errors.append(f"{pc}: missing health telemetry")
            continue
        if current.get("status") != "ok":
            errors.append(f"{pc}: {current.get('msg') or current.get('status')}")
            continue
        states = current.get("states") or {}
        old_ids = previous.get("frame_ids") or {}
        new_ids = current.get("frame_ids") or {}
        if not new_ids:
            errors.append(f"{pc}: no camera frame ids")
            continue
        for serial, new_id in new_ids.items():
            if states.get(serial) != "CAPTURING":
                errors.append(f"{pc}/{serial}: state={states.get(serial)!r}")
            if int(new_id) <= int(old_ids.get(serial, -1)):
                errors.append(f"{pc}/{serial}: frame id did not advance "
                              f"({old_ids.get(serial)} -> {new_id})")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pc-list", nargs="+", default=DEFAULT_PC_LIST)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--warmup-s", type=float, default=3.0)
    parser.add_argument("--sample-s", type=float, default=1.0)
    parser.add_argument("--min-snapshot-images", type=int, default=20)
    parser.add_argument("--snapshot-timeout-s", type=float, default=15.0)
    parser.add_argument("--snapshot-dir", default=None,
                        help="NFS-visible result directory (default: experiment smoke timestamp)")
    args = parser.parse_args()
    if (args.fps < 1 or args.warmup_s < 0 or args.sample_s <= 0
            or args.min_snapshot_images < 1 or args.snapshot_timeout_s <= 0):
        parser.error("fps/count/time arguments must be positive")

    output = (Path(args.snapshot_dir).expanduser() if args.snapshot_dir else
              Path.home() / "shared_data/AutoDex/experiment/continuous_basket/camera_smoke"
              / dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    from paradex.io.camera_system.remote_camera_controller import remote_camera_controller

    # Camera startup and NAS snapshot propagation are allowed to settle; this
    # only affects the non-motion readiness check, never the robot controller.
    rcc = remote_camera_controller("continuous_basket_camera_smoke", pc_list=args.pc_list,
                                   stall_timeout=max(15.0, args.sample_s * 3))
    armed = False
    try:
        rcc.arm(syncMode=False, fps=args.fps)
        armed = True
        rcc.set_stream(True)
        time.sleep(args.warmup_s)
        before = rcc.get_status()
        time.sleep(args.sample_s)
        after = rcc.get_status()
        errors = advancing_frame_errors(before, after, args.pc_list)
        if errors:
            raise RuntimeError("camera stream check failed: " + "; ".join(errors))
        image_count = capture_catalog_snapshot(
            rcc, output, min_images=args.min_snapshot_images,
            settle_timeout_s=args.snapshot_timeout_s,
            expected_serials=(
                serial for pc in args.pc_list
                for serial in ((after.get("pc") or {}).get(pc, {}).get("frame_ids") or {})
            ),
        )
        print(f"CAMERA_SMOKE_OK cameras={sum(len((after['pc'][pc].get('frame_ids') or {})) for pc in args.pc_list)} "
              f"snapshot_images={image_count} path={output}")
    finally:
        if armed:
            try:
                rcc.stop()
            except Exception as exc:
                print(f"[cleanup] rcc.stop failed: {exc!r}", file=sys.stderr)
        try:
            rcc.end()
        except Exception as exc:
            print(f"[cleanup] rcc.end failed: {exc!r}", file=sys.stderr)


if __name__ == "__main__":
    main()
