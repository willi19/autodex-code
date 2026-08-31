#!/usr/bin/env python3
"""Upload one continuous-basket recording from a capture PC.

Run this script *on a capture PC* after the robot run.  Unlike ParaDex's
generic uploader, it selects one run directory and processes one video at a
time.  This matters on the rig: one undistortion map occupies enough GPU
memory that the generic four-worker uploader can exhaust a capture PC's GPU.

The actual transform and upload remain ParaDex's established
``undistort_raw_video`` routine.  A successful upload removes only its source
AVI, exactly as the standard ParaDex uploader does.
"""
from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_EXPERIMENT_ROOT = Path(
    "AutoDex/experiment/continuous_basket_demo/franka_inspire"
)


def recording_video_paths(
    capture_roots: Iterable[Path], *, experiment_root: Path, run_id: str,
    stage: str = "capture",
) -> list[Path]:
    """Return only raw videos belonging to the requested continuous run."""
    run_rel = experiment_root / run_id / "raw" / stage / "videos"
    paths: list[Path] = []
    for root in capture_roots:
        paths.extend(path for path in (root / run_rel).glob("*.avi") if path.is_file())
    return sorted(paths)


class _Progress(dict):
    """Minimal progress sink required by ParaDex's single-video routine."""

    def __setitem__(self, key, value):  # type: ignore[override]
        super().__setitem__(key, value)
        message = value.get("message", "") if isinstance(value, dict) else ""
        if message:
            print(f"  {Path(str(key)).name}: {message}", flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="e.g. banana_continuous_003")
    parser.add_argument(
        "--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT,
        help="path below each local captures{1,2} root",
    )
    parser.add_argument("--stage", default="capture")
    parser.add_argument(
        "--dry-run", action="store_true", help="list selected AVI files without uploading"
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    # Import only on the capture PC, where the ParaDex/flir_env dependencies live.
    from paradex.utils.path import capture_path_list, shared_dir
    from paradex.video.raw_video_processor import undistort_raw_video

    capture_roots = [Path(path).expanduser() for path in capture_path_list]
    videos = recording_video_paths(
        capture_roots,
        experiment_root=args.experiment_root,
        run_id=args.run_id,
        stage=args.stage,
    )
    session_rel = args.experiment_root / args.run_id
    camparam = Path(shared_dir).expanduser() / session_rel / "cam_param" / "intrinsics.json"

    if not camparam.is_file():
        print(f"UPLOAD_RECORDING_FAILED missing cam-param: {camparam}", file=sys.stderr)
        return 2
    if not videos:
        print(f"UPLOAD_RECORDING_FAILED no raw AVI for {session_rel}", file=sys.stderr)
        return 3

    print(f"[upload] run={args.run_id} videos={len(videos)} workers=1")
    for video in videos:
        print(video)
    if args.dry_run:
        return 0

    failures: list[str] = []
    for index, video in enumerate(videos, start=1):
        print(f"[upload {index}/{len(videos)}] {video.name}", flush=True)
        progress = _Progress()
        result = undistort_raw_video(str(video), progress, str(video))
        status = progress.get(str(video), {}).get("status")
        if status != "completed":
            failures.append(f"{video}: {result}")
            print(f"[upload] FAILED {result}", file=sys.stderr)
        else:
            print(f"[upload] OK {result}")

        # The standard routine allocates its undistortion map on CUDA.  Release
        # cached tensors before proceeding to the next large AVI.
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    if failures:
        print("UPLOAD_RECORDING_FAILED\n" + "\n".join(failures), file=sys.stderr)
        return 1
    print("UPLOAD_RECORDING_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
