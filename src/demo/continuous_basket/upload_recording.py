#!/usr/bin/env python3
"""Upload exactly one continuous-basket recording to NAS.

Run this on the robot host after a take. The runner prints the exact command,
including its timestamped ``--session`` path::

    python src/demo/continuous_basket/upload_recording.py \
      --session AutoDex/experiment/continuous_basket/franka_inspire/banana/20260831_173000_123456

The host command pauses only the GPU-heavy Init/GoTrack capture daemons, starts
one serial uploader on each capture PC, verifies every expected camera video on
NAS, then restores those daemons. The raw AVI is removed from a capture PC only
after its individual upload succeeds.

``--worker`` is an implementation detail used by the host command after it
copies this file to each capture PC. It must not be used for a whole-rig run.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Iterable, Sequence

def _find_repo_root() -> Path:
    """Find AutoDex both in the checkout and when this script is copied to /tmp."""
    candidates = [Path(os.environ["AUTODEX_REPO_ROOT"]).expanduser()] if os.environ.get(
        "AUTODEX_REPO_ROOT"
    ) else []
    candidates.extend(Path(__file__).resolve().parents)
    candidates.append(Path.home() / "AutoDex")
    for candidate in candidates:
        # The copied capture-PC worker only needs a real AutoDex checkout as a
        # stable identity; it intentionally does not import its possibly older
        # source tree. The host uses the same root for current daemon scripts.
        if (candidate / "src").is_dir():
            return candidate
    raise RuntimeError("cannot locate AutoDex; set AUTODEX_REPO_ROOT")


REPO_ROOT = _find_repo_root()
DEFAULT_PC_LIST = ("capture1", "capture2", "capture3", "capture5", "capture6")
REMOTE_WORKER = "/tmp/autodex_upload_recording.py"

def parse_autodex_session_relative(value: str | Path) -> Path:
    """Validate the explicit capture/NAS-relative session path.

    Keep this tiny validator in the copied worker itself: capture PCs need not
    have the robot host's newest AutoDex checkout merely to upload its video.
    """
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "AutoDex":
        raise ValueError("session must be a relative path beginning with 'AutoDex/'")
    return path


def recording_video_paths(
    capture_roots: Iterable[Path], *, session_relative: Path, stage: str = "capture",
) -> list[Path]:
    """Return only the local raw AVI files for one explicit session.

    Do not use ParaDex's global recursive raw-video discovery here: capture
    PCs can contain old experiments or NAS mirror directories which are not
    valid new-acquisition roots.
    """
    video_relative = session_relative / "raw" / stage / "videos"
    paths: list[Path] = []
    for root in capture_roots:
        paths.extend(path for path in (root / video_relative).glob("*.avi") if path.is_file())
    return sorted(paths)


def expected_camera_serials(session_dir: Path) -> set[str]:
    """Read the cameras this take recorded, with a legacy-calibration fallback."""
    manifest = session_dir / "recording.json"
    if manifest.is_file():
        try:
            recorded = json.loads(manifest.read_text()).get("camera_serials")
            if isinstance(recorded, list) and recorded:
                return {str(serial) for serial in recorded}
        except (OSError, ValueError, json.JSONDecodeError):
            pass
    intrinsics = session_dir / "cam_param" / "intrinsics.json"
    if not intrinsics.is_file():
        raise FileNotFoundError(f"missing session cam-param: {intrinsics}")
    data = json.loads(intrinsics.read_text())
    if not isinstance(data, dict) or not data:
        raise ValueError(f"invalid session cam-param: {intrinsics}")
    return {str(serial) for serial in data}


def uploaded_video_paths(session_dir: Path, *, stage: str = "capture") -> dict[str, Path]:
    video_dir = session_dir / "videos" / stage
    return {path.stem: path for path in video_dir.glob("*.avi") if path.is_file()}


def verify_nas_recording(session_dir: Path, *, stage: str = "capture") -> tuple[bool, str]:
    expected = expected_camera_serials(session_dir)
    actual = uploaded_video_paths(session_dir, stage=stage)
    missing = sorted(expected - set(actual))
    empty = sorted(serial for serial, path in actual.items() if path.stat().st_size == 0)
    if missing or empty:
        detail = []
        if missing:
            detail.append("missing=" + ",".join(missing))
        if empty:
            detail.append("empty=" + ",".join(empty))
        return False, "; ".join(detail)
    return True, f"{len(expected)} camera videos"


class _Progress(dict):
    """Minimal progress sink required by ParaDex's single-video routine."""

    def __setitem__(self, key, value):  # type: ignore[override]
        super().__setitem__(key, value)
        message = value.get("message", "") if isinstance(value, dict) else ""
        if message:
            print(f"  {Path(str(key)).name}: {message}", flush=True)


def _worker(session_relative: Path, stage: str, dry_run: bool) -> int:
    """Run serially on one capture PC in ``flir_env``."""
    from paradex.utils.path import capture_path_list, shared_dir
    from paradex.video.raw_video_processor import undistort_raw_video

    capture_roots = [Path(path).expanduser() for path in capture_path_list]
    session_dir = Path(shared_dir).expanduser() / session_relative
    expected_camera_serials(session_dir)
    videos = recording_video_paths(
        capture_roots, session_relative=session_relative, stage=stage,
    )
    if not videos:
        # A resumed upload may already have removed this PC's raw inputs. The
        # host performs the authoritative all-camera NAS check after workers end.
        print("UPLOAD_RECORDING_WORKER_OK no local raw AVI remaining")
        return 0

    print(f"[upload] session={session_relative.as_posix()} videos={len(videos)} workers=1")
    for video in videos:
        print(video)
    if dry_run:
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

        # A new undistortion map is allocated on CUDA for every AVI. Explicit
        # cleanup keeps one worker well below a capture PC's GPU budget.
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    if failures:
        print("UPLOAD_RECORDING_WORKER_FAILED\n" + "\n".join(failures), file=sys.stderr)
        return 1
    print("UPLOAD_RECORDING_WORKER_OK")
    return 0


def _run_checked(command: Sequence[str], *, action: str) -> None:
    print("[upload-host]", shlex.join(command), flush=True)
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{action} failed with exit code {result.returncode}")


def _pause_capture_daemons() -> None:
    _run_checked(["bash", str(REPO_ROOT / "scripts/gotrack_daemons.sh"), "stop"],
                 action="stop GoTrack daemons")
    _run_checked(["bash", str(REPO_ROOT / "scripts/init_daemons.sh"), "stop"],
                 action="stop init daemons")


def _resume_capture_daemons() -> None:
    _run_checked(["bash", str(REPO_ROOT / "scripts/init_daemons.sh"), "start"],
                 action="start init daemons")
    _run_checked(["bash", str(REPO_ROOT / "scripts/gotrack_daemons.sh"), "start"],
                 action="start GoTrack daemons")


def _copy_worker(pc_list: Sequence[str]) -> None:
    for pc in pc_list:
        _run_checked(["scp", str(Path(__file__).resolve()), f"{pc}:{REMOTE_WORKER}"],
                     action=f"copy uploader to {pc}")


def _run_workers(pc_list: Sequence[str], *, session_relative: Path, stage: str) -> None:
    workers: list[tuple[str, subprocess.Popen]] = []
    for pc in pc_list:
        inner = (
            "source ~/anaconda3/etc/profile.d/conda.sh && "
            "conda activate flir_env && "
            "cd ~/paradex && "
            "AUTODEX_REPO_ROOT=$HOME/AutoDex "
            f"python {shlex.quote(REMOTE_WORKER)} --worker "
            f"--session {shlex.quote(session_relative.as_posix())} "
            f"--stage {shlex.quote(stage)}"
        )
        print(f"[upload-host] starting {pc}", flush=True)
        workers.append((pc, subprocess.Popen(["ssh", pc, f"bash -lc {shlex.quote(inner)}"])))
    failures = [pc for pc, process in workers if process.wait() != 0]
    if failures:
        raise RuntimeError("capture-PC upload failed: " + ", ".join(failures))


def _host_upload(
    session_relative: Path, *, pc_list: Sequence[str], stage: str,
    dry_run: bool, leave_daemons_stopped: bool,
) -> int:
    session_dir = Path.home() / "shared_data" / session_relative
    recording_manifest = session_dir / "recording.json"
    if not recording_manifest.is_file():
        raise FileNotFoundError(f"missing continuous recording manifest: {recording_manifest}")
    expected = expected_camera_serials(session_dir)
    print(f"[upload-host] session={session_relative.as_posix()} expected_cameras={len(expected)}")
    if dry_run:
        print("[upload-host] dry-run: no capture daemon was stopped and no file was changed")
        return 0

    daemon_cycle_started = False
    try:
        # Set before the first stop command: if the second stop fails, restore
        # the first daemon family rather than leaving the rig half-disabled.
        daemon_cycle_started = True
        _pause_capture_daemons()
        _copy_worker(pc_list)
        _run_workers(pc_list, session_relative=session_relative, stage=stage)
        ok, detail = verify_nas_recording(session_dir, stage=stage)
        if not ok:
            raise RuntimeError(f"NAS recording verification failed: {detail}")
        print(f"UPLOAD_RECORDING_OK {detail}")
        return 0
    finally:
        if daemon_cycle_started and not leave_daemons_stopped:
            original_error = sys.exc_info()[0] is not None
            try:
                _resume_capture_daemons()
            except Exception as exc:
                print(f"UPLOAD_RECORDING_DAEMON_RESTART_FAILED {exc}", file=sys.stderr)
                if not original_error:
                    raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session", required=True, type=parse_autodex_session_relative,
        help="exact timestamped session path printed by run_demo.py, beginning AutoDex/",
    )
    parser.add_argument("--stage", default="capture")
    parser.add_argument("--pc-list", nargs="+", default=list(DEFAULT_PC_LIST))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--leave-daemons-stopped", action="store_true",
        help="do not restore Init/GoTrack daemons after a host upload",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.worker:
        return _worker(args.session, args.stage, args.dry_run)
    return _host_upload(
        args.session, pc_list=args.pc_list, stage=args.stage,
        dry_run=args.dry_run, leave_daemons_stopped=args.leave_daemons_stopped,
    )


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"UPLOAD_RECORDING_FAILED {exc}", file=sys.stderr)
        raise SystemExit(1)
