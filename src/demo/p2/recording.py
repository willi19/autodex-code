"""P2-only execution recording using the standard AutoDex raw-video layout.

This module deliberately owns only the camera/video lifecycle.  Planning,
FoundPose, semantic routing, and robot motion remain in the inference demo.
Each episode gets the same layout consumed by ParaDex's generic uploader::

    capture PC:  AutoDex/experiment/v8_demo/inspire/<obj>/<stamp>/raw/exec/videos/*.avi
    NAS episode: AutoDex/experiment/v8_demo/inspire/<obj>/<stamp>/videos/exec/*.avi
"""
from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def autodex_session_relative(project_root: Path, session_dir: Path) -> Path:
    """Map a normal NAS episode to the capture daemon's ``AutoDex/...`` key.

    Kept local to P2 so this protocol has no runtime dependency on the
    separately maintained continuous-basket demo.
    """
    root = Path(project_root).resolve()
    session = Path(session_dir).resolve()
    try:
        return Path("AutoDex") / session.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"session {session} is outside AutoDex root {root}") from exc


def resolve_signal_generator_params(
    configured: Mapping[str, Any], *, device_root: Path = Path("/dev")
) -> tuple[dict[str, Any], str | None]:
    """Use the sole visible USBTMC node if the configured stale node is absent."""
    params = dict(configured)
    configured_addr = params.get("addr")
    if configured_addr and Path(str(configured_addr)).exists():
        return params, None
    candidates = sorted(path for path in device_root.glob("usbtmc*") if path.exists())
    if len(candidates) != 1:
        return params, None
    resolved = str(candidates[0])
    params["addr"] = resolved
    return params, (f"configured trigger {configured_addr!r} is unavailable; "
                    f"using discovered {resolved}")


class P2RecordingRuntime:
    """Reusable trigger/timestamp resources for P2 execution recordings.

    The runtime is created once per invocation of either P2 runner.  It is
    intentionally independent of object identity so the continuous P2 runner
    can use the same trigger hardware across consecutive episodes.
    """

    def __init__(
        self,
        *,
        rcc,
        pc_list: Sequence[str],
        serials: Sequence[str],
        video_fps: int = 30,
        enabled: bool = True,
    ) -> None:
        self.rcc = rcc
        self.pc_list = tuple(pc_list)
        self.serials = tuple(sorted(str(serial) for serial in serials))
        self.video_fps = int(video_fps)
        self.enabled = bool(enabled)
        self.sync_generator = None
        self.timestamp_monitor = None
        self.sync_mode = False
        self.sync_note: str | None = None
        self._active: P2ExecutionRecorder | None = None

        if not self.enabled:
            return
        if self.video_fps <= 0:
            raise ValueError("video_fps must be positive")

        # A disconnected/renumbered USB trigger must not stop the P2 grasp
        # demo before motion.  The cameras can record an explicit free-running
        # fallback and the manifest preserves that fact for later analysis.
        try:
            from paradex.io.camera_system.signal_generator import UTGE900
            from paradex.io.camera_system.timestamp_monitor import TimestampMonitor
            from paradex.utils.system import network_info

            params, note = resolve_signal_generator_params(
                network_info["signal_generator"]["param"])
            self.sync_generator = UTGE900(**params)
            self.timestamp_monitor = TimestampMonitor(
                **network_info["timestamp"]["param"])
            self.sync_mode = True
            self.sync_note = note
            if note:
                print(f"[p2-video] {note}")
        except Exception as exc:
            self.sync_generator = None
            self.timestamp_monitor = None
            self.sync_mode = False
            self.sync_note = repr(exc)
            print("[p2-video] hardware sync unavailable; recording "
                  f"free-running AVI instead ({exc!r})")

    def for_episode(self, *, run_dir: Path, project_root: Path, executor) -> "P2ExecutionRecorder":
        if self._active is not None and self._active.active:
            raise RuntimeError("cannot start a P2 episode while another recording is active")
        recorder = P2ExecutionRecorder(
            runtime=self,
            run_dir=Path(run_dir),
            project_root=Path(project_root),
            executor=executor,
        )
        self._active = recorder
        return recorder


class P2ExecutionRecorder:
    """Start/stop one P2 episode's cameras and robot-state recording."""

    def __init__(self, *, runtime: P2RecordingRuntime, run_dir: Path,
                 project_root: Path, executor) -> None:
        self.runtime = runtime
        self.run_dir = Path(run_dir)
        self.project_root = Path(project_root)
        self.executor = executor
        self.active = False
        self._manifest: dict[str, Any] | None = None

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "recording.json"

    def _write_manifest(self) -> None:
        if self._manifest is None:
            return
        self.manifest_path.write_text(json.dumps(self._manifest, indent=2) + "\n")

    def start(self) -> dict[str, Any]:
        """Switch stream -> per-PC AVI recording immediately before the task.

        The caller stops this take directly after release, before retreat or
        home reset, so the AVI is task-only evidence rather than a full robot
        session recording.
        """
        if not self.runtime.enabled:
            return {"enabled": False, "reason": "--no-video"}
        if self.active:
            raise RuntimeError("P2 execution recording is already active")

        from src.demo.banana_test.run_demo import _rcc_start, _safe_timestamp_start

        session_rel = autodex_session_relative(self.project_root, self.run_dir)
        capture_rel = session_rel / "raw" / "exec"
        raw_dir = self.run_dir / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        self._manifest = {
            "schema_version": 1,
            "protocol": "p2_object_diversity_semantic_routing",
            "enabled": True,
            "session_relative": session_rel.as_posix(),
            "camera_capture_relative": capture_rel.as_posix(),
            "nas_video_relative": (session_rel / "videos" / "exec").as_posix(),
            "stage": "exec",
            "video_fps": self.runtime.video_fps,
            "sync_mode": "hardware" if self.runtime.sync_mode else "free_running",
            "sync_note": self.runtime.sync_note,
            "capture_pc_list": list(self.runtime.pc_list),
            "camera_serials": list(self.runtime.serials),
            "robot_state_relative": (session_rel / "raw" / "robot").as_posix(),
            "started_at": dt.datetime.now().isoformat(timespec="seconds"),
            "upload_command": "python src/util/upload_video/main.py",
            "upload_state": "pending",
        }
        self._write_manifest()

        try:
            # Mark active before touching a sink.  If any later setup call
            # raises, ``stop()`` then still closes every resource that may
            # already have been armed.
            self.active = True
            self.runtime.rcc.stop()
            _rcc_start(self.runtime.rcc, "video", self.runtime.sync_mode,
                       capture_rel.as_posix(), fps=self.runtime.video_fps)
            self.executor.start_recording(str(raw_dir / "robot"))
            if self.runtime.sync_mode:
                self._manifest["timestamps_started"] = bool(
                    _safe_timestamp_start(self.runtime.timestamp_monitor,
                                          str(raw_dir / "timestamps")))
                self.runtime.sync_generator.start(fps=self.runtime.video_fps)
                self._manifest["sync_generator_started"] = True
            self._manifest["camera_started_at"] = dt.datetime.now().isoformat(
                timespec="seconds")
            self._write_manifest()
        except Exception:
            # Recording must never leave a video sink/trigger armed after a
            # setup failure.  The caller receives the error before commanding
            # robot motion.
            self.stop(restart_stream=False)
            raise
        print("[p2-video] recording execution AVI on "
              f"{len(self.runtime.serials)} cameras -> {capture_rel}")
        return dict(self._manifest)

    def stop(self, *, restart_stream: bool = False, stream_fps: int = 10) -> dict[str, Any]:
        """Flush the current AVI take; safe and idempotent on all exits."""
        from src.demo.banana_test.run_demo import (
            _rcc_start,
            _safe_timestamp_stop,
            _stop_with_timeout,
        )

        if not self.active:
            return dict(self._manifest or {"enabled": self.runtime.enabled,
                                             "active": False})
        # Camera stop must happen before trigger stop: capture workers need a
        # last pulse to flush their AVI buffers.  This is the same ordering as
        # run_auto.py and banana_test/run_demo.py.
        _stop_with_timeout("p2 video rcc.stop", self.runtime.rcc.stop)
        if self.runtime.sync_mode:
            _safe_timestamp_stop(self.runtime.timestamp_monitor)
            _stop_with_timeout("p2 video sync_generator.stop",
                               self.runtime.sync_generator.stop)
        _stop_with_timeout("p2 video executor.stop_recording",
                           self.executor.stop_recording)
        self.active = False
        if self._manifest is not None:
            self._manifest["stopped_at"] = dt.datetime.now().isoformat(
                timespec="seconds")
            self._manifest["upload_state"] = "raw_pending"
            self._write_manifest()
        if restart_stream:
            _rcc_start(self.runtime.rcc, "stream", False, fps=stream_fps)
        return dict(self._manifest or {})
