"""Small lifecycle wrapper around the distributed GoTrack runtime.

The core tracker intentionally exposes an infinite generator.  A demo loop
needs a different interface: start after FoundPose has located an object,
retrieve a fresh tracked pose after a robot action, then stop cleanly before
switching catalogue objects.  This wrapper provides that boundary without
changing GoTrack's existing daemon protocol.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class TrackSample:
    pose_world: np.ndarray
    wall_time: float
    frame_id: int
    n_inliers: int
    mean_residual_mm: float


class LiveGoTrackSession:
    """One active object tracker backed by existing capture-PC daemons.

    Imports are deliberately lazy.  Unit tests and offline planning can use
    the continuous-demo policy without ParaDex, ZMQ, the GoTrack checkout, or
    CUDA.  The capture PCs must already be running
    ``src/execution/daemon/gotrack_daemon.py``.
    """

    def __init__(
        self,
        *,
        pc_list: List[str],
        capture_ips: List[str],
        intrinsics: Dict[str, Dict[str, Any]],
        extrinsics: Dict[str, np.ndarray],
        anchor_root: Path,
        port_obs: int = 1235,
        port_prior: int = 1236,
        port_cmd: int = 6892,
        min_cams_per_frame: int = 6,
        min_inliers: int = 12,
    ):
        self.pc_list = list(pc_list)
        self.capture_ips = list(capture_ips)
        self.intrinsics = intrinsics
        self.extrinsics = extrinsics
        self.anchor_root = Path(anchor_root)
        self.port_obs = int(port_obs)
        self.port_prior = int(port_prior)
        self.port_cmd = int(port_cmd)
        self.min_cams_per_frame = int(min_cams_per_frame)
        self.min_inliers = int(min_inliers)

        self._cmd = None
        self._tracker = None
        self._worker: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._sample: Optional[TrackSample] = None
        self._sample_cv = threading.Condition()
        self._worker_error: Optional[str] = None
        self.obj_name: Optional[str] = None

    @staticmethod
    def _home_relative(path: Path) -> str:
        home = Path.home()
        try:
            return "~/" + str(path.expanduser().resolve().relative_to(home))
        except ValueError:
            return str(path)

    def _payload_calibration(self) -> tuple[dict, dict]:
        intrinsics = {
            serial: {
                "K": np.asarray(value["K_undist"], dtype=float).tolist(),
                "K_orig": np.asarray(value["K_orig"], dtype=float).tolist(),
                "dist_params": np.asarray(value["dist_params"], dtype=float).tolist(),
                "width": int(value["width"]), "height": int(value["height"]),
            }
            for serial, value in self.intrinsics.items()
        }
        extrinsics = {
            serial: np.asarray(value, dtype=float).reshape(4, 4).tolist()
            for serial, value in self.extrinsics.items()
        }
        return intrinsics, extrinsics

    def start(self, *, obj_name: str, mesh_path: Path, init_pose_world: np.ndarray,
              settle_s: float = 0.5) -> None:
        """Configure capture daemons and begin tracking a FoundPose result."""
        self.stop()
        anchor = self.anchor_root / f"{obj_name}.npz"
        if not anchor.is_file():
            raise FileNotFoundError(
                f"GoTrack anchor bank missing for {obj_name}: {anchor}; generate it before the demo"
            )
        from paradex.io.capture_pc.command_sender import CommandSender
        from autodex.perception.gotrack_tracker import GoTrackTracker

        intrinsics, extrinsics = self._payload_calibration()
        self._cmd = CommandSender(pc_list=self.pc_list, port=self.port_cmd)
        info = {
            "mesh_path": self._home_relative(Path(mesh_path)),
            "anchor_bank_path": self._home_relative(anchor),
            "object_id": 1, "object_name": obj_name,
            "intrinsics": intrinsics, "extrinsics": extrinsics,
            "mesh_scale": 1.0, "unit_scale_mode": "auto",
            "num_iters": 1, "first_frame_num_iters": 5,
        }
        self._cmd.send_command("init", wait=False, cmd_info=info)
        time.sleep(float(settle_s))
        self._tracker = GoTrackTracker(
            capture_pc_ips=self.capture_ips, port_obs=self.port_obs,
            port_prior=self.port_prior, min_cams_per_frame=self.min_cams_per_frame,
        )
        self._cmd.send_command(
            "start", wait=False,
            cmd_info={"trial_ts": f"basket_{obj_name}_{int(time.time() * 1000)}"},
        )
        self.obj_name = obj_name
        self._stop.clear()
        self._worker_error = None
        with self._sample_cv:
            self._sample = None

        initial = np.asarray(init_pose_world, dtype=np.float64).reshape(4, 4).copy()

        def _run() -> None:
            try:
                for frame_id, pose, info in self._tracker.track(initial):
                    if self._stop.is_set():
                        break
                    sample = TrackSample(
                        pose_world=np.asarray(pose, dtype=np.float64).copy(),
                        wall_time=time.time(), frame_id=int(frame_id),
                        n_inliers=int(info.get("n_inliers", 0)),
                        mean_residual_mm=float(info.get("mean_residual_mm", -1.0)),
                    )
                    with self._sample_cv:
                        self._sample = sample
                        self._sample_cv.notify_all()
            except Exception as exc:  # surfaced by wait_for_pose in the runner
                self._worker_error = f"{type(exc).__name__}: {exc}"
                with self._sample_cv:
                    self._sample_cv.notify_all()

        self._worker = threading.Thread(target=_run, name=f"gotrack-{obj_name}", daemon=True)
        self._worker.start()

    def wait_for_pose(self, *, since_wall_time: float = 0.0,
                      timeout_s: float = 1.0) -> Optional[TrackSample]:
        """Wait for a reliable pose newer than an action start timestamp."""
        deadline = time.monotonic() + float(timeout_s)
        with self._sample_cv:
            while True:
                sample = self._sample
                if (sample is not None and sample.wall_time >= since_wall_time
                        and sample.n_inliers >= self.min_inliers):
                    return sample
                if self._worker_error is not None:
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._sample_cv.wait(timeout=remaining)

    @property
    def worker_error(self) -> Optional[str]:
        return self._worker_error

    def stop(self) -> None:
        """Stop local consumption and leave capture daemons ready for reuse."""
        self._stop.set()
        if self._tracker is not None:
            self._tracker._stop.set()
        if self._worker is not None:
            self._worker.join(timeout=1.5)
        if self._cmd is not None:
            try:
                self._cmd.send_command("stop", wait=False, cmd_info={})
            except Exception:
                pass
        if self._tracker is not None:
            try:
                self._tracker.close()
            except Exception:
                pass
        if self._cmd is not None:
            # Do not broadcast `exit`: capture daemon processes should remain
            # warm between catalogue objects and demo takes.
            for sock in self._cmd.sockets.values():
                try:
                    sock.close()
                except Exception:
                    pass
        self._cmd = None
        self._tracker = None
        self._worker = None
        self.obj_name = None

    close = stop
