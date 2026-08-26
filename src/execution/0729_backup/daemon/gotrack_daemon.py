#!/usr/bin/env python3
"""GoTrack capture-PC daemon.

Each capture PC (capture1-6) runs one of these. It owns 4 physically attached
cameras, reads frames from shared memory (PySpin SHM publisher), runs GoTrack
stage 1-4 on its 4 cams, and publishes per-cam anchor observations to the
robot PC for stage 5-6 (triangulation + Kabsch fit).

Channels (paradex paths):
    PUB obs:           DataPublisher  port 1235  ("gotrack_obs")
    SUB prior_pose:    custom SUB      port 1236  ("gotrack_prior")
    REQ/REP control:   CommandReceiver port 6892  (init/start/stop/exit)

Initialization request from robot PC carries:
  - mesh_path
  - anchor_bank_path
  - intrinsics + extrinsics dict for *all* cameras (self picks own subset)
  - object_id, object_name

Run inside the `gotrack` conda env on each capture PC:

    python src/execution/daemon/gotrack_daemon.py \\
        --port-obs 1235 --port-prior 1236 --port-cmd 6892
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import zmq

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# paradex SHM camera reader + publisher.
from paradex.io.camera_system.camera_reader import MultiCameraReader  # noqa: E402
from paradex.io.capture_pc.data_sender import DataPublisher  # noqa: E402
from paradex.io.capture_pc.command_sender import CommandReceiver  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="[gotrack_daemon] %(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _np_to_bytes(arr: Optional[np.ndarray]) -> bytes:
    """Pack a numpy array as raw bytes (caller stores shape+dtype in metadata)."""
    if arr is None:
        return b""
    return np.ascontiguousarray(arr).tobytes()


def _arr_meta(arr: Optional[np.ndarray]) -> Dict[str, Any]:
    if arr is None:
        return {"shape": [], "dtype": ""}
    return {"shape": list(arr.shape), "dtype": str(arr.dtype)}


class PriorPoseSubscriber:
    """Subscribe to latest prior pose published by robot PC."""

    def __init__(self, robot_ip: str, port: int):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt_string(zmq.SUBSCRIBE, "")
        self.sock.setsockopt(zmq.CONFLATE, 1)  # keep only latest
        self.sock.connect(f"tcp://{robot_ip}:{port}")
        self._latest: Dict[str, Any] = {}
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info(f"[prior_sub] connected to {robot_ip}:{port}")

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                if self.sock.poll(timeout=100):
                    msg = self.sock.recv_json(flags=zmq.NOBLOCK)
                    with self._lock:
                        self._latest = msg
            except zmq.Again:
                pass
            except Exception as exc:
                logger.warning(f"[prior_sub] {exc}")

    def get_latest(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._latest)

    def close(self) -> None:
        self._stop.set()
        self._thread.join(timeout=1)
        self.sock.close()


class GoTrackDaemon:
    """Holds GoTrackEngine + paradex IO. Stays idle until /init command."""

    def __init__(
        self,
        port_obs: int,
        port_prior: int,
        port_cmd: int,
        robot_ip: str,
    ):
        self.port_obs = port_obs
        self.port_prior = port_prior
        self.port_cmd = port_cmd
        self.robot_ip = robot_ip

        # paradex publisher for anchor obs (one PUB per PC).
        self.publisher = DataPublisher(port=port_obs, name=f"gotrack_obs")

        # Subscribe to prior pose stream from robot PC.
        self.prior_sub = PriorPoseSubscriber(robot_ip, port_prior)

        # Camera reader (lazy — we need to wait until cameras exist in SHM).
        self.reader: Optional[MultiCameraReader] = None
        self.engine = None  # gotrack_engine.GoTrackEngine

        # Control signals.
        self.init_event = threading.Event()
        self.start_event = threading.Event()
        self.stop_event = threading.Event()
        self.exit_event = threading.Event()
        self._init_payload: Dict[str, Any] = {}

        self.cmd_receiver = CommandReceiver(
            event_dict={
                "init": self.init_event,
                "start": self.start_event,
                "stop": self.stop_event,
                "exit": self.exit_event,
            },
            port=port_cmd,
        )

    # ── lifecycle ──

    def _do_init(self) -> None:
        """Build GoTrackEngine from init payload sent by robot PC."""
        import cv2  # noqa: F401
        from autodex.perception.gotrack_engine import GoTrackEngine, CameraIntrinsics
        info = self.cmd_receiver.event_info.get("init", {}) or {}

        mesh_path = info["mesh_path"]
        anchor_bank_path = info["anchor_bank_path"]
        object_id = int(info.get("object_id", 1))
        object_name = info.get("object_name", "object")
        all_intrinsics = info["intrinsics"]   # {serial: {K (3x3), width, height} or with K_orig/K_undist/dist_params}
        all_extrinsics = info["extrinsics"]   # {serial: 4x4 world->cam}

        # Init MultiCameraReader for THIS PC's cameras.
        self.reader = MultiCameraReader()
        my_serials = list(self.reader.camera_names)
        logger.info(f"[init] this PC has {len(my_serials)} cameras: {my_serials}")

        # Build undistort maps so we can rectify SHM frames before feeding the
        # engine (which uses K_undist). Without this, only the cam whose object
        # happens to be near the image center survives — others fail because of
        # the distortion mismatch between raw image and K_undist.
        import cv2 as _cv2
        self.undistort_maps = {}

        cameras: List[CameraIntrinsics] = []
        for s in my_serials:
            if s not in all_intrinsics or s not in all_extrinsics:
                logger.warning(f"[init] no calib for {s}, skipping")
                continue
            intr = all_intrinsics[s]
            ext = np.asarray(all_extrinsics[s], dtype=np.float64).reshape(-1)
            ext = ext.reshape(3, 4) if ext.size == 12 else ext.reshape(4, 4)
            if ext.shape == (3, 4):
                ext = np.vstack([ext, [0, 0, 0, 1]])
            # K is K_undist (intrinsics_undistort) — engine works in undistorted space.
            K_undist = np.asarray(intr["K"], dtype=np.float64).reshape(3, 3)
            w, h = int(intr["width"]), int(intr["height"])
            cameras.append(CameraIntrinsics(
                serial=s, K=K_undist, extrinsic_cw=ext, width=w, height=h,
            ))
            # If K_orig + dist available, precompute undistort map.
            K_orig = intr.get("K_orig")
            dist = intr.get("dist_params")
            if K_orig is not None and dist is not None:
                K_orig = np.asarray(K_orig, dtype=np.float64).reshape(3, 3)
                dist = np.asarray(dist, dtype=np.float64).reshape(-1)
                mapx, mapy = _cv2.initUndistortRectifyMap(
                    K_orig, dist, None, K_undist, (w, h), _cv2.CV_16SC2)
                self.undistort_maps[s] = (mapx, mapy)
            else:
                self.undistort_maps[s] = None  # raw frame passed as-is
                logger.warning(f"[init] {s}: no K_orig+dist_params, skipping undistort "
                               "(anchor projection may be off)")
        if not cameras:
            raise RuntimeError("No cameras with calibration on this PC")

        self.engine = GoTrackEngine(
            mesh_path=mesh_path,
            anchor_bank_path=anchor_bank_path,
            cameras=cameras,
            object_id=object_id,
            object_name=object_name,
            mesh_scale=float(info.get("mesh_scale", 1.0)),
            unit_scale_mode=str(info.get("unit_scale_mode", "auto")),
            num_iters=int(info.get("num_iters", 1)),
            first_frame_num_iters=int(info.get("first_frame_num_iters", 5)),
            mask_free=True,
            skip_pnp=True,
        )
        logger.info("[init] engine ready")

    def _process_loop(self) -> None:
        """Main per-frame loop. Reads SHM, runs GoTrackEngine, publishes obs."""
        if self.reader is None or self.engine is None:
            logger.error("[loop] not initialized")
            return

        logger.info(f"[loop] entry: stop_event={self.stop_event.is_set()} "
                    f"exit_event={self.exit_event.is_set()}")
        # Defensive: clear stop_event left over from a prior session before we
        # enter the polling loop. Without this, a stale stop from the previous
        # run kills this run before it starts.
        if self.stop_event.is_set():
            logger.warning("[loop] stop_event was set at entry; clearing")
            self.stop_event.clear()

        # Pick up the trial_ts the robot PC sent with the start command, so
        # this trial's crops land under .../{obj}/{trial_ts}/{fid}/ and don't
        # collide with prior trials.
        start_info = self.cmd_receiver.event_info.get("start", {}) or {}
        trial_ts = str(start_info.get("trial_ts", "")) or None
        self.engine.set_trial_ts(trial_ts)

        my_serials = list(self.engine.cameras.keys())
        last_frame_ids = {s: 0 for s in my_serials}
        last_published_frame_id: Optional[int] = None
        n_published = 0
        n_skipped_frames = 0
        n_skipped_no_prior = 0
        last_log_t = time.perf_counter()

        while not self.stop_event.is_set() and not self.exit_event.is_set():
            now = time.perf_counter()
            if now - last_log_t > 2.0:
                logger.info(f"[loop] published={n_published}  "
                            f"skipped_frames={n_skipped_frames}  "
                            f"skipped_no_prior={n_skipped_no_prior}  "
                            f"last_fid={last_frame_ids}")
                last_log_t = now

            # 1. Pull latest frames from SHM, undistort to match K_undist that
            #    the engine uses. Dedupe at the frame_id level via
            #    last_published_frame_id below.
            import cv2 as _cv2
            images_data = self.reader.get_images(copy=True)
            frames_bgr: Dict[str, np.ndarray] = {}
            min_frame_id: Optional[int] = None
            for s in my_serials:
                img, fid = images_data.get(s, (None, 0))
                if img is None or fid <= 0:
                    continue
                maps = self.undistort_maps.get(s) if hasattr(self, "undistort_maps") else None
                if maps is not None:
                    img = _cv2.remap(img, maps[0], maps[1], _cv2.INTER_LINEAR)
                frames_bgr[s] = img
                last_frame_ids[s] = fid
                min_frame_id = fid if min_frame_id is None else min(min_frame_id, fid)

            # Need all cams to have at least some valid frame.
            if len(frames_bgr) != len(my_serials):
                n_skipped_frames += 1
                time.sleep(0.005)
                continue
            if min_frame_id == last_published_frame_id:
                time.sleep(0.005)
                continue

            # 2. Get latest prior pose. Skip if none yet (robot PC hasn't sent
            #    init pose).
            prior = self.prior_sub.get_latest()
            if "pose_world" not in prior:
                n_skipped_no_prior += 1
                time.sleep(0.01)
                continue
            prior_pose = np.asarray(prior["pose_world"], dtype=np.float64).reshape(4, 4)
            prior_frame_id = int(prior.get("frame_id", -1))

            # 3. Run GoTrackEngine.
            # diag: frames_bgr stats per cam (right before engine)
            if n_published < 3:
                for _s, _img in frames_bgr.items():
                    logger.info(f"[diag-frame] {_s}: shape={_img.shape} dtype={_img.dtype} "
                                f"mean={float(_img.mean()):.2f} std={float(_img.std()):.2f} "
                                f"nonzero={int((_img != 0).sum())}/{_img.size}")
            t0 = time.perf_counter()
            try:
                per_cam = self.engine.process_frame(
                    prior_pose_world=prior_pose,
                    frames_bgr=frames_bgr,
                    masks=None,
                    frame_index=int(min_frame_id),
                    include_debug_images=True,
                )
            except Exception as exc:
                logger.exception(f"[loop] engine error: {exc}")
                time.sleep(0.05)
                continue
            engine_sec = time.perf_counter() - t0

            # 4. Pack per-cam payload and publish.
            meta_items: List[Dict[str, Any]] = []
            binaries: List[bytes] = []
            for s, payload in per_cam.items():
                arr_keys = (
                    "uv_curr", "confidence", "valid_mask", "selected_mask",
                    "anchor_ids", "positions_o", "crop_intrinsic",
                    "T_world_from_crop_cam",
                )
                arr_meta = {}
                for k in arr_keys:
                    arr = payload.get(k)
                    arr_meta[k] = _arr_meta(arr) | {"data_index": len(binaries)}
                    binaries.append(_np_to_bytes(arr))
                meta_items.append({
                    "type": "gotrack_obs",
                    "name": f"{s}",  # camera serial
                    "frame_id": int(min_frame_id),
                    "prior_frame_id": prior_frame_id,
                    "status": str(payload.get("status", "ok")),
                    "engine_sec": engine_sec,
                    "arrays": arr_meta,
                })
            if meta_items:
                self.publisher.send_data(meta_items, binaries)
                last_published_frame_id = min_frame_id
                n_published += 1

    def run(self) -> None:
        logger.info("[daemon] waiting for /init from robot PC")
        while not self.exit_event.is_set():
            if self.init_event.is_set():
                self.init_event.clear()
                try:
                    self._do_init()
                except Exception as exc:
                    logger.exception(f"[init] failed: {exc}")
                    continue

            if self.start_event.is_set() and self.engine is not None:
                self.start_event.clear()
                logger.info("[daemon] entering process loop")
                self._process_loop()
                logger.info("[daemon] process loop stopped")
                self.stop_event.clear()
                # Background rsync local debug crops → NAS. Local writes are
                # fast and don't block engine; the upload runs after the run
                # ends so it doesn't affect tracking latency.
                self._upload_debug_crops()

            time.sleep(0.05)

        logger.info("[daemon] exit")

    def _upload_debug_crops(self) -> None:
        """rsync local /tmp/gotrack_crops/ to NAS, then wipe local."""
        import os, shutil, subprocess, threading
        local_root = "/tmp/gotrack_crops"
        if not os.path.isdir(local_root):
            return
        nas_root = os.path.expanduser("~/shared_data/AutoDex/debug/gotrack_crops")
        os.makedirs(nas_root, exist_ok=True)

        def _do_upload():
            try:
                logger.info(f"[diag-crops] uploading {local_root} → {nas_root}")
                subprocess.run(
                    ["rsync", "-rt", "--no-owner", "--no-group", "--remove-source-files",
                     f"{local_root}/", f"{nas_root}/"],
                    check=False,
                )
                # Remove empty dirs after --remove-source-files leaves them.
                shutil.rmtree(local_root, ignore_errors=True)
                logger.info("[diag-crops] upload complete")
            except Exception as exc:
                logger.warning(f"[diag-crops] upload failed: {exc}")

        threading.Thread(target=_do_upload, daemon=True).start()

    def close(self) -> None:
        self.exit_event.set()
        self.publisher.close()
        self.prior_sub.close()
        self.cmd_receiver.end()
        if self.reader is not None:
            self.reader.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port-obs", type=int, default=1235,
                        help="DataPublisher port for anchor obs (capture→robot)")
    parser.add_argument("--port-prior", type=int, default=1236,
                        help="ZMQ SUB port for prior pose (robot→capture)")
    parser.add_argument("--port-cmd", type=int, default=6892,
                        help="CommandReceiver port for init/start/stop")
    parser.add_argument("--robot-ip", type=str, default="192.168.0.2",
                        help="Robot PC IP for prior_pose SUB")
    args = parser.parse_args()

    daemon = GoTrackDaemon(
        port_obs=args.port_obs,
        port_prior=args.port_prior,
        port_cmd=args.port_cmd,
        robot_ip=args.robot_ip,
    )
    try:
        daemon.run()
    finally:
        daemon.close()


if __name__ == "__main__":
    main()
