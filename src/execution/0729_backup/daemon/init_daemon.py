#!/usr/bin/env python3
"""Init daemon for distributed FoundPose first-frame init.

Each capture PC (capture1-6) runs one of these. It owns 4 physically attached
cameras (via paradex SHM). On `run` command it:

  1. Snapshot 4 frames from SHM
  2. Undistort
  3. Per-cam SAM3 mask  ──┐
                          ├─→ PUB mask channel (port 5006) [async, fire-and-forget]
                          └─→ FoundPose Stage B → PUB pose channel (port 5007)

Channels:
    REQ/REP control:    CommandReceiver  port 6893  (init/run/exit)
    PUB masks:          DataPublisher    port 5006  ("init_mask")
    PUB poses:          DataPublisher    port 5007  ("init_pose")

Init payload (sent once when robot PC selects an object):
    {
        "obj_name":   str,
        "mesh_path":  str        (NFS path to mesh .obj),
        "assets_root": str       (NFS path to outputs/foundpose_assets/{obj}),
        "intrinsics": {serial: {K, K_undist, dist_params, width, height}},
        "extrinsics": {serial: 4x4 world->cam},
    }

Run payload (sent every time we want a new init pose):
    {
        "request_id": int,
        "prompt":     str        (SAM3 text prompt),
    }

Run inside `gotrack_cu128` env on each capture PC:

    python src/execution/daemon/init_daemon.py \\
        --port-mask 5006 --port-pose 5007 --port-cmd 6893
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paradex.io.camera_system.camera_reader import MultiCameraReader  # noqa: E402
from paradex.io.capture_pc.data_sender import DataPublisher  # noqa: E402
from paradex.io.capture_pc.command_sender import CommandReceiver  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="[init_daemon] %(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _to_44(ext: Any) -> np.ndarray:
    a = np.asarray(ext, dtype=np.float64).reshape(-1)
    if a.size == 12:
        a = np.vstack([a.reshape(3, 4), [0, 0, 0, 1]])
    elif a.size == 16:
        a = a.reshape(4, 4)
    else:
        raise ValueError(f"bad ext shape: {a.size}")
    return a


class InitDaemon:
    """Distributed FoundPose init daemon.

    Pipeline per camera: SHM read → undistort → SAM3 mask (PUB) → FPose (PUB).
    """

    def __init__(self, port_mask: int, port_pose: int, port_cmd: int):
        self.port_mask = port_mask
        self.port_pose = port_pose
        self.port_cmd = port_cmd

        self.pub_mask = DataPublisher(port=port_mask, name="init_mask")
        self.pub_pose = DataPublisher(port=port_pose, name="init_pose")

        # Preload SAM3 now (object-agnostic, slow ~5-30s first time). FoundPose
        # is object-specific so it loads on /init per-object.
        from autodex.perception.mask import Sam3ImageSegmentor
        t0 = time.perf_counter()
        self.sam3 = Sam3ImageSegmentor(gpu=0)
        logger.info(f"[startup] SAM3 preloaded in {time.perf_counter()-t0:.1f}s")

        # Lazy — built on /init.
        self.reader: Optional[MultiCameraReader] = None
        self.fp = None
        self.my_serials: List[str] = []
        self.K_undist: Dict[str, np.ndarray] = {}
        self.ext_cw: Dict[str, np.ndarray] = {}
        self.undistort_maps: Dict[str, tuple] = {}  # (mapx, mapy)

        self.init_event = threading.Event()
        self.run_event = threading.Event()
        self.exit_event = threading.Event()
        self.cmd_receiver = CommandReceiver(
            event_dict={
                "init": self.init_event,
                "run": self.run_event,
                "exit": self.exit_event,
            },
            port=port_cmd,
        )

    # ── lifecycle ──

    def _do_init(self) -> None:
        from autodex.perception.mask import Sam3ImageSegmentor
        from autodex.perception.foundpose_init import FoundPoseInit

        info = self.cmd_receiver.event_info.get("init", {}) or {}
        obj_name = info["obj_name"]
        mesh_path = str(Path(info["mesh_path"]).expanduser())
        assets_root = str(Path(info["assets_root"]).expanduser())
        all_intrinsics = info["intrinsics"]
        all_extrinsics = info["extrinsics"]
        self.mode = str(info.get("mode", "live"))  # "live" or "disk"
        explicit_serials = info.get("my_serials")

        if self.mode == "live":
            if self.reader is None:
                self.reader = MultiCameraReader()
            self.my_serials = list(self.reader.camera_names)
        else:
            self.reader = None
            if not explicit_serials:
                raise ValueError("disk mode requires 'my_serials' in init payload")
            self.my_serials = list(explicit_serials)
        logger.info(f"[init] mode={self.mode} cameras={len(self.my_serials)}: {self.my_serials}")

        self.K_undist = {}
        self.ext_cw = {}
        self.undistort_maps = {}
        for s in self.my_serials:
            if s not in all_intrinsics or s not in all_extrinsics:
                logger.warning(f"[init] no calib for {s}, skipping")
                continue
            intr = all_intrinsics[s]
            K_orig = np.asarray(intr["K_orig"], dtype=np.float64).reshape(3, 3)
            K_undist = np.asarray(intr["K_undist"], dtype=np.float64).reshape(3, 3)
            dist = np.asarray(intr["dist_params"], dtype=np.float64).reshape(-1)
            w = int(intr["width"]); h = int(intr["height"])
            self.K_undist[s] = K_undist
            self.ext_cw[s] = _to_44(all_extrinsics[s])
            if self.mode == "live":
                mapx, mapy = cv2.initUndistortRectifyMap(
                    K_orig, dist, None, K_undist, (w, h), cv2.CV_16SC2)
                self.undistort_maps[s] = (mapx, mapy)
            else:
                # disk mode: saved images are already undistorted
                self.undistort_maps[s] = None

        # SAM3 already preloaded at daemon startup. FoundPose is object-specific.
        if self.fp is None or getattr(self.fp, "obj_name", None) != obj_name:
            t0 = time.perf_counter()
            self.fp = FoundPoseInit(
                mesh_path=mesh_path,
                assets_root=assets_root,
                obj_name=obj_name,
                device="cuda:0",
            )
            logger.info(f"[init] FoundPose loaded in {time.perf_counter()-t0:.1f}s")

        self.init_event.clear()
        logger.info("[init] ready")

    def _publish_mask_async(self, req_id: int, serial: str,
                            mask_bool: np.ndarray, t_sam3: float) -> None:
        """Encode mask as PNG and publish in background thread."""
        def _work():
            try:
                mask_u8 = (mask_bool.astype(np.uint8) * 255)
                ok, buf = cv2.imencode(".png", mask_u8)
                if not ok:
                    return
                meta = [{
                    "req_id": int(req_id),
                    "serial": serial,
                    "h": int(mask_bool.shape[0]),
                    "w": int(mask_bool.shape[1]),
                    "t_sam3": float(t_sam3),
                    "ts": time.time(),
                }]
                self.pub_mask.send_data(meta, [buf.tobytes()])
            except Exception as exc:
                logger.warning(f"[mask_pub] {serial}: {exc}")
        threading.Thread(target=_work, daemon=True).start()

    def _publish_pose(self, req_id: int, serial: str,
                      result: Optional[Dict[str, Any]],
                      t_fp: float) -> None:
        if result is None:
            meta = [{
                "req_id": int(req_id), "serial": serial, "ok": False,
                "t_fp": float(t_fp), "ts": time.time(),
            }]
            self.pub_pose.send_data(meta, [b""])
            return
        pose = np.ascontiguousarray(result["pose_world"], dtype=np.float64)
        meta = [{
            "req_id": int(req_id), "serial": serial, "ok": True,
            "quality": float(result.get("quality", 0.0)),
            "inliers": int(result.get("inliers", 0)),
            "template_id": int(result.get("template_id", -1)),
            "mask_pixels": int(result.get("mask_pixels", 0)),
            "t_fp": float(t_fp), "ts": time.time(),
        }]
        self.pub_pose.send_data(meta, [pose.tobytes()])

    def _do_run(self) -> None:
        if self.sam3 is None or self.fp is None:
            logger.error("[run] not initialized (no models)")
            self.run_event.clear()
            return
        if self.mode == "live" and self.reader is None:
            logger.error("[run] live mode but no MultiCameraReader")
            self.run_event.clear()
            return

        info = self.cmd_receiver.event_info.get("run", {}) or {}
        prompt = info.get("prompt", "object")
        req_id = int(info.get("request_id", 0))
        capture_dir = info.get("capture_dir")  # disk mode only
        save_capture_dir = info.get("save_capture_dir")  # optional: save live undistorted frames
        if capture_dir:
            capture_dir = str(Path(capture_dir).expanduser())
        if save_capture_dir:
            save_capture_dir = str(Path(save_capture_dir).expanduser())
            (Path(save_capture_dir) / "images").mkdir(parents=True, exist_ok=True)

        # 1. snapshot
        t_snap = time.perf_counter()
        if self.mode == "live":
            # Sync wait: every cam must advance to a NEW frame within timeout.
            # Async per-cam latest grab caused random cams (still fid=0) to be
            # skipped — reverted to single-shot wait_for_new_frames.
            frames = self.reader.wait_for_new_frames(timeout=2.0)
        else:
            if not capture_dir:
                logger.error(f"[run {req_id}] disk mode but no capture_dir")
                self.run_event.clear()
                return
            frames = {}
            for s in self.my_serials:
                p = Path(capture_dir) / "images" / f"{s}.png"
                if p.exists():
                    bgr = cv2.imread(str(p))
                    if bgr is not None:
                        frames[s] = (bgr, 0)
        logger.info(f"[run {req_id}] snapshot {time.perf_counter()-t_snap:.3f}s "
                    f"({len(frames)} frames)")

        # 2. per-cam pipeline
        for s in self.my_serials:
            if s not in self.K_undist:
                continue
            f = frames.get(s)
            if f is None or f[0] is None:
                logger.warning(f"[run {req_id}] no frame for {s}")
                continue
            img_bgr, _ = f
            t0 = time.perf_counter()
            if self.mode == "live":
                mapx, mapy = self.undistort_maps[s]
                img_und = cv2.remap(img_bgr, mapx, mapy, cv2.INTER_LINEAR)
                rgb = cv2.cvtColor(img_und, cv2.COLOR_BGR2RGB)
                if save_capture_dir:
                    out_path = str(Path(save_capture_dir) / "images" / f"{s}.png")
                    threading.Thread(
                        target=cv2.imwrite, args=(out_path, img_und),
                        daemon=True,
                    ).start()
            else:
                # disk mode: image is already undistorted
                rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            t_und = time.perf_counter() - t0

            # SAM3
            t1 = time.perf_counter()
            mask = self.sam3.segment(rgb, prompt)  # bool HxW or None
            t_sam3 = time.perf_counter() - t1
            if mask is None or not mask.any():
                logger.warning(f"[run {req_id}] no mask for {s}")
                # still publish empty mask + None pose so robot knows
                empty = np.zeros(rgb.shape[:2], dtype=bool)
                self._publish_mask_async(req_id, s, empty, t_sam3)
                self._publish_pose(req_id, s, None, 0.0)
                continue

            # publish mask in background, run FoundPose in parallel
            self._publish_mask_async(req_id, s, mask.astype(bool), t_sam3)

            # FoundPose Stage B
            t2 = time.perf_counter()
            result = self.fp.estimate_one_view(
                image_rgb=rgb,
                mask_bool=mask.astype(bool),
                K=self.K_undist[s],
                ext_cw=self.ext_cw[s],
            )
            t_fp = time.perf_counter() - t2
            self._publish_pose(req_id, s, result, t_fp)
            q = float(result.get("quality", 0.0)) if result else 0.0
            logger.info(f"[run {req_id}] {s}: und {t_und*1000:.0f}ms "
                        f"sam3 {t_sam3*1000:.0f}ms fp {t_fp*1000:.0f}ms q={q:.2f}")

        self.run_event.clear()
        logger.info(f"[run {req_id}] done")

    def loop(self) -> None:
        logger.info(f"[daemon] cmd port {self.port_cmd}, mask port {self.port_mask}, "
                    f"pose port {self.port_pose}")
        while not self.exit_event.is_set():
            if self.init_event.is_set():
                try:
                    self._do_init()
                except Exception as exc:
                    logger.exception(f"[init] failed: {exc}")
                    self.init_event.clear()
                continue
            if self.run_event.is_set():
                try:
                    self._do_run()
                except Exception as exc:
                    logger.exception(f"[run] failed: {exc}")
                    self.run_event.clear()
                continue
            time.sleep(0.01)
        logger.info("[daemon] exit")

    def close(self) -> None:
        self.exit_event.set()
        self.pub_mask.close()
        self.pub_pose.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port-mask", type=int, default=5006)
    parser.add_argument("--port-pose", type=int, default=5007)
    parser.add_argument("--port-cmd", type=int, default=6893)
    args = parser.parse_args()
    d = InitDaemon(args.port_mask, args.port_pose, args.port_cmd)
    try:
        d.loop()
    except KeyboardInterrupt:
        pass
    finally:
        d.close()


if __name__ == "__main__":
    main()
