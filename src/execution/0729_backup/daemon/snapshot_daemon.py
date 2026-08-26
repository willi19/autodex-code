#!/usr/bin/env python3
"""Snapshot daemon — JPEG snapshot of every camera SHM on request.

Runs on each capture PC (capture1-3, 5, 6) alongside init_daemon. On a "snap"
command from the robot PC, reads ONE latest frame per camera from the
shared-memory ring buffer (paradex MultiCameraReader) and PUBs a JPEG of each
to the orchestrator. No undistort, no inference — purely a frame courier.

Used to grab a charuco lift-check image WITHOUT stopping the ongoing video
recording (rcc.start("video", ...) keeps writing the AVI; the daemon and the
video writer share the same SHM ring buffer that camera.py keeps populated
because we run cameras in "full" mode — see paradex camera.py line 224-225).

Channels:
    REQ/REP control:    CommandReceiver  port 6894
    PUB snapshots:      DataPublisher    port 5009  ("snapshot")

Snap payload:
    {"request_id": int, "save_dir": optional NFS path string (capture PC side)
     for the daemon to also dump the JPEG to disk in addition to PUBing}

Per-camera publish item:
    meta = {"req_id", "serial", "fid", "h", "w", "ts"}
    blob = JPEG bytes

Launch via scripts/snapshot_daemons.sh (one process per capture PC, same env
as init_daemon: gotrack_cu128).
"""
from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paradex.io.camera_system.camera_reader import MultiCameraReader  # noqa: E402
from paradex.io.capture_pc.data_sender import DataPublisher  # noqa: E402
from paradex.io.capture_pc.command_sender import CommandReceiver  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="[snapshot_daemon] %(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class SnapshotDaemon:
    def __init__(self, port_snap: int, port_cmd: int, jpeg_quality: int = 90):
        self.port_snap = port_snap
        self.port_cmd = port_cmd
        self.jpeg_quality = int(jpeg_quality)
        self.pub = DataPublisher(port=port_snap, name="snapshot")
        self.reader: Optional[MultiCameraReader] = None

        self.snap_event = threading.Event()
        self.exit_event = threading.Event()
        self.cmd_receiver = CommandReceiver(
            event_dict={"snap": self.snap_event, "exit": self.exit_event},
            port=port_cmd,
        )

    def _ensure_reader(self):
        if self.reader is None:
            self.reader = MultiCameraReader()
            logger.info(f"[reader] attached SHM for {len(self.reader.camera_names)} "
                        f"cameras: {self.reader.camera_names}")

    def _do_snap(self):
        info = self.cmd_receiver.event_info.get("snap", {}) or {}
        req_id = int(info.get("request_id", int(time.time() * 1000) & 0x7fffffff))
        save_dir = info.get("save_dir")
        if save_dir:
            save_dir = str(Path(save_dir).expanduser())
            Path(save_dir).mkdir(parents=True, exist_ok=True)

        try:
            self._ensure_reader()
        except Exception as exc:
            logger.error(f"[snap {req_id}] reader attach failed: {exc!r}")
            self.snap_event.clear()
            return

        # Sync wait: every cam must advance to a NEW frame within timeout.
        # Mirrors init_daemon's logic.
        frames = self.reader.wait_for_new_frames(timeout=2.0)

        params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        n_published = 0
        for s, (img, fid) in frames.items():
            try:
                ok, buf = cv2.imencode(".jpg", img, params)
                if not ok:
                    continue
                blob = buf.tobytes()
                meta = [{
                    "req_id": int(req_id), "serial": s, "fid": int(fid),
                    "h": int(img.shape[0]), "w": int(img.shape[1]),
                    "ts": time.time(),
                }]
                self.pub.send_data(meta, [blob])
                n_published += 1
                if save_dir:
                    out_path = Path(save_dir) / f"{s}.jpg"
                    threading.Thread(
                        target=lambda p=out_path, b=blob:
                            p.write_bytes(b),
                        daemon=True,
                    ).start()
            except Exception as exc:
                logger.warning(f"[snap {req_id}] {s}: {exc!r}")
        logger.info(f"[snap {req_id}] published {n_published}/{len(self.reader.camera_names)}")
        self.snap_event.clear()

    def loop(self):
        logger.info(f"[daemon] cmd port {self.port_cmd}, snap port {self.port_snap}")
        while not self.exit_event.is_set():
            if self.snap_event.is_set():
                try:
                    self._do_snap()
                except Exception as exc:
                    logger.exception(f"[snap] failed: {exc}")
                    self.snap_event.clear()
                continue
            time.sleep(0.01)
        logger.info("[daemon] exit")

    def close(self):
        self.exit_event.set()
        try:
            self.pub.close()
        except Exception:
            pass
        if self.reader is not None:
            try:
                self.reader.close()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port-snap", type=int, default=5009)
    parser.add_argument("--port-cmd", type=int, default=6894)
    parser.add_argument("--jpeg-quality", type=int, default=90)
    args = parser.parse_args()
    d = SnapshotDaemon(args.port_snap, args.port_cmd, args.jpeg_quality)
    try:
        d.loop()
    except KeyboardInterrupt:
        pass
    finally:
        d.close()


if __name__ == "__main__":
    main()
