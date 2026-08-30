"""Robot-PC GoTrack tracker (stages 5-6).

Collects per-frame, per-camera anchor observations from the 6 capture-PC
``gotrack_daemon`` processes, synchronises by frame_id, runs multi-view
triangulation + robust Kabsch fit, and publishes the resulting world pose
back to the daemons as the next prior.

Use after a successful FoundPose-based init pose has been produced. Init
pose is sent once via CommandSender (or just published as the initial
prior on the prior-pose PUB channel).

Channels (mirror gotrack_daemon defaults):
    SUB obs:          DataCollector  port 1235  (subscribes to 6 capture PCs)
    PUB prior_pose:   custom PUB      port 1236  (binds, daemons subscribe)
    REQ commands:     CommandSender   port 6892  (init/start/stop)

Run inside the `gotrack` conda env on the robot PC.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import zmq

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_GOTRACK_ROOT = Path(__file__).resolve().parent / "thirdparty/MV-GoTrack"
if str(_GOTRACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_GOTRACK_ROOT))

logger = logging.getLogger(__name__)

# Default sync / timing parameters.
_DEFAULT_FRAME_TIMEOUT_S = 0.5     # drop a frame if not all cams arrive in time
_DEFAULT_MAX_INFLIGHT_FRAMES = 8   # buffer cap


def _bytes_to_np(buf: bytes, shape: List[int], dtype: str) -> Optional[np.ndarray]:
    if not shape or not dtype or buf == b"":
        return None
    arr = np.frombuffer(buf, dtype=np.dtype(dtype))
    return arr.reshape(shape) if arr.size else None


def _parse_multipart(parts: List[bytes]) -> Tuple[Optional[Any], List[bytes]]:
    """Decode current ParaDex envelopes with a legacy JSON fallback.

    ``DataPublisher`` migrated from ``[topic, json, *blobs]`` to a msgpack
    envelope.  The init orchestrator already supports both forms; keeping the
    tracker JSON-only silently discards every anchor observation on current
    capture PCs and makes a seemingly healthy GoTrack session time out.
    """
    try:
        from paradex.io.capture_pc.envelope import decode
        msg = decode(parts)
        return msg.meta, list(msg.bufs)
    except Exception:
        pass
    if len(parts) < 2:
        return None, []
    try:
        return json.loads(parts[1].decode("utf-8")), list(parts[2:])
    except Exception:
        return None, []


def _unpack_payload(meta_item: Dict[str, Any], parts: List[bytes]) -> Dict[str, Any]:
    """Reconstruct numpy arrays for one cam from a multipart message."""
    out: Dict[str, Any] = {
        "frame_id": int(meta_item.get("frame_id", -1)),
        "prior_frame_id": int(meta_item.get("prior_frame_id", -1)),
        "status": str(meta_item.get("status", "")),
        "engine_sec": float(meta_item.get("engine_sec", 0.0)),
        "serial": str(meta_item.get("name", "")),
    }
    arrays = meta_item.get("arrays", {}) or {}
    for k, info in arrays.items():
        idx = int(info.get("data_index", -1))
        # data_index in meta is the per-message binary slot (0-based across all
        # arrays sent in this message). DataPublisher prepends [topic, json] so
        # binary parts start at parts[2 + idx].
        if 0 <= idx < len(parts):
            out[k] = _bytes_to_np(parts[idx], info["shape"], info["dtype"])
        else:
            out[k] = None
    return out


class PriorPosePublisher:
    """Bind PUB socket so all capture PCs can subscribe."""

    def __init__(self, port: int):
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.PUB)
        self.sock.bind(f"tcp://*:{port}")
        time.sleep(0.1)  # let subscribers connect
        logger.info(f"[prior_pub] bound on tcp://*:{port}")

    def publish(self, pose_world: np.ndarray, frame_id: int) -> None:
        msg = {
            "frame_id": int(frame_id),
            "pose_world": pose_world.tolist(),
            "ts": time.time(),
        }
        self.sock.send_json(msg)

    def close(self) -> None:
        self.sock.close()


class FrameSyncBuffer:
    """Per-frame buffer: collects per-cam payloads keyed by frame_id.

    Pops oldest frame once it has >= min_cams payloads, or after timeout.
    """

    def __init__(self, min_cams: int, timeout_s: float, max_inflight: int):
        self.min_cams = int(min_cams)
        self.timeout_s = float(timeout_s)
        self.max_inflight = int(max_inflight)
        self._buf: Dict[int, Dict[str, Dict[str, Any]]] = {}
        self._first_seen: Dict[int, float] = {}
        self._lock = threading.Lock()

    def add(self, frame_id: int, serial: str, payload: Dict[str, Any]) -> None:
        with self._lock:
            slot = self._buf.setdefault(frame_id, {})
            slot[serial] = payload
            self._first_seen.setdefault(frame_id, time.time())

    def pop_ready(self) -> Optional[Tuple[int, Dict[str, Dict[str, Any]]]]:
        """Return (frame_id, payloads) for oldest frame that satisfies threshold
        or has timed out. Caller decides whether to skip a timed-out frame.
        """
        with self._lock:
            if not self._buf:
                return None
            oldest = min(self._buf.keys())
            slot = self._buf[oldest]
            seen = self._first_seen[oldest]
            ready = len(slot) >= self.min_cams
            timed_out = time.time() - seen >= self.timeout_s
            if ready or timed_out:
                payloads = self._buf.pop(oldest)
                self._first_seen.pop(oldest, None)
                return oldest, payloads

            # Drop very old frames if buffer is overloaded.
            if len(self._buf) > self.max_inflight:
                # Drop everything older than half the buffer.
                cutoff = sorted(self._buf.keys())[len(self._buf) // 2]
                for fid in list(self._buf.keys()):
                    if fid < cutoff:
                        self._buf.pop(fid, None)
                        self._first_seen.pop(fid, None)
            return None


class GoTrackTracker:
    """Robot-PC tracker: receives anchor obs, fuses to world pose, publishes prior."""

    def __init__(
        self,
        capture_pc_ips: List[str],
        port_obs: int = 1235,
        port_prior: int = 1236,
        min_cams_per_frame: int = 6,
        max_triangulation_residual_mm: float = 25.0,
        kabsch_inlier_thresh_mm: float = 35.0,
        confidence_weight_mode: str = "linear",
        confidence_weight_alpha: float = 1.0,
        external_unit_scale_to_meter: float = 1.0,
        frame_timeout_s: float = _DEFAULT_FRAME_TIMEOUT_S,
        max_inflight_frames: int = _DEFAULT_MAX_INFLIGHT_FRAMES,
    ):
        self.capture_pc_ips = list(capture_pc_ips)
        self.port_obs = int(port_obs)
        self.port_prior = int(port_prior)
        self.max_triangulation_residual_mm = float(max_triangulation_residual_mm)
        self.kabsch_inlier_thresh_mm = float(kabsch_inlier_thresh_mm)
        self.confidence_weight_mode = str(confidence_weight_mode)
        self.confidence_weight_alpha = float(confidence_weight_alpha)
        self.external_unit_scale_to_meter = float(external_unit_scale_to_meter)
        self.min_cams_per_frame = int(min_cams_per_frame)

        self.sync_buffer = FrameSyncBuffer(
            min_cams=min_cams_per_frame,
            timeout_s=frame_timeout_s,
            max_inflight=max_inflight_frames,
        )

        # SUB sockets — one per capture PC.
        self.ctx = zmq.Context.instance()
        self.sub_sockets: Dict[str, zmq.Socket] = {}
        self.poller = zmq.Poller()
        for ip in self.capture_pc_ips:
            sock = self.ctx.socket(zmq.SUB)
            sock.setsockopt_string(zmq.SUBSCRIBE, "")
            sock.connect(f"tcp://{ip}:{port_obs}")
            self.sub_sockets[ip] = sock
            self.poller.register(sock, zmq.POLLIN)
        logger.info(f"[tracker] subscribed to {len(self.sub_sockets)} capture PCs")

        self.prior_pub = PriorPosePublisher(port_prior)

        # Live status for dashboard. Lock protects dict mutation.
        self._status_lock = threading.Lock()
        self._fps_window: "deque[float]" = deque(maxlen=30)
        self.status: Dict[str, Any] = {
            "obj_name": None,
            "init_done": False,
            "init_ts": None,
            "frame_id": -1,
            "fps": 0.0,
            "last_fit_ok": None,
            "fail_reason": None,
            "n_inliers": 0,
            "mean_residual_mm": -1.0,
            "current_pose": None,
            "per_pc_last_frame": {},
            "started_ts": time.time(),
            "capture_pc_ips": list(self.capture_pc_ips),
        }

        self._stop = threading.Event()
        self._sub_thread = threading.Thread(target=self._sub_loop, daemon=True)
        self._sub_thread.start()

    def _sub_loop(self) -> None:
        while not self._stop.is_set():
            try:
                socks = dict(self.poller.poll(timeout=100))
            except zmq.ZMQError:
                continue
            for ip, sock in self.sub_sockets.items():
                if sock not in socks:
                    continue
                try:
                    parts = sock.recv_multipart(flags=zmq.NOBLOCK)
                except zmq.Again:
                    continue
                meta, bin_parts = _parse_multipart(parts)
                if meta is None:
                    continue
                if isinstance(meta, list):
                    items = meta
                elif isinstance(meta, dict) and "items" in meta:
                    items = meta["items"]
                elif isinstance(meta, dict):
                    items = [meta]
                else:
                    continue
                for item in items:
                    if item.get("type") != "gotrack_obs":
                        continue
                    payload = _unpack_payload(item, bin_parts)
                    self.sync_buffer.add(payload["frame_id"], payload["serial"], payload)
                    with self._status_lock:
                        self.status["per_pc_last_frame"][ip] = {
                            "frame_id": int(payload["frame_id"]),
                            "ts": time.time(),
                        }

    def publish_prior(self, pose_world: np.ndarray, frame_id: int) -> None:
        self.prior_pub.publish(pose_world, frame_id)

    def fuse_one_frame(
        self, payloads: Dict[str, Dict[str, Any]]
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Triangulate + Kabsch fit on one frame's payloads.

        Returns (pose_world or None, info_dict).
        """
        from utils.multiview_geometry import (
            robust_fit_pose_from_anchors,
            triangulate_anchor_observations,
            build_fit_weights_from_triangulation_records,
        )

        observations_by_anchor: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
        intrinsics_map: Dict[str, np.ndarray] = {}
        extrinsics_map: Dict[str, np.ndarray] = {}

        for serial, p in payloads.items():
            uv = p.get("uv_curr")
            conf = p.get("confidence")
            sel = p.get("selected_mask")
            anchor_ids = p.get("anchor_ids")
            positions_o = p.get("positions_o")
            ci = p.get("crop_intrinsic")
            Tw = p.get("T_world_from_crop_cam")
            if uv is None or conf is None or sel is None or anchor_ids is None \
               or positions_o is None or ci is None or Tw is None:
                continue
            intrinsics_map[serial] = np.asarray(ci, dtype=np.float64)
            # extrinsics expected as world->cam by triangulate (matches paradex).
            extrinsics_map[serial] = np.linalg.inv(np.asarray(Tw, dtype=np.float64))

            sel_idx = np.where(sel)[0]
            for i in sel_idx:
                aid = int(anchor_ids[i])
                observations_by_anchor[aid].append({
                    "camera_id": serial,
                    "uv_curr": np.asarray(uv[i], dtype=np.float32),
                    "confidence": float(conf[i]),
                    "position_o": np.asarray(positions_o[i], dtype=np.float32),
                    "valid_flag": True,
                })

        if not observations_by_anchor:
            n_payloads = len(payloads)
            sel_sums = {s: int(np.asarray(p.get("selected_mask")).sum())
                        if p.get("selected_mask") is not None else -1
                        for s, p in payloads.items()}
            logger.warning(f"[fuse] no_observations: n_payloads={n_payloads} "
                           f"selected_mask_sums={sel_sums}")
            return None, {"reason": "no_observations"}

        # Diagnostic: how many anchors are seen by multiple cameras?
        n_anchors = len(observations_by_anchor)
        view_counts = [len(v) for v in observations_by_anchor.values()]
        n_multi = sum(1 for c in view_counts if c >= 2)
        if n_multi == 0:
            # Sample anchor_ids per cam to verify they're canonical bank indices.
            per_cam_sample = {}
            for s, p in payloads.items():
                aids = p.get("anchor_ids")
                sel = p.get("selected_mask")
                if aids is not None and sel is not None:
                    sel_ids = np.asarray(aids)[np.asarray(sel, dtype=bool)]
                    per_cam_sample[s] = (int(sel_ids.size),
                                          sel_ids[:5].tolist() if sel_ids.size else [],
                                          int(sel_ids.min()) if sel_ids.size else -1,
                                          int(sel_ids.max()) if sel_ids.size else -1)
            logger.warning(f"[fuse] no anchors seen by ≥2 cams: "
                           f"n_anchors={n_anchors} max_views_per_anchor={max(view_counts)} "
                           f"n_payloads={len(payloads)}")
            logger.warning(f"[fuse]   per-cam (n_selected, first5_ids, min, max): {per_cam_sample}")

        tri = triangulate_anchor_observations(
            observations_by_anchor=observations_by_anchor,
            intrinsics_map=intrinsics_map,
            extrinsics_map=extrinsics_map,
            min_views=2,
            external_unit_scale_to_meter=self.external_unit_scale_to_meter,
            weight_mode=self.confidence_weight_mode,
            weight_alpha=self.confidence_weight_alpha,
        )
        records = tri.get("records", [])
        if not records:
            return None, {"reason": "triangulation_empty", "tri": tri}

        # Optional residual filter.
        residuals_pre = [r.get("max_residual_mm", -1) for r in records]
        if self.max_triangulation_residual_mm > 0.0:
            keep = []
            for r in records:
                resid = r.get("max_residual_mm", 0.0)
                if resid is None or resid <= self.max_triangulation_residual_mm:
                    keep.append(r)
            records = keep
        if not records:
            import numpy as _np
            arr = _np.asarray(residuals_pre, dtype=float)
            logger.warning(
                f"[fuse] all_filtered_by_residual  n_pre={len(residuals_pre)}  "
                f"min={arr.min():.2f}mm  median={_np.median(arr):.2f}mm  "
                f"max={arr.max():.2f}mm  thresh={self.max_triangulation_residual_mm:.2f}mm")
            return None, {"reason": "all_filtered_by_residual", "residuals_mm": residuals_pre}

        weights = build_fit_weights_from_triangulation_records(
            records,
            mode="geometry",
        )
        source_points_o = np.asarray([r["position_o"] for r in records], dtype=np.float32)
        target_points_w = np.asarray([r["position_w"] for r in records], dtype=np.float32)
        fit = robust_fit_pose_from_anchors(
            source_points_o,
            target_points_w,
            weights,
            inlier_threshold_mm=self.kabsch_inlier_thresh_mm,
            external_unit_scale_to_meter=self.external_unit_scale_to_meter,
        )
        pose_world = fit.get("pose_world_from_object")
        if pose_world is None:
            return None, {"reason": "fit_failed", "fit": fit}
        return np.asarray(pose_world, dtype=np.float64), {
            "n_triangulated": len(records),
            "n_inliers": int(fit.get("num_inlier_anchors", 0)),
            "mean_residual_mm": float(fit.get("mean_residual_mm", -1)),
            "fit": fit,
        }

    def track(
        self, init_pose_world: np.ndarray
    ) -> Iterator[Tuple[int, np.ndarray, Dict[str, Any]]]:
        """Generator: yield (frame_id, pose_world, info) every frame."""
        # Send initial prior so daemons can start processing.
        # Burst-publish to defeat ZMQ slow-joiner: SUBs that connect before the
        # PUB binds (e.g. daemons) miss messages sent at startup. CONFLATE=1
        # on the subscriber side means only the latest survives, so it's safe
        # to spam.
        for _ in range(20):
            self.publish_prior(init_pose_world, frame_id=-1)
            time.sleep(0.02)
        prev_pose = init_pose_world.astype(np.float64).copy()
        with self._status_lock:
            self.status["init_done"] = True
            self.status["init_ts"] = time.time()
            self.status["current_pose"] = prev_pose.tolist()

        last_idle_republish = time.time()
        while not self._stop.is_set():
            ready = self.sync_buffer.pop_ready()
            if ready is None:
                # While idle waiting for first obs, keep republishing the prior
                # every 0.5s in case daemons missed earlier broadcasts.
                if time.time() - last_idle_republish > 0.5:
                    self.publish_prior(prev_pose, frame_id=-1)
                    last_idle_republish = time.time()
                time.sleep(0.005)
                continue
            frame_id, payloads = ready
            if len(payloads) < self.min_cams_per_frame:
                # Timed-out frame with too few cams — still try; otherwise skip.
                logger.debug(f"[track] frame {frame_id}: only {len(payloads)} cams")
            pose_world, info = self.fuse_one_frame(payloads)

            now = time.time()
            self._fps_window.append(now)
            fps = 0.0
            if len(self._fps_window) >= 2:
                dt = self._fps_window[-1] - self._fps_window[0]
                if dt > 0:
                    fps = (len(self._fps_window) - 1) / dt
            with self._status_lock:
                counts = self.status.setdefault("counts", {
                    "received": 0, "success": 0,
                    "fail_by_reason": {},
                })
                counts["received"] += 1
                if pose_world is not None:
                    counts["success"] += 1
                else:
                    reason = str(info.get("reason", "unknown"))
                    counts["fail_by_reason"][reason] = counts["fail_by_reason"].get(reason, 0) + 1
                self.status["frame_id"] = int(frame_id)
                self.status["fps"] = float(fps)
                self.status["last_fit_ok"] = pose_world is not None
                self.status["fail_reason"] = info.get("reason") if pose_world is None else None
                if pose_world is not None:
                    self.status["n_inliers"] = int(info.get("n_inliers", 0))
                    self.status["mean_residual_mm"] = float(info.get("mean_residual_mm", -1))
                    self.status["current_pose"] = pose_world.tolist()
                else:
                    self.status["n_inliers"] = 0
                    self.status["mean_residual_mm"] = -1.0

            if pose_world is None:
                logger.warning(f"[track] frame {frame_id} fit failed: {info.get('reason')}")
                # Republish previous prior to keep daemons moving.
                self.publish_prior(prev_pose, frame_id=frame_id)
                continue
            self.publish_prior(pose_world, frame_id=frame_id)
            prev_pose = pose_world
            yield frame_id, pose_world, info

    def start_dashboard(self, port: int = 8090) -> threading.Thread:
        """Start a Flask dashboard thread exposing live status at http://0.0.0.0:{port}/."""
        from autodex.dashboard.tracking_monitor import run_dashboard
        t = threading.Thread(target=run_dashboard, args=(self, port), daemon=True)
        t.start()
        logger.info(f"[dashboard] http://0.0.0.0:{port}")
        return t

    def close(self) -> None:
        self._stop.set()
        self._sub_thread.join(timeout=1)
        for s in self.sub_sockets.values():
            s.close()
        self.prior_pub.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-ips", type=str, nargs="+", required=True,
                        help="IPs of capture1..6 PCs (one per PC).")
    parser.add_argument("--port-obs", type=int, default=1235)
    parser.add_argument("--port-prior", type=int, default=1236)
    parser.add_argument("--min-cams-per-frame", type=int, default=6)
    parser.add_argument("--init-pose-npy", type=str, required=True,
                        help="Path to .npy with 4x4 init pose_world.")
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--web-port", type=int, default=8090,
                        help="Dashboard port (0 to disable).")
    parser.add_argument("--obj-name", type=str, default="",
                        help="Object name to show on dashboard.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s %(message)s")

    init_pose = np.load(args.init_pose_npy)
    tracker = GoTrackTracker(
        capture_pc_ips=args.capture_ips,
        port_obs=args.port_obs,
        port_prior=args.port_prior,
        min_cams_per_frame=args.min_cams_per_frame,
    )
    if args.obj_name:
        with tracker._status_lock:
            tracker.status["obj_name"] = args.obj_name
    if args.web_port > 0:
        tracker.start_dashboard(args.web_port)
    try:
        n = 0
        for frame_id, pose, info in tracker.track(init_pose):
            print(f"frame {frame_id}: t={pose[:3, 3].tolist()}  "
                  f"n_inl={info.get('n_inliers')}  "
                  f"resid_mm={info.get('mean_residual_mm', -1):.2f}")
            n += 1
            if args.max_frames > 0 and n >= args.max_frames:
                break
    finally:
        tracker.close()


if __name__ == "__main__":
    main()
