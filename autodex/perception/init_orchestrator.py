#!/usr/bin/env python3
"""Robot PC orchestrator for distributed FoundPose first-frame init.

Sends `init` + `run` commands to capture1-6 init_daemon instances. Subscribes
to per-cam mask + pose streams. Once enough payloads arrive, runs cross-view
IoU pose selection followed by silhouette refinement on this PC.

Channels (must match init_daemon.py):
    REQ:    CommandSender    port 6893  (control)
    SUB:    DataPublisher    port 5006  (init_mask)  — subscribed per capture IP
    SUB:    DataPublisher    port 5007  (init_pose)  — subscribed per capture IP
"""
from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import sys
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import zmq

from autodex.fast_selection import select_best_pose_by_quality

logger = logging.getLogger(__name__)


def _to_home_relative(p) -> str:
    """Convert /home/<user>/... paths to ~/... so capture PCs can resolve under their own home."""
    p = str(p)
    home = str(Path.home())
    if p.startswith(home + "/"):
        return "~/" + p[len(home) + 1:]
    return p


def _parse_multipart(parts: List[bytes]) -> Tuple[Optional[Dict], List[bytes]]:
    """paradex envelope format: [topic, msgpack_header{seq,ts,src,meta,n}, *bufs].

    The transport migrated from the OLD ``[b'data', metadata_json, *blobs]`` JSON
    layout to a msgpack envelope — parsing parts[1] as JSON silently dropped every
    payload (masks/poses 0/N). Use paradex's own decoder; ``msg.meta`` is the same
    per-item metadata that ``send_data(metadata, data)`` published, ``msg.bufs``
    the raw blobs. Falls back to the old JSON layout for safety."""
    try:
        from paradex.io.capture_pc.envelope import decode
        msg = decode(parts)
        return msg.meta, list(msg.bufs)
    except Exception:
        pass
    # legacy fallback: [b'data', metadata_json, *blobs]
    if len(parts) < 2 or parts[0] != b"data":
        return None, []
    try:
        return json.loads(parts[1].decode("utf-8")), list(parts[2:])
    except Exception:
        return None, []


def _progress(elapsed: float, n_mask: int, n_pose: int, n_expected: int,
              done: bool = False) -> None:
    """One self-overwriting progress line (\r), like tqdm.

    A wait can run tens of seconds at ~2 prints/s; on its own line each that is
    a hundred lines of scrollback burying whatever came before. Ends with a
    newline once, when done.
    """
    line = (f"  ... [{elapsed:5.1f}s] masks {n_mask}/{n_expected}  "
            f"poses {n_pose}/{n_expected}")
    # Pad to clear a previously longer line, since \r only moves the cursor.
    sys.stdout.write("\r" + line.ljust(72))
    if done:
        sys.stdout.write("\n")
    sys.stdout.flush()


class _Buffer:
    """Thread-safe buffer of {req_id: {serial: payload}}."""
    def __init__(self):
        self._d: Dict[int, Dict[str, Any]] = defaultdict(dict)
        self._lock = threading.Lock()

    def put(self, req_id: int, serial: str, payload: Any) -> None:
        with self._lock:
            self._d[req_id][serial] = payload

    def get(self, req_id: int) -> Dict[str, Any]:
        with self._lock:
            return dict(self._d.get(req_id, {}))

    def drop(self, req_id: int) -> None:
        with self._lock:
            self._d.pop(req_id, None)


class _SubThread(threading.Thread):
    """SUB to N capture PCs on one port, parse multipart, store in buffer."""
    def __init__(self, name: str, capture_ips: List[str], port: int,
                 buffer: _Buffer, on_message=None):
        super().__init__(daemon=True, name=name)
        self.ctx = zmq.Context.instance()
        self.sock = self.ctx.socket(zmq.SUB)
        self.sock.setsockopt_string(zmq.SUBSCRIBE, "")
        for ip in capture_ips:
            self.sock.connect(f"tcp://{ip}:{port}")
        self.buffer = buffer
        self.on_message = on_message
        # NOT `self._stop`: threading.Thread uses that name internally, and
        # shadowing it with an Event makes join() raise
        # "'Event' object is not callable" — the thread can never be waited on.
        self._stop_evt = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def run(self):
        try:
            self._run()
        finally:
            # The socket MUST be closed from the thread that polls it, and it
            # must be closed at all: zmq's context.term() blocks until every
            # socket in the context is closed, so leaking this one hangs
            # shutdown forever.
            try:
                self.sock.setsockopt(zmq.LINGER, 0)
                self.sock.close()
            except Exception:
                pass

    def _run(self):
        while not self._stop_evt.is_set():
            try:
                if self.sock.poll(timeout=100):
                    parts = self.sock.recv_multipart(flags=zmq.NOBLOCK)
                    msg, blobs = _parse_multipart(parts)
                    if msg is None or self.on_message is None:
                        continue
                    # New paradex envelope: msg.meta is a LIST of per-item dicts
                    # ([{"req_id","serial",...}]) with one blob per item. Support
                    # the old dict-wrapped ({"items":[...]}) and flat-dict layouts too.
                    if isinstance(msg, list):
                        for item, blob in zip(msg, blobs):
                            self.on_message(item, blob)
                    elif isinstance(msg, dict) and "items" in msg:
                        for item, blob in zip(msg["items"], blobs):
                            self.on_message(item, blob)
                    elif isinstance(msg, dict):
                        self.on_message(msg, blobs[0] if blobs else b"")
            except zmq.Again:
                pass
            except zmq.ContextTerminated:
                # Someone tore the context down under us — nothing left to
                # read. Return quietly so the finally: above closes the socket.
                return
            except Exception as exc:
                logger.warning(f"[{self.name}] {exc}")


class InitOrchestrator:
    """Coordinates distributed init across capture1-6.

    Parameters
    ----------
    pc_list : list of paradex PC names (e.g. ["capture1", ..., "capture6"]).
    capture_ips : list of IPs (one per PC, same order as pc_list).
    port_mask, port_pose : daemon's PUB ports.
    port_cmd : daemon's REQ/REP control port.
    """

    def __init__(
        self,
        pc_list: List[str],
        capture_ips: List[str],
        port_mask: int = 5006,
        port_pose: int = 5007,
        port_cmd: int = 6893,
        device: str = "cuda:0",
        command_timeout_ms: int = 60000,
        command_retries: int = 3,
    ):
        from paradex.io.capture_pc.command_sender import CommandSender

        assert len(pc_list) == len(capture_ips)
        if command_timeout_ms < 1 or command_retries < 1:
            raise ValueError("command timeout and retries must be positive")
        self.pc_list = pc_list
        self.capture_ips = capture_ips
        self.cmd = CommandSender(
            pc_list=pc_list, port=port_cmd, timeout=int(command_timeout_ms),
            retries=int(command_retries),
        )

        self.mask_buf = _Buffer()
        self.pose_buf = _Buffer()

        def _on_mask(item, blob):
            req = int(item["req_id"]); s = str(item["serial"])
            png = np.frombuffer(blob, dtype=np.uint8)
            mask_u8 = cv2.imdecode(png, cv2.IMREAD_GRAYSCALE)
            self.mask_buf.put(req, s, {
                "mask": (mask_u8 > 127) if mask_u8 is not None else None,
                "h": int(item["h"]), "w": int(item["w"]),
                "t_sam3": float(item.get("t_sam3", 0.0)),
                "ts": float(item.get("ts", 0.0)),
            })

        def _on_pose(item, blob):
            req = int(item["req_id"]); s = str(item["serial"])
            ok = bool(item.get("ok", False))
            entry = {"ok": ok, "t_fp": float(item.get("t_fp", 0.0)),
                     "ts": float(item.get("ts", 0.0))}
            if ok and len(blob) == 16 * 8:
                entry["pose_world"] = np.frombuffer(blob, dtype=np.float64).reshape(4, 4).copy()
                entry["quality"] = float(item.get("quality", 0.0))
                entry["inliers"] = int(item.get("inliers", 0))
                entry["mask_pixels"] = int(item.get("mask_pixels", 0))
            self.pose_buf.put(req, s, entry)

        self._mask_thread = _SubThread("init_mask", capture_ips, port_mask,
                                       self.mask_buf, _on_mask)
        self._pose_thread = _SubThread("init_pose", capture_ips, port_pose,
                                       self.pose_buf, _on_pose)
        self._mask_thread.start()
        self._pose_thread.start()
        time.sleep(0.3)  # let SUB sockets connect

        # robot-side state set by init_object()
        self.obj_name: Optional[str] = None
        self.intrinsics_undist: Dict[str, np.ndarray] = {}
        self.extrinsics: Dict[str, np.ndarray] = {}
        self.H: int = 0
        self.W: int = 0
        self._sil = None
        self.device = device

    # ── lifecycle ──

    def init_object(
        self,
        obj_name: str,
        mesh_path: str,
        assets_root: str,
        intrinsics_full: Dict[str, Dict[str, Any]],
        extrinsics_full: Dict[str, np.ndarray],
        image_hw: Tuple[int, int],
        mode: str = "live",
        pc_serials: Optional[Dict[str, List[str]]] = None,
        load_silhouette: bool = True,
    ) -> None:
        """Send init to all capture daemons and optionally load silhouette state.

        intrinsics_full : {serial: {K_orig (3x3), K_undist (3x3), dist_params (5,), width, height}}
        extrinsics_full : {serial: 4x4 world->cam}
        image_hw : (H, W) of undistorted images
        """
        self.obj_name = obj_name
        self.H, self.W = int(image_hw[0]), int(image_hw[1])
        # Robot-side per-cam params (ALL cams across all PCs).
        self.intrinsics_undist = {
            s: np.asarray(intrinsics_full[s]["K_undist"], dtype=np.float64).reshape(3, 3)
            for s in intrinsics_full
        }
        self.extrinsics = {
            s: np.asarray(extrinsics_full[s], dtype=np.float64).reshape(4, 4)
            for s in extrinsics_full
        }

        # Build cmd_info — must be JSON-serializable.
        intr_jsonable = {
            s: {
                "K_orig": np.asarray(v["K_orig"], dtype=np.float64).reshape(3, 3).tolist(),
                "K_undist": np.asarray(v["K_undist"], dtype=np.float64).reshape(3, 3).tolist(),
                "dist_params": np.asarray(v["dist_params"], dtype=np.float64).reshape(-1).tolist(),
                "width": int(v["width"]), "height": int(v["height"]),
            }
            for s, v in intrinsics_full.items()
        }
        extr_jsonable = {
            s: np.asarray(v, dtype=np.float64).reshape(4, 4).tolist()
            for s, v in extrinsics_full.items()
        }

        # In disk mode each PC needs to be told its serial subset (no SHM auto-detect).
        if mode == "disk":
            if pc_serials is None:
                raise ValueError("disk mode requires pc_serials={pc_name: [serials]}")
            with contextlib.redirect_stdout(io.StringIO()):
                for pc in self.pc_list:
                    info_pc = {
                        "obj_name": obj_name,
                        "mesh_path": _to_home_relative(mesh_path),
                        "assets_root": _to_home_relative(assets_root),
                        "intrinsics": intr_jsonable,
                        "extrinsics": extr_jsonable,
                        "mode": "disk",
                        "my_serials": list(pc_serials.get(pc, [])),
                    }
                    self.cmd._send_to_pc(pc, "init", wait=False, cmd_info=info_pc)
            logger.info(f"[orch] init (disk mode) dispatched to {len(self.pc_list)} PCs")
        else:
            info = {
                "obj_name": obj_name,
                "mesh_path": _to_home_relative(mesh_path),
                "assets_root": _to_home_relative(assets_root),
                "intrinsics": intr_jsonable,
                "extrinsics": extr_jsonable,
                "mode": "live",
            }
            logger.info(f"[orch] sending init (live) for {obj_name} to {len(self.pc_list)} PCs...")
            t0 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                self.cmd.send_command("init", wait=False, cmd_info=info)
            logger.info(f"[orch] init dispatched in {time.perf_counter()-t0:.1f}s")

        # The quality path used by the continuous demo never renders a mesh or
        # optimises a silhouette, so avoid paying to construct its CUDA state.
        # Keep the legacy default for existing callers and evaluation scripts.
        if not load_silhouette:
            return

        # Load silhouette optimizer locally (once per object).
        from autodex.perception.silhouette import SilhouetteOptimizer
        if self._sil is None or getattr(self._sil, "_obj_name", None) != obj_name:
            t0 = time.perf_counter()
            self._sil = SilhouetteOptimizer(str(mesh_path), device=self.device)
            self._sil._obj_name = obj_name
            logger.info(f"[orch] sil optimizer loaded in {time.perf_counter()-t0:.1f}s")

    def collect_payloads(
        self,
        prompt: str = "object",
        request_id: Optional[int] = None,
        n_expected_serials: Optional[int] = None,
        timeout_s: float = 15.0,
        capture_dir: Optional[str] = None,
        save_capture_dir: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        """Trigger one capture across all capture PCs and return raw payloads.

        Returns (masks, poses, timing). Each masks/poses key is a serial.
        """
        if request_id is None:
            request_id = int(time.time() * 1000) & 0x7fffffff
        n_expected = n_expected_serials or len(self.intrinsics_undist)

        for buf in (self.mask_buf, self.pose_buf):
            with buf._lock:
                buf._d.clear()

        t_dispatch = time.perf_counter()
        run_info = {"request_id": int(request_id), "prompt": prompt}
        if capture_dir is not None:
            run_info["capture_dir"] = _to_home_relative(capture_dir)
        if save_capture_dir is not None:
            run_info["save_capture_dir"] = _to_home_relative(save_capture_dir)
        with contextlib.redirect_stdout(io.StringIO()):
            self.cmd.send_command("run", wait=False, cmd_info=run_info)

        deadline = time.perf_counter() + timeout_s
        first_mask_t = None; first_pose_t = None
        last_print = 0.0
        last_n_mask = -1; last_n_pose = -1
        interrupted = False
        while time.perf_counter() < deadline:
            try:
                masks_now = self.mask_buf.get(request_id)
                poses_now = self.pose_buf.get(request_id)
            except KeyboardInterrupt:
                interrupted = True
                break
            if first_mask_t is None and masks_now:
                first_mask_t = time.perf_counter()
            if first_pose_t is None and poses_now:
                first_pose_t = time.perf_counter()
            now = time.perf_counter()
            if (now - last_print > 0.5
                    or len(masks_now) != last_n_mask
                    or len(poses_now) != last_n_pose):
                elapsed = now - t_dispatch
                _progress(elapsed, len(masks_now), len(poses_now), n_expected)
                last_print = now
                last_n_mask = len(masks_now); last_n_pose = len(poses_now)
            if len(masks_now) >= n_expected and len(poses_now) >= n_expected:
                break
            try:
                time.sleep(0.01)
            except KeyboardInterrupt:
                interrupted = True
                break
        if interrupted:
            # Ctrl-C during the wait = "stop waiting, use what arrived", not
            # "kill the run": the caller still has partial masks/poses and the
            # robot is mid-trial.
            sys.stdout.write("\n")
            logger.warning("[orch] wait interrupted by user — "
                           "continuing with whatever arrived")
        _progress(time.perf_counter() - t_dispatch, len(self.mask_buf.get(request_id)),
                  len(self.pose_buf.get(request_id)), n_expected, done=True)
        masks = self.mask_buf.get(request_id)
        poses = self.pose_buf.get(request_id)
        t_collected = time.perf_counter()
        self.mask_buf.drop(request_id); self.pose_buf.drop(request_id)
        timing = {
            "request_id": request_id,
            "dispatch_to_collected_s": t_collected - t_dispatch,
            "first_mask_arrived_s": (first_mask_t - t_dispatch) if first_mask_t else None,
            "first_pose_arrived_s": (first_pose_t - t_dispatch) if first_pose_t else None,
            "n_masks_recv": len(masks),
            "n_poses_recv": len(poses),
        }
        return masks, poses, timing

    def refine_from_payloads(
        self,
        masks: Dict[str, Any],
        poses: Dict[str, Any],
        subset_serials: Optional[List[str]] = None,
        sil_iters: int = 100,
        sil_lr: float = 0.002,
        sil_loss_threshold: float = 0.003,
        save_capture_dir: Optional[str] = None,
        sil_debug: bool = False,
        selection_mode: str = "iou",
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Select/refine already-collected payloads, restricted to a subset.

        subset_serials=None → use every serial known to the orchestrator.
        ``selection_mode='quality'`` chooses the highest FoundPose quality
        directly and intentionally does no mesh rendering or silhouette
        refinement.  It is for latency-sensitive live retries; ``'iou'``
        preserves the previous evaluation-grade behaviour.
        """
        if selection_mode not in {"iou", "quality"}:
            raise ValueError("selection_mode must be 'iou' or 'quality'")

        if subset_serials is None:
            subset = set(self.intrinsics_undist.keys())
        else:
            subset = set(subset_serials) & set(self.intrinsics_undist.keys())

        candidates: Dict[str, np.ndarray] = {}
        for s, p in poses.items():
            if s not in subset:
                continue
            if p.get("ok") and "pose_world" in p:
                candidates[s] = p["pose_world"]
        masks_bool: Dict[str, np.ndarray] = {
            s: m["mask"] for s, m in masks.items()
            if s in subset and m.get("mask") is not None and m["mask"].any()
        }
        if not candidates or (selection_mode == "iou" and not masks_bool):
            return None, {"reason": "no_candidates_or_masks",
                          "n_candidates": len(candidates),
                          "n_masks": len(masks_bool),
                          "selection_mode": selection_mode}

        if selection_mode == "quality":
            best_serial, best_pose, per_cand = select_best_pose_by_quality(
                candidates, poses)
            return best_pose, {
                "selection_mode": "quality",
                "sil_skipped": True,
                "iou_select_s": 0.0,
                "sil_refine_s": 0.0,
                "n_candidates": len(candidates),
                "n_masks": len(masks_bool),
                "best_serial": best_serial,
                "best_quality": float(per_cand.get(best_serial, 0.0)),
                "per_cand": per_cand,
            }

        from autodex.perception.pose_select import select_best_pose_by_iou
        if self._sil is None:
            raise RuntimeError("IoU selection requires init_object(load_silhouette=True)")

        t_iou0 = time.perf_counter()
        intr_subset = {s: self.intrinsics_undist[s] for s in masks_bool}
        extr_subset = {s: self.extrinsics[s] for s in masks_bool}
        best_serial, best_pose, best_iou, per_cand = select_best_pose_by_iou(
            candidates=candidates, masks=masks_bool,
            intrinsics=intr_subset, extrinsics=extr_subset,
            H=self.H, W=self.W,
            glctx=self._sil.glctx, mesh_tensors=self._sil.mesh_tensors,
        )
        t_iou = time.perf_counter() - t_iou0
        if best_pose is None:
            return None, {"reason": "iou_select_failed", "per_cand": per_cand}

        if sil_iters <= 0:
            return np.asarray(best_pose, dtype=np.float64), {
                "selection_mode": "iou",
                "sil_skipped": True,
                "iou_select_s": t_iou, "sil_refine_s": 0.0,
                "n_candidates": len(candidates), "n_masks": len(masks_bool),
                "best_serial": best_serial, "best_iou": float(best_iou),
                "pre_sil_pose": np.asarray(best_pose, dtype=np.float64).tolist(),
            }

        t_sil0 = time.perf_counter()
        views = [
            {"mask": (m.astype(np.uint8) * 255),
             "K": intr_subset[s], "extrinsic": extr_subset[s]}
            for s, m in masks_bool.items()
        ]
        sil_debug_dir = None
        if sil_debug and save_capture_dir is not None:
            sil_debug_dir = os.path.join(save_capture_dir, "sil_debug")
            os.makedirs(sil_debug_dir, exist_ok=True)
        refined, sil_loss = self._sil.optimize(
            initial_pose_world=best_pose,
            views=views,
            iters=sil_iters, lr=sil_lr,
            antialias=True,
            debug=sil_debug_dir is not None,
            debug_dir=sil_debug_dir,
            debug_every=10, debug_max_views=4,
        )
        t_sil = time.perf_counter() - t_sil0
        timing = {
            "iou_select_s": t_iou, "sil_refine_s": t_sil,
            "n_candidates": len(candidates), "n_masks": len(masks_bool),
            "best_serial": best_serial, "best_iou": float(best_iou),
            "sil_loss": float(sil_loss),
            "pre_sil_pose": np.asarray(best_pose, dtype=np.float64).tolist(),
        }
        if sil_loss > sil_loss_threshold:
            timing["sil_reject"] = True
            timing["reason"] = f"sil_loss_too_high ({sil_loss:.6f})"
            return None, timing
        return np.asarray(refined, dtype=np.float64), timing

    def trigger_init(
        self,
        prompt: str = "object",
        request_id: Optional[int] = None,
        n_expected_serials: Optional[int] = None,
        timeout_s: float = 15.0,
        sil_iters: int = 100,
        sil_lr: float = 0.002,
        capture_dir: Optional[str] = None,
        save_capture_dir: Optional[str] = None,
        sil_loss_threshold: float = 0.003,
        selection_mode: str = "iou",
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """Trigger one init across all capture PCs and select a pose on robot.

        Returns (pose_world, timing_dict). pose_world is None on failure.
        Pass ``selection_mode='quality'`` for the low-latency path: it selects
        the best FoundPose view by its quality/inlier metadata and skips both
        cross-view silhouette rendering and iterative silhouette refinement.
        """
        if selection_mode not in {"iou", "quality"}:
            raise ValueError("selection_mode must be 'iou' or 'quality'")

        if request_id is None:
            request_id = int(time.time() * 1000) & 0x7fffffff
        n_expected = n_expected_serials or len(self.intrinsics_undist)

        # Drop any buffered payloads from prior trials (only keep current req_id).
        for buf in (self.mask_buf, self.pose_buf):
            with buf._lock:
                buf._d.clear()

        # Send "run" to all PCs (silence paradex's per-PC print).
        t_dispatch = time.perf_counter()
        run_info = {"request_id": int(request_id), "prompt": prompt}
        if capture_dir is not None:
            run_info["capture_dir"] = _to_home_relative(capture_dir)
        if save_capture_dir is not None:
            run_info["save_capture_dir"] = _to_home_relative(save_capture_dir)
        with contextlib.redirect_stdout(io.StringIO()):
            self.cmd.send_command("run", wait=False, cmd_info=run_info)

        # Wait for masks + poses (poll buffers, progress every ~0.5s).
        deadline = time.perf_counter() + timeout_s
        first_mask_t = None; first_pose_t = None
        last_print = 0.0
        last_n_mask = -1; last_n_pose = -1
        interrupted = False
        while time.perf_counter() < deadline:
            try:
                masks_now = self.mask_buf.get(request_id)
                poses_now = self.pose_buf.get(request_id)
            except KeyboardInterrupt:
                interrupted = True
                break
            if first_mask_t is None and masks_now:
                first_mask_t = time.perf_counter()
            if first_pose_t is None and poses_now:
                first_pose_t = time.perf_counter()
            now = time.perf_counter()
            if (now - last_print > 0.5
                    or len(masks_now) != last_n_mask
                    or len(poses_now) != last_n_pose):
                elapsed = now - t_dispatch
                _progress(elapsed, len(masks_now), len(poses_now), n_expected)
                last_print = now
                last_n_mask = len(masks_now); last_n_pose = len(poses_now)
            if len(masks_now) >= n_expected and len(poses_now) >= n_expected:
                break
            try:
                time.sleep(0.01)
            except KeyboardInterrupt:
                interrupted = True
                break
        if interrupted:
            # Ctrl-C during the wait = "stop waiting, use what arrived", not
            # "kill the run": the caller still has partial masks/poses and the
            # robot is mid-trial.
            sys.stdout.write("\n")
            logger.warning("[orch] wait interrupted by user — "
                           "continuing with whatever arrived")
        _progress(time.perf_counter() - t_dispatch, len(self.mask_buf.get(request_id)),
                  len(self.pose_buf.get(request_id)), n_expected, done=True)
        masks = self.mask_buf.get(request_id)
        poses = self.pose_buf.get(request_id)
        t_collected = time.perf_counter()
        logger.info(f"[orch] req={request_id} collected: "
                    f"{len(masks)} masks / {len(poses)} poses in "
                    f"{t_collected-t_dispatch:.2f}s")
        expected_serials = set(self.intrinsics_undist.keys())
        missing_mask = sorted(expected_serials - set(masks.keys()))
        missing_pose = sorted(expected_serials - set(poses.keys()))
        if missing_mask:
            print(f"  [orch] missing masks ({len(missing_mask)}): "
                  f"{missing_mask}", flush=True)
        if missing_pose:
            print(f"  [orch] missing poses ({len(missing_pose)}): "
                  f"{missing_pose}", flush=True)

        # Build candidates: serial -> pose_world (only OK ones).
        candidates: Dict[str, np.ndarray] = {}
        for s, p in poses.items():
            if p.get("ok") and "pose_world" in p:
                candidates[s] = p["pose_world"]
        masks_bool: Dict[str, np.ndarray] = {
            s: m["mask"] for s, m in masks.items()
            if m.get("mask") is not None and m["mask"].any()
        }
        if not candidates or (selection_mode == "iou" and not masks_bool):
            self.mask_buf.drop(request_id); self.pose_buf.drop(request_id)
            return None, {
                "reason": "no_candidates_or_masks",
                "n_candidates": len(candidates), "n_masks": len(masks_bool),
                "dispatch_to_collected_s": t_collected - t_dispatch,
                "selection_mode": selection_mode,
            }

        if selection_mode == "quality":
            best_serial, best_pose, per_cand = select_best_pose_by_quality(
                candidates, poses)
            timing = {
                "dispatch_to_collected_s": t_collected - t_dispatch,
                "first_mask_arrived_s": (first_mask_t - t_dispatch) if first_mask_t else None,
                "first_pose_arrived_s": (first_pose_t - t_dispatch) if first_pose_t else None,
                "iou_select_s": 0.0,
                "sil_refine_s": 0.0,
                "total_s": time.perf_counter() - t_dispatch,
                "n_candidates": len(candidates), "n_masks": len(masks_bool),
                "best_serial": best_serial,
                "best_quality": float(per_cand.get(best_serial, 0.0)),
                "per_cand": per_cand,
                "selection_mode": "quality", "sil_skipped": True,
            }
            self.mask_buf.drop(request_id); self.pose_buf.drop(request_id)
            return best_pose, timing

        from autodex.perception.pose_select import select_best_pose_by_iou
        if self._sil is None:
            self.mask_buf.drop(request_id); self.pose_buf.drop(request_id)
            raise RuntimeError("IoU selection requires init_object(load_silhouette=True)")

        # Cross-view IoU select.
        t_iou0 = time.perf_counter()
        intr_subset = {s: self.intrinsics_undist[s] for s in masks_bool if s in self.intrinsics_undist}
        extr_subset = {s: self.extrinsics[s] for s in masks_bool if s in self.extrinsics}
        best_serial, best_pose, best_iou, per_cand = select_best_pose_by_iou(
            candidates=candidates,
            masks=masks_bool,
            intrinsics=intr_subset,
            extrinsics=extr_subset,
            H=self.H, W=self.W,
            glctx=self._sil.glctx,
            mesh_tensors=self._sil.mesh_tensors,
        )
        t_iou = time.perf_counter() - t_iou0
        logger.info(f"[orch] IoU select: best={best_serial} mean_iou={best_iou:.3f} "
                    f"(took {t_iou:.2f}s)")
        if best_pose is None:
            self.mask_buf.drop(request_id); self.pose_buf.drop(request_id)
            return None, {"reason": "iou_select_failed", "per_cand": per_cand}

        if sil_iters <= 0:
            timing = {
                "dispatch_to_collected_s": t_collected - t_dispatch,
                "first_mask_arrived_s": (first_mask_t - t_dispatch) if first_mask_t else None,
                "first_pose_arrived_s": (first_pose_t - t_dispatch) if first_pose_t else None,
                "iou_select_s": t_iou, "sil_refine_s": 0.0,
                "total_s": time.perf_counter() - t_dispatch,
                "n_candidates": len(candidates), "n_masks": len(masks_bool),
                "best_serial": best_serial, "best_iou": float(best_iou),
                "pre_sil_pose": np.asarray(best_pose, dtype=np.float64).tolist(),
                "selection_mode": "iou", "sil_skipped": True,
            }
            self.mask_buf.drop(request_id); self.pose_buf.drop(request_id)
            return np.asarray(best_pose, dtype=np.float64), timing

        # Sil refine on robot PC using collected masks.
        t_sil0 = time.perf_counter()
        views = []
        for s, m in masks_bool.items():
            if s not in intr_subset or s not in extr_subset:
                continue
            views.append({
                "mask": (m.astype(np.uint8) * 255),
                "K": intr_subset[s],
                "extrinsic": extr_subset[s],
            })
        sil_debug_dir = None
        refined, sil_loss = self._sil.optimize(
            initial_pose_world=best_pose,
            views=views,
            iters=sil_iters, lr=sil_lr,
            antialias=True,
            debug=False,
            debug_dir=sil_debug_dir,
            debug_every=10,
            debug_max_views=4,
        )
        t_sil = time.perf_counter() - t_sil0
        logger.info(f"[orch] sil refine: {t_sil:.2f}s ({sil_iters} iters, loss={sil_loss:.6f})")

        timing = {
            "dispatch_to_collected_s": t_collected - t_dispatch,
            "first_mask_arrived_s": (first_mask_t - t_dispatch) if first_mask_t else None,
            "first_pose_arrived_s": (first_pose_t - t_dispatch) if first_pose_t else None,
            "iou_select_s": t_iou,
            "sil_refine_s": t_sil,
            "total_s": time.perf_counter() - t_dispatch,
            "n_candidates": len(candidates),
            "n_masks": len(masks_bool),
            "best_serial": best_serial,
            "best_iou": float(best_iou),
            "sil_loss": float(sil_loss),
            "pre_sil_pose": np.asarray(best_pose, dtype=np.float64).tolist(),
        }
        self.mask_buf.drop(request_id); self.pose_buf.drop(request_id)

        # Threshold: silhouette matching loss above threshold means the refined
        # pose is unreliable. Set to float('inf') to disable.
        if sil_loss > sil_loss_threshold:
            logger.warning(f"[orch] sil loss {sil_loss:.6f} > {sil_loss_threshold} — pose unreliable, skipping")
            timing["sil_reject"] = True
            timing["reason"] = f"sil_loss_too_high ({sil_loss:.6f})"
            return None, timing

        return np.asarray(refined, dtype=np.float64), timing

    def close(self) -> None:
        # Stop our SUB threads. Do NOT call self.cmd.end() — that broadcasts
        # "exit" and kills the daemons, which we want to keep alive across
        # interactive sessions. Just close the local sockets.
        self._mask_thread.stop(); self._pose_thread.stop()
        # Wait for them to actually exit — stop() only sets a flag, and each
        # closes its SUB socket on the way out. Closing those sockets from
        # here instead would be a cross-thread close, which zmq forbids.
        for t in (self._mask_thread, self._pose_thread):
            try:
                t.join(timeout=3.0)
                if t.is_alive():
                    logger.warning(f"[close] {t.name} did not exit in 3s")
            except Exception:
                pass
        try:
            # LINGER=0 so close() doesn't wait for queued messages to drain to
            # a dead daemon — otherwise it hangs indefinitely on any REQ socket
            # left mid-cycle (wait=False sends).
            for s in self.cmd.sockets.values():
                try:
                    s.setsockopt(zmq.LINGER, 0)
                except Exception:
                    pass
                s.close()
        except Exception:
            pass
        # Deliberately NOT calling self.cmd.context.term(): CommandSender uses
        # zmq.Context.instance(), the process-wide singleton that the camera
        # controller and timestamp monitor also share. Terminating it kills
        # THEIR sockets too ("Context was terminated"), and it blocks until
        # every socket in the process is closed — which is what made shutdown
        # hang. Our own sockets are closed above; the rest is the OS's job at
        # exit.
