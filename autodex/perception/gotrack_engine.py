"""In-process GoTrack stage 1-4 wrapper for one capture PC (N cameras).

Wraps the MV-GoTrack online tracker so a daemon can call ``process_frame``
once per multi-cam frame, get back per-camera anchor observations, and
forward them to the central robot PC for triangulation + Kabsch fit (stages
5-6).

Stages handled here (per-camera, run on the local GPU):
  1. Template render (nvdiffrast) using the prior pose
  2. Crop (template + live image) to model.opts.crop_size
  3. DINOv2 forward → flow + confidence
  4. Anchor 2D observations (project canonical anchors → sample flow → live uv)

Stages NOT handled (run on robot PC after gathering 24-cam obs):
  5. Multi-view triangulation
  6. Robust Kabsch fit (RANSAC) → world pose

Must run inside the ``gotrack`` conda env (depends on torch 2.0 + xformers
+ the MV-GoTrack code under autodex/perception/thirdparty/MV-GoTrack/).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

_GOTRACK_ROOT = Path(__file__).resolve().parent / "thirdparty/MV-GoTrack"
if str(_GOTRACK_ROOT) not in sys.path:
    sys.path.insert(0, str(_GOTRACK_ROOT))

# Default GoTrack runtime knobs (matches gotrack_pipeline_debug.py).
_DEFAULT_CONFIG = "configs/model/gotrack.yaml"
_DEFAULT_CKPT = _GOTRACK_ROOT / "gotrack_checkpoint.pt"


@dataclass
class CameraIntrinsics:
    serial: str
    K: np.ndarray            # 3x3
    extrinsic_cw: np.ndarray  # 4x4 world->cam
    width: int
    height: int


def _build_camera_model(intr: CameraIntrinsics, translation_scale: float = 1.0):
    """Wrap our intrinsics+extrinsics into the GoTrack PinholePlaneCameraModel.

    translation_scale: applied to T_world_from_eye[:3, 3]. Must be 1.0 for the
    plain ``camera_models`` slot and ``unit_info.translation_scale_to_gotrack``
    for ``gotrack_camera_models`` (matches MV-GoTrack's video pipeline)."""
    from utils import structs
    T_world_from_eye = np.linalg.inv(np.asarray(intr.extrinsic_cw, dtype=np.float64))
    T_world_from_eye[:3, 3] *= float(translation_scale)
    return structs.PinholePlaneCameraModel(
        width=int(intr.width), height=int(intr.height),
        f=(float(intr.K[0, 0]), float(intr.K[1, 1])),
        c=(float(intr.K[0, 2]), float(intr.K[1, 2])),
        T_world_from_eye=T_world_from_eye,
    )


def _build_args_namespace(
    *,
    object_id: int = 1,
    mask_free: bool = True,
    skip_pnp: bool = True,
    confidence_threshold: float = 0.25,
    anchor_confidence_threshold: float = 0.25,
    anchor_depth_tolerance_m: float = 0.05,
    max_active_anchors_per_view: int = 256,
    mask_threshold: int = 127,
    min_mask_pixels: int = 200,
) -> argparse.Namespace:
    """Build a minimal argparse-style namespace consumed by GoTrack helpers."""
    return argparse.Namespace(
        object_id=object_id,
        mask_free=mask_free,
        skip_pnp=skip_pnp,
        confidence_threshold=confidence_threshold,
        anchor_confidence_threshold=anchor_confidence_threshold,
        anchor_depth_tolerance_m=anchor_depth_tolerance_m,
        max_active_anchors_per_view=max_active_anchors_per_view,
        mask_threshold=mask_threshold,
        min_mask_pixels=min_mask_pixels,
        # Optimized input pipeline knobs (off; we batch per-frame on this PC).
        optimized_input_pipeline=False,
        optimized_input_pipeline_v2=False,
        optim_crop_update_interval=0,
        optim_template_update_interval=1,
        optim_template_render_workers=1,
        optim_v2_crop_camera_workers=1,
        optim_v2_warp_grid_workers=1,
        template_renderer_backend="nvdiffrast",
    )


class GoTrackEngine:
    """One-PC GoTrack inference for N cameras.

    Args:
        mesh_path: object mesh (used for template render + anchor projection).
        anchor_bank_path: pre-generated anchor bank (.npz from
            scripts/generate_anchor_bank.py).
        cameras: list of CameraIntrinsics for the N cams attached to this PC.
        object_id: integer object id (1 for single-object setup).
        checkpoint_path: GoTrack checkpoint .pt.
        config_path: GoTrack model config .yaml.
        device: cuda device string.
        mesh_scale: 1.0 (mesh is in meters) or 1000.0 (mm).
        unit_scale_mode: "auto" | "meter" | "mm" — passed through to GoTrack
            unit-scale resolver.
        first_frame_num_iters / num_iters: refinement iterations. Online
            tracking uses 1 (warm prior); first frame can use 5 for safety.
    """

    def __init__(
        self,
        mesh_path: str,
        anchor_bank_path: str,
        cameras: List[CameraIntrinsics],
        object_id: int = 1,
        object_name: str = "object",
        checkpoint_path: Optional[str] = None,
        config_path: Optional[str] = None,
        device: str = "cuda:0",
        mesh_scale: float = 1.0,
        unit_scale_mode: str = "auto",
        num_iters: int = 1,
        first_frame_num_iters: int = 5,
        mask_free: bool = True,
        skip_pnp: bool = True,
    ):
        import torch
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
        os.environ.setdefault("EGL_PLATFORM", "surfaceless")

        self.device_str = device
        self.device = torch.device(device)
        if device.startswith("cuda"):
            torch.cuda.set_device(self.device)

        self.mesh_path = str(Path(mesh_path).expanduser().resolve())
        self.anchor_bank_path = str(Path(anchor_bank_path).expanduser().resolve())
        self.object_id = int(object_id)
        self.object_name = object_name

        self.cameras = {c.serial: c for c in cameras}
        self.serials = [c.serial for c in cameras]

        # GoTrack helpers (lazy import — these touch torch + nvdiffrast).
        from run_singleview_multi_object import (
            load_gotrack_model,
            setup_renderer,
        )
        from utils.anchor_bank import (
            load_anchor_bank,
            prepare_anchor_bank_for_gotrack,
        )
        from utils.unit_scale import resolve_unit_scale_for_object

        cfg_path = config_path or str(_GOTRACK_ROOT / _DEFAULT_CONFIG)
        ckpt_path = checkpoint_path or str(_DEFAULT_CKPT)
        if not Path(ckpt_path).exists():
            raise FileNotFoundError(f"GoTrack checkpoint missing: {ckpt_path}")

        # Cap iterations to first_frame value here; we manually swap to num_iters
        # after the first frame in process_frame().
        self._num_iters = int(num_iters)
        self._first_frame_num_iters = int(first_frame_num_iters)
        self._frame_count = 0
        # trial_ts is set by the daemon from the robot PC's start command, so
        # crops go to .../{obj}/{trial_ts}/{fid}/. If None, fall back to a
        # flat .../{obj}/{fid}/ layout (older behaviour).
        self._trial_ts: Optional[str] = None

        # Async crop saver: a background thread drains (path, np.uint8 BGR) jobs
        # so PNG encode + disk write doesn't block the engine loop. queue has a
        # bound so a slow disk can't grow memory without limit; over-bound jobs
        # are dropped with a counter.
        import queue, threading
        self._crop_save_q: "queue.Queue[tuple[str, np.ndarray] | None]" = queue.Queue(maxsize=256)
        self._crop_save_drops = 0

        def _crop_saver():
            import cv2 as _cv2
            while True:
                item = self._crop_save_q.get()
                if item is None:
                    break
                path, arr = item
                try:
                    _cv2.imwrite(path, arr)
                except Exception as exc:
                    logger.warning(f"[diag-crops] write failed for {path}: {exc}")
        self._crop_save_thread = threading.Thread(target=_crop_saver, daemon=True)
        self._crop_save_thread.start()

        logger.info(f"[GoTrackEngine] loading model on {device}")
        self.model = load_gotrack_model(
            config_path=cfg_path,
            checkpoint_path=ckpt_path,
            device=self.device,
            num_iters=self._first_frame_num_iters,
        )

        # Resolve unit scale (handles meter↔mm conversion between mesh and GoTrack).
        unit_info = resolve_unit_scale_for_object(
            mesh_path=self.mesh_path,
            mesh_scale=float(mesh_scale),
            mode=str(unit_scale_mode),
        )
        self._unit_info = unit_info
        self._translation_scale = float(unit_info.translation_scale_to_gotrack)

        self.renderer = setup_renderer(
            model=self.model,
            obj_ids=[self.object_id],
            mesh_paths=[self.mesh_path],
            unit_scale_infos={self.object_id: unit_info},
            backend="nvdiffrast",
        )
        # setup_renderer leaves model.gotrack_translation_scale unset, but
        # _process_group_for_timestep_anchor reads it via getattr(..., 1.0)
        # to scale init_pose_camera. If left at 1.0 while we scale the camera
        # T_world_from_eye to GoTrack's mm-unit (below), pose stays in meters
        # → unit mismatch inside scene_observation → crop_camera scale differs
        # wildly per cam. Mirror what model_loader.configure_renderer does.
        self.model.gotrack_translation_scale = float(self._translation_scale)
        self.renderer.gotrack_translation_scale = float(self._translation_scale)

        # Anchor bank — canonical FPS anchors on mesh (mesh frame, mm or meters
        # depending on mesh). prepare_anchor_bank_for_gotrack normalises into
        # GoTrack's internal unit system.
        raw_bank = load_anchor_bank(self.anchor_bank_path)
        self.anchor_bank = prepare_anchor_bank_for_gotrack(
            raw_bank,
            translation_scale_to_gotrack=unit_info.translation_scale_to_gotrack,
        )

        # Two camera-model dicts (matches MV-GoTrack video pipeline at
        # archive/run_multiview_gotrack_anchor_online.py:1116-1128):
        #   - camera_models:        translation_scale=1.0 (unscaled, used for
        #                           crop selection / unscaled geometry)
        #   - gotrack_camera_models: translation_scale=unit_info.translation_scale_to_gotrack
        #                            (cam pos in same scale as anchor/mesh in
        #                            gotrack frame — required for template
        #                            renderer + flow sampling).
        # Reusing one dict for both slots makes the template render empty.
        self.camera_models: Dict[str, Any] = {
            s: _build_camera_model(c, translation_scale=1.0)
            for s, c in self.cameras.items()
        }
        self.gotrack_camera_models: Dict[str, Any] = {
            s: _build_camera_model(c, translation_scale=self._translation_scale)
            for s, c in self.cameras.items()
        }

        # argparse-like namespace consumed by GoTrack helpers.
        self.args = _build_args_namespace(
            mask_free=mask_free,
            skip_pnp=skip_pnp,
        )
        self.mask_free = mask_free

    def set_trial_ts(self, ts: Optional[str]) -> None:
        """Daemon calls this when a 'start' command arrives so crops land
        under {obj}/{trial_ts}/{fid}/ instead of {obj}/{fid}/."""
        self._trial_ts = ts

    def _set_iters(self, n: int) -> None:
        """GoTrackOpts is a NamedTuple (immutable). Try to replace; fall back
        to no-op if model rejects assignment (in which case `num_iters` stays at
        whatever was set at load time)."""
        try:
            self.model.opts = self.model.opts._replace(num_iterations_test=int(n))
        except (AttributeError, TypeError):
            pass

    def process_frame(
        self,
        prior_pose_world: np.ndarray,
        frames_bgr: Dict[str, np.ndarray],
        masks: Optional[Dict[str, np.ndarray]] = None,
        frame_index: int = 0,
        time_sec: float = 0.0,
        include_debug_images: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """Run GoTrack stage 1-4 on N cameras for one frame.

        Args:
            prior_pose_world: 4x4 SE(3), object-in-world pose to use as prior.
            frames_bgr: {serial: HxWx3 uint8 BGR}. Must contain entries for
                every camera passed to __init__; missing cameras are skipped.
            masks: optional {serial: HxW uint8/bool} foreground mask. Required
                if mask_free=False.
            frame_index: monotonic frame id for record-keeping (passed back).
            time_sec: optional timestamp; passed through.
            include_debug_images: forward to GoTrack (heavy, off by default).

        Returns:
            {
              serial: {
                "frame_index": int,
                "status": "ok" | "empty_mask" | "missing",
                "anchor_uv_orig": np.float32 (M, 2)  — anchor (u, v) in original image,
                "anchor_conf":    np.float32 (M,),
                "valid_mask":     np.bool_  (M,)  — anchor was projected + flow-sampled successfully,
                "selected_mask":  np.bool_  (M,)  — top-K by conf (max_active_anchors_per_view),
                "compute_sec":    float,
              },
              ...
            }
            Anchor count M == len(self.anchor_bank["positions_o"]).
        """
        import time as _time
        import torch
        from run_multiview_gotrack_anchor_online import (
            _build_anchor_observations_for_frame,
            _process_group_for_timestep_anchor,
            DeviceState,
        )

        # Adjust iters based on first-frame.
        if self._frame_count == 0:
            self._set_iters(self._first_frame_num_iters)
        else:
            self._set_iters(self._num_iters)

        prior_pose_world = np.asarray(prior_pose_world, dtype=np.float64).reshape(4, 4)

        # Build frame_batch in GoTrack's expected shape.
        frame_batch: Dict[str, Dict[str, Any]] = {}
        for s in self.serials:
            if s not in frames_bgr:
                continue
            mask_frame = None
            if masks is not None and s in masks:
                m = masks[s]
                if m.dtype == np.bool_:
                    m = (m.astype(np.uint8) * 255)
                mask_frame = m
            frame_batch[s] = {
                "frame_bgr": frames_bgr[s],
                "mask_frame": mask_frame,
                "frame_index": int(frame_index),
                "time_sec": float(time_sec),
            }

        if not frame_batch:
            return {}

        # extrinsics dict + camera_models dict in expected shapes.
        extrinsics_map = {s: self.cameras[s].extrinsic_cw for s in frame_batch}
        camera_models = self.camera_models

        device_state = DeviceState(
            device=self.device,
            device_name=str(self.device),
            model=self.model,
            camera_ids=list(frame_batch.keys()),
        )

        t0 = _time.perf_counter()
        # Keep the end-to-end online update in FP32.  GoTrack passes renderer
        # outputs and confidence maps through nvdiffrast and NumPy in this
        # call chain; BF16 autocast leaks into both boundaries and fails on
        # current capture-PC builds.  The demo's correctness takes priority
        # over the optional xFormers BF16 speed path.
        with torch.autocast(device_type="cuda", enabled=False):
            per_camera_records = _process_group_for_timestep_anchor(
                device_state=device_state,
                frame_batch=frame_batch,
                camera_models=self.camera_models,
                gotrack_camera_models=self.gotrack_camera_models,
                extrinsics_map=extrinsics_map,
                init_pose_world=prior_pose_world,
                init_pose_source="prior",
                args=self.args,
                include_debug_images=include_debug_images,
            )

        # _process_group_for_timestep_anchor stores debug payloads inside each
        # frame_record under "debug_data". Extract them for stage 4.
        per_camera_debug: Dict[str, Dict[str, Any]] = {}
        for cam, rec in per_camera_records.items():
            dbg = rec.pop("debug_data", None) or rec.get("debug", None)
            if dbg is not None:
                per_camera_debug[cam] = dbg

        # === DIAG: queue crop images for async save to
        # ~/shared_data/AutoDex/debug/gotrack_crops/{obj}/{frame:06d}/{serial}_*.png.
        # PNG encode + disk write run on a background thread so this stays
        # off the engine hot path.
        try:
            import os, cv2 as _cv2, queue as _q
            # Save to LOCAL disk during the run; gotrack_daemon rsyncs to NAS
            # on stop. NAS writes are slow enough to back the saver queue up.
            trial_seg = f"{self._trial_ts}/" if self._trial_ts else ""
            save_dir = (f"/tmp/gotrack_crops/{self.object_name}/"
                        f"{trial_seg}{int(frame_index):06d}")
            os.makedirs(save_dir, exist_ok=True)

            def _prep(arr):
                a = np.asarray(arr)
                if a.dtype != np.uint8:
                    a = np.clip(a * 255 if a.max() <= 1.0 else a, 0, 255).astype(np.uint8)
                if a.ndim == 3 and a.shape[2] == 3:
                    a = _cv2.cvtColor(a, _cv2.COLOR_RGB2BGR)
                return a

            # Save per-cam bbox files: one JSON per serial. Previously we
            # wrote a single bbox.json per fid_dir, but each PC's daemon
            # writes to the same NAS path on rsync — the last PC's bbox.json
            # overwrites everyone else's, leaving the offline video missing
            # bbox for all cams except the last-rsynced PC's.
            for s in self.serials:
                dbg = per_camera_debug.get(s)
                if dbg is None:
                    continue
                # Require all fields needed for bbox up-front so frame + bbox
                # stay paired. If engine status was non-OK (empty_mask/missing)
                # some fields are absent — skip this cam entirely so the user
                # doesn't see a frame without its bbox.
                ci = dbg.get("crop_intrinsic")
                Tw_crop = dbg.get("T_world_from_crop_cam")
                Tw_orig = dbg.get("T_world_from_orig_cam")
                if ci is None or Tw_crop is None or Tw_orig is None:
                    continue
                # crops
                for key, suffix in (("query_rgb_crop", "query"),
                                    ("template_rgb_crop", "template")):
                    arr = dbg.get(key)
                    if arr is None:
                        continue
                    try:
                        self._crop_save_q.put_nowait((f"{save_dir}/{s}_{suffix}.png", _prep(arr)))
                    except _q.Full:
                        self._crop_save_drops += 1
                # downscaled full frame (undistorted) as JPG — 1/4 scale
                frame = frames_bgr.get(s)
                if frame is not None:
                    try:
                        h, w = frame.shape[:2]
                        small = _cv2.resize(frame, (w // 4, h // 4),
                                            interpolation=_cv2.INTER_AREA)
                        try:
                            self._crop_save_q.put_nowait((f"{save_dir}/{s}_frame.jpg", small))
                        except _q.Full:
                            self._crop_save_drops += 1
                    except Exception:
                        pass
                # 4-corner bbox in original undistorted image coords. Same-position
                # rotation only: ray dir in crop_cam → rotate to world → rotate to
                # orig_cam → project. Robust to depth.
                K_crop = np.asarray(ci, dtype=np.float64)
                R_wc = np.asarray(Tw_crop, dtype=np.float64)[:3, :3]
                R_wo = np.asarray(Tw_orig, dtype=np.float64)[:3, :3]
                K_orig = np.asarray(self.cameras[s].K, dtype=np.float64)
                cw, ch = 280, 280  # crop size; matches model.opts.crop_size
                corners_crop = np.array([
                    [0, 0, 1], [cw, 0, 1], [cw, ch, 1], [0, ch, 1],
                ], dtype=np.float64)
                rays_crop = (np.linalg.inv(K_crop) @ corners_crop.T).T
                rays_world = (R_wc @ rays_crop.T).T
                rays_orig = (R_wo.T @ rays_world.T).T
                uv_orig = (K_orig @ rays_orig.T).T
                uv_orig = uv_orig[:, :2] / uv_orig[:, 2:3]
                try:
                    import json
                    with open(f"{save_dir}/{s}_bbox.json", "w") as fh:
                        json.dump(uv_orig.round(2).tolist(), fh)
                except Exception:
                    pass
        except Exception as exc:
            logger.warning(f"[diag-crops] enqueue failed: {exc}")

        # === DIAG: per-cam debug state right after batch refinement ===
        diag_records: Dict[str, str] = {}
        for s in self.serials:
            rec = per_camera_records.get(s)
            dbg = per_camera_debug.get(s)
            if rec is None:
                diag_records[s] = "rec=None"
                continue
            status = rec.get("status", "?")
            if dbg is None:
                diag_records[s] = f"status={status} dbg=None"
                continue
            fmap = dbg.get("flow_map")
            cmap = dbg.get("confidence_map")
            tw = dbg.get("T_world_from_crop_cam")
            ci = dbg.get("crop_intrinsic")
            cis = dbg.get("crop_image_size")
            f_stat = (
                f"flow shape={list(fmap.shape) if hasattr(fmap, 'shape') else None}"
            ) if hasattr(fmap, "min") else "flow=None"
            tw_str = "Tw=None"
            if tw is not None:
                t_arr = np.asarray(tw)
                tw_str = f"Tw=[t={t_arr[:3,3].round(3).tolist()}]"
            ci_str = "ci=None"
            if ci is not None:
                ci_arr = np.asarray(ci)
                ci_str = f"ci_fx={ci_arr[0,0]:.1f} cx={ci_arr[0,2]:.1f}"
            diag_records[s] = (f"status={status} {f_stat} {tw_str} {ci_str} cis={cis}")
        engine_sec = _time.perf_counter() - t0
        logger.info(f"[engine.diag] fid={int(frame_index)} engine_sec={engine_sec:.3f} per-cam debug after refine batch:")
        for s, txt in diag_records.items():
            logger.info(f"  {s}: {txt}")

        external_unit_scale_to_meter = float(self._unit_info.external_unit_scale_to_meter)
        per_view_anchor_data, _obs_by_anchor, _summary = _build_anchor_observations_for_frame(
            anchor_bank=self.anchor_bank,
            per_camera_records=per_camera_records,
            per_camera_debug=per_camera_debug,
            frame_batch=frame_batch,
            camera_models=camera_models,
            args=self.args,
            external_unit_scale_to_meter=external_unit_scale_to_meter,
        )

        # === DIAG: per-cam valid_mask after anchor obs build ===
        logger.info(f"[engine.diag] fid={int(frame_index)} per-cam anchor obs:")
        for s in self.serials:
            obs = per_view_anchor_data.get(s)
            if obs is None:
                logger.info(f"  {s}: obs=None")
                continue
            vm = obs.get("valid_mask")
            sm = obs.get("selected_mask")
            conf = obs.get("confidence")
            v_sum = int(np.asarray(vm).sum()) if vm is not None else -1
            s_sum = int(np.asarray(sm).sum()) if sm is not None else -1
            c_stat = ""
            if conf is not None:
                ca = np.asarray(conf)
                c_stat = f"conf[min={ca.min():.3f} max={ca.max():.3f} mean={ca.mean():.3f}]"
            logger.info(f"  {s}: valid={v_sum}/256 selected={s_sum} {c_stat}")
        compute_sec = _time.perf_counter() - t0
        self._frame_count += 1

        # Pack per-cam payload for the robot PC. `uv_curr` is in CROP-image
        # coordinates and must be back-projected with `crop_intrinsic` (also
        # cropped). `T_world_from_crop_cam` lets the robot PC build the world-
        # frame ray. `position_o` is the canonical anchor in mesh frame (used
        # by Kabsch fit).
        out: Dict[str, Dict[str, Any]] = {}
        for s, rec in per_camera_records.items():
            obs = per_view_anchor_data.get(s)
            base = {
                "frame_index": int(frame_index),
                "status": rec.get("status", "missing"),
                "compute_sec": compute_sec / max(len(per_camera_records), 1),
            }
            if obs is None:
                out[s] = base
                continue
            dbg = per_camera_debug.get(s, {})
            crop_intrinsic = dbg.get("crop_intrinsic")
            T_world_from_crop_cam = dbg.get("T_world_from_crop_cam")
            base.update({
                "uv_curr":               np.asarray(obs["uv_curr"], dtype=np.float32),
                "confidence":            np.asarray(obs["confidence"], dtype=np.float32),
                "valid_mask":            np.asarray(obs["valid_mask"], dtype=np.bool_),
                "selected_mask":         np.asarray(obs["selected_mask"], dtype=np.bool_),
                "anchor_ids":            np.asarray(obs["anchor_ids"], dtype=np.int64),
                "positions_o":           np.asarray(obs["positions_o"], dtype=np.float32),
                "crop_intrinsic":        (np.asarray(crop_intrinsic, dtype=np.float32)
                                          if crop_intrinsic is not None else None),
                "T_world_from_crop_cam": (np.asarray(T_world_from_crop_cam, dtype=np.float64)
                                          if T_world_from_crop_cam is not None else None),
            })
            out[s] = base
        return out
