#!/usr/bin/env python3
"""
Silhouette Optimization Server

Runs silhouette-based pose refinement on a dedicated machine.
Communicates via socket using pickled numpy arrays.
"""

import argparse
import logging
import pickle
import socket
import struct
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import trimesh

_FP_ROOT = Path(__file__).parent / "thirdparty/FoundationPose"
if str(_FP_ROOT) not in sys.path:
    sys.path.insert(0, str(_FP_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='[Silhouette] [%(levelname)s] %(message)s'
)


def _resolve_mesh_path(mesh_arg: Optional[str], obj_name: Optional[str]) -> str:
    if mesh_arg:
        mesh_path = Path(mesh_arg)
        if not mesh_path.exists():
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")
        return str(mesh_path)
    if not obj_name:
        raise ValueError("Either --mesh or --obj_name must be provided.")
    mesh_path, _ = get_mesh_path(obj_name)
    mesh_path = Path(mesh_path)
    if not mesh_path.exists():
        raise FileNotFoundError(f"Mesh not found for object '{obj_name}': {mesh_path}")
    return str(mesh_path)


def _resolve_mesh_override(request: Dict[str, Any]) -> Optional[str]:
    mesh_arg = request.get("mesh_path")
    obj_name = request.get("obj_name")
    if not mesh_arg and not obj_name:
        return None
    return _resolve_mesh_path(mesh_arg, obj_name)


class SilhouetteOptimizer:
    def __init__(self, mesh_path: str, device: str = "cuda"):
        self.mesh_path = mesh_path
        self.device = device
        self.mesh = self._load_mesh(mesh_path)
        self.mesh_tensors = None
        self.glctx = None
        self._init_renderer()

    def _load_mesh(self, mesh_path: str) -> trimesh.Trimesh:
        loaded = trimesh.load(str(mesh_path), process=False)
        if isinstance(loaded, trimesh.Scene):
            meshes = [
                geom for geom in loaded.geometry.values()
                if isinstance(geom, trimesh.Trimesh)
            ]
            if not meshes:
                raise RuntimeError("Scene contains no valid Trimesh objects")
            return trimesh.util.concatenate(meshes)
        return loaded

    def _init_renderer(self):
        import nvdiffrast.torch as dr
        from Utils import make_mesh_tensors

        self.mesh_tensors = make_mesh_tensors(self.mesh, device=self.device)
        self.glctx = dr.RasterizeCudaContext()
        self._pos_homo = None
        self._glcam_t = None
        self._warmup_renderer()

    def reset_mesh(self, mesh_path: str):
        if mesh_path == self.mesh_path:
            return
        logging.info("Resetting mesh to %s", mesh_path)
        self.mesh_path = mesh_path
        self.mesh = self._load_mesh(mesh_path)
        self._init_renderer()

    def _warmup_renderer(self):
        try:
            if self._pos_homo is None:
                from Utils import to_homo_torch
                self._pos_homo = to_homo_torch(self.mesh_tensors["pos"])
            if self._glcam_t is None:
                from Utils import glcam_in_cvcam
                self._glcam_t = torch.tensor(
                    glcam_in_cvcam, device=self.device, dtype=torch.float32
                )
            dummy_pose = torch.eye(4, device=self.device, dtype=torch.float32).reshape(1, 4, 4)
            dummy_K = np.array(
                [[50.0, 0.0, 32.0], [0.0, 50.0, 32.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
            _ = self._render_silhouette(
                K=dummy_K,
                H=64,
                W=64,
                ob_in_cams=dummy_pose,
                glctx=self.glctx,
                mesh_tensors=self.mesh_tensors,
                antialias=False,
            )
            logging.info("Silhouette renderer warmup completed")
        except Exception as exc:
            logging.warning(f"Silhouette warmup failed: {exc}")

    def optimize(
        self,
        initial_pose_world: np.ndarray,
        views: List[Dict[str, Any]],
        iters: int = 200,
        lr: float = 1e-3,
        mask_blur_ksize: int = 1,
        mask_blur_sigma: float = 0.0,
        use_iou: bool = False,
        iou_weight: float = 1.0,
        antialias: bool = False,
        debug: bool = False,
        debug_dir: Optional[str] = None,
        debug_every: int = 1,
        debug_max_views: int = 4,
        frame_idx: int = 0,
        outlier_filter: bool = True,
        outlier_mult: float = 3.0,
        outlier_min_keep: int = 3,
    ) -> np.ndarray:
        import nvdiffrast.torch as dr
        from Utils import projection_matrix_from_intrinsics, to_homo_torch, glcam_in_cvcam

        logging.info(f"Running silhouette optimization with {len(views)} views")

        pose_world_init_t = torch.tensor(
            initial_pose_world, device=self.device, dtype=torch.float32
        )

        if self._pos_homo is None:
            self._pos_homo = to_homo_torch(self.mesh_tensors["pos"])
        if self._glcam_t is None:
            self._glcam_t = torch.tensor(glcam_in_cvcam, device=self.device, dtype=torch.float32)

        masks_raw: List[np.ndarray] = []
        cam_ids: List[str] = []
        mask_t_list: List[torch.Tensor] = []
        extrinsic_list: List[torch.Tensor] = []
        proj_list: List[torch.Tensor] = []
        H_ref: Optional[int] = None
        W_ref: Optional[int] = None
        for view in views:
            if "mask_path" in view:
                mask = cv2.imread(view["mask_path"], cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise FileNotFoundError(f"Mask not found: {view['mask_path']}")
                if debug:
                    logging.info("Silhouette mask path: %s", view["mask_path"])
            else:
                mask = view["mask"]
            mask_soft = self._make_soft_mask(mask, ksize=mask_blur_ksize, sigma=mask_blur_sigma)
            mask_t = torch.tensor(mask_soft, device=self.device, dtype=torch.float32)
            extrinsic_t = torch.tensor(view["extrinsic"], device=self.device, dtype=torch.float32)
            H, W = mask_t.shape
            if H_ref is None:
                H_ref, W_ref = H, W
            elif (H, W) != (H_ref, W_ref):
                raise ValueError(
                    f"All views must share (H, W); got {(H, W)} vs {(H_ref, W_ref)}"
                )
            proj = projection_matrix_from_intrinsics(view["K"], height=H, width=W, znear=0.001, zfar=100)
            proj_t = torch.as_tensor(proj.reshape(4, 4), device=self.device, dtype=torch.float32)

            masks_raw.append(mask)
            cam_ids.append(view.get("cam_id", ""))
            mask_t_list.append(mask_t)
            extrinsic_list.append(extrinsic_t)
            proj_list.append(proj_t)

        N = len(mask_t_list)
        if N == 0:
            raise ValueError("optimize() requires at least one view")

        mask_batch = torch.stack(mask_t_list, dim=0)            # (N, H, W)
        extrinsic_batch = torch.stack(extrinsic_list, dim=0)    # (N, 4, 4)
        proj_batch = torch.stack(proj_list, dim=0)              # (N, 4, 4)
        H, W = H_ref, W_ref

        # ── per-view outlier filter ──────────────────────────────────────
        # Render at the initial pose, compute per-view MSE, drop views whose
        # loss is >= outlier_mult × median (bad mask / mis-aligned candidate
        # / wrong segmentation). A few junk views can dominate the joint
        # gradient and stall the optimization (loss barely moves). Keep at
        # least ``outlier_min_keep`` views regardless of threshold.
        if outlier_filter and N > outlier_min_keep:
            with torch.no_grad():
                pose_cam_batch0 = extrinsic_batch @ pose_world_init_t
                alpha0 = self._render_silhouette_batched(
                    H=H, W=W,
                    ob_in_cams=pose_cam_batch0,
                    glctx=self.glctx,
                    mesh_tensors=self.mesh_tensors,
                    pos_homo=self._pos_homo,
                    proj_t=proj_batch,
                    glcam_t=self._glcam_t,
                    antialias=False,
                )                                                       # (N, H, W, 1)
                render0 = self._blur_mask_torch_batched(
                    alpha0[..., 0], mask_blur_ksize, mask_blur_sigma)
                per_view_mse = ((render0 - mask_batch) ** 2).mean(dim=(1, 2))   # (N,)
            losses = per_view_mse.cpu().numpy()
            median_loss = float(np.median(losses))
            thr = outlier_mult * median_loss
            keep_mask = losses <= thr
            n_keep = int(keep_mask.sum())
            if n_keep < outlier_min_keep:
                # Threshold too strict — keep the best ``outlier_min_keep`` views.
                order = np.argsort(losses)
                keep_mask = np.zeros_like(keep_mask, dtype=bool)
                keep_mask[order[:outlier_min_keep]] = True
                n_keep = outlier_min_keep
            dropped = [(cam_ids[i] or f"v{i}", float(losses[i]))
                       for i in range(N) if not keep_mask[i]]
            kept_losses = losses[keep_mask]
            logging.info(
                f"[sil filter] median_init_loss={median_loss:.5f}  "
                f"thr={thr:.5f} (x{outlier_mult})  kept {n_keep}/{N}  "
                f"kept_max={float(kept_losses.max()):.5f}")
            if dropped:
                logging.info(f"[sil filter] dropped: " + ", ".join(
                    f"{cid}({l:.4f})" for cid, l in dropped))
            keep_idx_t = torch.tensor(np.where(keep_mask)[0],
                                       device=self.device, dtype=torch.long)
            mask_batch       = mask_batch.index_select(0, keep_idx_t)
            extrinsic_batch  = extrinsic_batch.index_select(0, keep_idx_t)
            proj_batch       = proj_batch.index_select(0, keep_idx_t)
            cam_ids          = [cam_ids[i] for i in np.where(keep_mask)[0]]
            masks_raw        = [masks_raw[i] for i in np.where(keep_mask)[0]]
            N = n_keep

        r6d_init = self._matrix_to_rotation_6d(pose_world_init_t[:3, :3].unsqueeze(0))[0]
        t_init = pose_world_init_t[:3, 3]
        optim_r6d = torch.nn.Parameter(r6d_init.clone())
        optim_t = torch.nn.Parameter(t_init.clone())
        optimizer = torch.optim.Adam([optim_r6d, optim_t], lr=lr)

        for it in range(iters):
            optimizer.zero_grad()
            pose_world = self._build_pose_from_r6d_t(optim_r6d, optim_t)

            pose_cam_batch = extrinsic_batch @ pose_world                  # (N, 4, 4)
            alpha_t = self._render_silhouette_batched(
                H=H, W=W,
                ob_in_cams=pose_cam_batch,
                glctx=self.glctx,
                mesh_tensors=self.mesh_tensors,
                pos_homo=self._pos_homo,
                proj_t=proj_batch,
                glcam_t=self._glcam_t,
                antialias=antialias,
            )                                                              # (N, H, W, 1)
            render_masks = alpha_t[..., 0]                                 # (N, H, W)
            render_masks = self._blur_mask_torch_batched(
                render_masks, mask_blur_ksize, mask_blur_sigma
            )                                                              # (N, H, W)

            loss = F.mse_loss(render_masks, mask_batch, reduction="mean")
            if use_iou:
                loss = loss + iou_weight * self._silhouette_iou_loss_batched(
                    render_masks, mask_batch
                )

            if debug and (it % debug_every == 0):
                for view_idx in range(min(debug_max_views, N)):
                    self._save_debug_pair(
                        frame_idx=frame_idx,
                        iter_idx=it + 1,
                        cam_id=cam_ids[view_idx] or f"view{view_idx}",
                        render_mask=render_masks[view_idx],
                        target_mask=masks_raw[view_idx],
                        debug_dir=debug_dir,
                    )

            loss.backward()
            optimizer.step()

            if it == 0 or (it + 1) % 50 == 0 or it + 1 == iters:
                logging.info(f"  Iter {it + 1}/{iters} - loss: {loss.item():.6f}")

        pose_world_opt = self._build_pose_from_r6d_t(optim_r6d, optim_t).detach().cpu().numpy()
        return pose_world_opt, loss.item()

    @staticmethod
    def _make_soft_mask(mask: np.ndarray, ksize: int, sigma: float) -> np.ndarray:
        mask_f = mask.astype(np.float32)
        if mask_f.max() > 1.0:
            mask_f = mask_f / 255.0
        if ksize <= 1:
            return mask_f
        if ksize % 2 == 0:
            ksize += 1
        blurred = cv2.GaussianBlur(mask_f, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)
        return np.clip(blurred, 0.0, 1.0)

    @staticmethod
    def _silhouette_iou_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        intersection = (pred * target).sum()
        union = (pred + target).clamp(0, 1).sum()
        return 1 - (intersection / (union + 1e-9))

    @staticmethod
    def _silhouette_iou_loss_batched(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        intersection = (pred * target).sum(dim=(1, 2))
        union = (pred + target).clamp(0, 1).sum(dim=(1, 2))
        per_view = 1 - (intersection / (union + 1e-9))
        return per_view.mean()

    @staticmethod
    def _blur_mask_torch(mask: torch.Tensor, ksize: int, sigma: float) -> torch.Tensor:
        if (ksize is None or ksize <= 1) and (sigma is None or sigma <= 0):
            return mask
        if ksize is None or ksize <= 1:
            ksize = int(max(1, 2 * round(3 * sigma) + 1))
        if ksize <= 1:
            return mask
        if ksize % 2 == 0:
            ksize += 1
        mask_t = mask[None, None, :, :]
        blurred = F.avg_pool2d(mask_t, kernel_size=ksize, stride=1, padding=ksize // 2)
        return blurred[0, 0]

    @staticmethod
    def _blur_mask_torch_batched(masks: torch.Tensor, ksize: int, sigma: float) -> torch.Tensor:
        if (ksize is None or ksize <= 1) and (sigma is None or sigma <= 0):
            return masks
        if ksize is None or ksize <= 1:
            ksize = int(max(1, 2 * round(3 * sigma) + 1))
        if ksize <= 1:
            return masks
        if ksize % 2 == 0:
            ksize += 1
        m = masks.unsqueeze(1)
        blurred = F.avg_pool2d(m, kernel_size=ksize, stride=1, padding=ksize // 2)
        return blurred.squeeze(1)

    @staticmethod
    def _save_debug_pair(
        frame_idx: int,
        iter_idx: int,
        cam_id: str,
        render_mask: torch.Tensor,
        target_mask: np.ndarray,
        debug_dir: Optional[str],
    ) -> None:
        if not debug_dir:
            return
        debug_path = Path(debug_dir)
        debug_path.mkdir(parents=True, exist_ok=True)

        render_np = render_mask.detach().cpu().numpy()
        render_np = np.clip(render_np, 0.0, 1.0)
        render_u8 = (render_np * 255.0).astype(np.uint8)

        target_np = target_mask.astype(np.float32)
        if target_np.max() > 1.0:
            target_np = target_np / 255.0
        target_u8 = np.clip(target_np * 255.0, 0, 255).astype(np.uint8)

        render_h, render_w = render_u8.shape[:2]
        target_h, target_w = target_u8.shape[:2]
        logging.info(
            "Silhouette debug: %s render=%sx%s target=%sx%s",
            cam_id,
            render_w,
            render_h,
            target_w,
            target_h,
        )

        render_rgb = cv2.cvtColor(render_u8, cv2.COLOR_GRAY2BGR)
        target_rgb = cv2.cvtColor(target_u8, cv2.COLOR_GRAY2BGR)
        side_by_side = np.concatenate([render_rgb, target_rgb], axis=1)

        frame_dir = debug_path / f"frame_{frame_idx:06d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        out_path = frame_dir / f"iter_{iter_idx:04d}_{cam_id}.png"
        cv2.imwrite(str(out_path), side_by_side)

    def _render_silhouette(self, K, H, W, ob_in_cams, glctx, mesh_tensors, antialias: bool):
        import nvdiffrast.torch as dr
        from Utils import projection_matrix_from_intrinsics, to_homo_torch, glcam_in_cvcam

        pos = mesh_tensors["pos"]
        faces = mesh_tensors["faces"]
        glcam = torch.tensor(glcam_in_cvcam, device=ob_in_cams.device, dtype=ob_in_cams.dtype)
        ob_in_glcams = glcam[None] @ ob_in_cams
        proj = projection_matrix_from_intrinsics(K, height=H, width=W, znear=0.001, zfar=100)
        proj = torch.as_tensor(proj.reshape(-1, 4, 4), device=ob_in_cams.device, dtype=ob_in_cams.dtype)
        pos_homo = to_homo_torch(pos)
        pos_clip = (proj @ ob_in_glcams)[:, None] @ pos_homo[None, ..., None]
        pos_clip = pos_clip[..., 0]
        rast_out, _ = dr.rasterize(glctx, pos_clip, faces, resolution=np.asarray([H, W]))
        alpha = torch.clamp(rast_out[..., -1:], 0, 1)
        if antialias:
            alpha = dr.antialias(alpha, rast_out, pos_clip, faces)
        alpha = torch.flip(alpha, dims=[1])
        return alpha

    def _render_silhouette_fast(self, H, W, ob_in_cams, glctx, mesh_tensors, pos_homo, proj_t, glcam_t, antialias: bool):
        import nvdiffrast.torch as dr

        pos = mesh_tensors["pos"]
        faces = mesh_tensors["faces"]
        ob_in_glcams = glcam_t[None] @ ob_in_cams
        pos_clip = (proj_t @ ob_in_glcams)[:, None] @ pos_homo[None, ..., None]
        pos_clip = pos_clip[..., 0]
        rast_out, _ = dr.rasterize(glctx, pos_clip, faces, resolution=np.asarray([H, W]))
        alpha = torch.clamp(rast_out[..., -1:], 0, 1)
        if antialias:
            alpha = dr.antialias(alpha, rast_out, pos_clip, faces)
        alpha = torch.flip(alpha, dims=[1])
        return alpha

    def _render_silhouette_batched(self, H, W, ob_in_cams, glctx, mesh_tensors,
                                    pos_homo, proj_t, glcam_t, antialias: bool):
        """Batched silhouette render across N cameras in a single rasterize call.

        ob_in_cams: (N, 4, 4) world->cam pose per view (== extrinsic @ pose_world)
        proj_t:     (N, 4, 4) GL projection matrix per view
        glcam_t:    (4, 4)    glcam_in_cvcam (shared)
        pos_homo:   (V, 4)    mesh vertices in homogeneous coords (shared)
        Returns alpha: (N, H, W, 1)
        """
        import nvdiffrast.torch as dr

        faces = mesh_tensors["faces"]
        ob_in_glcams = glcam_t[None] @ ob_in_cams              # (N, 4, 4)
        proj_pose = proj_t @ ob_in_glcams                       # (N, 4, 4)
        pos_clip = proj_pose[:, None] @ pos_homo[None, ..., None]   # (N, V, 4, 1)
        pos_clip = pos_clip[..., 0]                             # (N, V, 4)
        rast_out, _ = dr.rasterize(glctx, pos_clip, faces, resolution=np.asarray([H, W]))
        alpha = torch.clamp(rast_out[..., -1:], 0, 1)
        if antialias:
            alpha = dr.antialias(alpha, rast_out, pos_clip, faces)
        alpha = torch.flip(alpha, dims=[1])
        return alpha

    @staticmethod
    def _rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
        a1, a2 = d6[..., :3], d6[..., 3:]
        b1 = F.normalize(a1, dim=-1)
        b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
        b2 = F.normalize(b2, dim=-1)
        b3 = torch.cross(b1, b2, dim=-1)
        return torch.stack((b1, b2, b3), dim=-2)

    @staticmethod
    def _matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
        batch_dim = matrix.size()[:-2]
        return matrix[..., :2, :].clone().reshape(batch_dim + (6,))

    def _build_pose_from_r6d_t(self, r6d: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        R = self._rotation_6d_to_matrix(r6d)
        T = torch.eye(4, device=r6d.device, dtype=r6d.dtype)
        T[:3, :3] = R
        T[:3, 3] = t
        return T


def run_socket_mode(args):
    optimizer = SilhouetteOptimizer(mesh_path=args.mesh, device=args.device)

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((args.bind_host, args.port))
    server.listen(1)

    logging.info(f"Silhouette server listening on port {args.port}")

    restart_requested = False
    while True:
        conn = None
        try:
            conn, addr = server.accept()
            logging.info(f"Connection from {addr}")

            length_bytes = conn.recv(8)
            if not length_bytes:
                conn.close()
                continue

            data_length = struct.unpack('>Q', length_bytes)[0]
            data = b''
            while len(data) < data_length:
                chunk = conn.recv(min(65536, data_length - len(data)))
                if not chunk:
                    break
                data += chunk

            _t_deser_start = time.perf_counter()
            request = pickle.loads(data)
            _t_deser = time.perf_counter() - _t_deser_start

            mesh_override = _resolve_mesh_override(request)
            if mesh_override:
                optimizer.reset_mesh(mesh_override)
            initial_pose_world = request["initial_pose_world"]
            views = request["views"]
            iters = request.get("iters", 200)
            lr = request.get("lr", 1e-3)
            mask_blur_ksize = request.get("mask_blur_ksize", 1)
            mask_blur_sigma = request.get("mask_blur_sigma", 0.0)
            use_iou = request.get("use_iou", False)
            iou_weight = request.get("iou_weight", 1.0)
            antialias = bool(request.get("antialias", False))
            debug = request.get("debug", False)
            debug_dir = request.get("debug_dir")
            debug_every = request.get("debug_every", 1)
            debug_max_views = request.get("debug_max_views", 4)
            frame_idx = request.get("frame_idx", 0)

            _t_infer_start = time.perf_counter()
            optimized_pose = optimizer.optimize(
                initial_pose_world=initial_pose_world,
                views=views,
                iters=iters,
                lr=lr,
                mask_blur_ksize=mask_blur_ksize,
                mask_blur_sigma=mask_blur_sigma,
                use_iou=use_iou,
                iou_weight=iou_weight,
                antialias=antialias,
                debug=debug,
                debug_dir=debug_dir,
                debug_every=debug_every,
                debug_max_views=debug_max_views,
                frame_idx=frame_idx,
            )
            _t_infer = time.perf_counter() - _t_infer_start

            _t_ser_start = time.perf_counter()
            response = pickle.dumps({
                "optimized_pose_world": optimized_pose,
                "server_timing": {
                    "deserialize": _t_deser,
                    "optimization": _t_infer,
                    "num_views": len(views),
                    "iters": iters,
                },
            })
            _t_ser = time.perf_counter() - _t_ser_start
            logging.info(
                "  Timing: deser=%.3fs optimize=%.3fs ser=%.3fs (%d views, %d iters)",
                _t_deser, _t_infer, _t_ser, len(views), iters,
            )
            conn.sendall(struct.pack('>Q', len(response)))
            conn.sendall(response)
            conn.close()

        except KeyboardInterrupt:
            logging.info("Shutting down server...")
            break
        except Exception as exc:
            logging.error(f"Error: {exc}")
            traceback.print_exc()
            try:
                response = pickle.dumps({"error": str(exc)})
                if conn is not None:
                    conn.sendall(struct.pack('>Q', len(response)))
                    conn.sendall(response)
                    conn.close()
            except Exception:
                pass
            if args.restart_on_error:
                restart_requested = True
                break

    server.close()
    if restart_requested:
        raise RuntimeError("Restarting silhouette server after error")


def _run_with_restart(run_fn, args, label):
    while True:
        try:
            run_fn(args)
            return
        except KeyboardInterrupt:
            logging.info("Shutting down %s...", label)
            return
        except Exception as exc:
            logging.error("%s crashed: %s", label, exc)
            if not args.restart_on_error:
                raise
            delay = max(0.0, float(args.restart_delay))
            logging.info("Restarting %s in %.1fs", label, delay)
            time.sleep(delay)


def main():
    parser = argparse.ArgumentParser(description="Silhouette Optimization Server")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    parser.add_argument("--mesh", type=str, default=None, help="Path to mesh file")
    parser.add_argument("--obj_name", type=str, default=None, help="Object name to resolve mesh")
    parser.add_argument("--port", type=int, default=5004, help="Port (socket mode)")
    parser.add_argument("--bind_host", type=str, default="127.0.0.1", help="Bind host (socket mode)")
    parser.add_argument(
        "--restart_on_error",
        dest="restart_on_error",
        action="store_true",
        default=True,
        help="Restart server on error",
    )
    parser.add_argument(
        "--no_restart_on_error",
        dest="restart_on_error",
        action="store_false",
        help="Disable restart on error",
    )
    parser.add_argument(
        "--restart_delay",
        type=float,
        default=2.0,
        help="Delay before restart in seconds",
    )

    args = parser.parse_args()
    args.mesh = _resolve_mesh_path(args.mesh, args.obj_name)
    _run_with_restart(run_socket_mode, args, "Silhouette server")


if __name__ == "__main__":
    main()
