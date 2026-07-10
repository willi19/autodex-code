#!/usr/bin/env python3
"""Single-shot perception debug runner.

Runs ONE init pipeline cycle and saves everything that's useful for figuring
out *why* the perception loss is high:

  - raw undistorted images per camera (`images/{serial}.png`)
  - per-camera SAM3/YOLOE masks (`masks/{serial}.png`)
  - per-camera FoundPose initial pose candidates (`poses/{serial}.npy`)
  - the cross-view IoU-selected initial pose (`pre_sil_pose.npy`)
  - the silhouette-refined pose if it converged (`refined_pose.npy`,
    saved even when sil_loss > threshold, with a note in `result.json`)
  - mesh wireframe overlays at the initial pose
    (`overlay_initial/{serial}.png`) and refined pose
    (`overlay_refined/{serial}.png`)
  - `result.json` with iou / loss / all the timing the orchestrator returns

Prerequisites:
    bash scripts/init_daemons.sh start

Usage:
    python src/experiment/reset/perception_debug.py --obj attached_container
    python src/experiment/reset/perception_debug.py --obj brown_ramen --hand inspire_left
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.utils.system import get_pc_ip, get_camera_list
from paradex.calibration.utils import save_current_camparam, save_current_C2R, load_c2r

from autodex.utils.path import project_dir, obj_path
from autodex.perception.init_orchestrator import InitOrchestrator

from src.execution.run_auto import (
    DEFAULT_PC_LIST, ASSETS_BASE, MESH_BASE, CAM_PARAM_ROOT, _load_calib,
)

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")


PROMPT = "object on the checkerboard"
SIL_ITERS = 100
SIL_LR = 0.002
INIT_TIMEOUT_S = 120.0
STREAM_FPS = 10
STREAM_WARMUP_S = 2.0


def _project_points(pts_world: np.ndarray, K: np.ndarray, T_cam_world: np.ndarray):
    """Project Nx3 world points to image pixels. Returns Nx2 (u, v) and Z mask."""
    pts_h = np.hstack([pts_world, np.ones((pts_world.shape[0], 1))])
    pts_cam = (T_cam_world @ pts_h.T).T[:, :3]
    z = pts_cam[:, 2]
    valid = z > 1e-6
    uvw = (K @ pts_cam.T).T
    uv = np.zeros((len(pts_world), 2), dtype=np.float32)
    uv[valid, 0] = uvw[valid, 0] / uvw[valid, 2]
    uv[valid, 1] = uvw[valid, 1] / uvw[valid, 2]
    return uv, valid


def _wireframe_overlay(img: np.ndarray, mesh: trimesh.Trimesh,
                       pose_world: np.ndarray, K: np.ndarray,
                       T_cam_world: np.ndarray,
                       color=(0, 255, 0), thickness=1) -> np.ndarray:
    """Draw mesh edges projected at `pose_world` onto `img`. Returns copy."""
    out = img.copy()
    H, W = img.shape[:2]
    # Transform mesh vertices to world frame.
    verts = np.asarray(mesh.vertices)
    verts_h = np.hstack([verts, np.ones((len(verts), 1))])
    verts_world = (pose_world @ verts_h.T).T[:, :3]
    uv, valid = _project_points(verts_world, K, T_cam_world)

    edges = mesh.edges_unique
    for e0, e1 in edges:
        if not (valid[e0] and valid[e1]):
            continue
        p0 = (int(round(uv[e0, 0])), int(round(uv[e0, 1])))
        p1 = (int(round(uv[e1, 0])), int(round(uv[e1, 1])))
        # Cheap clip — skip lines fully outside.
        if (p0[0] < -5000 or p0[0] > W + 5000 or p1[0] < -5000 or p1[0] > W + 5000):
            continue
        cv2.line(out, p0, p1, color, thickness, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--hand", type=str, default="inspire_left",
                        choices=["allegro", "inspire", "inspire_left"])
    parser.add_argument("--out", type=str, default=None,
                        help="Output debug dir (default: experiment/perception_debug/{obj}/{ts})")
    args = parser.parse_args()

    mesh_path = MESH_BASE / args.obj / "raw_mesh" / f"{args.obj}.obj"
    assets_root = ASSETS_BASE / args.obj
    if not mesh_path.exists():
        sys.exit(f"mesh not found: {mesh_path}")
    if not (assets_root / "object_repre/v1" / args.obj / "1/repre.pth").exists():
        sys.exit(f"repre.pth missing for {args.obj}")

    calib_dir = sorted(CAM_PARAM_ROOT.iterdir())[-1]
    print(f"calib: {calib_dir.name}")
    intrinsics_full, extrinsics_full, H, W = _load_calib(calib_dir)

    pc_ips = [get_pc_ip(p) for p in DEFAULT_PC_LIST]
    pc_serials = {p: get_camera_list(p) for p in DEFAULT_PC_LIST}
    active = {s for pc in DEFAULT_PC_LIST for s in pc_serials[pc]}
    intrinsics_full = {s: v for s, v in intrinsics_full.items() if s in active}
    extrinsics_full = {s: v for s, v in extrinsics_full.items() if s in active}
    print(f"  {len(intrinsics_full)} cams active ({H}x{W})")

    # Output dir.
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else (
        Path(project_dir) / "experiment" / "perception_debug" / args.obj / ts
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "images").mkdir(exist_ok=True)
    (out_dir / "masks").mkdir(exist_ok=True)
    (out_dir / "poses").mkdir(exist_ok=True)
    (out_dir / "overlay_initial").mkdir(exist_ok=True)
    (out_dir / "overlay_refined").mkdir(exist_ok=True)
    save_current_C2R(str(out_dir))
    save_current_camparam(str(out_dir))
    print(f"out: {out_dir}")

    # Stream + orchestrator.
    rcc = remote_camera_controller(f"perception_debug_{os.getpid()}", pc_list=DEFAULT_PC_LIST)
    rcc.start("stream", False, fps=STREAM_FPS)
    time.sleep(STREAM_WARMUP_S)

    orch = InitOrchestrator(
        pc_list=DEFAULT_PC_LIST, capture_ips=pc_ips,
        port_mask=5006, port_pose=5007, port_cmd=6893,
    )
    orch.init_object(
        obj_name=args.obj,
        mesh_path=str(mesh_path), assets_root=str(assets_root),
        intrinsics_full=intrinsics_full, extrinsics_full=extrinsics_full,
        image_hw=(H, W), mode="live", pc_serials=pc_serials,
    )

    # Hook the orchestrator's buffer drops so we grab mask + per-cam pose data
    # BEFORE they get cleared at the end of trigger_init.
    captured = {"masks": {}, "poses": {}}
    orig_mask_drop = orch.mask_buf.drop
    orig_pose_drop = orch.pose_buf.drop

    def _snap_mask_drop(req_id):
        captured["masks"] = dict(orch.mask_buf.get(req_id))
        return orig_mask_drop(req_id)

    def _snap_pose_drop(req_id):
        captured["poses"] = dict(orch.pose_buf.get(req_id))
        return orig_pose_drop(req_id)

    orch.mask_buf.drop = _snap_mask_drop
    orch.pose_buf.drop = _snap_pose_drop

    print(f"[perception] triggering init...")
    t0 = time.time()
    pose_world, timing = orch.trigger_init(
        prompt=PROMPT,
        save_capture_dir=str(out_dir),   # writes images/*.png
        sil_iters=SIL_ITERS, sil_lr=SIL_LR,
        timeout_s=INIT_TIMEOUT_S,
    )
    print(f"[perception] done in {time.time()-t0:.2f}s  sil_loss={timing.get('sil_loss')}")

    # Save per-cam masks.
    for s, m in captured["masks"].items():
        mask = m.get("mask")
        if mask is None:
            continue
        cv2.imwrite(str(out_dir / "masks" / f"{s}.png"),
                    (mask.astype(np.uint8) * 255))

    # Save per-cam initial pose candidates.
    per_cam = {}
    for s, p in captured["poses"].items():
        if p.get("ok") and "pose_world" in p:
            np.save(out_dir / "poses" / f"{s}.npy", p["pose_world"])
            per_cam[s] = {
                "ok": True,
                "quality": p.get("quality"),
                "inliers": p.get("inliers"),
            }
        else:
            per_cam[s] = {"ok": False}

    # Save pre_sil and refined poses.
    pre_sil_pose = None
    if "pre_sil_pose" in (timing or {}):
        pre_sil_pose = np.asarray(timing["pre_sil_pose"], dtype=np.float64)
        np.save(out_dir / "pre_sil_pose.npy", pre_sil_pose)

    refined_pose = None
    if pose_world is not None:
        refined_pose = np.asarray(pose_world)
        np.save(out_dir / "refined_pose.npy", refined_pose)

    # Render mesh wireframe overlays.
    print(f"[overlay] rendering mesh wireframe on each cam...")
    mesh = trimesh.load(str(mesh_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    for s in captured["masks"].keys():
        img_path = out_dir / "images" / f"{s}.png"
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        if s not in intrinsics_full or s not in extrinsics_full:
            continue
        K = np.asarray(intrinsics_full[s]["K_undist"], dtype=np.float64)
        T_cam_world = np.asarray(extrinsics_full[s], dtype=np.float64)
        if T_cam_world.shape == (3, 4):
            T_cam_world = np.vstack([T_cam_world, [0, 0, 0, 1]])

        if pre_sil_pose is not None:
            ov = _wireframe_overlay(img, mesh, pre_sil_pose, K, T_cam_world,
                                    color=(0, 165, 255), thickness=1)
            cv2.imwrite(str(out_dir / "overlay_initial" / f"{s}.png"), ov)
        if refined_pose is not None:
            ov = _wireframe_overlay(img, mesh, refined_pose, K, T_cam_world,
                                    color=(0, 255, 0), thickness=1)
            cv2.imwrite(str(out_dir / "overlay_refined" / f"{s}.png"), ov)

    # Save result.json.
    result = {
        "obj": args.obj, "hand": args.hand,
        "ts": ts,
        "pose_world_returned": refined_pose.tolist() if refined_pose is not None else None,
        "pre_sil_pose": pre_sil_pose.tolist() if pre_sil_pose is not None else None,
        "timing": timing,
        "per_cam": per_cam,
        "calib_dir": str(calib_dir),
        "mesh_path": str(mesh_path),
    }
    with open(out_dir / "result.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n[done] saved to {out_dir}")
    print(f"  images/   {len(list((out_dir/'images').glob('*.png')))} files")
    print(f"  masks/    {len(list((out_dir/'masks').glob('*.png')))} files")
    print(f"  poses/    {len(list((out_dir/'poses').glob('*.npy')))} files")
    print(f"  overlay_initial/  {len(list((out_dir/'overlay_initial').glob('*.png')))} files")
    print(f"  overlay_refined/  {len(list((out_dir/'overlay_refined').glob('*.png')))} files")
    print(f"  sil_loss={timing.get('sil_loss')}  best_iou={timing.get('best_iou')}")

    # Cleanup.
    try:
        rcc.stop()
        rcc.end()
    except Exception:
        pass
    try:
        orch.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
