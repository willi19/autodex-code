"""Overlay the SAVED init pose (pose_world.npy) onto the init_capture images.

Unlike overlay_pose_outliers (RE-RUNS FoundPose) and overlay_init_pose (legacy
outlier pose vs mask), this just renders the already-saved ``pose_world.npy`` mesh
silhouette over each init_capture camera image, so you can eyeball whether the
stored init pose is correct. Writes ``{trial}/outputs/init_pose_overlay.png``.

Projection follows overlay_pose_outliers exactly: pose_cam = ext_cw @ pose_world
(pose_world is in the camera-calibration world frame; no C2R needed).

Runs in the `gotrack` env (nvdiffrast for the render):
    EGL_PLATFORM=surfaceless ~/miniconda3/envs/gotrack/bin/python \
      -m src.dataset.overlay_saved_pose --trial <dir> [--mesh_frame op|paradex]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

from src.validation.perception.foundpose_overlay_grid import _make_grid
from src.dataset.recompute_pose import load_cam_param, MESH_BASE, OP_MESH_BASE
from src.dataset.overlay_pose_outliers import _mesh_sil


def _mesh_path(obj, frame):
    if frame == "op":
        return OP_MESH_BASE / obj / "processed_data" / "mesh" / "simplified.obj"
    return MESH_BASE / obj / "raw_mesh" / f"{obj}.obj"


def _green_overlay(image_rgb, mesh_sil, color=(0, 220, 0), alpha=0.45):
    out = image_rgb.copy().astype(np.float32)
    out[mesh_sil] = out[mesh_sil] * (1 - alpha) + np.array(color, np.float32) * alpha
    out = out.astype(np.uint8)
    mc, _ = cv2.findContours(mesh_sil.astype(np.uint8) * 255,
                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, mc, -1, (0, 255, 0), 3)
    return out


def _load_init_images(trial_dir):
    d = os.path.join(trial_dir, "init_capture", "images")
    out = {}
    for f in sorted(os.listdir(d)):
        if f.endswith(".png"):
            img = cv2.imread(os.path.join(d, f))
            if img is not None:
                out[os.path.splitext(f)[0]] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trial", required=True, help="trial dir with pose_world.npy")
    ap.add_argument("--mesh_frame", choices=["paradex", "op"], default="paradex")
    ap.add_argument("--out_name", default="init_pose_overlay.png")
    args = ap.parse_args()

    td = args.trial
    obj = Path(td).parent.name
    pose_world = np.load(os.path.join(td, "pose_world.npy")).astype(np.float64)
    K_all, ext_all, (H, W) = load_cam_param(td)
    images = _load_init_images(td)

    from autodex.perception.silhouette import SilhouetteOptimizer
    sil = SilhouetteOptimizer(str(_mesh_path(obj, args.mesh_frame)))

    ovs = {}
    for s in sorted(images):
        if s not in K_all or s not in ext_all:
            continue
        msil = _mesh_sil(pose_world, K_all[s], ext_all[s], H, W,
                         sil.glctx, sil.mesh_tensors)
        ovs[s] = cv2.cvtColor(_green_overlay(images[s], msil), cv2.COLOR_RGB2BGR)

    grid = _make_grid(ovs, cols=6)
    out_dir = os.path.join(td, "outputs")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.out_name)
    cv2.imwrite(out_path, grid)
    print(f"[done] {len(ovs)} cams  mesh={args.mesh_frame} -> {out_path}")


if __name__ == "__main__":
    main()
