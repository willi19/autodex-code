"""Render the object mesh at each uncovered trial's saved pose_world over all
camera images, as one grid PNG per trial, so a wrong 6D pose is visible by eye.

Uses the EXISTING pose_world.npy (no re-estimation) + the object_processing mesh
(the canonical frame tabletop/coverage use). Green fill = mesh at pose_world;
red contour = object mask (if present). Runs render-only (no FoundPose) so it is
memory-cheap and coexists with a live BODex job.

    ~/miniconda3/envs/gotrack/bin/python -m src.dataset.overlay_uncovered --out <dir>
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
from src.dataset.recompute_pose import load_cam_param, load_images, load_masks  # noqa: E402
from src.dataset.overlay_pose_outliers import _mesh_sil, _combined_overlay  # noqa: E402
from src.validation.perception.foundpose_overlay_grid import _make_grid  # noqa: E402

OP = "/home/mingi/shared_data/object_processing"

# (dataset, obj, ts) of the 11 genuine uncovered trials.
UNCOVERED = [
    ("selected_100", "plant_mister", "20260125_151303"),
    ("selected_100", "washing_brush", "20260127_090911"),
    ("selected_100", "washing_brush", "20260127_091027"),
    ("selected_100", "washing_brush", "20260127_091142"),
    ("selected_100", "washing_brush", "20260127_092147"),
    ("selected_100", "washing_brush2", "20260124_054725"),
    ("corl_selected_100", "beige_brush", "20260405_073010"),
    ("corl_selected_100", "blue_alarm", "20260328_012741"),
    ("corl_selected_100", "brown_ramen", "20260327_002626"),
    ("corl_selected_100", "plant_mister", "20260326_162611"),
    ("corl_selected_100", "plant_mister", "20260326_165254"),
]
DS_ROOT = "/home/mingi/shared_data/autodex_dataset"


def render_trial(trial_dir, mesh_path, sil_cache):
    from autodex.perception.silhouette import SilhouetteOptimizer
    if mesh_path not in sil_cache:
        if not sil_cache:
            sil_cache["_sil"] = SilhouetteOptimizer(mesh_path)
        else:
            sil_cache["_sil"].reset_mesh(mesh_path)
        sil_cache[mesh_path] = True
    sil = sil_cache["_sil"]

    pose_world = np.load(os.path.join(trial_dir, "pose_world.npy"))
    K_all, ext_all, (H, W) = load_cam_param(trial_dir)
    images = load_images(trial_dir)
    masks = load_masks(trial_dir, H, W)

    tiles = {}
    for s in sorted(images):
        if s not in K_all or s not in ext_all:
            continue
        msil = _mesh_sil(pose_world, K_all[s], ext_all[s], H, W, sil.glctx, sil.mesh_tensors)
        mb = masks.get(s)
        tiles[s] = _combined_overlay(images[s], mb if mb is not None else np.zeros((H, W), bool), msil)
    if not tiles:
        return None
    return _make_grid(tiles)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(REPO_ROOT / "outputs/uncovered_overlay"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    sil_cache = {}
    for ds, obj, ts in UNCOVERED:
        trial_dir = os.path.join(DS_ROOT, ds, obj, ts)
        mesh = os.path.join(OP, obj, "processed_data", "mesh", "simplified.obj")
        if not os.path.isdir(trial_dir) or not os.path.exists(mesh):
            print(f"  skip {obj}/{ts} (missing)")
            continue
        grid = render_trial(trial_dir, mesh, sil_cache)
        if grid is None:
            print(f"  no views {obj}/{ts}")
            continue
        out_p = os.path.join(args.out, f"{obj}__{ds[:4]}__{ts}.png")
        cv2.imwrite(out_p, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
        print(f"  wrote {out_p}")


if __name__ == "__main__":
    main()
