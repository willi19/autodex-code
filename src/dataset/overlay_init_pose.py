"""Overlay the INITIAL (capture-time / legacy) object pose on every camera.

The initial pose is ``optimized_pose_world.txt`` (the pose actually used at
capture). For each pose-outlier trial, render that pose's mesh silhouette (green
fill) with the old SAM3 mask (red contour) on all cameras, so the initial pose's
misalignment is directly visible. Banner shows the initial-pose sil loss.

    green = initial(legacy) mesh   red = mask

Run in the `gotrack` env (nvdiffrast):
    EGL_PLATFORM=surfaceless ~/miniconda3/envs/gotrack/bin/python \
      -m src.dataset.overlay_init_pose
"""
from __future__ import annotations

import json
import os

import cv2
import numpy as np

from src.dataset.recompute_pose import load_cam_param, load_masks, MESH_BASE, SIL_LR
from src.dataset.overlay_pose_outliers import _mesh_sil, _combined_overlay
from src.validation.perception.foundpose_overlay_grid import _make_grid
from src.dataset.legacy_pose_sil_loss import legacy_pose_path

OUTLIER_ROOT = "/home/mingi/shared_data/autodex_dataset/selected_100_pose_outlier"
THRESHOLD = 0.003


def iter_trials(root, obj_filter=None):
    for obj in sorted(os.listdir(root)):
        if obj_filter and obj != obj_filter:
            continue
        od = os.path.join(root, obj)
        if not os.path.isdir(od):
            continue
        for ts in sorted(os.listdir(od)):
            if os.path.isdir(os.path.join(od, ts)):
                yield obj, ts


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=OUTLIER_ROOT)
    ap.add_argument("--obj")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from autodex.perception.silhouette import SilhouetteOptimizer

    work = list(iter_trials(args.root, args.obj))
    if args.limit:
        work = work[:args.limit]
    print(f"[init-overlay] {len(work)} trials")

    sil, cur_obj, done = None, None, 0
    for obj, ts in work:
        td = os.path.join(args.root, obj, ts)
        out_p = os.path.join(td, "overlay", "init_overlay.png")
        if os.path.exists(out_p) and not args.overwrite:
            continue
        lp = legacy_pose_path(obj, ts)
        if not os.path.exists(lp):
            print(f"  no legacy pose: {obj}/{ts}")
            continue
        if obj != cur_obj:
            mesh = str(MESH_BASE / obj / "raw_mesh" / f"{obj}.obj")
            if sil is None:
                sil = SilhouetteOptimizer(mesh)
            else:
                sil.reset_mesh(mesh)
            cur_obj = obj

        legacy = np.loadtxt(lp).astype(np.float64)
        K_all, ext_all, (H, W) = load_cam_param(td)
        masks = load_masks(td, H, W)
        ljson = os.path.join(td, "overlay", "legacy_loss.json")
        loss = json.load(open(ljson))["legacy_sil_loss"] if os.path.exists(ljson) else None

        ovs = {}
        for s in sorted(masks.keys()):
            if s not in K_all or s not in ext_all or not masks[s].any():
                continue
            msil = _mesh_sil(legacy, K_all[s], ext_all[s], H, W, sil.glctx, sil.mesh_tensors)
            comb = _combined_overlay(_rgb(td, s), masks[s], msil)
            ovs[s] = cv2.cvtColor(comb, cv2.COLOR_RGB2BGR)
        grid = _make_grid(ovs, cols=6)
        if grid is not None:
            banner = (f"init(legacy) sil_loss={loss:.5f}  (thr {THRESHOLD})   "
                      f"green=init_pose  red=mask") if loss is not None else "green=init_pose red=mask"
            for col, th in ((0, 0, 0), 12), ((0, 255, 255), 5):
                cv2.putText(grid, banner, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.2, col, th, cv2.LINE_AA)
            os.makedirs(os.path.dirname(out_p), exist_ok=True)
            cv2.imwrite(out_p, grid)
            done += 1
            print(f"  [{obj}/{ts}] loss={loss:.5f} -> {out_p}")
    print(f"[init-overlay] done {done}")


def _rgb(td, serial):
    """Undistorted RGB for a camera: init_capture/images first, else source images."""
    from src.dataset.recompute_pose import load_images
    if not hasattr(_rgb, "_cache") or _rgb._cache_td != td:
        _rgb._cache = load_images(td)
        _rgb._cache_td = td
    return _rgb._cache[serial]


if __name__ == "__main__":
    main()
