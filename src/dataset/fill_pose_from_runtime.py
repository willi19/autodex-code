"""Fill missing pose_world.npy from the episode's RUNTIME pose, gated by sil loss.

Each executed episode already recognized an object pose at run time, saved as
``{RSS_SRC}/{obj}/{ts}/outputs/{obj}_pose/optimized_pose_world.txt`` (+ per-view
masks in the same ``outputs/{obj}_pose/masks/``). For every dataset trial that is
missing ``pose_world.npy``:

  1. seed the silhouette optimizer with the runtime pose + runtime masks,
  2. refine and measure sil loss,
  3. sil_loss <= SIL_LOSS_THRESHOLD  -> save the refined pose as pose_world.npy,
     sil_loss >  threshold          -> leave it (needs a full FoundPose recompute).

Runs render-only (no FoundPose), so it is cheap and coexists with a live GPU job.

    ~/miniconda3/envs/gotrack/bin/python -m src.dataset.fill_pose_from_runtime \
        --root <dataset_root> --src <rss_experiment_root> [--refine_scale 0.5]
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

from src.dataset.recompute_pose import (MESH_BASE, SIL_LOSS_THRESHOLD, load_cam_param,
                                        _mesh_path, _scale_K)

DEFAULT_ROOT = "/home/mingi/shared_data/autodex_dataset/selected_100"
DEFAULT_SRC = "/home/mingi/shared_data/RSS2026_Mingi/experiment/selected_100"
SIL_ITERS = 100


def _runtime_pose(src_trial, obj):
    p = os.path.join(src_trial, "outputs", f"{obj}_pose", "optimized_pose_world.txt")
    if not os.path.exists(p):
        return None
    return np.loadtxt(p).reshape(4, 4)


def _runtime_masks(src_trial, obj, H, W):
    d = os.path.join(src_trial, "outputs", f"{obj}_pose", "masks")
    out = {}
    if not os.path.isdir(d):
        return out
    for f in os.listdir(d):
        if not f.endswith(".png"):
            continue
        m = cv2.imread(os.path.join(d, f), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        out[f[:-4]] = m > 127
    return out


def fill_trial(trial_dir, src_trial, obj, sil, refine_scale):
    pose0 = _runtime_pose(src_trial, obj)
    if pose0 is None:
        return {"reject": True, "reason": "no_runtime_pose"}
    K_all, ext_all, (H, W) = load_cam_param(trial_dir)
    masks = _runtime_masks(src_trial, obj, H, W)
    masks = {s: m for s, m in masks.items() if s in K_all and s in ext_all and m.any()}
    if not masks:
        return {"reject": True, "reason": "no_runtime_masks"}

    intr = {s: K_all[s] for s in masks}
    Hr, Wr = H, W
    if refine_scale < 1.0:
        Hr, Wr = int(round(H * refine_scale)), int(round(W * refine_scale))
        intr = {s: _scale_K(intr[s], Wr / W, Hr / H) for s in masks}
        masks = {s: cv2.resize(m.astype(np.uint8), (Wr, Hr), interpolation=cv2.INTER_NEAREST) > 0
                 for s, m in masks.items()}

    views = [{"mask": (m.astype(np.uint8) * 255), "K": intr[s], "extrinsic": ext_all[s]}
             for s, m in masks.items()]
    refined, sil_loss = sil.optimize(initial_pose_world=pose0, views=views,
                                     iters=SIL_ITERS, antialias=True)
    return {"sil_loss": float(sil_loss), "reject": bool(sil_loss > SIL_LOSS_THRESHOLD),
            "n_masks": len(masks), "source": "runtime", "pose": np.asarray(refined, np.float64)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--refine_scale", type=float, default=0.5)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--all", action="store_true",
                    help="process ALL trials (init-first): if init sil <= thresh, use the "
                         "init pose (overwriting an existing recompute pose); if init sil is "
                         "large, leave the existing pose_world untouched (recompute stays).")
    args = ap.parse_args()

    from autodex.perception.silhouette import SilhouetteOptimizer

    # collect trials, grouped by object so the mesh loads once
    todo = []
    for obj in sorted(os.listdir(args.root)):
        od = os.path.join(args.root, obj)
        if not os.path.isdir(od):
            continue
        for ts in sorted(os.listdir(od)):
            d = os.path.join(od, ts)
            if not os.path.isdir(d):
                continue
            if args.all or not os.path.exists(os.path.join(d, "pose_world.npy")):
                todo.append((obj, ts))
    print(f"[fill] {len(todo)} trials ({'ALL, init-first' if args.all else 'missing-pose'})")

    sil = None
    cur = None
    stats = {"filled": 0, "reject": 0, "no_runtime": 0}
    for obj, ts in todo:
        d = os.path.join(args.root, obj, ts)
        src_trial = os.path.join(args.src, obj, ts)
        if obj != cur:
            mp = str(_mesh_path(obj))
            if sil is None:
                sil = SilhouetteOptimizer(mp)
            else:
                sil.reset_mesh(mp)
            cur = obj
        try:
            r = fill_trial(d, src_trial, obj, sil, args.refine_scale)
        except Exception as e:
            print(f"  ERR {obj}/{ts}: {type(e).__name__} {e}")
            continue
        if r.get("reason"):
            stats["no_runtime"] += 1
            print(f"  [no-runtime] {obj}/{ts} {r['reason']}")
            continue
        key = "outlier(sil>thr)" if r["reject"] else "ok"
        stats["reject" if r["reject"] else "filled"] += 1
        print(f"  [{key}] {obj}/{ts} sil_loss={r['sil_loss']:.5f} n={r['n_masks']}")
        if args.write:
            # ALWAYS save the sil-refined pose. High init-sil trials need refining the
            # MOST -- pose_world = refined pose for every trial. sil_loss (reject flag)
            # is just a quality marker, never a reason to leave pose_world empty.
            info = {k: v for k, v in r.items() if k != "pose"}
            json.dump(info, open(os.path.join(d, "recompute_pose.json"), "w"), indent=1)
            np.save(os.path.join(d, "pose_world.npy"), r["pose"])
    print("summary:", stats)


if __name__ == "__main__":
    main()
