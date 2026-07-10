"""Silhouette loss of the CAPTURE-TIME (legacy) pose for the pose outliers.

Each outlier trial was rejected on the *FoundPose recompute*'s sil_loss (>0.003,
no pose_world.npy). But the pose actually used at capture/execution is the legacy
``optimized_pose_world.txt`` (old SAM3+FoundationPose). This measures that legacy
pose's silhouette loss against the same masks, using the same loss the pipeline
uses. If the legacy loss is low, the capture-time pose is fine even though the
recompute flipped/failed -- a recovery candidate.

    legacy pose : SRC_ROOT/{obj}/{ts}/outputs/{obj}_pose/optimized_pose_world.txt
    masks       : reused old SAM3 masks (same as recompute)
    loss        : sil.optimize(legacy, views, iters=1)  -> loss AT the legacy pose
                  (antialias irrelevant to the loss value; the pipeline's own
                   outlier filter measures MSE the same antialias-free way)

Runs in the `gotrack` env (nvdiffrast):
    EGL_PLATFORM=surfaceless ~/miniconda3/envs/gotrack/bin/python \
      -m src.dataset.legacy_pose_sil_loss
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from src.dataset.recompute_pose import load_cam_param, load_masks, MESH_BASE, SIL_LR

SRC_ROOT = "/home/mingi/shared_data/RSS2026_Mingi/experiment/selected_100"
CLEAN_ROOT = "/home/mingi/shared_data/autodex_dataset/selected_100"
OUTLIER_ROOT = "/home/mingi/shared_data/autodex_dataset/selected_100_pose_outlier"
ROOTS = [CLEAN_ROOT, OUTLIER_ROOT]
OUT_JSON = "/home/mingi/shared_data/autodex_dataset/legacy_sil_loss.json"
THRESHOLD = 0.003


def legacy_pose_path(obj, ts):
    return os.path.join(SRC_ROOT, obj, ts, "outputs", f"{obj}_pose",
                        "optimized_pose_world.txt")


def iter_outliers(roots):
    for root in roots:
        if not os.path.isdir(root):
            continue
        for obj in sorted(os.listdir(root)):
            od = os.path.join(root, obj)
            if not os.path.isdir(od):
                continue
            for ts in sorted(os.listdir(od)):
                td = os.path.join(od, ts)
                if os.path.isdir(td):
                    yield obj, ts, td


def main():
    from autodex.perception.silhouette import SilhouetteOptimizer

    work = sorted(iter_outliers(ROOTS), key=lambda x: x[0])
    print(f"[legacy] {len(work)} RSS trials across {len(ROOTS)} roots")

    sil, cur_obj = None, None
    rows, missing = [], 0
    for obj, ts, td in work:
        in_outlier = td.startswith(OUTLIER_ROOT)
        lp = legacy_pose_path(obj, ts)
        if not os.path.exists(lp):
            missing += 1
            continue
        if obj != cur_obj:
            mesh = MESH_BASE / obj / "raw_mesh" / f"{obj}.obj"
            if sil is None:
                sil = SilhouetteOptimizer(str(mesh))
            else:
                sil.reset_mesh(str(mesh))
            cur_obj = obj

        legacy = np.loadtxt(lp).astype(np.float64)
        K_all, ext_all, (H, W) = load_cam_param(td)
        masks = load_masks(td, H, W)
        views = [{"mask": (m.astype(np.uint8) * 255), "K": K_all[s], "extrinsic": ext_all[s]}
                 for s, m in masks.items() if s in K_all and s in ext_all and m.any()]
        if not views:
            missing += 1
            continue
        _, loss = sil.optimize(initial_pose_world=legacy, views=views,
                               iters=1, lr=SIL_LR, antialias=False)
        loss = float(loss)
        rp = os.path.join(td, "recompute_pose.json")
        recompute_loss = json.load(open(rp)).get("sil_loss") if os.path.exists(rp) else None
        rows.append({"obj": obj, "ts": ts, "legacy_sil_loss": loss,
                     "recompute_sil_loss": recompute_loss, "in_outlier": in_outlier})
        odir = os.path.join(td, "overlay")
        os.makedirs(odir, exist_ok=True)
        json.dump({"legacy_sil_loss": loss, "recompute_sil_loss": recompute_loss,
                   "threshold": THRESHOLD, "legacy_ok": loss <= THRESHOLD},
                  open(os.path.join(odir, "legacy_loss.json"), "w"), indent=1)

    ll = np.array([r["legacy_sil_loss"] for r in rows])
    bad = ll > THRESHOLD                                  # new-criterion outliers
    # cross-tab vs the current (recompute-based) split.
    cur_out = np.array([r["in_outlier"] for r in rows])
    print(f"\n[legacy] computed {len(rows)}  (missing/no-legacy {missing})")
    print(f"legacy_sil_loss  p50={np.percentile(ll,50):.5f} p90={np.percentile(ll,90):.5f} "
          f"min={ll.min():.5f} max={ll.max():.5f}")
    for thr in (0.003, 0.004, 0.005, 0.006):
        print(f"  legacy > {thr:.3f}: {int((ll>thr).sum()):4d}  (new outliers)")
    print(f"\nNew criterion (legacy > {THRESHOLD}): {int(bad.sum())} outliers / {len(rows)}")
    print(f"  now-clean but new-outlier (in clean, legacy>thr): {int((bad & ~cur_out).sum())}")
    print(f"  now-outlier but new-clean (in outlier, legacy<=thr): {int((~bad & cur_out).sum())}")
    out = OUT_JSON
    json.dump({"n": len(rows), "threshold": THRESHOLD,
               "n_new_outlier": int(bad.sum()), "trials": rows},
              open(out, "w"), indent=1)
    print(f"-> {out}")


if __name__ == "__main__":
    main()
