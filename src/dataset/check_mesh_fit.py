"""Per-object mesh-fit check: does the object_processing mesh match the data?

The captured objects had several mesh versions. For each object we render the
canonical object_processing mesh at the trial's frame-0 pose (pose_world) and
compare its silhouette to the frame-0 SAM3/gotrack masks (MSE, the same loss the
outlier filter uses). A LOW loss means the op mesh fits the observed object; a
HIGH loss (even though the pose was accepted) means the mesh geometry is the
wrong version for that object.

Mask sources differ per dataset:
  selected_100         : {RSS}/{obj}/{ts}/outputs/{obj}_pose/masks   (old SAM3)
  corl_selected_100    : {EXP}/allegro/{obj}/{ts}/_pipeline_tmp/masks (gotrack)
  selected_100_inspire : {EXP}/inspire/{obj}/{ts}/_pipeline_tmp/masks (gotrack)

    EGL_PLATFORM=surfaceless ~/miniconda3/envs/gotrack/bin/python \
        -m src.dataset.check_mesh_fit [--dataset all] [--max_trials 5] [--refine_scale 0.5]
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

import src.dataset.recompute_pose as R
from src.dataset.recompute_pose import load_cam_param, _mesh_path, _scale_K
from src.dataset.overlay_pose_outliers import _mesh_sil

DS_BASE = "/home/mingi/shared_data/autodex_dataset"
RSS = "/home/mingi/shared_data/RSS2026_Mingi/experiment/selected_100"
EXP = "/home/mingi/shared_data/AutoDex/experiment/selected_100"

# (dataset name) -> function(obj, ts) -> mask dir
MASK_DIRS = {
    "selected_100": lambda o, t: os.path.join(RSS, o, t, "outputs", f"{o}_pose", "masks"),
    "corl_selected_100": lambda o, t: os.path.join(EXP, "allegro", o, t, "_pipeline_tmp", "masks"),
    "selected_100_inspire": lambda o, t: os.path.join(EXP, "inspire", o, t, "_pipeline_tmp", "masks"),
}


def load_masks(mask_dir, H, W):
    out = {}
    if not os.path.isdir(mask_dir):
        return out
    for f in os.listdir(mask_dir):
        if not f.endswith(".png"):
            continue
        m = cv2.imread(os.path.join(mask_dir, f), cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        out[f[:-4]] = m > 127
    return out


def trial_loss(trial_dir, mask_dir, sil, refine_scale):
    pw = np.load(os.path.join(trial_dir, "pose_world.npy"))
    K_all, ext_all, (H, W) = load_cam_param(trial_dir)
    masks = load_masks(mask_dir, H, W)
    masks = {s: m for s, m in masks.items() if s in K_all and s in ext_all and m.any()}
    if not masks:
        return None
    intr = {s: K_all[s] for s in masks}
    Hr, Wr = H, W
    if refine_scale < 1.0:
        Hr, Wr = int(round(H * refine_scale)), int(round(W * refine_scale))
        intr = {s: _scale_K(intr[s], Wr / W, Hr / H) for s in masks}
        masks = {s: cv2.resize(m.astype(np.uint8), (Wr, Hr), interpolation=cv2.INTER_NEAREST) > 0
                 for s, m in masks.items()}
    mses = []
    for s, m in masks.items():
        msil = _mesh_sil(pw, intr[s], ext_all[s], Hr, Wr, sil.glctx, sil.mesh_tensors)
        mses.append(float(((msil.astype(np.float32) - m.astype(np.float32)) ** 2).mean()))
    return float(np.mean(mses))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="all", choices=["all", *MASK_DIRS])
    ap.add_argument("--max_trials", type=int, default=5, help="trials sampled per object")
    ap.add_argument("--thr", type=float, default=0.003, help="sil above this = re-run candidate")
    ap.add_argument("--refine_scale", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.join(DS_BASE, "mesh_fit.json"))
    args = ap.parse_args()

    from autodex.perception.silhouette import SilhouetteOptimizer
    R._USE_OP_MESH = True  # render the object_processing mesh

    datasets = list(MASK_DIRS) if args.dataset == "all" else [args.dataset]
    sil, cur = None, None
    rows = []          # per-object summary
    trials = []        # per-trial sil (the re-run candidates)
    for name in datasets:
        root = os.path.join(DS_BASE, name)
        if not os.path.isdir(root):
            continue
        for obj in sorted(os.listdir(root)):
            od = os.path.join(root, obj)
            if not os.path.isdir(od):
                continue
            ts_list = sorted(t for t in os.listdir(od)
                             if os.path.isdir(os.path.join(od, t))
                             and os.path.exists(os.path.join(od, t, "pose_world.npy")))[:args.max_trials]
            if not ts_list:
                continue
            mp = str(_mesh_path(obj))
            if not os.path.exists(mp):
                rows.append({"dataset": name, "obj": obj, "median_loss": None, "n": 0,
                             "note": "no_op_mesh"})
                print(f"  [no-mesh] {name}/{obj}")
                continue
            sil = SilhouetteOptimizer(mp) if sil is None else (sil.reset_mesh(mp) or sil)
            cur = obj
            losses = []
            for t in ts_list:
                try:
                    v = trial_loss(os.path.join(od, t), MASK_DIRS[name](obj, t), sil, args.refine_scale)
                except Exception as e:
                    print(f"  ERR {name}/{obj}/{t}: {type(e).__name__} {e}")
                    continue
                if v is None:
                    continue
                losses.append(v)
                trials.append({"dataset": name, "obj": obj, "ts": t, "sil": v,
                               "high": bool(v > args.thr)})
            if not losses:
                rows.append({"dataset": name, "obj": obj, "median_loss": None, "n": 0,
                             "note": "no_mask"})
                print(f"  [no-mask] {name}/{obj}")
                continue
            med = float(np.median(losses))
            nhi = sum(v > args.thr for v in losses)
            rows.append({"dataset": name, "obj": obj, "median_loss": med, "n": len(losses),
                         "max_loss": float(np.max(losses)), "n_high": nhi})
            flag = f"  <-- {nhi} high" if nhi else ""
            print(f"  {name:20s} {obj:34s} med={med:.5f} (n={len(losses)}){flag}")

    high = sorted([t for t in trials if t["high"]], key=lambda t: -t["sil"])
    json.dump({"thr": args.thr, "n_trials": len(trials), "n_high": len(high),
               "objects": rows, "trials": trials}, open(args.out, "w"), indent=1)
    print(f"\n[mesh-fit] {len(trials)} trials measured; {len(high)} high (sil>{args.thr}) = re-run set")
    import collections
    hc = collections.Counter(f"{t['dataset']}/{t['obj']}" for t in high)
    for k, c in hc.most_common():
        print(f"  {c:3d}  {k}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
