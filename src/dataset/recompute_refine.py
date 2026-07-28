"""Re-derive op-frame pose for the op-misfit trials (transform didn't fix them).

For each failing trial (op_sil_unverified.json high=True) we seed from pose_world,
try a small set of object-local rotations (to escape a wrong-orientation track),
sil-optimize each against the op mesh (antialias), and keep the best. Then:

  best sil <= thr -> adopt it as pose_world (orig kept as pose_world_prev.npy).
  best sil >  thr -> move the trial to {dataset}_pose_outlier.

    EGL_PLATFORM=surfaceless ~/miniconda3/envs/gotrack/bin/python \
        -m src.dataset.recompute_refine [--thr 0.003] [--iters 120] [--write]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil

import cv2
import numpy as np
import trimesh

import src.dataset.recompute_pose as R
from src.dataset.recompute_pose import _mesh_path, load_cam_param, _scale_K
from src.dataset.check_mesh_fit import MASK_DIRS, load_masks, DS_BASE

IN = "/home/mingi/shared_data/autodex_dataset/op_sil_unverified.json"
OUT = "/home/mingi/shared_data/autodex_dataset/recompute_refine.json"


def rot_inits():
    """object-local rotations to try (as pose_world @ Rlocal)."""
    inits = [np.eye(4)]
    for ax in ([0, 0, 1], [1, 0, 0], [0, 1, 0]):
        for a in (np.pi / 2, np.pi, -np.pi / 2):
            inits.append(trimesh.transformations.rotation_matrix(a, ax))
    return inits


def views_of(trial_dir, mask_dir, refine_scale):
    K_all, ext_all, (H, W) = load_cam_param(trial_dir)
    masks = load_masks(mask_dir, H, W)
    masks = {s: m for s, m in masks.items() if s in K_all and s in ext_all and m.any()}
    if not masks:
        return None
    intr = {s: K_all[s] for s in masks}
    if refine_scale < 1.0:
        Hr, Wr = int(round(H * refine_scale)), int(round(W * refine_scale))
        intr = {s: _scale_K(intr[s], Wr / W, Hr / H) for s in masks}
        masks = {s: cv2.resize(m.astype(np.uint8), (Wr, Hr), interpolation=cv2.INTER_NEAREST) > 0
                 for s, m in masks.items()}
    return [{"mask": (m.astype(np.uint8) * 255), "K": intr[s], "extrinsic": ext_all[s]}
            for s, m in masks.items()]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thr", type=float, default=0.003)
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--refine_scale", type=float, default=0.5)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    from autodex.perception.silhouette import SilhouetteOptimizer
    R._USE_OP_MESH = True

    high = [t for t in json.load(open(IN))["trials"] if t["high"]]
    high.sort(key=lambda t: (t["obj"], t["dataset"], t["ts"]))
    print(f"[refine] {len(high)} trials, {len(set(t['obj'] for t in high))} objects")

    sil, cur = None, None
    rows = []
    inits = rot_inits()
    for t in high:
        ds, obj, ts = t["dataset"], t["obj"], t["ts"]
        if obj != cur:
            mp = str(_mesh_path(obj))
            sil = SilhouetteOptimizer(mp) if sil is None else (sil.reset_mesh(mp) or sil)
            cur = obj
        td = os.path.join(DS_BASE, ds, obj, ts)
        views = views_of(td, MASK_DIRS[ds](obj, ts), args.refine_scale)
        if views is None:
            rows.append({**t, "status": "no_mask"})
            continue
        pw = np.load(os.path.join(td, "pose_world.npy"))
        best_pose, best_loss = None, 1e9
        for Rl in inits:
            try:
                p, l = sil.optimize(initial_pose_world=pw @ Rl, views=views,
                                    iters=args.iters, antialias=True)
            except Exception as e:
                print(f"  ERR {obj}/{ts}: {type(e).__name__} {e}")
                continue
            if float(l) < best_loss:
                best_loss, best_pose = float(l), np.asarray(p, np.float64)
        ok = best_pose is not None and best_loss <= args.thr
        rows.append({**t, "sil_orig": round(t["sil"], 5),
                     "sil_refined": None if best_pose is None else round(best_loss, 5),
                     "status": "fixed" if ok else "outlier"})
        print(f"  {'FIX' if ok else 'OUT'} {ds}/{obj}/{ts}: {t['sil']:.5f} -> "
              f"{'None' if best_pose is None else f'{best_loss:.5f}'}")
        if args.write and best_pose is not None:
            if ok:
                if not os.path.exists(os.path.join(td, "pose_world_prev.npy")):
                    np.save(os.path.join(td, "pose_world_prev.npy"), pw)
                np.save(os.path.join(td, "pose_world.npy"), best_pose)
                rp = os.path.join(td, "recompute_pose.json")
                info = json.load(open(rp)) if os.path.exists(rp) else {}
                info.update({"sil_loss": best_loss, "reject": False,
                             "source": "op_refine"})
                json.dump(info, open(rp, "w"), indent=1)
            else:
                dst = os.path.join(f"{DS_BASE}/{ds}_pose_outlier", obj, ts)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(td, dst)

    fixed = [r for r in rows if r.get("status") == "fixed"]
    out = [r for r in rows if r.get("status") == "outlier"]
    json.dump({"thr": args.thr, "n": len(rows), "n_fixed": len(fixed),
               "n_outlier": len(out), "wrote": bool(args.write), "trials": rows},
              open(args.out, "w"), indent=1)
    import collections
    print(f"\n[refine] fixed {len(fixed)}/{len(rows)}  outlier {len(out)}  written={args.write}")
    for name in ("selected_100", "corl_selected_100", "selected_100_inspire"):
        s = [r for r in rows if r["dataset"] == name]
        if s:
            print(f"  {name:22s} fixed {sum(r['status']=='fixed' for r in s)}/{len(s)}")
    print("  outliers by object:")
    for k, c in collections.Counter(f"{r['dataset']}/{r['obj']}" for r in out).most_common():
        print(f"    {c:3d}  {k}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
