"""Authoritative op-mesh sil loss for the op-UNVERIFIED trials, measured the exact
way the gate did: object_processing mesh + antialiased silhouette loss at the
pose (sil.optimize iters=1, antialias=True). One fixed definition -> one count.

op-unverified set:
  selected_100         : recompute_pose.json source == 'runtime'  (paradex-gated)
  corl_selected_100    : all  (gotrack = paradex)
  selected_100_inspire : all  (gotrack = paradex)

    EGL_PLATFORM=surfaceless ~/miniconda3/envs/gotrack/bin/python \
        -m src.dataset.measure_op_sil [--refine_scale 0.5] [--thr 0.003]
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

import src.dataset.recompute_pose as R
from src.dataset.recompute_pose import load_cam_param, _mesh_path, _scale_K
from src.dataset.check_mesh_fit import MASK_DIRS, load_masks, DS_BASE

THR_OUT = "/home/mingi/shared_data/autodex_dataset/op_sil_unverified.json"


def op_unverified():
    """yield (dataset, obj, ts)."""
    # selected_100: source == runtime
    d = os.path.join(DS_BASE, "selected_100")
    for o in sorted(os.listdir(d)):
        od = os.path.join(d, o)
        if not os.path.isdir(od):
            continue
        for t in sorted(os.listdir(od)):
            rp = os.path.join(od, t, "recompute_pose.json")
            if os.path.exists(rp) and (json.load(open(rp)).get("source") == "runtime"):
                yield "selected_100", o, t
    # corl + inspire: all with pose_world
    for name in ("corl_selected_100", "selected_100_inspire"):
        d = os.path.join(DS_BASE, name)
        for o in sorted(os.listdir(d)):
            od = os.path.join(d, o)
            if not os.path.isdir(od):
                continue
            for t in sorted(os.listdir(od)):
                if os.path.exists(os.path.join(od, t, "pose_world.npy")):
                    yield name, o, t


def aa_loss(trial_dir, mask_dir, sil, refine_scale):
    pw = np.load(os.path.join(trial_dir, "pose_world.npy"))
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
    views = [{"mask": (m.astype(np.uint8) * 255), "K": intr[s], "extrinsic": ext_all[s]}
             for s, m in masks.items()]
    _, loss = sil.optimize(initial_pose_world=pw, views=views, iters=1, antialias=True)
    return float(loss)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refine_scale", type=float, default=0.5)
    ap.add_argument("--thr", type=float, default=0.003)
    ap.add_argument("--out", default=THR_OUT)
    args = ap.parse_args()

    from autodex.perception.silhouette import SilhouetteOptimizer
    R._USE_OP_MESH = True

    # sanity: a known op-gated trial (source=None) should reproduce its stored value.
    san_obj, san_ts = "black_holder_with_handle", "20260206_172428"
    std = os.path.join(DS_BASE, "selected_100", san_obj, san_ts)
    sil = SilhouetteOptimizer(str(_mesh_path(san_obj)))
    try:
        v = aa_loss(std, MASK_DIRS["selected_100"](san_obj, san_ts), sil, args.refine_scale)
        stored = json.load(open(os.path.join(std, "recompute_pose.json"))).get("sil_loss")
        print(f"[sanity] {san_obj}/{san_ts}: op+aa={v:.5f}  stored(op)={stored:.5f}")
    except Exception as e:
        print(f"[sanity] failed: {e}")

    work = list(op_unverified())
    print(f"[op-sil] {len(work)} op-unverified trials")
    cur = san_obj
    rows = []
    for i, (name, o, t) in enumerate(work):
        if o != cur:
            sil.reset_mesh(str(_mesh_path(o)))
            cur = o
        td = os.path.join(DS_BASE, name, o, t)
        try:
            v = aa_loss(td, MASK_DIRS[name](o, t), sil, args.refine_scale)
        except Exception as e:
            print(f"  ERR {name}/{o}/{t}: {type(e).__name__} {e}")
            continue
        if v is None:
            continue
        rows.append({"dataset": name, "obj": o, "ts": t, "sil": v, "high": bool(v > args.thr)})
        if (i + 1) % 100 == 0:
            print(f"  ..{i+1}/{len(work)}")

    high = [r for r in rows if r["high"]]
    json.dump({"thr": args.thr, "n": len(rows), "n_high": len(high), "trials": rows},
              open(args.out, "w"), indent=1)
    import collections
    print(f"\n[op-sil] measured {len(rows)}; sil>{args.thr}: {len(high)}")
    for name in ("selected_100", "corl_selected_100", "selected_100_inspire"):
        s = [r for r in rows if r["dataset"] == name]
        hi = sum(r["high"] for r in s)
        print(f"  {name:22s} {len(s):4d} trials  >{args.thr}: {hi}")
    print("  high by object:")
    for k, c in collections.Counter(f"{r['dataset']}/{r['obj']}" for r in high).most_common():
        print(f"    {c:3d}  {k}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
