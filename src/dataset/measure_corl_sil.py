"""corl 각 trial의 pose_world silhouette sil loss를 측정 (pose는 안 건드림).

selected_100 outlier 기준과 동일: pose AT the trial's pose_world의 sil loss(iters=1,
refine 없이 loss만). 마스크는 원본 experiment의 _pipeline_tmp/masks 사용. 결과를
recompute_pose.json에 {sil_loss, source:'gotrack', reject: sil>0.003}로 기록.

    ~/miniconda3/envs/gotrack/bin/python -m src.dataset.measure_corl_sil [--refine_scale 0.5]
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

from src.dataset.recompute_pose import load_cam_param, _mesh_path, _scale_K
from src.dataset.overlay_pose_outliers import _mesh_sil

DS = "/home/mingi/shared_data/autodex_dataset/corl_selected_100"
EXP = "/home/mingi/shared_data/AutoDex/experiment/selected_100/allegro"
THR = 0.003


def masks_from_src(obj, ts, H, W):
    d = os.path.join(EXP, obj, ts, "_pipeline_tmp", "masks")
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


def measure(trial_dir, obj, ts, sil, refine_scale):
    pw = np.load(os.path.join(trial_dir, "pose_world.npy"))
    K_all, ext_all, (H, W) = load_cam_param(trial_dir)
    masks = masks_from_src(obj, ts, H, W)
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
    # Pure forward: render mesh silhouette at pose_world per view, MSE vs mask.
    # (blur ksize=1 in the optimizer = no blur, so bool silhouette MSE matches.)
    mses = []
    for s, m in masks.items():
        msil = _mesh_sil(pw, intr[s], ext_all[s], Hr, Wr, sil.glctx, sil.mesh_tensors)
        mses.append(float(((msil.astype(np.float32) - m.astype(np.float32)) ** 2).mean()))
    return float(np.mean(mses)), len(masks)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refine_scale", type=float, default=0.5)
    args = ap.parse_args()
    from autodex.perception.silhouette import SilhouetteOptimizer

    todo = []
    for o in sorted(os.listdir(DS)):
        if not os.path.isdir(os.path.join(DS, o)):
            continue
        for t in sorted(os.listdir(os.path.join(DS, o))):
            d = os.path.join(DS, o, t)
            if os.path.isdir(d) and os.path.exists(os.path.join(d, "pose_world.npy")):
                todo.append((o, t))
    print(f"[measure-corl] {len(todo)} trials")
    sil = None
    cur = None
    stats = {"ok": 0, "reject": 0, "no_mask": 0}
    for o, t in todo:
        d = os.path.join(DS, o, t)
        if o != cur:
            mp = str(_mesh_path(o))
            sil = SilhouetteOptimizer(mp) if sil is None else (sil.reset_mesh(mp) or sil)
            cur = o
        try:
            r = measure(d, o, t, sil, args.refine_scale)
        except Exception as e:
            print(f"  ERR {o}/{t}: {type(e).__name__} {e}")
            continue
        if r is None:
            stats["no_mask"] += 1
            print(f"  [no-mask] {o}/{t}")
            continue
        loss, nm = r
        rej = loss > THR
        stats["reject" if rej else "ok"] += 1
        rp = os.path.join(d, "recompute_pose.json")
        info = json.load(open(rp)) if os.path.exists(rp) else {}
        info.update({"sil_loss": loss, "source": "gotrack", "reject": bool(rej), "n_masks": nm})
        json.dump(info, open(rp, "w"), indent=1)
        print(f"  [{'reject' if rej else 'ok'}] {o}/{t} sil_loss={loss:.5f} n={nm}")
    print("summary:", stats)


if __name__ == "__main__":
    main()
