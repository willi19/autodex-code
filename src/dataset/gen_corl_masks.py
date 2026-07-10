"""Generate per-camera object masks for CORL trials (they ship without masks).

RSS trials reuse old SAM3 masks; CORL trials (``corl_selected_100/``) have only
``raw/images/{serial}.png`` and no masks, so the initial-pose silhouette loss
can't be computed. This runs the SAME segmentation the live init pipeline uses
(``Sam3ImageSegmentor`` + prompt "object on the checkerboard") on every CORL
image and writes ``{trial}/obj_mask/{serial}.png`` (uint8 0/255).

Run in the `sam3` env:
    ~/miniconda3/envs/sam3/bin/python -m src.dataset.gen_corl_masks
    ~/miniconda3/envs/sam3/bin/python -m src.dataset.gen_corl_masks --obj banana
"""
from __future__ import annotations

import argparse
import os

import cv2

CORL_ROOT = "/home/mingi/shared_data/autodex_dataset/corl_selected_100"
PROMPT = "object on the checkerboard"


def iter_trials(root, obj_filter=None):
    for obj in sorted(os.listdir(root)):
        if obj_filter and obj != obj_filter:
            continue
        od = os.path.join(root, obj)
        if not os.path.isdir(od):
            continue
        for ts in sorted(os.listdir(od)):
            td = os.path.join(od, ts)
            if os.path.isdir(td):
                yield obj, ts, td


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=CORL_ROOT)
    ap.add_argument("--obj", help="restrict to one object")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from autodex.perception.mask import Sam3ImageSegmentor
    seg = Sam3ImageSegmentor(gpu=0)

    work = list(iter_trials(args.root, args.obj))
    print(f"[corl-mask] {len(work)} trials  prompt={args.prompt!r}")

    n_img, n_none, n_trial = 0, 0, 0
    for obj, ts, td in work:
        img_dir = os.path.join(td, "raw", "images")
        if not os.path.isdir(img_dir):
            print(f"  no images: {obj}/{ts}")
            continue
        out_dir = os.path.join(td, "obj_mask")
        os.makedirs(out_dir, exist_ok=True)
        serials = sorted(f[:-4] for f in os.listdir(img_dir) if f.endswith(".png"))
        done = 0
        for s in serials:
            out_p = os.path.join(out_dir, f"{s}.png")
            if os.path.exists(out_p) and not args.overwrite:
                done += 1
                continue
            bgr = cv2.imread(os.path.join(img_dir, f"{s}.png"))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            mask = seg.segment(rgb, args.prompt)
            if mask is None:
                n_none += 1
                mask = (bgr[:, :, 0] * 0).astype("uint8")  # empty mask placeholder
            cv2.imwrite(out_p, mask)
            n_img += 1
            done += 1
        n_trial += 1
        print(f"  [{n_trial}/{len(work)}] {obj}/{ts}: {done}/{len(serials)} masks")

    print(f"[corl-mask] done. wrote {n_img} masks ({n_none} empty/none) across {n_trial} trials")


if __name__ == "__main__":
    main()
