"""Generate obj_mask/{serial}.png for an explicit list of trial dirs.

Same SAM3 segmentation as gen_corl_masks (prompt "object on the checkerboard"),
but targets specific trials (e.g. the handful of uncovered RSS/corl trials that
need re-masking) instead of a whole dataset root. Reads undistorted frames from
each trial's raw/images.

    ~/miniconda3/envs/sam3/bin/python -m src.dataset.gen_trial_masks \
        /path/to/trialA /path/to/trialB ...
"""
from __future__ import annotations

import argparse
import os

import cv2

PROMPT = "object on the checkerboard"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("trials", nargs="+", help="trial dirs (each with raw/images)")
    ap.add_argument("--prompt", default=PROMPT)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from autodex.perception.mask import Sam3ImageSegmentor
    seg = Sam3ImageSegmentor(gpu=0)

    n_img = n_none = 0
    for td in args.trials:
        img_dir = os.path.join(td, "raw", "images")
        if not os.path.isdir(img_dir):
            print(f"  no images: {td}")
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
            mask = seg.segment(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), args.prompt)
            if mask is None:
                n_none += 1
                mask = (bgr[:, :, 0] * 0).astype("uint8")
            cv2.imwrite(out_p, mask)
            n_img += 1
            done += 1
        print(f"  {os.path.basename(td)}: {done}/{len(serials)} masks")
    print(f"[trial-mask] wrote {n_img} masks ({n_none} empty) across {len(args.trials)} trials")


if __name__ == "__main__":
    main()
