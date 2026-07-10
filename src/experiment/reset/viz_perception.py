#!/usr/bin/env python3
"""Render mesh-overlay PNGs for a trial's perception result.

For each camera that has a saved undistorted image in ``init_capture/images/``,
projects the obj mesh at the saved ``pose_world.npy`` and writes the overlay
to ``perception_overlay/{serial}.png``.

Quick way to eyeball whether the perception pose lined up with the actual
object in every camera (so you can tell if a high sil_loss is from bad pose,
bad mask, occlusion, glare, etc.).

Usage:
    # latest pepsi trial
    python src/experiment/reset/viz_perception.py --obj pepsi

    # explicit trial
    python src/experiment/reset/viz_perception.py \\
        --trial_dir ~/shared_data/AutoDex/experiment/reset_test/reorient_drop/inspire_left/pepsi/20260524_112431
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from paradex.image.image_dict import ImageDict


def _resolve_trial_dir(args) -> Path:
    if args.trial_dir:
        p = Path(args.trial_dir).expanduser()
        if not p.exists():
            sys.exit(f"trial dir not found: {p}")
        return p
    root = (Path.home() / "shared_data" / "AutoDex" / "experiment"
            / args.exp_name / args.hand / args.obj)
    if not root.exists():
        sys.exit(f"obj experiment dir not found: {root}")
    cands = sorted([p for p in root.iterdir() if p.is_dir()
                    and p.name[:1].isdigit()])
    if not cands:
        sys.exit(f"no trial dirs under {root}")
    return cands[-1]


def _infer_obj(trial_dir: Path, override: str | None) -> str:
    return override or trial_dir.parent.name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial_dir", default=None)
    parser.add_argument("--obj", default=None)
    parser.add_argument("--hand", default="inspire_left")
    parser.add_argument("--exp_name", default="reset_test/reorient_drop")
    parser.add_argument("--out_dir", default=None,
                        help="Where to save overlays. Default = "
                             "<trial_dir>/perception_overlay/")
    parser.add_argument("--prompt", default="object on the checkerboard",
                        help="SAM3 text prompt — defaults to the same prompt "
                             "reorient_drop.py uses.")
    args = parser.parse_args()

    trial_dir = _resolve_trial_dir(args)
    obj_name = _infer_obj(trial_dir, args.obj)
    print(f"[viz] trial = {trial_dir}")
    print(f"[viz] obj   = {obj_name}")

    img_dir = trial_dir / "init_capture"
    if not (img_dir / "images").exists():
        sys.exit(f"no init_capture/images/ in {trial_dir}")

    # ImageDict needs cam_param/ next to images/. The trial dir holds them at
    # the top level — give ImageDict the trial root (cam_param + init_capture
    # both live there).
    # Symlink images into a temp combined dir if needed; simplest: copy intr
    # files into init_capture/ for ImageDict.from_path.
    cam_param_src = trial_dir / "cam_param"
    cam_param_dst = img_dir / "cam_param"
    if not cam_param_dst.exists() and cam_param_src.exists():
        try:
            cam_param_dst.symlink_to(cam_param_src.resolve())
        except FileExistsError:
            pass

    img_dict = ImageDict.from_path(str(img_dir))
    n_cams = len(img_dict.images)
    print(f"[viz] loaded {n_cams} cam images from {img_dir / 'images'}")
    if n_cams == 0:
        sys.exit("no images loaded.")

    import cv2
    out_dir = (Path(args.out_dir) if args.out_dir
               else trial_dir / "perception_overlay")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Use SAM3 — the SAME segmentor init_daemon runs, so the resulting mask
    # matches what the perception pipeline actually saw.
    print(f"[viz] loading SAM3 (prompt={args.prompt!r})...")
    from autodex.perception.mask import Sam3ImageSegmentor
    seg = Sam3ImageSegmentor()
    n_mask_ok = 0
    for serial, img in img_dict.images.items():
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = seg.segment(rgb, prompt=args.prompt)
        out_img = img.copy()
        if mask is not None and mask.any():
            n_mask_ok += 1
            red = np.zeros_like(out_img); red[..., 2] = 255
            blend = cv2.addWeighted(out_img, 1.0, red, 0.45, 0.0)
            m3 = (mask > 0)[..., None]
            out_img = np.where(m3, blend, out_img)
            cnts, _ = cv2.findContours((mask > 0).astype(np.uint8),
                                        cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out_img, cnts, -1, (0, 255, 255), 2)
            label = "MASK OK"; color = (0, 255, 0)
        else:
            label = "MASK MISSING"; color = (0, 0, 255)
        cv2.putText(out_img, f"{serial}  {label}", (16, 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.imwrite(str(out_dir / f"{serial}_mask.png"), out_img)
    print(f"[viz] mask: {n_mask_ok}/{n_cams} cams got mask "
          f"→ {out_dir}/*_mask.png")


if __name__ == "__main__":
    main()
