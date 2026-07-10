#!/usr/bin/env python3
"""Walk past trial result.json files, find contact events, extract grid
frames from the corresponding multi-camera videos, save to
``{REPO_ROOT}/contact_snaps/{obj}/{label}/{timestamp}.jpg``.

Two contact event types handled:
  - place_early_contact (timing.place.stopped_on_contact + descended<target)
    → frame at contact_t_s into the PLACE video.
  - approach contact (exception with ContactDetected)
    → frame near the END of the EXEC video (exact timing unrecorded).

Usage:
    python src/experiment/num_camera/extract_contact_grids.py \\
        --root ~/shared_data/AutoDex/experiment/v7_prev_1136/inspire_left \\
              ~/shared_data/AutoDex/experiment/v7/inspire_left ...
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = os.path.expanduser("~/AutoDex")
OUT_ROOT = os.path.join(REPO_ROOT, "contact_snaps")


def _grid(frames, target_w=480):
    """Stack list of BGR frames into a grid image."""
    if not frames:
        return None
    rsz = []
    for f in frames:
        scale = target_w / f.shape[1]
        rsz.append(cv2.resize(f, (target_w, int(f.shape[0] * scale))))
    n = len(rsz)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    h, w = rsz[0].shape[:2]
    canvas = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    for i, im in enumerate(rsz):
        r, c = i // cols, i % cols
        canvas[r*h:(r+1)*h, c*w:(c+1)*w] = im
    return canvas


def _videos_in(d):
    if not os.path.isdir(d):
        return []
    return sorted(glob.glob(os.path.join(d, "*.avi")) +
                   glob.glob(os.path.join(d, "*.mp4")))


def _grab_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    idx = max(0, min(frame_idx, n - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None


def _extract(video_dir, t_sec, fallback="last"):
    """Grab one frame per camera video at ``t_sec`` seconds in (or 'last'
    frame as fallback). Returns list of frames."""
    vids = _videos_in(video_dir)
    if not vids:
        return []
    frames = []
    for v in vids:
        cap = cv2.VideoCapture(v)
        if not cap.isOpened():
            continue
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        if t_sec is None or t_sec < 0:
            idx = n - 1   # last frame
        else:
            idx = min(int(t_sec * fps), n - 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        cap.release()
        if ok:
            frames.append(frame)
    return frames


def process_trial(trial_dir: Path, obj: str):
    rp = trial_dir / "result.json"
    if not rp.exists():
        return
    try:
        with open(rp) as f:
            r = json.load(f)
    except Exception:
        return
    ts = trial_dir.name

    # place early contact
    place = (r.get("timing") or {}).get("place") or {}
    if place.get("stopped_on_contact"):
        _d = place.get("descended", 0.0)
        _t = place.get("target", 0.0)
        if _t and _d < _t - 0.005:
            ct = place.get("contact_t_s")
            place_vid_dir = trial_dir / "videos" / "place"
            frames = _extract(place_vid_dir, ct)
            grid = _grid(frames)
            if grid is not None:
                out_d = Path(OUT_ROOT) / obj / "place_early"
                out_d.mkdir(parents=True, exist_ok=True)
                out_p = out_d / f"{ts}.jpg"
                cv2.imwrite(str(out_p), grid)
                print(f"  place_early  → {out_p}")

    # approach contact
    exc = str(r.get("exception", ""))
    if "ContactDetected" in exc and "_move_joints" in exc:
        exec_vid_dir = trial_dir / "videos" / "exec"
        # exact timing not saved; grab last frame.
        frames = _extract(exec_vid_dir, None)
        grid = _grid(frames)
        if grid is not None:
            out_d = Path(OUT_ROOT) / obj / "approach"
            out_d.mkdir(parents=True, exist_ok=True)
            out_p = out_d / f"{ts}.jpg"
            cv2.imwrite(str(out_p), grid)
            print(f"  approach     → {out_p}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, nargs="+",
                    help="one or more roots of trial dirs")
    ap.add_argument("--obj", default=None)
    args = ap.parse_args()

    roots = [Path(os.path.expanduser(r)) for r in args.root]
    objs = set()
    for root in roots:
        if not root.is_dir():
            continue
        for d in root.iterdir():
            if d.is_dir() and (args.obj is None or d.name == args.obj):
                objs.add(d.name)
    objs = sorted(objs)

    for obj in objs:
        print(f"\n=== {obj} ===")
        for root in roots:
            obj_dir = root / obj
            if not obj_dir.is_dir():
                continue
            for trial in sorted(obj_dir.iterdir()):
                if not trial.is_dir() or not re.match(r"^2026", trial.name):
                    continue
                process_trial(trial, obj)


if __name__ == "__main__":
    main()
