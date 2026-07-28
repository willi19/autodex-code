"""Re-sync arm/hand qpos onto the 30fps VIDEO-frame axis (not the 33fps triggers).

precompute_synced_qpos resampled arm/hand onto timestamp.npy (the 33.3fps trigger
clock, 232 frames). But the cameras record at 30fps, so the saved videos (and
gotrack object poses) have ~203 frames and drift from the 33fps qpos. This
resamples the RAW arm/hand streams onto the true video-frame times so frame f of
every stream (arm/hand/object video) is the same instant.

Video-frame times:
  - if object_tracking exists: vt[0] + gotrack time_sec  (the actual 30fps frames)
  - else: vt[0] + arange(N)/30  with N = min camera .avi frame count

Overwrites (video-frame axis, same +0.03 offset as before):
  arm/state.npy  arm/action.npy(cart)  arm/action_qpos.npy  hand/state.npy  hand/action.npy

Then re-run fix_lift_action_ik to repair action_qpos lift frames on the new axis.

    ~/miniconda3/envs/mingi/bin/python -m src.dataset.resync_video_axis \
        --roots <dataset_root> [--obj X] [--write]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import cv2
import numpy as np
from scipy.interpolate import interp1d

DEFAULT_ROOTS = ["/home/mingi/shared_data/autodex_dataset/selected_100"]
OFFSET = 0.03


def video_frame_times(trial):
    """Absolute times of the 30fps video frames. Prefer gotrack time_sec."""
    vt = np.load(os.path.join(trial, "raw/timestamps/timestamp.npy"))
    wp = os.path.join(trial, "object_tracking/gotrack_output/world_pose_records.json")
    if os.path.exists(wp):
        recs = [r for r in json.load(open(wp)) if r.get("status") == "ok"]
        if recs:
            return vt[0] + np.array([float(r["time_sec"]) for r in recs]), "gotrack"
    avis = glob.glob(os.path.join(trial, "videos", "*.avi"))
    if not avis:
        return None, "no_video"
    ns = []
    for v in avis:
        c = cv2.VideoCapture(v); ns.append(int(c.get(cv2.CAP_PROP_FRAME_COUNT))); c.release()
    N = min(ns)
    return vt[0] + np.arange(N) / 30.0, "uniform30"


def resample_to(src_time, src_val, tgt):
    m = min(len(src_time), len(src_val))
    f = interp1d(src_time[:m], src_val[:m], axis=0, bounds_error=False,
                 fill_value=(src_val[0], src_val[m - 1]))
    return f(tgt).astype(np.float32)


def trigger_fps(trial):
    vt = np.load(os.path.join(trial, "raw/timestamps/timestamp.npy"))
    if len(vt) < 2 or vt[-1] <= vt[0]:
        return 0.0
    return (len(vt) - 1) / (vt[-1] - vt[0])


def resync_trial(trial, write):
    # Only trials whose trigger clock is the fast 33fps (mismatched with the 30fps
    # video) need re-syncing; ones already at ~30fps are left untouched.
    fps = trigger_fps(trial)
    if fps < 31.5:
        return "skip_30fps", 0
    ft, src = video_frame_times(trial)
    if ft is None:
        return "no_video", 0
    rarm = os.path.join(trial, "raw/arm"); rhand = os.path.join(trial, "raw/hand")
    need = [f"{rarm}/time.npy", f"{rarm}/position.npy", f"{rarm}/action_qpos.npy",
            f"{rarm}/action.npy", f"{rhand}/time.npy", f"{rhand}/position.npy",
            f"{rhand}/action.npy"]
    if not all(os.path.exists(p) for p in need):
        return "no_raw", 0

    at = np.load(f"{rarm}/time.npy") + OFFSET
    ht = np.load(f"{rhand}/time.npy") + OFFSET
    out = {
        "arm/state.npy":       resample_to(at, np.load(f"{rarm}/position.npy"), ft),
        "arm/action_qpos.npy": resample_to(at, np.load(f"{rarm}/action_qpos.npy"), ft),
        "arm/action.npy":      resample_to(at, np.load(f"{rarm}/action.npy"), ft),
        "hand/state.npy":      resample_to(ht, np.load(f"{rhand}/position.npy"), ft),
        "hand/action.npy":     resample_to(ht, np.load(f"{rhand}/action.npy"), ft),
    }
    if write:
        for rel, arr in out.items():
            np.save(os.path.join(trial, rel), arr)
    return f"ok:{src}", len(ft)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", nargs="+", default=DEFAULT_ROOTS)
    ap.add_argument("--obj")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    trials = []
    for root in args.roots:
        for obj in sorted(os.listdir(root)):
            if args.obj and obj != args.obj:
                continue
            od = os.path.join(root, obj)
            if not os.path.isdir(od):
                continue
            for ts in sorted(os.listdir(od)):
                d = os.path.join(od, ts)
                if os.path.isdir(d):
                    trials.append(d)
    if args.limit:
        trials = trials[: args.limit]
    print(f"[resync] {len(trials)} trials  write={args.write}")

    from collections import Counter
    stats = Counter()
    for d in trials:
        try:
            r, n = resync_trial(d, args.write)
        except Exception as e:
            print(f"  ERR {os.path.basename(os.path.dirname(d))}/{os.path.basename(d)}: {type(e).__name__} {e}")
            stats["err"] += 1
            continue
        stats[r.split(":")[0]] += 1
    print("summary:", dict(stats))


if __name__ == "__main__":
    main()
