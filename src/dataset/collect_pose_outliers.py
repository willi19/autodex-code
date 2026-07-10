"""Quarantine trials whose recomputed pose is unreliable (high silhouette loss).

The recompute step (``recompute_pose.json``) refines each trial's pose by
silhouette matching and REJECTS the result when the match loss exceeds 0.003
(``reject: true``, no ``pose_world.npy`` written). Those rejected trials still
sit in the clean set as bare directories with no valid pose, so we MOVE them
out of ``selected_100/`` into ``selected_100_pose_outlier/{obj}/{ts}/``.

The gate is on the silhouette loss of the *newly recomputed* pose:

    flagged  <=>  reject == true   (equivalently sil_loss >= THRESHOLD, default 0.003)

Trials already quarantined into ``selected_100_wireout/`` are left alone.
Trials with no ``recompute_pose.json`` at all (never recomputed) are reported
but NOT moved here — that is a separate failure class.

Usage:
    python -m src.dataset.collect_pose_outliers                    # dry-run report
    python -m src.dataset.collect_pose_outliers --threshold 0.002  # stricter cut
    python -m src.dataset.collect_pose_outliers --write            # move trials
"""

import argparse
import json
import os
import shutil

DATASET_ROOT = "/home/mingi/shared_data/autodex_dataset/selected_100"
OUTLIER_ROOT = "/home/mingi/shared_data/autodex_dataset/selected_100_pose_outlier"
SIL_THRESHOLD = 0.003


def iter_trials(root):
    for obj in sorted(os.listdir(root)):
        obj_dir = os.path.join(root, obj)
        if not os.path.isdir(obj_dir):
            continue
        for ts in sorted(os.listdir(obj_dir)):
            if os.path.isdir(os.path.join(obj_dir, ts)):
                yield obj, ts


def detect(dataset_root, threshold):
    """Split trials into sil-loss outliers vs. trials never recomputed."""
    flagged, no_json = [], []
    for obj, ts in iter_trials(dataset_root):
        rj = os.path.join(dataset_root, obj, ts, "recompute_pose.json")
        if not os.path.exists(rj):
            no_json.append({"obj": obj, "ts": ts})
            continue
        d = json.load(open(rj))
        sil = d.get("sil_loss")
        rejected = bool(d.get("reject")) or (sil is not None and sil >= threshold)
        if rejected:
            flagged.append({"obj": obj, "ts": ts, "sil_loss": sil,
                            "reject": bool(d.get("reject"))})
    flagged.sort(key=lambda f: -(f["sil_loss"] or 0))
    return flagged, no_json


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset_root", default=DATASET_ROOT)
    ap.add_argument("--outlier_root", default=OUTLIER_ROOT)
    ap.add_argument("--threshold", type=float, default=SIL_THRESHOLD,
                    help="sil_loss at/above which a trial is unreliable")
    ap.add_argument("--write", action="store_true", help="move trials (default: dry-run)")
    args = ap.parse_args()

    flagged, no_json = detect(args.dataset_root, args.threshold)
    print(f"sil-loss outliers flagged: {len(flagged)}  [sil_loss >= {args.threshold}]")
    print(f"(trials with no recompute_pose.json, NOT moved here: {len(no_json)})")

    moved = 0
    for f in flagged:
        src = os.path.join(args.dataset_root, f["obj"], f["ts"])
        dst = os.path.join(args.outlier_root, f["obj"], f["ts"])
        tag = "MOVE" if args.write else "would move"
        sil = f"{f['sil_loss']:.5f}" if f["sil_loss"] is not None else "  -  "
        print(f"  {tag} sil={sil}  {f['obj']}/{f['ts']}")
        if args.write:
            if not os.path.isdir(src):
                print(f"    ! src missing (already moved?): {src}")
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            moved += 1

    if args.write:
        os.makedirs(args.outlier_root, exist_ok=True)
        mpath = os.path.join(args.outlier_root, "pose_outlier_manifest.json")
        json.dump({"count": moved, "threshold_sil": args.threshold,
                   "criterion": f"reject or sil_loss >= {args.threshold}",
                   "trials": flagged}, open(mpath, "w"), indent=1)
        print(f"\nmoved {moved} trials -> {args.outlier_root}")
        print(f"manifest -> {mpath} ({moved} moved)")
    else:
        print("\n(dry-run) re-run with --write to move.")


if __name__ == "__main__":
    main()
