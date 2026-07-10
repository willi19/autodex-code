"""Re-split clean vs pose-outlier by the INITIAL (capture-time) pose sil loss.

The first outlier pass used the FoundPose *recompute* sil loss. The correct
criterion is the *initial* pose (the pose actually used at capture: legacy
``optimized_pose_world.txt``), measured by ``legacy_pose_sil_loss.py`` into
``legacy_sil_loss.json``. A trial is an outlier iff its initial-pose sil loss
exceeds the threshold. This moves trials between ``selected_100/`` and
``selected_100_pose_outlier/`` so the split matches that criterion.

    outlier  <=>  legacy_sil_loss > THRESHOLD   (default 0.003)

Usage:
    python -m src.dataset.reclassify_by_initial              # dry-run report
    python -m src.dataset.reclassify_by_initial --write      # move trials
"""
import argparse
import json
import os
import shutil

CLEAN_ROOT = "/home/mingi/shared_data/autodex_dataset/selected_100"
OUTLIER_ROOT = "/home/mingi/shared_data/autodex_dataset/selected_100_pose_outlier"
LEGACY_JSON = "/home/mingi/shared_data/autodex_dataset/legacy_sil_loss.json"
THRESHOLD = 0.003


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clean_root", default=CLEAN_ROOT)
    ap.add_argument("--outlier_root", default=OUTLIER_ROOT)
    ap.add_argument("--legacy_json", default=LEGACY_JSON)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    rows = json.load(open(args.legacy_json))["trials"]
    to_outlier, to_clean = [], []
    for r in rows:
        obj, ts = r["obj"], r["ts"]
        bad = r["legacy_sil_loss"] > args.threshold
        in_outlier = r["in_outlier"]
        if bad and not in_outlier:
            to_outlier.append(r)
        elif not bad and in_outlier:
            to_clean.append(r)

    print(f"reclassify by initial sil loss > {args.threshold}")
    print(f"  clean -> outlier (newly bad): {len(to_outlier)}")
    print(f"  outlier -> clean (recovered): {len(to_clean)}")

    def move(rows_, src_root, dst_root, tag):
        moved = 0
        for r in rows_:
            src = os.path.join(src_root, r["obj"], r["ts"])
            dst = os.path.join(dst_root, r["obj"], r["ts"])
            mark = "MOVE" if args.write else "would move"
            print(f"  {mark} [{tag}] legacy={r['legacy_sil_loss']:.5f}  {r['obj']}/{r['ts']}")
            if args.write:
                if not os.path.isdir(src):
                    print(f"    ! src missing: {src}")
                    continue
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.move(src, dst)
                moved += 1
        return moved

    m1 = move(to_outlier, args.clean_root, args.outlier_root, "->outlier")
    m2 = move(to_clean, args.outlier_root, args.clean_root, "->clean")

    if args.write:
        manifest = {"threshold": args.threshold,
                    "moved_to_outlier": m1, "moved_to_clean": m2,
                    "criterion": f"initial(legacy) sil_loss > {args.threshold}"}
        json.dump(manifest, open(os.path.join(args.outlier_root,
                  "reclassify_manifest.json"), "w"), indent=1)
        print(f"\nmoved {m1} -> outlier, {m2} -> clean")
    else:
        print("\n(dry-run) re-run with --write to move.")


if __name__ == "__main__":
    main()
