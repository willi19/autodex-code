"""Copy each trial's undistorted init frames into the dataset (v7 layout).

The current perception framework computes the object init pose from undistorted
per-camera frames. The legacy captures store those at ``{obj}/{ts}/images/``;
the v7 dataset layout expects them at ``{obj}/{ts}/init_capture/images/``. This
copies src ``images/`` -> dataset ``init_capture/images/`` so the pose can be
recomputed with the current framework.

    src {obj}/{ts}/images/  ->  dataset {obj}/{ts}/init_capture/images/

Usage:
    python -m src.dataset.copy_init_images            # dry-run report
    python -m src.dataset.copy_init_images --write    # actually copy
"""

import argparse
import os
import shutil

SRC_ROOT = "/home/mingi/shared_data/RSS2026_Mingi/experiment/selected_100"
DATASET_ROOTS = [
    "/home/mingi/shared_data/autodex_dataset/selected_100",
    "/home/mingi/shared_data/autodex_dataset/selected_100_wireout",
]
SRC_SUBDIR = "images"
DST_SUBDIR = os.path.join("init_capture", "images")


def iter_trials(root):
    for obj in sorted(os.listdir(root)):
        obj_dir = os.path.join(root, obj)
        if not os.path.isdir(obj_dir):
            continue
        for ts in sorted(os.listdir(obj_dir)):
            if os.path.isdir(os.path.join(obj_dir, ts)):
                yield obj, ts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src_root", default=SRC_ROOT)
    ap.add_argument("--roots", nargs="+", default=DATASET_ROOTS)
    ap.add_argument("--write", action="store_true", help="perform the copy (default: dry-run)")
    ap.add_argument("--overwrite", action="store_true", help="re-copy even if dst exists")
    args = ap.parse_args()

    stats = {"copied": 0, "skip_exist": 0, "no_src": 0}
    for root in args.roots:
        if not os.path.isdir(root):
            continue
        for obj, ts in iter_trials(root):
            src = os.path.join(args.src_root, obj, ts, SRC_SUBDIR)
            dst = os.path.join(root, obj, ts, DST_SUBDIR)
            if not os.path.isdir(src):
                stats["no_src"] += 1
                continue
            if os.path.isdir(dst) and not args.overwrite:
                stats["skip_exist"] += 1
                continue
            if args.write:
                if os.path.isdir(dst) and args.overwrite:
                    shutil.rmtree(dst)
                shutil.copytree(src, dst)
            stats["copied"] += 1

    verb = "copied" if args.write else "would copy"
    print("=== summary ===")
    print(f"  {verb:12s}: {stats['copied']}")
    print(f"  {'skip_exist':12s}: {stats['skip_exist']}")
    print(f"  {'no_src':12s}: {stats['no_src']}")
    if not args.write:
        print("\n(dry-run) re-run with --write to perform the copy.")


if __name__ == "__main__":
    main()
