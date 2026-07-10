"""Copy the ``raw/`` directory of each legacy trial into the dataset.

The dataset trials under ``autodex_dataset/selected_100/{obj}/{ts}/`` were
seeded with only undistorted videos + cam_param. This brings over the original
``raw/`` dir (per-camera raw frames, timestamps, arm/hand robot state) from the
source captures, aligned by ``{obj}/{ts}``.

    src {obj}/{ts}/raw/  ->  dataset {obj}/{ts}/raw/

Usage:
    python -m src.dataset.copy_raw            # dry-run report only
    python -m src.dataset.copy_raw --write    # actually copy
"""

import argparse
import os
import shutil

SRC_ROOT = "/home/mingi/shared_data/RSS2026_Mingi/experiment/selected_100"
DST_ROOT = "/home/mingi/shared_data/autodex_dataset/selected_100"


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
    ap.add_argument("--dst_root", default=DST_ROOT)
    ap.add_argument("--subdir", default="raw", help="trial subdirectory to copy")
    ap.add_argument("--write", action="store_true", help="perform the copy (default: dry-run)")
    ap.add_argument("--overwrite", action="store_true", help="re-copy even if dst subdir exists")
    args = ap.parse_args()

    stats = {"copied": 0, "skip_exist": 0, "no_src": 0, "no_dst_trial": 0}
    for obj, ts in iter_trials(args.dst_root):
        src = os.path.join(args.src_root, obj, ts, args.subdir)
        dst = os.path.join(args.dst_root, obj, ts, args.subdir)

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
        else:
            stats["copied"] += 1  # "would copy"

    print("=== summary ===")
    verb = "copied" if args.write else "would copy"
    print(f"  {verb:14s}: {stats['copied']}")
    print(f"  {'skip_exist':14s}: {stats['skip_exist']}")
    print(f"  {'no_src':14s}: {stats['no_src']}")
    if not args.write:
        print("\n(dry-run) re-run with --write to perform the copy.")


if __name__ == "__main__":
    main()
