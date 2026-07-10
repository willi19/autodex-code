"""Precompute video-synced arm/hand qpos for every dataset trial.

Wraps ``autodex.utils.sync.precompute_synced_qpos`` over all trials in the
dataset. Reads each trial's ``raw/arm``, ``raw/hand``, ``raw/timestamps`` and
writes video-frame-aligned, URDF-joint-order streams to the trial:

    {trial}/arm/state.npy   (F,6)   measured arm qpos
    {trial}/arm/action.npy  (F,6)   commanded arm qpos (action_qpos)
    {trial}/hand/state.npy  (F,16)  measured hand   (URDF order)
    {trial}/hand/action.npy (F,16)  commanded hand  (URDF order)

F = number of video timestamps (raw/timestamps/timestamp.npy). arm and hand end
up on the same index, so trial[i] arm and hand are simultaneous.

Usage:
    python -m src.dataset.sync_arm_hand            # dry-run report
    python -m src.dataset.sync_arm_hand --write    # compute + write
"""

import argparse
import os

from autodex.utils.sync import precompute_synced_qpos

ROOTS = [
    "/home/mingi/shared_data/autodex_dataset/selected_100",
    "/home/mingi/shared_data/autodex_dataset/selected_100_wireout",
]


def iter_trials(roots):
    for root in roots:
        if not os.path.isdir(root):
            continue
        for obj in sorted(os.listdir(root)):
            obj_dir = os.path.join(root, obj)
            if not os.path.isdir(obj_dir):
                continue
            for ts in sorted(os.listdir(obj_dir)):
                td = os.path.join(obj_dir, ts)
                if os.path.isdir(td):
                    yield td


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", nargs="+", default=ROOTS)
    ap.add_argument("--hand", default="allegro")
    ap.add_argument("--write", action="store_true", help="compute + write (default: dry-run)")
    ap.add_argument("--overwrite", action="store_true", help="recompute even if synced files exist")
    args = ap.parse_args()

    stats = {"ok": 0, "skip_exist": 0, "no_timestamps": 0, "error": 0}
    for td in iter_trials(args.roots):
        need = ("raw/arm", "raw/hand", "raw/timestamps/timestamp.npy")
        if not all(os.path.exists(os.path.join(td, p)) for p in need):
            stats["no_timestamps"] += 1
            continue
        done = all(os.path.exists(os.path.join(td, p)) for p in
                   ("arm/state.npy", "arm/action.npy", "hand/state.npy", "hand/action.npy"))
        if done and not args.overwrite:
            stats["skip_exist"] += 1
            continue
        if not args.write:
            stats["ok"] += 1  # would compute
            continue
        try:
            precompute_synced_qpos(td, args.hand, overwrite=args.overwrite)
            stats["ok"] += 1
        except Exception as e:
            stats["error"] += 1
            print(f"  ERROR {td}: {e}")

    print("=== summary ===")
    for k, v in stats.items():
        print(f"  {k:14s}: {v}")
    if not args.write:
        print("\n(dry-run) re-run with --write to compute.")


if __name__ == "__main__":
    main()
