"""Fix the (wrong) C2R.npy of previous captures using hand-eye recalibration.

Each legacy trial in ``experiment/selected_100/{obj}/{ts}/`` carries a wrong
``C2R.npy`` (T_world_robot). The same wrong matrix was produced by exactly one
hand-eye calibration episode under ``handeye_calibration/{episode}/0/C2R.npy``.
That episode is being recalibrated, yielding a corrected ``c2r_new.npy`` beside
the old one. This script matches each trial to its episode by the (wrong) C2R
value, then writes the episode's ``c2r_new.npy`` into the dataset trial as
``C2R.npy``.

    src trial C2R.npy  ==  handeye {episode}/0/C2R.npy  ->  copy c2r_new.npy
                                                            into dataset C2R.npy

Usage:
    python -m src.dataset.fix_c2r                 # dry-run report only
    python -m src.dataset.fix_c2r --write         # actually write C2R.npy
"""

import argparse
import glob
import json
import os

import numpy as np

SRC_ROOT = "/home/mingi/shared_data/RSS2026_Mingi/experiment/selected_100"
DST_ROOT = "/home/mingi/shared_data/autodex_dataset/selected_100"
# Hand-eye episodes live across a couple of roots. The main calibration folder
# plus the standalone 01-26 session (source of the white_hand_shower/screwdriver
# trials, which was captured outside the main folder).
HANDEYE_ROOTS = [
    "/home/mingi/shared_data/handeye_calibration",
    "/home/mingi/shared_data/handeyecalib_0126",
]

# Corrected C2R file candidates inside a handeye episode's "0/" dir.
NEW_C2R_NAMES = ("c2r_new.npy", "C2R_new.npy")
# Exact match expected (same array written to both places); tiny tol for safety.
MATCH_TOL = 1e-6


def build_handeye_index(handeye_roots=HANDEYE_ROOTS):
    """Load every handeye episode's (wrong) C2R.npy into memory.

    Scans each root for ``{episode}/0/C2R.npy``. Returns list of dicts:
    {episode, dir, c2r, new_path}. ``new_path`` is the corrected c2r file if
    present, else None.
    """
    if isinstance(handeye_roots, str):
        handeye_roots = [handeye_roots]
    index = []
    c2r_paths = []
    for root in handeye_roots:
        c2r_paths.extend(glob.glob(os.path.join(root, "*", "0", "C2R.npy")))
    for c2r_path in sorted(c2r_paths):
        ep_dir = os.path.dirname(c2r_path)
        episode = c2r_path.split(os.sep)[-3]
        try:
            c2r = np.load(c2r_path)
        except Exception:
            continue
        if c2r.shape != (4, 4):
            continue
        new_path = None
        for name in NEW_C2R_NAMES:
            p = os.path.join(ep_dir, name)
            if os.path.exists(p):
                new_path = p
                break
        index.append({"episode": episode, "dir": ep_dir, "c2r": c2r, "new_path": new_path})
    return index


def find_episode(old_c2r, index):
    """Return the unique handeye entry whose C2R matches ``old_c2r``.

    Raises LookupError if zero or more than one episode matches within MATCH_TOL.
    """
    matches = []
    for entry in index:
        if np.abs(entry["c2r"] - old_c2r).max() <= MATCH_TOL:
            matches.append(entry)
    if len(matches) == 0:
        raise LookupError("no matching handeye episode")
    if len(matches) > 1:
        eps = ", ".join(m["episode"] for m in matches)
        raise LookupError(f"ambiguous: {len(matches)} episodes match ({eps})")
    return matches[0]


def iter_trials(root):
    """Yield (obj, ts) for every trial dir under root."""
    for obj in sorted(os.listdir(root)):
        obj_dir = os.path.join(root, obj)
        if not os.path.isdir(obj_dir):
            continue
        for ts in sorted(os.listdir(obj_dir)):
            if os.path.isdir(os.path.join(obj_dir, ts)):
                yield obj, ts


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src_root", default=SRC_ROOT, help="legacy trials with wrong C2R.npy")
    ap.add_argument("--dst_root", default=DST_ROOT, help="dataset trials to receive fixed C2R.npy")
    ap.add_argument("--handeye_root", nargs="+", default=HANDEYE_ROOTS,
                    help="one or more roots holding {episode}/0/C2R.npy")
    ap.add_argument("--write", action="store_true", help="write C2R.npy (default: dry-run report)")
    ap.add_argument("--overwrite", action="store_true", help="overwrite existing dst C2R.npy")
    ap.add_argument("--report", default=None, help="optional path to write a JSON report")
    args = ap.parse_args()

    index = build_handeye_index(args.handeye_root)
    n_new = sum(1 for e in index if e["new_path"])
    print(f"handeye episodes: {len(index)} indexed, {n_new} have corrected c2r_new.npy")

    stats = {"ok": 0, "no_match": 0, "ambiguous": 0, "no_new": 0, "no_src": 0, "skip_exist": 0, "written": 0}
    rows = []
    for obj, ts in iter_trials(args.dst_root):
        src_c2r = os.path.join(args.src_root, obj, ts, "C2R.npy")
        dst_c2r = os.path.join(args.dst_root, obj, ts, "C2R.npy")
        row = {"obj": obj, "ts": ts, "status": None, "episode": None}

        if not os.path.exists(src_c2r):
            row["status"] = "no_src"
            stats["no_src"] += 1
            rows.append(row)
            continue

        old = np.load(src_c2r)
        try:
            entry = find_episode(old, index)
        except LookupError as e:
            msg = str(e)
            row["status"] = "ambiguous" if msg.startswith("ambiguous") else "no_match"
            stats[row["status"]] += 1
            rows.append(row)
            continue

        row["episode"] = entry["episode"]
        if entry["new_path"] is None:
            row["status"] = "no_new"
            stats["no_new"] += 1
            rows.append(row)
            continue

        stats["ok"] += 1
        if os.path.exists(dst_c2r) and not args.overwrite:
            row["status"] = "skip_exist"
            stats["skip_exist"] += 1
            rows.append(row)
            continue

        row["status"] = "ready"
        if args.write:
            new_c2r = np.load(entry["new_path"])
            np.save(dst_c2r, new_c2r)
            row["status"] = "written"
            stats["written"] += 1
        rows.append(row)

    print("\n=== summary ===")
    for k, v in stats.items():
        print(f"  {k:12s}: {v}")
    print(f"  {'total':12s}: {len(rows)}")

    # Surface problems explicitly.
    for label in ("no_match", "ambiguous", "no_src"):
        bad = [r for r in rows if r["status"] == label]
        if bad:
            print(f"\n{label} ({len(bad)}):")
            for r in bad[:20]:
                print(f"    {r['obj']}/{r['ts']}")
            if len(bad) > 20:
                print(f"    ... +{len(bad) - 20} more")

    if not args.write:
        print("\n(dry-run) re-run with --write once c2r_new.npy exists for all episodes.")

    if args.report:
        with open(args.report, "w") as f:
            json.dump({"stats": stats, "rows": rows}, f, indent=1)
        print(f"\nreport -> {args.report}")


if __name__ == "__main__":
    main()
