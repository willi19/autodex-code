"""Quarantine trials with dead hand feedback into a sibling dataset dir.

Some allegro captures have a dead/disconnected hand-feedback line: the commanded
``raw/hand/action.npy`` moves normally, but the measured ``raw/hand/position.npy``
never changes across the whole trial. This detects those trials and MOVES them
out of the clean set into ``selected_100_wireout/{obj}/{ts}/``.

A long measured-frozen run alone is NOT enough: the hand legitimately holds
still during the arm-only phases (lift / place / return), and there BOTH the
command and the measurement sit constant. The true dead-feedback signature is
commanded-moves-but-measured-frozen. So we take the longest measured-frozen run
and ask whether the *commanded* action still varies across that same span:

    frozen_run  > MIN_FROZEN     (measured hand stuck for a while)  AND
    act_ptp_in_run > MIN_ACT_PTP (commanded hand kept moving there)

One rule catches both classes and excludes legit holds (act_ptp_in_run ~ 0):
    - whole-trial dead : the frozen run spans the entire trial
    - mid-motion cut   : the hand moves, then the feedback dies for the tail

Usage:
    python -m src.dataset.collect_wireout                  # dry-run report
    python -m src.dataset.collect_wireout --write          # move trials
"""

import argparse
import json
import os
import shutil

import numpy as np

DATASET_ROOT = "/home/mingi/shared_data/autodex_dataset/selected_100"
WIREOUT_ROOT = "/home/mingi/shared_data/autodex_dataset/selected_100_wireout"
SRC_ROOT = "/home/mingi/shared_data/RSS2026_Mingi/experiment/selected_100"
MIN_FROZEN = 300       # measured-hand frozen run length (samples ~ 3 s)
MIN_ACT_PTP = 0.3      # commanded hand range (rad) within that run => real motion


def _ptp(x):
    return float(np.abs(x.max(0) - x.min(0)).max()) if len(x) else 0.0


def frozen_runs(pos):
    """Yield (start, end) index spans where all hand dofs stay exactly equal."""
    if len(pos) < 2:
        return
    same = np.all(np.diff(pos, axis=0) == 0, axis=1)
    i = 0
    while i < len(same):
        if same[i]:
            j = i
            while j < len(same) and same[j]:
                j += 1
            yield i, j + 1          # +1: run covers pos[i .. j]
            i = j
        else:
            i += 1


def dead_feedback(pos, act):
    """Longest frozen run whose commanded action still moves.

    Returns (frozen_len, act_ptp_in_run, start_idx) for the worst such run, or
    (0, 0.0, 0) if none qualifies.
    """
    n = min(len(pos), len(act))
    pos, act = pos[:n], act[:n]
    worst = (0, 0.0, 0)
    for a, b in frozen_runs(pos):
        run_len = b - a
        if run_len <= MIN_FROZEN:
            continue
        act_ptp = _ptp(act[a:b])
        if act_ptp > MIN_ACT_PTP and run_len > worst[0]:
            worst = (run_len, act_ptp, a)
    return worst


def iter_trials(root):
    for obj in sorted(os.listdir(root)):
        obj_dir = os.path.join(root, obj)
        if not os.path.isdir(obj_dir):
            continue
        for ts in sorted(os.listdir(obj_dir)):
            if os.path.isdir(os.path.join(obj_dir, ts)):
                yield obj, ts


def detect(dataset_root, src_root):
    """Return trials whose measured hand feedback dies while commanded moves."""
    flagged = []
    for obj, ts in iter_trials(dataset_root):
        hand_dir = os.path.join(src_root, obj, ts, "raw", "hand")
        pos_f = os.path.join(hand_dir, "position.npy")
        act_f = os.path.join(hand_dir, "action.npy")
        if not (os.path.exists(pos_f) and os.path.exists(act_f)):
            continue
        pos, act = np.load(pos_f), np.load(act_f)
        frozen_len, act_ptp, start = dead_feedback(pos, act)
        if frozen_len == 0:
            continue
        n = min(len(pos), len(act))
        flagged.append({
            "obj": obj, "ts": ts, "n": int(n),
            "frozen": int(frozen_len), "act_ptp_in_run": round(float(act_ptp), 3),
            "frozen_start": int(start),
            "kind": "whole" if frozen_len > 0.9 * n else "mid",
        })
    return flagged


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset_root", default=DATASET_ROOT)
    ap.add_argument("--wireout_root", default=WIREOUT_ROOT)
    ap.add_argument("--src_root", default=SRC_ROOT)
    ap.add_argument("--write", action="store_true", help="move trials (default: dry-run)")
    args = ap.parse_args()

    flagged = detect(args.dataset_root, args.src_root)
    whole = sum(1 for f in flagged if f["kind"] == "whole")
    mid = sum(1 for f in flagged if f["kind"] == "mid")
    print(f"dead-hand-feedback flagged: {len(flagged)}  (whole={whole}, mid={mid})  "
          f"[frozen>{MIN_FROZEN}, act_ptp_in_run>{MIN_ACT_PTP}]")

    moved = 0
    for f in sorted(flagged, key=lambda x: -x["frozen"]):
        src = os.path.join(args.dataset_root, f["obj"], f["ts"])
        dst = os.path.join(args.wireout_root, f["obj"], f["ts"])
        tag = "MOVE" if args.write else "would move"
        print(f"  {tag} [{f['kind']:5s}] froz={f['frozen']:5d} "
              f"act_in_run={f['act_ptp_in_run']:.2f}  {f['obj']}/{f['ts']}")
        if args.write:
            if not os.path.isdir(src):
                print(f"    ! src trial missing (already moved?): {src}")
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(src, dst)
            moved += 1

    if args.write:
        os.makedirs(args.wireout_root, exist_ok=True)
        # Rebuild the manifest from everything currently in the wireout dir, so
        # earlier-quarantined trials stay recorded alongside the new moves.
        all_wire = detect(args.wireout_root, args.src_root)
        with open(os.path.join(args.wireout_root, "wireout_manifest.json"), "w") as fp:
            json.dump({"count": len(all_wire),
                       "criterion": f"frozen>{MIN_FROZEN} & act_ptp_in_run>{MIN_ACT_PTP}",
                       "trials": all_wire}, fp, indent=1)
        print(f"\nmoved {moved} trials -> {args.wireout_root}")
        print(f"manifest -> {os.path.join(args.wireout_root, 'wireout_manifest.json')} "
              f"({len(all_wire)} total)")
    else:
        print("\n(dry-run) re-run with --write to move.")


if __name__ == "__main__":
    main()
