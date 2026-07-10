#!/usr/bin/env python3
"""Walk past trial result.json files and tally approach / place contact rates.

Usage:
    python src/experiment/num_camera/contact_stats.py \\
        --root ~/shared_data/AutoDex/experiment/v7_prev_1136/inspire_left \\
        [--obj attached_container]
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=str, nargs="+",
                    help="one or more roots of trial dirs, e.g. "
                         "v7/inspire_left v7_prev_1136/inspire_left")
    ap.add_argument("--obj", default=None,
                    help="filter to this obj subdir (default: all)")
    args = ap.parse_args()

    roots = [Path(os.path.expanduser(r)) for r in args.root]

    # Collect all obj names across roots.
    objs = set()
    for root in roots:
        if not root.is_dir():
            continue
        for d in root.iterdir():
            if d.is_dir() and (args.obj is None or d.name == args.obj):
                objs.add(d.name)
    objs = sorted(objs)

    grand = Counter()
    for obj in objs:
        c = Counter()
        for root in roots:
            obj_dir = root / obj
            if not obj_dir.is_dir():
                continue
            for trial in sorted(obj_dir.iterdir()):
                if not trial.is_dir() or not re.match(r"^2026", trial.name):
                    continue
                rp = trial / "result.json"
                if not rp.exists():
                    continue
                try:
                    with open(rp) as f:
                        r = json.load(f)
                except Exception:
                    continue
                c["n_trials"] += 1
                if r.get("success") is True:
                    c["n_success"] += 1
                exc = str(r.get("exception", ""))
                if "ContactDetected" in exc and "_move_joints" in exc:
                    c["approach_contact"] += 1
                place = (r.get("timing") or {}).get("place") or {}
                if place.get("stopped_on_contact"):
                    _d = place.get("descended", 0.0)
                    _t = place.get("target", 0.0)
                    if _t and _d < _t - 0.005:
                        c["place_early_contact"] += 1
                    else:
                        c["place_full_contact"] += 1
        if c["n_trials"]:
            print(f"\n{obj}: {c['n_trials']} trials  "
                  f"({c['n_success']} success)")
            for k in ("approach_contact", "place_early_contact",
                      "place_full_contact"):
                v = c[k]
                pct = 100 * v / c["n_trials"]
                print(f"  {k}: {v} ({pct:.1f}%)")
            grand += c

    if grand["n_trials"]:
        print(f"\n=== TOTAL: {grand['n_trials']} trials "
              f"({grand['n_success']} success) ===")
        for k in ("approach_contact", "place_early_contact",
                  "place_full_contact"):
            v = grand[k]
            pct = 100 * v / grand["n_trials"]
            print(f"  {k}: {v} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
