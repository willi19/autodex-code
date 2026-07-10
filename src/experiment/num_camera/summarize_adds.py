#!/usr/bin/env python3
"""Aggregate ADD-S results from compute_adds.py across all trials per obj.

Walks ~/shared_data/AutoDex/experiment/num_camera_adds/{obj}/{ts}/result.json
and outputs per-(obj, k) mean / median / std / n.

Usage:
    python src/experiment/num_camera/summarize_adds.py
    python src/experiment/num_camera/summarize_adds.py --exp num_camera_adds
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(Path.home() / "shared_data/AutoDex/experiment"))
    p.add_argument("--exp", default="num_camera_adds")
    args = p.parse_args()

    root = Path(args.root) / args.exp
    if not root.is_dir():
        print(f"not found: {root}")
        return

    # (obj, k) → list of adds (mm)
    bucket = defaultdict(list)
    ks_seen = set()
    for obj_dir in sorted(root.iterdir()):
        if not obj_dir.is_dir():
            continue
        obj = obj_dir.name
        for tr in sorted(obj_dir.iterdir()):
            rj = tr / "result.json"
            if not rj.exists():
                continue
            try:
                d = json.load(open(rj))
            except Exception:
                continue
            entries = d.get("k_entries", {}) or {}
            for k_str, e in entries.items():
                try:
                    k = int(k_str)
                except Exception:
                    continue
                v = e.get("adds_m")
                if v is None:
                    continue
                bucket[(obj, k)].append(v * 1000.0)
                ks_seen.add(k)

    if not bucket:
        print("no ADD-S entries found")
        return

    ks_sorted = sorted(ks_seen)
    objs = sorted({o for (o, _) in bucket})
    obj_totals = defaultdict(list)

    def fmt_cell(vals):
        if not vals:
            return f"  {'—':>16}"
        a = np.array(vals)
        return f"  {a.mean():>5.1f}/{np.median(a):>4.1f}/{a.std():<4.0f}"

    head = f"{'obj':<22} " + " ".join(f"k={k:>2}".rjust(18) for k in ks_sorted) + "    n"
    print(head)
    print("-" * len(head))
    for obj in objs:
        row = f"{obj:<22} "
        n_obj = 0
        for k in ks_sorted:
            vals = bucket.get((obj, k), [])
            row += fmt_cell(vals)
            n_obj = max(n_obj, len(vals))
            obj_totals[k].extend(vals)
        row += f"   {n_obj:>3}"
        print(row)

    print("-" * len(head))
    row = f"{'ALL':<22} "
    n_all = 0
    for k in ks_sorted:
        vals = obj_totals.get(k, [])
        row += fmt_cell(vals)
        n_all = max(n_all, len(vals))
    row += f"   {n_all:>3}"
    print(row)
    print(f"\nformat: mean / median / std (mm)")


if __name__ == "__main__":
    main()
