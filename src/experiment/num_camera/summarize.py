#!/usr/bin/env python3
"""Summarize num_camera_k* trial outcomes.

Categories per trial result.json:
    SUCC    success=true (manual y at lift, OR auto-flipped place_contact)
    FAIL_N  success=false, "lift" reached → user pressed n at lift
    FAIL_A  success=false, "lift" NOT reached (approach/grasp aborted before lift)
            includes execution_states ending at init/approach/pregrasp/grasp/squeeze
    EXCL    plan/perception/tabletop failures, NO_RESULT, success=null

Success rate = SUCC / (SUCC + FAIL_N + FAIL_A).

Usage:
    python src/experiment/num_camera/summarize.py
    python src/experiment/num_camera/summarize.py --root ~/shared_data/AutoDex/experiment
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict
from pathlib import Path


REACHED_LIFT_STATES = {"lift", "lift_done", "place", "place_done",
                       "hand_open", "seq_base_shoulder", "plan_wrist",
                       "reset_done", "arm_retract"}


def categorize(rj: dict) -> str:
    """SUCC / FAIL_N / FAIL_A / EXCL."""
    succ = rj.get("success")
    if succ is True:
        return "SUCC"
    if succ is None:
        return "EXCL"
    reason = rj.get("reason") or ""
    if reason in ("planning_failed_all_candidates", "all_scenes_done",
                  "tabletop_unclassified", "perception_failed",
                  "scene_already_done", "sil_loss_too_high"):
        return "EXCL"
    if reason.startswith("perception") or reason.startswith("sil_"):
        return "EXCL"
    # success=False with no fail-stage reason → user labeled n
    # Distinguish approach-aborted vs lift-then-dropped via execution_states.
    states = (rj.get("timing", {}) or {}).get("execution_states") or []
    state_names = {s.get("state") for s in states if isinstance(s, dict)}
    reached_lift = bool(state_names & REACHED_LIFT_STATES)
    return "FAIL_N" if reached_lift else "FAIL_A"


def scan_root(root: Path):
    """Yield (k, obj, trial_dir, category)."""
    for k_dir in sorted(root.glob("num_camera_k*")):
        m = re.match(r"num_camera_k(\d+)$", k_dir.name)
        if not m:
            continue
        k = int(m.group(1))
        for obj_dir in sorted(k_dir.glob("*/inspire_left/*")):
            obj = obj_dir.name
            for tr in sorted(obj_dir.iterdir()):
                if not tr.is_dir():
                    continue
                rj_path = tr / "result.json"
                if not rj_path.exists():
                    yield k, obj, tr, "EXCL"  # NO_RESULT
                    continue
                try:
                    rj = json.load(open(rj_path))
                except Exception:
                    yield k, obj, tr, "EXCL"
                    continue
                yield k, obj, tr, categorize(rj)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(Path.home() / "shared_data/AutoDex/experiment"))
    p.add_argument("--verbose", action="store_true", help="Print every trial")
    args = p.parse_args()

    counts = defaultdict(lambda: {"SUCC": 0, "FAIL_N": 0, "FAIL_A": 0, "EXCL": 0})
    rows = list(scan_root(Path(args.root)))
    for k, obj, tr, cat in rows:
        counts[(k, obj)][cat] += 1
        if args.verbose:
            print(f"k={k} {obj} {tr.name}  {cat}")

    if not counts:
        print("no trials found")
        return

    # Per-k tables (separate block per k).
    by_k = defaultdict(dict)
    for (k, obj), c in counts.items():
        by_k[k][obj] = c
    for k in sorted(by_k):
        print(f"\n=== k={k} ===")
        print(f"{'obj':<22}  {'all':>3} {'ok':>3} {'fail_n':>6} {'fail_a':>6}  {'rate':>6}")
        print("-" * 55)
        k_tot = {"SUCC": 0, "FAIL_N": 0, "FAIL_A": 0}
        for obj in sorted(by_k[k]):
            c = by_k[k][obj]
            denom = c["SUCC"] + c["FAIL_N"] + c["FAIL_A"]
            if denom == 0:
                continue
            rate = f"{100*c['SUCC']/denom:.0f}%"
            print(f"{obj:<22}  {denom:>3} {c['SUCC']:>3} {c['FAIL_N']:>6} "
                  f"{c['FAIL_A']:>6}  {rate:>6}")
            for cat in k_tot:
                k_tot[cat] += c[cat]
        print("-" * 55)
        denom = sum(k_tot.values())
        rate = f"{100*k_tot['SUCC']/denom:.0f}%" if denom else "—"
        print(f"{'TOTAL':<22}  {denom:>3} {k_tot['SUCC']:>3} {k_tot['FAIL_N']:>6} "
              f"{k_tot['FAIL_A']:>6}  {rate:>6}")


if __name__ == "__main__":
    main()
