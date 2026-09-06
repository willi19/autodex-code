"""
Per-grasp place precomputation.

For each candidate seed under candidates/{hand}/reset_{h_cm}/{obj}/reorient_{h_cm}/{i}_{j}/{seed},
search a (place_x, place_y, place_tz) grid for a place location that yields
IK-feasible:
  - q_placed                (always)
  - q_apex_j  (rotate apex) (always)
  - q_depart  (release-and-lift) (h_cm == 0 only — for put-down case)

Save the chosen place as `place.npy` (4×4 T_obj_end) and metadata as
`place_meta.json` in each seed directory. Seeds with no feasible grid point
have no place.npy and will be skipped by downstream sweep.

Usage:
    # one (i, j) pair
    python src/grasp_generation/reorient/compute_place.py \
        --obj attached_container --i 0 --j 16 --h_cm 0 --hand inspire_left

    # all available (i, j) pairs for this object/h_cm
    python src/grasp_generation/reorient/compute_place.py \
        --obj attached_container --all_pairs --h_cm 0 --hand inspire_left
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

from autodex.utils.path import get_obj_root

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan_reset import (  # noqa: E402
    APEX_Z, DEPART_DZ,
    _reset_candidate_path, _ik_solve,
    init_planner, load_candidates_object_frame, load_tabletop_pose,
    make_obj_pose, compute_apex,
)


def grid_place_options(place_x_vals, place_y_vals, place_tz_vals,
                        Tj_can: np.ndarray, h_m: float):
    """Yield (place_x, place_y, place_tz, T_obj_end) for every grid point."""
    for px in place_x_vals:
        for py in place_y_vals:
            for ptz in place_tz_vals:
                T_end = make_obj_pose(
                    Tj_can,
                    np.array([float(px), float(py), Tj_can[2, 3] + h_m]),
                    float(ptz),
                )
                yield float(px), float(py), float(ptz), T_end


def find_place_for_seed(planner, *, wrist_se3_obj, grasp_q,
                         Tj_can, h_m, full,
                         place_x_vals, place_y_vals, place_tz_vals,
                         retract_init):
    """Iterate (x, y, tz) grid; return first feasible (T_end, metadata) or None.

    Feasibility: IK at q_placed + q_apex_j + (q_depart only if full).
    """
    apex_z = APEX_Z
    open_q = np.zeros_like(grasp_q)

    for px, py, ptz, T_end in grid_place_options(
        place_x_vals, place_y_vals, place_tz_vals, Tj_can, h_m,
    ):
        wrist_place = T_end @ wrist_se3_obj
        q_placed = _ik_solve(planner, wrist_place, grasp_q, retract_q=retract_init)
        if q_placed is None:
            continue

        T_apex_j = compute_apex(T_end, wrist_se3_obj, apex_z)
        wrist_apex = T_apex_j @ wrist_se3_obj
        q_apex_j = _ik_solve(planner, wrist_apex, grasp_q, retract_q=q_placed)
        if q_apex_j is None:
            continue

        if full:
            wrist_depart = wrist_place.copy()
            wrist_depart[2, 3] += DEPART_DZ
            q_depart = _ik_solve(planner, wrist_depart, open_q, retract_q=q_placed)
            if q_depart is None:
                continue

        return {
            "place_x": px, "place_y": py, "place_tz": ptz,
            "T_obj_end": T_end,
        }
    return None


def discover_pairs(obj_name: str, hand: str, h_cm: int, *, version: str = "v8"):
    base = _reset_candidate_path(hand, h_cm, version) / obj_name / f"reorient_{h_cm}"
    if not base.exists():
        return []
    pairs = []
    for d in sorted(base.iterdir()):
        if d.is_dir() and "_" in d.name:
            try:
                i_str, j_str = d.name.split("_", 1)
                pairs.append((int(i_str), int(j_str)))
            except ValueError:
                continue
    return pairs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obj", required=True)
    p.add_argument("--i", type=int, default=None)
    p.add_argument("--j", type=int, default=None)
    p.add_argument("--all_pairs", action="store_true")
    p.add_argument("--pairs", default=None,
                    help='explicit "0_16,0_1" list')
    p.add_argument("--h_cm", type=int, default=0)
    p.add_argument("--hand", default="inspire_left",
                    choices=["inspire_left", "inspire", "allegro"])
    p.add_argument("--version", default="v8",
                   help="v8 reset/tabletop asset contract (only supported value)")
    p.add_argument("--x_min", type=float, default=0.35)
    p.add_argument("--x_max", type=float, default=0.55)
    p.add_argument("--x_step", type=float, default=0.05)
    p.add_argument("--y_min", type=float, default=-0.10)
    p.add_argument("--y_max", type=float, default=0.10)
    p.add_argument("--y_step", type=float, default=0.05)
    p.add_argument("--tz_step", type=float, default=30.0)
    p.add_argument("--overwrite", action="store_true",
                    help="recompute even if place.npy already exists")
    args = p.parse_args()
    if args.version != "v8":
        p.error("compute_place supports only --version v8; legacy reset assets are not used")
    obj_root = get_obj_root(args.version)

    # Default = all pairs.
    if args.pairs:
        pairs = [tuple(int(x) for x in tok.strip().split("_"))
                 for tok in args.pairs.split(",")]
    elif args.i is not None and args.j is not None:
        pairs = [(args.i, args.j)]
    else:
        pairs = discover_pairs(args.obj, args.hand, args.h_cm, version=args.version)
        if not pairs:
            print(f"[compute_place] no pairs discovered for {args.obj} h={args.h_cm}cm")
            return

    place_x_vals = np.arange(args.x_min, args.x_max + 1e-9, args.x_step).round(3)
    place_y_vals = np.arange(args.y_min, args.y_max + 1e-9, args.y_step).round(3)
    place_tz_vals = np.arange(0.0, 360.0, args.tz_step).round(3)
    n_grid = len(place_x_vals) * len(place_y_vals) * len(place_tz_vals)
    h_m = args.h_cm / 100.0
    full = (args.h_cm == 0)

    print(f"[compute_place] obj={args.obj} h={args.h_cm}cm hand={args.hand}")
    print(f"  pairs: {pairs}")
    print(f"  grid: x={place_x_vals.tolist()}")
    print(f"        y={place_y_vals.tolist()}")
    print(f"        tz=0..330 step {args.tz_step}° ({len(place_tz_vals)} vals)")
    print(f"  per-seed grid size: {n_grid}")
    print(f"  full (depart/retract): {full}")

    print(f"[compute_place] init planner ...")
    t0 = time.time()
    planner, _ = init_planner(args.hand)
    retract_init = planner._init_state.copy()
    print(f"[compute_place] planner warmup: {time.time() - t0:.1f}s")

    Tj_cache = {}
    grand_ok = grand_total = 0
    overall = time.time()

    for (i, j) in pairs:
        if j not in Tj_cache:
            Tj_cache[j] = load_tabletop_pose(args.obj, j, obj_root)
        Tj_can = Tj_cache[j]

        try:
            wrist_o_all, _preg, grasp_all, seed_ids, _places = load_candidates_object_frame(
                args.obj, args.hand, args.h_cm, i, j, version=args.version,
            )
        except FileNotFoundError as e:
            print(f"[pair {i}_{j}] SKIP — {e}")
            continue

        candidate_dir = (_reset_candidate_path(args.hand, args.h_cm, args.version)
                         / args.obj / f"reorient_{args.h_cm}" / f"{i}_{j}")
        n_ok = 0
        pair_t0 = time.time()
        for k, seed in enumerate(seed_ids):
            seed_dir = candidate_dir / seed
            place_npy = seed_dir / "place.npy"
            if place_npy.exists() and not args.overwrite:
                n_ok += 1
                continue

            result = find_place_for_seed(
                planner, wrist_se3_obj=wrist_o_all[k], grasp_q=grasp_all[k],
                Tj_can=Tj_can, h_m=h_m, full=full,
                place_x_vals=place_x_vals,
                place_y_vals=place_y_vals,
                place_tz_vals=place_tz_vals,
                retract_init=retract_init,
            )
            if result is None:
                # Remove stale place.npy if exists
                if place_npy.exists():
                    place_npy.unlink()
                meta_path = seed_dir / "place_meta.json"
                if meta_path.exists():
                    meta_path.unlink()
                continue

            np.save(place_npy, result["T_obj_end"].astype(np.float32))
            with open(seed_dir / "place_meta.json", "w") as f:
                json.dump({
                    "place_x": result["place_x"],
                    "place_y": result["place_y"],
                    "place_tz": result["place_tz"],
                    "h_cm": args.h_cm,
                    "checked": ["q_placed", "q_apex_j"] + (["q_depart"] if full else []),
                }, f, indent=2)
            n_ok += 1

        elapsed = time.time() - pair_t0
        print(f"[pair {i}_{j}] {n_ok}/{len(seed_ids)} seeds placed  ({elapsed:.1f}s)")
        grand_ok += n_ok
        grand_total += len(seed_ids)

    total = time.time() - overall
    print(f"\n[compute_place] done: {grand_ok}/{grand_total} seeds placed "
          f"across {len(pairs)} pairs ({total:.1f}s)")


if __name__ == "__main__":
    main()
