"""
Sweep reset planner over a (pickup_x, pickup_theta_z) grid, optionally across
multiple (i, j) tabletop-pose pairs in one process (one planner warmup).

Output (per pair):
    outputs/reset_cache/{hand}/{obj}/reorient_{h_cm}/{i}_{j}/
        sweep_summary.json
        x{xx}_tz{zz}/{seed_id}/trajectory.npz
        x{xx}_tz{zz}/{seed_id}/meta.json

Usage:
    # Single pair
    python src/grasp_generation/reorient/sweep_reset.py \
        --obj attached_container --i 0 --j 16 --h_cm 0 --hand inspire_left

    # All available (i, j) pairs for this object
    python src/grasp_generation/reorient/sweep_reset.py \
        --obj attached_container --all_pairs --h_cm 0 --hand inspire_left

    # Explicit pair list
    python src/grasp_generation/reorient/sweep_reset.py \
        --obj attached_container --pairs 0_16,0_1,1_0,16_0 --h_cm 0
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from autodex.utils.path import get_obj_root, repo_dir

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plan_reset import (  # noqa: E402
    DEFAULT_PLACE_XY, DEFAULT_PLACE_TZ, APEX_Z, HAND_Z_MIN, phase_names_for,
    init_planner, load_tabletop_pose, make_obj_pose, plan_one_cell, save_plan,
    load_fk_urdf, load_object_vertices, _reset_candidate_path,
    dominant_fail_category,
)


def discover_pairs(obj_name: str, hand: str, h_cm: int, *, version: str = "v8"):
    """List direct v8 (i, j) cells from the runtime reset candidate tree."""
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


def sweep_one_pair(planner, base_world, urdf_fk, ee_link, obj_verts, *,
                    obj_name, hand, i, j, h_cm, xs, tzs,
                    place_x, place_y, place_tz, max_seeds, sweep_root, skip_done,
                    obj_root, version: str = "v8"):
    """Sweep (x, tz) grid for one (i, j) pair. Writes per-cell + summary."""
    sweep_root.mkdir(parents=True, exist_ok=True)
    h_m = h_cm / 100.0
    Ti = load_tabletop_pose(obj_name, i, obj_root)

    # Per-seed place search: iterate these tz values to find feasible place.
    place_tz_grid = list(np.arange(0.0, 360.0, 30.0))

    summary = {
        "obj_name": obj_name, "hand": hand,
        "i": i, "j": j, "h_cm": h_cm,
        "x_values": xs.tolist(), "tz_values": tzs.tolist(),
        "place_x": place_x, "place_y": place_y,
        "place_search_tzs": place_tz_grid,
        "phase_names": phase_names_for(h_cm), "apex_z": APEX_Z, "hand_z_min": HAND_Z_MIN,
        "max_seeds": max_seeds,
        "cells": [],
    }

    n_ok = 0
    t_pair = time.time()
    for x in xs:
        for tz in tzs:
            cell_name = f"x{x:.2f}_tz{int(round(tz)):03d}"
            cell_dir = sweep_root / cell_name
            cached = None
            if skip_done and cell_dir.exists():
                for sd in cell_dir.iterdir():
                    if (sd / "trajectory.npz").exists() and (sd / "meta.json").exists():
                        cached = sd
                        break
            if cached is not None:
                print(f"  [{i}_{j}] {cell_name}: cached -> {cached.name}")
                summary["cells"].append({
                    "x": float(x), "tz": float(tz), "status": "ok",
                    "seed_id": cached.name, "elapsed_s": 0.0, "cached": True,
                })
                n_ok += 1
                continue

            T_obj_start = make_obj_pose(
                Ti, np.array([float(x), 0.0, Ti[2, 3]]), float(tz),
            )
            t1 = time.time()
            result = plan_one_cell(
                planner, obj_name=obj_name, hand=hand,
                h_cm=h_cm, i=i, j=j,
                obj_root=obj_root, version=version,
                T_obj_start=T_obj_start,
                place_xy=(place_x, place_y),
                place_search_tzs=place_tz_grid,
                base_world=base_world, max_seeds=max_seeds, verbose=False,
                urdf_fk=urdf_fk, ee_link=ee_link, obj_verts=obj_verts,
            )
            elapsed = time.time() - t1
            if result is None or result.get("status") != "ok":
                fail_counts = result.get("fail_counts", {}) if result else {}
                cat = dominant_fail_category(fail_counts)
                top = max(fail_counts.items(), key=lambda kv: kv[1])[0] if fail_counts else "no_candidates"
                print(f"  [{i}_{j}] {cell_name}: FAIL [{cat}/{top}] ({elapsed:.1f}s)")
                summary["cells"].append({
                    "x": float(x), "tz": float(tz), "status": "fail",
                    "fail_category": cat, "fail_top": top,
                    "fail_counts": fail_counts,
                    "elapsed_s": round(elapsed, 2),
                })
            else:
                n_ok += 1
                ptz_used = result.get("place_tz_used")
                ptz_str = f"  place_tz={ptz_used:.0f}°" if ptz_used is not None else ""
                print(f"  [{i}_{j}] {cell_name}: ok seed={result['seed_id']}{ptz_str} ({elapsed:.1f}s)")
                out_dir = cell_dir / result["seed_id"]
                meta = {
                    "obj_name": obj_name, "hand": hand,
                    "i": i, "j": j, "h_cm": h_cm,
                    "pickup_x": float(x), "pickup_tz": float(tz),
                    "place_x": place_x, "place_y": place_y,
                    "place_tz": ptz_used if ptz_used is not None else place_tz,
                    "seed_id": result["seed_id"], "phase_names": phase_names_for(h_cm),
                    "wrist_se3_obj": result["wrist_se3_obj"].tolist(),
                    "T_obj_start": T_obj_start.tolist(),
                    "T_obj_apex_i": result["T_obj_apex_i"].tolist(),
                    "T_obj_apex_j": result["T_obj_apex_j"].tolist(),
                    "T_obj_end": result["T_obj_end"].tolist(),
                    "apex_z": APEX_Z, "hand_z_min": HAND_Z_MIN,
                }
                save_plan(out_dir, result["trajs"], meta)
                summary["cells"].append({
                    "x": float(x), "tz": float(tz), "status": "ok",
                    "seed_id": result["seed_id"], "place_tz": ptz_used,
                    "elapsed_s": round(elapsed, 2),
                })
            with open(sweep_root / "sweep_summary.json", "w") as f:
                json.dump(summary, f, indent=2)

    total = time.time() - t_pair
    print(f"[pair {i}_{j}] done: {n_ok}/{len(summary['cells'])} ok  ({total:.1f}s)")
    return n_ok, len(summary["cells"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obj", required=True)
    # Pair selection: --i/--j (single) | --all_pairs | --pairs "0_16,0_1"
    p.add_argument("--i", type=int, default=None)
    p.add_argument("--j", type=int, default=None)
    p.add_argument("--all_pairs", action="store_true",
                    help="auto-discover all (i, j) pairs for this object")
    p.add_argument("--pairs", default=None,
                    help='explicit comma-separated list, e.g. "0_16,0_1,1_0"')
    p.add_argument("--h_cm", type=int, default=0)
    p.add_argument("--hand", default="inspire_left",
                    choices=["inspire_left", "inspire", "allegro"])
    p.add_argument("--version", default="v8",
                   help="v8 reset/tabletop asset contract (only supported value)")
    p.add_argument("--x_min", type=float, default=0.30)
    p.add_argument("--x_max", type=float, default=0.55)
    p.add_argument("--x_step", type=float, default=0.05)
    p.add_argument("--tz_min", type=float, default=0.0)
    p.add_argument("--tz_max", type=float, default=330.0)
    p.add_argument("--tz_step", type=float, default=30.0)
    p.add_argument("--place_x", type=float, default=DEFAULT_PLACE_XY[0])
    p.add_argument("--place_y", type=float, default=DEFAULT_PLACE_XY[1])
    p.add_argument("--place_tz", type=float, default=DEFAULT_PLACE_TZ)
    p.add_argument("--max_seeds", type=int, default=20)
    p.add_argument("--out_root", default=None,
                    help="override output root (default: ~/AutoDex/outputs/reset_cache/{hand}/{obj}/reorient_{h_cm})")
    p.add_argument("--skip_done", action="store_true",
                    help="skip cells with existing trajectory")
    args = p.parse_args()
    if args.version != "v8":
        p.error("sweep_reset supports only --version v8; legacy reset assets are not used")
    obj_root = get_obj_root(args.version)

    # Resolve pair list. Default = all pairs (most common case).
    if args.pairs:
        pairs = []
        for tok in args.pairs.split(","):
            i_s, j_s = tok.strip().split("_")
            pairs.append((int(i_s), int(j_s)))
    elif args.i is not None and args.j is not None:
        pairs = [(args.i, args.j)]
    else:
        pairs = discover_pairs(args.obj, args.hand, args.h_cm, version=args.version)
        if not pairs:
            print(f"[sweep] no pairs discovered for {args.obj} h={args.h_cm}cm")
            return

    xs = np.arange(args.x_min, args.x_max + 1e-6, args.x_step).round(3)
    tzs = np.arange(args.tz_min, args.tz_max + 1e-6, args.tz_step).round(3)

    print(f"[sweep] obj={args.obj} h={args.h_cm}cm hand={args.hand}")
    print(f"[sweep] {len(pairs)} pair(s): {pairs}")
    print(f"[sweep] x: {xs.tolist()}")
    print(f"[sweep] tz: {tzs.tolist()}")
    print(f"[sweep] place=({args.place_x:.2f}, {args.place_y:.2f}, tz={args.place_tz:.0f}°)")
    print(f"[sweep] cells per pair: {len(xs) * len(tzs)}")

    print(f"[sweep] init planner ...")
    t0 = time.time()
    planner, base_world = init_planner(args.hand)
    urdf_fk, ee_link = load_fk_urdf(args.hand)
    obj_verts = load_object_vertices(args.obj, obj_root)
    print(f"[sweep] planner warmup: {time.time() - t0:.1f}s ({len(obj_verts)} mesh verts)")

    out_root = (Path(args.out_root) if args.out_root else
                Path(repo_dir) / "outputs" / "reset_cache" / args.version / args.hand / args.obj
                / f"reorient_{args.h_cm}")

    overall = time.time()
    grand_ok = grand_total = 0
    for (i, j) in pairs:
        sweep_root = out_root / f"{i}_{j}"
        print(f"\n=== pair {i}_{j} ===")
        try:
            n_ok, n_tot = sweep_one_pair(
                planner, base_world, urdf_fk, ee_link, obj_verts,
                obj_name=args.obj, hand=args.hand, i=i, j=j, h_cm=args.h_cm,
                xs=xs, tzs=tzs,
                place_x=args.place_x, place_y=args.place_y, place_tz=args.place_tz,
                max_seeds=args.max_seeds, sweep_root=sweep_root,
                skip_done=args.skip_done, obj_root=obj_root, version=args.version,
            )
            grand_ok += n_ok
            grand_total += n_tot
        except FileNotFoundError as e:
            print(f"[pair {i}_{j}] SKIP — {e}")

    total = time.time() - overall
    print(f"\n[sweep] all done: {grand_ok}/{grand_total} ok across {len(pairs)} pairs "
          f"({total:.1f}s, avg {total/max(len(pairs),1):.1f}s/pair)")


if __name__ == "__main__":
    main()
