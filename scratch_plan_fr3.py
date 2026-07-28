#!/usr/bin/env python3
"""Planning-only smoke test: run the reorient 8-phase chain with the FR3 arm.

Arm/hand split (no executor, no perception):
  planner       = GraspPlanner(hand="fr3_inspire")   # FR3 arm + inspire right
  candidate hand = "inspire"                          # candidates/inspire/reset/...

Uses plan_reset.plan_one_cell unchanged (now DOF-generic). Prints per-phase
trajectory shapes and saves to outputs/reset_plans/fr3_inspire/...
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, "src/grasp_generation/reorient")
from plan_reset import (  # noqa: E402
    init_planner, load_fk_urdf, load_object_vertices, load_tabletop_pose,
    make_obj_pose, plan_one_cell, save_plan, phase_names_for,
    DEFAULT_PLACE_XY, APEX_Z, HAND_Z_MIN,
)

CAND_HAND = "inspire"       # where the grasp seeds live
PLANNER_HAND = "fr3_inspire"  # FR3 arm + inspire right hand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="apple")
    ap.add_argument("--i", type=int, default=0)
    ap.add_argument("--j", type=int, default=65)
    ap.add_argument("--h_cm", type=int, default=0)
    ap.add_argument("--pickup_x", type=float, default=0.40)
    ap.add_argument("--pickup_tz", type=float, default=0.0)
    ap.add_argument("--place_x", type=float, default=DEFAULT_PLACE_XY[0])
    ap.add_argument("--place_y", type=float, default=DEFAULT_PLACE_XY[1])
    ap.add_argument("--max_seeds", type=int, default=30)
    args = ap.parse_args()

    h_m = args.h_cm / 100.0
    Ti = load_tabletop_pose(args.obj, args.i)
    Tj = load_tabletop_pose(args.obj, args.j)
    T_obj_start = make_obj_pose(Ti, np.array([args.pickup_x, 0.0, Ti[2, 3]]),
                               args.pickup_tz)
    place_search_tzs = list(np.arange(0.0, 360.0, 30.0))

    print(f"[fr3-plan] obj={args.obj} cell={args.i}_{args.j} h={args.h_cm}cm "
          f"planner={PLANNER_HAND} cand={CAND_HAND}")
    t0 = time.time()
    planner, base_world = init_planner(PLANNER_HAND)
    urdf_fk, ee_link = load_fk_urdf(PLANNER_HAND)
    obj_verts = load_object_vertices(args.obj)
    print(f"[fr3-plan] warmup {time.time()-t0:.1f}s | init_state dim={len(planner._init_state)} "
          f"| {len(obj_verts)} verts")

    result = plan_one_cell(
        planner, obj_name=args.obj, hand=CAND_HAND,
        h_cm=args.h_cm, i=args.i, j=args.j,
        T_obj_start=T_obj_start, T_obj_end=None,
        place_xy=(args.place_x, args.place_y), place_search_tzs=place_search_tzs,
        base_world=base_world, max_seeds=args.max_seeds,
        urdf_fk=urdf_fk, ee_link=ee_link, obj_verts=obj_verts,
    )

    if result is None or result.get("status") != "ok":
        print(f"[fr3-plan] NO success — fail_counts={result.get('fail_counts') if result else None}")
        return

    trajs = result["trajs"]
    print(f"\n[fr3-plan] SUCCESS  seed={result['seed_id']} (idx {result['seed_idx']})")
    print(f"  place_tz_used={result.get('place_tz_used')}")
    print("  phases:")
    for name, tr in trajs.items():
        print(f"    {name:14s} shape={np.asarray(tr).shape}")

    out = (Path("outputs/reset_plans") / PLANNER_HAND / args.obj
           / f"reorient_{args.h_cm}" / f"{args.i}_{args.j}"
           / f"x{args.pickup_x:.2f}_tz{int(round(args.pickup_tz)):03d}"
           / result["seed_id"])
    ptz_used = result.get("place_tz_used")
    # Field names/shape must match what view_reset.py reads.
    save_plan(out, trajs, {
        "obj_name": args.obj, "hand": PLANNER_HAND,
        "i": args.i, "j": args.j, "h_cm": args.h_cm,
        "pickup_x": args.pickup_x, "pickup_tz": args.pickup_tz,
        "place_x": args.place_x, "place_y": args.place_y,
        "place_tz": ptz_used,
        "seed_id": result["seed_id"], "phase_names": phase_names_for(args.h_cm),
        "wrist_se3_obj": result["wrist_se3_obj"].tolist(),
        "T_obj_start": T_obj_start.tolist(),
        "T_obj_apex_i": result["T_obj_apex_i"].tolist(),
        "T_obj_apex_j": result["T_obj_apex_j"].tolist(),
        "T_obj_end": result["T_obj_end"].tolist(),
        "apex_z": APEX_Z, "hand_z_min": HAND_Z_MIN,
        "cand_hand": CAND_HAND,
    })
    print(f"\n[fr3-plan] saved -> {out}")
    print(f"[fr3-plan] view: python src/grasp_generation/reorient/view_reset.py "
          f"--plan_dir {out} --port 8090")


if __name__ == "__main__":
    main()
