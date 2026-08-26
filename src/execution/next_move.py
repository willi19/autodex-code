#!/usr/bin/env python3
"""Report where to move the object next to cover more scenes. Computes only.

Three ways to say where the object is now:

    # from a finished trial dir (pose_world.npy + C2R.npy)
    python src/execution/next_move.py --obj toothbrush_holder --hand inspire \
        --version v8 --trial ~/shared_data/AutoDex/experiment/v8/inspire/toothbrush_holder/20260811_120000

    # from explicit pose files
    python src/execution/next_move.py --obj toothbrush_holder --hand inspire \
        --version v8 --pose_world pose_world.npy --c2r C2R.npy

    # hypothetical: object resting on tabletop stem at (x, yaw), no perception
    python src/execution/next_move.py --obj toothbrush_holder --hand inspire \
        --version v8 --stem 002 --x 0.45 --yaw 30

Add --json for machine-readable output, --no_ik to skip the reposition search
(no planner load, ~10s faster, reorient + coverage numbers still reported).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from autodex.utils.next_move import plan_next_move, format_next_move
from autodex.utils.path import get_obj_root
from autodex.utils.reposition import place_T


def _pose_from_args(args) -> np.ndarray:
    """Object pose in the ROBOT frame, however the caller chose to specify it."""
    if args.stem is not None:
        root = get_obj_root(args.version)
        p = os.path.join(root, args.obj, "processed_data", "info",
                         "tabletop", f"{args.stem}.npy")
        T = np.load(p)
        if T.shape == (3, 3):
            M = np.eye(4); M[:3, :3] = T; T = M
        return place_T(T, args.x, np.deg2rad(args.yaw))

    pw, c2r = args.pose_world, args.c2r
    if args.trial:
        pw = pw or os.path.join(args.trial, "pose_world.npy")
        c2r = c2r or os.path.join(args.trial, "C2R.npy")
    if not pw or not os.path.exists(pw):
        sys.exit(f"pose_world.npy not found: {pw}")
    pose_world = np.load(pw)
    if not c2r or not os.path.exists(c2r):
        sys.exit(f"C2R.npy not found: {c2r}")
    return np.linalg.inv(np.load(c2r)) @ pose_world


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True)
    ap.add_argument("--hand", default="inspire")
    ap.add_argument("--version", default="v8")
    ap.add_argument("--h_cm", type=int, default=0,
                    help="reorient descent height folder to check cells in")
    ap.add_argument("--top_k", type=int, default=8,
                    help="how many still-useful grasps the reposition search IKs")

    ap.add_argument("--trial", default=None, help="trial dir holding pose_world.npy + C2R.npy")
    ap.add_argument("--pose_world", default=None)
    ap.add_argument("--c2r", default=None)
    ap.add_argument("--stem", default=None, help="hypothetical: tabletop stem, e.g. 002")
    ap.add_argument("--x", type=float, default=0.50, help="with --stem: object x")
    ap.add_argument("--yaw", type=float, default=0.0, help="with --stem: object yaw (deg)")

    ap.add_argument("--no_ik", action="store_true",
                    help="skip the reposition search (no planner load)")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    pose_robot = _pose_from_args(args)

    planner = None
    if not args.no_ik:
        from autodex.planner import GraspPlanner
        from autodex.planner.planner import _to_curobo_world
        from autodex.planner.obstacles import TABLE_CUBOID
        planner = GraspPlanner(hand=args.hand)
        # ik_pose_batch needs a solver; run_auto gets one as a side effect of
        # plan(), but standalone we have to build it. Table-only world: the
        # reposition search asks "can the arm reach this wrist pose", and the
        # scene obstacles are already baked into which grasps are candidates.
        planner._init_ik_solver(
            _to_curobo_world({"mesh": {}, "cuboid": {"table": TABLE_CUBOID}}))

    res = plan_next_move(args.obj, args.hand, args.version, pose_robot,
                         planner=planner, h_cm=args.h_cm, top_k=args.top_k)

    if args.json:
        print(json.dumps(res, indent=2, default=str))
    else:
        print(format_next_move(res))


if __name__ == "__main__":
    main()
