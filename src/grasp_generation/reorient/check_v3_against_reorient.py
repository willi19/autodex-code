"""Quick diagnostic: take existing v8 grasps and check how many clear BOTH
table_i and table_j of each v8 reorient scene.

Wrist_se3 is object-frame in BODex output, so it transfers cleanly across
scenes that have the same object — just plug into the new scene's object pose.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from autodex.planner.planner import GraspPlanner
from autodex.utils.path import get_candidate_path, get_obj_root


def _xyzquat_to_T(p):
    T = np.eye(4)
    T[:3, 3] = p[:3]
    T[:3, :3] = R.from_quat([p[4], p[5], p[6], p[3]]).as_matrix()
    return T


def gather_v8(hand, obj, max_g):
    root = Path(get_candidate_path(hand)) / "v8" / obj
    if not root.is_dir():
        return None, None
    wrists, gposes = [], []
    for st in root.iterdir():
        if not st.is_dir():
            continue
        for s in st.iterdir():
            if not s.is_dir():
                continue
            for seed in s.iterdir():
                w = seed / "wrist_se3.npy"
                g = seed / "grasp_pose.npy"
                if w.exists() and g.exists():
                    wrists.append(np.load(w))
                    gposes.append(np.load(g))
                    if len(wrists) >= max_g:
                        return np.stack(wrists), np.stack(gposes)
    if not wrists:
        return None, None
    return np.stack(wrists), np.stack(gposes)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", default="inspire_left")
    parser.add_argument("--obj", required=True)
    parser.add_argument("--scene_type", default="reorient_0")
    parser.add_argument("--version", default="v8",
                        help="v8 candidate/tabletop asset contract")
    parser.add_argument("--max", type=int, default=500)
    args = parser.parse_args()
    if args.version != "v8":
        parser.error("this diagnostic supports only --version v8")

    wrists_obj, gposes = gather_v8(args.hand, args.obj, args.max)
    if wrists_obj is None:
        print(f"no v8 grasps for {args.hand}/{args.obj}")
        return
    print(f"collected {len(wrists_obj)} v8 grasps")

    planner = GraspPlanner(hand=args.hand)

    scene_dir = Path(get_obj_root(args.version)) / args.obj / "scene" / args.scene_type
    for sjson in sorted(scene_dir.glob("*.json")):
        scene = json.load(open(sjson))
        obj_T_world = _xyzquat_to_T(scene["scene"]["mesh"]["target"]["pose"])
        wrists_world = np.einsum("ij,njk->nik", obj_T_world, wrists_obj)

        world_cfg = {"cuboid": {
            k: scene["scene"]["cuboid"][k]
            for k in ("table_i", "table_j") if k in scene["scene"]["cuboid"]
        }}

        coll = planner._check_collision(world_cfg, wrists_world, gposes)
        n_pass = int((~coll).sum())
        print(f"  {sjson.stem:<12}  collision-free: {n_pass}/{len(wrists_obj)}")


if __name__ == "__main__":
    main()
