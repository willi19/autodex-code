"""Build scene_list.json (scene_type/scene_id per valid_array row).

compute_order.py drops scenes where no grasp is collision-free, so valid_array
rows lose their scene mapping. This script recomputes the scene loop and saves:
  - scene_list.json: list of {scene_type, scene_id} per row (full, no drop)
  - valid_array_full.npy: (S_all, G) — rows for dropped scenes are all-False

The existing valid_array.npy / setcover_order.json / stats.json are not touched.

Usage:
    python src/grasp_generation/order/build_scene_list.py --hand inspire_left --version v3 --obj smallbowl
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm

from autodex.utils.path import obj_path, repo_dir, project_dir, get_scene_dir
from autodex.utils.conversion import cart2se3
from autodex.planner.planner import GraspPlanner, _to_curobo_world


def load_grasps(candidate_root, obj_name):
    wrist_list, pregrasp_list, info_list = [], [], []
    root = os.path.join(candidate_root, obj_name)
    if not os.path.isdir(root):
        return [], np.array([]), np.array([])
    for scene_type in sorted(os.listdir(root)):
        st_dir = os.path.join(root, scene_type)
        if not os.path.isdir(st_dir):
            continue
        for scene_name in sorted(os.listdir(st_dir)):
            sd = os.path.join(st_dir, scene_name)
            if not os.path.isdir(sd):
                continue
            for grasp_name in sorted(os.listdir(sd)):
                gd = os.path.join(sd, grasp_name)
                if not os.path.isdir(gd):
                    continue
                wrist_list.append(np.load(os.path.join(gd, "wrist_se3.npy")))
                pregrasp_list.append(np.load(os.path.join(gd, "pregrasp_pose.npy")))
                info_list.append((obj_name, scene_type, scene_name, grasp_name))
    return info_list, np.array(wrist_list), np.array(pregrasp_list)


def build(obj_name, hand, version):
    candidate_root = os.path.join(repo_dir, "candidates", hand, version)
    order_root = os.path.join(repo_dir, "order", hand, version, obj_name)

    if not os.path.isdir(order_root):
        print(f"  No order dir for {obj_name}: {order_root}")
        return

    info_list, wrist_arr, pregrasp_arr = load_grasps(candidate_root, obj_name)
    if len(info_list) == 0:
        print(f"  No grasps for {obj_name}")
        return

    hand_cfg = os.path.join(project_dir, "content", "configs", "robot", f"{hand}_floating.yml")
    planner = GraspPlanner(hand_cfg_path=hand_cfg)

    scene_root = get_scene_dir(hand, obj_name)
    if not os.path.isdir(scene_root):
        print(f"  No scene dir: {scene_root} — skipping")
        return
    scene_list = []
    rows = []

    for scene_type in sorted(os.listdir(scene_root)):
        st_dir = os.path.join(scene_root, scene_type)
        if not os.path.isdir(st_dir):
            continue
        for scene_file in tqdm(sorted(os.listdir(st_dir)), desc=f"    {scene_type}", leave=False):
            if not scene_file.endswith(".json"):
                continue
            scene_cfg = json.load(open(os.path.join(st_dir, scene_file)))["scene"]

            obj_se3 = cart2se3(scene_cfg["mesh"]["target"]["pose"])
            wrist_world = np.einsum("ij,ajk->aik", obj_se3, wrist_arr)
            world_cfg = _to_curobo_world(scene_cfg)
            collided = planner._check_collision(world_cfg, wrist_world, pregrasp_arr)

            valid = ~collided
            scene_id = scene_file.replace(".json", "")
            scene_list.append({
                "scene_type": scene_type,
                "scene_id": scene_id,
                "obj_pose": scene_cfg["mesh"]["target"]["pose"],
                "n_valid": int(valid.sum()),
            })
            rows.append(valid)

    full = np.stack(rows, axis=0)
    print(f"  {obj_name}: full valid_array {full.shape}, kept rows {(full.any(axis=1)).sum()}")

    with open(os.path.join(order_root, "scene_list.json"), "w") as f:
        json.dump(scene_list, f, indent=2)
    np.save(os.path.join(order_root, "valid_array_full.npy"), full)
    print(f"  Saved scene_list.json + valid_array_full.npy in {order_root}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", default="inspire_left")
    parser.add_argument("--version", default="v3")
    parser.add_argument("--obj", default=None,
                        help="Single object. If omitted, runs all objects under order dir.")
    args = parser.parse_args()

    order_dir = os.path.join(repo_dir, "order", args.hand, args.version)
    if args.obj:
        objs = [args.obj]
    else:
        objs = sorted([d for d in os.listdir(order_dir)
                       if os.path.isdir(os.path.join(order_dir, d))])

    for obj in objs:
        print(f"\n{obj}:")
        build(obj, args.hand, args.version)

    print("\nDone!")
