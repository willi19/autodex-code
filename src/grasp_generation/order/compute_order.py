"""Compute greedy set cover ordering for grasp candidates.

For each object, loads all candidates, checks collision-free reachability
across all scenes, then ranks grasps by greedy set cover.

Usage:
    python src/grasp_generation/order/compute_order.py
    python src/grasp_generation/order/compute_order.py --hand allegro --version v3
    python src/grasp_generation/order/compute_order.py --obj attached_container
"""

import os
import json
import argparse
import numpy as np
from tqdm import tqdm

from autodex.utils.path import obj_path, candidate_path, get_scene_dir
from autodex.utils.conversion import cart2se3, se32action
from autodex.planner.planner import GraspPlanner, _to_curobo_world

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def load_grasp_data(candidate_root, obj_name):
    """Load all candidates for an object."""
    wrist_se3_list = []
    pregrasp_pose_list = []
    grasp_info_list = []

    root_dir = os.path.join(candidate_root, obj_name)
    if not os.path.isdir(root_dir):
        return [], np.array([]), np.array([])

    for scene_type in sorted(os.listdir(root_dir)):
        scene_type_dir = os.path.join(root_dir, scene_type)
        if not os.path.isdir(scene_type_dir):
            continue
        for scene_name in sorted(os.listdir(scene_type_dir)):
            scene_dir = os.path.join(scene_type_dir, scene_name)
            if not os.path.isdir(scene_dir):
                continue
            for grasp_name in sorted(os.listdir(scene_dir)):
                grasp_dir = os.path.join(scene_dir, grasp_name)
                if not os.path.isdir(grasp_dir):
                    continue

                wrist_se3 = np.load(os.path.join(grasp_dir, "wrist_se3.npy"))
                pregrasp = np.load(os.path.join(grasp_dir, "pregrasp_pose.npy"))

                grasp_info_list.append((obj_name, scene_type, scene_name, grasp_name))
                wrist_se3_list.append(wrist_se3)
                pregrasp_pose_list.append(pregrasp)

    if not wrist_se3_list:
        return [], np.array([]), np.array([])
    return grasp_info_list, np.array(wrist_se3_list), np.array(pregrasp_pose_list)


def setcover_order(valid_array):
    """Greedy set cover with robust reset.

    Args:
        valid_array: (S, G) bool — valid_array[s, g] = True if grasp g is collision-free in scene s.

    Returns:
        order: list of grasp indices in set cover order.
        stats: list of dicts with per-step coverage info.
    """
    S, G = valid_array.shape
    uncovered = np.ones(S, dtype=bool)
    available = np.ones(G, dtype=bool)
    order = []
    stats = []
    cycle = 1

    for _ in tqdm(range(G), desc="  Set cover", leave=False):
        cover_counts = np.sum(valid_array[uncovered][:, available], axis=0)

        if np.all(cover_counts == 0):
            uncovered = np.ones(S, dtype=bool)
            cycle += 1
            cover_counts = np.sum(valid_array[uncovered][:, available], axis=0)

            if np.all(cover_counts == 0):
                total_counts = np.sum(valid_array[:, available], axis=0)
                best_local = np.argmax(total_counts)
            else:
                best_local = np.argmax(cover_counts)
        else:
            best_local = np.argmax(cover_counts)

        best_idx = np.flatnonzero(available)[best_local]

        newly_covered = int(np.sum(valid_array[:, best_idx] & uncovered))
        order.append(int(best_idx))
        available[best_idx] = False
        uncovered &= ~valid_array[:, best_idx]
        coverage_pct = float((1 - np.sum(uncovered) / S) * 100)

        stats.append({
            "grasp_idx": int(best_idx),
            "newly_covered_count": newly_covered,
            "current_coverage_pct": coverage_pct,
            "cycle": cycle,
        })

    return order, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", type=str, default="allegro")
    parser.add_argument("--version", type=str, default="v3")
    parser.add_argument("--obj", type=str, default=None,
                        help="Single object. If omitted, reads obj_list.txt")
    parser.add_argument("--candidate_root", type=str, default=None,
                        help="Override candidate root directory")
    parser.add_argument("--output_root", type=str, default=None,
                        help="Override output root directory")
    parser.add_argument("--obj_root_dir", type=str, default=None,
                        help="Override object root dir (default: paradex)")
    parser.add_argument("--obj_list_file", type=str, default=None,
                        help="Object list file (default: src/grasp_generation/obj_list.txt)")
    args = parser.parse_args()

    # Object list
    if args.obj:
        obj_list = [args.obj]
    else:
        obj_list_file = args.obj_list_file or os.path.join(
            REPO_ROOT, "src", "grasp_generation", "obj_list.txt"
        )
        with open(obj_list_file) as f:
            obj_list = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    obj_root = args.obj_root_dir or obj_path

    from autodex.utils.path import repo_dir
    candidate_root = args.candidate_root or os.path.join(repo_dir, "candidates", args.hand, args.version)
    output_root = args.output_root or os.path.join(repo_dir, "order", args.hand, args.version)

    from autodex.utils.path import project_dir
    hand_cfg_map = {
        "allegro": os.path.join(project_dir, "content", "configs", "robot", "allegro_floating.yml"),
        "inspire": os.path.join(project_dir, "content", "configs", "robot", "inspire_floating.yml"),
        "inspire_left": os.path.join(project_dir, "content", "configs", "robot", "inspire_left_floating.yml"),
    }
    hand_cfg = hand_cfg_map.get(args.hand)
    planner = GraspPlanner(hand_cfg_path=hand_cfg)

    for obj_name in tqdm(obj_list, desc="Objects"):
        print(f"\n{obj_name}:")

        order_path = os.path.join(output_root, obj_name, "setcover_order.json")
        if os.path.exists(order_path):
            print("  Already processed, skipping.")
            continue

        # Load candidates
        grasp_info_list, wrist_se3_list, pregrasp_list = load_grasp_data(candidate_root, obj_name)
        if len(grasp_info_list) == 0:
            print("  No candidates found, skipping.")
            continue
        print(f"  Loaded {len(grasp_info_list)} candidates")

        # Build valid_array: for each scene, check collision-free reachability
        scene_root = get_scene_dir(args.hand, obj_name)
        valid_array = []

        for scene_type in tqdm(sorted(os.listdir(scene_root)), desc="  Scene types", leave=False):
            scene_type_dir = os.path.join(scene_root, scene_type)
            if not os.path.isdir(scene_type_dir):
                continue
            for scene_file in tqdm(sorted(os.listdir(scene_type_dir)), desc=f"    {scene_type}", leave=False):
                if not scene_file.endswith(".json"):
                    continue
                scene_cfg = json.load(open(os.path.join(scene_type_dir, scene_file)))["scene"]

                obj_se3 = cart2se3(scene_cfg["mesh"]["target"]["pose"])
                wrist_world = np.einsum("ij,ajk->aik", obj_se3, wrist_se3_list)

                world_cfg = _to_curobo_world(scene_cfg)
                collided = planner._check_collision(world_cfg, wrist_world, pregrasp_list)

                if np.sum(~collided) == 0:
                    continue

                valid_array.append(~collided)

        if not valid_array:
            print("  No reachable scenes, skipping.")
            continue

        valid_array = np.array(valid_array)
        print(f"  Valid array: {valid_array.shape} (scenes x grasps)")

        # Greedy set cover
        cover_order, cover_stats = setcover_order(valid_array)

        # Save order
        ordered_info = []
        for i, idx in enumerate(cover_order):
            info = list(grasp_info_list[idx])
            info.append(idx)
            ordered_info.append(info)
            cover_stats[i]["scene_info"] = info

        os.makedirs(os.path.join(output_root, obj_name), exist_ok=True)
        with open(order_path, "w") as f:
            json.dump(ordered_info, f, indent=2)
        np.save(os.path.join(output_root, obj_name, "valid_array.npy"), valid_array)

        # Save stats
        stats_path = os.path.join(output_root, obj_name, "stats.json")
        with open(stats_path, "w") as f:
            json.dump({
                "object": obj_name,
                "total_scenes": int(valid_array.shape[0]),
                "total_grasps": int(valid_array.shape[1]),
                "steps": cover_stats,
            }, f, indent=2)

        print(f"  Saved to {order_path}")

    print("\nDone!")