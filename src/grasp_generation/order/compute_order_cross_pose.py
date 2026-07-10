"""Cross-pose set cover.

For each (candidate, scene) pair, compute validity = collision-free AND
MuJoCo-passes (with the scene's tabletop-pose gravity direction). Then run
greedy set cover over the resulting (C, S) validity matrix.

A candidate generated for pose A may still be valid in a scene at pose B if
both (a) it's collision-free in B's obstacles and (b) it can hold the object
under B's gravity direction. Per-pose MuJoCo results are cached per candidate.

Usage:
    python src/grasp_generation/order/compute_order_cross_pose.py \
        --hand inspire_left --version v4 --obj donut
"""

import os
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "grasp_generation", "order"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "grasp_generation", "sim_filter"))

from autodex.utils.path import obj_path as default_obj_path, repo_dir, project_dir, get_scene_dir
from autodex.utils.conversion import cart2se3
from autodex.planner.planner import GraspPlanner, _to_curobo_world

from compute_order import setcover_order  # reuse greedy
from run_sim_filter import (  # type: ignore
    eval_single_grasp, INSPIRE_MIMIC_MAP, INSPIRE_F1_MIMIC_MAP,
    HAND_PATHS, OBJ_MASS,
)
from autodex.simulator.hand_object import MjHO


HAND_PLANNER_CFG = {
    "allegro": "allegro_floating.yml",
    "inspire": "inspire_floating.yml",
    "inspire_left": "inspire_left_floating.yml",
}

SCENE_TYPES_DEFAULT = ["wall", "shelf", "box"]


def load_candidates(candidate_root, obj_name):
    """Return list of dicts with grasp arrays + source scene/pose."""
    cands = []
    obj_root = os.path.join(candidate_root, obj_name)
    if not os.path.isdir(obj_root):
        return cands
    for scene_type in sorted(os.listdir(obj_root)):
        st_dir = os.path.join(obj_root, scene_type)
        if not os.path.isdir(st_dir):
            continue
        for sid in sorted(os.listdir(st_dir)):
            sid_dir = os.path.join(st_dir, sid)
            if not os.path.isdir(sid_dir):
                continue
            for cname in sorted(os.listdir(sid_dir)):
                c_dir = os.path.join(sid_dir, cname)
                if not os.path.isdir(c_dir):
                    continue
                try:
                    w = np.load(os.path.join(c_dir, "wrist_se3.npy"))
                    pre = np.load(os.path.join(c_dir, "pregrasp_pose.npy"))
                    grasp = np.load(os.path.join(c_dir, "grasp_pose.npy"))
                except FileNotFoundError:
                    continue
                cands.append({
                    "wrist_se3": w, "pregrasp": pre, "grasp": grasp,
                    "src_scene_type": scene_type, "src_scene_id": sid,
                    "name": cname,
                    "dir": c_dir,
                })
    return cands


def load_scenes(hand, obj_root_dir, obj_name, scene_types):
    """Return list of dicts with scene info (parsed JSON + pose_idx + gravity_dir)."""
    scenes = []
    for st in scene_types:
        d = get_scene_dir(hand, obj_name, st)
        if not os.path.isdir(d):
            continue
        for f in sorted(os.listdir(d)):
            if not f.endswith(".json"):
                continue
            sid = f[:-5]
            cfg = json.load(open(os.path.join(d, f)))
            scene_cfg = cfg["scene"]
            pose_idx = cfg.get("meta", {}).get("pose_idx", "0")
            obj_se3 = cart2se3(scene_cfg["mesh"]["target"]["pose"])
            gravity_dir = obj_se3[:3, :3].T @ np.array([0.0, 0.0, -1.0])
            scenes.append({
                "scene_type": st, "scene_id": sid, "pose_idx": pose_idx,
                "scene_cfg": scene_cfg, "obj_se3": obj_se3,
                "gravity_dir": gravity_dir,
            })
    return scenes


def collision_matrix(planner, candidates, scenes):
    """Return bool array (C, S): True = collision-FREE (valid by collision)."""
    C, S = len(candidates), len(scenes)
    out = np.zeros((C, S), dtype=bool)
    if C == 0 or S == 0:
        return out
    wrist_obj = np.stack([c["wrist_se3"] for c in candidates])  # (C, 4, 4)
    pre_arr = np.stack([c["pregrasp"] for c in candidates])
    for j, sc in enumerate(tqdm(scenes, desc="  coll/scene", leave=False)):
        world_cfg = _to_curobo_world(sc["scene_cfg"])
        wrist_world = np.einsum("ij,cjk->cik", sc["obj_se3"], wrist_obj)
        try:
            collided = planner._check_collision(world_cfg, wrist_world, pre_arr)
        except Exception:
            import traceback; traceback.print_exc()
            collided = np.ones(C, dtype=bool)
        out[:, j] = ~collided
    return out


def mujoco_matrix_per_pose(mj, candidates, scenes, hand):
    """For each (candidate, scene), run MuJoCo with scene's gravity_dir.
    Cache by pose_idx (gravity_dir is determined by pose_idx).

    Returns bool array (C, S).
    Also writes per-candidate cache file: {cand.dir}/cross_pose_eval.json
    """
    C, S = len(candidates), len(scenes)
    out = np.zeros((C, S), dtype=bool)
    if C == 0 or S == 0:
        return out

    # Group scenes by pose_idx (gravity is same within a pose_idx)
    pose_idxs = sorted({sc["pose_idx"] for sc in scenes})
    pose_to_gravity = {}
    for sc in scenes:
        pose_to_gravity.setdefault(sc["pose_idx"], sc["gravity_dir"])

    pose_to_scene_cols = {p: [j for j, sc in enumerate(scenes) if sc["pose_idx"] == p]
                          for p in pose_idxs}

    if hand in ("inspire", "inspire_left"):
        mimic_map = INSPIRE_MIMIC_MAP
    elif hand in ("inspire_f1", "inspire_f1_left"):
        mimic_map = INSPIRE_F1_MIMIC_MAP
    else:
        mimic_map = None
    apply_r_delta = (hand == "allegro")

    for ci, c in enumerate(tqdm(candidates, desc="  mujoco/cand", leave=False)):
        cache_path = os.path.join(c["dir"], "cross_pose_eval.json")
        cache = {}
        if os.path.exists(cache_path):
            try:
                cache = json.load(open(cache_path))
            except Exception:
                cache = {}

        for p in pose_idxs:
            if p in cache:
                ok = bool(cache[p])
            else:
                try:
                    succ, _ = eval_single_grasp(
                        mj, c["wrist_se3"], c["pregrasp"], c["grasp"],
                        mimic_map=mimic_map, apply_r_delta=apply_r_delta,
                        gravity_dir=pose_to_gravity[p],
                    )
                    ok = bool(succ)
                except Exception:
                    import traceback; traceback.print_exc()
                    ok = False
                cache[p] = ok
            for j in pose_to_scene_cols[p]:
                out[ci, j] = ok

        with open(cache_path, "w") as f:
            json.dump(cache, f)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", required=True, choices=list(HAND_PLANNER_CFG))
    parser.add_argument("--version", required=True)
    parser.add_argument("--obj", default=None)
    parser.add_argument("--obj_list_file", default=None)
    parser.add_argument("--obj_root", default=default_obj_path)
    parser.add_argument("--scene_types", nargs="+", default=SCENE_TYPES_DEFAULT)
    parser.add_argument("--output_root", default=None,
                        help="Default: REPO/order/{hand}/{version}/")
    args = parser.parse_args()

    candidate_root = os.path.join(REPO_ROOT, "candidates", args.hand, args.version)
    output_root = args.output_root or os.path.join(REPO_ROOT, "order", args.hand, args.version)

    if args.obj:
        obj_list = [args.obj]
    else:
        obj_list_file = args.obj_list_file or os.path.join(
            REPO_ROOT, "src", "grasp_generation", "obj_list.txt"
        )
        with open(obj_list_file) as f:
            obj_list = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    # Init planner once (shared across objs — needs world swap per call)
    hand_cfg = os.path.join(project_dir, "content", "configs", "robot",
                            HAND_PLANNER_CFG[args.hand])
    planner = GraspPlanner(hand_cfg_path=hand_cfg)

    for obj_name in tqdm(obj_list, desc="Objects"):
        print(f"\n=== {obj_name} ===")
        cands = load_candidates(candidate_root, obj_name)
        scenes = load_scenes(args.hand, args.obj_root, obj_name, args.scene_types)
        print(f"  candidates: {len(cands)}, scenes: {len(scenes)}")
        if not cands or not scenes:
            print("  skip (no candidates or no scenes)")
            continue

        # Per-obj MuJoCo session
        hand_xml = HAND_PATHS[args.hand]
        mj = MjHO(obj_name, hand_xml["path"], weld_body_name=hand_xml["weld_body"],
                  obj_mass=OBJ_MASS, debug_viewer=False,
                  obj_root_dir=args.obj_root)

        coll_ok = collision_matrix(planner, cands, scenes)
        muj_ok = mujoco_matrix_per_pose(mj, cands, scenes, args.hand)
        mj.close()

        valid = coll_ok & muj_ok   # (C, S)
        print(f"  valid matrix: {valid.shape}, total True: {int(valid.sum())}")

        # setcover_order expects (S, G) = (scenes, grasps) → transpose
        order, stats = setcover_order(valid.T)

        # Save: ordered list of candidates + per-step coverage stats + valid matrix
        out_dir = os.path.join(output_root, obj_name)
        os.makedirs(out_dir, exist_ok=True)
        ordered = []
        for rank, idx in enumerate(order):
            c = cands[idx]
            ordered.append({
                "rank": rank,
                "src_scene_type": c["src_scene_type"],
                "src_scene_id": c["src_scene_id"],
                "candidate_name": c["name"],
                "candidate_idx": int(idx),
                "newly_covered": stats[rank]["newly_covered_count"],
                "coverage_pct": stats[rank]["current_coverage_pct"],
                "cycle": stats[rank]["cycle"],
            })
        scene_keys = [f"{s['scene_type']}/{s['scene_id']}" for s in scenes]
        with open(os.path.join(out_dir, "setcover_order.json"), "w") as f:
            json.dump({
                "scene_keys": scene_keys,
                "n_candidates": len(cands),
                "n_scenes": len(scenes),
                "order": ordered,
            }, f, indent=2)
        np.save(os.path.join(out_dir, "valid_matrix.npy"), valid)
        print(f"  → {out_dir}/setcover_order.json")


if __name__ == "__main__":
    main()
