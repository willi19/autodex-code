"""Generate shelf/wall/packed scenes with adaptive gap.

For each (scene_type, tabletop_pose, face_combo, z_rot) tuple, walk
GAP_SWEEP in ascending order and keep the smallest gap whose scene admits
at least one collision-free grasp from the candidate pool. Tuples for
which no gap up to 0.08 m works are dropped — those scenes are infeasible.

Run after BODex candidates exist; otherwise there is no grasp pool to
validate against.

Usage:
    python src/scene_generation/adaptive_gap_scene.py --hand inspire_left --version v3
    python src/scene_generation/adaptive_gap_scene.py --hand allegro --version v3 --obj donut
"""

import os
import json
import shutil
import argparse
from itertools import product

import numpy as np
import tqdm

from autodex.utils.path import obj_path as default_obj_path, repo_dir, project_dir, get_scene_dir
from autodex.planner.planner import GraspPlanner, _to_curobo_world

import sys
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "grasp_generation", "order"))
from compute_order import load_grasp_data  # noqa: E402

from generate_scene import (  # noqa: E402
    get_shelf_scene, get_wall_scene, get_packed_scene,
)


GAP_SWEEP = [0.0, 0.02, 0.04, 0.06, 0.08]
VERTICAL_THRESH = 0.95  # cos(~18 deg): symmetry axis treated as vertical
Z_ROTS_DEFAULT = [0, 72, 144, 216, 288]


HAND_CFG = {
    "allegro": "allegro_floating.yml",
    "inspire": "inspire_floating.yml",
    "inspire_left": "inspire_left_floating.yml",
}

AXIS_VEC = {"x": np.array([1, 0, 0]), "y": np.array([0, 1, 0]), "z": np.array([0, 0, 1])}


def load_symmetry_registry():
    p = os.path.join(os.path.dirname(__file__), "symmetry.json")
    with open(p) as f:
        reg = json.load(f)
    reg.pop("_comment", None)
    return reg


def z_rots_for_pose(obj_name, tabletop_pose, symmetry_reg):
    """If object has revolute symmetry axis that aligns with world z under
    this tabletop pose, collapse z_rots to [0]."""
    info = symmetry_reg.get(obj_name)
    if info is None or info.get("type") != "revolute":
        return Z_ROTS_DEFAULT
    axis_obj = AXIS_VEC[info["axis"]]
    axis_world = tabletop_pose[:3, :3] @ axis_obj
    if abs(axis_world[2]) >= VERTICAL_THRESH:
        return [0]
    return Z_ROTS_DEFAULT


def _scene_valid(planner, scene_dict, wrist_se3_list, pregrasp_list, tabletop_pose):
    obj_se3 = tabletop_pose
    wrist_world = np.einsum("ij,ajk->aik", obj_se3, wrist_se3_list)
    world_cfg = _to_curobo_world(scene_dict)
    collided = planner._check_collision(world_cfg, wrist_world, pregrasp_list)
    return np.sum(~collided) > 0


def _backup_and_clear(out_dir):
    """First run: rename out_dir → out_dir+'_prev' to preserve baseline.
    Subsequent runs: keep _prev intact, drop current out_dir for clean regen.
    """
    if os.path.isdir(out_dir):
        prev = out_dir + "_prev"
        if not os.path.isdir(prev):
            os.rename(out_dir, prev)
        else:
            shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)


def _adapt_gap(builder, scene_kwargs, planner, wrist_se3_list, pregrasp_list, tabletop_pose):
    """Try GAP_SWEEP in order; return (scene_dict, gap) for first feasible, else None."""
    for gap in GAP_SWEEP:
        scene = builder(gap=gap, **scene_kwargs)
        if scene is None:
            continue
        if _scene_valid(planner, scene, wrist_se3_list, pregrasp_list, tabletop_pose):
            return scene, gap
    return None


def gen_wall(obj_name, hand, obj_root, planner, wrist_se3_list, pregrasp_list, symmetry_reg):
    obj_dir = os.path.join(obj_root, obj_name)
    out_dir = get_scene_dir(hand, obj_name, "wall")
    _backup_and_clear(out_dir)

    tabletop_pose_path = os.path.join(obj_dir, "processed_data", "info", "tabletop")
    obb_info = json.load(open(os.path.join(obj_dir, "processed_data", "info", "simplified.json")))

    scene_cnt = 0
    skipped = 0
    for fname in sorted(os.listdir(tabletop_pose_path)):
        pose_idx = fname.split(".")[0]
        pose = np.load(os.path.join(tabletop_pose_path, fname))

        z_rots = z_rots_for_pose(obj_name, pose, symmetry_reg)
        for z_rot in z_rots:
            def builder(gap, _pose=pose, _z_rot=z_rot):
                return get_wall_scene(obj_name, _pose, obb_info, _z_rot, gap)
            res = _adapt_gap(builder, {}, planner, wrist_se3_list, pregrasp_list, pose)
            if res is None:
                skipped += 1
                continue
            scene, gap = res
            scene_cfg = {
                "scene": scene,
                "meta": {
                    "pose_idx": pose_idx,
                    "param": {"z_rotation_deg": z_rot, "gap": gap},
                },
            }
            with open(os.path.join(out_dir, f"{scene_cnt}.json"), "w") as f:
                json.dump(scene_cfg, f, indent=2)
            scene_cnt += 1
    return scene_cnt, skipped


def gen_shelf(obj_name, hand, obj_root, planner, wrist_se3_list, pregrasp_list, symmetry_reg):
    obj_dir = os.path.join(obj_root, obj_name)
    out_dir = get_scene_dir(hand, obj_name, "shelf")
    _backup_and_clear(out_dir)

    tabletop_pose_path = os.path.join(obj_dir, "processed_data", "info", "tabletop")
    obb_info = json.load(open(os.path.join(obj_dir, "processed_data", "info", "simplified.json")))

    face_combos = [
        (up, side, back)
        for up, side, back in product([True, False], [True, False], [True])
        if (up or side)
    ]

    scene_cnt = 0
    skipped = 0
    for fname in sorted(os.listdir(tabletop_pose_path)):
        pose_idx = fname.split(".")[0]
        pose = np.load(os.path.join(tabletop_pose_path, fname))

        z_rots = z_rots_for_pose(obj_name, pose, symmetry_reg)
        for z_rot, (up, side, back) in product(z_rots, face_combos):
            def builder(gap, _pose=pose, _z=z_rot, _u=up, _s=side, _b=back):
                return get_shelf_scene(
                    obj_name, _pose, obb_info, _z, gap,
                    up=_u, side=_s, back=_b,
                )
            res = _adapt_gap(builder, {}, planner, wrist_se3_list, pregrasp_list, pose)
            if res is None:
                skipped += 1
                continue
            scene, gap = res
            scene_cfg = {
                "scene": scene,
                "meta": {
                    "pose_idx": pose_idx,
                    "param": {
                        "z_rotation_deg": z_rot, "gap": gap,
                        "up": up, "side": side, "back": back,
                    },
                },
            }
            with open(os.path.join(out_dir, f"{scene_cnt}.json"), "w") as f:
                json.dump(scene_cfg, f, indent=2)
            scene_cnt += 1
    return scene_cnt, skipped


def gen_packed(obj_name, hand, obj_root, planner, wrist_se3_list, pregrasp_list, symmetry_reg):  # noqa: ARG001
    obj_dir = os.path.join(obj_root, obj_name)
    out_dir = get_scene_dir(hand, obj_name, "packed")
    _backup_and_clear(out_dir)

    tabletop_pose_path = os.path.join(obj_dir, "processed_data", "info", "tabletop")
    obb_info = json.load(open(os.path.join(obj_dir, "processed_data", "info", "simplified.json")))

    side_combos = [
        c for c in product([False, True], repeat=4)
        if any(c)
    ]  # (front, right, left, back), at least one True

    scene_cnt = 0
    skipped = 0
    for fname in sorted(os.listdir(tabletop_pose_path)):
        pose_idx = fname.split(".")[0]
        pose = np.load(os.path.join(tabletop_pose_path, fname))

        for front, right, left, back in side_combos:
            def builder(gap, _pose=pose, _f=front, _r=right, _l=left, _b=back):
                return get_packed_scene(
                    obj_name, _pose, obb_info, _f, _r, _l, _b, gap,
                )
            res = _adapt_gap(builder, {}, planner, wrist_se3_list, pregrasp_list, pose)
            if res is None:
                skipped += 1
                continue
            scene, gap = res
            scene_cfg = {
                "scene": scene,
                "meta": {
                    "pose_idx": pose_idx,
                    "param": {
                        "front": front, "right": right, "left": left, "back": back,
                        "gap": gap,
                    },
                },
            }
            with open(os.path.join(out_dir, f"{scene_cnt}.json"), "w") as f:
                json.dump(scene_cfg, f, indent=2)
            scene_cnt += 1
    return scene_cnt, skipped


SCENE_GENS = {
    "wall": gen_wall,
    "shelf": gen_shelf,
    "packed": gen_packed,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", required=True, choices=list(HAND_CFG))
    parser.add_argument("--version", required=True, help="Candidate version (e.g. v3)")
    parser.add_argument("--obj", default=None, help="Single object name")
    parser.add_argument("--obj_list_file", default=None)
    parser.add_argument("--scenes", nargs="+", default=["shelf", "wall", "packed"],
                        choices=list(SCENE_GENS))
    parser.add_argument("--obj_root", default=default_obj_path)
    parser.add_argument("--candidate_root", default=None)
    args = parser.parse_args()

    candidate_root = args.candidate_root or os.path.join(
        repo_dir, "candidates", args.hand, args.version
    )

    if args.obj:
        obj_list = [args.obj]
    else:
        obj_list_file = args.obj_list_file or os.path.join(
            REPO_ROOT, "src", "grasp_generation", "obj_list.txt"
        )
        with open(obj_list_file) as f:
            obj_list = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    hand_cfg = os.path.join(project_dir, "content", "configs", "robot", HAND_CFG[args.hand])
    planner = GraspPlanner(hand_cfg_path=hand_cfg)
    symmetry_reg = load_symmetry_registry()

    summary = {}
    for obj_name in tqdm.tqdm(obj_list, desc="Objects"):
        grasp_info_list, wrist_se3_list, pregrasp_list = load_grasp_data(candidate_root, obj_name)
        if len(grasp_info_list) == 0:
            print(f"{obj_name}: no candidates, skip")
            continue

        per_obj = {}
        for scene_type in args.scenes:
            kept, skipped = SCENE_GENS[scene_type](
                obj_name, args.hand, args.obj_root, planner, wrist_se3_list, pregrasp_list, symmetry_reg,
            )
            per_obj[scene_type] = (kept, skipped)
            print(f"{obj_name}/{scene_type}: kept={kept}, skipped={skipped}")
        summary[obj_name] = per_obj

    print("\n=== Summary ===")
    for obj, st in summary.items():
        for stype, (k, s) in st.items():
            print(f"{obj} {stype}: kept={k}, skipped={s}")
