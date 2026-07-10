"""Scene generation with correct OBB center handling.

The original MeshProcess/src/generate_scene.py had a bug: OBB corners were computed
relative to the mesh origin (pose[:3, 3]) instead of the actual OBB center
(pose[:3,:3] @ obb_transform[:3,3] + pose[:3,3]). This caused obstacles (walls,
shelves, packed objects) to be misplaced whenever the OBB center didn't coincide
with the mesh origin.

Fix: always transform obb_transform[:3,3] to world frame before computing corners.
"""

import os
import json
from itertools import product

import numpy as np
import transforms3d
import tqdm

from autodex.utils.path import get_scene_dir

obj_path = "/home/mingi/shared_data/AutoDex/object/paradex"
# Override via --obj_path CLI flag (set in __main__).
# Scenes are written per-hand (get_scene_dir); HAND is rebound in __main__.
HAND = "allegro"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _obb_corners_world(obb_info: dict, pose: np.ndarray) -> np.ndarray:
    """Compute 8 OBB corners in world frame, accounting for OBB center offset.

    Args:
        obb_info: dict with 'obb' (extents) and 'obb_transform' (4x4).
        pose: 4x4 object-to-world transform.

    Returns:
        (8, 3) array of corner positions in world frame.
    """
    obb_tf = np.array(obb_info["obb_transform"])
    R_obb = obb_tf[:3, :3]
    t_obb = obb_tf[:3, 3]  # OBB center in mesh-local frame
    half_ext = np.array(obb_info["obb"]) / 2.0

    R_w = pose[:3, :3]
    t_w = pose[:3, 3]

    # OBB axes and center in world frame
    axes_w = R_w @ R_obb
    center_w = R_w @ t_obb + t_w

    corners = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                local = np.array([sx, sy, sz]) * half_ext
                corners.append(axes_w @ local + center_w)
    return np.array(corners)


def _obb_axes_world(obb_info: dict, pose: np.ndarray):
    """Return OBB axes in world frame and extents."""
    obb_tf = np.array(obb_info["obb_transform"])
    R_obb = obb_tf[:3, :3]
    extents = np.array(obb_info["obb"])
    axes_w = pose[:3, :3] @ R_obb  # columns are OBB axes in world
    return axes_w, extents


def get_mesh_dict(obj_name: str, pose: np.ndarray) -> dict:
    return {
        "scale": [1.0, 1.0, 1.0],
        "pose": [
            pose[0, 3], pose[1, 3], pose[2, 3],
            *transforms3d.quaternions.mat2quat(pose[:3, :3]),
        ],
        "file_path": os.path.join(
            obj_path, obj_name, "processed_data", "mesh", "simplified.obj"
        ),
        "urdf_path": os.path.join(
            obj_path, obj_name, "processed_data", "urdf", "coacd.urdf"
        ),
    }


def get_tabletop_scene(obj_name: str, tabletop_pose: np.ndarray) -> dict:
    ret = {
        "mesh": {},
        "cuboid": {
            "table": {
                "dims": [2.0, 2.0, 0.2],
                "pose": [0.0, 0.0, -0.1, 1, 0, 0, 0.0],
            }
        },
    }
    ret["mesh"]["target"] = get_mesh_dict(obj_name, tabletop_pose)
    return ret


# ---------------------------------------------------------------------------
# Scene generators (with OBB center fix)
# ---------------------------------------------------------------------------

def get_packed_scene(obj_name, tabletop_pose, obb_info, front, right, left, back, gap):
    axes_w, extents = _obb_axes_world(obb_info, tabletop_pose)
    v1, v2, v3 = axes_w[:, 0], axes_w[:, 1], axes_w[:, 2]
    l1, l2, l3 = extents

    # Find most-vertical axis
    z_components = [abs(v1[2]), abs(v2[2]), abs(v3[2])]
    vertical_idx = int(np.argmax(z_components))

    all_axes = [v1, v2, v3]
    all_extents = [l1, l2, l3]

    horizontal_axes = [
        np.array([all_axes[i][0], all_axes[i][1], 0])
        for i in range(3) if i != vertical_idx
    ]
    horizontal_extents = [
        (all_extents[i] * (1 + z_components[i] / z_components[vertical_idx])) + gap
        for i in range(3) if i != vertical_idx
    ]

    # FIX: compute OBB center offset in world frame for correct packing direction
    obb_tf = np.array(obb_info["obb_transform"])
    t_obb = obb_tf[:3, 3]
    obb_center_offset = tabletop_pose[:3, :3] @ t_obb  # offset from mesh origin

    scene = get_tabletop_scene(obj_name, tabletop_pose)
    if front:
        new_pose = tabletop_pose.copy()
        new_pose[:3, 3] += horizontal_axes[0] * horizontal_extents[0]
        scene["mesh"]["obs_front"] = get_mesh_dict(obj_name, new_pose)
    if right:
        new_pose = tabletop_pose.copy()
        new_pose[:3, 3] += horizontal_axes[1] * horizontal_extents[1]
        scene["mesh"]["obs_right"] = get_mesh_dict(obj_name, new_pose)
    if left:
        new_pose = tabletop_pose.copy()
        new_pose[:3, 3] -= horizontal_axes[1] * horizontal_extents[1]
        scene["mesh"]["obs_left"] = get_mesh_dict(obj_name, new_pose)
    if back:
        new_pose = tabletop_pose.copy()
        new_pose[:3, 3] -= horizontal_axes[0] * horizontal_extents[0]
        scene["mesh"]["obs_back"] = get_mesh_dict(obj_name, new_pose)
    return scene


def get_wall_scene(obj_name, tabletop_pose, obb_info, z_rotation_deg, gap):
    angle_rad = np.radians(z_rotation_deg)
    z_rotation = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0, 0],
        [np.sin(angle_rad),  np.cos(angle_rad), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    rotated_pose = z_rotation @ tabletop_pose

    # FIX: use _obb_corners_world which accounts for OBB center offset
    corners_world = _obb_corners_world(obb_info, rotated_pose)

    min_y = corners_world[:, 1].min()
    wall_y = min_y - gap

    scene = get_tabletop_scene(obj_name, rotated_pose)
    scene["cuboid"]["wall"] = {
        "dims": [1.0, 0.02, 1.0],
        "pose": [0.0, wall_y - 0.01, 0.5, 1, 0, 0, 0],
    }
    return scene


def get_shelf_scene(obj_name, tabletop_pose, obb_info,
                    z_rotation_deg, gap, up=True, side=True, back=True):
    angle = np.radians(z_rotation_deg)
    z_rot = np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [np.sin(angle),  np.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    pose = z_rot @ tabletop_pose

    # FIX: use _obb_corners_world which accounts for OBB center offset
    corners = _obb_corners_world(obb_info, pose)

    THICK = 0.1

    min_x, max_x = corners[:, 0].min(), corners[:, 0].max()
    min_y, max_y = corners[:, 1].min(), corners[:, 1].max()
    max_z = corners[:, 2].max()

    up_z = max_z + gap + THICK / 2
    wall_height = up_z - THICK / 2
    wall_center_z = wall_height / 2

    y_min_wall = min_y - gap - THICK
    y_max_wall = max_y + gap
    wall_len_y = y_max_wall - y_min_wall

    scene = get_tabletop_scene(obj_name, pose)

    # back wall
    back_x_len = (max_x - min_x) + 2 * gap + 2 * THICK
    back_center_x = (max_x + min_x) / 2

    if back:
        scene["cuboid"]["back"] = {
            "dims": [back_x_len, THICK, wall_height],
            "pose": [
                back_center_x,
                y_min_wall + THICK / 2,
                wall_center_z,
                1, 0, 0, 0,
            ],
        }

    # side walls
    if side:
        scene["cuboid"]["side_pos"] = {
            "dims": [THICK, wall_len_y, wall_height],
            "pose": [
                max_x + gap + THICK / 2,
                (y_min_wall + y_max_wall) / 2,
                wall_center_z,
                1, 0, 0, 0,
            ],
        }
        scene["cuboid"]["side_neg"] = {
            "dims": [THICK, wall_len_y, wall_height],
            "pose": [
                min_x - gap - THICK / 2,
                (y_min_wall + y_max_wall) / 2,
                wall_center_z,
                1, 0, 0, 0,
            ],
        }

    # ceiling
    ceil_x_len = (max_x - min_x) + 2 * gap + 2 * THICK
    ceil_y_len = wall_len_y

    if up:
        scene["cuboid"]["up"] = {
            "dims": [ceil_x_len, ceil_y_len, THICK],
            "pose": [
                (max_x + min_x) / 2,
                (y_min_wall + y_max_wall) / 2,
                up_z,
                1, 0, 0, 0,
            ],
        }

    return scene


def get_box_scene(obj_name, tabletop_pose, height_offset):
    import trimesh

    mesh_path = os.path.join(
        obj_path, obj_name, "processed_data", "mesh", "simplified.obj"
    )
    mesh = trimesh.load(mesh_path)

    vertices_world = (tabletop_pose[:3, :3] @ mesh.vertices.T).T + tabletop_pose[:3, 3]
    max_z = vertices_world[:, 2].max()

    floor_z = max_z - height_offset
    if floor_z < 0:
        return None

    scene = get_tabletop_scene(obj_name, tabletop_pose)
    scene["cuboid"]["floor"] = {
        "dims": [2.0, 2.0, 0.2],
        "pose": [0.0, 0.0, floor_z - 0.1, 1, 0, 0, 0],
    }
    return scene


# ---------------------------------------------------------------------------
# Save functions
# ---------------------------------------------------------------------------

def save_float_scene(obj_name):
    obj_dir = os.path.join(obj_path, obj_name)
    out_path = get_scene_dir(HAND, obj_name)
    os.makedirs(os.path.join(out_path, "float"), exist_ok=True)

    scene_cfg = {
        "scene": {
            "mesh": {"target": get_mesh_dict(obj_name, np.eye(4))},
            "cuboid": {},
        },
        "meta": {"param": {}},
    }
    with open(os.path.join(out_path, "float", "0.json"), "w") as f:
        json.dump(scene_cfg, f, indent=2)


def save_tabletop_scene(obj_name):
    obj_dir = os.path.join(obj_path, obj_name)
    out_path = get_scene_dir(HAND, obj_name)
    tabletop_pose_path = os.path.join(obj_dir, "processed_data", "info", "tabletop")
    os.makedirs(os.path.join(out_path, "table"), exist_ok=True)

    for i, fname in enumerate(
        tqdm.tqdm(os.listdir(tabletop_pose_path), desc=f"table/{obj_name}")
    ):
        pose = np.load(os.path.join(tabletop_pose_path, fname))
        scene_cfg = {
            "scene": get_tabletop_scene(obj_name, pose),
            "meta": {"pose_idx": fname.split(".")[0], "param": {}},
        }
        with open(os.path.join(out_path, "table", f"{i}.json"), "w") as f:
            json.dump(scene_cfg, f, indent=2)


def save_packed_scene(obj_name):
    obj_dir = os.path.join(obj_path, obj_name)
    out_path = get_scene_dir(HAND, obj_name)
    tabletop_pose_path = os.path.join(obj_dir, "processed_data", "info", "tabletop")
    os.makedirs(os.path.join(out_path, "packed"), exist_ok=True)

    params = {
        "front": [False, True],
        "right": [False, True],
        "left": [False, True],
        "back": [False, True],
        "gap": [0.0, 0.01, 0.02, 0.05, 0.07],
    }
    combos = list(product(
        params["front"], params["right"], params["left"],
        params["back"], params["gap"],
    ))

    obb_info = json.load(
        open(os.path.join(obj_dir, "processed_data", "info", "simplified.json"))
    )

    scene_cnt = 0
    for fname in tqdm.tqdm(os.listdir(tabletop_pose_path), desc=f"packed/{obj_name}"):
        pose_idx = fname.split(".")[0]
        pose = np.load(os.path.join(tabletop_pose_path, fname))

        for front, right, left, back, gap in combos:
            if not (front or right or left or back):
                continue

            scene_cfg = {
                "scene": get_packed_scene(
                    obj_name, pose, obb_info, front, right, left, back, gap
                ),
                "meta": {
                    "pose_idx": pose_idx,
                    "param": {
                        "front": front, "right": right,
                        "left": left, "back": back, "gap": gap,
                    },
                },
            }
            with open(os.path.join(out_path, "packed", f"{scene_cnt}.json"), "w") as f:
                json.dump(scene_cfg, f, indent=2)
            scene_cnt += 1


def save_wall_scene(obj_name):
    obj_dir = os.path.join(obj_path, obj_name)
    out_path = get_scene_dir(HAND, obj_name)
    tabletop_pose_path = os.path.join(obj_dir, "processed_data", "info", "tabletop")
    os.makedirs(os.path.join(out_path, "wall"), exist_ok=True)

    params = {
        "z_rotation_deg": [0, 72, 144, 216, 288],
        "gap": [0.0, 0.05],
    }
    combos = list(product(params["z_rotation_deg"], params["gap"]))

    obb_info = json.load(
        open(os.path.join(obj_dir, "processed_data", "info", "simplified.json"))
    )

    scene_cnt = 0
    for fname in tqdm.tqdm(os.listdir(tabletop_pose_path), desc=f"wall/{obj_name}"):
        pose_idx = fname.split(".")[0]
        pose = np.load(os.path.join(tabletop_pose_path, fname))

        for z_rot, gap in combos:
            scene_cfg = {
                "scene": get_wall_scene(obj_name, pose, obb_info, z_rot, gap),
                "meta": {
                    "pose_idx": pose_idx,
                    "param": {"z_rotation_deg": z_rot, "gap": gap},
                },
            }
            with open(os.path.join(out_path, "wall", f"{scene_cnt}.json"), "w") as f:
                json.dump(scene_cfg, f, indent=2)
            scene_cnt += 1


def save_shelf_scene(obj_name):
    obj_dir = os.path.join(obj_path, obj_name)
    out_path = get_scene_dir(HAND, obj_name)
    tabletop_pose_path = os.path.join(obj_dir, "processed_data", "info", "tabletop")
    os.makedirs(os.path.join(out_path, "shelf"), exist_ok=True)

    params = {
        "z_rotation_deg": [0, 72, 144, 216, 288],
        "gap": [0.0, 0.05],
        "up": [True, False],
        "side": [True, False],
        "back": [True],
    }
    combos = list(product(
        params["z_rotation_deg"], params["gap"],
        params["up"], params["side"], params["back"],
    ))

    obb_info = json.load(
        open(os.path.join(obj_dir, "processed_data", "info", "simplified.json"))
    )

    scene_cnt = 0
    for fname in tqdm.tqdm(os.listdir(tabletop_pose_path), desc=f"shelf/{obj_name}"):
        pose_idx = fname.split(".")[0]
        pose = np.load(os.path.join(tabletop_pose_path, fname))

        for z_rot, gap, up, side, back in combos:
            if not (up or side):
                continue
            if up and side and gap == 0.0:
                continue

            scene_cfg = {
                "scene": get_shelf_scene(
                    obj_name, pose, obb_info, z_rot, gap,
                    up=up, side=side, back=back,
                ),
                "meta": {
                    "pose_idx": pose_idx,
                    "param": {
                        "z_rotation_deg": z_rot, "gap": gap,
                        "up": up, "side": side, "back": back,
                    },
                },
            }
            with open(os.path.join(out_path, "shelf", f"{scene_cnt}.json"), "w") as f:
                json.dump(scene_cfg, f, indent=2)
            scene_cnt += 1


def save_box_scene(obj_name):
    obj_dir = os.path.join(obj_path, obj_name)
    out_path = get_scene_dir(HAND, obj_name)
    tabletop_pose_path = os.path.join(obj_dir, "processed_data", "info", "tabletop")
    os.makedirs(os.path.join(out_path, "box"), exist_ok=True)

    params = {"height_offset": [0.05, 0.1]}
    combos = list(product(params["height_offset"]))

    scene_cnt = 0
    for fname in tqdm.tqdm(os.listdir(tabletop_pose_path), desc=f"box/{obj_name}"):
        pose_idx = fname.split(".")[0]
        pose = np.load(os.path.join(tabletop_pose_path, fname))

        for (height_offset,) in combos:
            scene = get_box_scene(obj_name, pose, height_offset)
            if scene is None:
                continue

            scene_cfg = {
                "scene": scene,
                "meta": {
                    "pose_idx": pose_idx,
                    "param": {"height_offset": height_offset},
                },
            }
            with open(os.path.join(out_path, "box", f"{scene_cnt}.json"), "w") as f:
                json.dump(scene_cfg, f, indent=2)
            scene_cnt += 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate scene JSONs for objects")
    parser.add_argument(
        "--objects", nargs="+", default=None,
        help="Object names. If omitted, uses all objects in obj_path.",
    )
    parser.add_argument(
        "--scenes", nargs="+",
        default=["table", "wall", "shelf", "float", "box"],
        help="Scene types to generate (default: all)",
    )
    parser.add_argument(
        "--obj_path", default=obj_path,
        help="Root dir containing object subdirs (default: paradex)",
    )
    parser.add_argument(
        "--hand", default="allegro",
        help="Hand whose scene subtree these scenes are written under.",
    )
    args = parser.parse_args()
    obj_path = args.obj_path
    # Helpers reference obj_path / HAND at module scope; rebind them.
    globals()["obj_path"] = obj_path
    globals()["HAND"] = args.hand

    if args.objects is None:
        obj_list = sorted([
            d for d in os.listdir(obj_path)
            if os.path.isdir(os.path.join(obj_path, d, "processed_data", "info", "tabletop"))
        ])
    else:
        obj_list = args.objects

    scene_funcs = {
        "table": save_tabletop_scene,
        "packed": save_packed_scene,
        "wall": save_wall_scene,
        "shelf": save_shelf_scene,
        "float": save_float_scene,
        "box": save_box_scene,
    }

    for obj_name in tqdm.tqdm(obj_list, desc="Objects"):
        for scene_type in args.scenes:
            if scene_type in scene_funcs:
                scene_funcs[scene_type](obj_name)