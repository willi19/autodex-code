import argparse
import os
import json
import glob
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot

from paradex.visualization.visualizer.viser import ViserViewer
from autodex.utils.path import obj_path, get_scene_dir

parser = argparse.ArgumentParser()
parser.add_argument(
    "--obj_path", default=obj_path,
    help="Root dir of object subdirs (default: paradex from autodex.utils.path).",
)
cli_args = parser.parse_args()
obj_path = cli_args.obj_path

# ── Colors (0-1 float, matching recorder_scene.py style) ─────────────────────
COLOR_TABLE    = (0.94, 0.94, 0.96)
COLOR_OBSTACLE = (0.47, 0.53, 0.60)
COLOR_OBB      = (0.27, 0.51, 1.00)
COLOR_AXIS_X   = (1.00, 0.20, 0.20)
COLOR_AXIS_Y   = (0.20, 0.85, 0.20)
COLOR_AXIS_Z   = (0.20, 0.20, 1.00)

TABLE_OPACITY    = 0.9
OBSTACLE_OPACITY = 0.5
LINE_WIDTH_OBB   = 4.0
LINE_WIDTH_AXIS  = 3.0
# ──────────────────────────────────────────────────────────────────────────────

available_objects = sorted([
    d for d in os.listdir(obj_path)
    if os.path.isdir(get_scene_dir("allegro", d, "table"))
])

print(f"Found {len(available_objects)} objects with scenes")

vis = ViserViewer()


def load_mesh(path):
    mesh = trimesh.load(path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh


def parse_pose(pose_list):
    """Parse [x, y, z, qw, qx, qy, qz] into a 4x4 matrix."""
    pose = np.eye(4)
    pose[:3, 3] = np.array(pose_list[:3])
    quat = pose_list[3:]  # wxyz
    pose[:3, :3] = Rot.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    return pose


def clear_scene():
    for name in list(vis.obj_dict.keys()):
        try:
            vis.obj_dict[name]['frame'].remove()
        except Exception:
            pass
    vis.obj_dict.clear()
    vis.frame_nodes.clear()


def add_obb(parent_frame, obb_extents, obb_transform, axis_len):
    """Add OBB wireframe and axes as children of parent_frame."""
    half = obb_extents / 2
    corners_local = np.array([
        [-half[0], -half[1], -half[2]],
        [ half[0], -half[1], -half[2]],
        [ half[0],  half[1], -half[2]],
        [-half[0],  half[1], -half[2]],
        [-half[0], -half[1],  half[2]],
        [ half[0], -half[1],  half[2]],
        [ half[0],  half[1],  half[2]],
        [-half[0],  half[1],  half[2]],
    ])
    corners = (obb_transform[:3, :3] @ corners_local.T).T + obb_transform[:3, 3]

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]
    for ei, (i, j) in enumerate(edges):
        vis.server.scene.add_spline_catmull_rom(
            name=f"{parent_frame}/obb_edge_{ei}",
            positions=np.array([corners[i], corners[j]]),
            color=COLOR_OBB,
            line_width=LINE_WIDTH_OBB,
        )

    origin = np.zeros(3)
    for axis_dir, axis_color, label in [
        (np.array([1, 0, 0]), COLOR_AXIS_X, "axis_x"),
        (np.array([0, 1, 0]), COLOR_AXIS_Y, "axis_y"),
        (np.array([0, 0, 1]), COLOR_AXIS_Z, "axis_z"),
    ]:
        vis.server.scene.add_spline_catmull_rom(
            name=f"{parent_frame}/{label}",
            positions=np.array([origin, axis_dir * axis_len]),
            color=axis_color,
            line_width=LINE_WIDTH_AXIS,
        )


def get_scene_files(obj_name, scene_type):
    scene_dir = get_scene_dir("allegro", obj_name, scene_type)
    if not os.path.isdir(scene_dir):
        return []
    return sorted(
        glob.glob(os.path.join(scene_dir, "*.json")),
        key=lambda x: int(os.path.basename(x).split('.')[0]),
    )


def load_scene():
    clear_scene()

    obj_name = obj_dropdown.value
    scene_type = scene_type_dropdown.value
    if not obj_name or not scene_type:
        return

    scene_files = get_scene_files(obj_name, scene_type)
    if not scene_files or scene_idx_slider.value >= len(scene_files):
        return

    scene_path = scene_files[scene_idx_slider.value]
    with open(scene_path, 'r') as f:
        cfg = json.load(f)

    scene = cfg["scene"]

    # Load OBB info
    info_path = os.path.join(obj_path, obj_name, "processed_data", "info", "simplified.json")
    with open(info_path, 'r') as f:
        info = json.load(f)
    obb_extents = np.array(info['obb'])
    obb_transform = np.array(info['obb_transform'])
    axis_len = float(np.max(obb_extents)) * 0.6

    # Render meshes
    for mesh_name, mesh_info in scene.get("mesh", {}).items():
        if mesh_name == "target":
            mesh_path = os.path.join(
                obj_path, obj_name, "processed_data", "mesh", "simplified.obj"
            )
            mesh = load_mesh(mesh_path)
        else:
            mesh = trimesh.load(mesh_info["file_path"], force="mesh")

        pose = parse_pose(mesh_info["pose"])
        vis.add_object(mesh_name, mesh, obj_T=pose)

        if mesh_name == "target":
            fp = f"/objects/{mesh_name}_frame"
            add_obb(fp, obb_extents, obb_transform, axis_len)

    # Render cuboids
    for cuboid_name, cuboid_info in scene.get("cuboid", {}).items():
        box = trimesh.creation.box(extents=cuboid_info["dims"])
        pose = parse_pose(cuboid_info["pose"])
        vis.add_object(cuboid_name, box, obj_T=pose)

        if cuboid_name == "table":
            vis.change_color(cuboid_name, COLOR_TABLE + (TABLE_OPACITY,))
        else:
            vis.change_color(cuboid_name, COLOR_OBSTACLE + (OBSTACLE_OPACITY,))

    scene_idx_str = os.path.basename(scene_path).replace(".json", "")
    print(f"[SceneViewer] {obj_name} / {scene_type} / {scene_idx_str}")


def update_scene_types():
    """Update scene type dropdown for current object."""
    obj_name = obj_dropdown.value
    scene_root = get_scene_dir("allegro", obj_name)
    if os.path.isdir(scene_root):
        types = sorted(
            d for d in os.listdir(scene_root)
            if os.path.isdir(os.path.join(scene_root, d))
        )
    else:
        types = []

    scene_type_dropdown.options = tuple(types) if types else ("",)
    if types:
        scene_type_dropdown.value = types[0]
    update_scene_index()


def update_scene_index():
    """Update scene index slider for current object + scene type."""
    scene_files = get_scene_files(obj_dropdown.value, scene_type_dropdown.value)
    if scene_files:
        scene_idx_slider.max = len(scene_files) - 1
        scene_idx_slider.value = 0
        load_scene()
    else:
        scene_idx_slider.max = 1
        scene_idx_slider.value = 0
        clear_scene()


# ── GUI ────────────────────────────────────────────────────────────────────────
with vis.server.gui.add_folder("Scene"):
    obj_dropdown = vis.server.gui.add_dropdown(
        "Object",
        options=tuple(available_objects),
        initial_value=available_objects[0] if available_objects else "",
    )
    scene_type_dropdown = vis.server.gui.add_dropdown(
        "Scene Type",
        options=("",),
        initial_value="",
    )
    scene_idx_slider = vis.server.gui.add_slider(
        "Scene Index",
        min=0,
        max=1,
        step=1,
        initial_value=0,
    )

with vis.server.gui.add_folder("Appearance"):
    wall_opacity_slider = vis.server.gui.add_slider(
        "Wall Opacity", min=0.0, max=1.0, step=0.05, initial_value=OBSTACLE_OPACITY,
    )
    table_opacity_slider = vis.server.gui.add_slider(
        "Table Opacity", min=0.0, max=1.0, step=0.05, initial_value=TABLE_OPACITY,
    )

WALL_NAMES = ["back", "side_pos", "side_neg", "up", "wall",
              "box_front", "box_back", "box_left", "box_right", "floor"]


@obj_dropdown.on_update
def _(_) -> None:
    update_scene_types()

@scene_type_dropdown.on_update
def _(_) -> None:
    update_scene_index()

@scene_idx_slider.on_update
def _(_) -> None:
    load_scene()

@wall_opacity_slider.on_update
def _(_) -> None:
    for name in WALL_NAMES:
        if name in vis.obj_dict:
            vis.change_color(name, COLOR_OBSTACLE + (wall_opacity_slider.value,))

@table_opacity_slider.on_update
def _(_) -> None:
    if "table" in vis.obj_dict:
        vis.change_color("table", COLOR_TABLE + (table_opacity_slider.value,))

# Load initial
if available_objects:
    update_scene_types()

vis.add_floor(0.0)
vis.start_viewer()
