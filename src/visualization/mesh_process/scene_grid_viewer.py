"""
Grid viewer for all scenes of a given object + scene type.
Uses procedural scene generation (get_wall_scene, get_box_scene, get_shelf_scene)
following the pattern from Visualization/paper/figure1/recorder_scene.py.
"""
import os
import json
import glob
import numpy as np
import trimesh
import transforms3d
from scipy.spatial.transform import Rotation as R
import shapely.geometry as geom

from autodex.utils.path import obj_path, get_scene_dir
from autodex.utils.conversion import cart2se3

# ── Colors ────────────────────────────────────────────────────────────────────
COLORS = {
    "target_obj": (0, 100, 0),
    "obstacle":   (119, 136, 153),
    "table":      (240, 240, 245),
}

COLOR_OBB    = (0.27, 0.51, 1.00)
COLOR_AXIS_X = (1.00, 0.20, 0.20)
COLOR_AXIS_Y = (0.20, 0.85, 0.20)
COLOR_AXIS_Z = (0.20, 0.20, 1.00)

LINE_WIDTH_OBB  = 4.0
LINE_WIDTH_AXIS = 3.0
# ──────────────────────────────────────────────────────────────────────────────


# ── Procedural scene builders (from recorder_scene.py) ────────────────────────

def rotz(theta):
    c, s = np.cos(theta), np.sin(theta)
    T = np.eye(4)
    T[0, 0] = c; T[0, 1] = -s
    T[1, 0] = s; T[1, 1] = c
    return T


def transl(xyz):
    T = np.eye(4)
    T[:3, 3] = xyz
    return T


def mat4_to_pose(T):
    q = R.from_matrix(T[:3, :3]).as_quat()  # xyzw
    return [T[0, 3], T[1, 3], T[2, 3], q[3], q[0], q[1], q[2]]


def get_mesh_dict(obj_name, pose):
    return {
        "scale": [1.0, 1.0, 1.0],
        "pose": [
            pose[0, 3], pose[1, 3], pose[2, 3],
            *transforms3d.quaternions.mat2quat(pose[:3, :3])
        ],
        "file_path": os.path.join(obj_path, obj_name, "processed_data", "mesh", "simplified.obj"),
    }


def get_tabletop_scene(obj_name, tabletop_pose):
    ret = {
        "mesh": {},
        "cuboid": {
            "table": {
                "dims": [2.0, 2.0, 0.2],
                "pose": [0.0, 0.0, -0.1, 1, 0, 0, 0.0],
            }
        }
    }
    ret["mesh"]["target"] = get_mesh_dict(obj_name, tabletop_pose)
    return ret


def get_wall_scene(obj_name, tabletop_pose, obb_info, z_rotation_deg, gap):
    angle_rad = np.radians(z_rotation_deg)
    z_rotation = np.array([
        [np.cos(angle_rad), -np.sin(angle_rad), 0, 0],
        [np.sin(angle_rad),  np.cos(angle_rad), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    rotated_pose = z_rotation @ tabletop_pose

    obb_transform = np.array(obb_info['obb_transform'])
    R_obb = obb_transform[:3, :3]
    obb_extents = np.array(obb_info['obb'])

    R_world = rotated_pose[:3, :3]
    t_world = rotated_pose[:3, 3]
    obb_world = R_world @ R_obb

    corners_local = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                corners_local.append(np.array([i, j, k]) * obb_extents / 2)
    corners_world = np.array([(obb_world @ c + t_world) for c in corners_local])

    min_y = corners_world[:, 1].min()
    max_z = corners_world[:, 2].max()
    wall_height = max_z + 0.1
    wall_y = min_y - gap

    scene = get_tabletop_scene(obj_name, rotated_pose)
    scene["cuboid"]["wall"] = {
        "dims": [0.3, 0.02, wall_height],
        "pose": [0.0, wall_y - 0.01, wall_height / 2, 1, 0, 0, 0],
    }
    return scene


def get_box_scene(obj_name, tabletop_pose, height_offset):
    mesh_path = os.path.join(obj_path, obj_name, "processed_data", "mesh", "simplified.obj")
    mesh = trimesh.load(mesh_path, force="mesh")

    verts = mesh.vertices
    R_obj = tabletop_pose[:3, :3]
    t_obj = tabletop_pose[:3, 3]
    verts_w = (R_obj @ verts.T).T + t_obj

    points_xy = verts_w[:, :2]
    poly = geom.MultiPoint(points_xy).convex_hull
    rect = poly.minimum_rotated_rectangle
    rect_pts = np.array(rect.exterior.coords)[:4]

    cx, cy = rect.centroid.coords[0]
    edge = rect_pts[1] - rect_pts[0]
    yaw = np.arctan2(edge[1], edge[0])

    width = np.linalg.norm(rect_pts[1] - rect_pts[0])
    depth = np.linalg.norm(rect_pts[2] - rect_pts[1])

    max_z = verts_w[:, 2].max()
    wall_height = max_z - height_offset
    if wall_height <= 0:
        return get_tabletop_scene(obj_name, tabletop_pose)

    THICK = 0.02
    scene = get_tabletop_scene(obj_name, tabletop_pose)

    T_box_center = np.eye(4)
    T_box_center[:3, :3] = rotz(yaw)[:3, :3]
    T_box_center[:3, 3] = [cx, cy, wall_height / 2]

    hw, hd = width / 2, depth / 2

    def add_wall(name, local_x, local_y, w, d):
        T_local = np.eye(4)
        T_local[:3, 3] = [local_x, local_y, 0]
        T_wall = T_box_center @ T_local
        scene["cuboid"][name] = {
            "dims": [w, d, wall_height],
            "pose": mat4_to_pose(T_wall),
        }

    add_wall("box_front",  hw + THICK / 2, 0, THICK, depth + 2 * THICK)
    add_wall("box_back",  -hw - THICK / 2, 0, THICK, depth + 2 * THICK)
    add_wall("box_right", 0,  hd + THICK / 2, width + 2 * THICK, THICK)
    add_wall("box_left",  0, -hd - THICK / 2, width + 2 * THICK, THICK)

    return scene


def get_shelf_scene(obj_name, tabletop_pose, obb_info, z_rotation_deg, gap,
                    up=True, side=True, back=True):
    angle = np.radians(z_rotation_deg)
    z_rot = np.array([
        [np.cos(angle), -np.sin(angle), 0, 0],
        [np.sin(angle),  np.cos(angle), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    pose = z_rot @ tabletop_pose

    obb_tf = np.array(obb_info["obb_transform"])
    R_obb = obb_tf[:3, :3]
    ext = np.array(obb_info["obb"]) / 2.0

    Rw = pose[:3, :3]
    tw = pose[:3, 3]
    axes = Rw @ R_obb

    corners = []
    for sx in [-1, 1]:
        for sy in [-1, 1]:
            for sz in [-1, 1]:
                corners.append(axes @ (np.array([sx, sy, sz]) * ext) + tw)
    corners = np.array(corners)

    THICK = 0.02
    min_x, max_x = corners[:, 0].min(), corners[:, 0].max()
    min_y, max_y = corners[:, 1].min(), corners[:, 1].max()
    max_z = corners[:, 2].max()

    inner_y_back  = min_y - gap
    inner_y_front = max_y + gap
    inner_x_min   = min_x - gap
    inner_x_max   = max_x + gap
    inner_z_top   = max_z + gap
    wall_h = inner_z_top

    scene = get_tabletop_scene(obj_name, pose)
    full_width = (inner_x_max - inner_x_min) + 2 * THICK

    if back:
        scene["cuboid"]["back"] = {
            "dims": [full_width, THICK, wall_h],
            "pose": [
                (inner_x_min + inner_x_max) / 2,
                inner_y_back - THICK / 2,
                wall_h / 2,
                1, 0, 0, 0
            ],
        }

    full_depth = (inner_y_front - inner_y_back) + THICK
    if side:
        scene["cuboid"]["side_pos"] = {
            "dims": [THICK, full_depth - THICK, wall_h],
            "pose": [
                inner_x_max + THICK / 2,
                (inner_y_back - THICK / 2 + inner_y_front) / 2,
                wall_h / 2,
                1, 0, 0, 0
            ],
        }
        scene["cuboid"]["side_neg"] = {
            "dims": [THICK, full_depth - THICK, wall_h],
            "pose": [
                inner_x_min - THICK / 2,
                (inner_y_back - THICK / 2 + inner_y_front) / 2,
                wall_h / 2,
                1, 0, 0, 0
            ],
        }

    if up:
        scene["cuboid"]["up"] = {
            "dims": [full_width, full_depth, THICK],
            "pose": [
                (inner_x_min + inner_x_max) / 2,
                (inner_y_back - THICK + inner_y_front) / 2,
                inner_z_top + THICK / 2,
                1, 0, 0, 0
            ],
        }

    return scene


# ── Helpers ───────────────────────────────────────────────────────────────────


def load_mesh(path):
    mesh = trimesh.load(path)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return mesh


def parse_pose(pose_list):
    pose = np.eye(4)
    pose[:3, 3] = np.array(pose_list[:3])
    quat = pose_list[3:]  # wxyz
    pose[:3, :3] = R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
    return pose


def clear_all():
    for name in list(vis.obj_dict.keys()):
        try:
            vis.obj_dict[name]['frame'].remove()
        except Exception:
            pass
    vis.obj_dict.clear()
    vis.frame_nodes.clear()


def add_obb(parent_frame, obb_extents, obb_transform, axis_len):
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


def build_scene(obj_name, scene_type, cfg, obb_info):
    """Build scene dict using procedural generation, following recorder_scene.py."""
    tabletop_pose = cart2se3(cfg['scene']['mesh']['target']['pose'])
    param = cfg.get('meta', {}).get('param', {})

    if scene_type == "wall":
        return get_wall_scene(
            obj_name=obj_name,
            tabletop_pose=tabletop_pose,
            obb_info=obb_info,
            z_rotation_deg=0.0,
            gap=param.get('gap', 0.03),
        )
    elif scene_type == "box":
        return get_box_scene(
            obj_name=obj_name,
            tabletop_pose=tabletop_pose,
            height_offset=param.get('height_offset', 0.1),
        )
    elif scene_type == "shelf":
        return get_shelf_scene(
            obj_name=obj_name,
            tabletop_pose=tabletop_pose,
            obb_info=obb_info,
            z_rotation_deg=0.0,
            gap=param.get('gap', 0.03),
            up=param.get('up', True),
            side=param.get('side', True),
            back=param.get('back', True),
        )
    elif scene_type in ("table", "float"):
        return get_tabletop_scene(obj_name, tabletop_pose)
    else:
        return get_tabletop_scene(obj_name, tabletop_pose)


def load_grid():
    clear_all()

    obj_name = obj_dropdown.value
    scene_type = scene_type_dropdown.value
    if not obj_name or not scene_type:
        return

    scene_files = get_scene_files(obj_name, scene_type)
    if not scene_files:
        print(f"No scenes found for {obj_name}/{scene_type}")
        return

    # Load OBB info
    info_path = os.path.join(obj_path, obj_name, "processed_data", "info", "simplified.json")
    with open(info_path, 'r') as f:
        obb_info = json.load(f)
    obb_extents = np.array(obb_info['obb'])
    obb_transform = np.array(obb_info['obb_transform'])
    axis_len = float(np.max(obb_extents)) * 0.6

    # Load raw mesh once
    raw_path = os.path.join(obj_path, obj_name, "raw_mesh", f"{obj_name}.obj")
    raw_mesh = load_mesh(raw_path)

    # Grid params
    margin = 0.1
    grid_spacing = float(np.linalg.norm(obb_extents)) * 3.0 + margin
    grid_cols = int(np.ceil(np.sqrt(len(scene_files))))

    print(f"Loading {len(scene_files)} {scene_type} scenes in {grid_cols}x grid...")

    for idx, scene_path in enumerate(scene_files):
        with open(scene_path, 'r') as f:
            cfg = json.load(f)

        scene_idx_str = os.path.basename(scene_path).replace(".json", "")

        # Build scene procedurally
        scene = build_scene(obj_name, scene_type, cfg, obb_info)
        if scene is None:
            continue

        # Grid offset
        row = idx // grid_cols
        col = idx % grid_cols
        offset = np.array([
            (col - grid_cols // 2) * grid_spacing,
            (row - grid_cols // 2) * grid_spacing,
            0.0,
        ])

        # Render meshes
        for mesh_name, mesh_info in scene.get("mesh", {}).items():
            if mesh_name == "target":
                mesh = raw_mesh
            else:
                mesh = trimesh.load(mesh_info["file_path"], force="mesh")

            pose = parse_pose(mesh_info["pose"])
            pose[:3, 3] += offset

            name = f"s{scene_idx_str}_{mesh_name}"
            vis.add_object(name, mesh, obj_T=pose)

            if mesh_name == "target":
                fp = f"/objects/{name}_frame"
                add_obb(fp, obb_extents, obb_transform, axis_len)

        # Render cuboids
        for cuboid_name, cuboid_info in scene.get("cuboid", {}).items():
            box = trimesh.creation.box(extents=cuboid_info["dims"])
            pose = parse_pose(cuboid_info["pose"])
            pose[:3, 3] += offset

            name = f"s{scene_idx_str}_{cuboid_name}"
            vis.add_object(name, box, obj_T=pose)

            if cuboid_name == "table":
                color = tuple(c / 255.0 for c in COLORS["table"])
                vis.change_color(name, color + (0.9,))
            else:
                color = tuple(c / 255.0 for c in COLORS["obstacle"])
                vis.change_color(name, color + (0.5,))



    print(f"Loaded {len(scene_files)} {scene_type} scenes for {obj_name}")


def update_scene_types():
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


# ── GUI ────────────────────────────────────────────────────────────────────────

vis = None
available_objects = []
obj_dropdown = None
scene_type_dropdown = None


def _main():
    from paradex.visualization.visualizer.viser import ViserViewer
    global vis, available_objects, obj_dropdown, scene_type_dropdown

    available_objects = sorted([
        d for d in os.listdir(obj_path)
        if os.path.isdir(get_scene_dir("allegro", d))
    ])
    print(f"Found {len(available_objects)} objects with scenes")

    vis = ViserViewer()

    with vis.server.gui.add_folder("Scene Grid"):
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
        load_btn = vis.server.gui.add_button("Load Grid")

    @obj_dropdown.on_update
    def _(_) -> None:
        update_scene_types()

    @load_btn.on_click
    def _(_) -> None:
        load_grid()

    if available_objects:
        update_scene_types()
        load_grid()

    vis.start_viewer()


if __name__ == "__main__":
    _main()
