"""Virtual obstacle generators for different scene types.

Each function takes an object pose (4x4 SE3 in robot frame) and returns
a dict of cuboid obstacles to merge into scene_cfg["cuboid"].

Scene types:
    - table: just the table (no extra obstacles)
    - wall: a vertical wall behind the object
    - shelf: open-front shelf box (3 walls + top + bottom)
    - cluttered: random cylinders/cubes around the object

All poses are 7D [x, y, z, qw, qx, qy, qz] in robot frame.
All dims are [width, depth, height] matching cuRobo cuboid convention.
"""
import numpy as np
from scipy.spatial.transform import Rotation


TABLE_CUBOID = {
    "dims": [2, 3, 0.2],
    "pose": [1.1, 0, -0.1 + 0.035, 1, 0, 0, 0],
}


def _quat_identity():
    return [1, 0, 0, 0]


def _quat_from_euler(roll=0, pitch=0, yaw=0):
    """Returns [qw, qx, qy, qz]."""
    r = Rotation.from_euler("xyz", [roll, pitch, yaw])
    xyzw = r.as_quat()
    return [float(xyzw[3]), float(xyzw[0]), float(xyzw[1]), float(xyzw[2])]


def get_table_obstacles(obj_pose):
    """Table only — no extra obstacles."""
    return {"table": TABLE_CUBOID}


def get_wall_obstacles(obj_pose, wall_gap=0.04, wall_angle=0.0,
                       wall_thickness=0.02, wall_width=0.5, wall_height=0.4):
    """Wall placed around object, rotated by wall_angle around object center.

    wall_angle=0: wall behind (+y), 90: right (+x), 180: front (-y), 270: left (-x).

    Args:
        obj_pose: (4,4) SE3 in robot frame
        wall_gap: distance from object center to wall front face (meters)
        wall_angle: rotation angle (degrees) around object z-axis. 0=+y.
        wall_thickness: wall thickness
        wall_width: wall extent (tangent direction)
        wall_height: wall extent along z
    """
    obj_xyz = obj_pose[:3, 3]
    table_z = TABLE_CUBOID["pose"][2] + TABLE_CUBOID["dims"][2] / 2

    angle_rad = np.radians(wall_angle)
    # Direction from object center to wall center
    dx = -np.sin(angle_rad)
    dy = np.cos(angle_rad)
    dist = wall_gap + wall_thickness / 2

    wall_center = [
        float(obj_xyz[0] + dx * dist),
        float(obj_xyz[1] + dy * dist),
        float(table_z + wall_height / 2),
    ]

    # Wall orientation: normal faces toward object
    wall_quat = _quat_from_euler(yaw=angle_rad)

    return {
        "table": TABLE_CUBOID,
        "wall": {
            "dims": [wall_width, wall_thickness, wall_height],
            "pose": wall_center + wall_quat,
        },
    }


def get_shelf_obstacles(obj_pose, shelf_width=0.30, shelf_depth=0.30,
                        shelf_height=0.30, shelf_gap=0.02, thickness=0.01,
                        back=True, sides=True, top=True, shelf_angle=0.0):
    """Shelf around object with selectable panels.

    Args:
        obj_pose: (4,4) SE3 in robot frame
        shelf_width: inner width (x direction)
        shelf_depth: inner depth (y direction)
        shelf_height: inner height (z direction)
        shelf_gap: gap between object and shelf walls
        thickness: panel thickness
        back: include back wall
        sides: include left+right walls
        top: include top panel
    """
    obj_xyz = obj_pose[:3, 3]
    table_z = TABLE_CUBOID["pose"][2] + TABLE_CUBOID["dims"][2] / 2

    cx = float(obj_xyz[0])
    cy = float(obj_xyz[1])
    cz = float(table_z + shelf_height / 2)

    hw = shelf_width / 2
    hd = shelf_depth / 2
    angle_rad = np.radians(shelf_angle)

    def _rotate_xy(dx, dy):
        """Rotate (dx, dy) offset around object center by shelf_angle."""
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        return cx + c * dx - s * dy, cy + s * dx + c * dy

    cuboids = {"table": TABLE_CUBOID}
    quat = _quat_from_euler(yaw=angle_rad)

    if back:
        bx, by = _rotate_xy(0, hd + shelf_gap + thickness / 2)
        cuboids["shelf_back"] = {
            "dims": [shelf_width + 2 * thickness, thickness, shelf_height],
            "pose": [bx, by, cz] + quat,
        }
    if sides:
        lx, ly = _rotate_xy(-hw - shelf_gap - thickness / 2, 0)
        cuboids["shelf_left"] = {
            "dims": [thickness, shelf_depth, shelf_height],
            "pose": [lx, ly, cz] + quat,
        }
        rx, ry = _rotate_xy(hw + shelf_gap + thickness / 2, 0)
        cuboids["shelf_right"] = {
            "dims": [thickness, shelf_depth, shelf_height],
            "pose": [rx, ry, cz] + quat,
        }
    if top:
        tx, ty = _rotate_xy(0, thickness / 2)
        cuboids["shelf_top"] = {
            "dims": [shelf_width + 2 * thickness, shelf_depth + thickness, thickness],
            "pose": [tx, ty, cz + shelf_height / 2 + shelf_gap + thickness / 2] + quat,
        }
    return cuboids


def get_cluttered_obstacles(obj_pose, n_obstacles=4, seed=None,
                            min_dist=0.08, max_dist=0.20,
                            min_size=0.03, max_size=0.10,
                            min_height=0.05, max_height=0.15):
    """Random cubes/cylinders (approximated as cubes) around the object.

    Places obstacles on the table around the object at random angles,
    avoiding the approach corridor (front 90 degrees toward robot).

    Args:
        obj_pose: (4,4) SE3 in robot frame
        n_obstacles: number of obstacles to place
        seed: random seed for reproducibility
        min_dist/max_dist: distance range from object center (horizontal)
        min_size/max_size: obstacle width/depth range
        min_height/max_height: obstacle height range
    """
    rng = np.random.RandomState(seed)
    obj_xyz = obj_pose[:3, 3]
    table_z = TABLE_CUBOID["pose"][2] + TABLE_CUBOID["dims"][2] / 2

    cuboids = {"table": TABLE_CUBOID}

    for i in range(n_obstacles):
        # Random angle, avoid front 90deg (approach direction = negative y)
        # Blocked range: [-45, +45] deg from -y direction = [225, 315] deg
        # Allowed: [0, 225) or (315, 360)
        angle = rng.uniform(0, 270)
        if angle > 225:
            angle += 90  # skip 225-315 range
        angle_rad = np.radians(angle)

        dist = rng.uniform(min_dist, max_dist)
        sx = rng.uniform(min_size, max_size)
        sy = rng.uniform(min_size, max_size)
        sz = rng.uniform(min_height, max_height)

        cx = float(obj_xyz[0] + dist * np.cos(angle_rad))
        cy = float(obj_xyz[1] + dist * np.sin(angle_rad))
        cz = float(table_z + sz / 2)

        # Random yaw rotation
        yaw = rng.uniform(0, np.pi)

        cuboids[f"clutter_{i}"] = {
            "dims": [float(sx), float(sy), float(sz)],
            "pose": [cx, cy, cz] + _quat_from_euler(yaw=yaw),
        }

    return cuboids


# ── Public API ───────────────────────────────────────────────────────────

SCENE_TYPES = {
    "table": get_table_obstacles,
    "wall": get_wall_obstacles,
    "shelf": get_shelf_obstacles,
    "cluttered": get_cluttered_obstacles,
}


def add_obstacles(scene_cfg, scene_type, seed=None, wall_gap=0.04, wall_angle=0.0,
                  clutter_min_dist=0.12, clutter_max_dist=0.20, clutter_n=4,
                  shelf_width=0.30, shelf_depth=0.30, shelf_height=0.30, shelf_gap=0.02,
                  shelf_back=True, shelf_sides=True, shelf_top=True):
    """Add virtual obstacles to scene_cfg based on scene type.

    Args:
        scene_cfg: dict with "mesh" and "cuboid" keys
        scene_type: one of "table", "wall", "shelf", "cluttered"
        seed: random seed (only used for "cluttered")
        wall_gap: distance from object center to wall (only for "wall")
        wall_angle: wall rotation around object in degrees (only for "wall")

    Returns:
        scene_cfg with updated "cuboid" dict
    """
    from autodex.utils.conversion import cart2se3

    if scene_type not in SCENE_TYPES:
        raise ValueError(f"Unknown scene type: {scene_type}. Choose from {list(SCENE_TYPES.keys())}")

    # Get object pose in robot frame from scene_cfg
    obj_pose_7d = scene_cfg["mesh"]["target"]["pose"]
    obj_pose = cart2se3(obj_pose_7d)

    if scene_type == "cluttered":
        cuboids = get_cluttered_obstacles(obj_pose, seed=seed, n_obstacles=clutter_n, min_dist=clutter_min_dist, max_dist=clutter_max_dist)
    elif scene_type == "wall":
        cuboids = get_wall_obstacles(obj_pose, wall_gap=wall_gap, wall_angle=wall_angle)
    elif scene_type == "shelf":
        cuboids = get_shelf_obstacles(obj_pose, shelf_width=shelf_width, shelf_depth=shelf_depth,
                                      shelf_height=shelf_height, shelf_gap=shelf_gap,
                                      back=shelf_back, sides=shelf_sides, top=shelf_top,
                                      shelf_angle=wall_angle)
    else:
        cuboids = SCENE_TYPES[scene_type](obj_pose)

    scene_cfg["cuboid"] = cuboids
    return scene_cfg