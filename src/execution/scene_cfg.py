"""Build cuRobo scene_cfg from a perceived object pose.

Extracted from src/execution_prev/run_auto.py so it can be shared by the new
init-pipeline-based runner (and any future entry points).
"""
from __future__ import annotations

import glob
import os
from typing import Optional

import numpy as np
import trimesh

from autodex.utils.conversion import se32cart
from autodex.utils.path import obj_path


# Object bottom must stay >= this z (robot frame) when snapping to table.
TABLE_SURFACE_Z = -0.1 + 0.039 + 0.1  # 0.039

# Objects with y-axis cylindrical symmetry — snap to nearest tabletop pose.
CYLINDER_OBJECTS = [
    "pepper_tuna", "pepper_tuna_light", "pepsi", "pepsi_light",
    "smallbowl", "jja_ramen", "open_short_pringles",
    "beige_brush",
]

# Spherical objects — use first tabletop pose rotation directly.
SPHERE_OBJECTS = ["baseball", "tennis_ball"]


def find_planning_mesh(obj_name: str) -> str:
    p = os.path.join(obj_path, obj_name, "processed_data", "mesh", "simplified.obj")
    if os.path.exists(p):
        return p
    p2 = os.path.join(obj_path, obj_name, "raw_mesh", f"{obj_name}.obj")
    if os.path.exists(p2):
        return p2
    raise FileNotFoundError(f"No planning mesh for {obj_name}")


def _snap_z_to_table(pose_robot: np.ndarray, mesh_path: str) -> np.ndarray:
    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    verts = np.asarray(mesh.vertices)
    verts_h = np.hstack([verts, np.ones((len(verts), 1))])
    verts_robot = (pose_robot @ verts_h.T).T[:, :3]
    bottom_z = verts_robot[:, 2].min()

    if bottom_z < TABLE_SURFACE_Z:
        delta = TABLE_SURFACE_Z - bottom_z
        print(f"    [snap] Object bottom {bottom_z:.4f} < table {TABLE_SURFACE_Z:.4f}, raising by {delta:.4f}m")
        pose_robot = pose_robot.copy()
        pose_robot[2, 3] += delta
    return pose_robot


def _snap_cylinder_pose(pose_robot: np.ndarray, obj_name: str) -> np.ndarray:
    tabletop_dir = os.path.join(obj_path, obj_name, "processed_data", "info", "tabletop")
    if not os.path.isdir(tabletop_dir):
        return pose_robot
    tabletop_files = sorted(glob.glob(os.path.join(tabletop_dir, "*.npy")))
    if not tabletop_files:
        return pose_robot

    R_est = pose_robot[:3, :3]
    y_est = R_est @ np.array([0, 1, 0])

    best_diff = float("inf")
    best_R_tab = R_est
    for tf in tabletop_files:
        R_tab = np.load(tf)[:3, :3]
        y_tab_z = R_tab[2, 1]
        diff = np.abs(np.abs(y_est[2]) - np.abs(y_tab_z))
        if diff < best_diff:
            best_diff = diff
            best_R_tab = R_tab.copy()
            if y_est[2] * y_tab_z < 0:
                best_R_tab = best_R_tab @ np.diag([1, -1, -1]).astype(float)

    y_tab = best_R_tab[:, 1]
    phi = np.arctan2(y_est[1], y_est[0]) - np.arctan2(y_tab[1], y_tab[0])
    c, s = np.cos(phi), np.sin(phi)
    R_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    best_R = R_z @ best_R_tab

    print(f"    [cylinder] Snapped (y-z diff={best_diff:.3f}, z-rot={np.degrees(phi):.1f}deg)")
    pose_robot = pose_robot.copy()
    pose_robot[:3, :3] = best_R
    return pose_robot


def _snap_sphere_pose(pose_robot: np.ndarray, obj_name: str) -> np.ndarray:
    tabletop_dir = os.path.join(obj_path, obj_name, "processed_data", "info", "tabletop")
    if not os.path.isdir(tabletop_dir):
        return pose_robot
    tabletop_files = sorted(glob.glob(os.path.join(tabletop_dir, "*.npy")))
    if not tabletop_files:
        return pose_robot

    R_tab = np.load(tabletop_files[0])[:3, :3]
    print(f"    [sphere] Replaced rotation with tabletop pose 0")
    pose_robot = pose_robot.copy()
    pose_robot[:3, :3] = R_tab
    return pose_robot


def pose_world_to_scene_cfg(pose_world: np.ndarray, c2r: np.ndarray, obj_name: str) -> dict:
    """Convert world-frame 4x4 pose to a scene_cfg dict for GraspPlanner.plan()."""
    pose_robot = np.linalg.inv(c2r) @ pose_world
    if obj_name in SPHERE_OBJECTS:
        pose_robot = _snap_sphere_pose(pose_robot, obj_name)
    elif obj_name in CYLINDER_OBJECTS:
        pose_robot = _snap_cylinder_pose(pose_robot, obj_name)
    return {
        "mesh": {
            "target": {
                "pose": se32cart(pose_robot).tolist(),
                "file_path": find_planning_mesh(obj_name),
            }
        },
        "cuboid": {
            "table": {
                "dims": [2, 3, 0.2],
                "pose": [1.1, 0, -0.1 + 0.037, 1, 0, 0, 0],
            }
        },
    }
