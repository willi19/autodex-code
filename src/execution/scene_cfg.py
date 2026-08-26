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


def find_planning_mesh(obj_name: str, obj_root: Optional[str] = None) -> str:
    """Planning mesh for ``obj_name`` under ``obj_root`` (default: legacy obj_path).

    Pass ``get_obj_root(version)`` to resolve a v8 pool against
    object_processing — its simplified.obj is a different mesh frame than the
    paradex one, so the root must match the candidate pool.
    """
    root = obj_root or obj_path
    p = os.path.join(root, obj_name, "processed_data", "mesh", "simplified.obj")
    if os.path.exists(p):
        return p
    p2 = os.path.join(root, obj_name, "raw_mesh", f"{obj_name}.obj")
    if os.path.exists(p2):
        return p2
    raise FileNotFoundError(f"No planning mesh for {obj_name} under {root}")


def check_mesh_frame_match(obj_name: str, perception_mesh: str,
                           obj_root: Optional[str] = None,
                           tol_m: float = 0.002) -> tuple:
    """Verify the perception mesh and the planning asset tree share one frame.

    FoundPose estimates the pose of ``perception_mesh``; the planner places
    ``find_planning_mesh(obj, obj_root)``. If the two roots hold *different
    geometry* for the object, the estimated pose is expressed in the wrong
    frame and every grasp is silently offset.

    For 98 of 104 objects the paradex and object_processing ``raw_mesh`` files
    are byte-identical, so the check is free. The rest (paradex shipped a crude
    primitive, op has a real scan) genuinely need re-onboarding against op.

    Returns ``(ok: bool, msg: str)``.
    """
    import filecmp

    root = obj_root or obj_path
    if os.path.realpath(root) == os.path.realpath(obj_path):
        return True, "planning root == legacy obj_path"

    ref = os.path.join(root, obj_name, "raw_mesh", f"{obj_name}.obj")
    if not os.path.exists(ref):
        return True, f"no raw_mesh under {root} — cannot compare, assuming ok"
    if not os.path.exists(perception_mesh):
        return False, f"perception mesh missing: {perception_mesh}"
    if filecmp.cmp(perception_mesh, ref, shallow=False):
        return True, "raw_mesh byte-identical across roots"

    # Different bytes: fall back to geometry. Same frame => same vertex count
    # and a centroid/extent match well under a grasp-relevant tolerance.
    a = trimesh.load(perception_mesh, process=False)
    b = trimesh.load(ref, process=False)
    if isinstance(a, trimesh.Scene):
        a = a.dump(concatenate=True)
    if isinstance(b, trimesh.Scene):
        b = b.dump(concatenate=True)
    d_cent = float(np.linalg.norm(np.asarray(a.centroid) - np.asarray(b.centroid)))
    d_ext = float(np.max(np.abs(np.asarray(a.extents) - np.asarray(b.extents))))
    if d_cent <= tol_m and d_ext <= tol_m:
        return True, f"geometry matches (dcentroid={d_cent:.4f} dextent={d_ext:.4f})"
    return False, (
        f"{obj_name}: perception mesh and planning root disagree "
        f"(dcentroid={d_cent:.4f}m dextent={d_ext:.4f}m, "
        f"V {len(a.vertices)} vs {len(b.vertices)}). Re-onboard FoundPose "
        f"against {ref} before running this pool."
    )


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


def _snap_cylinder_pose(pose_robot: np.ndarray, obj_name: str,
                        obj_root: Optional[str] = None) -> np.ndarray:
    tabletop_dir = os.path.join(obj_root or obj_path, obj_name,
                                "processed_data", "info", "tabletop")
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


def _snap_sphere_pose(pose_robot: np.ndarray, obj_name: str,
                      obj_root: Optional[str] = None) -> np.ndarray:
    tabletop_dir = os.path.join(obj_root or obj_path, obj_name,
                                "processed_data", "info", "tabletop")
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


def pose_world_to_scene_cfg(pose_world: np.ndarray, c2r: np.ndarray, obj_name: str,
                            obj_root: Optional[str] = None) -> dict:
    """Convert world-frame 4x4 pose to a scene_cfg dict for GraspPlanner.plan().

    ``obj_root`` selects which asset tree the planning mesh and the tabletop
    snap poses come from — pass ``get_obj_root(version)`` so a v8 pool reads
    object_processing. Defaults to the legacy ``obj_path``.
    """
    pose_robot = np.linalg.inv(c2r) @ pose_world
    if obj_name in SPHERE_OBJECTS:
        pose_robot = _snap_sphere_pose(pose_robot, obj_name, obj_root)
    elif obj_name in CYLINDER_OBJECTS:
        pose_robot = _snap_cylinder_pose(pose_robot, obj_name, obj_root)
    return {
        "mesh": {
            "target": {
                "pose": se32cart(pose_robot).tolist(),
                "file_path": find_planning_mesh(obj_name, obj_root),
            }
        },
        "cuboid": {
            "table": {
                "dims": [2, 3, 0.2],
                "pose": [1.1, 0, -0.1 + 0.037, 1, 0, 0, 0],
            }
        },
    }
