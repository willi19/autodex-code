"""Table-contact corrections for live object poses.

FoundPose estimates a full 6-D pose. A few millimetres of vertical error can
put the planning mesh below the virtual table, causing every otherwise-valid
approach to be rejected as a hand/table collision. The physical object is
known to be resting on the table during the pick stage, so only raise a pose
just enough to put its transformed mesh bottom on that same table surface.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np

# Keep this equal to ``autodex.planner.obstacles.TABLE_CUBOID``'s upper
# surface.  It lives here as a scalar so the light-weight readiness/unit-test
# path does not import cuRobo or initialize CUDA merely to inspect a pose.
TABLE_SURFACE_Z = 0.035


def table_surface_z() -> float:
    """Planner table-top height in robot coordinates."""
    return TABLE_SURFACE_Z


def mesh_bottom_z(pose_robot: np.ndarray, vertices: np.ndarray) -> float:
    """Return the z coordinate of a mesh transformed by ``pose_robot``."""
    pose = np.asarray(pose_robot, dtype=np.float64)
    points = np.asarray(vertices, dtype=np.float64)
    if pose.shape != (4, 4):
        raise ValueError(f"expected pose shape (4, 4), got {pose.shape}")
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        raise ValueError("vertices must have shape (N, 3), N > 0")
    return float((pose[:3, :3] @ points.T)[2].min() + pose[2, 3])


def raise_to_table(
    pose_robot: np.ndarray,
    vertices: np.ndarray,
    *,
    surface_z: float | None = None,
    max_raise_m: float = 0.015,
) -> Tuple[np.ndarray, float, float]:
    """Raise an implausibly sunken pose without ever lowering it.

    Returns ``(corrected_pose, applied_raise_m, original_bottom_z)``. A large
    correction is refused: it more likely means that FoundPose selected a wrong
    instance or has a bad transform than a table-contact measurement error.
    """
    if max_raise_m < 0:
        raise ValueError("max_raise_m must be non-negative")
    surface_z = table_surface_z() if surface_z is None else float(surface_z)
    original_bottom = mesh_bottom_z(pose_robot, vertices)
    delta = max(0.0, surface_z - original_bottom)
    corrected = np.asarray(pose_robot, dtype=np.float64).copy()
    if 0.0 < delta <= max_raise_m:
        corrected[2, 3] += delta
        return corrected, delta, original_bottom
    return corrected, 0.0, original_bottom


def load_mesh_vertices(mesh_path: str | Path) -> np.ndarray:
    """Load only the vertices needed for the inexpensive table-contact test."""
    import trimesh

    mesh = trimesh.load(str(mesh_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    return np.asarray(mesh.vertices, dtype=np.float64)


def snap_pose_world_to_table(
    pose_world: np.ndarray,
    c2r: np.ndarray,
    vertices: np.ndarray,
    *,
    max_raise_m: float = 0.015,
) -> Tuple[np.ndarray, dict]:
    """Apply :func:`raise_to_table` while preserving the external world pose API."""
    pose_robot = np.linalg.inv(np.asarray(c2r, dtype=np.float64)) @ np.asarray(
        pose_world, dtype=np.float64
    )
    corrected_robot, applied, bottom = raise_to_table(
        pose_robot, vertices, max_raise_m=max_raise_m
    )
    return np.asarray(c2r, dtype=np.float64) @ corrected_robot, {
        "surface_z_robot": table_surface_z(),
        "mesh_bottom_z_robot": bottom,
        "raise_m": applied,
        "max_raise_m": float(max_raise_m),
        "applied": bool(applied > 0.0),
    }
