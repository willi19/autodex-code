"""Geometry helpers for a marker-defined basket release reference."""
from __future__ import annotations

import numpy as np


def release_reference_from_marker(
    center_robot: np.ndarray,
    pose_robot: np.ndarray,
    marker_offset_m: np.ndarray,
) -> np.ndarray:
    """Return the basket release point from a detected standalone marker.

    ``marker_offset_m`` is expressed in the detected marker's local frame, not
    the robot frame. Attach the marker horizontally on a rigid basket fixture;
    its local +z then points upward and the offset can reach the open interior.
    A zero offset means that the marker centre itself is the release reference.
    """
    center = np.asarray(center_robot, dtype=np.float64).reshape(-1)
    pose = np.asarray(pose_robot, dtype=np.float64)
    offset = np.asarray(marker_offset_m, dtype=np.float64).reshape(-1)
    if center.shape != (3,):
        raise ValueError(f"marker center must have shape (3,), got {center.shape}")
    if pose.shape != (4, 4):
        raise ValueError(f"marker pose must have shape (4, 4), got {pose.shape}")
    if offset.shape != (3,):
        raise ValueError(f"marker offset must have shape (3,), got {offset.shape}")
    if not (np.isfinite(center).all() and np.isfinite(pose).all() and np.isfinite(offset).all()):
        raise ValueError("marker geometry must contain only finite values")
    # ``locate_marker.marker_frame`` was deliberately defined for a marker on
    # the table plane. Reject a side-mounted tag instead of silently treating
    # its ambiguous normal as the basket's vertical release axis.
    if float(pose[2, 2]) < 0.9:
        raise ValueError("basket marker must be mounted horizontally (local +z upward)")
    return center + pose[:3, :3] @ offset
