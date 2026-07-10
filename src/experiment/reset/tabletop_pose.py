"""Classify a 4x4 robot-frame pose against an object's discrete tabletop poses.

Each object has a directory ``{obj_path}/{obj}/processed_data/info/tabletop/*.npy``
with 4x4 poses representing stable resting orientations on the table. The
classifier picks the closest one by rotation geodesic distance (deg).
"""
from __future__ import annotations

import glob
import os
from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from autodex.utils.path import obj_path
from autodex.utils.symmetry import get_cyl_axis_local


_CACHE: Dict[str, list] = {}


def _load_tabletop_poses(obj_name: str):
    """Return list of (filename, 4x4 pose) for this object, cached."""
    if obj_name in _CACHE:
        return _CACHE[obj_name]
    tabletop_dir = os.path.join(obj_path, obj_name, "processed_data",
                                "info", "tabletop")
    if not os.path.isdir(tabletop_dir):
        _CACHE[obj_name] = []
        return []
    files = sorted(glob.glob(os.path.join(tabletop_dir, "*.npy")))
    out = []
    for f in files:
        pose = np.load(f)
        if pose.shape == (4, 4):
            out.append((os.path.basename(f), pose))
        elif pose.shape == (3, 3):
            T = np.eye(4)
            T[:3, :3] = pose
            out.append((os.path.basename(f), T))
    _CACHE[obj_name] = out
    return out


def _rot_geodesic_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Geodesic distance between two rotations in degrees."""
    cos = (np.trace(R_a.T @ R_b) - 1.0) / 2.0
    cos = float(np.clip(cos, -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _z_aligned_geodesic_deg(R_est: np.ndarray, R_tab: np.ndarray) -> float:
    """Geodesic distance AFTER optimally aligning via world z-axis rotation.

    The tabletop pose can sit on the table at any yaw — so two poses that
    differ only by a rotation around world z represent the SAME tabletop
    class. Find the optimal R_z(theta) such that R_z(theta) @ R_tab is as
    close to R_est as possible, then return the residual geodesic distance.

    Closed form: with M = R_est @ R_tab.T, the optimal yaw is
        theta* = atan2(M[1,0] - M[0,1], M[0,0] + M[1,1]).
    """
    M = R_est @ R_tab.T
    theta = np.arctan2(M[1, 0] - M[0, 1], M[0, 0] + M[1, 1])
    c, s = np.cos(theta), np.sin(theta)
    R_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return _rot_geodesic_deg(R_est, R_z @ R_tab)


def _cyl_z_aligned_geodesic_deg(R_est: np.ndarray, R_tab: np.ndarray,
                                  axis_local: np.ndarray,
                                  n_cyl: int = 36) -> float:
    """Aligned geodesic distance for cylinder-symmetric objects.

    Two tabletop classes that differ by a rotation around the object's local
    symmetry axis are physically identical (e.g. a pepsi can rolled around
    its axis looks the same). Search optimal cyl_yaw around the axis (in
    object/tabletop frame) AND world-z yaw (closed form), return minimum
    residual geodesic.

    R_aligned = R_z_world(*) @ R_tab @ R_axis_local(theta)
    """
    axis = np.asarray(axis_local, dtype=np.float64).reshape(3)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    best = float("inf")
    for theta in np.linspace(0.0, 2.0 * np.pi, n_cyl, endpoint=False):
        R_cyl = Rotation.from_rotvec(axis * float(theta)).as_matrix()
        R_tab_rot = R_tab @ R_cyl
        d = _z_aligned_geodesic_deg(R_est, R_tab_rot)
        if d < best:
            best = d
    return best


def classify_tabletop_pose(pose_robot: np.ndarray,
                            obj_name: str) -> Optional[Dict[str, Any]]:
    """Find the closest tabletop pose to ``pose_robot`` by rotation.

    Returns None if no tabletop poses are defined for the object. Otherwise:
        {
          "idx": int,                # index in sorted file list
          "filename": str,           # e.g. "006.npy"
          "rot_err_deg": float,
          "n_candidates": int,
          "all_err_deg": list[float] # distance to every candidate
        }
    """
    poses = _load_tabletop_poses(obj_name)
    if not poses:
        return None
    R_est = pose_robot[:3, :3]
    # Cylinder-symmetric objects: also fold over the object's symmetry axis
    # so a rolled-around-its-axis cylinder still matches its tabletop class.
    axis_local = get_cyl_axis_local(obj_name)
    if axis_local is not None:
        errs = [_cyl_z_aligned_geodesic_deg(R_est, T[:3, :3], axis_local)
                for _, T in poses]
    else:
        errs = [_z_aligned_geodesic_deg(R_est, T[:3, :3]) for _, T in poses]
    idx = int(np.argmin(errs))
    fname, _ = poses[idx]
    return {
        "idx": idx,
        "filename": fname,
        "rot_err_deg": float(errs[idx]),
        "n_candidates": len(poses),
        "all_err_deg": [float(e) for e in errs],
    }
