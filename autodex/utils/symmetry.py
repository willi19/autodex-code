"""Per-object rotational symmetry lookup.

Two sources of symmetry info:

1. ``src/scene_generation/symmetry.json`` — global registry. Currently only
   ``"type": "revolute"`` + ``"axis": "x"/"y"/"z"`` (continuous rotation).
2. ``{obj_path}/{obj}/processed_data/info/pose_symmetry.json`` — per-object
   tabletop-pose equivalence groups. A group of >1 sorted-index entries means
   those tabletop poses are physically the same under some discrete symmetry
   rotation. The axis+angle is derived from the rotation between paired
   tabletop pose 4×4 transforms on disk.

``get_cyl_axis_local`` checks both: revolute → axis from registry; otherwise
inspects per-obj pose_symmetry.json and derives axis from first multi-element
group. ``get_cyl_yaw_grid`` returns the matching angle grid (continuous N for
revolute; discrete order-N for pose-symmetric).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

_AXIS_VEC = {
    "x": np.array([1.0, 0.0, 0.0]),
    "y": np.array([0.0, 1.0, 0.0]),
    "z": np.array([0.0, 0.0, 1.0]),
}

_REGISTRY_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "src" / "scene_generation" / "symmetry.json"
)

_OBJ_ROOT = Path.home() / "shared_data" / "AutoDex" / "object" / "paradex"

_cache: Optional[dict] = None
_pose_sym_cache: dict = {}


def _load() -> dict:
    global _cache
    if _cache is not None:
        return _cache
    if not _REGISTRY_PATH.is_file():
        _cache = {}
        return _cache
    with open(_REGISTRY_PATH) as f:
        reg = json.load(f)
    reg.pop("_comment", None)
    _cache = reg
    return _cache


def _derive_pose_symmetry(obj_name: str) -> Optional[Tuple[np.ndarray, int]]:
    """Inspect ``{obj}/processed_data/info/pose_symmetry.json`` and the
    matching tabletop pose .npy files. Find the first equivalence group with
    >1 entry and derive the (axis_unit_vec_3, order) of the discrete
    rotational symmetry from the rotation between the two tabletop pose 4×4s.

    Returns (axis, order) or ``None`` if no per-obj symmetry / no multi-elem
    group / degenerate rotation.
    """
    if obj_name in _pose_sym_cache:
        return _pose_sym_cache[obj_name]
    psp = _OBJ_ROOT / obj_name / "processed_data" / "info" / "pose_symmetry.json"
    tt_dir = _OBJ_ROOT / obj_name / "processed_data" / "info" / "tabletop"
    if not psp.is_file() or not tt_dir.is_dir():
        _pose_sym_cache[obj_name] = None
        return None
    try:
        groups = json.load(open(psp))["groups"]
    except Exception:
        _pose_sym_cache[obj_name] = None
        return None
    # tabletop files are zero-padded 3-digit (e.g. 001.npy) per the pipeline
    # convention. Group entries are int pose indices.
    for grp in groups:
        if not isinstance(grp, (list, tuple)) or len(grp) < 2:
            continue
        a, b = int(grp[0]), int(grp[1])
        T_a = _load_tabletop_pose(tt_dir, a)
        T_b = _load_tabletop_pose(tt_dir, b)
        if T_a is None or T_b is None:
            continue
        # Rotation in object frame s.t. T_b ≈ R_sym @ T_a (rotation part only).
        # → R_sym = T_b.R @ T_a.R.T
        R_sym = T_b[:3, :3] @ T_a[:3, :3].T
        rotvec = Rotation.from_matrix(R_sym).as_rotvec()
        angle = float(np.linalg.norm(rotvec))
        if angle < 1e-3:
            continue
        axis = rotvec / angle
        order = max(2, int(round(2.0 * np.pi / angle)))
        _pose_sym_cache[obj_name] = (axis.astype(np.float64), order)
        return _pose_sym_cache[obj_name]
    _pose_sym_cache[obj_name] = None
    return None


def _load_tabletop_pose(tt_dir: Path, idx: int) -> Optional[np.ndarray]:
    for stem in (f"{idx:03d}", str(idx)):
        p = tt_dir / f"{stem}.npy"
        if p.is_file():
            arr = np.load(p)
            if arr.shape == (3, 3):
                T = np.eye(4); T[:3, :3] = arr
                return T
            if arr.shape == (4, 4):
                return arr
    return None


def get_cyl_axis_local(obj_name: str) -> Optional[np.ndarray]:
    """Return the object-frame symmetry axis as a (3,) unit vector, or ``None``
    if no symmetry is registered.

    Checks (in order):
      1. global ``symmetry.json`` (revolute OR discrete) — axis field.
      2. per-object ``pose_symmetry.json`` — derives axis from first
         multi-element group's tabletop pose pair.
    """
    info = _load().get(obj_name)
    if info is not None and info.get("axis") in _AXIS_VEC:
        t = info.get("type")
        if t in ("revolute", "discrete"):
            return _AXIS_VEC[info["axis"]].copy()
    derived = _derive_pose_symmetry(obj_name)
    if derived is None:
        return None
    axis, _order = derived
    return axis.copy()


def get_cyl_yaw_grid(obj_name: str,
                      n_continuous: int = 8) -> Optional[np.ndarray]:
    """Return a (N,) array of yaw angles (rad) to enumerate around the
    object's symmetry axis. ``None`` if no symmetry.

      - Continuous revolute (``symmetry.json type=revolute``):
        N = ``n_continuous``.
      - Discrete in registry (``symmetry.json type=discrete + order``):
        N = ``order``.
      - Discrete from per-obj ``pose_symmetry.json``: N = derived order.
    """
    info = _load().get(obj_name)
    if info is not None:
        t = info.get("type")
        if t == "revolute":
            return np.linspace(0.0, 2.0 * np.pi, n_continuous, endpoint=False)
        if t == "discrete":
            order = max(2, int(info.get("order", 2)))
            return np.linspace(0.0, 2.0 * np.pi, order, endpoint=False)
    derived = _derive_pose_symmetry(obj_name)
    if derived is None:
        return None
    _axis, order = derived
    return np.linspace(0.0, 2.0 * np.pi, order, endpoint=False)
