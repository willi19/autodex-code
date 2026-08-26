"""Map tabletop-pose indices between the object_processing and paradex trees.

The two asset trees enumerate the same physical resting poses in a DIFFERENT
order, and different parts of the pipeline are numbered against different trees:

* scene JSONs (``meta.pose_idx``) and v8 candidate ``pose_idx`` — object_processing
* reset/reorient cell dirs ``candidates/{hand}/reset/{obj}/reorient_{h}/{i}_{j}/``
  and their ``openpose_{i:03d}.npy`` files — paradex, because the generator
  (``src/grasp_generation/reorient/plan_reset.py:load_tabletop_pose``) reads
  ``{obj_path}/.../tabletop/{idx:03d}.npy`` with ``obj_path`` = paradex

So a v8 reorient target named by an object_processing stem cannot index a reset
cell directly. For attached_container ``op 000 ≡ paradex 001`` and vice versa —
using the raw int silently looks up the REVERSED transition, and op 002/007/010
find no cell at all and drop out of the target list without an error.

This maps between them by matching tabletop ROTATIONS (the poses are identical
geometry, only renumbered), so nothing has to be regenerated. The grasp payloads
themselves need no fixing: both trees' ``raw_mesh`` are byte-identical for these
objects, so the object frame — and every ``wrist_se3.npy`` in it — is shared.
"""
from __future__ import annotations

import glob
import os
from typing import Dict, Optional

import numpy as np

from autodex.utils.path import obj_path, get_obj_root

# A match must be this close, AND this much better than the runner-up. The
# margin is the important half: a wrong-but-close match is worse than failing,
# because it silently reorients the object to the wrong face.
TOL_DEG = 5.0
MARGIN_DEG = 2.0

_CACHE: Dict[tuple, Dict[str, int]] = {}


def rot_geodesic_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Geodesic distance between two rotations, in degrees."""
    cos = (np.trace(R_a.T @ R_b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(float(np.clip(cos, -1.0, 1.0)))))


def z_aligned_geodesic_deg(R_est: np.ndarray, R_tab: np.ndarray) -> float:
    """Geodesic distance AFTER optimally aligning via a world-z rotation.

    A tabletop pose can rest at any yaw, so two rotations differing only by a
    world-z turn are the SAME tabletop class. With ``M = R_est @ R_tab.T`` the
    optimal yaw is ``atan2(M[1,0] - M[0,1], M[0,0] + M[1,1])``.
    """
    M = R_est @ R_tab.T
    theta = np.arctan2(M[1, 0] - M[0, 1], M[0, 0] + M[1, 1])
    c, s = np.cos(theta), np.sin(theta)
    R_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return rot_geodesic_deg(R_est, R_z @ R_tab)


def cyl_aligned_geodesic_deg(R_est, R_tab, axis, n_cyl: int = 72) -> float:
    """``z_aligned_geodesic_deg`` that also folds over a symmetry axis.

    For a body of revolution, two tabletop poses differing by a rotation about
    its axis are the SAME resting pose. Comparing them with world-z alignment
    alone reports a large angle and the match is rejected — apple's op stem 015
    "differs" from its closest paradex pose by 104 degrees purely this way.
    """
    from scipy.spatial.transform import Rotation as _R
    axis = np.asarray(axis, dtype=float).reshape(3)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    best = float("inf")
    for th in np.linspace(0.0, 2.0 * np.pi, n_cyl, endpoint=False):
        R_cyl = _R.from_rotvec(axis * float(th)).as_matrix()
        d = z_aligned_geodesic_deg(R_est, R_tab @ R_cyl)
        if d < best:
            best = d
    return best


def symmetry_rotations(sym, n_cont: int = 72):
    """Every rotation that maps the object onto itself, as (3,3) matrices.

    ``sym`` is what ``get_asset_symmetry`` returns: a list of (axis, fold),
    fold=None meaning continuous. The set is the product over axes -- a Dinf
    can is Cinf about its long axis AND 2-fold about two perpendicular ones, so
    an end-over-end flip must be folded too or its two tabletop poses look
    180 deg apart. Always contains identity, so a symmetry-free object reduces
    to a plain comparison.
    """
    from scipy.spatial.transform import Rotation as _R
    mats = [np.eye(3)]
    for axis, fold in (sym or []):
        axis = np.asarray(axis, dtype=float).reshape(3)
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        angles = (np.linspace(0.0, 2 * np.pi, n_cont, endpoint=False)
                  if fold is None else
                  np.arange(int(fold)) * (2 * np.pi / int(fold)))
        step = [_R.from_rotvec(axis * float(t)).as_matrix() for t in angles]
        mats = [M @ S for M in mats for S in step]
    return mats


def _load_rots(root: str, obj_name: str) -> Dict[str, np.ndarray]:
    d = os.path.join(root, obj_name, "processed_data", "info", "tabletop")
    out: Dict[str, np.ndarray] = {}
    for f in sorted(glob.glob(os.path.join(d, "*.npy"))):
        T = np.load(f)
        out[os.path.basename(f)[:-4]] = T[:3, :3] if T.shape == (4, 4) else T
    return out


def reset_index_map(
    obj_name: str,
    obj_root: Optional[str] = None,
    legacy_root: Optional[str] = None,
    tol_deg: float = TOL_DEG,
    margin_deg: float = MARGIN_DEG,
) -> Dict[str, int]:
    """``{obj_root stem -> paradex int}`` for reset/reorient cell lookups.

    Identity (``{stem: int(stem)}``) when ``obj_root`` already IS the legacy
    tree, so callers can apply this unconditionally.

    Raises ``ValueError`` if any stem has no confident match — better a loud
    failure than reorienting to the wrong face.
    """
    root = obj_root or obj_path
    legacy = legacy_root or obj_path
    key = (obj_name, os.path.realpath(root), os.path.realpath(legacy))
    if key in _CACHE:
        return _CACHE[key]

    src = _load_rots(root, obj_name)
    if os.path.realpath(root) == os.path.realpath(legacy):
        out = {s: int(s) for s in src}
        _CACHE[key] = out
        return out

    dst = _load_rots(legacy, obj_name)
    if not src or not dst:
        raise ValueError(f"{obj_name}: missing tabletop poses "
                         f"(src={len(src)} under {root}, dst={len(dst)} under {legacy})")

    # A body of revolution rests identically at any angle about its axis, so
    # fold over that axis before comparing or every match looks wrong.
    from autodex.utils.symmetry import get_asset_symmetry
    sym = get_asset_symmetry(obj_name, root) or get_asset_symmetry(obj_name, legacy)
    rots = symmetry_rotations(sym)

    def _dist(Ra, Rb):
        return min(z_aligned_geodesic_deg(Ra, Rb @ S) for S in rots)

    out: Dict[str, int] = {}
    for stem, R in src.items():
        errs = sorted(((_dist(R, Rd), k) for k, Rd in dst.items()))
        best_err, best_k = errs[0]
        runner_up = errs[1][0] if len(errs) > 1 else float("inf")
        if best_err > tol_deg:
            raise ValueError(
                f"{obj_name}: tabletop stem {stem} has no legacy match "
                f"(best {best_k} @ {best_err:.2f}deg > {tol_deg}deg tolerance)")
        if (runner_up - best_err) < margin_deg and runner_up > tol_deg:
            # Ambiguous: the runner-up is close behind but OUT of tolerance, so
            # the two are not interchangeable and we cannot tell which is meant.
            raise ValueError(
                f"{obj_name}: tabletop stem {stem} is ambiguous "
                f"(best {best_k} @ {best_err:.2f}deg, runner-up @ {runner_up:.2f}deg)")
        # runner_up also within tolerance => the legacy tree lists two poses
        # that are the same class (french_mustard has a pair 0.7deg apart).
        # Either maps correctly, so take the closer one instead of failing.
        out[stem] = int(best_k)
    _CACHE[key] = out
    return out


def to_reset_index(obj_name: str, stem, obj_root: Optional[str] = None) -> int:
    """Reset-cell (paradex) index for a tabletop ``stem`` in ``obj_root``.

    ``stem`` may be a stem string (``'007'``) or the int the caller already
    derived from one (``7``) — both resolve to the same entry.
    """
    m = reset_index_map(obj_name, obj_root)
    s = str(stem)
    if s in m:
        return m[s]
    for k, v in m.items():           # caller passed int('007') == 7
        if int(k) == int(s):
            return v
    raise KeyError(f"{obj_name}: no tabletop stem {stem!r} under "
                   f"{obj_root or obj_path} (have {sorted(m)})")


def describe_map(obj_name: str, version: str = "v8") -> str:
    """Human-readable ``op stem -> paradex cell index`` table, for logs."""
    root = get_obj_root(version)
    try:
        m = reset_index_map(obj_name, root)
    except ValueError as e:
        return f"{obj_name}: MAPPING FAILED — {e}"
    return f"{obj_name} ({version}): " + ", ".join(
        f"{s}->{i}" for s, i in sorted(m.items()))
