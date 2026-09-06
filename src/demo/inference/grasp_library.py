"""Load the fixed Inspire success-grasp library used by the demo runner.

The library deliberately has no candidate-version or hand arguments.  It is
the union of exactly the two real-execution stores agreed for the demo:

* ``~/shared_data/autodex_dataset/selected_100_inspire`` — positive human
  reviews, replayed from ``executed_grasp``;
* ``~/shared_data/AutoDex/experiment/v8/inspire`` — successful v8 runs.

All returned wrist transforms are object-frame transforms, so callers can
apply a newly perceived object pose without needing to know where the source
trial was performed.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


SELECTED_100_INSPIRE_ROOT = Path.home() / "shared_data/autodex_dataset/selected_100_inspire"
V8_INSPIRE_EXPERIMENT_ROOT = Path.home() / "shared_data/AutoDex/experiment/v8/inspire"
# Episodes written by this demo runner.  It stores under its own version so it
# never writes into the collection tree, but a demo success is a real physical
# success and stays a first-class source here.
V8_DEMO_INSPIRE_EXPERIMENT_ROOT = (
    Path.home() / "shared_data/AutoDex/experiment/v8_demo/inspire")
# This is not a third success source: it supplies the original object-frame
# candidate files referenced by a successful v8 episode's ``scene_info``.
# The episode result remains the sole v8 success criterion.
V8_INSPIRE_CANDIDATE_ROOT = Path.home() / "shared_data/AutoDex/candidates/inspire/v8"
# selected_100's executed episodes retain the real wrist/hand state but not a
# stable pointer to their original generated candidate.  The sibling pool
# restores the collision-checked pregrasp + grasp joint values.
SELECTED_100_INSPIRE_CANDIDATE_ROOT = (
    Path.home() / "shared_data/AutoDex/candidates/inspire/selected_100")


@dataclass(frozen=True)
class DemoGrasp:
    """One physically successful Inspire grasp expressed in the object frame."""

    source: str
    episode: Path
    wrist_obj: np.ndarray
    pregrasp: np.ndarray
    grasp: np.ndarray


def _read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _read_array(path: Path, shape: tuple[int, ...]) -> np.ndarray | None:
    try:
        value = np.asarray(np.load(path), dtype=np.float32)
    except (OSError, ValueError, EOFError):
        return None
    if value.shape != shape or not np.isfinite(value).all():
        return None
    return value


def _read_invertible_transform(path: Path) -> np.ndarray | None:
    """Read a finite 4x4 transform that downstream planning can invert.

    A malformed archive must remove only that source episode, not crash the
    entire live demo during the marker reachability preflight.
    """
    value = _read_array(path, (4, 4))
    if value is None:
        return None
    try:
        if not np.isfinite(np.linalg.det(value)) or abs(np.linalg.det(value)) < 1e-8:
            return None
        np.linalg.inv(value)
    except np.linalg.LinAlgError:
        return None
    return value


def inspire_action_to_qpos(action: np.ndarray) -> np.ndarray:
    """Invert the Inspire controller-action ordering used by the executors.

    ``squeeze_hand.npy`` in a v8 run stores controller units (0=open,
    1000=closed) ordered pinky through thumb.  The planner needs radians in
    ``[thumb_yaw, thumb_pitch, index, middle, ring, pinky]`` order.
    """
    action = np.asarray(action, dtype=np.float32).reshape(-1)
    if action.shape != (6,) or not np.isfinite(action).all():
        raise ValueError("Inspire action must be six finite values")
    limits = np.array([1.15, 0.55, 1.6, 1.6, 1.6, 1.6], dtype=np.float32)
    controller_order = np.array([action[5], action[4], action[3],
                                 action[2], action[1], action[0]])
    return np.clip((1.0 - controller_order / 1000.0) * limits, 0.0, limits)


def _selected_candidate_hand_poses(obj_name: str, candidate_root: Path):
    """Return valid original candidate hand poses for one selected_100 object."""
    object_root = candidate_root / obj_name
    if not object_root.is_dir():
        return []
    candidates = []
    for pregrasp_path in object_root.rglob("pregrasp_pose.npy"):
        base = pregrasp_path.parent
        wrist = _read_invertible_transform(base / "wrist_se3.npy")
        pregrasp = _read_array(pregrasp_path, (6,))
        grasp = _read_array(base / "grasp_pose.npy", (6,))
        if wrist is not None and pregrasp is not None and grasp is not None:
            candidates.append((base, wrist, pregrasp, grasp))
    return candidates


def _restore_selected_hand_pose(recorded_wrist: np.ndarray,
                                recorded_pose: np.ndarray,
                                candidates) -> tuple[np.ndarray, np.ndarray, Path | None]:
    """Recover the source candidate fingers for a selected successful replay.

    selected_100's ``executed_grasp/grasp_pose.npy`` is the hand state at the
    real grasp event. For legacy records it matches either the source
    candidate's pregrasp or grasp vector; it is *not* valid to replace it with
    a universal zero/open pose. Prefer the exact candidate match and keep the
    source candidate's pair of collision-checked hand configurations. If an
    archive is incomplete, use the observed hand state for both phases rather
    than inventing the unsafe all-zero pose.
    """
    if candidates:
        def score(candidate):
            _base, wrist, pregrasp, grasp = candidate
            hand_error = min(np.linalg.norm(pregrasp - recorded_pose),
                             np.linalg.norm(grasp - recorded_pose))
            # Executed FK can differ slightly from its nominal candidate
            # wrist, so use it only as a tie-breaker after finger matching.
            wrist_error = np.linalg.norm(wrist - recorded_wrist)
            return hand_error, wrist_error

        best = min(candidates, key=score)
        if score(best)[0] <= 1e-4:
            base, _wrist, pregrasp, grasp = best
            return pregrasp.copy(), grasp.copy(), base

    return recorded_pose.copy(), recorded_pose.copy(), None


def _iter_selected_100(obj_name: str, root: Path,
                       candidate_root: Path) -> Iterable[DemoGrasp]:
    object_root = root / obj_name
    if not object_root.is_dir():
        return
    candidates = _selected_candidate_hand_poses(obj_name, candidate_root)
    for episode in sorted(path for path in object_root.iterdir() if path.is_dir()):
        label = _read_json(episode / "human_success_label.json")
        if not label or label.get("reviewed") is not True or label.get("human_success") is not True:
            continue
        wrist = _read_invertible_transform(episode / "executed_grasp/wrist_se3.npy")
        grasp = _read_array(episode / "executed_grasp/grasp_pose.npy", (6,))
        if wrist is None or grasp is None:
            continue
        pregrasp, grasp, _source_candidate = _restore_selected_hand_pose(
            wrist, grasp, candidates)
        yield DemoGrasp(
            source="selected_100_inspire",
            episode=episode,
            wrist_obj=wrist,
            grasp=grasp,
            pregrasp=pregrasp,
        )


def _v8_candidate_grasp(result: dict, obj_name: str, candidate_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Resolve the exact generated candidate selected by a successful v8 run.

    This preserves the original object-frame convention for cylinders and
    spheres, where the planning scene may have snapped an observed orientation
    before saving ``plan/wrist_se3.npy``.  Older/incomplete episodes fall back
    to their self-contained plan files below.
    """
    scene_info = result.get("scene_info")
    if not isinstance(scene_info, (list, tuple)) or len(scene_info) != 3:
        return None
    object_root = candidate_root / obj_name
    if not object_root.is_dir():
        # On the NAS v8 pools are commonly ``<object>.tar.gz`` beside the
        # expanded object directory. Reuse the generic archive cache used by
        # the fallback planner so successful v8 episodes retain their exact
        # original candidate hand pose for every object.
        from autodex.utils.path import resolve_candidate_object_path

        resolved = resolve_candidate_object_path(
            candidate_root.parent, candidate_root.name, obj_name)
        if resolved is None:
            return None
        object_root = Path(resolved)
    source = object_root.joinpath(*(str(part) for part in scene_info))
    wrist = _read_invertible_transform(source / "wrist_se3.npy")
    pregrasp = _read_array(source / "pregrasp_pose.npy", (6,))
    grasp = _read_array(source / "grasp_pose.npy", (6,))
    if wrist is None or pregrasp is None or grasp is None:
        return None
    return wrist, pregrasp, grasp


def _iter_v8(obj_name: str, root: Path, candidate_root: Path) -> Iterable[DemoGrasp]:
    object_root = root / obj_name
    if not object_root.is_dir():
        return
    for episode in sorted(path for path in object_root.iterdir() if path.is_dir()):
        result = _read_json(episode / "result.json")
        if not result or result.get("success") is not True:
            continue
        candidate = _v8_candidate_grasp(result, obj_name, candidate_root)
        if candidate is not None:
            wrist, pregrasp, grasp = candidate
            yield DemoGrasp(
                source="v8_inspire",
                episode=episode,
                wrist_obj=wrist,
                pregrasp=pregrasp,
                grasp=grasp,
            )
            continue
        wrist_robot = _read_invertible_transform(episode / "plan/wrist_se3.npy")
        pose_world = _read_array(episode / "pose_world.npy", (4, 4))
        c2r = _read_array(episode / "C2R.npy", (4, 4))
        squeeze_action = _read_array(episode / "squeeze_hand.npy", (6,))
        try:
            trajectory = np.asarray(np.load(episode / "plan/traj.npy"), dtype=np.float32)
        except (OSError, ValueError, EOFError):
            continue
        if (wrist_robot is None or pose_world is None or c2r is None
                or squeeze_action is None or trajectory.ndim != 2
                or trajectory.shape[0] == 0 or trajectory.shape[1] < 6
                or not np.isfinite(trajectory).all()):
            continue
        try:
            # ``plan/wrist_se3.npy`` is in the source robot frame.  Re-express
            # it in the source object frame so it can be reused at today's pose.
            source_robot_obj = np.linalg.inv(c2r) @ pose_world
            wrist_obj = np.linalg.inv(source_robot_obj) @ wrist_robot
            grasp = inspire_action_to_qpos(squeeze_action)
        except np.linalg.LinAlgError:
            continue
        if not np.isfinite(np.linalg.det(wrist_obj)) or abs(np.linalg.det(wrist_obj)) < 1e-8:
            continue
        yield DemoGrasp(
            source="v8_inspire",
            episode=episode,
            wrist_obj=wrist_obj.astype(np.float32),
            # Arm DOF differs between source experiments, but the final six
            # columns are always Inspire joints.
            pregrasp=trajectory[-1, -6:].astype(np.float32),
            grasp=grasp.astype(np.float32),
        )


def load_demo_grasps(
    obj_name: str,
    *,
    selected_root: Path = SELECTED_100_INSPIRE_ROOT,
    selected_candidate_root: Path = SELECTED_100_INSPIRE_CANDIDATE_ROOT,
    v8_root: Path = V8_INSPIRE_EXPERIMENT_ROOT,
    v8_candidate_root: Path = V8_INSPIRE_CANDIDATE_ROOT,
    v8_demo_root: Path = V8_DEMO_INSPIRE_EXPERIMENT_ROOT,
) -> list[DemoGrasp]:
    """Return all valid, physically successful demo grasps for ``obj_name``.

    Source order is demo episodes, then the v8 collection tree, then
    selected_100, so the most recent executions are attempted before the older
    dataset while retaining every fixed source.  Both v8 namespaces reference
    the same candidate pool, so they share ``v8_candidate_root``.
    """
    grasps = [*_iter_v8(obj_name, Path(v8_demo_root), Path(v8_candidate_root)),
              *_iter_v8(obj_name, Path(v8_root), Path(v8_candidate_root)),
              *_iter_selected_100(obj_name, Path(selected_root),
                                  Path(selected_candidate_root))]
    if not grasps:
        return []

    # Identical episodes can be re-exported.  Do not waste a robot attempt on
    # duplicates, but preserve different finger configurations at one wrist.
    unique: list[DemoGrasp] = []
    fingerprints: set[bytes] = set()
    for grasp in grasps:
        fingerprint = np.round(
            np.concatenate([grasp.wrist_obj.reshape(-1), grasp.grasp]), 5
        ).astype(np.float32).tobytes()
        if fingerprint not in fingerprints:
            fingerprints.add(fingerprint)
            unique.append(grasp)
    return unique


def demo_symmetry_rotations(
    obj_name: str,
    *,
    obj_root: str | Path | None = None,
    n_continuous: int = 8,
) -> tuple[np.ndarray, dict]:
    """Return object-frame rotations that preserve the demo object's shape.

    A successful wrist pose is stored in the object's coordinate frame.  When
    FoundPose picks a different, but physically equivalent, orientation for a
    symmetric object, replaying only that one transform can make a previously
    successful grasp look unreachable.  These rotations expand that wrist
    pose before planning: ``T_object @ R_sym @ T_wrist_object``.

    Discrete symmetries are enumerated exactly.  A continuous symmetry needs a
    finite live-demo budget, so it is sampled uniformly (eight orientations by
    default, matching the main planner's cylinder-symmetry search).
    """
    if n_continuous < 1:
        raise ValueError("n_continuous must be positive")

    # Keep these imports lazy: the grasp-library reader is also used by small
    # offline tools that deliberately do not install the complete planner
    # stack.
    from scipy.spatial.transform import Rotation
    from autodex.utils.symmetry import (
        get_asset_symmetry,
        get_cyl_axis_local,
        get_cyl_yaw_grid,
    )

    sym = get_asset_symmetry(obj_name, obj_root)
    source = "asset" if sym else "none"
    has_continuous_axis = bool(sym and any(fold is None for _axis, fold in sym))
    rotations = [np.eye(3, dtype=np.float64)]

    if sym:
        # Take the product over all declared axes.  This matters for Dinf
        # objects: their continuous long-axis symmetry alone does not include
        # the physically equivalent end-over-end flips.
        for axis, fold in sym:
            axis = np.asarray(axis, dtype=np.float64).reshape(3)
            norm = np.linalg.norm(axis)
            if norm <= 1e-9:
                continue
            axis /= norm
            angles = (
                np.linspace(0.0, 2.0 * np.pi, n_continuous, endpoint=False)
                if fold is None
                else np.arange(max(1, int(fold)), dtype=np.float64)
                * (2.0 * np.pi / max(1, int(fold)))
            )
            per_axis = [Rotation.from_rotvec(axis * float(angle)).as_matrix()
                        for angle in angles]
            rotations = [current @ variant
                         for current in rotations for variant in per_axis]
    else:
        # Older assets can lack object_processing's symmetry.json.  Preserve
        # the established registry/tabletop fallback rather than silently
        # losing symmetry support for those objects.
        axis = get_cyl_axis_local(obj_name)
        angles = get_cyl_yaw_grid(obj_name)
        if axis is not None and angles is not None and len(angles) > 1:
            source = "registry_or_tabletop"
            axis = np.asarray(axis, dtype=np.float64).reshape(3)
            axis /= np.linalg.norm(axis)
            rotations = [Rotation.from_rotvec(axis * float(angle)).as_matrix()
                         for angle in angles]

    # A product of multiple discrete declarations can contain equivalent
    # matrices.  Do not send duplicate motion plans to the live robot.
    unique: list[np.ndarray] = []
    seen: set[bytes] = set()
    for rotation in rotations:
        rotation = np.asarray(rotation, dtype=np.float64)
        key = np.round(rotation, decimals=8).tobytes()
        if key not in seen:
            seen.add(key)
            unique.append(rotation)
    return np.asarray(unique, dtype=np.float64), {
        "source": source,
        "n_variants": len(unique),
        "continuous_samples": int(n_continuous) if has_continuous_axis else None,
    }


def demo_planner_candidates(
    grasps: Iterable[DemoGrasp],
    object_pose: np.ndarray,
    symmetry_rotations: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[list[str]]]:
    """Convert fixed success grasps to AutoDex planner-ready candidates.

    The returned wrists are already in the current robot frame.  They can be
    passed to :meth:`GraspPlanner.plan` through its explicit-candidate input,
    which preserves the normal AutoDex collision filter, grasp/lift IK checks,
    near-start IK selection, and finger-refined joint-space planning.
    """
    object_pose = np.asarray(object_pose, dtype=np.float64)
    rotations = np.asarray(symmetry_rotations, dtype=np.float64)
    if object_pose.shape != (4, 4):
        raise ValueError("object_pose must have shape (4, 4)")
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError("symmetry_rotations must have shape (N, 3, 3)")

    wrist_se3, pregrasp, grasp_pose, scene_info = [], [], [], []
    for grasp in grasps:
        for symmetry_index, rotation in enumerate(rotations):
            T_sym = np.eye(4, dtype=np.float64)
            T_sym[:3, :3] = rotation
            wrist_se3.append(object_pose @ T_sym @ grasp.wrist_obj)
            pregrasp.append(np.asarray(grasp.pregrasp, dtype=np.float32))
            grasp_pose.append(np.asarray(grasp.grasp, dtype=np.float32))
            # Keep the source episode and variant in the normal PlanResult
            # scene_info slot so executor/result logging stays uniform.
            scene_info.append([
                "fixed-inspire",
                grasp.source,
                str(grasp.episode),
                str(symmetry_index),
            ])
    if not wrist_se3:
        return (
            np.empty((0, 4, 4), dtype=np.float64),
            np.empty((0, 6), dtype=np.float32),
            np.empty((0, 6), dtype=np.float32),
            [],
        )
    return (
        np.asarray(wrist_se3, dtype=np.float64),
        np.asarray(pregrasp, dtype=np.float32),
        np.asarray(grasp_pose, dtype=np.float32),
        scene_info,
    )
