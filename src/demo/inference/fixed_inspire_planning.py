"""Planning helpers specific to the fixed-success Inspire inference demo.

These helpers deliberately live outside ``banana_test``: the inference demo
replays object-frame grasps from its own physical-success library, whereas the
banana experiment selects entries from a tabletop candidate catalogue.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from autodex.utils.conversion import cart2se3
from src.demo.inference.grasp_library import (
    DemoGrasp,
    demo_planner_candidates,
)


# Keep the fixed-library reach screen consistent with the original banana demo
# lift used when these helpers lived there.
FIXED_INSPIRE_LIFT_HEIGHT_M = 0.10


def _rot_z(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _place_wrist(
    T_obj_grasp: np.ndarray,
    T_obj_in_wrist: np.ndarray,
    target_xy: Sequence[float],
    yaw_deg: float,
    obj_z: float,
    wrist_z: float,
) -> np.ndarray:
    """Return a wrist pose that holds the object over ``target_xy`` at yaw."""
    T = np.eye(4)
    T[:3, :3] = _rot_z(np.deg2rad(yaw_deg)) @ T_obj_grasp[:3, :3]
    T[:3, 3] = [target_xy[0], target_xy[1], obj_z]
    wrist = T @ np.linalg.inv(T_obj_in_wrist)
    wrist[2, 3] = wrist_z
    return wrist


def filter_fixed_inspire_by_place_reach(
    planner,
    grasps: Sequence[DemoGrasp],
    T_obj_grasp: np.ndarray,
    target_xy: Sequence[float],
    yaws: Sequence[float],
    symmetry_rotations: Optional[np.ndarray] = None,
) -> list[DemoGrasp]:
    """Keep fixed-success grasps with at least one reachable drop pose.

    A fixed grasp is stored as an object-frame wrist transform.  Test every
    declared symmetry equivalent and candidate drop yaw in a batched IK call;
    this is only a screen, so the normal planner still validates the selected
    approach/lift/carry trajectory afterwards.
    """
    if symmetry_rotations is None:
        symmetry_rotations = np.eye(3, dtype=np.float64)[None]
    probes: list[np.ndarray] = []
    owners: list[int] = []
    for index, grasp in enumerate(grasps):
        for rotation in symmetry_rotations:
            T_sym = np.eye(4)
            T_sym[:3, :3] = np.asarray(rotation, dtype=np.float64)
            wrist_obj = T_sym @ grasp.wrist_obj
            wrist_grasp = T_obj_grasp @ wrist_obj
            object_in_wrist = np.linalg.inv(wrist_obj)
            wrist_z = float(wrist_grasp[2, 3]) + FIXED_INSPIRE_LIFT_HEIGHT_M
            object_z = float(T_obj_grasp[2, 3]) + FIXED_INSPIRE_LIFT_HEIGHT_M
            for yaw in yaws:
                probes.append(_place_wrist(
                    T_obj_grasp, object_in_wrist, target_xy, float(yaw),
                    object_z, wrist_z,
                ))
                owners.append(index)
    if not probes:
        return list(grasps)
    feasible = np.asarray(planner.ik_pose_batch(np.asarray(probes))).reshape(-1)
    reachable = {owners[index] for index, ok in enumerate(feasible) if ok}
    return [grasp for index, grasp in enumerate(grasps) if index in reachable]


def plan_fixed_inspire_grasp(
    planner,
    scene_cfg: dict,
    grasps: Sequence[DemoGrasp],
    symmetry_rotations: Optional[np.ndarray] = None,
):
    """Plan one fixed-success Inspire grasp through the normal AutoDex path."""
    if symmetry_rotations is None:
        symmetry_rotations = np.eye(3, dtype=np.float64)[None]
    symmetry_rotations = np.asarray(symmetry_rotations, dtype=np.float64)
    if symmetry_rotations.ndim != 3 or symmetry_rotations.shape[1:] != (3, 3):
        raise ValueError("symmetry_rotations must have shape (N, 3, 3)")

    object_pose = cart2se3(scene_cfg["mesh"]["target"]["pose"])
    candidates = demo_planner_candidates(grasps, object_pose, symmetry_rotations)
    print("    [fixed-inspire] AutoDex planner: "
          f"{len(candidates[0])} grasp/symmetry candidates")
    result = planner.plan(
        scene_cfg, obj_name="fixed-inspire", grasp_version="fixed-inspire",
        skip_done=False, success_only=False, hand="inspire",
        candidate_order=[], candidate_override=candidates,
    )
    timing = dict(result.timing or {})
    timing.update({
        "source": "fixed-inspire",
        "n_source_candidates": len(grasps),
        "n_symmetry_variants": len(symmetry_rotations),
        "n_candidates": len(candidates[0]),
    })
    result.timing = timing
    return result
