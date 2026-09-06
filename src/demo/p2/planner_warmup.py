"""One-time cuRobo prewarm used by P2 session runners.

The helpers here do not plan a grasp and never command the robot.  They build
only a structurally equivalent table/object world so ``GraspPlanner`` can
materialise CUDA graphs and joint-space trajopt before episode timing begins.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any


def planner_prewarm_scene(*, preferred_object: str,
                          object_root: Path) -> tuple[str, dict]:
    """Build a table world structurally identical to P2's first plan world.

    The synthetic pose is deliberately far from the retract configuration.
    A later real plan replaces it with the perceived target pose; it never
    consumes this placeholder pose as a physical grasp target.  ``apple`` is
    the known P2 fallback asset when an initially suggested generic object has
    not been prepared locally.
    """
    from src.execution.scene_cfg import (
        TABLE_SURFACE_Z,
        TABLE_THICKNESS_Z,
        find_planning_mesh,
    )

    objects = [preferred_object]
    if preferred_object != "apple":
        objects.append("apple")
    mesh_path = None
    selected = None
    for obj_name in objects:
        try:
            mesh_path = find_planning_mesh(obj_name, str(object_root))
        except FileNotFoundError:
            continue
        selected = obj_name
        break
    if mesh_path is None or selected is None:
        raise FileNotFoundError(
            "cannot find a planning mesh for planner prewarm: "
            + ", ".join(repr(value) for value in objects))
    return selected, {
        "mesh": {
            "target": {
                "pose": [1.85, 1.20, 0.25, 1.0, 0.0, 0.0, 0.0],
                "file_path": mesh_path,
            },
        },
        # Match pose_world_to_scene_cfg(...), 'table' exactly.  Same-object
        # first trials then need only a target-mesh pose update.
        "cuboid": {"table": {
            "dims": [2, 3, TABLE_THICKNESS_Z],
            "pose": [1.1, 0, TABLE_SURFACE_Z - TABLE_THICKNESS_Z / 2,
                     1, 0, 0, 0],
        }},
    }


def prewarm_planner(*, planner, preferred_object: str,
                    object_root: Path) -> dict[str, Any]:
    """Pay MotionGen/IK cold-start cost during unmeasured session setup."""
    warmup_object, scene_cfg = planner_prewarm_scene(
        preferred_object=preferred_object, object_root=object_root)
    print("[planner] prewarm: MotionGen graph + joint-space trajopt "
          f"({warmup_object}/table placeholder)...")
    info = dict(planner.warmup(scene_cfg, warmup_js_trajopt=True))
    info["placeholder_object"] = warmup_object
    print("[planner] prewarm complete: "
          f"{info['total_s']:.2f}s (first measured plan is warm)")
    return info
