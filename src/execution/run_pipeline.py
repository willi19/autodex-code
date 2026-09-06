#!/usr/bin/env python3
"""Integrated run_auto pipeline with in-process pose-recovery actions.

This entry point intentionally reuses :mod:`src.execution.run_auto` for the
normal trial lifecycle: calibration, one camera controller, FoundPose
orchestrator, CUDA planner, executor, execution, labelling, coverage, and
cleanup are unchanged. Its only behavioural difference is that two recovery
branches reuse those already-live resources:

    exhausted/no-candidate/failed-plan tabletop -> in-process reset reorient
    -> fresh perception -> normal run_auto trial
    failed plan -> search (x, yaw) -> in-process rotate -> fresh perception
    -> normal run_auto trial

Both recoveries use the failed trial's perception instead of starting another
process or redoing hardware/FoundPose/planner initialisation. Perception is
deliberately repeated *after* the physical placement; that is a new scene
observation used by the normal trial, not duplicated setup work.
"""
from __future__ import annotations

import os
import sys
import time

# ``python src/execution/run_pipeline.py`` puts only this directory on
# sys.path. Match the existing execution scripts so direct CLI invocation can
# import the ``src`` package as well as project modules.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from src.execution import run_auto
from src.execution.rotate_obj_yaw import rotate_from_live_scene
from src.execution.scene_cfg import pose_world_to_scene_cfg
from autodex.planner.obstacles import add_obstacles
from autodex.utils.robot_config import CHARUCO_BOARD_11_CENTER_XY
from src.experiment.reset.reorient import reorient_from_live_scene


DEFAULT_HELD_SPEED_SCALE = 0.25


def _restart_stream_or_mark_failure(rcc, args, info: dict, recovery: str) -> dict:
    """Return the live controller to stream mode after an in-process recovery."""
    try:
        run_auto._rcc_start(rcc, "stream", False, fps=args.stream_fps)
        if args.stream_warmup_s > 0:
            time.sleep(args.stream_warmup_s)
    except Exception as stream_exc:
        info = dict(info)
        info["success"] = False
        info["reason"] = f"{recovery}_stream_restart_failed"
        info["stream_exception"] = repr(stream_exc)
    return info


def _rotate_in_process(**context) -> dict:
    """Adapter for ``run_auto.run_single_trial``'s recovery callback."""
    args = context["args"]
    rcc = context["rcc"]
    info = None
    try:
        info = rotate_from_live_scene(
            obj=context["obj"], hand=context["hand"], arm=args.arm,
            grasp_version=context["grasp_version"],
            planner=context["planner"], executor=context["executor"],
            scene_cfg=context["scene_cfg"],
            target_x=context["target_x"],
            target_y=context.get("target_y", float(CHARUCO_BOARD_11_CENTER_XY[1])),
            target_yaw_deg=context["target_yaw_deg"],
            tabletop_pose_stem=context["tabletop_pose_stem"],
            candidate_order=context["candidate_order"],
            priority_map=context["priority_map"],
            scene_type_filter=context["scene_type_filter"],
            scene_id=context["scene_id"],
            success_only=context["success_only"],
            skip_done=context["skip_done"],
            skip_scenes_with_success=context["skip_scenes_with_success"],
            cyl_axis_local=context["cyl_axis_local"],
            cyl_yaw_grid=context["cyl_yaw_grid"],
            held_speed_scale=DEFAULT_HELD_SPEED_SCALE,
            rcc=rcc,
        )
    finally:
        # rotate_from_live_scene stops the stream before physical movement.
        # Re-arm the SAME controller before run_auto begins the next normal
        # trial's FoundPose step; no second controller registration occurs.
        if info is not None:
            info = _restart_stream_or_mark_failure(rcc, args, info, "rotation")
        else:
            # Let run_auto's callback wrapper record the original exception,
            # but still make a best effort to leave cameras ready for an
            # operator-driven retry.
            _restart_stream_or_mark_failure(
                rcc, args, {"success": False}, "rotation")
    return info


def _reorient_in_process(**context) -> dict:
    """Adapter that reuses run_auto's perception, hardware and planner.

    The standalone reset script owns a full camera/FoundPose/planner/executor
    lifecycle.  Here only its reset-candidate planning and physical motion are
    invoked; the next ordinary run_auto trial supplies the one necessary fresh
    perception after the object has been placed.
    """
    args = context["args"]
    rcc = context["rcc"]
    sub = (f"{context['scene_prefix']}/{context['hand']}"
           if context["scene_prefix"] else context["hand"])
    lift_rel = os.path.join(
        "shared_data", "AutoDex", "experiment", args.exp_name, sub,
        context["obj"], os.path.basename(context["img_dir"]),
        "reorient_lift_check", "raw",
    )
    lift_abs = os.path.join(
        context["img_dir"], "reorient_lift_check", "raw", "images")
    # Reset candidates are table transitions.  Do not inherit wall/shelf/
    # clutter obstacles from the failed collection scene; that would make this
    # recovery differ from the retained standalone reorient policy.
    table_scene = pose_world_to_scene_cfg(
        context["pose_world"], context["c2r"], context["obj"],
        context["obj_root"],
    )
    table_scene = add_obstacles(table_scene, "table")

    info = None
    try:
        info = reorient_from_live_scene(
            obj=context["obj"], hand=context["hand"], arm=args.arm,
            target_j=context["target_j"], planner=context["planner"],
            executor=context["executor"], rcc=rcc, scene_cfg=table_scene,
            obj_root=context["obj_root"], grasp_version=args.grasp_version,
            lift_label_rel=lift_rel,
            lift_label_abs=lift_abs, held_speed_scale=DEFAULT_HELD_SPEED_SCALE,
        )
    finally:
        if info is not None:
            info = _restart_stream_or_mark_failure(rcc, args, info, "reorient")
        else:
            _restart_stream_or_mark_failure(
                rcc, args, {"success": False}, "reorient")
    return info


def main() -> None:
    run_auto.main(
        pose_adjust_handler=_rotate_in_process,
        reorient_handler=_reorient_in_process,
    )


if __name__ == "__main__":
    main()
