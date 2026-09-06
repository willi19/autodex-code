#!/usr/bin/env python3
"""Run consecutive P2 semantic-routing trials for Franka speed experiments.

This is the speed-study counterpart of :mod:`src.demo.p2.run_auto`.  It
reuses the normal P2/inference pipeline unchanged (capture, FoundPose,
silhouette refinement, Qwen semantic routing, grasp selection, cuRobo
planning, task-only AVI recording, and Franka execution), but changes the
*experimental condition* recorded for each take:

* object identity is still selected before every trial (Enter reuses it);
* P2 grid-location labels are deliberately not collected;
* ``arm_speed_scale``, ``traj_speed``, ``joint_vmax``, and ``accel_max`` are
  selected before every trial (Enter restores the original ``FrankaExecutor``
  default for that field, rather than reusing the preceding trial's value);
* the complete relevant ``FrankaExecutor`` configuration and the four tuned
  fields are written into the normal episode metadata; and
* episodes are isolated below ``experiment/v8_speed/inspire/<object>/<stamp>``.

Run after the normal P2 capture daemons are available::

    bash scripts/init_daemons.sh start --p2-semantic
    python src/demo/p2/run_auto_speed.py --arm franka --execute

All unrecognised options are forwarded unchanged to
``src/demo/inference/run_demo.py``.  This runner requires ``--arm franka``:
the recorded controls are specific to ``FrankaExecutor``.
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.demo.p2.recording import P2RecordingRuntime
from src.demo.p2.planner_warmup import prewarm_planner
from src.demo.p2.run_auto import (
    _TRIAL_SEPARATOR,
    _base_inference_args,
    _episode_args,
    _format_seconds,
    _make_episode_dir,
    _prepare_object,
    _prompt_object,
    _write_json,
)
from src.demo.p2.semantic_router import P2_QWEN_MODEL, P2SemanticRouter


SPEED_PROTOCOL_ID = "p2_execution_speed"
SPEED_DEMO_VERSION = "v8_speed"
SPEED_TUNABLE_FIELDS = (
    "arm_speed_scale",
    "traj_speed",
    "joint_vmax",
    "accel_max",
)
# These are all public scalar constructor controls of FrankaExecutor.  They
# are copied on every take so a result is self-describing even if executor
# defaults evolve after this speed study has finished.
FRANKA_EXECUTOR_RECORD_FIELDS = (
    "hand_name",
    "dt",
    "squeeze_level",
    "arm_speed_scale",
    "ctrl_dt",
    "joint_vmax",
    "pos_kp",
    "follow_tol",
    "vel_smooth",
    "traj_dt",
    "traj_speed",
    "held_speed_scale",
    "max_lead",
    "land_tol",
    "follow_timeout_s",
    "follow_log_every_s",
    "accel_max",
)


def _parse_args(argv: list[str] | None) -> tuple[argparse.Namespace, list[str]]:
    """Parse runner-only controls and leave normal inference flags untouched."""
    parser = argparse.ArgumentParser(
        description=__doc__, add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--obj", metavar="OBJECT",
                        help="initial object shown at the first prompt")
    parser.add_argument("--max-trials", type=int, default=0,
                        help="stop after this many trials; 0 keeps prompting")
    parser.add_argument("--semantic-port", type=int, default=5010,
                        help="P2 crop PUB port (must match init_daemon)")
    parser.add_argument("--semantic-timeout-s", type=float, default=20.0,
                        help="wait for generic Qwen route after FoundPose collection")
    parser.add_argument("--vlm-model", default=P2_QWEN_MODEL,
                        help="Qwen2.5-VL-3B-Instruct model id, loaded NF4")
    parser.add_argument("--no-video", action="store_true",
                        help="do not record execution AVI files on capture PCs")
    parser.add_argument("--video-fps", type=int, default=30,
                        help="execution AVI FPS on every capture PC")
    parser.add_argument("-h", "--help", action="store_true")
    speed_args, remaining = parser.parse_known_args(argv)
    if speed_args.help:
        parser.print_help()
        print("\nAll remaining options are forwarded unchanged to "
              "src/demo/inference/run_demo.py.")
        raise SystemExit(0)
    if speed_args.max_trials < 0:
        parser.error("--max-trials must be non-negative")
    if not 1 <= speed_args.semantic_port <= 65535:
        parser.error("--semantic-port must be in 1..65535")
    if speed_args.semantic_timeout_s <= 0:
        parser.error("--semantic-timeout-s must be positive")
    if speed_args.video_fps <= 0:
        parser.error("--video-fps must be positive")
    return speed_args, remaining


def _executor_config(executor) -> dict[str, Any]:
    """Return only stable, operator-meaningful FrankaExecutor controls."""
    return {
        field: getattr(executor, field)
        for field in FRANKA_EXECUTOR_RECORD_FIELDS
    }


def _speed_controls(config: dict[str, Any]) -> dict[str, float]:
    return {field: float(config[field]) for field in SPEED_TUNABLE_FIELDS}


def _format_control(value: float) -> str:
    """Keep prompts concise while retaining exact normal Python float values."""
    return f"{float(value):g}"


def _prompt_positive_control(label: str, *, default: float, units: str) -> float | None:
    """Prompt one speed control; blank means the original executor default."""
    while True:
        raw = input(
            f"{label} ({units}; Enter=default {_format_control(default)}, q=finish): "
        ).strip()
        if raw.lower() in {"q", "quit", "exit"}:
            return None
        if not raw:
            return float(default)
        try:
            value = float(raw)
        except ValueError:
            print("Enter a positive finite number, or Enter for the default.")
            continue
        if np.isfinite(value) and value > 0.0:
            return value
        print("Enter a positive finite number, or Enter for the default.")


def _prompt_speed_controls(defaults: dict[str, float]) -> dict[str, float] | None:
    """Prompt the four execution controls, always defaulting to baseline.

    ``defaults`` is captured just after FrankaExecutor construction.  It is
    intentionally never updated, so a blank entry cannot accidentally retain
    the previous take's experimental speed setting.
    """
    print("[speed] Enter uses the original FrankaExecutor default for this trial.")
    requested: dict[str, float] = {}
    for field, units in (
        ("arm_speed_scale", "blocking-move speed scale"),
        ("traj_speed", "trajectory timing scale"),
        ("joint_vmax", "rad/s"),
        ("accel_max", "rad/s^2"),
    ):
        value = _prompt_positive_control(field, default=defaults[field], units=units)
        if value is None:
            return None
        requested[field] = value
    return requested


def _apply_speed_controls(executor, requested: dict[str, float]) -> None:
    """Set exactly the documented per-trial Franka execution controls."""
    for field in SPEED_TUNABLE_FIELDS:
        setattr(executor, field, float(requested[field]))


def _print_trial_summary(*, trial_index: int, run_dir: Path, object_name: str,
                         executor_config: dict[str, Any], result: dict[str, Any]) -> None:
    """Print the speed condition plus automatic pipeline timing only.

    Speed trials intentionally have no operator f/g/p/c/a annotation.  The
    printed record therefore describes only the automatic pipeline state and
    the execution condition that is persisted for later timing analysis.
    """
    timing = result.get("timing") if isinstance(result.get("timing"), dict) else {}
    perception = timing.get("perception") if isinstance(timing.get("perception"), dict) else {}
    planning = timing.get("planning") if isinstance(timing.get("planning"), dict) else {}
    execution = timing.get("execution") if isinstance(timing.get("execution"), dict) else {}
    task = execution.get("task") if isinstance(execution.get("task"), dict) else {}
    reset = execution.get("reset") if isinstance(execution.get("reset"), dict) else {}
    semantic = result.get("semantic") if isinstance(result.get("semantic"), dict) else {}
    predicted = semantic.get("prediction")
    route = (f"{predicted} -> {semantic.get('basket')} / "
             f"{float(semantic['bearing_deg']):+.1f} deg"
             if predicted and semantic.get("bearing_deg") is not None else "n/a")
    auto = "SUCCESS" if result.get("success") is True else str(
        result.get("reason", "not_run"))
    reset_text = (_format_seconds(reset.get("total_s"))
                  if reset.get("performed") else "not performed")
    controls = _speed_controls(executor_config)

    print(f"\n{_TRIAL_SEPARATOR}")
    print(f"[P2 SPEED TRIAL {trial_index:02d} COMPLETE] {object_name}")
    print(f"  speed    : arm_speed_scale={controls['arm_speed_scale']:g}, "
          f"traj_speed={controls['traj_speed']:g}, "
          f"joint_vmax={controls['joint_vmax']:g} rad/s, "
          f"accel_max={controls['accel_max']:g} rad/s^2")
    print(f"  pipeline : {auto}")
    print(f"  route    : {route}")
    print("  timing   : "
          f"perception={_format_seconds(perception.get('total_s'))}, "
          f"planning={_format_seconds(planning.get('total_s'))}, "
          f"task={_format_seconds(task.get('total_s'))}, "
          f"reset={reset_text}, "
          f"execution={_format_seconds(execution.get('total_s'))}")
    print(f"  saved    : {run_dir / 'result.json'}")
    print(f"{_TRIAL_SEPARATOR}\n")


def main(argv: list[str] | None = None) -> None:
    speed_args, remaining = _parse_args(argv)
    # Parse before connecting hardware so unsupported inference settings are
    # rejected before a camera/robot session is acquired.
    initial_object = speed_args.obj or "apple"
    base_args = _base_inference_args(initial_object, remaining)
    if base_args.arm != "franka":
        raise SystemExit("run_auto_speed.py requires --arm franka")

    from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
    from paradex.utils.system import get_camera_list, get_pc_ip

    from autodex.perception.init_orchestrator import InitOrchestrator
    from autodex.planner import GraspPlanner
    from autodex.utils.path import get_obj_root, project_dir
    from src.demo.banana_test.run_demo import (
        ASSETS_BASE,
        CAM_PARAM_ROOT,
        _clear_camera_errors,
        _ensure_camera_lock,
        _load_calib,
        _planner_robot,
        _rcc_start,
        _safe,
        _stop_with_timeout,
        _warn_if_not_streaming,
        quiet_curobo,
    )
    from src.demo.inference import run_demo as inference
    from src.execution.franka_executor import FrankaExecutor
    from src.execution.scene_cfg import check_mesh_frame_match

    # v8's object-processing root is shared by the real trial and by the
    # one-time planner warmup.  Keeping the roots identical avoids hiding a
    # mesh-frame/configuration problem during setup.
    object_root = Path(get_obj_root(inference.GRASP_ASSET_VERSION))

    print(f"[p2-speed] protocol={SPEED_PROTOCOL_ID}; results={SPEED_DEMO_VERSION}; "
          "FRUIT -> left/+50.0 deg, NON_FRUIT -> right/-30.0 deg")
    print("[p2-speed] one-time setup: cameras/calibration, planner, robot, Qwen "
          "and recording trigger. Per trial: speed condition, FoundPose, VLM, "
          "plan, execute.")

    calib_dir = (Path(base_args.calib_dir).expanduser() if base_args.calib_dir
                 else sorted(CAM_PARAM_ROOT.iterdir())[-1])
    intrinsics, extrinsics, height, width = _load_calib(calib_dir)
    pc_ips = [get_pc_ip(pc) for pc in base_args.pc_list]
    pc_serials = {pc: get_camera_list(pc) for pc in base_args.pc_list}
    active = {serial for pc in base_args.pc_list for serial in pc_serials[pc]}
    intrinsics = {serial: value for serial, value in intrinsics.items()
                  if serial in active}
    extrinsics = {serial: value for serial, value in extrinsics.items()
                  if serial in active}
    print(f"calib: {calib_dir.name}  ({len(intrinsics)} cams, {height}x{width})")

    rcc = remote_camera_controller("p2_auto_speed", pc_list=base_args.pc_list,
                                   stall_timeout=15.0)
    orch = planner = executor = semantic_router = recording_runtime = None
    planner_prewarm: dict[str, Any] | None = None
    stream_started = False
    try:
        if not _ensure_camera_lock(rcc):
            raise RuntimeError("camera daemons are owned by another controller")
        if not _clear_camera_errors(rcc):
            raise RuntimeError("capture cameras remain in an error state")
        print(f"[stream] start @ {base_args.stream_fps} FPS...")
        _rcc_start(rcc, "stream", False, fps=base_args.stream_fps)
        stream_started = True
        time.sleep(base_args.stream_warmup_s)
        if not _warn_if_not_streaming(rcc):
            raise RuntimeError("camera stream did not start")

        orch = InitOrchestrator(
            pc_list=base_args.pc_list, capture_ips=pc_ips,
            port_mask=base_args.port_mask, port_pose=base_args.port_pose,
            port_cmd=base_args.port_cmd,
        )
        planner_robot = _planner_robot(base_args.arm, "inspire")
        print(f"[planner] warmup once ({planner_robot})...")
        planner = GraspPlanner(hand=planner_robot)
        quiet_curobo()
        # GraspPlanner normally creates MotionGen lazily inside the first
        # ``planner.plan`` call, after perception.  Do that work now, before
        # any trial begins, including the first joint-space trajopt kernel.
        planner_prewarm = prewarm_planner(
            planner=planner, preferred_object=initial_object,
            object_root=object_root)
        print("[executor] connect once (franka)...")
        executor = FrankaExecutor(hand_name="inspire")
        executor.set_speed_profile_planner(planner)
        # Snapshot immediately after construction.  A blank input on any later
        # trial must reset that one field to these original code defaults.
        speed_defaults = _speed_controls(_executor_config(executor))
        print("[speed] FrankaExecutor defaults: "
              f"arm_speed_scale={speed_defaults['arm_speed_scale']:g}, "
              f"traj_speed={speed_defaults['traj_speed']:g}, "
              f"joint_vmax={speed_defaults['joint_vmax']:g} rad/s, "
              f"accel_max={speed_defaults['accel_max']:g} rad/s^2")

        semantic_router = P2SemanticRouter(
            capture_ips=pc_ips, pc_serials=pc_serials,
            port=speed_args.semantic_port, model_id=speed_args.vlm_model,
            timeout_s=speed_args.semantic_timeout_s,
        )
        print("[p2-vlm] preload Qwen once...")
        semantic_router.preload()
        if not speed_args.no_video:
            serials = [serial for pc in base_args.pc_list
                       for serial in pc_serials[pc]]
            recording_runtime = P2RecordingRuntime(
                rcc=rcc, pc_list=base_args.pc_list, serials=serials,
                video_fps=speed_args.video_fps,
            )

        # Keep the normal P2 session boundary: clear-view once before the
        # first prompted take.  No blind home is issued after a failed take.
        print("[executor] moving once to clear-view home")
        executor.home(clear_view=True)

        previous_object = speed_args.obj
        initialized_object: str | None = None
        trial_index = 0

        while speed_args.max_trials == 0 or trial_index < speed_args.max_trials:
            object_name = _prompt_object(previous_object)
            if object_name is None:
                break
            requested_controls = _prompt_speed_controls(speed_defaults)
            if requested_controls is None:
                break
            _apply_speed_controls(executor, requested_controls)
            executor_config = _executor_config(executor)
            previous_object = object_name
            trial_index += 1
            args = _episode_args(object_name, remaining)
            run_dir = _make_episode_dir(Path(project_dir), object_name,
                                        SPEED_DEMO_VERSION)
            print(f"\n{_TRIAL_SEPARATOR}")
            print(f"[P2 SPEED TRIAL {trial_index:02d} START] {object_name}")
            print("  speed    : "
                  f"arm_speed_scale={requested_controls['arm_speed_scale']:g}, "
                  f"traj_speed={requested_controls['traj_speed']:g}, "
                  f"joint_vmax={requested_controls['joint_vmax']:g} rad/s, "
                  f"accel_max={requested_controls['accel_max']:g} rad/s^2")
            print(f"  episode  : {run_dir}")
            print(f"{_TRIAL_SEPARATOR}")

            preparation_started = time.perf_counter()
            object_reused = object_name == initialized_object
            trial_object_setup: dict[str, Any] | None = None
            try:
                if not object_reused:
                    trial_object_setup = _prepare_object(
                        object_name=object_name, args=args, orch=orch,
                        intrinsics=intrinsics, extrinsics=extrinsics,
                        image_hw=(height, width), pc_serials=pc_serials,
                        assets_base=ASSETS_BASE, object_root=object_root,
                        check_mesh_frame_match=check_mesh_frame_match,
                    )
                    initialized_object = object_name
                else:
                    trial_object_setup = {
                        "object": object_name,
                        "reused_from_previous_trial": True,
                    }
                semantic_router.set_evaluation_object(object_name)
                grasps = inference.load_demo_grasps(object_name)
                counts = {source: sum(g.source == source for g in grasps)
                          for source in ("v8_inspire", "selected_100_inspire")}
                print(f"[library] {len(grasps)} fixed Inspire successes: {counts}")
            except Exception as exc:
                # Preserve the durable episode-record behavior of normal P2.
                # This is a preflight abort: it never starts robot execution.
                result: dict[str, Any] | None = {
                    "object": object_name,
                    "success": False,
                    "reason": "object_prepare_failed",
                    "error": repr(exc),
                    "timing": {"preparation": {
                        "total_s": round(time.perf_counter() - preparation_started, 3)}},
                }
                grasps = []
            else:
                result = None

            trial_record: dict[str, Any] = {
                "protocol": SPEED_PROTOCOL_ID,
                "experiment_type": "execution_speed",
                "trial_index": trial_index,
                "created_at": dt.datetime.now().isoformat(timespec="seconds"),
                "object": object_name,
                "conditions": {
                    "arm": args.arm,
                    "hand": "inspire",
                    "grasp_version": inference.GRASP_ASSET_VERSION,
                    "franka_executor": executor_config,
                    "speed_tuned_fields": list(SPEED_TUNABLE_FIELDS),
                    "speed_defaults_at_session_start": speed_defaults,
                    "perception": args.perception,
                    "prompt": args.prompt,
                    "silhouette": {
                        "iterations": args.sil_iters,
                        "learning_rate": args.sil_lr,
                        "loss_max": args.sil_loss_max,
                        "loss_gate_ignored": args.ignore_sil_loss,
                    },
                    "motion": {
                        "lift_height_m": 0.15,
                        "transfer_mode": args.transfer_mode,
                        "drop_target": args.drop_target,
                        "box_xy": list(args.box_xy),
                        "carry_clearance_m": args.carry_clearance,
                    },
                    "semantic": {
                        "model_id": speed_args.vlm_model,
                        "crop_port": speed_args.semantic_port,
                        "timeout_s": speed_args.semantic_timeout_s,
                    },
                    "video": {
                        "enabled": not speed_args.no_video,
                        "fps": None if speed_args.no_video else speed_args.video_fps,
                    },
                    "calibration_dir": str(calib_dir),
                    "capture_pcs": list(base_args.pc_list),
                },
                "route_contract": {
                    "FRUIT": {"basket": "left", "bearing_deg": 50.0},
                    "NON_FRUIT": {"basket": "right", "bearing_deg": -30.0},
                },
                "resource_reuse": {
                    "camera_calibration_stream": True,
                    "planner_executor_qwen": True,
                    "planner_prewarm": planner_prewarm,
                    "foundpose_object_reused": object_reused,
                    "foundpose_object_setup": trial_object_setup,
                    "execution_video_enabled": not speed_args.no_video,
                },
                "timing": {
                    "runner_preparation_s": round(
                        time.perf_counter() - preparation_started, 3),
                },
            }
            _write_json(run_dir / "p2_trial.json", trial_record,
                        jsonable=inference._jsonable)
            target_info = {
                "target_type": "fixed_box",
                "center_robot": np.array([args.box_xy[0], args.box_xy[1], 0.04]),
                "release_target_robot": np.array([args.box_xy[0], args.box_xy[1], 0.04]),
            }
            _write_json(run_dir / "place_target.json", target_info,
                        jsonable=inference._jsonable)

            recorder = (recording_runtime.for_episode(
                run_dir=run_dir, project_root=Path(project_dir), executor=executor)
                if recording_runtime is not None else None)
            pipeline_started = result is None
            if pipeline_started:
                run_raised = False
                try:
                    result = inference.run_once(
                        args, orch=orch, planner=planner, executor=executor,
                        rcc=rcc,
                        target_xyz=np.asarray(target_info["center_robot"]),
                        run_dir=run_dir, grasps=grasps,
                        semantic_router=semantic_router,
                        execution_recorder=recorder,
                    )
                except KeyboardInterrupt:
                    run_raised = True
                    result = {"success": False, "reason": "interrupted"}
                    print("\n[interrupted trial]")
                except Exception as exc:
                    run_raised = True
                    # Do not command a surprise release/reset while an operator
                    # may need to inspect a held object.
                    result = {
                        "success": False,
                        "reason": "fatal",
                        "error": repr(exc),
                        "action": "stopped_without_robot_reset",
                    }
                    print(f"\n[FATAL] trial stopped without robot reset: {exc!r}")
                finally:
                    if recorder is not None and recorder.active:
                        _safe("p2 speed execution_recorder.stop",
                              lambda: recorder.stop(
                                  restart_stream=True, stream_fps=args.stream_fps))
                    elif recorder is None and run_raised:
                        # Match the normal P2 no-video failure recovery.
                        try:
                            rcc.stop()
                            _rcc_start(rcc, "stream", False, fps=args.stream_fps)
                        except Exception as exc:
                            print(f"[rcc] stream restore after trial failed: {exc!r}")

            assert result is not None  # set by preflight or run_once above
            pipeline_timing = result.get("timing") if isinstance(result, dict) else None
            trial_record["pipeline"] = result
            trial_record["timing"]["pipeline"] = pipeline_timing
            _write_json(run_dir / "p2_trial.json", trial_record,
                        jsonable=inference._jsonable)

            result["p2"] = {
                "protocol": SPEED_PROTOCOL_ID,
                "experiment_type": "execution_speed",
                "object": object_name,
                "franka_executor": executor_config,
                "speed_tuned_fields": list(SPEED_TUNABLE_FIELDS),
            }
            # Keep the automatic semantic-routing verdict.  This runner does
            # not request or store a physical human outcome label.
            if isinstance(result.get("semantic"), dict):
                result["p2"]["semantic_evaluation"] = result["semantic"].get(
                    "semantic_evaluation")
            trial_record["pipeline"] = result
            _write_json(run_dir / "result.json", result, jsonable=inference._jsonable)
            _write_json(run_dir / "p2_trial.json", trial_record,
                        jsonable=inference._jsonable)
            print(f"[p2-speed] saved timing/configuration -> "
                  f"{run_dir / 'p2_trial.json'}")
            _print_trial_summary(
                trial_index=trial_index,
                run_dir=run_dir,
                object_name=object_name,
                executor_config=executor_config,
                result=result,
            )
    finally:
        if recording_runtime is not None and recording_runtime._active is not None:
            _safe("p2 speed final recorder.stop", recording_runtime._active.stop)
        if executor is not None:
            _safe("executor.shutdown", executor.shutdown)
        if orch is not None:
            _safe("orch.close", orch.close)
        if semantic_router is not None:
            _safe("semantic_router.close", semantic_router.close)
        if stream_started:
            _stop_with_timeout("rcc.stop", rcc.stop)
        _stop_with_timeout("rcc.end", rcc.end)


if __name__ == "__main__":
    main()
