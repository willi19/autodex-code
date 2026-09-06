#!/usr/bin/env python3
"""Run consecutive, operator-labelled P2 semantic-routing trials.

This is a reuse-oriented wrapper around :mod:`src.demo.inference.run_demo`:
it calls that module's unchanged ``run_once`` for every physical trial rather
than maintaining a second perception, grasp-planning, or execution pipeline.
Only the P2 semantic router, video recorder, interactive protocol choices,
and annotations are layered around it.

Hardware/session resources created once
---------------------------------------
* camera-controller ownership, calibration and live stream;
* one FoundPose ``InitOrchestrator`` connection, one cuRobo planner and one
  robot executor connection;
* the Qwen2.5-VL-3B NF4 model plus its three-PC crop subscriber;
* the optional trigger/timestamp resources for per-PC AVI recording.

Per-object / per-episode resources
-----------------------------------
* changing the object reinitialises the existing FoundPose daemons with that
  object's mesh/template; reusing the same object does not;
* every episode captures FoundPose/SAM3, obtains the generic semantic route,
  plans, executes, and writes the normal ``v8_demo/inspire/<obj>/<stamp>``
  artifact tree, including ``raw/exec/videos/*.avi`` on each capture PC;
* the operator enters its location and the furthest verified outcome (f/g/p/c),
  or records an explicitly unscored aborted take (a)
  and an optional memo.  Both are committed to that episode's ``result.json``
  and ``p2_trial.json``.

Start P2-capable capture daemons first::

    bash scripts/init_daemons.sh start --p2-semantic
    python src/demo/p2/run_auto.py --arm franka --execute

Use ``--obj apple --location 0`` only to seed the first prompt.  Enter at a
subsequent object/location prompt repeats its immediately previous value.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.demo.p2.protocol import P2_PROTOCOL_ID
from src.demo.p2.planner_warmup import prewarm_planner
from src.demo.p2.recording import P2RecordingRuntime
from src.demo.p2.semantic_router import P2_QWEN_MODEL, P2SemanticRouter


P2_LOCATIONS: dict[str, dict[str, str]] = {
    "0": {"name": "upper_right", "description": "upper right"},
    "1": {"name": "center", "description": "center"},
    "2": {"name": "lower_left", "description": "lower left"},
}

P2_OUTCOMES: dict[str, dict[str, Any]] = {
    "f": {"name": "fail", "G": False, "P": False, "C": False},
    "g": {"name": "grasp", "G": True, "P": False, "C": False},
    "p": {"name": "place", "G": True, "P": True, "C": False},
    "c": {"name": "correct_bin", "G": True, "P": True, "C": True},
    # The operator stopped this take without a scientific outcome (for
    # example, an external interruption or setup issue).  Keep it durable in
    # the episode tree, but make it impossible for summary code to mistake it
    # for either a failure or a success.
    "a": {"name": "aborted", "G": None, "P": None, "C": None,
          "scored": False},
}

_TRIAL_SEPARATOR = "=" * 78
_OBJECT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


def _parse_args(argv: list[str] | None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description=__doc__, add_help=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--obj", metavar="OBJECT",
                        help="initial object shown at the first prompt")
    parser.add_argument("--location", choices=sorted(P2_LOCATIONS),
                        help="initial P2 location shown at the first prompt")
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
    p2_args, remaining = parser.parse_known_args(argv)
    if p2_args.help:
        parser.print_help()
        print("\nAll remaining options are forwarded unchanged to "
              "src/demo/inference/run_demo.py.")
        raise SystemExit(0)
    if p2_args.max_trials < 0:
        parser.error("--max-trials must be non-negative")
    if not 1 <= p2_args.semantic_port <= 65535:
        parser.error("--semantic-port must be in 1..65535")
    if p2_args.semantic_timeout_s <= 0:
        parser.error("--semantic-timeout-s must be positive")
    if p2_args.video_fps <= 0:
        parser.error("--video-fps must be positive")
    return p2_args, remaining


def _base_inference_args(initial_object: str, remaining: list[str]):
    """Validate P2's immutable motion contract through the normal parser."""
    from src.demo.inference import run_demo as inference
    from src.demo.p2 import run_demo as p2_demo

    forwarded = p2_demo._inference_argv(
        SimpleNamespace(obj=initial_object), remaining)
    parsed = inference.parse_args(forwarded)
    if not parsed.execute:
        raise SystemExit("P2 run_auto drives physical trials; pass --execute")
    return parsed


def _episode_args(object_name: str, remaining: list[str]):
    """Return a fresh regular inference Namespace for this selected object."""
    return _base_inference_args(object_name, remaining)


def _make_episode_dir(project_root: Path, object_name: str, demo_version: str) -> Path:
    """Create the exact existing inference-demo result hierarchy safely."""
    parent = (Path(project_root) / "experiment" / demo_version / "inspire"
              / object_name)
    parent.mkdir(parents=True, exist_ok=True)
    base = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = parent / base
    suffix = 1
    while True:
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            candidate = parent / f"{base}_{suffix:02d}"
            suffix += 1


def _write_json(path: Path, value: Any, *, jsonable) -> None:
    path.write_text(json.dumps(jsonable(value), indent=2, default=str) + "\n")


def _prompt_object(previous: str | None) -> str | None:
    while True:
        suffix = f" [{previous}]" if previous else ""
        raw = input(f"\nObject (asset name; Enter=reuse, q=finish){suffix}: ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            return None
        if not raw and previous is not None:
            return previous
        if _OBJECT_NAME.fullmatch(raw):
            return raw
        print("Use a non-empty asset name containing only letters, numbers, "
              "underscore, dot, or hyphen.")


def _prompt_location(previous: str | None) -> str | None:
    choices = "0=upper_right, 1=center, 2=lower_left"
    while True:
        suffix = f" [{previous}={P2_LOCATIONS[previous]['name']}]" if previous else ""
        raw = input(f"Location ({choices}; Enter=reuse, q=finish){suffix}: ").strip()
        if raw.lower() in {"q", "quit", "exit"}:
            return None
        if not raw and previous is not None:
            return previous
        if raw in P2_LOCATIONS:
            return raw
        print("Enter 0, 1, or 2.")


def _prompt_outcome() -> tuple[str, str, float]:
    """Return the furthest verified P2 outcome, memo, and prompt duration."""
    started = time.perf_counter()
    while True:
        raw = input("Outcome [f=fail, g=grasp, p=place, c=correct bin, "
                    "a=aborted]: ").strip().lower()
        if raw in P2_OUTCOMES:
            break
        print("Enter exactly f, g, p, c, or a.")
    memo = input("Memo (optional; Enter to save): ").strip()
    return raw, memo, round(time.perf_counter() - started, 3)


def _format_seconds(value: Any) -> str:
    """Compact, operator-facing duration formatting without inventing zeroes."""
    try:
        return f"{float(value):.2f}s"
    except (TypeError, ValueError):
        return "n/a"


def _print_trial_summary(*, trial_index: int, run_dir: Path, object_name: str,
                         location: dict[str, Any], result: dict[str, Any],
                         human: dict[str, Any]) -> None:
    """Print only the completed P2 take's decision, label, and core timing."""
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
    gpc = "unscored" if not human.get("scored", True) else (
        f"G={human.get('G')} P={human.get('P')} C={human.get('C')}")
    reset_text = (_format_seconds(reset.get("total_s"))
                  if reset.get("performed") else "not performed")

    print(f"\n{_TRIAL_SEPARATOR}")
    print(f"[P2 TRIAL {trial_index:02d} COMPLETE] {object_name} / "
          f"{location['name']} ({location['id']})")
    print(f"  pipeline : {auto}")
    print(f"  route    : {route}")
    print(f"  operator : {human['name']} ({gpc})")
    print("  timing   : "
          f"perception={_format_seconds(perception.get('total_s'))}, "
          f"planning={_format_seconds(planning.get('total_s'))}, "
          f"task={_format_seconds(task.get('total_s'))}, "
          f"reset={reset_text}, "
          f"execution={_format_seconds(execution.get('total_s'))}")
    print(f"  saved    : {run_dir / 'result.json'}")
    print(f"{_TRIAL_SEPARATOR}\n")


def _prepare_object(*, object_name: str, args, orch, intrinsics, extrinsics,
                    image_hw: tuple[int, int], pc_serials, assets_base: Path,
                    object_root: Path, check_mesh_frame_match) -> dict[str, Any]:
    """Switch the already-connected FoundPose daemons to one P2 object."""
    mesh_path = object_root / object_name / "raw_mesh" / f"{object_name}.obj"
    assets_root = assets_base / object_name
    if not mesh_path.is_file():
        raise FileNotFoundError(f"mesh not found: {mesh_path}")
    frame_ok, frame_msg = check_mesh_frame_match(
        object_name, str(mesh_path), str(object_root))
    if not frame_ok:
        raise RuntimeError(f"[mesh_frame] {frame_msg}")
    repre = assets_root / "object_repre" / "v1" / object_name / "1" / "repre.pth"
    if not repre.is_file():
        raise FileNotFoundError(f"FoundPose representation missing: {repre}")
    print(f"[mesh_frame] {frame_msg}")
    started = time.perf_counter()
    print(f"[orch] init/switch object -> {object_name}...")
    orch.init_object(
        obj_name=object_name, mesh_path=str(mesh_path), assets_root=str(assets_root),
        intrinsics_full=intrinsics, extrinsics_full=extrinsics,
        image_hw=image_hw, mode="live", pc_serials=pc_serials,
    )
    return {
        "object": object_name,
        "mesh_path": str(mesh_path),
        "foundpose_assets": str(assets_root),
        "object_init_s": round(time.perf_counter() - started, 3),
        "mesh_frame": frame_msg,
    }


def main(argv: list[str] | None = None) -> None:
    p2_args, remaining = _parse_args(argv)
    # Parsing once before hardware connects catches unsupported inference flags
    # early.  The selected object itself is prompted immediately after setup.
    initial_object = p2_args.obj or "apple"
    base_args = _base_inference_args(initial_object, remaining)

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
    from src.execution.scene_cfg import check_mesh_frame_match

    object_root = Path(get_obj_root(inference.GRASP_ASSET_VERSION))

    print(f"[p2-auto] protocol={P2_PROTOCOL_ID}; "
          "FRUIT -> left/+50.0 deg, NON_FRUIT -> right/-30.0 deg")
    print("[p2-auto] one-time setup: cameras/calibration, planner, robot, "
          "Qwen and recording trigger. Per trial: FoundPose, VLM, plan, execute.")

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

    rcc = remote_camera_controller("p2_auto", pc_list=base_args.pc_list,
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

        # InitOrchestrator is transport state only here.  Its object-specific
        # template/model is loaded lazily when that object is selected.
        orch = InitOrchestrator(
            pc_list=base_args.pc_list, capture_ips=pc_ips,
            port_mask=base_args.port_mask, port_pose=base_args.port_pose,
            port_cmd=base_args.port_cmd,
        )
        planner_robot = _planner_robot(base_args.arm, "inspire")
        print(f"[planner] warmup once ({planner_robot})...")
        planner = GraspPlanner(hand=planner_robot)
        quiet_curobo()
        # Build/capture all cuRobo graphs before any episode starts.  This
        # moves first-plan latency into one-time setup and leaves the normal
        # P2 perception/planning/execution path unchanged per trial.
        planner_prewarm = prewarm_planner(
            planner=planner, preferred_object=initial_object,
            object_root=object_root)
        print(f"[executor] connect once ({base_args.arm})...")
        if base_args.arm == "franka":
            from src.execution.franka_executor import FrankaExecutor
            executor = FrankaExecutor(hand_name="inspire")
            executor.set_speed_profile_planner(planner)
        else:
            from autodex.executor.real import RealExecutor
            executor = RealExecutor(hand_name="inspire")

        semantic_router = P2SemanticRouter(
            capture_ips=pc_ips, pc_serials=pc_serials,
            port=p2_args.semantic_port, model_id=p2_args.vlm_model,
            timeout_s=p2_args.semantic_timeout_s,
        )
        print("[p2-vlm] preload Qwen once...")
        semantic_router.preload()
        if not p2_args.no_video:
            serials = [serial for pc in base_args.pc_list
                       for serial in pc_serials[pc]]
            recording_runtime = P2RecordingRuntime(
                rcc=rcc, pc_list=base_args.pc_list, serials=serials,
                video_fps=p2_args.video_fps,
            )

        # Same first clear-view home as the single P2 runner.  Later trials
        # rely on run_once's normal successful reset; no blind home is issued
        # after a failed grasp or an unexpected post-motion exception.
        print("[executor] moving once to clear-view home")
        executor.home(clear_view=True)

        previous_object = p2_args.obj
        previous_location = p2_args.location
        initialized_object: str | None = None
        trial_index = 0

        while p2_args.max_trials == 0 or trial_index < p2_args.max_trials:
            object_name = _prompt_object(previous_object)
            if object_name is None:
                break
            location_id = _prompt_location(previous_location)
            if location_id is None:
                break
            previous_object, previous_location = object_name, location_id
            trial_index += 1
            args = _episode_args(object_name, remaining)
            run_dir = _make_episode_dir(Path(project_dir), object_name,
                                        inference.DEMO_VERSION)
            print(f"\n{_TRIAL_SEPARATOR}")
            print(f"[P2 TRIAL {trial_index:02d} START] {object_name} / "
                  f"{P2_LOCATIONS[location_id]['name']} ({location_id})")
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
                if object_reused:
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
                # An unavailable asset must still have a durable episode record
                # and operator label; do not tear down the reusable hardware.
                result = {
                    "object": object_name, "success": False,
                    "reason": "object_prepare_failed", "error": repr(exc),
                    "timing": {"preparation": {
                        "total_s": round(time.perf_counter() - preparation_started, 3)}},
                }
                grasps = []
            else:
                result = None

            location = {"id": int(location_id), **P2_LOCATIONS[location_id]}
            trial_record: dict[str, Any] = {
                "protocol": P2_PROTOCOL_ID,
                "trial_index": trial_index,
                "created_at": dt.datetime.now().isoformat(timespec="seconds"),
                "object": object_name,
                "location": location,
                "conditions": {
                    "arm": args.arm,
                    "hand": "inspire",
                    "grasp_version": inference.GRASP_ASSET_VERSION,
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
                        "model_id": p2_args.vlm_model,
                        "crop_port": p2_args.semantic_port,
                        "timeout_s": p2_args.semantic_timeout_s,
                    },
                    "video": {
                        "enabled": not p2_args.no_video,
                        "fps": None if p2_args.no_video else p2_args.video_fps,
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
                    "execution_video_enabled": not p2_args.no_video,
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
            # A missing mesh/FoundPose representation is a preflight failure:
            # no perception, planning, or robot task has run.  Do not make the
            # operator label it as a physical P2 result (the old behaviour
            # immediately displayed the f/g/p/c prompt in this situation).
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
                    # Preserve the standard inference policy: do not command a
                    # surprise release/reset while the operator may need to
                    # inspect a held object.
                    result = {"success": False, "reason": "fatal",
                              "error": repr(exc),
                              "action": "stopped_without_robot_reset"}
                    print(f"\n[FATAL] trial stopped without robot reset: {exc!r}")
                finally:
                    if recorder is not None and recorder.active:
                        _safe("p2 execution_recorder.stop",
                              lambda: recorder.stop(restart_stream=True,
                                                    stream_fps=args.stream_fps))
                    elif recorder is None and run_raised:
                        # The no-video path changes the RCC sink itself.  Put
                        # the live stream back after an unexpected error.  A
                        # normal no-video run_once already does this, so do
                        # not spend an extra camera stop/start between takes.
                        try:
                            rcc.stop()
                            _rcc_start(rcc, "stream", False, fps=args.stream_fps)
                        except Exception as exc:
                            print(f"[rcc] stream restore after trial failed: {exc!r}")

            pipeline_timing = result.get("timing") if isinstance(result, dict) else None
            trial_record["pipeline"] = result
            trial_record["timing"]["pipeline"] = pipeline_timing
            _write_json(run_dir / "p2_trial.json", trial_record,
                        jsonable=inference._jsonable)

            if pipeline_started:
                print(f"\n[p2] episode {run_dir.name}: record the furthest verified result.")
                outcome_code, memo, label_s = _prompt_outcome()
                human = {"code": outcome_code, **P2_OUTCOMES[outcome_code],
                         "memo": memo, "annotated_at": dt.datetime.now().isoformat(
                             timespec="seconds"),
                         "operator_label_s": label_s}
            else:
                outcome_code, label_s = "a", 0.0
                reason = str(result.get("reason", "preflight_failed"))
                human = {
                    "code": outcome_code,
                    **P2_OUTCOMES[outcome_code],
                    "memo": f"Automatic preflight abort: {reason}",
                    "annotated_at": dt.datetime.now().isoformat(timespec="seconds"),
                    "operator_label_s": label_s,
                    "automatic": True,
                }
                print(f"\n[p2] preflight failed ({reason}); no robot task ran. "
                      "Recorded automatic aborted result; skipping operator outcome prompt.")
            result["p2"] = {
                "protocol": P2_PROTOCOL_ID,
                "object": object_name,
                "location": location,
                "human_evaluation": human,
            }
            # Keep automatic semantic C and human physical C distinct.  The
            # former asks whether Qwen chose fruit/non-fruit correctly; the
            # latter verifies the object actually reached the selected bin.
            if isinstance(result.get("semantic"), dict):
                result["p2"]["semantic_evaluation"] = result["semantic"].get(
                    "semantic_evaluation")
            result.setdefault("timing", {})["operator_annotation"] = {
                "total_s": label_s,
            }
            trial_record["human_evaluation"] = human
            trial_record["timing"]["operator_annotation_s"] = label_s
            trial_record["pipeline"] = result
            _write_json(run_dir / "result.json", result, jsonable=inference._jsonable)
            _write_json(run_dir / "p2_trial.json", trial_record,
                        jsonable=inference._jsonable)
            print(f"[p2] saved {P2_OUTCOMES[outcome_code]['name']} -> "
                  f"{run_dir / 'result.json'}")
            _print_trial_summary(
                trial_index=trial_index, run_dir=run_dir,
                object_name=object_name, location=location,
                result=result, human=human,
            )
    finally:
        # These are session-level resources; a normal episode has already
        # stopped/restarted the recorder.  This final pass covers Ctrl-C or a
        # failure between two prompts without leaving a capture sink armed.
        if recording_runtime is not None and recording_runtime._active is not None:
            _safe("p2 final recorder.stop", recording_runtime._active.stop)
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
