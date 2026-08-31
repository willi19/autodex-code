#!/usr/bin/env python3
"""Automated mode: distributed FoundPose init -> planning -> execute -> label.

Replaces the legacy SAM3+FPose perception pipeline used by
`src/execution_prev/run_auto.py`. Per trial we run one FoundPose init across
the capture PCs and treat that pose as the object pose for planning.

Prerequisites:
    bash scripts/init_daemons.sh start    # init_daemon on capture1-3, 5, 6
    bash scripts/init_daemons.sh status   # 1 daemon per PC

Usage:
    python src/execution/run_auto.py --obj attached_container
    python src/execution/run_auto.py --obj brown_ramen --scene wall --wall_angle 0
    python src/execution/run_auto.py --obj brown_ramen --success_only --viz
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import chime
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from paradex.io.robot_controller import get_arm, get_hand  # noqa: F401  (warm import)
from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.io.camera_system.signal_generator import UTGE900
from paradex.io.camera_system.timestamp_monitor import TimestampMonitor
from paradex.utils.system import network_info, get_pc_ip, get_camera_list
from paradex.calibration.utils import save_current_camparam, save_current_C2R, load_c2r

from autodex.utils.path import project_dir, get_obj_root
from autodex.planner import GraspPlanner
from autodex.planner.obstacles import add_obstacles
from autodex.planner.visualizer import ScenePlanVisualizer
from autodex.executor.real import RealExecutor
from autodex.perception.init_orchestrator import InitOrchestrator
from autodex.perception.snapshot_orchestrator import SnapshotOrchestrator

from src.execution.scene_cfg import pose_world_to_scene_cfg
from src.execution.label import auto_label_charuco, get_label

# Board id lives in src/execution/label.py — one place to swap.
from src.execution.label import CHARUCO_BOARD  # noqa: E402

# Candidate pools driven by precomputed scene coverage: the runner ranks
# candidates by remaining-uncovered scenes, bounces to a reorient target when
# the current tabletop is fully covered, and stops when nothing is left.
# Other pools (selected_100, table_only, ...) have no coverage json and run
# the plain candidate sweep instead.
COVERAGE_VERSIONS = ("v7", "v8")


def _is_coverage_pool(version: str) -> bool:
    return version in COVERAGE_VERSIONS


def _stop_with_timeout(name: str, fn, timeout: float = 20.0) -> bool:
    """Run a shutdown call with a deadline, naming it if it does not return.

    Every stop() on this path ends in an untimed Event.wait() -- rcc waits on
    `sending_event`, the timestamp monitor on `event["stop"]` -- so a device
    that died mid-capture hangs the whole trial with no clue which one it was.
    An untimed wait is also not Ctrl-C interruptible, so the operator can only
    kill the process. Run each in its own thread and move on if it overruns:
    a leaked stop is far cheaper than a wedged experiment.

    Returns True if it returned in time.
    """
    import threading
    done = threading.Event()
    err = []

    def _run():
        try:
            fn()
        except Exception as exc:      # a failing stop must not kill the trial
            err.append(exc)
        finally:
            done.set()

    threading.Thread(target=_run, daemon=True, name=f"stop-{name}").start()
    if not done.wait(timeout):
        print(f"[shutdown] {name}.stop() did not return in {timeout:.0f}s — "
              f"leaving it and continuing")
        return False
    if err:
        print(f"[shutdown] {name}.stop() raised: {err[0]!r}")
    return True


def _safe_timestamp_start(tsm, save_path) -> bool:
    """Start the sync-timestamp monitor, refusing to block on a dead one.

    TimestampMonitor.run() gives up when its camera cannot be opened: it logs
    "continuing WITHOUT sync timestamps", sets error+connection+stop, and the
    thread RETURNS. It guards stop() against that ("stop() blocks on this;
    without it a later stop() would hang forever") but not start() -- which
    ends in `self.event["acquisition"].wait()` with no timeout, waiting on a
    thread that is already gone. The wait is not Ctrl-C interruptible, so the
    trial hangs until the process is killed.

    The error flag is one-shot (start() clears it via the "is in ERROR state"
    branch), which is why the hang lands on the SECOND trial rather than the
    first. Check the thread itself instead: no live capture thread means no
    one will ever set `acquisition`.

    Returns True if the monitor was started, False if it was skipped.
    """
    th = getattr(tsm, "capture_thread", None)
    alive = th.is_alive() if th is not None else False
    cam_ok = getattr(tsm, "camera", None) is not None
    if not alive or not cam_ok:
        print(f"[timestamp] monitor is dead (thread_alive={alive} "
              f"camera={'ok' if cam_ok else 'None'}) — skipping, "
              f"recording WITHOUT sync timestamps")
        tsm._autodex_started = False
        return False
    tsm.start(save_path)
    tsm._autodex_started = True
    return True


def _safe_timestamp_stop(tsm) -> None:
    """Stop the monitor only if we actually started it.

    stop() ends in an untimed event["stop"].wait(). When start() was skipped
    there is nothing to stop and nobody left to set that event, so calling it
    burns the full shutdown timeout on every trial for no reason.
    """
    if not getattr(tsm, "_autodex_started", False):
        return
    _stop_with_timeout("timestamp_monitor", tsm.stop)


def _rcc_start(rcc, mode, sync_mode, save_path=None, fps=30):
    """Start a capture, translating the retired 'stream'/'video' modes.

    paradex's camera API dropped both: a capture now arms in 'acquire' and its
    outputs are toggled as SINKS (set_stream / set_record). The capture PCs
    reject the old names outright --
        invalid mode 'stream': use 'image' (single frame) or 'acquire' + ...
    -- which silently left every trial without frames. Keeping the old call
    shape here means the trial flow below reads unchanged.
    """
    if mode == "stream":
        rcc.arm(syncMode=sync_mode, fps=fps)
        rcc.set_stream(True)
        _warn_if_not_streaming(rcc)
    elif mode == "full":
        # "full" was video AVI + SHM stream at once (snapshot_daemon reads the
        # stream while the AVI records). Both are just sinks now.
        rcc.arm(syncMode=sync_mode, fps=fps)
        rcc.set_record(save_path=save_path, on=True)
        rcc.set_stream(True)
    elif mode == "video":
        rcc.arm(syncMode=sync_mode, fps=fps)
        rcc.set_record(save_path=save_path, on=True)
    else:                       # 'image' is still a real capture mode
        rcc.start(mode, sync_mode, save_path, fps=fps)


def _warn_if_not_streaming(rcc, timeout_s: float = 4.0, poll_s: float = 0.5) -> bool:
    """Warn if the cameras are not actually capturing after the stream is armed.

    The failure this catches is silent: without a running capture the init
    pipeline just sits on 0/20 masks until it times out. Poll rather than read
    one status snapshot — ``running`` is reported by the daemons' health PUB and
    lags the sink command by a beat, so a single read right after set_stream
    reports False on healthy cameras.
    """
    deadline = time.time() + timeout_s
    dead: list = []
    while time.time() < deadline:
        try:
            dead = [pc for pc, s in (rcc.get_status().get("pc") or {}).items()
                    if not s.get("running")]
        except Exception as exc:
            print(f"[rcc] status check failed: {exc!r}")
            return True                      # don't block the run on telemetry
        if not dead:
            return True
        time.sleep(poll_s)
    print(f"[rcc] WARNING stream armed but not capturing on: {dead}")
    return False


def _ensure_camera_lock(rcc, settle_s: float = 1.5) -> bool:
    """Make sure THIS controller owns the daemons, taking over if it does not.

    A crashed run leaves its lock behind, and the next ``register`` is refused
    ("locked by run_auto_<earlier>"). ``register()`` only logs that and sets
    ``_registered = True`` anyway, so every later arm/set_stream is silently
    dropped by the daemon: cameras never capture, the init pipeline waits out
    its timeout on 0/20 masks, and nothing says why. Check ownership against
    the daemons' own report instead of trusting registration.
    """
    time.sleep(settle_s)
    try:
        pcs = (rcc.get_status().get("pc") or {})
        foreign = {pc: s.get("controller") for pc, s in pcs.items()
                   if s.get("controller") and s.get("controller") != rcc.name}
        if not foreign:
            return True
        print(f"[rcc] daemons held by another controller: {foreign}")
        print("[rcc] forcing takeover")
        rcc.force_takeover()
        time.sleep(1.0)
        pcs = (rcc.get_status().get("pc") or {})
        still = {pc: s.get("controller") for pc, s in pcs.items()
                 if s.get("controller") and s.get("controller") != rcc.name}
        if still:
            print(f"[rcc] TAKEOVER FAILED, still held by: {still}")
            return False
        print("[rcc] takeover ok")
        return True
    except Exception as exc:
        print(f"[rcc] ownership check failed: {exc!r}")
        return False


def _clear_camera_errors(rcc, settle_s: float = 1.5, reload_wait_s: float = 6.0,
                         attempts: int = 2) -> bool:
    """Reload the capture daemons' cameras if they are stuck in an error state.

    A camera that failed to start latches its error and never clears it: on the
    error path ``Camera.start()`` returns BEFORE setting ``event["start"]``,
    while ``error_reset()`` only fires from ``stop()`` when that same event was
    set. So one bad start (e.g. a retired capture mode) poisons the camera for
    every later run, and every ``start`` after it returns early — trials then
    run with no frames at all. Reloading rebuilds the daemon's CameraLoader,
    which is the only thing that clears it.

    Returns True if the cameras are healthy when this returns.
    """
    time.sleep(settle_s)
    if not rcc.is_error():
        return True
    for i in range(attempts):
        print(f"[rcc] cameras in error state — reloading "
              f"({i + 1}/{attempts})")
        try:
            rcc.force_takeover()      # a dead session may still hold the lock
        except Exception as exc:
            print(f"[rcc] force_takeover failed: {exc!r}")
        try:
            rcc.reload_cameras()
        except Exception as exc:
            print(f"[rcc] reload_cameras failed: {exc!r}")
        time.sleep(reload_wait_s)
        if not rcc.is_error():
            print("[rcc] cameras recovered")
            return True
    print("[rcc] STILL in error after reload — check the capture PCs:\n"
          f"      {rcc.get_status()}")
    return False



def _list_v7_scenes(hand: str, obj: str, version: str = "v7"):
    """List all (scene_type, scene_id) pairs under v7/{obj}/.
    Returns sorted list of tuples."""
    from autodex.utils.path import get_candidate_path
    root = Path(get_candidate_path(hand)) / version / obj
    if not root.is_dir():
        return []
    out = []
    for st_dir in sorted(root.iterdir()):
        if not st_dir.is_dir():
            continue
        for sid_dir in sorted(st_dir.iterdir()):
            if sid_dir.is_dir():
                out.append((st_dir.name, sid_dir.name))
    return out


def _pick_v7_scene(hand: str, obj: str, version: str = "v7"):
    """Pick the first scene under v7/{obj}/ that has no successful grasp yet.
    Returns (scene_type, scene_id) or None if all scenes are done."""
    for st, sid in _list_v7_scenes(hand, obj, version):
        if not _scene_has_success(hand, version, obj, st, sid):
            return (st, sid)
    return None


def _scene_has_success(hand: str, version: str, obj: str,
                        scene_type: str, scene_id: str) -> bool:
    """Return True if any grasp candidate under
    candidates/{hand}/{version}/{obj}/{scene_type}/{scene_id}/ already has
    a result.json marking success=True. Used to skip a whole scene once any
    grasp in it has worked (user policy: 한 scene 성공하면 그 scene 통째로 제외)."""
    from autodex.utils.path import get_candidate_path
    base = Path(get_candidate_path(hand)) / version / obj
    if scene_type:
        base = base / scene_type / scene_id
    else:
        base = base / scene_id
    if not base.is_dir():
        return False
    for grasp_dir in base.iterdir():
        if not grasp_dir.is_dir():
            continue
        rj = grasp_dir / "result.json"
        if not rj.exists():
            continue
        try:
            with open(rj) as f:
                if json.load(f).get("success", False):
                    return True
        except Exception:
            continue
    return False


logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")


def quiet_curobo(level=logging.WARNING) -> None:
    """Silence cuRobo's per-solve chatter ("Updating problem kernel", "Ran TO",
    "breaking reference", ...).

    setLevel alone is not enough here: whichever library configures logging
    first owns the root handler, and cuRobo's records still reach it by
    propagation. Cutting propagate keeps them off stdout no matter who set up
    the root, and the NullHandler stops the "no handlers" fallback from
    printing them anyway. A single plan emits dozens of these lines, which bury
    the [planner]/[place] output that actually says what happened.
    """
    lg = logging.getLogger("curobo")
    lg.setLevel(level)
    lg.propagate = False
    if not lg.handlers:
        lg.addHandler(logging.NullHandler())


quiet_curobo()

def _planner_robot(arm: str, hand: str) -> str:
    """cuRobo robot config name for an (arm, hand) pair.

    The xarm configs are keyed by hand alone (``xarm_allegro`` etc. are selected
    inside GraspPlanner from the hand string), so xarm keeps passing the hand
    through unchanged. The FR3 has its own config per hand.
    """
    if arm != "franka":
        return hand
    if hand == "inspire":
        return "fr3_inspire"
    # NOTE: GraspPlanner.HAND_CONFIGS.get(hand, allegro) SILENTLY falls back to
    # the allegro xarm config for an unknown key, so an unsupported combination
    # must be rejected here rather than planned with the wrong robot.
    raise SystemExit(
        f"--arm franka does not support --hand {hand}: no fr3 config for it "
        f"(GraspPlanner.HAND_CONFIGS has fr3_inspire only, and "
        f"{project_dir}/content/configs/robot/ ships fr3_inspire.yml only). "
        f"Use --hand inspire.")


DEFAULT_PC_LIST = ["capture1", "capture2", "capture3", "capture5", "capture6"]
ASSETS_BASE = Path.home() / "shared_data/AutoDex/foundpose_assets"
CAM_PARAM_ROOT = Path.home() / "shared_data/cam_param"


def _wait_for_object_on_table(rcc, args, scene_prefix: str, trial_idx: int,
                                required_board: str = CHARUCO_BOARD) -> bool:
    """Block until charuco `required_board` is NOT fully visible (= obj covers it).
    Used as a pre-flight check before each --auto trial.

    Returns True to proceed with trial, False if user pressed 'q' to quit.
    """
    sub = f"{scene_prefix}/{args.hand}" if scene_prefix else args.hand
    attempt = 0
    while True:
        attempt += 1
        check_rel = os.path.join(
            "shared_data", "AutoDex", "experiment", args.exp_name, sub,
            args.obj, "_precheck", f"trial{trial_idx:03d}_{attempt:02d}", "raw"
        )
        check_abs = os.path.join(
            project_dir, "experiment", args.exp_name, sub, args.obj,
            "_precheck", f"trial{trial_idx:03d}_{attempt:02d}", "raw", "images"
        )
        _stop_with_timeout("rcc", rcc.stop)
        rcc.start("image", False, check_rel)
        rcc.stop()
        time.sleep(0.3)
        board_visible, info = auto_label_charuco(check_abs, required_board=required_board)
        # board_visible=True → board fully detected = no obj on table → prompt.
        # board_visible=False → board partially hidden = obj on table → start.
        # board_visible=None → no images / board not in cfg → start anyway (don't block).
        if board_visible is None:
            print(f"[precheck] {info.get('reason', 'unknown')} — proceeding without check")
            _rcc_start(rcc, "stream", False, fps=args.stream_fps)
            return True
        if not board_visible:
            print(f"[precheck] obj on table (board covered "
                  f"{info.get('covered')}/{info.get('expected')}). Starting trial.")
            _rcc_start(rcc, "stream", False, fps=args.stream_fps)
            return True
        print(f"[precheck] charuco fully visible "
              f"({info.get('covered')}/{info.get('expected')}) — no obj on table.")
        try:
            cmd = input("    Place obj then press Enter (q to quit): ").strip().lower()
        except KeyboardInterrupt:
            return False
        if cmd == "q":
            return False


# ── calibration ──────────────────────────────────────────────────────────────

def _load_calib(calib_dir: Path):
    """Read intrinsics.json + extrinsics.json into the dict shape InitOrchestrator wants."""
    with open(calib_dir / "intrinsics.json") as f:
        intr_raw = json.load(f)
    with open(calib_dir / "extrinsics.json") as f:
        extr_raw = json.load(f)

    intrinsics_full, extrinsics_full = {}, {}
    for s, d in intr_raw.items():
        intrinsics_full[s] = {
            "K_orig": np.asarray(d["original_intrinsics"], dtype=np.float64).reshape(3, 3),
            "K_undist": np.asarray(d["intrinsics_undistort"], dtype=np.float64).reshape(3, 3),
            "dist_params": np.asarray(d["dist_params"], dtype=np.float64).reshape(-1),
            "width": int(d["width"]), "height": int(d["height"]),
        }
    for s, ext in extr_raw.items():
        a = np.asarray(ext, dtype=np.float64).reshape(-1)
        a = (np.vstack([a.reshape(3, 4), [0, 0, 0, 1]]) if a.size == 12 else a.reshape(4, 4))
        extrinsics_full[s] = a

    first = next(iter(intrinsics_full.values()))
    return intrinsics_full, extrinsics_full, int(first["height"]), int(first["width"])


# ── single trial ─────────────────────────────────────────────────────────────

_active_vis: Optional[ScenePlanVisualizer] = None


def run_single_trial(
    args,
    *,
    scene_prefix: str,
    orch: InitOrchestrator,
    planner: GraspPlanner,
    executor,          # RealExecutor (xarm) or FrankaExecutor (fr3)
    rcc,
    sync_generator,
    timestamp_monitor,
) -> dict:
    global _active_vis
    if _active_vis is not None:
        try:
            _active_vis.server.stop()
        except Exception:
            pass
        _active_vis = None

    obj = args.obj
    hand = args.hand
    # xarm = 6, FR3 = 7. Every arm/hand column split below uses this instead of
    # a literal 6, so the same trial body drives both arms.
    adof = getattr(executor, "arm_dof", 6)
    dir_idx = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sub = f"{scene_prefix}/{hand}" if scene_prefix else hand
    img_dir = os.path.join(project_dir, "experiment", args.exp_name, sub, obj, dir_idx)
    os.makedirs(img_dir, exist_ok=True)
    timing: dict = {}

    def _ts() -> str:
        return datetime.datetime.now().isoformat()

    def _stamp_end(result_dict):
        """Stamp trial_end in shared timing and return the dict unchanged."""
        timing["trial_end"] = _ts()
        return result_dict

    timing["trial_start"] = _ts()

    # ── 1. save calib (raw images come from init pipeline itself) ────────────
    print(f"\n{'='*60}")
    print(f"[1/6] Trial dir -> {dir_idx}")
    save_current_C2R(img_dir)
    save_current_camparam(img_dir)

    # ── 2. Distributed FoundPose init ───────────────────────────────────────
    print(f"[2/6] Init pipeline (FoundPose distributed)...")
    timing["perception_start"] = _ts()
    t0 = time.time()
    save_capture_dir = os.path.join(img_dir, "init_capture")
    pose_world, perc_timing = orch.trigger_init(
        prompt=args.prompt,
        save_capture_dir=save_capture_dir,
        sil_iters=args.sil_iters, sil_lr=args.sil_lr,
        timeout_s=args.init_timeout_s,
    )
    timing["perception_s"] = round(time.time() - t0, 2)
    if perc_timing:
        timing["perception_detail"] = perc_timing

    if pose_world is None:
        reason = (perc_timing or {}).get("reason", "perception_failed")
        print(f"    Perception FAILED ({reason})")
        chime.error()
        # Pause for human — bad pose estimate likely needs operator to
        # reposition object / lights / camera before next attempt.
        try:
            cmd = input("    [perception_failed] Fix the scene then press Enter "
                        "(q to quit): ").strip().lower()
        except KeyboardInterrupt:
            cmd = "q"
        fail = {"dir_idx": dir_idx, "scene_type": args.scene, "success": False,
                "reason": reason, "timing": timing}
        if cmd == "q":
            fail["all_done"] = True
            fail["reason"] = "user_quit_perception_failed"
        with open(os.path.join(img_dir, "result.json"), "w") as f:
            json.dump(fail, f, indent=2, default=str)
        return _stamp_end(fail)

    print(f"    Perception: {timing['perception_s']}s")
    np.save(os.path.join(img_dir, "pose_world.npy"), pose_world)

    # ── 2.5 Reposition detection ─────────────────────────────────────────────
    # If charuco board "1" is fully visible RIGHT AFTER perception succeeded,
    # the obj is somewhere but NOT covering the board → enter reposition mode
    # (grasp obj, place at r=0.4, y=0 on the board).
    reposition_mode = False
    if args.auto:
        _stop_with_timeout("rcc", rcc.stop)
        repo_check_rel = os.path.join(
            "shared_data", "AutoDex", "experiment", args.exp_name, sub, obj,
            dir_idx, "_repo_check", "raw"
        )
        repo_check_abs = os.path.join(img_dir, "_repo_check", "raw", "images")
        rcc.start("image", False, repo_check_rel)
        rcc.stop()
        time.sleep(0.3)
        board_vis, board_info = auto_label_charuco(
            repo_check_abs, required_board=CHARUCO_BOARD
        )
        timing["repo_charuco_before"] = board_info
        _rcc_start(rcc, "stream", False, fps=args.stream_fps)
        if board_vis is True:
            print(f"\n    [reposition] charuco "
                  f"{board_info.get('covered')}/{board_info.get('expected')} "
                  f"fully visible → reposition mode")
            reposition_mode = True

    # ── 3. scene_cfg + plan ──────────────────────────────────────────────────
    print(f"[3/6] Planning (version={args.grasp_version}, scene={args.scene})...")
    timing["planning_start"] = _ts()
    t0 = time.time()
    c2r = load_c2r(img_dir)
    # Asset root for planning mesh + tabletop poses. v8 candidates/scenes were
    # generated against object_processing, whose tabletop stems and simplified
    # mesh differ from the legacy paradex tree — resolve by version so the two
    # never mix (which would mislabel pose_idx).
    obj_root = get_obj_root(args.grasp_version)
    scene_cfg = pose_world_to_scene_cfg(pose_world, c2r, obj, obj_root)
    scene_cfg = add_obstacles(
        scene_cfg, args.scene,
        wall_gap=args.wall_gap, wall_angle=args.wall_angle,
        seed=args.clutter_seed,
        clutter_min_dist=args.clutter_min_dist,
        clutter_max_dist=args.clutter_max_dist,
        clutter_n=args.clutter_n,
        shelf_width=args.shelf_width, shelf_depth=args.shelf_depth,
        shelf_height=args.shelf_height, shelf_gap=args.shelf_gap,
        shelf_back=not args.no_shelf_back,
        shelf_sides=not args.no_shelf_sides,
        shelf_top=not args.no_shelf_top,
    )
    # ── tabletop classification + cylinder freedom (mirrors run_debug) ─────
    from src.experiment.reset.tabletop_pose import classify_tabletop_pose
    from autodex.utils.symmetry import get_cyl_axis_local
    pose_robot = np.linalg.inv(c2r) @ pose_world
    tb_before = classify_tabletop_pose(pose_robot, obj, obj_root)
    timing["tabletop_before"] = tb_before
    if tb_before is not None:
        scene_id = str(tb_before["idx"])
        pose_stem = tb_before["filename"].replace(".npy", "")
        print(f"    [tabletop] idx={tb_before['idx']} "
              f"({tb_before['filename']}) err={tb_before['rot_err_deg']:.1f}°")
        # Policy: once any grasp in this scene has succeeded, skip the WHOLE
        # scene (including the successful grasp itself). Only applies for the
        # table scene (other scenes don't persist per-candidate result.json).
        scene_type_for_check = args.scene if args.scene != "table" else "table"
        if _scene_has_success(hand, args.grasp_version, obj,
                               scene_type_for_check, scene_id):
            print(f"    [scene_skip] scene_type={scene_type_for_check} "
                  f"scene_id={scene_id} already has a successful grasp — "
                  f"skipping this trial")
            skip = {"dir_idx": dir_idx, "scene_type": args.scene,
                    "success": None, "reason": "scene_already_done",
                    "scene_id": scene_id, "timing": timing}
            with open(os.path.join(img_dir, "result.json"), "w") as f:
                json.dump(skip, f, indent=2, default=str)
            return _stamp_end(skip)
    else:
        scene_id = None
        pose_stem = None
    # Symmetry-axis enumerate. Continuous-revolute → 8 angles. Discrete
    # (e.g. blue_alarm 180°) → order-N grid from pose_symmetry.json.
    from autodex.utils.symmetry import get_cyl_yaw_grid as _get_cyl_yaw_grid
    _cyl_axis = get_cyl_axis_local(obj)
    _cyl_grid = _get_cyl_yaw_grid(obj)
    # v7 candidates use scene_type=wall/shelf with BODex-sequential scene_ids
    # that DON'T map to sorted tabletop indices. We treat all v7 scenes as a
    # single pool — load EVERY (wall+shelf) candidate, then let
    # skip_scenes_with_success drop done scene_ids. obstacles for run_auto
    # are still controlled by --scene independently.
    # Reposition mode overrides the grasp source and obstacles for this trial:
    # use table_only candidates (any scene_type filtered to "table") sorted
    # by stats-based success rate.
    if reposition_mode:
        _eff_grasp_version = "table_only"
        _plan_scene_id = None
        _plan_scene_type_filter = "table"
        _plan_tabletop_stem = pose_stem
        from autodex.utils.coverage import table_only_grasp_order_by_stats
        _plan_candidate_order = table_only_grasp_order_by_stats(obj, hand=hand)
        _plan_priority_map = None
        print(f"    [reposition] table_only candidates ranked by stats: "
              f"{len(_plan_candidate_order)} grasps")
    elif _is_coverage_pool(args.grasp_version):
        _eff_grasp_version = args.grasp_version
        _plan_scene_id = None
        # A newly generated tabletop pool has no coverage JSON yet. Select it
        # directly but keep skip-done and the one-success-per-tabletop safety
        # policy, so real Franka collection does not repeat a known failure or
        # overwrite a successful record.
        _plan_scene_type_filter = args.candidate_scene_type or (
            args.scene if args.scene in ("wall", "shelf", "box") else None
        )
        # Trim by tabletop pose: keep only candidates whose scene
        # meta.pose_idx == current tabletop stem.
        _plan_tabletop_stem = pose_stem
        # NO pre-filter (= all tabletop-matching candidates loaded). After
        # IK+collision in planner.plan, sort survivors by coverage count
        # desc via priority_map, then plan_single_js in that order.
        if args.candidate_scene_type is not None:
            _plan_candidate_order = None
            _plan_priority_map = None
            print(f"    [collection] direct candidate scene type="
                  f"{args.candidate_scene_type}; coverage ordering deferred")
        elif args.ignore_coverage:
            _plan_candidate_order = None
            _plan_priority_map = None
            print(f"    [coverage] IGNORED (--ignore_coverage) — full pool")
        else:
            from autodex.utils.coverage import load_v7_coverage_map
            _cov = load_v7_coverage_map(
                obj, tabletop_pose_stem=pose_stem,
                hand=hand, version=args.grasp_version, arm=args.arm)
            if _cov is None:
                # No coverage json at all. Without this the run reads as
                # "0/0 candidates -> all scenes done" and stops, which looks
                # identical to a finished object. It is a missing precompute.
                sys.exit(
                    f"[coverage] no coverage json for {obj}/{args.grasp_version}.\n"
                    f"  expected: {project_dir}/experiment/{args.grasp_version}"
                    f"/coverage/cov_{args.grasp_version}_cand_{obj}.json\n"
                    f"  build it with:\n"
                    f"    python src/dataset/compute_v8_coverage.py --obj {obj} "
                    f"--hand {hand} --version {args.grasp_version}\n"
                    f"  (and make sure candidates/{hand}/{args.grasp_version}/{obj}/ "
                    f"is extracted from its .tar.gz first)")
            _useful = {k: v for k, v in _cov.items() if v > 0}
            _plan_candidate_order = sorted(_useful, key=lambda k: -_useful[k])
            _plan_priority_map = None
            # Dropped = remaining-uncovered == 0. Early on that is NOT
            # "already covered" — those are grasps whose `covers` list is
            # empty, i.e. collision-free in none of this tabletop's scenes.
            # Only once successes accumulate does the count mean progress.
            _n_drop = len(_cov) - len(_useful)
            _n_empty = sum(1 for k, v in _cov.items() if v == 0)
            print(f"    [coverage] {len(_useful)}/{len(_cov)} candidates open "
                  f"uncovered scenes"
                  + (f" (dropped {_n_drop}: cover nothing left)" if _n_drop else ""))
        # Pre-plan reorient check (skipped under --ignore_coverage).
        from autodex.utils.coverage import uncovered_scenes, pick_reorient_target
        _rem = (None if args.ignore_coverage else
                uncovered_scenes(obj, pose_stem, hand=hand,
                                  version=args.grasp_version, arm=args.arm))
        if _rem is not None and len(_rem) == 0:
            target = pick_reorient_target(obj, pose_stem, hand=hand,
                                           version=args.grasp_version,
                                           obj_root=obj_root, arm=args.arm)
            print(f"\n    [reorient] tabletop {pose_stem} fully covered.")
            if target is None:
                print(f"    All tabletops covered — nothing left for {obj}.")
                done = {"dir_idx": dir_idx, "scene_type": args.scene,
                        "success": None, "reason": "all_tabletops_covered",
                        "all_done": True, "timing": timing}
                with open(os.path.join(img_dir, "result.json"), "w") as f:
                    json.dump(done, f, indent=2, default=str)
                return _stamp_end(done)
            j_int, stem, n_rem = target
            print(f"    target_j={j_int} (pose {stem}) has {n_rem} uncovered scenes.")
            _reorient_cmd = (f"python src/experiment/reset/reorient.py "
                             f"--obj {obj} --hand {hand} "
                             f"--target_j {j_int} --auto "
                             f"--version {args.grasp_version}")
            print(f"    Suggested:\n      {_reorient_cmd}")
            if args.arm == "franka":
                print("    [reorient] NOTE: reorient.py drives the XARM "
                      "(RealExecutor + xarm URDFs) — do not run it for the FR3.")
            try:
                _cmd = input("    Press Enter to RUN reorient now, "
                             "'s' to skip-and-continue (you ran it manually), "
                             "'q' to quit: ").strip().lower()
            except KeyboardInterrupt:
                _cmd = "q"
            done = {"dir_idx": dir_idx, "scene_type": args.scene,
                    "success": None, "reason": "reorient_needed",
                    "reorient_target_j": j_int,
                    "reorient_target_stem": stem,
                    "reorient_uncovered_n": n_rem,
                    "timing": timing}
            if _cmd == "q":
                done["all_done"] = True
                done["reason"] = "user_quit_reorient"
            elif _cmd == "":
                # Auto-launch DISABLED: reorient is a human-supervised step —
                # run_auto only reports the target and hands off. Print the
                # command and let the operator run it, same as the 's' path.
                # import subprocess
                # print(f"    [auto] running: {_reorient_cmd}")
                # rc = subprocess.call(_reorient_cmd, shell=True)
                # done["reorient_subprocess_rc"] = rc
                # if rc != 0:
                #     print(f"    [auto] reorient returned {rc}; main loop "
                #           f"will continue but obj may still be at the same pose.")
                print(f"    [manual] auto-launch disabled — run it yourself:\n"
                      f"      {_reorient_cmd}")
            # else: 's' = user already ran it manually, fall through
            with open(os.path.join(img_dir, "result.json"), "w") as f:
                json.dump(done, f, indent=2, default=str)
            return _stamp_end(done)
    else:
        _eff_grasp_version = args.grasp_version
        _plan_scene_id = scene_id
        _plan_scene_type_filter = None
        _plan_tabletop_stem = None
        _plan_candidate_order = None
        _plan_priority_map = None
    # Reposition: ignore skip_done / skip_scenes_with_success so stats-ranked
    # candidates can be retried.
    _skip_done_eff = False if (reposition_mode or args.ignore_coverage) else True
    _skip_scenes_eff = False if (reposition_mode or args.ignore_coverage) else True
    result = planner.plan(
        scene_cfg, obj, _eff_grasp_version,
        skip_done=_skip_done_eff,
        success_only=args.success_only, hand=hand,
        scene_id=_plan_scene_id,
        scene_type_filter=_plan_scene_type_filter,
        skip_scenes_with_success=_skip_scenes_eff,
        openpose_pose_stem=pose_stem,
        cyl_axis_local=_cyl_axis,
        cyl_yaw_grid=_cyl_grid,
        tabletop_pose_stem=_plan_tabletop_stem,
        candidate_order=_plan_candidate_order,
        priority_map=_plan_priority_map,
    )
    timing["plan_s"] = round(time.time() - t0, 2)
    print(f"    Plan: {timing['plan_s']}s  success={result.success}")

    if not result.success:
        # n_total == 0 means load_candidate returned empty — every scene is
        # already done OR no grasp candidate matches the current tabletop.
        # Either way we need to switch to a different tabletop pose.
        n_total = (result.timing or {}).get("n_total", -1)
        if n_total == 0:
            print(f"    No grasp candidates left at tabletop {pose_stem} "
                  f"for {obj} ({args.grasp_version}).")
            if _is_coverage_pool(args.grasp_version):
                from autodex.utils.coverage import pick_reorient_target
                target = pick_reorient_target(obj, pose_stem, hand=hand,
                                               version=args.grasp_version,
                                               obj_root=obj_root, arm=args.arm)
                if target is None:
                    print(f"    No reorient target with uncovered scenes — "
                          f"nothing left for {obj}.")
                    done = {"dir_idx": dir_idx, "scene_type": args.scene,
                            "success": None, "reason": "all_tabletops_covered",
                            "all_done": True, "timing": timing}
                    with open(os.path.join(img_dir, "result.json"), "w") as f:
                        json.dump(done, f, indent=2, default=str)
                    return _stamp_end(done)
                j_int, stem, n_rem = target
                print(f"    [reorient] target_j={j_int} (pose {stem}) "
                      f"has {n_rem} uncovered scenes.")
                print(f"    Suggested:")
                print(f"      python src/experiment/reset/reorient.py "
                      f"--obj {obj} --hand {hand} --target_j {j_int} --auto "
                      f"--version {args.grasp_version}")
                try:
                    _cmd = input("    Run reorient then press Enter (q to quit): "
                                 ).strip().lower()
                except KeyboardInterrupt:
                    _cmd = "q"
                done = {"dir_idx": dir_idx, "scene_type": args.scene,
                        "success": None, "reason": "reorient_needed",
                        "reorient_target_j": j_int,
                        "reorient_target_stem": stem,
                        "reorient_uncovered_n": n_rem,
                        "timing": timing}
                if _cmd == "q":
                    done["all_done"] = True
                    done["reason"] = "user_quit_reorient"
                with open(os.path.join(img_dir, "result.json"), "w") as f:
                    json.dump(done, f, indent=2, default=str)
                return _stamp_end(done)
            print(f"    All scenes already done — nothing left to try.")
            done = {"dir_idx": dir_idx, "scene_type": args.scene,
                    "success": None, "reason": "all_scenes_done",
                    "all_done": True, "timing": timing}
            with open(os.path.join(img_dir, "result.json"), "w") as f:
                json.dump(done, f, indent=2, default=str)
            return _stamp_end(done)
        # cuRobo IK is stochastic — random seeds occasionally find 0
        # feasible at an obj pose where the next run finds several. Before
        # declaring "reorient needed", retry the plan up to 2 more times.
        for _retry in range(1, 3):
            print(f"    Planning failed (attempt {_retry}/2 retry)...")
            t_re = time.time()
            result = planner.plan(
                scene_cfg, obj, _eff_grasp_version,
                skip_done=_skip_done_eff,
                success_only=args.success_only, hand=hand,
                scene_id=_plan_scene_id,
                scene_type_filter=_plan_scene_type_filter,
                skip_scenes_with_success=_skip_scenes_eff,
                openpose_pose_stem=pose_stem,
                cyl_axis_local=_cyl_axis,
                cyl_yaw_grid=_cyl_grid,
                tabletop_pose_stem=_plan_tabletop_stem,
                candidate_order=_plan_candidate_order,
                priority_map=_plan_priority_map,
            )
            timing[f"plan_retry_{_retry}_s"] = round(time.time() - t_re, 2)
            if result.success:
                print(f"    Plan retry #{_retry} success after {timing[f'plan_retry_{_retry}_s']}s")
                break
        if result.success:
            timing["plan_s"] = sum(v for k, v in timing.items()
                                    if k.startswith("plan") and k.endswith("_s"))
            # Fall through to the normal post-plan code path below.
        else:
            print("    Planning FAILED after retries — launching visualizer to inspect...")
        # Match planner.plan()'s actual candidate pool: skip_done=True and
        # skip_scenes_with_success=True so the viewer shows only what was
        # actually attempted (not the full disk pool).
        wrist_se3, _, grasp_pose, filtered, ik_failed = planner.get_candidates(
            scene_cfg, obj, _eff_grasp_version,
            success_only=args.success_only,
            skip_done=_skip_done_eff, hand=hand, run_ik=True,
            scene_id=_plan_scene_id,
            scene_type_filter=_plan_scene_type_filter,
            tabletop_pose_stem=_plan_tabletop_stem,
            candidate_order=_plan_candidate_order,
            cyl_axis_local=_cyl_axis,
            cyl_yaw_grid=_cyl_grid,
            skip_scenes_with_success=_skip_scenes_eff,
        )
        fv = ScenePlanVisualizer(scene_cfg, None, port=8080,
                                 hand=_planner_robot(args.arm, hand))
        fv.add_candidates(wrist_se3, grasp_pose, filtered, ik_failed=ik_failed)
        fv.start_viewer(use_thread=True)
        _active_vis = fv
        chime.error()
        # v7: before bouncing to reorient, try (r, yaw) sweep at the
        # CURRENT tabletop — same obj orientation, just translate /
        # rotate around vertical. If any (r, yaw) makes ≥1 candidate
        # IK-feasible, prefer that (rotate_obj_yaw) over reorienting
        # to a different tabletop.
        if _is_coverage_pool(args.grasp_version):
            _ros_yaw = None
            _ros_x = None
            try:
                from autodex.utils.conversion import cart2se3 as _cart2se3
                T_obj_now = _cart2se3(scene_cfg["mesh"]["target"]["pose"])
                obj_z_now = float(T_obj_now[2, 3])
                R_obj = T_obj_now[:3, :3]
                # wrist_se3 already in WORLD frame (transformed by obj pose
                # via load_candidate). Recover obj-local wrist = inv(T_obj_now) @ world.
                _wrist_obj_local = np.einsum(
                    "ij,Njk->Nik", np.linalg.inv(T_obj_now), wrist_se3)
                _xs = np.arange(0.30, 0.66, 0.05)
                _yaws = np.deg2rad(np.arange(0, 360, 30))
                _combos = [(float(x), float(y)) for x in _xs for y in _yaws]
                # For each (x, yaw), build all wrists and IK batch-check.
                # Cap candidates to first 32 to keep grid fast.
                _cand_cap = min(32, len(_wrist_obj_local))
                _wlocal = _wrist_obj_local[:_cand_cap]
                _best_pick = None   # (n_ok, x, yaw)
                for _x, _yaw in _combos:
                    c, s = np.cos(_yaw), np.sin(_yaw)
                    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                    T_new = np.eye(4)
                    T_new[:3, :3] = Rz @ R_obj
                    T_new[:3, 3] = [_x, 0.0, obj_z_now]
                    _world_wrists = T_new[None] @ _wlocal
                    _succ = planner.ik_pose_batch(_world_wrists)
                    _n = int(_succ.sum())
                    if _best_pick is None or _n > _best_pick[0]:
                        _best_pick = (_n, _x, float(np.degrees(_yaw)))
                if _best_pick is not None and _best_pick[0] > 0:
                    _, _ros_x, _ros_yaw = _best_pick
                    print(f"\n    [pose_search] move obj to (x={_ros_x:.2f}, y=0) "
                          f"+ rotate {_ros_yaw:.0f}° → {_best_pick[0]}/{_cand_cap} "
                          f"v7 candidates IK-feasible at CURRENT tabletop")
            except Exception as _se:
                print(f"    [pose_search] failed: {_se!r}")
            if _ros_yaw is not None:
                # --arm MUST be forwarded: rotate_obj_yaw defaults to xarm,
                # so without it an FR3 run launches XArmController and dies on
                # "connect socket failed" (there is no xarm on this setup).
                _cmd_ros = (f"python src/execution/rotate_obj_yaw.py "
                             f"--obj {obj} --hand {hand} --arm {args.arm} "
                             f"--target_yaw_deg {_ros_yaw:.0f} "
                             f"--target_x {_ros_x:.2f} "
                             f"--grasp_version {args.grasp_version}")
                print(f"    Suggested (same-tabletop, no reorient):\n      {_cmd_ros}")
                try:
                    _cmd = input("    Press Enter to RUN rotate_obj_yaw now, "
                                 "'s' to skip-and-continue, 'q' to quit: "
                                 ).strip().lower()
                except KeyboardInterrupt:
                    _cmd = "q"
                fail_record = {"dir_idx": dir_idx, "scene_type": args.scene,
                               "success": False,
                               "reason": "pose_adjust_needed",
                               "pose_adjust": {"x": _ros_x, "yaw_deg": _ros_yaw},
                               "timing": timing}
                if _cmd == "q":
                    fail_record["all_done"] = True
                    fail_record["reason"] = "user_quit_pose_adjust"
                elif _cmd == "":
                    import subprocess
                    print(f"    [auto] running: {_cmd_ros}")
                    rc = subprocess.call(_cmd_ros, shell=True)
                    fail_record["rotate_obj_yaw_rc"] = rc
                with open(os.path.join(img_dir, "result.json"), "w") as f:
                    json.dump(fail_record, f, indent=2, default=str)
                return _stamp_end(fail_record)
            from autodex.utils.coverage import pick_reorient_target
            target = pick_reorient_target(obj, pose_stem, hand=hand,
                                           version=args.grasp_version,
                                           obj_root=obj_root, arm=args.arm)
            if target is not None:
                j_int, stem, n_rem = target
                print(f"\n    [reorient] all candidates at tabletop {pose_stem} "
                      f"failed (IK/collision).")
                print(f"    target_j={j_int} (pose {stem}) has {n_rem} "
                      f"uncovered scenes.")
                print(f"    Suggested:")
                print(f"      python src/experiment/reset/reorient.py "
                      f"--obj {obj} --hand {hand} --target_j {j_int} --auto "
                      f"--version {args.grasp_version}")
                try:
                    _cmd = input("    Run reorient then press Enter (q to quit): "
                                 ).strip().lower()
                except KeyboardInterrupt:
                    _cmd = "q"
                fail = {"dir_idx": dir_idx, "scene_type": args.scene,
                        "success": False, "reason": "reorient_needed_ik_fail",
                        "reorient_target_j": j_int,
                        "reorient_target_stem": stem,
                        "reorient_uncovered_n": n_rem,
                        "timing": timing}
                if _cmd == "q":
                    fail["all_done"] = True
                    fail["reason"] = "user_quit_reorient"
                with open(os.path.join(img_dir, "result.json"), "w") as f:
                    json.dump(fail, f, indent=2, default=str)
                return _stamp_end(fail)
        # Fall-through: non-v7, or no reorient target. Stop the cycle.
        fail = {"dir_idx": dir_idx, "scene_type": args.scene, "success": False,
                "reason": "planning_failed_all_candidates",
                "all_done": True, "timing": timing}
        with open(os.path.join(img_dir, "result.json"), "w") as f:
            json.dump(fail, f, indent=2, default=str)
        return _stamp_end(fail)

    plan_dir = os.path.join(img_dir, "plan")
    os.makedirs(plan_dir, exist_ok=True)
    np.save(os.path.join(plan_dir, "traj.npy"), result.traj)
    np.save(os.path.join(plan_dir, "wrist_se3.npy"), result.wrist_se3)
    if result.timing:
        with open(os.path.join(plan_dir, "timing.json"), "w") as f:
            json.dump(result.timing, f, indent=2)
    print(f"    Scene info: {result.scene_info}")

    # Precomputed trajectories (shared with viz so what user sees == what
    # robot executes). Set inside the viz block below if --viz.
    _precomputed_lift_traj = None
    _precomputed_repo_traj = None

    if args.viz:
        print("    Launching visualizer (http://localhost:8080)...")
        sv = ScenePlanVisualizer(scene_cfg, result, port=8080,
                                 hand=_planner_robot(args.arm, hand))
        # Pre-compute lift / repose / place trajectories so viz shows the
        # entire planned motion before execution starts. Obj follows the
        # wrist rigidly through these phases.
        try:
            from autodex.utils.conversion import cart2se3 as _cart2se3
            import torch as _torch
            from scipy.spatial.transform import Rotation as _R
            T_obj_grasp_world = _cart2se3(scene_cfg["mesh"]["target"]["pose"])

            def _fk_wrist(qpos: np.ndarray) -> np.ndarray:
                """Compute WRIST 4×4 pose (= cuRobo ee_link = base_link)."""
                kin = planner._motion_gen.kinematics.get_state(
                    _torch.tensor(qpos, dtype=_torch.float32,
                                  device=planner._tensor_args.device).unsqueeze(0)
                )
                pos = kin.ee_position[0].detach().cpu().numpy()
                quat = kin.ee_quaternion[0].detach().cpu().numpy()
                Rmat = _R.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()
                T = np.eye(4)
                T[:3, :3] = Rmat
                T[:3, 3] = pos
                return T

            grasp_end_qpos = np.asarray(result.traj[-1], dtype=np.float32)
            grasp_end_arm = grasp_end_qpos[:adof]
            T_wrist_grasp_end = _fk_wrist(grasp_end_qpos)
            # Use FK-derived wrist so obj viz at grasp end exactly matches
            # scene_cfg obj pose (no jump at lift start).
            T_obj_in_wrist = np.linalg.inv(T_wrist_grasp_end) @ T_obj_grasp_world

            def _obj_traj_along(robot_traj: np.ndarray) -> np.ndarray:
                """Compute (N, 4, 4) obj poses along trajectory (rigid grasp)."""
                out = np.zeros((len(robot_traj), 4, 4))
                for i, q in enumerate(robot_traj):
                    T_wrist = _fk_wrist(np.asarray(q, dtype=np.float32))
                    out[i] = T_wrist @ T_obj_in_wrist
                return out

            # 1. Lift traj — target wrist pose with z+0.10
            lift_wrist = T_wrist_grasp_end.copy()
            lift_wrist[2, 3] += 0.10
            grasp_full = np.concatenate([
                grasp_end_arm, np.asarray(result.grasp_pose, dtype=np.float32)
            ])
            lift_traj = planner.plan_pose_constrained(
                grasp_full, lift_wrist,
                hold_vec_weight=[1, 1, 1, 1, 1, 0],
                scene_cfg=scene_cfg, include_obj_obstacle=False,
            )
            if lift_traj is not None:
                _precomputed_lift_traj = lift_traj
                lift_obj_traj = _obj_traj_along(lift_traj)
                sv.add_traj("lift", {"traj_robot": lift_traj},
                            obj_traj={"mesh_target": lift_obj_traj})

                # 2. Repose traj (only for v7)
                if _is_coverage_pool(args.grasp_version) and result.scene_info is not None:
                    lift_end_qpos = np.asarray(lift_traj[-1], dtype=np.float32)
                    lift_end_arm = lift_end_qpos[:adof]
                    T_wrist_lift_end = _fk_wrist(lift_end_qpos)
                    T_obj_lift_end = T_wrist_lift_end @ T_obj_in_wrist
                    R_PLACE_VIZ = 0.55
                    obj_z_now = float(T_obj_lift_end[2, 3])
                    R_obj_canonical = T_obj_grasp_world[:3, :3]
                    T_obj_repo = np.eye(4)
                    T_obj_repo[:3, :3] = R_obj_canonical
                    T_obj_repo[:3, 3] = [R_PLACE_VIZ, 0.0, obj_z_now]
                    T_wrist_repo = T_obj_repo @ np.linalg.inv(T_obj_in_wrist)
                    T_wrist_repo[2, 3] = T_wrist_lift_end[2, 3]  # force wrist z match
                    lift_full = np.concatenate([
                        lift_end_arm, np.asarray(result.grasp_pose, dtype=np.float32)
                    ])
                    repo_traj = planner.plan_pose_constrained(
                        lift_full, T_wrist_repo,
                        hold_vec_weight=[0, 0, 0, 0, 0, 1],
                        scene_cfg=scene_cfg, include_obj_obstacle=False,
                    )
                    if repo_traj is not None:
                        _precomputed_repo_traj = repo_traj
                        repo_obj_traj = _obj_traj_along(repo_traj)
                        sv.add_traj("repose", {"traj_robot": repo_traj},
                                    obj_traj={"mesh_target": repo_obj_traj})

                        # 3. Place traj — wrist z descend
                        repo_end_qpos = np.asarray(repo_traj[-1], dtype=np.float32)
                        repo_end_arm = repo_end_qpos[:adof]
                        T_wrist_repo_end = _fk_wrist(repo_end_qpos)
                        place_wrist = T_wrist_repo_end.copy()
                        place_wrist[2, 3] -= 0.10
                        repo_full = np.concatenate([
                            repo_end_arm, np.asarray(result.grasp_pose, dtype=np.float32)
                        ])
                        place_traj = planner.plan_pose_constrained(
                            repo_full, place_wrist,
                            hold_vec_weight=[1, 1, 1, 1, 1, 0],
                            scene_cfg=scene_cfg, include_obj_obstacle=False,
                        )
                        if place_traj is not None:
                            place_obj_traj = _obj_traj_along(place_traj)
                            sv.add_traj("place", {"traj_robot": place_traj},
                                        obj_traj={"mesh_target": place_obj_traj})
        except Exception as _viz_e:
            print(f"    [viz] phase precompute failed: {_viz_e!r}")
        sv.start_viewer(use_thread=True)
        _active_vis = sv

    # ── 4. Execute (stream off, video on) ───────────────────────────────────
    print(f"[4/6] Executing on robot...")
    timing["execution_start"] = _ts()
    # Pause the always-on stream so video can take its place.
    try:
        rcc.stop()
    except Exception:
        pass

    raw_rel = os.path.join("AutoDex", "experiment", args.exp_name, sub, obj, dir_idx, "raw")
    exec_rel = os.path.join(raw_rel, "exec")
    place_rel = os.path.join(raw_rel, "place")
    raw_dir = os.path.join(img_dir, "raw")
    _rcc_start(rcc, "video", True, exec_rel)
    _safe_timestamp_start(timestamp_monitor, os.path.join(raw_dir, "timestamps"))
    executor.start_recording(raw_dir)
    sync_generator.start(fps=30)

    t0 = time.time()
    DEBUG_DUMP_DIR = "/tmp/pose_constrained_debug"
    try:
        s_hand = executor.execute(
            result, planner=planner, scene_cfg=scene_cfg,
            debug_dump_dir=DEBUG_DUMP_DIR,
            lift_traj_override=_precomputed_lift_traj,
        )   # grasp + lift; lift uses precomputed traj if viz on
    except Exception as _exec_e:
        # Contact monitor / SDK errors / etc. during execute would otherwise
        # propagate up to main() and kill the whole experiment loop. Recover
        # the robot (open hand, retract to clear-view), mark the candidate
        # as fail (skip_done filters it next trial), log, then continue.
        print(f"    [execute FAIL] {type(_exec_e).__name__}: {_exec_e!r}")
        try:
            print(f"    [recovery] reset_fallback (hand open + retract) ...")
            executor.reset_fallback(result)
        except Exception as _re:
            print(f"    [recovery] reset_fallback FAILED: {_re!r}")
        _stop_with_timeout("executor.recording", executor.stop_recording)
        # timestamp_monitor FIRST: its stop() waits on event["stop"], which the
        # capture loop only sets after camera.get_timestamp() returns — and that
        # blocks for the next frame. Kill the trigger first and no frame ever
        # arrives, so stop() waits forever. (Every other call site in this file
        # already stops it in this order.)
        _safe_timestamp_stop(timestamp_monitor)
        _stop_with_timeout("sync_generator", sync_generator.stop)
        # NOTE: rcc.stop() pauses the current capture (record / stream).
        # Do NOT call rcc.end() here — that tears down the remote camera
        # controller, which the next trial still needs.
        _stop_with_timeout("rcc", rcc.stop)
        # Persist per-candidate fail so we don't pick the same one again.
        if result.scene_info is not None:
            sei = result.scene_info
            if isinstance(sei, (list, tuple)) and len(sei) == 3:
                from autodex.utils.path import get_candidate_path
                cand_result_path = os.path.join(
                    get_candidate_path(hand), args.grasp_version, obj,
                    sei[0], sei[1], sei[2], "result.json",
                )
                try:
                    with open(cand_result_path, "w") as _f:
                        json.dump({"success": False, "dir_idx": dir_idx,
                                   "arm": args.arm,
                                   "reason": f"execute_{type(_exec_e).__name__}"
                                   }, _f)
                except Exception: pass
        fail = {"dir_idx": dir_idx, "scene_type": args.scene,
                "success": False, "reason": "execute_exception",
                "exception": repr(_exec_e), "timing": timing}
        try:
            with open(os.path.join(img_dir, "result.json"), "w") as _f:
                json.dump(fail, _f, indent=2, default=str)
        except Exception: pass
        return _stamp_end(fail)

    # --auto: lift-time charuco snapshot — object is up, marker on the table
    # should be uncovered. Pause video, capture image set, resume video.
    auto_succ_lift = None
    auto_label_info = {}
    if args.auto:
        _stop_with_timeout("rcc", rcc.stop)
        label_lift_rel = os.path.join("shared_data", "AutoDex", "experiment",
                                       args.exp_name, sub, obj, dir_idx,
                                       "label_at_lift", "raw")
        label_lift_abs = os.path.join(img_dir, "label_at_lift", "raw", "images")
        rcc.start("image", False, label_lift_rel)
        rcc.stop()
        time.sleep(0.3)
        auto_succ_lift, auto_label_info = auto_label_charuco(
            label_lift_abs, required_board=CHARUCO_BOARD)
        if auto_label_info.get("reason"):
            print(f"    [auto-label] FAILED ({auto_label_info['reason']})")
        else:
            print(f"    [auto-label] success={auto_succ_lift}  "
                  f"covered {auto_label_info.get('covered')}/"
                  f"{auto_label_info.get('expected')}")

        # Charuco fail → don't place, recover via reset_hybrid (self-collision
        # + placed-obj collision aware). Record fail to candidate dir so this
        # grasp is skip_done-filtered on the next trial.
        # None = the label could not be JUDGED (no images captured), which is
        # not the robot failing. Recording it as a candidate failure would
        # blacklist a grasp that may well have worked, so the trial is voided
        # instead: recover the arm, write nothing to the candidate dir.
        _label_unjudgeable = auto_succ_lift is None
        if not auto_succ_lift:
            if _label_unjudgeable:
                print(f"    [auto-label] UNJUDGEABLE "
                      f"({auto_label_info.get('reason')}) — voiding this trial, "
                      f"candidate NOT marked failed")
            else:
                print("    [auto-label] charuco FAIL — recovering (reset_hybrid)")
            _stop_with_timeout("executor.recording", executor.stop_recording)
            _safe_timestamp_stop(timestamp_monitor)
            _stop_with_timeout("sync_generator", sync_generator.stop)
            # Release (squeeze→grasp→pregrasp gradient) so reset_hybrid's
            # pregrasp→openpose slow interp starts from the right state.
            try:
                executor.release(result)
            except Exception as re_e:
                print(f"    release FAILED (continuing to retract): {re_e!r}")
            try:
                fb_log = executor.reset(result, planner, scene_cfg)
                timing["retract"] = fb_log
            except Exception as re_e2:
                print(f"    reset FAILED ({re_e2!r}), trying reset_hybrid")
                try:
                    fb_log = executor.reset_hybrid(result, planner, scene_cfg)
                    timing["retract"] = fb_log
                except Exception as fe:
                    timing["retract_error"] = repr(fe)
                    print(f"    reset_hybrid FAILED: {fe!r}")
            _rcc_start(rcc, "stream", False, fps=args.stream_fps)
            # Persist fail to the candidate dir (skip_done filter on next trial).
            # Skipped when the label was unjudgeable — see above.
            if result.scene_info is not None and not _label_unjudgeable:
                from autodex.utils.path import get_candidate_path
                sei = result.scene_info
                if isinstance(sei, (list, tuple)) and len(sei) == 3:
                    cand_result_path = os.path.join(
                        get_candidate_path(hand), args.grasp_version, obj,
                        sei[0], sei[1], sei[2], "result.json",
                    )
                    with open(cand_result_path, "w") as f:
                        json.dump({"success": False, "dir_idx": dir_idx,
                                   "arm": args.arm,
                                   "reason": "charuco_fail"}, f)
            fail = {"dir_idx": dir_idx, "scene_type": args.scene,
                    "success": None if _label_unjudgeable else False,
                    "reason": ("label_unjudgeable" if _label_unjudgeable
                               else "charuco_fail"),
                    "auto_label": auto_label_info, "timing": timing}
            with open(os.path.join(img_dir, "result.json"), "w") as f:
                json.dump(fail, f, indent=2, default=str)
            return _stamp_end(fail)

        # Resume video for place phase.
        _rcc_start(rcc, "video", True, place_rel)

    # Reposition obj at (x=R_PLACE, y=0, current_z) before place. For v7,
    # pick yaw that makes the NEXT cov-greedy grasp IK-reachable. Hold z so
    # obj doesn't dip / rise during reposition (plan_pose_constrained).
    from autodex.utils.conversion import cart2se3
    from autodex.utils.coverage import next_grasp_after_success
    from autodex.utils.path import get_candidate_path

    R_PLACE_DEFAULT = 0.50
    R_PLACE = R_PLACE_DEFAULT   # overridden below if yaw_search picks better r
    T_wrist_now = executor.arm.get_data()["position"] @ executor._link6_to_wrist
    T_obj_grasp = cart2se3(scene_cfg["mesh"]["target"]["pose"])
    T_obj_in_wrist = np.linalg.inv(result.wrist_se3) @ T_obj_grasp
    T_obj_now = T_wrist_now @ T_obj_in_wrist
    obj_z = float(T_obj_now[2, 3])
    # IMPORTANT: use the perception-time obj orientation, NOT the lifted one.
    # plan_pose_constrained's lift can drift wrist orientation by a few
    # degrees → T_obj_now picks that up → target tabletop ends up tilted /
    # standing instead of preserved. Using T_obj_grasp's rotation as the
    # reference means we ROTATE the obj back to the original tabletop pose
    # (rigid grasp guarantees obj will follow whatever wrist motion).
    R_obj_now = T_obj_grasp[:3, :3]

    chosen_yaw = 0.0
    yaw_feasible_n = 0
    _repos = None
    if _is_coverage_pool(args.grasp_version) and pose_stem is not None:
        # Coverage-driven placement: score every (r, yaw) by the UNION of
        # scenes the grasps it makes reachable would newly cover, instead of
        # only chasing the single next set-cover pick below.
        from autodex.utils.reposition import pick_reposition_target, describe
        try:
            _repos = pick_reposition_target(
                obj, pose_stem, hand, args.grasp_version,
                planner=planner, R_obj_robot=R_obj_now, obj_z=obj_z,
                x_preferred=R_PLACE_DEFAULT,
            )
        except Exception as _re:
            print(f"    [place_yaw] reposition search failed: {_re!r}")
            _repos = None
        print(f"    [place_yaw] coverage: {describe(_repos)}")
        if _repos is not None:
            R_PLACE = _repos["x"]
            chosen_yaw = _repos["yaw_rad"]
            yaw_feasible_n = _repos["n_feasible_grasps"]

    # Fallback: no coverage json (or nothing left to open) — aim the placement
    # at whichever grasp the set-cover would pick next, as before.
    if _repos is None and _is_coverage_pool(args.grasp_version) \
            and result.scene_info is not None:
        cur_key = tuple(str(x) for x in result.scene_info)
        next_key = next_grasp_after_success(
            obj, cur_key, tabletop_pose_stem=pose_stem,
            hand=hand, version=args.grasp_version, arm=args.arm,
        )
        if next_key is not None:
            next_path = os.path.join(
                get_candidate_path(hand), args.grasp_version, obj,
                next_key[0], next_key[1], next_key[2], "wrist_se3.npy",
            )
            if os.path.exists(next_path):
                next_wrist_obj = np.load(next_path)
                # Sweep (r, yaw): r in [0.40 .. 0.65] step 0.05, yaw 0..350° step 10°.
                # Picks (r, yaw) that makes next grasp's wrist target IK-feasible.
                # Prefer r closest to default (0.55) — keeps placements predictable
                # on the board when multiple radii work.
                yaws = np.deg2rad(np.arange(0, 360, 10))
                rs = np.arange(0.40, 0.66, 0.05)
                combos = [(r, y) for r in rs for y in yaws]
                wrist_targets = np.zeros((len(combos), 4, 4))
                for i, (r, yaw) in enumerate(combos):
                    c, s = np.cos(yaw), np.sin(yaw)
                    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                    T_obj_target = np.eye(4)
                    T_obj_target[:3, :3] = Rz @ R_obj_now
                    T_obj_target[:3, 3] = [float(r), 0.0, obj_z]
                    wrist_targets[i] = T_obj_target @ next_wrist_obj
                succ = planner.ik_pose_batch(wrist_targets)
                ok_idx = np.where(succ)[0]
                yaw_feasible_n = int(succ.sum())
                if len(ok_idx) > 0:
                    # rank by |r - R_PLACE_DEFAULT|, tie-break first yaw
                    best = min(ok_idx,
                                key=lambda k: (abs(combos[k][0] - R_PLACE_DEFAULT),
                                                combos[k][1]))
                    R_PLACE = float(combos[best][0])
                    chosen_yaw = float(combos[best][1])
                    print(f"    [place_yaw] next={next_key} → "
                          f"r={R_PLACE:.2f}  yaw={np.degrees(chosen_yaw):.0f}°  "
                          f"({yaw_feasible_n}/{len(combos)} feasible)")
                else:
                    print(f"    [place_yaw] next={next_key} 0 (r,yaw) feasible "
                          f"— using r={R_PLACE_DEFAULT}, yaw=0°")
            else:
                print(f"    [place_yaw] next grasp wrist_se3.npy missing: "
                      f"{next_path}")
        else:
            print(f"    [place_yaw] no next grasp in cov order")

    # Move arm so obj ends at (R_PLACE, 0, obj_z) with chosen yaw, holding z.
    c, s = np.cos(chosen_yaw), np.sin(chosen_yaw)
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    T_obj_target = np.eye(4)
    T_obj_target[:3, :3] = Rz @ R_obj_now
    T_obj_target[:3, 3] = [R_PLACE, 0.0, obj_z]
    T_wrist_target = T_obj_target @ np.linalg.inv(T_obj_in_wrist)
    # Force goal wrist z = current wrist z exactly so cuRobo's held-z check
    # passes. Mathematically z is preserved under world-z rotation already,
    # but floating-point chain ops may drift by 1e-7 which trips the check.
    T_wrist_now_world = (executor.arm.get_data()["position"]
                          @ executor._link6_to_wrist)
    T_wrist_target[2, 3] = T_wrist_now_world[2, 3]
    start_full = np.concatenate([
        np.asarray(executor.arm.get_data()["qpos"][:adof], dtype=np.float32),
        np.asarray(result.grasp_pose, dtype=np.float32),
    ])
    if _precomputed_repo_traj is not None:
        print(f"    [reposition] using precomputed repo_traj "
              f"shape={_precomputed_repo_traj.shape}")
        traj_repose = _precomputed_repo_traj
    else:
        traj_repose = planner.plan_pose_constrained(
            start_full, T_wrist_target,
            hold_vec_weight=[0, 0, 0, 0, 0, 1],   # hold z only
            scene_cfg=scene_cfg,
            include_obj_obstacle=False,
            debug_dump_dir=DEBUG_DUMP_DIR,
        )
    timing["place_yaw_deg"] = round(np.degrees(chosen_yaw), 1)
    timing["place_yaw_feasible_n"] = yaw_feasible_n
    if traj_repose is not None:
        arm_repose = traj_repose[:, :adof]
        # Hold hand at squeeze pose during repose (planner traj's hand
        # portion = grasp_pose which is less closed than s_hand and would
        # open fingers mid-motion → drop obj).
        hand_repose = np.tile(s_hand, (len(traj_repose), 1))
        executor._move_joints(arm_repose, hand_repose)
        print(f"    [reposition] obj → (r={R_PLACE}, y=0, "
              f"yaw={np.degrees(chosen_yaw):.0f}°)")
    else:
        print(f"    [reposition] plan_pose_constrained failed — placing here")

    # planner+scene_cfg → cuRobo straight descent (mirror of lift), with the
    # cartesian admittance descent kept as the fallback.
    # In reposition mode the arm was moved AFTER the lift, so the planned grasp
    # wrist is no longer above the object. The xarm's place always descends from
    # where it is; the FR3's default target is the planned grasp wrist, so tell
    # it to descend from the current wrist instead.
    _place_kw = {}
    if args.arm == "franka" and reposition_mode:
        _place_kw["use_current_wrist"] = True
    place_info = executor.place(result, planner=planner, scene_cfg=scene_cfg,
                                debug_dump_dir=DEBUG_DUMP_DIR, **_place_kw)
    timing["execute_s"] = round(time.time() - t0, 2)
    timing["execution_states"] = executor.state_timestamps
    timing["place"] = place_info

    # Release the obj BEFORE stopping cameras so the hand-open moment is
    # captured in the place video. Only release on normal full-descent
    # path; early-contact branch keeps the grasp closed for reset chain.
    _descended_pre = place_info.get("descended", 0.0)
    _target_d_pre = place_info.get("target", 0.0)
    _released_in_video = False
    if not (place_info.get("stopped_on_contact")
            and (_target_d_pre - _descended_pre) > 0.005):
        print(f"[6/6] Releasing (in-video)...")
        executor.release(result)
        _released_in_video = True

    # STOP order: rcc (cameras) first WHILE sync_generator still pulsing
    # — cameras need pulses to flush buffers during stop. Then timestamp
    # and sync last.
    _stop_with_timeout("rcc", rcc.stop)
    _safe_timestamp_stop(timestamp_monitor)
    _stop_with_timeout("sync_generator", sync_generator.stop)

    # Place hit contact mid-descent → object didn't reach the table. Stay at
    # current (mid-descent) height, keep grasp, go straight to reset chain.
    # No release (object is still in hand and will be carried up by reset).
    # Treat as failure only when contact stopped descent BEFORE the target
    # depth — that means the obj hit something mid-descent (table itself at
    # full depth is the EXPECTED stop, not a failure). Threshold 5mm guards
    # against floating-point noise at exactly target.
    _descended = place_info.get("descended", 0.0)
    _target_d = place_info.get("target", 0.0)
    _early_contact = (place_info.get("stopped_on_contact")
                       and (_target_d - _descended) > 0.005)
    if _early_contact:
        print(f"    [place] EARLY contact stop "
              f"({_descended*1000:.1f}mm of "
              f"{_target_d*1000:.1f}mm) — recovering (no release)")
        _stop_with_timeout("executor.recording", executor.stop_recording)
        try:
            fb_log = executor.reset(result, planner, scene_cfg)
            timing["retract"] = fb_log
        except Exception as re_e2:
            print(f"    reset FAILED ({re_e2!r}), trying reset_hybrid")
            try:
                fb_log = executor.reset_hybrid(result, planner, scene_cfg)
                timing["retract"] = fb_log
            except Exception as fe:
                timing["retract_error"] = repr(fe)
                print(f"    reset_hybrid FAILED: {fe!r}")
        _rcc_start(rcc, "stream", False, fps=args.stream_fps)
        if reposition_mode:
            # Reposition fail (contact stop) → update stats with fail.
            if result.scene_info is not None:
                from autodex.utils.coverage import update_grasp_stats
                from autodex.utils.path import get_candidate_path
                sei = result.scene_info
                if isinstance(sei, (list, tuple)) and len(sei) == 3:
                    gd = os.path.join(
                        get_candidate_path(hand), "table_only", obj,
                        sei[0], sei[1], sei[2]
                    )
                    a, s = update_grasp_stats(gd, False)
                    print(f"    [stats] {sei} now {s}/{a} (rate={s/a if a else 0:.2f})")
                    timing["repo_stats"] = {"attempts": a, "successes": s}
        elif result.scene_info is not None:
            from autodex.utils.path import get_candidate_path
            sei = result.scene_info
            if isinstance(sei, (list, tuple)) and len(sei) == 3:
                cand_result_path = os.path.join(
                    get_candidate_path(hand), args.grasp_version, obj,
                    sei[0], sei[1], sei[2], "result.json",
                )
                with open(cand_result_path, "w") as f:
                    json.dump({"success": bool(auto_succ_lift),
                               "dir_idx": dir_idx,
                               "arm": args.arm,
                               "reason": "place_early_contact"}, f)
        # Grasp success criterion = charuco at LIFT (auto_succ_lift). Place
        # quality is a separate metric — early contact during descent does
        # not invalidate a successful grasp. Keep trial.success = grasp
        # success, attach place_info as diagnostic.
        trial_success = bool(auto_succ_lift)
        record = {
            "dir_idx": dir_idx, "scene_type": args.scene,
            "success": trial_success,
            "reason": ("place_early_contact" if trial_success
                       else "charuco_fail_then_place_early_contact"),
            "auto_label_lift": auto_label_info,
            "place": place_info, "timing": timing,
        }
        with open(os.path.join(img_dir, "result.json"), "w") as f:
            json.dump(record, f, indent=2, default=str)
        return _stamp_end(record)

    # ── 5. Label ─────────────────────────────────────────────────────────────
    timing["label_start"] = _ts()
    print(f"[5/6] Label the result")
    if args.auto:
        if reposition_mode:
            # Reposition success = obj covers the charuco board after place
            # (= board NOT fully visible).
            _stop_with_timeout("rcc", rcc.stop)
            repo_post_rel = os.path.join(
                "shared_data", "AutoDex", "experiment", args.exp_name, sub,
                obj, dir_idx, "_repo_check_post", "raw"
            )
            repo_post_abs = os.path.join(
                img_dir, "_repo_check_post", "raw", "images"
            )
            rcc.start("image", False, repo_post_rel)
            rcc.stop()
            time.sleep(0.3)
            post_vis, post_info = auto_label_charuco(
                repo_post_abs, required_board=CHARUCO_BOARD
            )
            timing["repo_charuco_after"] = post_info
            succ = (post_vis is False)  # covered = success
            note = (f"repo post-charuco "
                    f"{post_info.get('covered')}/{post_info.get('expected')}")
            print(f"    [reposition] post={post_info.get('covered')}/"
                  f"{post_info.get('expected')} → success={succ}")
            # Update stats for the chosen table_only grasp.
            if result.scene_info is not None:
                from autodex.utils.coverage import update_grasp_stats
                from autodex.utils.path import get_candidate_path as _gcp
                sei = result.scene_info
                if isinstance(sei, (list, tuple)) and len(sei) == 3:
                    grasp_dir = os.path.join(
                        _gcp(hand), "table_only", obj,
                        sei[0], sei[1], sei[2]
                    )
                    a, s = update_grasp_stats(grasp_dir, succ)
                    print(f"    [stats] {sei} now {s}/{a} "
                          f"(rate={s/a if a else 0:.2f})")
                    timing["repo_stats"] = {"attempts": a, "successes": s}
        else:
            succ = bool(auto_succ_lift)
            note = (auto_label_info.get("reason")
                    or f"charuco covered "
                       f"{auto_label_info.get('covered')}/"
                       f"{auto_label_info.get('expected')}")
            timing["auto_label"] = auto_label_info
            print(f"    auto_label: success={succ}  note={note}")
    else:
        label_rel = os.path.join("shared_data", "AutoDex", "experiment",
                                 args.exp_name, sub, obj, dir_idx, "label", "raw")
        rcc.start("image", False, label_rel)
        rcc.stop()
        try:
            succ, note = get_label()
        except KeyboardInterrupt:
            print("\n[interrupted] Releasing and cleaning up...")
            executor.release(result)
            executor.stop_recording()
            raise

    # ── 6. Release & save ────────────────────────────────────────────────────
    if not _released_in_video:
        print(f"[6/6] Releasing...")
        executor.release(result)
    else:
        print(f"[6/6] Release already done in-video — skipping")

    # reset_hybrid now does the slow pregrasp→openpose interp internally,
    # then keeps hand at openpose during sequential [1,2,0] + cuRobo wrist.
    #     replan around placed object, back to XARM_INIT.
    try:
        fb_log = executor.reset(result, planner, scene_cfg)
        timing["retract"] = fb_log
        print(f"    retract OK  final_qpos_err={fb_log.get('final_qpos_err'):.4f}")
    except Exception as re_e:
        print(f"    reset FAILED ({re_e!r}), trying reset_hybrid")
        try:
            fb_log = executor.reset_hybrid(result, planner, scene_cfg)
            timing["retract"] = fb_log
            print(f"    retract OK (hybrid)  final_qpos_err={fb_log.get('final_qpos_err'):.4f}")
        except Exception as fb_e:
            print(f"    reset_hybrid FAILED: {fb_e!r}, falling back")
            try:
                executor.reset_fallback(result)
            except Exception as ff_e:
                print(f"    reset_fallback FAILED: {ff_e!r}")

    executor.stop_recording()

    if s_hand is not None:
        np.save(os.path.join(img_dir, "squeeze_hand.npy"), s_hand)

    trial_result = {
        "dir_idx": dir_idx,
        "scene_type": args.scene,
        "arm": args.arm,
        "success": succ,
        "scene_info": result.scene_info,
        "candidate_idx": result.timing.get("candidate_idx") if result.timing else None,
        "tabletop_before": tb_before,
        "timing": timing,
    }
    if note is not None:
        trial_result["note"] = note
    with open(os.path.join(img_dir, "result.json"), "w") as f:
        json.dump(trial_result, f, indent=2, default=str)

    # Persist result back to the candidate dir for ALL scenes (table, wall,
    # shelf, v7 wall/shelf, etc.) — both success AND fail. This is what
    # skip_done / skip_scenes_with_success read on the next trial to avoid
    # re-attempting the same candidate. Reposition trials don't write here;
    # their stats.json is updated separately above.
    if (succ is not None and result.scene_info is not None
            and not reposition_mode):
        from autodex.utils.path import get_candidate_path
        sei = result.scene_info
        if isinstance(sei, (list, tuple)) and len(sei) == 3:
            cand_result_path = os.path.join(
                get_candidate_path(hand), args.grasp_version, obj,
                sei[0], sei[1], sei[2], "result.json",
            )
            with open(cand_result_path, "w") as f:
                json.dump({"success": succ, "dir_idx": dir_idx,
                           "arm": args.arm}, f)

    status = "SUCCESS" if succ else ("ISSUE" if succ is None else "FAIL")
    print(f"    Result: {status}  saved to {img_dir}/result.json")

    # Resume the stream so the next trial's init has live SHM frames.
    _rcc_start(rcc, "stream", False, fps=args.stream_fps)

    return _stamp_end(trial_result)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--grasp_version", type=str, default="selected_100")
    parser.add_argument("--exp_name", type=str, default=None, help="Defaults to grasp_version")
    parser.add_argument("--hand", type=str, default="allegro",
                        choices=["allegro", "inspire", "inspire_left"])
    parser.add_argument("--arm", type=str, default="xarm",
                        choices=["xarm", "franka"],
                        help="franka = FR3 + inspire (planner robot fr3_inspire, "
                             "7-DOF arm, FrankaExecutor)")
    parser.add_argument("--scene", type=str, default="table",
                        choices=["table", "wall", "shelf", "cluttered"])
    parser.add_argument("--success_only", action="store_true")
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--max_trials", type=int, default=0,
                        help="0=unlimited. With --ignore_coverage: cap demo run.")
    parser.add_argument("--ignore_coverage", action="store_true",
                        help="Run regardless of scene coverage / past success: "
                             "no coverage filter, no skip_done, no reorient "
                             "suggestion. v7_demo use case.")
    parser.add_argument("--candidate-scene-type", default=None,
                        choices=["table", "wall", "shelf", "box"],
                        help="Use only this candidate scene type while retaining normal skip-done semantics. "
                             "For new Franka collection use 'table' before coverage is computed.")

    # scene-specific args (pass-through to autodex.planner.obstacles.add_obstacles)
    parser.add_argument("--wall_gap", type=float, default=0.04)
    parser.add_argument("--wall_angle", type=float, default=0.0)
    parser.add_argument("--clutter_seed", type=int, default=42)
    parser.add_argument("--clutter_min_dist", type=float, default=0.12)
    parser.add_argument("--clutter_max_dist", type=float, default=0.20)
    parser.add_argument("--clutter_n", type=int, default=4)
    parser.add_argument("--shelf_width", type=float, default=0.30)
    parser.add_argument("--shelf_depth", type=float, default=0.30)
    parser.add_argument("--shelf_height", type=float, default=0.30)
    parser.add_argument("--shelf_gap", type=float, default=0.02)
    parser.add_argument("--no_shelf_back", action="store_true")
    parser.add_argument("--no_shelf_sides", action="store_true")
    parser.add_argument("--no_shelf_top", action="store_true")

    # init pipeline args
    parser.add_argument("--pc_list", type=str, nargs="+", default=DEFAULT_PC_LIST)
    parser.add_argument("--port_mask", type=int, default=5006)
    parser.add_argument("--port_pose", type=int, default=5007)
    parser.add_argument("--port_cmd", type=int, default=6893)
    parser.add_argument("--port_snap", type=int, default=5009)
    parser.add_argument("--port_snap_cmd", type=int, default=6894)
    parser.add_argument("--auto", action="store_true",
                        help="Auto-label via charuco snapshot at lift-time. "
                             "Default off — falls back to manual get_label() prompt.")
    parser.add_argument("--prompt", type=str, default="object on the checkerboard")
    parser.add_argument("--sil_iters", type=int, default=100)
    parser.add_argument("--sil_lr", type=float, default=0.002)
    parser.add_argument("--init_timeout_s", type=float, default=60.0,
                        help="Max wait for masks+poses. Ctrl-C during the "
                             "wait cuts it short and continues with "
                             "whatever arrived.")
    parser.add_argument("--calib_dir", type=str, default=None,
                        help="Camera calib dir. Default: latest under ~/shared_data/cam_param/.")
    parser.add_argument("--stream_fps", type=int, default=10)
    parser.add_argument("--stream_warmup_s", type=float, default=2.0)

    args = parser.parse_args()
    if args.exp_name is None:
        args.exp_name = args.grasp_version

    # scene_prefix: '' (table), 'wall', 'shelf', 'cluttered', plus '_success_only' suffix.
    scene_prefix = args.scene if args.scene != "table" else ""
    if args.success_only:
        scene_prefix = f"{scene_prefix}_success_only" if scene_prefix else "success_only"

    # Mesh / FoundPose assets sanity check.
    # v8 candidates are expressed against object_processing. FoundPose must
    # initialize against that exact mesh frame; otherwise physical Franka
    # collection can mark grasps that are offset when the continuous runner
    # later consumes them.
    mesh_root = Path(get_obj_root(args.grasp_version))
    mesh_path = mesh_root / args.obj / "raw_mesh" / f"{args.obj}.obj"
    assets_root = ASSETS_BASE / args.obj
    if not mesh_path.exists():
        sys.exit(f"mesh not found: {mesh_path}")
    if not (assets_root / "object_repre/v1" / args.obj / "1/repre.pth").exists():
        sys.exit(f"repre.pth missing for {args.obj} (expected under {assets_root})")

    # FoundPose and the planner now share the version-resolved mesh. Keep this
    # check because a caller may still point a non-v8 pool at a mismatched
    # custom object tree.
    from src.execution.scene_cfg import check_mesh_frame_match
    _frame_ok, _frame_msg = check_mesh_frame_match(
        args.obj, str(mesh_path), get_obj_root(args.grasp_version))
    if not _frame_ok:
        sys.exit(f"[mesh_frame] {_frame_msg}")
    print(f"[mesh_frame] {_frame_msg}")

    # Calibration.
    if args.calib_dir:
        calib_dir = Path(args.calib_dir).expanduser()
    else:
        calib_dir = sorted(CAM_PARAM_ROOT.iterdir())[-1]
    print(f"calib: {calib_dir.name}")
    intrinsics_full, extrinsics_full, H, W = _load_calib(calib_dir)

    pc_ips = [get_pc_ip(p) for p in args.pc_list]
    pc_serials = {p: get_camera_list(p) for p in args.pc_list}
    active_serials = {s for pc in args.pc_list for s in pc_serials[pc]}
    intrinsics_full = {s: v for s, v in intrinsics_full.items() if s in active_serials}
    extrinsics_full = {s: v for s, v in extrinsics_full.items() if s in active_serials}
    print(f"  {len(intrinsics_full)} cams active across {len(args.pc_list)} PCs  ({H}x{W})")

    # Hardware init.
    # stall_timeout > the arm-to-trigger gap. Cameras are armed with
    # syncMode=True and produce nothing until sync_generator.start() runs a
    # few statements later, so the default 3 s logs a 20-camera "no new
    # frames" block on every single trial that means nothing.
    rcc = remote_camera_controller("run_auto", pc_list=args.pc_list,
                                   stall_timeout=15.0)
    _ensure_camera_lock(rcc)
    _clear_camera_errors(rcc)
    sync_generator = UTGE900(**network_info["signal_generator"]["param"])
    timestamp_monitor = TimestampMonitor(**network_info["timestamp"]["param"])

    print(f"[stream] starting on {len(args.pc_list)} PCs @ {args.stream_fps} FPS...")
    _rcc_start(rcc, "stream", False, fps=args.stream_fps)
    if args.stream_warmup_s > 0:
        time.sleep(args.stream_warmup_s)

    # Init orchestrator (FoundPose distributed).
    print(f"[orch] initializing for {args.obj}...")
    orch = InitOrchestrator(
        pc_list=args.pc_list, capture_ips=pc_ips,
        port_mask=args.port_mask, port_pose=args.port_pose, port_cmd=args.port_cmd,
    )
    orch.init_object(
        obj_name=args.obj,
        mesh_path=str(mesh_path), assets_root=str(assets_root),
        intrinsics_full=intrinsics_full, extrinsics_full=extrinsics_full,
        image_hw=(H, W), mode="live", pc_serials=pc_serials,
    )

    # The planner's robot config keys on arm+hand; the CANDIDATE pool keys on
    # hand alone (fr3 and xarm share the inspire grasp pools), which is why
    # planner.plan() below still gets args.hand.
    planner_robot = _planner_robot(args.arm, args.hand)
    print(f"[planner] warming up ({planner_robot})...")
    planner = GraspPlanner(hand=planner_robot)
    quiet_curobo()   # GraspPlanner init re-runs curobo's own logger setup
    print(f"[executor] connecting to robot ({args.arm})...")
    if args.arm == "franka":
        from src.execution.franka_executor import FrankaExecutor
        executor = FrankaExecutor(hand_name=args.hand)
        # Park OUT of the cameras' view before the first perception so the arm
        # never occludes the object (the xarm's INIT pose is already clear).
        print("[executor] homing to clear-view...")
        executor.home(clear_view=True)
    else:
        executor = RealExecutor(hand_name=args.hand)

    def _cleanup():
        print("\n[cleanup] Stopping hardware...")
        # Order: rcc first (cameras need pulses to flush during stop), then
        # timestamp, sync_generator last.
        for fn in (rcc.stop, timestamp_monitor.stop, sync_generator.stop,
                   executor.stop_recording):
            try:
                fn()
            except Exception:
                pass

    results: List[dict] = []
    trial = 0
    try:
        while True:
            trial += 1
            print(f"\n{'#'*60}\n# Trial {trial}\n{'#'*60}")
            chime.info()
            if not args.auto:
                try:
                    cmd = input("Press Enter to start trial, 'q' to quit: ").strip().lower()
                except KeyboardInterrupt:
                    _cleanup()
                    break
                if cmd == "q":
                    break
            # --auto: no pre-perception charuco check; perception runs first
            # and decides:
            #   pose_world None        → perception_failed prompt (case 1)
            #   pose_world OK + ...    → normal trial flow

            # Coverage snapshot BEFORE the trial — used to compute how
            # many new scenes the trial just covered.
            if _is_coverage_pool(args.grasp_version):
                from autodex.utils.coverage import (
                    uncovered_scenes, _tabletop_stems,
                )
                _stems_before = _tabletop_stems(
                    args.obj, get_obj_root(args.grasp_version))
                _rem_before = {}
                for _s in _stems_before:
                    _u = uncovered_scenes(args.obj, _s, hand=args.hand,
                                          version=args.grasp_version,
                                          arm=args.arm)
                    _rem_before[_s] = (len(_u) if _u is not None else None)

            tr = run_single_trial(
                args, scene_prefix=scene_prefix,
                orch=orch, planner=planner, executor=executor,
                rcc=rcc, sync_generator=sync_generator,
                timestamp_monitor=timestamp_monitor,
            )
            results.append(tr)
            n_succ = sum(1 for r in results if r.get("success"))
            print(f"\n    Running total: {n_succ}/{len(results)} success")

            # After-trial coverage summary (v7 only). Show per-tabletop
            # remaining uncovered count + how many scenes this trial just
            # covered (delta vs before).
            if _is_coverage_pool(args.grasp_version) and tr.get("success"):
                from autodex.utils.coverage import uncovered_scenes
                lines = []
                total_now = 0
                total_before = 0
                for _s, _b in _rem_before.items():
                    _u = uncovered_scenes(args.obj, _s, hand=args.hand,
                                          version=args.grasp_version,
                                          arm=args.arm)
                    _n = (len(_u) if _u is not None else None)
                    if _b is None or _n is None:
                        lines.append(f"      pose={_s}: N/A")
                        continue
                    _delta = _b - _n
                    lines.append(f"      pose={_s}: {_n} uncovered "
                                 f"(was {_b}, -{_delta} this trial)")
                    total_now += _n
                    total_before += _b
                print(f"    [coverage] after trial (success):")
                for ln in lines:
                    print(ln)
                print(f"      TOTAL remaining: {total_now} "
                      f"(was {total_before}, "
                      f"-{total_before - total_now} this trial)")

            if tr.get("all_done") and not args.ignore_coverage:
                print(f"\n    All scenes done for {args.obj} — stopping loop.")
                break
            if args.max_trials and len(results) >= args.max_trials:
                print(f"\n    --max_trials {args.max_trials} reached — stopping loop.")
                break
    finally:
        # Summary + cleanup.
        print(f"\n{'='*60}\nSUMMARY: {args.obj} x {len(results)} trials")
        n_succ = sum(1 for r in results if r.get("success"))
        if results:
            print(f"  Success: {n_succ}/{len(results)} ({100*n_succ/len(results):.0f}%)")
        for r in results:
            status = "OK" if r.get("success") else r.get("reason", "FAIL")
            print(f"  {r['dir_idx']}: {status}")

        sub = f"{scene_prefix}/{args.hand}" if scene_prefix else args.hand
        summary_path = os.path.join(project_dir, "experiment", args.exp_name, sub,
                                    args.obj, "summary.json")
        os.makedirs(os.path.dirname(summary_path), exist_ok=True)
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2, default=str)

        try:
            executor.shutdown()
        except Exception:
            pass
        try:
            orch.close()
        except Exception:
            pass
        for fn in (timestamp_monitor.end, sync_generator.end, rcc.stop, rcc.end):
            try:
                fn()
            except Exception:
                pass


if __name__ == "__main__":
    main()
