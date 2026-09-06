#!/usr/bin/env python3
"""Banana pick-and-place demo: grasp the object, put it on the marked spot.

Same shape as ``src/execution/run_auto.py`` (stream -> FoundPose init -> plan ->
video-recorded execution -> human label -> artifacts on the NAS), with the
experiment logic stripped out:

  * the GRASP is not searched — it replays a candidate that already SUCCEEDED
    at the tabletop pose the object is lying in right now (``success_grasps``)
  * NO scene obstacles besides the table; the demo table is bare
  * the PLACE target is not a fixed x — it is the center of the standalone
    ArUco marker on the table, triangulated once at start (``place_target``)
  * the DROP ORIENTATION is free: the pre-flight sweeps world-z yaw and keeps
    the reachable one closest to as-picked, and grasps that cannot reach the
    marker are dropped from the pool BEFORE planning
  * the object is carried at the lift height and released ``--drop_h`` (3 cm)
    below it — it is never lowered back onto the table
  * success is NOT auto-judged: the operator labels each trial (y/n/c), then
    repositions the object, types the GRID INDEX it was placed at and presses
    Enter for the next one (run_auto style). The index is typed rather than
    inferred: the protocol's grid cells are what the numbers get reported
    against, and nobody can read the object's center in robot frame off a table
  * every trial records PER-MODULE timing (perception / grasp plan / place
    filter / pre-flight / grasp exec / carry / place / retract) and the report
    breaks the success rate down per (location, orientation) — the numbers the
    OpenArm comparison protocol asks for

Everything lands under ``{project_dir}/experiment/banana_demo/`` (override with
``--exp_name``).

    bash scripts/init_daemons.sh start          # capture PCs
    ~/paradex/cpp/franka_daemon/run_daemon.sh   # franka PC

    # target marker is re-triangulated at start
    python src/demo/banana_test/run_demo.py --obj banana

    # reuse an earlier capture for the target (no new image set)
    python src/demo/banana_test/run_demo.py --obj banana \
        --target_capture_dir ~/shared_data/mingi_erasethis/20260826_182132

    # force a drop yaw / a different release height
    python src/demo/banana_test/run_demo.py --obj banana --target_yaw_deg 90
    python src/demo/banana_test/run_demo.py --obj banana --drop_h 0.05
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
from typing import List, Optional

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

import chime

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.io.camera_system.signal_generator import UTGE900
from paradex.io.camera_system.timestamp_monitor import TimestampMonitor
from paradex.utils.system import network_info, get_pc_ip, get_camera_list
from paradex.calibration.utils import save_current_C2R, save_current_camparam, load_c2r

from autodex.utils.path import project_dir, get_obj_root
from autodex.utils.conversion import cart2se3
from autodex.utils.symmetry import get_cyl_axis_local, get_cyl_yaw_grid
from autodex.planner import GraspPlanner
from autodex.planner.planner import _to_curobo_world
from autodex.planner.obstacles import TABLE_CUBOID
from autodex.planner.obstacles import add_obstacles
from autodex.perception.init_orchestrator import InitOrchestrator

from src.execution.scene_cfg import pose_world_to_scene_cfg
from src.execution.label import get_label
from src.experiment.reset.tabletop_pose import classify_tabletop_pose

from src.demo.banana_test.place_target import locate_marker, capture_images
from src.demo.banana_test.success_grasps import success_keys_at_pose

DEFAULT_PC_LIST = ["capture1", "capture2", "capture3", "capture5", "capture6"]
ASSETS_BASE = Path.home() / "shared_data/AutoDex/foundpose_assets"
MESH_BASE = Path.home() / "shared_data/AutoDex/object/paradex"
CAM_PARAM_ROOT = Path.home() / "shared_data/cam_param"
# Capture the place marker was last triangulated from. Reused by default so a
# run does not spend an image capture re-finding a marker that has not moved.
DEFAULT_TARGET_CAPTURE = str(Path.home() /
                             "shared_data/mingi_erasethis/20260826_200751")
LIFT_HEIGHT = 0.10


# ── camera helpers (same semantics as run_auto) ──────────────────────────────

def _rcc_start(rcc, mode, sync_mode, save_path=None, fps=30):
    """'stream' / 'video' / 'full' are sinks on an armed capture now."""
    if mode == "stream":
        rcc.arm(syncMode=sync_mode, fps=fps)
        rcc.set_stream(True)
    elif mode == "full":
        rcc.arm(syncMode=sync_mode, fps=fps)
        rcc.set_record(save_path=save_path, on=True)
        rcc.set_stream(True)
    elif mode == "video":
        rcc.arm(syncMode=sync_mode, fps=fps)
        rcc.set_record(save_path=save_path, on=True)
    else:
        rcc.start(mode, sync_mode, save_path, fps=fps)


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



def _fk_wrist(planner, qpos_full) -> np.ndarray:
    """cuRobo's own FK for the wrist (= ee_link), for comparing against what
    the robot reports. A mismatch here is a frame/calibration problem, not a
    planning one, and it shows up as "asked to descend 3cm, descended 4.6cm"."""
    import torch
    from scipy.spatial.transform import Rotation as R
    kin = planner._motion_gen.kinematics.get_state(
        torch.tensor(np.asarray(qpos_full, dtype=np.float32),
                     device=planner._tensor_args.device).unsqueeze(0))
    T = np.eye(4)
    q = kin.ee_quaternion[0].detach().cpu().numpy()          # wxyz
    T[:3, :3] = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    T[:3, 3] = kin.ee_position[0].detach().cpu().numpy()
    return T


def _wrist_now(planner, executor, adof: int, hand_qpos) -> np.ndarray:
    """Current wrist pose IN THE PLANNER'S FRAME.

    Do NOT use ``arm.get_data()["position"] @ _link6_to_wrist`` to build cuRobo
    targets: measured on this robot, that frame sits 107.2 mm (and 180 deg) away
    from cuRobo's ee_link, constant in the wrist's own frame. Feeding a target
    built from it into plan_pose_constrained shifts the goal by that offset's
    world-z component -- which swings between +4 mm and -62 mm depending on the
    wrist's orientation, and is why a 3 cm descend came out as 4.6 cm.
    """
    q = np.concatenate([
        np.asarray(executor.arm.get_data()["qpos"][:adof], dtype=np.float32),
        np.asarray(hand_qpos, dtype=np.float32)])
    return _fk_wrist(planner, q)


def _stop_with_timeout(name: str, fn, timeout: float = 20.0) -> bool:
    """Run a shutdown/capture call with a deadline, naming it if it hangs.

    rcc.stop(), the timestamp monitor and executor.stop_recording() all end in
    an UNTIMED Event.wait(), which is not even Ctrl-C interruptible: one dead
    camera wedges the trial with no clue which call is stuck. run_auto has the
    same guard for the same reason.
    """
    import threading
    done, err = threading.Event(), []

    def _run():
        try:
            fn()
        except Exception as exc:
            err.append(exc)
        finally:
            done.set()

    threading.Thread(target=_run, daemon=True, name=f"stop-{name}").start()
    if not done.wait(timeout):
        print(f"    [shutdown] {name} did not return in {timeout:.0f}s — "
              f"leaving it and continuing")
        return False
    if err:
        print(f"    [shutdown] {name} raised: {err[0]!r}")
    return True


def _safe(name, fn, *a, **kw):
    try:
        return fn(*a, **kw)
    except Exception as e:
        print(f"    [{name}] {e!r}")
        return None


def quiet_curobo(level=logging.WARNING) -> None:
    for n in ("curobo", "curobo.util.logger", "curobo.wrap.reacher.motion_gen"):
        logging.getLogger(n).setLevel(level)


def _planner_robot(arm: str, hand: str) -> str:
    if arm == "xarm":
        return hand
    if hand != "inspire":
        raise SystemExit("FR3 demo currently supports only --hand inspire")
    return "fr3_inspire"


def _load_calib(calib_dir: Path):
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


def _rot_z(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _yaw_grid(forced: Optional[float], step: int) -> List[float]:
    """Drop yaws to try, closest-to-as-picked first (0, +step, -step, ...)."""
    if forced is not None:
        return [float(forced)]
    grid = [0.0]
    for d in range(step, 181, step):
        grid += [float(d), float(-d)]
    return grid


def _place_wrist(T_obj_grasp, T_obj_in_wrist, target_xy, yaw_deg,
                 obj_z, wrist_z) -> np.ndarray:
    """Wrist pose that holds the object over ``target_xy`` rotated by ``yaw``."""
    T = np.eye(4)
    T[:3, :3] = _rot_z(np.deg2rad(yaw_deg)) @ T_obj_grasp[:3, :3]
    T[:3, 3] = [target_xy[0], target_xy[1], obj_z]
    W = T @ np.linalg.inv(T_obj_in_wrist)
    W[2, 3] = wrist_z
    return W


def filter_by_place_reach(planner, keys, obj, hand, version, T_obj_grasp,
                          target_xy, yaws) -> List[tuple]:
    """Keep only grasps that can still REACH the drop spot while holding the obj.

    The planner picks a grasp without knowing where the object has to go, so a
    grasp that is perfectly graspable can leave the arm unable to reach the
    marker -- which the post-plan pre-flight would then reject, wasting the
    trial. Screening here (IK only, one batch) drops those up front.

    A candidate's ``wrist_se3.npy`` is stored in the OBJECT frame, so the
    object-in-wrist transform is just its inverse.
    """
    from autodex.utils.path import get_candidate_path
    root = os.path.join(get_candidate_path(hand), version, obj)
    probes, owner = [], []
    for i, (t, sid, gid) in enumerate(keys):
        f = os.path.join(root, t, sid, gid, "wrist_se3.npy")
        if not os.path.exists(f):
            continue
        w_obj = np.load(f)
        T_obj_in_wrist = np.linalg.inv(w_obj)
        wrist_z = float((T_obj_grasp @ w_obj)[2, 3]) + LIFT_HEIGHT
        obj_z = float(T_obj_grasp[2, 3]) + LIFT_HEIGHT
        for y in yaws:
            probes.append(_place_wrist(T_obj_grasp, T_obj_in_wrist,
                                        target_xy, y, obj_z, wrist_z))
            owner.append(i)
    if not probes:
        return list(keys)
    ok = np.asarray(planner.ik_pose_batch(np.array(probes))).reshape(-1)
    reachable = {owner[j] for j, f in enumerate(ok) if f}
    return [k for i, k in enumerate(keys) if i in reachable]


# ── one trial ────────────────────────────────────────────────────────────────

def run_trial(args, *, orch, planner, executor, rcc, sync_generator,
              timestamp_monitor, target_xyz: np.ndarray, run_dir: str) -> dict:
    obj, hand = args.obj, args.hand
    adof = getattr(executor, "arm_dof", 6)
    dir_idx = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    img_dir = os.path.join(run_dir, dir_idx)
    os.makedirs(img_dir, exist_ok=True)
    timing: dict = {"trial_start": datetime.datetime.now().isoformat()}

    t_trial0 = time.time()

    def _finish(d: dict) -> dict:
        timing["trial_end"] = datetime.datetime.now().isoformat()
        # Wall clock minus the human's own thinking time — the label prompt
        # blocks, so leaving it in would make the pipeline look arbitrarily slow.
        timing["trial_wall_s"] = round(time.time() - t_trial0, 2)
        timing["pipeline_s"] = round(timing["trial_wall_s"]
                                     - timing.get("m_label_s", 0.0), 2)
        d["modules_s"] = {k[2:-2]: v for k, v in timing.items()
                          if k.startswith("m_") and k.endswith("_s")}
        d["pipeline_s"] = timing["pipeline_s"]
        d["timing"] = timing
        with open(os.path.join(img_dir, "result.json"), "w") as f:
            json.dump(d, f, indent=2, default=str)
        return d

    print(f"\n{'='*60}\n[1/6] Trial dir -> {dir_idx}")
    save_current_C2R(img_dir)
    save_current_camparam(img_dir)

    # ── 2. perception ────────────────────────────────────────────────────────
    print(f"[2/6] Init pipeline (FoundPose distributed)...")
    t0 = time.time()
    pose_world, perc_timing = orch.trigger_init(
        prompt=args.prompt,
        save_capture_dir=os.path.join(img_dir, "init_capture"),
        sil_iters=args.sil_iters, sil_lr=args.sil_lr,
        timeout_s=args.init_timeout_s,
    )
    timing["m_perception_s"] = round(time.time() - t0, 2)
    timing["perception_detail"] = perc_timing
    if pose_world is None:
        # Perception failing is not a grasp failure -- counting it would poison
        # the success rate, and it always needs a human (lighting, occlusion,
        # object outside the cameras' overlap). Stop the session instead.
        reason = (perc_timing or {}).get("reason", "perception_failed")
        print(f"    Perception FAILED ({reason}) — stopping the session.")
        chime.error()
        return _finish({"dir_idx": dir_idx, "success": None, "reason": reason,
                        "grid_idx": args.loc, "ori_idx": args.ori,
                        "quit": True})
    print(f"    Perception: {timing['m_perception_s']}s")
    np.save(os.path.join(img_dir, "pose_world.npy"), pose_world)

    # ── 3. scene + grasp selection ───────────────────────────────────────────
    print(f"[3/6] Planning (version={args.grasp_version}, table only)...")
    c2r = load_c2r(img_dir)
    obj_root = get_obj_root(args.grasp_version)
    scene_cfg = pose_world_to_scene_cfg(pose_world, c2r, obj, obj_root)
    scene_cfg = add_obstacles(scene_cfg, "table")
    pose_robot = np.linalg.inv(c2r) @ pose_world
    tb = classify_tabletop_pose(pose_robot, obj, obj_root)
    pose_stem = tb["filename"].replace(".npy", "") if tb else None
    timing["tabletop_before"] = tb
    print(f"    obj pos (robot): {pose_robot[:3, 3].round(3)}  tabletop={pose_stem}")

    at_pose, any_pose = success_keys_at_pose(
        obj, hand, args.grasp_version, pose_stem, arm=args.arm)
    if at_pose:
        cand_order = at_pose
        src = f"success@tabletop {pose_stem}"
    elif any_pose and args.allow_other_pose:
        cand_order = any_pose
        src = "success@any tabletop (--allow_other_pose)"
        print(f"    [grasp] no success at tabletop {pose_stem} — "
              f"falling back to all {len(any_pose)} successes")
    else:
        print(f"    [grasp] no successful grasp recorded for "
              f"{obj}/{hand}/{args.grasp_version} at tabletop {pose_stem} "
              f"(arm={args.arm}). Reorient the object, or pass "
              f"--allow_other_pose.")
        _safe("input", input, "    Enter to continue (q to quit): ")
        return _finish({"dir_idx": dir_idx, "success": False,
                        "reason": "no_success_grasp_at_tabletop",
                        "tabletop": pose_stem})
    # Drop the grasps from which the marker is out of reach, BEFORE planning.
    T_obj_grasp = cart2se3(scene_cfg["mesh"]["target"]["pose"])
    yaws = _yaw_grid(args.target_yaw_deg, args.yaw_step)
    if getattr(planner, "_ik_solver", None) is None:
        # plan() builds the solver as a side effect; on the very first trial we
        # need it BEFORE planning, so build a table-only one here.
        planner._init_ik_solver(_to_curobo_world(
            {"mesh": {}, "cuboid": {"table": TABLE_CUBOID}}))
    _t = time.time()
    reach = filter_by_place_reach(planner, cand_order, obj, hand,
                                  args.grasp_version, T_obj_grasp,
                                  target_xyz[:2], yaws)
    timing["m_place_filter_s"] = round(time.time() - _t, 3)
    print(f"    [grasp] {len(cand_order)} success candidates from {src}; "
          f"{len(reach)} can also reach the drop spot")
    if reach:
        cand_order = reach
    else:
        print(f"    [grasp] NONE can reach the marker — planning anyway, "
              f"the pre-flight will decide")
    timing["candidate_pool"] = {"source": src, "keys": cand_order,
                                "n_place_reachable": len(reach)}

    t0 = time.time()
    result = planner.plan(
        scene_cfg, obj, args.grasp_version,
        skip_done=False, success_only=False, hand=hand,
        scene_id=None, scene_type_filter=None,
        skip_scenes_with_success=False,
        openpose_pose_stem=pose_stem,
        cyl_axis_local=get_cyl_axis_local(obj),
        cyl_yaw_grid=get_cyl_yaw_grid(obj),
        candidate_order=cand_order,
    )
    timing["m_grasp_plan_s"] = round(time.time() - t0, 2)
    if not result.success:
        print(f"    Plan FAILED: {result.timing}")
        chime.error()
        return _finish({"dir_idx": dir_idx, "success": False,
                        "reason": "plan_failed", "tabletop": pose_stem})
    print(f"    plan OK  scene_info={result.scene_info}")

    # Pre-flight: can we even repose the object onto the marker once lifted?
    # Check BEFORE grasping so a failure costs nothing (same guard as
    # rotate_obj_yaw.py). The DROP ORIENTATION DOES NOT MATTER for this demo,
    # so instead of one fixed yaw we sweep world-z yaw and keep the feasible
    # one closest to "as picked" (least wrist rotation on the way over).
    T_obj_in_wrist = np.linalg.inv(result.wrist_se3) @ T_obj_grasp
    T_wrist_lift = result.wrist_se3.copy()
    T_wrist_lift[2, 3] += LIFT_HEIGHT
    obj_z_lifted = float((T_wrist_lift @ T_obj_in_wrist)[2, 3])

    def _wrist_at(yaw_deg: float, obj_z: float, wrist_z: float) -> np.ndarray:
        return _place_wrist(T_obj_grasp, T_obj_in_wrist, target_xyz,
                            yaw_deg, obj_z, wrist_z)

    _t = time.time()
    probe = np.array([_wrist_at(y, obj_z_lifted, T_wrist_lift[2, 3])
                      for y in yaws])
    ok = np.asarray(planner.ik_pose_batch(probe)).reshape(-1)
    timing["m_preflight_s"] = round(time.time() - _t, 3)
    feasible = [y for y, f in zip(yaws, ok) if f]
    if not feasible:
        print(f"    [pre-flight] place target {np.round(target_xyz, 3)} is "
              f"IK-infeasible at every yaw ({len(yaws)} tried) — "
              f"refusing to grasp.")
        chime.error()
        return _finish({"dir_idx": dir_idx, "success": False,
                        "reason": "place_target_ik_infeasible",
                        "target_xyz": target_xyz.tolist(),
                        "tabletop": pose_stem})
    place_yaw = feasible[0]
    timing["place_yaw_deg"] = place_yaw
    timing["place_yaw_feasible"] = feasible
    print(f"    [pre-flight] place yaw = {place_yaw:+.0f}deg "
          f"({len(feasible)}/{len(yaws)} yaws feasible)")
    # Dry-run the WHOLE motion before touching the robot. planner.plan() only
    # IK-checks the lift (`Lift IK check`); the trajectory optimiser is what
    # returns INVALID_START_STATE_WORLD_COLLISION, and finding that out after
    # the object is already squeezed leaves the arm stuck holding it with no
    # plan for anywhere to go. Fail here instead, before the grasp.
    _t = time.time()
    grasp_end = np.asarray(result.traj[-1], dtype=np.float32)
    T_wrist_grasp = _fk_wrist(planner, grasp_end)
    # FK-derived (not result.wrist_se3) so the lift starts exactly where the
    # planner thinks the executed trajectory ends.
    T_oiw = np.linalg.inv(T_wrist_grasp) @ T_obj_grasp

    def _full(q_arm):
        return np.concatenate([np.asarray(q_arm[:adof], dtype=np.float32),
                               np.asarray(result.grasp_pose, dtype=np.float32)])

    lift_wrist = T_wrist_grasp.copy()
    lift_wrist[2, 3] += LIFT_HEIGHT
    lift_traj = planner.plan_pose_constrained(
        _full(grasp_end), lift_wrist, hold_vec_weight=[1, 1, 1, 1, 1, 0],
        scene_cfg=scene_cfg, include_obj_obstacle=False)
    if lift_traj is None:
        print(f"    [dry-run] LIFT plan failed — refusing to grasp "
              f"(the arm would end up holding the object with nowhere to go)")
        chime.error()
        return _finish({"dir_idx": dir_idx, "success": False,
                        "reason": "lift_plan_failed", "tabletop": pose_stem,
                        "scene_info": result.scene_info})

    lift_end = np.asarray(lift_traj[-1], dtype=np.float32)
    T_wrist_lift = _fk_wrist(planner, lift_end)
    obj_z_lift = float((T_wrist_lift @ T_oiw)[2, 3])
    carry_wrist = _place_wrist(T_obj_grasp, T_oiw, target_xyz, place_yaw,
                               obj_z_lift, float(T_wrist_lift[2, 3]))
    carry_traj = planner.plan_pose_constrained(
        _full(lift_end), carry_wrist, hold_vec_weight=[0, 0, 0, 0, 0, 1],
        scene_cfg=scene_cfg, include_obj_obstacle=False)
    if carry_traj is None:
        print(f"    [dry-run] CARRY plan failed — refusing to grasp")
        chime.error()
        return _finish({"dir_idx": dir_idx, "success": False,
                        "reason": "carry_plan_failed", "tabletop": pose_stem,
                        "scene_info": result.scene_info})
    timing["m_dryrun_s"] = round(time.time() - _t, 2)
    print(f"    [dry-run] lift + carry both plan OK "
          f"({timing['m_dryrun_s']}s)")

    np.save(os.path.join(img_dir, "plan_traj.npy"), np.asarray(result.traj))
    np.save(os.path.join(img_dir, "grasp_wrist_se3.npy"), result.wrist_se3)

    # ── 4. execute (stream off, video on) ────────────────────────────────────
    print(f"[4/6] Executing (grasp + lift)...")
    _safe("rcc.stop", rcc.stop)
    raw_rel = os.path.join("AutoDex", "experiment", args.exp_name,
                           f"{args.arm}_{hand}", obj, dir_idx, "raw")
    raw_dir = os.path.join(img_dir, "raw")
    if not args.no_video:
        _rcc_start(rcc, "video", True, os.path.join(raw_rel, "exec"))
        _safe_timestamp_start(timestamp_monitor,
                              os.path.join(raw_dir, "timestamps"))
        _safe("executor.start_recording", executor.start_recording, raw_dir)
        try:
            sync_generator.start(fps=30)
        except Exception as _se:
            # No trigger = no frames on syncMode=True cameras: the trial would
            # record a fraction of a second and then stall silently.
            print(f"    [sync_generator] START FAILED: {_se!r} — the cameras "
                  f"are armed for external trigger and will NOT record.")

    t0 = time.time()
    try:
        s_hand = executor.execute(result, planner=planner, scene_cfg=scene_cfg,
                                  lift_height=LIFT_HEIGHT,
                                  lift_traj_override=lift_traj)
    except Exception as e:
        print(f"    execute FAILED: {e!r}")
        for fn, nm in ((rcc.stop, "rcc.stop"),
                       (lambda: _safe_timestamp_stop(timestamp_monitor),
                        "timestamp_monitor.stop"),
                       (sync_generator.stop, "sync_generator.stop"),
                       (executor.stop_recording, "executor.stop_recording")):
            _stop_with_timeout(nm, fn)
        _safe("reset_fallback", executor.reset_fallback, result)
        _rcc_start(rcc, "stream", False, fps=args.stream_fps)
        return _finish({"dir_idx": dir_idx, "success": False,
                        "reason": "execute_exception", "exception": repr(e)})
    timing["m_grasp_exec_s"] = round(time.time() - t0, 2)

    # ── 5. repose over the marker, then place ────────────────────────────────
    print(f"[5/6] Carry to marker (constant height) + drop {np.round(target_xyz, 3)} + place...")
    T_wrist_now = _wrist_now(planner, executor, adof, result.grasp_pose)
    T_obj_now = T_wrist_now @ T_oiw
    # Constant height: the goal wrist z IS the current wrist z, and
    # plan_pose_constrained holds z along the way. Rotation comes off the
    # PERCEPTION-time orientation so lift drift doesn't accumulate.
    T_wrist_target = _wrist_at(place_yaw, float(T_obj_now[2, 3]),
                               float(T_wrist_now[2, 3]))
    start_full = np.concatenate([
        np.asarray(executor.arm.get_data()["qpos"][:adof], dtype=np.float32),
        np.asarray(result.grasp_pose, dtype=np.float32),
    ])
    t0 = time.time()
    traj_repose = planner.plan_pose_constrained(
        start_full, T_wrist_target,
        hold_vec_weight=[0, 0, 0, 0, 0, 1],     # hold z only
        scene_cfg=scene_cfg, include_obj_obstacle=False,
    )
    timing["m_carry_plan_s"] = round(time.time() - t0, 2)
    if traj_repose is None:
        # The re-plan starts from the arm's ACTUAL config (tracking error), so
        # it can fail where the dry run succeeded. The dry-run trajectory is
        # still collision-checked for this scene -- use it rather than dropping
        # the object where it stands.
        print(f"    carry re-plan failed — using the dry-run trajectory")
        traj_repose = carry_traj
        timing["carry_source"] = "dryrun"
    if traj_repose is not None:
        _t = time.time()
        executor._move_joints(traj_repose[:, :adof],
                              np.tile(s_hand, (len(traj_repose), 1)))
        timing["m_carry_exec_s"] = round(time.time() - _t, 2)
        timing["repose"] = "ok"
        print(f"    repose OK")
    else:
        timing["repose"] = "plan_failed"
        print(f"    repose plan FAILED — placing where the object is")

    # The object is carried at the lift height and let go ``--drop_h`` below it
    # -- NOT lowered back onto the table the way run_auto's place does. The
    # target is built here from cuRobo FK and handed over explicitly;
    # use_current_wrist would have place() rebuild it from the measured frame,
    # which is the 107mm-offset one (see _wrist_now).
    T_place = _wrist_now(planner, executor, adof, result.grasp_pose)
    z_carry = float(T_place[2, 3])
    T_place[2, 3] -= args.drop_h
    place_kwargs = {"grasp_wrist": T_place} if args.arm == "franka" else {}
    # Record both frames: the gap is the constant 107mm ee_link offset seen
    # through the current wrist orientation, and place_info["descended"] is
    # measured in the OTHER frame, so it will not equal --drop_h.
    try:
        _q_now = np.concatenate([
            np.asarray(executor.arm.get_data()["qpos"][:adof], dtype=np.float32),
            np.asarray(s_hand, dtype=np.float32)])
        _z_fk = float(_fk_wrist(planner, _q_now)[2, 3])
        _z_meas = float((executor.arm.get_data()["position"]
                         @ executor._link6_to_wrist)[2, 3])
        timing["z_wrist_measured"] = round(_z_meas, 4)
        timing["z_wrist_fk"] = round(_z_fk, 4)
        print(f"    wrist z: measured={_z_meas:.4f}  curobo_fk={_z_fk:.4f}  "
              f"delta={_z_fk - _z_meas:+.4f}")
    except Exception as _fe:
        print(f"    [fk check] {_fe!r}")
    _t = time.time()
    place_info = executor.place(result, planner=planner, scene_cfg=scene_cfg,
                                lift_height=args.drop_h, **place_kwargs)
    timing["m_place_s"] = round(time.time() - _t, 2)
    timing["place"] = place_info
    print(f"    place: {place_info}")
    if args.arm != "franka":
        _safe("release", executor.release, result)

    # Straight up and OUT before anything else moves. Retracting from where
    # place() left the wrist (reset() plans a joint-space path home) sweeps the
    # hand through the object and the setup around it -- that is what pushed the
    # rig off the table on the very first trial. Climb --retreat_h from HERE,
    # and hold the fingers where release left them: opening them at this height
    # extends them downward and scrapes the table.
    _t = time.time()
    T_up = _wrist_now(planner, executor, adof,
                      getattr(executor, "_last_hand_qpos", result.pregrasp_pose)).copy()
    z_up = float(T_up[2, 3]) + args.retreat_h
    T_up[2, 3] = z_up
    # Hold the shape place()'s release left the fingers in (PREGRASP) all the
    # way up -- do NOT open them further next to the object.
    hand_now = np.asarray(getattr(executor, "_last_hand_qpos", result.pregrasp_pose),
                          dtype=np.float32)                       # RADIANS (planner units)
    up_traj = planner.plan_pose_constrained(
        np.concatenate([np.asarray(executor.arm.get_data()["qpos"][:adof],
                                    dtype=np.float32), hand_now]),
        T_up, hold_vec_weight=[1, 1, 1, 1, 1, 0],
        scene_cfg=scene_cfg, include_obj_obstacle=False)
    if up_traj is not None:
        # _move_joints -> _follow expects the hand columns in CONTROLLER UNITS
        # (0-1000), which is what execute() returns as s_hand. Handing it the
        # planner's radians (0.3-1.2) clips to ~0 = fingers slammed shut on the
        # object we just put down.
        hand_cmd = np.asarray(executor._convert(hand_now.astype(np.float64)),
                              dtype=np.float64)
        executor._move_joints(up_traj[:, :adof],
                              np.tile(hand_cmd, (len(up_traj), 1)))
        timing["retreat_up"] = "ok"
        print(f"    retreat up {args.retreat_h*100:.0f}cm "
              f"({T_wrist_now[2, 3]:.3f} -> {z_up:.3f}) OK")
    else:
        timing["retreat_up"] = "plan_failed"
        print(f"    retreat-up plan FAILED — reset() will handle it")
    timing["m_retreat_s"] = round(time.time() - _t, 2)

    # ORDER MATTERS: rcc FIRST -- the cameras need trigger pulses still running
    # to flush their buffers during stop(), so killing the sync generator before
    # rcc.stop() leaves it waiting on frames that can never arrive (untimed
    # Event.wait, not even Ctrl-C interruptible). Same order as run_auto's
    # _cleanup(). The timeouts are the second line of defence, not the fix.
    if not args.no_video:
        for fn, nm in ((rcc.stop, "rcc.stop"),
                       (lambda: _safe_timestamp_stop(timestamp_monitor),
                        "timestamp_monitor.stop"),
                       (sync_generator.stop, "sync_generator.stop"),
                       (executor.stop_recording, "executor.stop_recording")):
            _stop_with_timeout(nm, fn)
    else:
        _stop_with_timeout("rcc.stop", rcc.stop)

    # ── 6. label (human) + retract ───────────────────────────────────────────
    label_rel = os.path.join("shared_data", "AutoDex", "experiment",
                             args.exp_name, f"{args.arm}_{hand}", obj, dir_idx,
                             "label", "raw")
    # Photo of the end state for the record. Never let it block the prompt --
    # if the cameras are wedged we still want the operator to be able to label.
    _stop_with_timeout("rcc.start(image)", lambda: rcc.start("image", False, label_rel),
                       timeout=15.0)
    _stop_with_timeout("rcc.stop(image)", rcc.stop, timeout=15.0)
    print(f"\n[6/6] LABEL — did the {obj} end up ON the marker?")
    print(f"      y  = SUCCESS: robot picked it up and it is sitting on the "
          f"marker/board")
    print(f"      n  = FAIL: never grasped it, dropped it, or it landed off "
          f"the marker")
    print(f"      c  = VOID: not the robot's fault (operator error, hardware "
          f"glitch) — excluded from the rate")
    print(f"      ym / nm = same as y / n but type a memo afterwards")
    _t = time.time()
    try:
        succ, note = get_label()
    except KeyboardInterrupt:
        _safe("release", executor.release, result)
        _safe("stop_recording", executor.stop_recording)
        raise

    timing["m_label_s"] = round(time.time() - _t, 2)

    _t = time.time()
    for fn in (lambda: executor.reset(result, planner, scene_cfg),
               lambda: executor.reset_hybrid(result, planner, scene_cfg),
               lambda: executor.reset_fallback(result)):
        try:
            timing["retract"] = fn()
            break
        except Exception as e:
            print(f"    retract step failed: {e!r}")
    timing["m_retract_s"] = round(time.time() - _t, 2)

    if s_hand is not None:
        np.save(os.path.join(img_dir, "squeeze_hand.npy"), s_hand)
    _stop_with_timeout("rcc stream restart",
                       lambda: _rcc_start(rcc, "stream", False, fps=args.stream_fps))

    d = {
        "dir_idx": dir_idx, "arm": args.arm, "hand": hand, "obj": obj,
        "success": succ,
        "scene_info": result.scene_info,
        "tabletop": pose_stem,
        "grid_idx": args.loc, "ori_idx": args.ori,
        "obj_xy": [round(float(pose_robot[0, 3]), 4),
                    round(float(pose_robot[1, 3]), 4)],
        "obj_yaw_deg": round(float(np.rad2deg(np.arctan2(
            pose_robot[1, 0], pose_robot[0, 0]))), 1),
        "target_xyz": np.asarray(target_xyz).tolist(),
        "place_yaw_deg": place_yaw,
        "drop_h": args.drop_h,
        "obj_pose_robot": pose_robot.tolist(),
    }
    if note:
        d["note"] = note
    status = "SUCCESS" if succ else ("ISSUE" if succ is None else "FAIL")
    print(f"    Result: {status}  saved to {img_dir}/result.json")
    return _finish(d)


def load_trials(run_dir: str) -> List[dict]:
    """Every ``result.json`` already written under ``run_dir``, oldest first."""
    out = []
    for d in sorted(Path(run_dir).iterdir()):
        rp = d / "result.json"
        if not d.is_dir() or not rp.exists():
            continue
        try:
            rec = json.loads(rp.read_text())
        except Exception:
            continue
        rec.setdefault("dir_idx", d.name)
        out.append(rec)
    return out


def tally(recs: List[dict]):
    """(successes, judged, total). Aborted trials (no human label) are total-only."""
    judged = [r for r in recs if r.get("success") is not None]
    return sum(1 for r in judged if r["success"]), len(judged), len(recs)


def _cell(rec: dict):
    """(location, orientation) key a trial is reported under.

    Explicit ``--loc`` / ``--ori`` tags win (that is what the OpenArm protocol
    grid gives us); otherwise fall back to the measured object position rounded
    to a 5 cm cell and the detected tabletop pose stem.
    """
    loc = rec.get("grid_idx", rec.get("loc"))
    if loc is None or loc == "":
        xy = rec.get("obj_xy")
        loc = (f"auto:x{0.05 * round(xy[0] / 0.05):.2f}"
               f"_y{0.05 * round(xy[1] / 0.05):+.2f}" if xy else "?")
    ori = rec.get("ori_idx", rec.get("ori")) or rec.get("tabletop") or "?"
    return f"grid{loc}" if str(loc).isdigit() else str(loc), str(ori)


def report(recs: List[dict]) -> dict:
    """Print + return the success rate overall and per (location, orientation)."""
    n_succ, n_judged, n_all = tally(recs)
    rate = (n_succ / n_judged) if n_judged else 0.0
    print(f"\n{'='*64}")
    print(f"SUCCESS RATE: {n_succ}/{n_judged} = {100*rate:.1f}%   "
          f"({n_all} trials, {n_all - n_judged} unjudged/aborted)")

    cells: dict = {}
    for r in recs:
        cells.setdefault(_cell(r), []).append(r)
    print(f"\n  per location x orientation")
    print(f"  {'grid/location':<18} {'ori':<6} {'succ':>7}  {'rate':>6}  "
          f"{'pipeline_s':>10}")
    per_cell = {}
    for key in sorted(cells):
        rs = cells[key]
        s_, j_, a_ = tally(rs)
        pipes = [r["pipeline_s"] for r in rs if r.get("pipeline_s")]
        mean_p = (sum(pipes) / len(pipes)) if pipes else float("nan")
        print(f"  {key[0]:<18} {key[1]:<6} {s_:>3}/{j_:<3} "
              f"{(100*s_/j_ if j_ else 0):>5.1f}%  {mean_p:>10.1f}")
        per_cell[f"{key[0]}|{key[1]}"] = {
            "location": key[0], "orientation": key[1],
            "n_trials": a_, "n_judged": j_, "n_success": s_,
            "success_rate": (s_ / j_) if j_ else None,
            "mean_pipeline_s": mean_p,
        }

    # Module timings: mean over trials that got that far.
    mods: dict = {}
    for r in recs:
        for k, v in (r.get("modules_s") or {}).items():
            mods.setdefault(k, []).append(v)
    if mods:
        print(f"\n  mean module time (s), n = trials that reached the module")
        order = ["perception", "place_filter", "grasp_plan", "preflight",
                 "grasp_exec", "carry_plan", "carry_exec", "place", "retract",
                 "label"]
        keys = [k for k in order if k in mods] + \
               [k for k in mods if k not in order]
        for k in keys:
            v = mods[k]
            print(f"    {k:<14} {sum(v)/len(v):>7.2f}  (n={len(v)})")
    per_mod = {k: {"mean_s": sum(v) / len(v), "n": len(v)}
               for k, v in mods.items()}
    pipes = [r["pipeline_s"] for r in recs if r.get("pipeline_s")]
    if pipes:
        print(f"\n  full pipeline (excl. human labelling): "
              f"mean {sum(pipes)/len(pipes):.1f}s over {len(pipes)} trials")
    return {"n_trials": n_all, "n_judged": n_judged, "n_success": n_succ,
            "success_rate": rate, "per_cell": per_cell, "modules_s": per_mod,
            "mean_pipeline_s": (sum(pipes) / len(pipes)) if pipes else None}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obj", default="banana")
    p.add_argument("--hand", default="inspire", choices=["allegro", "inspire",
                                                          "inspire_left"])
    p.add_argument("--arm", default="franka", choices=["xarm", "franka"])
    p.add_argument("--grasp_version", default="v8")
    p.add_argument("--exp_name", default="banana_demo")
    p.add_argument("--loc", "--grid", dest="loc", default=None,
                   help="starting GRID INDEX of the object's location in the "
                        "comparison protocol's layout. Re-asked before every "
                        "trial; stored as grid_idx and used to group the "
                        "success-rate report")
    p.add_argument("--ori", default=None,
                   help="starting orientation index. Asked with the grid idx; "
                        "falls back to the detected tabletop stem")
    p.add_argument("--allow_other_pose", action="store_true",
                   help="if no success grasp exists at the CURRENT tabletop "
                        "pose, fall back to successes at any tabletop pose")
    # place target
    p.add_argument("--target_capture_dir", default=DEFAULT_TARGET_CAPTURE,
                   help="image set to triangulate the marker from; pass '' to "
                        "capture a fresh one")
    p.add_argument("--marker_id", type=int, default=None,
                   help="aruco id of the target marker (default: the only "
                        "non-charuco-board id in view)")
    p.add_argument("--marker_dict", default="6X6_1000")
    p.add_argument("--target_yaw_deg", type=float, default=None,
                   help="force this world-z yaw for the drop (default: pick "
                        "the IK-feasible yaw closest to as-picked -- the demo "
                        "does not care how the object lands)")
    p.add_argument("--retreat_h", type=float, default=0.15,
                   help="how far straight up the wrist climbs after releasing, "
                        "before the retract home (m)")
    p.add_argument("--drop_h", type=float, default=0.03,
                   help="descend this far below the carry height before "
                        "releasing (m); the object is never lowered back to "
                        "the table")
    p.add_argument("--yaw_step", type=int, default=10,
                   help="yaw sweep resolution (deg) for the auto drop yaw")
    # perception / cameras
    p.add_argument("--pc_list", nargs="+", default=DEFAULT_PC_LIST)
    p.add_argument("--port_mask", type=int, default=5006)
    p.add_argument("--port_pose", type=int, default=5007)
    p.add_argument("--port_cmd", type=int, default=6893)
    p.add_argument("--prompt", default="banana")
    p.add_argument("--sil_iters", type=int, default=100)
    p.add_argument("--sil_lr", type=float, default=0.002)
    p.add_argument("--init_timeout_s", type=float, default=60.0)
    p.add_argument("--stream_fps", type=int, default=10)
    p.add_argument("--stream_warmup_s", type=float, default=2.0)
    p.add_argument("--calib_dir", default=None)
    p.add_argument("--no_video", action="store_true",
                   help="skip AVI recording (no sync generator / timestamps)")
    args = p.parse_args()

    planner_robot = _planner_robot(args.arm, args.hand)
    mesh_path = MESH_BASE / args.obj / "raw_mesh" / f"{args.obj}.obj"
    assets_root = ASSETS_BASE / args.obj
    if not mesh_path.exists():
        sys.exit(f"mesh not found: {mesh_path}")
    if not (assets_root / "object_repre/v1" / args.obj / "1/repre.pth").exists():
        sys.exit(f"repre.pth missing for {args.obj}")

    calib_dir = (Path(args.calib_dir).expanduser() if args.calib_dir
                 else sorted(CAM_PARAM_ROOT.iterdir())[-1])
    print(f"calib: {calib_dir.name}")
    intrinsics_full, extrinsics_full, H, W = _load_calib(calib_dir)
    pc_ips = [get_pc_ip(pc) for pc in args.pc_list]
    pc_serials = {pc: get_camera_list(pc) for pc in args.pc_list}
    active = {s for pc in args.pc_list for s in pc_serials[pc]}
    intrinsics_full = {s: v for s, v in intrinsics_full.items() if s in active}
    extrinsics_full = {s: v for s, v in extrinsics_full.items() if s in active}
    print(f"  {len(intrinsics_full)} cams active ({H}x{W})")

    run_dir = os.path.join(project_dir, "experiment", args.exp_name,
                           f"{args.arm}_{args.hand}", args.obj)
    os.makedirs(run_dir, exist_ok=True)

    rcc = remote_camera_controller("banana_demo", pc_list=args.pc_list,
                                   stall_timeout=15.0)
    # A crashed run leaves the daemon lock behind and a bad start LATCHES a
    # camera error; either one makes every later arm/set_record a silent no-op,
    # so trials record a couple of frames and stop. run_auto clears both before
    # it touches the cameras -- so must this.
    _ensure_camera_lock(rcc)
    _clear_camera_errors(rcc)
    sync_generator = UTGE900(**network_info["signal_generator"]["param"])
    timestamp_monitor = TimestampMonitor(**network_info["timestamp"]["param"])

    # ── place target (marker) ────────────────────────────────────────────────
    def _locate_target() -> dict:
        cap = (os.path.expanduser(args.target_capture_dir)
               if args.target_capture_dir else capture_images(pc_list=args.pc_list))
        info = locate_marker(cap, dict_type=args.marker_dict,
                             marker_id=args.marker_id)
        print(f"[target] {args.marker_dict} id={info['marker_id']} "
              f"({info['n_views']} views)  center_robot="
              f"{info['center_robot'].round(4)}")
        with open(os.path.join(run_dir, "place_target.json"), "w") as f:
            json.dump({k: (v.tolist() if isinstance(v, np.ndarray) else v)
                       for k, v in info.items()}, f, indent=1)
        return info

    target_info = _locate_target()

    print(f"[stream] start @ {args.stream_fps} FPS...")
    _rcc_start(rcc, "stream", False, fps=args.stream_fps)
    time.sleep(args.stream_warmup_s)
    _warn_if_not_streaming(rcc)

    print(f"[orch] init for {args.obj}...")
    orch = InitOrchestrator(
        pc_list=args.pc_list, capture_ips=pc_ips,
        port_mask=args.port_mask, port_pose=args.port_pose,
        port_cmd=args.port_cmd,
    )
    orch.init_object(
        obj_name=args.obj, mesh_path=str(mesh_path),
        assets_root=str(assets_root),
        intrinsics_full=intrinsics_full, extrinsics_full=extrinsics_full,
        image_hw=(H, W), mode="live", pc_serials=pc_serials,
    )

    print(f"[planner] warmup ({planner_robot})...")
    planner = GraspPlanner(hand=planner_robot)
    quiet_curobo()
    print(f"[executor] connect ({args.arm})...")
    if args.arm == "franka":
        from src.execution.franka_executor import FrankaExecutor
        executor = FrankaExecutor(hand_name=args.hand)
        executor.set_speed_profile_planner(planner)
        executor.home(clear_view=True)
    else:
        from autodex.executor.real import RealExecutor
        executor = RealExecutor(hand_name=args.hand)

    n_new = 0
    try:
        while True:
            # Tags are asked BEFORE the trial, not derived from the object pose:
            # the operator knows which protocol cell they just put the object
            # in, and nobody can eyeball the object's center in robot frame.
            cur = (f"[grid {args.loc}"
                   + (f", ori {args.ori}]" if args.ori is not None else "]")
                   ) if args.loc is not None else "[unset]"
            cmd = input(
                f"\nPlace the {args.obj} for the next trial.\n"
                f"  grid idx {cur} — Enter to keep, "
                f"'<grid> [ori]' to set, q to stop: ").strip()
            if cmd.lower() == "q":
                break
            if cmd:
                parts = cmd.split()
                args.loc = parts[0]
                args.ori = parts[1] if len(parts) > 1 else args.ori
            if args.loc is None:
                print("    (no grid idx given — this trial groups under the "
                      "measured object xy instead)")
            print(f"\n{'#'*60}\n# Trial {n_new + 1} "
                  f"(grid={args.loc if args.loc is not None else 'auto'}, "
                  f"ori={args.ori if args.ori is not None else 'auto'})"
                  f"\n{'#'*60}")
            chime.info()
            tr = run_trial(args, orch=orch, planner=planner, executor=executor,
                           rcc=rcc, sync_generator=sync_generator,
                           timestamp_monitor=timestamp_monitor,
                           target_xyz=np.asarray(target_info["center_robot"]),
                           run_dir=run_dir)
            n_new += 1
            status = ("SUCCESS" if tr.get("success") else
                      ("ISSUE" if tr.get("success") is None
                       else tr.get("reason", "FAIL")))
            print(f"\nTRIAL {tr.get('dir_idx')}: {status}")
            n_s, n_j, _ = tally(load_trials(run_dir))
            print(f"    running rate: {n_s}/{n_j} = "
                  f"{(100*n_s/n_j if n_j else 0):.1f}%")
            if tr.get("quit"):
                break
    except KeyboardInterrupt:
        print("\n[interrupted]")
    finally:
        recs = load_trials(run_dir)
        summary = report(recs)
        summary.update({"obj": args.obj, "arm": args.arm, "hand": args.hand,
                        "grasp_version": args.grasp_version,
                        "n_trials_this_session": n_new,
                        "target": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                                   for k, v in target_info.items()},
                        "trials": recs})
        stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = os.path.join(run_dir, f"summary_{stamp}.json")
        with open(out, "w") as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n  saved: {out}")

        _safe("executor.shutdown", executor.shutdown)
        _safe("orch.close", orch.close)
        for fn, nm in ((rcc.stop, "rcc.stop"),
                       (timestamp_monitor.end, "timestamp_monitor.end"),
                       (sync_generator.end, "sync_generator.end"),
                       (rcc.end, "rcc.end")):
            _stop_with_timeout(nm, fn)


if __name__ == "__main__":
    main()
