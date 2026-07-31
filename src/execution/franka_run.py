"""Minimal franka(fr3)+inspire grasp runner — no reorient, no coverage.

Per trial: FoundPose init (InitOrchestrator, live) -> pose_world -> scene_cfg
-> plan (fr3_inspire on v8 candidates) -> FrankaExecutor.execute (grasp+lift)
-> charuco lift-snapshot success -> place back -> save.

This is the stripped franka counterpart of ``src/execution/run_auto.py`` (which
adds coverage-driven reorient, cylinder snapping, success-only modes, etc). Here
we just grasp whatever the planner selects for the object at its current pose.

Prereqs:
  * franka daemon up:   ./cpp/franka_daemon/run_daemon.sh   (on the franka PC)
  * init daemons up:    bash scripts/init_daemons.sh start   (capture PCs)
  * cameras + calib present.

    python src/execution/franka_run.py --obj servingbowl_small --n_trials 1
    (or: python -m src.execution.franka_run --obj servingbowl_small)
"""
import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

# Allow direct execution (`python src/execution/franka_run.py`) without -m by
# putting the repo root on sys.path so the absolute `src.` / `autodex.` imports
# resolve.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.io.camera_system.signal_generator import UTGE900
from paradex.io.camera_system.timestamp_monitor import TimestampMonitor
from paradex.utils.system import network_info, get_pc_ip, get_camera_list
from paradex.calibration.utils import save_current_camparam, save_current_C2R, load_c2r

from autodex.perception.init_orchestrator import InitOrchestrator
from autodex.planner import GraspPlanner
from autodex.utils.path import project_dir

from src.execution.scene_cfg import pose_world_to_scene_cfg
from src.execution.label import auto_label_charuco
from src.execution.franka_executor import FrankaExecutor, ContactAbort
# reuse run_auto's calib loader + asset roots (module import has no hardware side effects)
from src.execution.run_auto import (
    _load_calib, MESH_BASE, ASSETS_BASE, CAM_PARAM_ROOT, DEFAULT_PC_LIST,
)

CHARUCO_BOARD = "1"
_WATCHDOG = None            # external process that hard-kills us if we hang


def _wait_for_object_on_table(rcc, args, trial_idx: int,
                              required_board: str = CHARUCO_BOARD) -> bool:
    """Block until charuco ``required_board`` is NOT fully visible (= the object
    covers it). Pre-flight check before each trial, ported from run_auto.py.

    Returns True to proceed, False if the user pressed 'q'.
    """
    attempt = 0
    while True:
        attempt += 1
        stem = os.path.join("_precheck", f"trial{trial_idx:03d}_{attempt:02d}", "raw")
        check_rel = os.path.join("shared_data", "AutoDex", "experiment",
                                 args.exp_name, args.hand, args.obj, stem)
        check_abs = os.path.join(project_dir, "experiment", args.exp_name,
                                 args.hand, args.obj, stem, "images")
        try: rcc.stop()
        except Exception: pass
        rcc.start("image", False, check_rel)
        rcc.stop()
        time.sleep(0.3)
        board_visible, info = auto_label_charuco(check_abs, required_board=required_board)

        def _resume_stream():
            # new capture-PC API: acquire continuously + enable the SHM sink
            rcc.start("acquire", False, fps=args.stream_fps)
            rcc.set_sink(stream=True)
            if args.stream_warmup_s > 0:
                time.sleep(args.stream_warmup_s)

        # visible=True  -> board fully detected  = nothing on the table -> prompt
        # visible=False -> board partly hidden   = object present       -> start
        # visible=None  -> no images / board not configured             -> start
        if board_visible is None:
            print(f"[precheck] {info.get('reason', 'unknown')} — proceeding without check")
            _resume_stream()
            return True
        if not board_visible:
            print(f"[precheck] obj on table (board covered "
                  f"{info.get('covered')}/{info.get('expected')}). Starting trial.")
            _resume_stream()
            return True
        print(f"[precheck] charuco fully visible "
              f"({info.get('covered')}/{info.get('expected')}) — no obj on table.")
        try:
            cmd = input("    Place obj then press Enter (q to quit): ").strip().lower()
        except KeyboardInterrupt:
            return False
        if cmd == "q":
            return False


def run_trial(args, orch, planner, executor, rcc, sync_generator,
              timestamp_monitor, trial_idx):
    obj, hand = args.obj, args.hand
    dir_idx = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    img_dir = os.path.join(project_dir, "experiment", args.exp_name, hand, obj, dir_idx)
    os.makedirs(img_dir, exist_ok=True)
    timing = {"trial_start": datetime.datetime.now().isoformat()}
    print(f"\n{'='*60}\n[trial {trial_idx}] {obj} -> {dir_idx}")

    # 1. calib snapshot
    save_current_C2R(img_dir)
    save_current_camparam(img_dir)

    # 2. FoundPose init -> pose_world
    print("[init] FoundPose distributed...")
    t0 = time.time()
    pose_world, perc_timing = orch.trigger_init(
        prompt=args.prompt,
        save_capture_dir=os.path.join(img_dir, "init_capture"),
        sil_iters=args.sil_iters, sil_lr=args.sil_lr, timeout_s=args.init_timeout_s,
    )
    timing["perception_s"] = round(time.time() - t0, 2)
    if pose_world is None:
        reason = (perc_timing or {}).get("reason", "perception_failed")
        print(f"[init] FAILED ({reason})")
        return _save(img_dir, dict(dir_idx=dir_idx, success=False, reason=reason, timing=timing))
    np.save(os.path.join(img_dir, "pose_world.npy"), pose_world)

    # 3. scene_cfg + plan
    c2r = load_c2r(img_dir)
    scene_cfg = pose_world_to_scene_cfg(pose_world, c2r, obj)
    t0 = time.time()
    result = planner.plan(scene_cfg, obj, args.grasp_version, hand=hand)
    timing["plan_s"] = round(time.time() - t0, 2)
    print(f"[plan] success={result.success}  ({timing['plan_s']}s)")
    if not result.success:
        return _save(img_dir, dict(dir_idx=dir_idx, success=False,
                                   reason="plan_failed", timing=timing))

    # 4. execute (record window)
    raw_dir = os.path.join(img_dir, "raw")
    if timestamp_monitor is not None:
        timestamp_monitor.start(os.path.join(raw_dir, "timestamps"))
    executor.start_recording(raw_dir)
    if not args.no_video:
        # Video sink ON for the execute window. set_sink toggles it live on the
        # running acquire capture, so the SHM stream sink stays up (no
        # stop/start dance like run_auto's rcc.start("video", ...)).
        # save_path is CAPTURE-PC-relative; run_auto's convention writes the
        # .avi to each capture PC's local disk under ~/AutoDex/...
        exec_rel = os.path.join("AutoDex", "experiment", args.exp_name, hand,
                                obj, dir_idx, "raw", "exec")
        print(f"[video] recording to <capture PC>:~/{exec_rel}")
        rcc.set_record(exec_rel, on=True)
        sync_generator.start(fps=30)
    aborted = False
    try:
        executor.execute(result, planner=planner, scene_cfg=scene_cfg,
                         lift_height=args.lift_height)
    except ContactAbort as e:
        print(f"[execute] {e} — aborting trial")
        aborted = True
    # ORDER MATTERS: stop the timestamp monitor BEFORE killing the trigger.
    # TimestampMonitor.stop() waits on event["stop"], which its capture loop only
    # sets after `camera.get_timestamp()` returns — and that call blocks for the
    # next frame. Stop the signal generator first and no frame ever arrives, so
    # stop() waits forever (this is what hung the trial before _save). _timed
    # bounds it either way.
    if timestamp_monitor is not None:
        _timed("timestamp_monitor.stop", timestamp_monitor.stop, timeout=10)
    if not args.no_video:
        try: sync_generator.stop()
        except Exception: pass
        try: rcc.set_record(on=False)          # video sink off, stream stays on
        except Exception as e: print(f"[video] set_record(off) failed: {e!r}")

    if aborted:
        pass  # stop_recording moved to _shutdown (bounded by short watchdog)
        _timed("reset", lambda: executor.reset(result, planner, scene_cfg), timeout=40)
        return _save(img_dir, dict(dir_idx=dir_idx, success=False,
                                   reason="contact_abort", scene_info=result.scene_info,
                                   timing=timing))

    # 5. lift-time charuco snapshot: object up -> board reappears -> success
    success = None
    if not args.no_charuco:
        try: rcc.stop()
        except Exception: pass
        label_rel = os.path.join("shared_data", "AutoDex", "experiment",
                                 args.exp_name, hand, obj, dir_idx, "label_at_lift", "raw")
        label_abs = os.path.join(img_dir, "label_at_lift", "raw", "images")
        rcc.start("image", False, label_rel)
        rcc.stop()
        time.sleep(0.3)
        success, info = auto_label_charuco(label_abs, required_board=CHARUCO_BOARD)
        print(f"[label] success={success}  covered {info.get('covered')}/{info.get('expected')}")

    # 6. place back + reset (retract to clear-view avoiding the placed object)
    executor.place(planner=planner, scene_cfg=scene_cfg,
                   grasp_wrist=result.wrist_se3, hand_qpos=result.grasp_pose,
                   pregrasp_qpos=result.pregrasp_pose)
    executor.reset(result, planner, scene_cfg)
    pass  # stop_recording moved to _shutdown (bounded by short watchdog)
    # timestamp_monitor was already stopped right after execute, while the
    # trigger was still running (see the ordering note above).

    return _save(img_dir, dict(dir_idx=dir_idx, success=bool(success) if success is not None else None,
                               scene_info=result.scene_info, timing=timing,
                               candidate_idx=result.timing.get("candidate_idx")))


def _timed(label, fn, timeout=8.0):
    """Run fn in a daemon thread with a hard timeout so a blocking stop()/end()
    can NEVER hang the trial (worst case: abandoned after timeout)."""
    import threading
    t = threading.Thread(target=lambda: _swallow(label, fn), daemon=True)
    t.start()
    t.join(timeout)
    if t.is_alive():
        print(f"  {label}: TIMEOUT after {timeout}s — abandoning")


def _swallow(label, fn):
    try:
        fn()
    except Exception as e:
        print(f"  {label}: {e!r}")


def _save(img_dir, result_dict):
    result_dict["trial_end"] = datetime.datetime.now().isoformat()
    with open(os.path.join(img_dir, "result.json"), "w") as f:
        json.dump(result_dict, f, indent=2, default=str)
    print(f"[trial] result: success={result_dict.get('success')} "
          f"reason={result_dict.get('reason', '-')}")
    return result_dict


def main():
    # Ctrl-C / kill must ALWAYS terminate immediately — bypass every cleanup path
    # (which is where it hangs). os._exit does not run finally/atexit/thread joins.
    import signal
    def _force_exit(signum, _frame):
        print(f"\n[signal {signum}] force exit", flush=True)
        os._exit(130)
    for _sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(_sig, _force_exit)
        except Exception:
            pass

    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", required=True)
    ap.add_argument("--hand", default="inspire", choices=["inspire", "inspire_left"])
    ap.add_argument("--grasp_version", default="v8")
    ap.add_argument("--exp_name", default=None, help="default: grasp_version")
    ap.add_argument("--n_trials", type=int, default=1)
    ap.add_argument("--lift_height", type=float, default=0.10)
    ap.add_argument("--no_charuco", action="store_true",
                    help="skip the charuco lift snapshot (success stays None)")
    ap.add_argument("--no_precheck", action="store_true",
                    help="skip the pre-trial charuco check (start even with no object on the table)")
    ap.add_argument("--no_video", action="store_true", help="skip sync_generator video recording")
    ap.add_argument("--prompt", type=str, default="object on the checkerboard")
    ap.add_argument("--sil_iters", type=int, default=100)
    ap.add_argument("--sil_lr", type=float, default=0.002)
    ap.add_argument("--init_timeout_s", type=float, default=120.0)
    ap.add_argument("--calib_dir", type=str, default=None)
    ap.add_argument("--pc_list", nargs="+", default=DEFAULT_PC_LIST)
    ap.add_argument("--port_mask", type=int, default=5006)
    ap.add_argument("--port_pose", type=int, default=5007)
    ap.add_argument("--port_cmd", type=int, default=6893)
    ap.add_argument("--stream_fps", type=int, default=10)
    ap.add_argument("--stream_warmup_s", type=float, default=2.0)
    ap.add_argument("--watchdog_s", type=float, default=None,
                    help="hard SIGKILL budget (default: 420 + 300*n_trials)")
    args = ap.parse_args()

    # BULLETPROOF termination: a separate watchdog PROCESS (immune to GIL / hung
    # threads / blocked C calls in this process) SIGKILLs us if we ever hang past
    # the budget. It only fires while we are still alive (a real hang) — normal
    # completion kills the watchdog in _shutdown before os._exit, so there is no
    # pid-reuse risk. This is what GUARANTEES the program ends.
    global _WATCHDOG
    import subprocess as _sp
    # Budget must cover ONE-TIME setup (init_object dispatch, cuRobo MotionGen
    # warmup, franka connect) plus each trial (perception + plan + execute +
    # lift/place/reset, each of which runs its own cuRobo plan). The old
    # 90 + 120*n killed healthy runs mid-plan — this is a HANG backstop, so it
    # should sit well above the normal worst case.
    _wd_secs = args.watchdog_s if args.watchdog_s else 420 + args.n_trials * 300
    # start_new_session=True puts the watchdog in ITS OWN session so a terminal
    # Ctrl-C (which goes to the foreground group) does NOT kill it — it survives
    # and SIGKILLs us even while we're stuck inside a cuRobo C++/CUDA call (where
    # the Python SIGINT handler cannot run because the C call holds the GIL).
    _WATCHDOG = _sp.Popen(
        [sys.executable, "-c",
         f"import time,os,signal\ntime.sleep({_wd_secs})\n"
         f"try:\n os.kill({os.getpid()}, signal.SIGKILL)\nexcept Exception: pass"],
        start_new_session=True)
    print(f"[watchdog] armed (pid {_WATCHDOG.pid}): hard SIGKILL of THIS pid "
          f"{os.getpid()} in {_wd_secs}s. Manual kill: kill -9 {os.getpid()}", flush=True)

    if args.exp_name is None:
        args.exp_name = args.grasp_version

    # asset sanity
    mesh_path = MESH_BASE / args.obj / "raw_mesh" / f"{args.obj}.obj"
    assets_root = ASSETS_BASE / args.obj
    if not mesh_path.exists():
        raise SystemExit(f"mesh not found: {mesh_path}")

    calib_dir = Path(args.calib_dir).expanduser() if args.calib_dir \
        else sorted(CAM_PARAM_ROOT.iterdir())[-1]
    print(f"calib: {calib_dir.name}")
    intrinsics_full, extrinsics_full, H, W = _load_calib(calib_dir)

    pc_ips = [get_pc_ip(p) for p in args.pc_list]
    pc_serials = {p: get_camera_list(p) for p in args.pc_list}
    active = {s for pc in args.pc_list for s in pc_serials[pc]}
    intrinsics_full = {s: v for s, v in intrinsics_full.items() if s in active}
    extrinsics_full = {s: v for s, v in extrinsics_full.items() if s in active}

    # All hardware handles start as None so cleanup works no matter WHERE setup
    # dies. sync_generator + timestamp_monitor are only for synchronized video;
    # skipped under --no_video (the local PySpin timestamp camera may be on a
    # wrong subnet and is not needed to grasp).
    rcc = sync_generator = timestamp_monitor = executor = None
    try:
        rcc = remote_camera_controller("franka_run", pc_list=args.pc_list)
        if not args.no_video:
            sync_generator = UTGE900(**network_info["signal_generator"]["param"])
            timestamp_monitor = TimestampMonitor(**network_info["timestamp"]["param"])
        print(f"[stream] {len(args.pc_list)} PCs @ {args.stream_fps} FPS")
        # New capture-PC API: acquire continuously, then enable the shared-memory
        # stream sink ('stream' mode was removed). The init pipeline reads SHM.
        rcc.start("acquire", False, fps=args.stream_fps)
        rcc.set_sink(stream=True)
        if args.stream_warmup_s > 0:
            time.sleep(args.stream_warmup_s)

        orch = InitOrchestrator(
            pc_list=args.pc_list, capture_ips=pc_ips,
            port_mask=args.port_mask, port_pose=args.port_pose, port_cmd=args.port_cmd)
        orch.init_object(
            obj_name=args.obj, mesh_path=str(mesh_path), assets_root=str(assets_root),
            intrinsics_full=intrinsics_full, extrinsics_full=extrinsics_full,
            image_hw=(H, W), mode="live", pc_serials=pc_serials)

        print("[planner] warming up (fr3_inspire)...")
        planner = GraspPlanner(hand="fr3_inspire")
        print("[executor] connecting to franka...")
        executor = FrankaExecutor(hand_name=args.hand)
        # Move to clear-view (arm out of the cameras' view of the object) BEFORE
        # the first perception so the arm never occludes the object.
        print("[executor] homing to clear-view...")
        executor.home(clear_view=True)

        for i in range(args.n_trials):
            if not args.no_precheck:
                if not _wait_for_object_on_table(rcc, args, i):
                    print("[precheck] quit requested — stopping")
                    break
            run_trial(args, orch, planner, executor, rcc,
                      sync_generator, timestamp_monitor, i)
    except KeyboardInterrupt:
        print("\n[interrupt] Ctrl-C — shutting down cleanly")
    except Exception as e:
        import traceback
        print(f"\n[error] {e!r} — shutting down cleanly")
        traceback.print_exc()
    finally:
        _shutdown(rcc, executor, sync_generator, timestamp_monitor)


def _shutdown(rcc, executor, sync_generator, timestamp_monitor):
    """Always-safe teardown: works with any subset of handles already created,
    in an order that leaves no camera streaming and no recording open. Every
    call is run in a daemon thread with a hard timeout so a blocking stop()
    (e.g. a hung camera thread) can NEVER hang the shutdown — worst case it is
    abandoned after `timeout` and we move on."""
    import threading
    import signal
    # HARD backstop: a thread-based timeout is defeated when a cleanup call blocks
    # inside a C extension holding the GIL (no other Python thread can run, so the
    # join never times out — that is why the process hung after OOM). SIGALRM is
    # delivered to the main thread and force-exits regardless.
    try:
        signal.signal(signal.SIGALRM, lambda *a: (print("[cleanup] hard timeout — force exit"), os._exit(0)))
        signal.alarm(25)
    except Exception:
        pass
    # SHORT-FUSE watchdog: the work is DONE, so bound the whole teardown hard.
    # A separate process (own session, immune to Ctrl-C, immune to this process's
    # GIL) SIGKILLs us in 15s no matter what cleanup call blocks. This is what
    # actually guarantees the prompt returns after the arm finishes its cycle.
    _short_wd = None
    try:
        import subprocess as _sp
        _short_wd = _sp.Popen([sys.executable, "-c",
                   f"import time,os,signal\ntime.sleep(15)\n"
                   f"try:\n os.kill({os.getpid()}, signal.SIGKILL)\nexcept Exception: pass"],
                  start_new_session=True)
    except Exception:
        pass

    def _threads():
        return [f"{t.name}{'(d)' if t.daemon else '(NONDAEMON)'}"
                for t in threading.enumerate() if t.is_alive() and t is not threading.current_thread()]

    print(f"[cleanup] stopping hardware... alive threads BEFORE: {_threads()}", flush=True)

    def _try(label, fn, timeout=8.0):
        print(f"  [cleanup] -> {label} ...", flush=True)
        done = {"ok": False}

        def _run():
            try:
                fn()
                done["ok"] = True
            except Exception as e:
                print(f"  [cleanup] {label}: {e!r}", flush=True)
        t = threading.Thread(target=_run, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            print(f"  [cleanup] {label}: TIMEOUT {timeout}s — blocked here, abandoning", flush=True)
        else:
            print(f"  [cleanup] {label}: done ({'ok' if done['ok'] else 'err'})", flush=True)

    if executor is not None:
        _try("stop_recording", executor.stop_recording)
    if sync_generator is not None:
        _try("sync_generator.stop", sync_generator.stop)
    if timestamp_monitor is not None:
        _try("timestamp_monitor.stop", timestamp_monitor.stop)
    if rcc is not None:
        _try("rcc set_sink(stream off)", lambda: rcc.set_sink(stream=False, video=False))
        _try("rcc.stop", rcc.stop)
        _try("rcc.end", rcc.end)
    if executor is not None:
        _try("executor.shutdown", executor.shutdown)
    print("[cleanup] done.")
    # clean exit reached — disarm BOTH watchdogs so neither later kills a reused pid.
    for _wd in (_WATCHDOG, _short_wd):
        try:
            if _wd is not None:
                _wd.kill()
        except Exception:
            pass
    # daemon threads may still be blocked in native code; force process exit so
    # the shell prompt returns instead of hanging on a stuck stop().
    os._exit(0)


if __name__ == "__main__":
    main()
