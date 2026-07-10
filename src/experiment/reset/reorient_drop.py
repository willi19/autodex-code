#!/usr/bin/env python3
"""Reorient + drop policy.

Per-cycle (structure mirrors naive_drop.py — DO NOT diverge):
    perception -> classify tabletop pose (before) -> plan
    -> execute (grasp + lift 25cm)
    -> charuco check (is the object actually lifted?)
        * fail  -> reset_fallback (open hand, sequential retract)  -> next cycle
        * pass  -> reorient wrist mid-air so the held object matches the
                   user-specified target tabletop rotation (z-aligned to the
                   current object yaw so motion is minimal)
                -> cartesian descent so the held object hovers ~15cm release
                   height above the table (object mesh-bottom must stay above
                   TABLE_SURFACE_Z — the ~10cm descent is bounded so it never
                   crashes the object into the table even if mesh is tall)
                -> release (reverse squeeze -> grasp -> pregrasp)
                -> reset_fallback (open hand -> sequential retract)
                -> post-drop perception -> classify tabletop pose (after)

The target tabletop pose is selected ONCE at script start via
``--target_tabletop`` (filename stem, e.g. ``002`` for ``002.npy``) and
shared across all cycles.

Prerequisites:
    bash scripts/init_daemons.sh start

Usage:
    python src/experiment/reset/reorient_drop.py --obj brown_ramen \\
        --target_tabletop 002 --auto
"""
from __future__ import annotations

import argparse
import atexit
import datetime
import glob
import json
import logging
import os
import signal
import sys
import time
from pathlib import Path

import chime
import numpy as np
import trimesh
import yourdfpy
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.io.camera_system.signal_generator import UTGE900
from paradex.io.camera_system.timestamp_monitor import TimestampMonitor
from paradex.utils.system import network_info, get_pc_ip, get_camera_list
from paradex.calibration.utils import save_current_camparam, save_current_C2R, load_c2r

from autodex.utils.path import project_dir, obj_path, load_openpose_for_candidates
from autodex.utils.conversion import cart2se3
from autodex.planner import GraspPlanner
from autodex.planner.planner import PlanResult, _to_curobo_world
from autodex.planner.obstacles import TABLE_CUBOID, add_obstacles
from autodex.planner.visualizer import ScenePlanVisualizer
from autodex.executor.real import RealExecutor
from autodex.perception.init_orchestrator import InitOrchestrator
from autodex.perception.snapshot_orchestrator import SnapshotOrchestrator

from src.execution.scene_cfg import pose_world_to_scene_cfg
from autodex.utils.symmetry import get_cyl_axis_local
from src.execution.run_auto import (
    DEFAULT_PC_LIST, ASSETS_BASE, MESH_BASE, CAM_PARAM_ROOT, _load_calib,
)
from src.execution.label import auto_label_charuco
from src.experiment.reset.tabletop_pose import classify_tabletop_pose

# viser viz helpers (mirror view_reorient.py).
from paradex.visualization.visualizer.viser import ViserViewer

_URDF_ROOT = Path.home() / "shared_data" / "AutoDex" / "content" / "assets" / "robot"
URDF_BY_HAND_VIZ = {
    "inspire_left": _URDF_ROOT / "inspire_left_description" / "xarm_inspire_left.urdf",
    "inspire":      _URDF_ROOT / "inspire_description"      / "xarm_inspire.urdf",
    "allegro":      _URDF_ROOT / "allegro_description"      / "xarm_allegro.urdf",
}
FLOATING_URDF_BY_HAND = {
    "inspire_left": _URDF_ROOT / "inspire_description" / "inspire_left_floating.urdf",
    "inspire":      _URDF_ROOT / "inspire_description" / "inspire_floating.urdf",
    "allegro":      _URDF_ROOT / "allegro_description" / "allegro_floating.urdf",
}
EE_LINK = "base_link"


def _fk_ee(urdf, joint_traj):
    out = np.tile(np.eye(4), (len(joint_traj), 1, 1))
    for t, q in enumerate(joint_traj):
        urdf.update_cfg(q)
        out[t] = urdf.get_transform(EE_LINK, urdf.base_link)
    return out

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logging.getLogger("curobo").setLevel(logging.WARNING)


# ── Hardcoded defaults (rarely changed across runs) ──────────────────────────
GRASP_VERSION = "table_only"
LIFT_HEIGHT_M = 0.25         # +25cm above grasp pose
RELEASE_HEIGHT_M = 0.25      # release at same height as lift (no descent)
TABLE_SURFACE_Z = TABLE_CUBOID["pose"][2] + TABLE_CUBOID["dims"][2] / 2  # 0.039
EXP_NAME = "reset_test/reorient_drop"
SCENE = "table"
SUCCESS_ONLY = False
PC_LIST = DEFAULT_PC_LIST
PORT_MASK = 5006
PORT_POSE = 5007
PORT_CMD = 6893
# Snapshot daemon ports (scripts/snapshot_daemons.sh start)
PORT_SNAP = 5009
PORT_SNAP_CMD = 6894
PROMPT = "object on the checkerboard"
SIL_ITERS = 100
SIL_LR = 0.002
INIT_TIMEOUT_S = 120.0
POST_INIT_TIMEOUT_S = 60.0
STREAM_FPS = 30
STREAM_WARMUP_S = 2.0
VIDEO_FPS = 30
CYCLE_SLEEP_S = 2.0
POST_DROP_SETTLE_S = 1.0
CHARUCO_BOARD = "1"


class _SoftSkip(Exception):
    """Perception/plan/lift/charuco failure: log the cycle and continue."""


def _now() -> str:
    return datetime.datetime.now().isoformat()


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _stop_video(rcc, sync_generator, timestamp_monitor):
    """Stop video recording. Order matters — see naive_drop.py for rationale."""
    rcc.stop()
    time.sleep(0.3)
    sync_generator.stop()
    timestamp_monitor.stop()


def _viz_plan_fail(stage: str, scene_cfg: dict, vis_prev, obj_name: str,
                   port: int, hand: str,
                   T_obj_target=None, x_grid=None, yaw_grid=None,
                   T_obj_now=None, T_wrist_target=None) -> object:
    """Open a viser viewer showing what was being attempted when ``stage``
    failed: target obj pose, the (x, yaw) grid of obj poses tried, current
    obj pose (hand-attached for lift/reorient/descent), and the wrist target
    if known. Blocks on Enter so the user can inspect.
    """
    if vis_prev is not None:
        try: vis_prev.stop_viewer()
        except Exception: pass
    try:
        new_vis = ScenePlanVisualizer(scene_cfg, None, port=port, hand=hand)
        mesh_path = Path(obj_path) / obj_name / "raw_mesh" / f"{obj_name}.obj"
        target_mesh = (trimesh.load(str(mesh_path), process=False)
                       if mesh_path.exists() else None)
        if T_obj_target is not None and target_mesh is not None:
            new_vis.add_object(f"{stage}_target_obj",
                               target_mesh, T_obj_target)
            new_vis.add_frame(f"{stage}_target_axes", T_obj_target)
        if (x_grid is not None and yaw_grid is not None
                and T_obj_target is not None):
            R_base = T_obj_target[:3, :3]
            for xi, x_try in enumerate(x_grid):
                for yi, yaw_try in enumerate(yaw_grid):
                    cc, ss = np.cos(yaw_try), np.sin(yaw_try)
                    R_z = np.array([[cc, -ss, 0.0],
                                    [ss,  cc, 0.0],
                                    [0.0, 0.0, 1.0]])
                    T_o = np.eye(4)
                    T_o[:3, :3] = R_z @ R_base
                    T_o[0, 3] = float(x_try)
                    T_o[1, 3] = float(T_obj_target[1, 3])
                    T_o[2, 3] = float(T_obj_target[2, 3])
                    try:
                        new_vis.add_frame(
                            f"grid_x{xi:02d}_y{yi:02d}", T_o)
                    except Exception: pass
                    if target_mesh is not None:
                        try:
                            new_vis.add_object(
                                f"grid_mesh_x{xi:02d}_y{yi:02d}",
                                target_mesh.copy(), T_o)
                        except Exception: pass
        if T_obj_now is not None and target_mesh is not None:
            new_vis.add_object("obj_now", target_mesh, T_obj_now)
            new_vis.add_frame("obj_now_axes", T_obj_now)
        if T_wrist_target is not None:
            new_vis.add_frame(f"{stage}_wrist_target", T_wrist_target)
        new_vis.start_viewer(use_thread=True)
        print(f"[viz] {stage} plan FAILED — viewer at "
              f"http://localhost:{port}")
        input("[viz] press Enter to continue...")
        return new_vis
    except Exception as ve:
        print(f"[viz] {stage} viz setup failed: {ve!r}")
        return vis_prev


def _viz_cand_failures(cand_log, obj_name, port, hand, mesh_path,
                       vis_prev=None):
    """Dropdown viewer: xarm at last_qpos + floating hand at T_wrist_target
    + object at fail-time pose. Blocks on Enter."""
    if vis_prev is not None:
        try: vis_prev.stop_viewer()
        except Exception: pass
    if not cand_log:
        print("[viz] no candidate failures to show")
        return vis_prev
    try:
        urdf_path = URDF_BY_HAND_VIZ[hand]
        floating_urdf_path = FLOATING_URDF_BY_HAND[hand]
        target_mesh = (trimesh.load(str(mesh_path), process=False)
                       if mesh_path.exists() else None)
        vis = ViserViewer(port_number=port)
        vis.add_robot("xarm", str(urdf_path))
        vis.add_robot("floating_hand", str(floating_urdf_path))
        # Pre-add 10 extra floating hands for sample wrist viz (reorient grid).
        N_SAMPLE = 10
        for k in range(N_SAMPLE):
            vis.add_robot(f"sample_hand_{k}", str(floating_urdf_path))
        vis.add_floor(height=0.0)
        first = cand_log[0]
        init_T_obj = first.get("T_obj_at_fail")
        if target_mesh is not None:
            vis.add_object("obj", target_mesh,
                           init_T_obj if init_T_obj is not None else np.eye(4))

        labels = [
            f"#{e['cand_idx']:02d} [{e['stage']}] {e.get('reason') or ''}"
            for e in cand_log
        ]
        cand_dd = vis.server.gui.add_dropdown(
            "Failed candidate", options=tuple(labels))
        info_md = vis.server.gui.add_markdown("(select a candidate)")

        def _apply(idx):
            e = cand_log[idx]
            qpos = e.get("last_qpos")
            # xarm at last reached qpos
            if qpos is not None:
                try: vis.robot_dict["xarm"].update_cfg(np.asarray(qpos))
                except Exception as ue:
                    print(f"[viz] xarm update_cfg: {ue!r}")
            # floating hand at T_wrist_target
            T_w = e.get("T_wrist_target")
            if T_w is not None and "floating_hand" in vis.robot_dict:
                fh = vis.robot_dict["floating_hand"]
                fh._visual_root_frame.position = T_w[:3, 3]
                fh._visual_root_frame.wxyz = (
                    R.from_matrix(T_w[:3, :3]).as_quat()[[3, 0, 1, 2]])
                grasp_q = e.get("grasp_qpos")
                if grasp_q is not None:
                    try: fh.update_cfg(np.asarray(grasp_q))
                    except Exception: pass
            # object at fail-time pose
            T_obj = e.get("T_obj_at_fail")
            if T_obj is not None and "obj" in vis.obj_dict:
                fr = vis.obj_dict["obj"]["frame"]
                fr.position = T_obj[:3, 3]
                fr.wxyz = R.from_matrix(T_obj[:3, :3]).as_quat()[[3, 0, 1, 2]]
            # Sample wrist targets (reorient grid) — extra floating hands
            samples = e.get("T_wrist_targets_sample") or []
            grasp_q = e.get("grasp_qpos")
            for k in range(N_SAMPLE):
                key = f"sample_hand_{k}"
                if key not in vis.robot_dict:
                    continue
                fh_k = vis.robot_dict[key]
                if k < len(samples):
                    T_k = samples[k]
                    fh_k._visual_root_frame.position = T_k[:3, 3]
                    fh_k._visual_root_frame.wxyz = (
                        R.from_matrix(T_k[:3, :3]).as_quat()[[3, 0, 1, 2]])
                    fh_k._visual_root_frame.visible = True
                    if grasp_q is not None:
                        try: fh_k.update_cfg(np.asarray(grasp_q))
                        except Exception: pass
                else:
                    fh_k._visual_root_frame.visible = False
            # Print wrist target xyz for diagnosis
            xyz_str = (f"main wrist xyz="
                       f"{np.round(T_w[:3, 3], 4).tolist() if T_w is not None else None}")
            samp_xyz = [np.round(t[:3, 3], 3).tolist() for t in samples[:5]]
            print(f"  [viz] cand#{e['cand_idx']} stage={e['stage']} {xyz_str} "
                  f"  sample_xyz(first 5)={samp_xyz}")
            info_md.content = (
                f"**cand #{e['cand_idx']}** — fail at **{e['stage']}**  \n"
                f"reason: `{e.get('reason')}`  \n"
                f"sample wrist count: {len(samples)}"
            )

        @cand_dd.on_update
        def _(_):
            _apply(labels.index(cand_dd.value))

        _apply(0)
        vis.start_viewer(use_thread=True)
        print(f"[viz] cand-fail viewer at "
              f"http://localhost:{port} ({len(cand_log)} cands)")
        input("[viz] press Enter to continue...")
        return vis
    except Exception as ve:
        import traceback; traceback.print_exc()
        print(f"[viz] cand-fail viz setup failed: {ve!r}")
        return vis_prev


def _load_target_tabletop(obj_name: str, key: str) -> tuple[str, int, np.ndarray]:
    """Return (filename, sorted_idx, 4x4 pose) for the requested tabletop.

    ``key`` is the FILENAME stem (e.g. ``"002"`` for ``002.npy``) — matches
    what the user sees on disk. ``sorted_idx`` is the 0-based position in the
    sorted glob, identical to what ``classify_tabletop_pose`` returns, so we
    can compare against ``tb_after['idx']`` for hit-target accounting.
    """
    tabletop_dir = os.path.join(obj_path, obj_name, "processed_data",
                                "info", "tabletop")
    files = sorted(glob.glob(os.path.join(tabletop_dir, "*.npy")))
    if not files:
        sys.exit(f"no tabletop poses found at {tabletop_dir}")
    fname = f"{key}.npy"
    matched_idx = next((i for i, f in enumerate(files)
                        if os.path.basename(f) == fname), None)
    if matched_idx is None:
        avail = ", ".join(os.path.basename(f).replace(".npy", "") for f in files)
        sys.exit(f"tabletop {fname!r} not found in {tabletop_dir} — "
                 f"available: [{avail}]")
    pose = np.load(files[matched_idx])
    if pose.shape == (3, 3):
        T = np.eye(4); T[:3, :3] = pose
        pose = T
    return fname, matched_idx, pose


def _z_align_yaw(R_est: np.ndarray, R_tab: np.ndarray) -> np.ndarray:
    """Return R_z(theta) @ R_tab where theta is the optimal yaw aligning R_tab
    to R_est. Mirrors _z_aligned_geodesic_deg in tabletop_pose.py — the
    tabletop class is yaw-invariant, so we pick the yaw that makes the
    reorient motion minimal w.r.t. the currently-grasped object orientation.
    """
    M = R_est @ R_tab.T
    theta = np.arctan2(M[1, 0] - M[0, 1], M[0, 0] + M[1, 1])
    c, s = np.cos(theta), np.sin(theta)
    R_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return R_z @ R_tab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--hand", type=str, default="inspire_left",
                        choices=["allegro", "inspire", "inspire_left"])
    parser.add_argument("--target_tabletop", type=str, required=True,
                        help="Filename stem of the desired tabletop pose "
                             "(e.g. '002' for 002.npy in "
                             "{obj}/processed_data/info/tabletop/).")
    parser.add_argument("--auto", action="store_true",
                        help="Skip per-cycle Enter prompt, fully autonomous.")
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--port_viser", type=int, default=8080)
    args = parser.parse_args()

    # Asset sanity.
    mesh_path = MESH_BASE / args.obj / "raw_mesh" / f"{args.obj}.obj"
    assets_root = ASSETS_BASE / args.obj
    if not mesh_path.exists():
        sys.exit(f"mesh not found: {mesh_path}")
    if not (assets_root / "object_repre/v1" / args.obj / "1/repre.pth").exists():
        sys.exit(f"repre.pth missing for {args.obj}")

    # Target tabletop pose (robot-frame, fixed for whole run).
    tt_filename, tt_sorted_idx, target_tabletop_robot = _load_target_tabletop(
        args.obj, args.target_tabletop)
    R_target_robot = target_tabletop_robot[:3, :3]
    print(f"[target] tabletop file={tt_filename} "
          f"(sorted_idx={tt_sorted_idx} — matches classify_tabletop_pose)")

    # Calibration.
    calib_dir = sorted(CAM_PARAM_ROOT.iterdir())[-1]
    print(f"calib: {calib_dir.name}")
    intrinsics_full, extrinsics_full, H, W = _load_calib(calib_dir)

    pc_ips = [get_pc_ip(p) for p in PC_LIST]
    pc_serials = {p: get_camera_list(p) for p in PC_LIST}
    active = {s for pc in PC_LIST for s in pc_serials[pc]}
    intrinsics_full = {s: v for s, v in intrinsics_full.items() if s in active}
    extrinsics_full = {s: v for s, v in extrinsics_full.items() if s in active}
    print(f"  {len(intrinsics_full)} cams active across {len(PC_LIST)} PCs ({H}x{W})")

    # Hardware stream.
    client_name = f"reorient_drop_{os.getpid()}"
    rcc = remote_camera_controller(client_name, pc_list=PC_LIST)
    print(f"[stream] starting on {len(PC_LIST)} PCs @ {STREAM_FPS} FPS "
          f"(client={client_name})...")
    rcc.start("stream", False, fps=STREAM_FPS)
    time.sleep(STREAM_WARMUP_S)

    sync_generator = UTGE900(**network_info["signal_generator"]["param"])
    timestamp_monitor = TimestampMonitor(**network_info["timestamp"]["param"])
    print(f"[video] @ {VIDEO_FPS} FPS")

    sub = args.hand
    obj_root = Path(project_dir) / "experiment" / EXP_NAME / sub / args.obj
    obj_root.mkdir(parents=True, exist_ok=True)

    # Init orchestrator (1x).
    print(f"[orch] init for {args.obj}...")
    orch = InitOrchestrator(
        pc_list=PC_LIST, capture_ips=pc_ips,
        port_mask=PORT_MASK, port_pose=PORT_POSE, port_cmd=PORT_CMD,
    )
    # Snapshot orchestrator for charuco lift-check (no video stop needed).
    snap_orch = SnapshotOrchestrator(
        pc_list=PC_LIST, capture_ips=pc_ips,
        port_snap=PORT_SNAP, port_cmd=PORT_SNAP_CMD,
    )
    n_cams_total = sum(len(pc_serials[p]) for p in PC_LIST)
    orch.init_object(
        obj_name=args.obj,
        mesh_path=str(mesh_path), assets_root=str(assets_root),
        intrinsics_full=intrinsics_full, extrinsics_full=extrinsics_full,
        image_hw=(H, W), mode="live", pc_serials=pc_serials,
    )

    print("[planner] warming up curobo...")
    planner = GraspPlanner(hand=args.hand)
    # cuRobo's MotionGenConfig resets the "curobo" logger to INFO inside
    # GraspPlanner. Silence it after construction.
    from curobo.util.logger import setup_curobo_logger
    setup_curobo_logger("warning")
    print("[executor] connecting to robot...")
    executor = RealExecutor(hand_name=args.hand)

    trials: list = []
    summary_path = obj_root / "summary.json"
    cycle = 0
    vis = None

    # ── Guaranteed cleanup (atexit + signal handlers) ────────────────────
    _cleanup_done = [False]

    def _do_cleanup():
        if _cleanup_done[0]:
            return
        _cleanup_done[0] = True
        print("\n[cleanup] tearing down resources...")
        nonlocal vis

        import threading
        def _call_with_timeout(label, fn, timeout=5):
            done = threading.Event()
            err_holder: list = []
            def _wrap():
                try:
                    fn()
                except Exception as ce:
                    err_holder.append(ce)
                finally:
                    done.set()
            t = threading.Thread(target=_wrap, daemon=True)
            t.start()
            done.wait(timeout=timeout)
            if not done.is_set():
                print(f"[cleanup] {label} TIMED OUT after {timeout}s — skipping")
                return False
            if err_holder:
                print(f"[cleanup] {label} failed: {err_holder[0]!r}")
            return True

        for label, fn in (
            ("vis.stop_viewer", (lambda: vis.stop_viewer()) if vis else None),
            ("executor.stop_recording", executor.stop_recording),
            ("executor.shutdown", executor.shutdown),
            ("orch.close", orch.close),
            ("rcc.stop", rcc.stop),
            ("sync_generator.stop", sync_generator.stop),
            ("timestamp_monitor.stop", timestamp_monitor.stop),
            ("sync_generator.end", sync_generator.end),
            ("timestamp_monitor.end", timestamp_monitor.end),
            ("rcc.end", rcc.end),
        ):
            if fn is None:
                continue
            _call_with_timeout(label, fn, timeout=5)
        print("[cleanup] done — forcing exit")
        os._exit(0)

    def _signal_handler(signum, frame):
        print(f"\n[signal] received {signal.Signals(signum).name}, cleaning up...")
        _do_cleanup()
        sys.exit(128 + signum)

    atexit.register(_do_cleanup)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGINT, _signal_handler)

    try:
        while True:
            print(f"\n{'#'*60}\n# Cycle {cycle}\n{'#'*60}")
            chime.info()
            if not args.auto:
                try:
                    cmd = input(f"[cycle {cycle}] Enter=start, q=quit: ").strip().lower()
                except KeyboardInterrupt:
                    print("\n[loop] KeyboardInterrupt, stopping.")
                    break
                if cmd == "q":
                    break

            trial_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            cdir = obj_root / trial_ts
            (cdir / "plan").mkdir(parents=True, exist_ok=True)
            save_current_C2R(str(cdir))
            save_current_camparam(str(cdir))

            rec: dict = {
                "cycle": cycle,
                "trial_ts": trial_ts,
                "start": _now(),
                "obj": args.obj, "hand": args.hand, "scene": SCENE,
                "target_tabletop": {
                    "key": args.target_tabletop,
                    "filename": tt_filename,
                    "sorted_idx": tt_sorted_idx,
                },
                "status": "started",
                "progress": {
                    "perception": None, "tabletop_before": None, "plan": None,
                    "execute": None, "lift": None, "charuco": None,
                    "reorient": None, "descent": None, "release": None,
                    "retract": None, "post_perception": None,
                    "tabletop_after": None,
                },
                "timing": {},
                "tabletop_before": None,
                "tabletop_after": None,
                "tabletop_changed": None,
                "drop_quality": None,
                "files": {
                    "pose_world": "pose_world.npy",
                    "pose_world_after_drop": "pose_world_after_drop.npy",
                    "perception_images": "init_capture/",
                    "label_images": "label_at_lift/",
                    "post_drop_images": "post_drop_capture/",
                    "plan_dir": "plan/",
                    "actual_robot": "raw/",
                    "cam_param": "cam_param/",
                    "raw_video": "raw/",
                    "timestamps": "raw/timestamps",
                },
            }

            result = None
            scene_cfg = None
            planned_obj_pose = None
            T_obj_in_wrist = None
            video_started = False

            try:
                # 1. Perception (always — no lock_pose).
                print(f"[cycle {cycle}] Perception...")
                t0 = time.time()
                pose_world, perc_timing = orch.trigger_init(
                    prompt=PROMPT,
                    save_capture_dir=str(cdir / "init_capture"),
                    sil_iters=SIL_ITERS, sil_lr=SIL_LR,
                    timeout_s=INIT_TIMEOUT_S,
                )
                rec["timing"]["perception_s"] = round(time.time() - t0, 2)
                if perc_timing:
                    rec["timing"]["perception_detail"] = perc_timing
                if pose_world is None:
                    reason = (perc_timing or {}).get("reason", "perception_failed")
                    rec["progress"]["perception"] = f"failed: {reason}"
                    rec["status"] = "perception_failed"
                    rec["reason"] = reason
                    print(f"    perception FAILED ({reason}) — skipping cycle")
                    raise _SoftSkip
                rec["progress"]["perception"] = "ok"
                np.save(cdir / "pose_world.npy", pose_world)

                # 2. Tabletop classification (before).
                c2r = load_c2r(str(cdir))
                pose_robot_before = np.linalg.inv(c2r) @ pose_world
                tb_before = classify_tabletop_pose(pose_robot_before, args.obj)
                rec["tabletop_before"] = tb_before
                rec["progress"]["tabletop_before"] = (
                    f"idx={tb_before['idx']} ({tb_before['rot_err_deg']:.1f}deg)"
                    if tb_before else "no_tabletop_data"
                )
                if tb_before:
                    print(f"    [tabletop before] idx={tb_before['idx']} "
                          f"({tb_before['filename']}) err={tb_before['rot_err_deg']:.1f}deg")

                # 3. Plan — enumerate ALL IK-feasible grasp candidates, then
                #    pick the first one whose (x, yaw) reorient pre-check also
                #    passes. Mirrors view_reorient.py's enumeration. Avoids
                #    grasping a candidate that can't be placed at target
                #    orientation (which would waste lift + charuco + reorient
                #    only to fail).
                print(f"[cycle {cycle}] Planning (scene={SCENE})...")
                t0 = time.time()
                scene_cfg = pose_world_to_scene_cfg(pose_world, c2r, args.obj)
                scene_cfg = add_obstacles(scene_cfg, SCENE)
                _write_json(cdir / "scene_cfg.json", scene_cfg)

                start_scene_id = (str(tb_before["idx"])
                                  if tb_before is not None else None)

                # Clear motion_gen world cache + reset RNG seeds so plan-fail
                # count doesn't drift across cycles. Skip graph_planner
                # reset_buffer() (heavy; ~20s in our setup).
                if planner._motion_gen is not None:
                    planner._motion_gen.clear_world_cache()
                    planner._motion_gen.reset_seed()

                # solve_ik (retract + lift IK check now matches plan()).
                # Cylinder freedom for approach IK — expand candidates by
                # cyl_yaw rotations around the object's symmetry axis when the
                # axis is not (nearly) world-vertical. For a lying pepsi this
                # turns 30 candidates → 30×8 = 240 candidates → much higher
                # chance of finding an IK-feasible grasp from current arm INIT.
                start_cyl_axis = get_cyl_axis_local(args.obj)
                if start_cyl_axis is not None:
                    _pose_robot_start = np.linalg.inv(c2r) @ pose_world
                    _axis_world_start = _pose_robot_start[:3, :3] @ start_cyl_axis
                    if abs(_axis_world_start[2]) >= 0.95:
                        start_cyl_grid = None   # standing — degenerate
                    else:
                        start_cyl_grid = np.linspace(
                            0, 2 * np.pi, 8, endpoint=False)
                else:
                    start_cyl_grid = None
                ik_res = planner.solve_ik(
                    scene_cfg, args.obj, GRASP_VERSION,
                    hand=args.hand, scene_id=start_scene_id,
                    cyl_axis_local=start_cyl_axis,
                    cyl_yaw_grid=start_cyl_grid)
                ik_ok = list(np.where(ik_res["ik_success"])[0])
                np.random.shuffle(ik_ok)
                n_grasp_total = ik_res["n_total"]
                planner._ik_solver = None  # rebuild for next plan_obj_placement
                # Load openpose (wider-open finger config) for each candidate,
                # keyed by the observed start tabletop pose. Used for the
                # approach IK goal so the hand is more open during approach.
                openpose_pose_stem = (
                    tb_before["filename"].replace(".npy", "")
                    if tb_before is not None else None)
                if openpose_pose_stem is not None:
                    openpose_list = load_openpose_for_candidates(
                        args.obj, ik_res["scene_info"], args.hand,
                        GRASP_VERSION, openpose_pose_stem)
                    n_op = sum(1 for op in openpose_list if op is not None)
                    print(f"    openpose: {n_op}/{n_grasp_total} candidates "
                          f"have openpose_{openpose_pose_stem}.npy")
                else:
                    openpose_list = [None] * n_grasp_total
                print(f"    grasp IK: {len(ik_ok)}/{n_grasp_total} feasible")
                if len(ik_ok) == 0:
                    rec["progress"]["plan"] = (
                        f"no_ik_feasible_grasp ({n_grasp_total} total)")
                    rec["status"] = "plan_failed"
                    raise _SoftSkip

                # motion_gen world for approach planning: obj mesh + table
                # (so the approach path is collision-aware against obj).
                # plan_wrist_reorient / plan_obj_placement auto-switch to
                # scene_lift_pre (table only) inside their own calls.
                T_obj_grasp_world_full = cart2se3(scene_cfg["mesh"]["target"]["pose"])
                R_target_obj_world_pre = target_tabletop_robot[:3, :3]
                obj_target_pos_world_pre = np.array([
                    0.0,        # x — overridden by X_GRID in plan_obj_placement
                    0.0,        # y — placing y default
                    float(target_tabletop_robot[2, 3])
                        + TABLE_SURFACE_Z + LIFT_HEIGHT_M,
                ])
                T_obj_target_world_pre = np.eye(4)
                T_obj_target_world_pre[:3, :3] = R_target_obj_world_pre
                T_obj_target_world_pre[:3, 3] = obj_target_pos_world_pre
                print(f"    [target] tabletop_robot xyz="
                      f"{np.round(target_tabletop_robot[:3, 3], 4).tolist()} "
                      f"-> obj_target_pos="
                      f"{np.round(obj_target_pos_world_pre, 4).tolist()} "
                      f"(TABLE_SURFACE_Z={TABLE_SURFACE_Z:.4f}, "
                      f"LIFT_HEIGHT_M={LIFT_HEIGHT_M:.4f})")
                scene_lift_pre = {"mesh": {},
                                  "cuboid": dict(scene_cfg["cuboid"])}
                # Target x range fixed to [0.40, 0.45] — placing zone in front
                # of robot. If IK infeasible there, fall back to in-place
                # release (skip reorient+descent stages).
                X_GRID_PRE = np.array([0.40, 0.45])
                YAW_GRID_PRE = np.linspace(0, 2 * np.pi, 8, endpoint=False)
                # Continuous-revolute objects: free DoF about the object's
                # local symmetry axis (from src/scene_generation/symmetry.json,
                # via get_cyl_axis_local), applied IN OBJECT FRAME. Final
                # target rotation is Rz(world_yaw) @ R_target @ R_local(cyl).
                # If the symmetry axis ends up (nearly) world-vertical at the
                # target, R_local(axis, cyl) is degenerate with the world yaw
                # — collapse cyl grid to a single point to avoid wasted IK.
                from autodex.utils.symmetry import get_cyl_yaw_grid
                CYL_AXIS_LOCAL = get_cyl_axis_local(args.obj)
                CYL_YAW_GRID_PRE = get_cyl_yaw_grid(args.obj)
                # Standing case (axis ≈ world z) collapses cyl with world yaw
                # — keep single-point grid for placement IK if so.
                if (CYL_AXIS_LOCAL is not None
                        and CYL_YAW_GRID_PRE is not None
                        and abs((R_target_obj_world_pre @ CYL_AXIS_LOCAL)[2])
                            >= 0.95):
                    CYL_YAW_GRID_PRE = np.array([0.0])

                # Init motion_gen with obj-world for approach planning.
                world_approach = _to_curobo_world(scene_cfg)
                planner._init_motion_gen(world_approach)
                planner._cached_world = world_approach

                # Constants for the in-air chain (same for every candidate).
                obj_target_pos_descent = np.array([
                    0.0, 0.0,
                    float(target_tabletop_robot[2, 3])
                        + TABLE_SURFACE_Z + RELEASE_HEIGHT_M,
                ])

                # Candidate loop: each cand must pass approach → lift → reorient
                # → descent. First cand whose full chain succeeds wins.
                chosen_cand_idx = None
                approach_traj_chosen = None
                lift_traj_chosen = None
                reorient_traj_chosen = None
                descent_traj_chosen = None
                T_obj_in_wrist_chosen = None
                T_wrist_lift_chosen = None
                T_wrist_descent_chosen = None
                pre_info = None
                pregrasp_chosen = None
                grasp_chosen = None
                scene_info_chosen = None
                result = None
                n_approach_fail = 0
                n_lift_fail = 0
                n_reorient_fail = 0
                n_descent_fail = 0
                last_fail_stage = None
                last_fail_info = None
                last_fail_lift_T = None
                last_fail_T_obj_in_wrist = None
                last_fail_reorient_info = None
                cand_log: list = []   # per-cand fail entry for viz dropdown

                for cand_idx in ik_ok:
                    cand_idx = int(cand_idx)
                    wrist_grasp_cand = ik_res["wrist_se3"][cand_idx]
                    grasp_cand = ik_res["grasp"][cand_idx]
                    pregrasp_cand = ik_res["pregrasp"][cand_idx]
                    T_obj_in_wrist_cand = (
                        np.linalg.inv(wrist_grasp_cand) @ T_obj_grasp_world_full)

                    # (a) approach — restore obj-world if a previous cand's
                    # plan_wrist_reorient/plan_obj_placement switched to
                    # table-only. Use openpose (wider hand) for the approach-
                    # end finger config if available; fall back to pregrasp.
                    if planner._world_structure_changed(world_approach):
                        planner._update_world(world_approach)
                        planner._cached_world = world_approach
                    approach_goal = ik_res["ik_qpos"][cand_idx].copy()
                    if openpose_list[cand_idx] is not None:
                        approach_goal[6:] = openpose_list[cand_idx]
                    ok_ap, approach_traj_cand = planner._refine_fingers(
                        planner._init_state, approach_goal)
                    if not ok_ap:
                        n_approach_fail += 1
                        cand_log.append({
                            "cand_idx": cand_idx, "stage": "approach",
                            "reason": "refine_fingers_failed",
                            "T_wrist_target": wrist_grasp_cand,
                            "T_obj_at_fail": T_obj_grasp_world_full,
                            "last_qpos": planner._init_state.copy(),
                            "grasp_qpos": grasp_cand,
                        })
                        print(f"  cand#{cand_idx}: approach FAIL")
                        continue

                    # (b) lift
                    T_wrist_lift_cand = wrist_grasp_cand.copy()
                    T_wrist_lift_cand[2, 3] += LIFT_HEIGHT_M
                    cur_qpos_lift = np.concatenate(
                        [approach_traj_cand[-1, :6], pregrasp_cand])
                    lift_traj_cand, lift_info_cand = planner.plan_wrist_reorient(
                        scene_lift_pre, cur_qpos_lift, T_wrist_lift_cand,
                        hold_hand_qpos=pregrasp_cand, n_yaw=8)
                    if lift_traj_cand is None:
                        n_lift_fail += 1
                        last_fail_stage = "lift"
                        last_fail_info = lift_info_cand
                        last_fail_lift_T = T_wrist_lift_cand
                        last_fail_T_obj_in_wrist = T_obj_in_wrist_cand
                        cand_log.append({
                            "cand_idx": cand_idx, "stage": "lift",
                            "reason": lift_info_cand.get("reason"),
                            "T_wrist_target": T_wrist_lift_cand,
                            "T_obj_at_fail": T_wrist_lift_cand @ T_obj_in_wrist_cand,
                            "last_qpos": cur_qpos_lift.copy(),
                            "grasp_qpos": grasp_cand,
                        })
                        print(f"  cand#{cand_idx}: lift FAIL "
                              f"({lift_info_cand.get('reason')})")
                        continue

                    # (c+d) reorient + descent — straight-down. Get sorted
                    # IK-feasible (x, yaw) list from reorient skip_plan, then
                    # try each in order: both reorient AND descent must plan.
                    # First (x, yaw) where both stages succeed wins.
                    cur_qpos_reorient = lift_traj_cand[-1].copy()
                    _, sorted_info = planner.plan_obj_placement(
                        scene_lift_pre, cur_qpos_reorient, T_obj_in_wrist_cand,
                        R_target_obj_world_pre, obj_target_pos_world_pre,
                        hold_hand_qpos=pregrasp_cand,
                        x_grid=X_GRID_PRE, yaw_grid=YAW_GRID_PRE,
                        cyl_yaw_grid=CYL_YAW_GRID_PRE,
                        cyl_axis_local=CYL_AXIS_LOCAL,
                        skip_plan=True)
                    sorted_cands = sorted_info.get("sorted_candidates", [])
                    # Re-sort: prefer x near grid center (further from robot
                    # body) over arm-distance-minimizing best.
                    x_center = 0.5 * (float(X_GRID_PRE[0]) + float(X_GRID_PRE[-1]))
                    sorted_cands = sorted(
                        sorted_cands, key=lambda sc: abs(sc["x"] - x_center))
                    reorient_traj_cand = None
                    descent_traj_cand = None
                    pre_info_cand = None
                    descent_info_cand = None
                    last_reorient_fail = None
                    last_descent_fail = None
                    print(f"  [dbg] cand#{cand_idx}: inner sc loop "
                          f"len(sorted_cands)={len(sorted_cands)}")
                    for sc_i, sc in enumerate(sorted_cands):
                        sx, syaw, scyl = sc["x"], sc["yaw"], sc["cyl_yaw"]
                        x1 = np.array([sx]); y1 = np.array([syaw])
                        cyl1 = (np.array([scyl])
                                if CYL_YAW_GRID_PRE is not None else None)
                        # reorient at LIFT_HEIGHT
                        r_traj, r_info = planner.plan_obj_placement(
                            scene_lift_pre, cur_qpos_reorient, T_obj_in_wrist_cand,
                            R_target_obj_world_pre, obj_target_pos_world_pre,
                            hold_hand_qpos=pregrasp_cand,
                            x_grid=x1, yaw_grid=y1,
                            cyl_yaw_grid=cyl1,
                            cyl_axis_local=CYL_AXIS_LOCAL,
                            skip_plan=False)
                        print(f"    [dbg] sc#{sc_i} x={sx:.3f} yaw={syaw:.3f} "
                              f"reorient={'OK' if r_traj is not None else 'FAIL'} "
                              f"reason={r_info.get('reason')} "
                              f"n_feas={r_info.get('n_feasible')} "
                              f"n_attempts={r_info.get('n_plan_attempts')}")
                        if r_traj is None:
                            last_reorient_fail = (sc, r_info)
                            continue
                        # descent at RELEASE_HEIGHT, same (x, yaw, cyl_yaw)
                        cur_qpos_descent = r_traj[-1].copy()
                        d_traj, d_info = planner.plan_obj_placement(
                            scene_lift_pre, cur_qpos_descent, T_obj_in_wrist_cand,
                            R_target_obj_world_pre, obj_target_pos_descent,
                            hold_hand_qpos=pregrasp_cand,
                            x_grid=x1, yaw_grid=y1,
                            cyl_yaw_grid=cyl1,
                            cyl_axis_local=CYL_AXIS_LOCAL,
                            skip_plan=False)
                        print(f"    [dbg] sc#{sc_i} descent="
                              f"{'OK' if d_traj is not None else 'FAIL'} "
                              f"reason={d_info.get('reason')} "
                              f"n_feas={d_info.get('n_feasible')} "
                              f"n_attempts={d_info.get('n_plan_attempts')}")
                        if d_traj is None:
                            last_descent_fail = (sc, d_info,
                                                 cur_qpos_descent.copy())
                            # Per-sc viz entry: reorient OK + descent FAIL.
                            # Use reorient-end qpos so viz shows the IK that
                            # succeeded just before descent failed.
                            cy_p, sy_p = (np.cos(syaw), np.sin(syaw))
                            Rz_p = np.array([[cy_p, -sy_p, 0.0],
                                             [sy_p,  cy_p, 0.0],
                                             [0.0,   0.0,  1.0]])
                            if CYL_AXIS_LOCAL is not None:
                                R_cyl_p = R.from_rotvec(
                                    CYL_AXIS_LOCAL * float(scyl)
                                ).as_matrix()
                            else:
                                R_cyl_p = np.eye(3)
                            T_obj_descent_p = np.eye(4)
                            T_obj_descent_p[:3, :3] = (
                                Rz_p @ R_target_obj_world_pre @ R_cyl_p)
                            T_obj_descent_p[0, 3] = float(sx)
                            T_obj_descent_p[1, 3] = float(
                                obj_target_pos_descent[1])
                            T_obj_descent_p[2, 3] = float(
                                obj_target_pos_descent[2])
                            cand_log.append({
                                "cand_idx": cand_idx,
                                "stage": f"descent_sc{sc_i}",
                                "reason": d_info.get("reason"),
                                "T_wrist_target": d_info.get("T_wrist_target"),
                                "T_obj_at_fail": T_obj_descent_p,
                                "last_qpos": cur_qpos_descent.copy(),
                                "grasp_qpos": grasp_cand,
                            })
                            continue
                        reorient_traj_cand = r_traj
                        descent_traj_cand = d_traj
                        pre_info_cand = r_info
                        descent_info_cand = d_info
                        break

                    if reorient_traj_cand is None:
                        # No (x, yaw) made it through the chain. Record the
                        # latest fail stage seen.
                        if last_descent_fail is not None:
                            n_descent_fail += 1
                            # Per-sc entries already appended inside sc loop.
                            print(f"  cand#{cand_idx}: descent FAIL on every "
                                  f"reorient-feasible (x, yaw) "
                                  f"({len(sorted_cands)} tried)")
                        else:
                            n_reorient_fail += 1
                            # Show one reorient target wrist (grid[0]) so the
                            # floating-hand viz reflects the flipped pose the
                            # reorient was trying to achieve, NOT the un-flipped
                            # lift pose.
                            T_w_show = sorted_info.get("T_wrist_target")
                            if T_w_show is None:
                                T_w_show = T_wrist_lift_cand
                            # Build up to 10 sample wrist targets from the
                            # 56-point grid so viz can show several attempted
                            # reorient poses (not just one).
                            T_inv = np.linalg.inv(T_obj_in_wrist_cand)
                            sample_wrists = []
                            n_max = 10
                            grid_pairs = [(x, y) for x in X_GRID_PRE
                                          for y in YAW_GRID_PRE]
                            step = max(1, len(grid_pairs) // n_max)
                            for (xv, yv) in grid_pairs[::step][:n_max]:
                                cc, ss = np.cos(yv), np.sin(yv)
                                Rz = np.array([[cc, -ss, 0.0],
                                               [ss,  cc, 0.0],
                                               [0.0, 0.0, 1.0]])
                                T_obj = np.eye(4)
                                T_obj[:3, :3] = Rz @ R_target_obj_world_pre
                                T_obj[0, 3] = float(xv)
                                T_obj[1, 3] = float(obj_target_pos_world_pre[1])
                                T_obj[2, 3] = float(obj_target_pos_world_pre[2])
                                sample_wrists.append(T_obj @ T_inv)
                            # Show reorient target obj pose (flipped to
                            # target orientation), NOT the lift-time obj.
                            cand_log.append({
                                "cand_idx": cand_idx, "stage": "reorient",
                                "reason": sorted_info.get("reason"),
                                "T_wrist_target": T_w_show,
                                "T_wrist_targets_sample": sample_wrists,
                                "T_obj_at_fail": T_obj_target_world_pre,
                                "last_qpos": cur_qpos_reorient.copy(),
                                "grasp_qpos": grasp_cand,
                            })
                            print(f"  cand#{cand_idx}: reorient FAIL "
                                  f"(no_ik feasible, "
                                  f"n_candidates={sorted_info.get('n_candidates')})")
                        continue

                    # All four stages passed — commit.
                    chosen_cand_idx = cand_idx
                    pre_info = pre_info_cand
                    approach_traj_chosen = approach_traj_cand
                    lift_traj_chosen = lift_traj_cand
                    reorient_traj_chosen = reorient_traj_cand
                    descent_traj_chosen = descent_traj_cand
                    T_obj_in_wrist_chosen = T_obj_in_wrist_cand
                    T_wrist_lift_chosen = T_wrist_lift_cand
                    T_wrist_descent_chosen = descent_info_cand.get("T_wrist_target")
                    grasp_chosen = grasp_cand
                    pregrasp_chosen = pregrasp_cand
                    openpose_chosen = openpose_list[cand_idx]
                    scene_info_chosen = ik_res["scene_info"][cand_idx]
                    result = PlanResult(
                        success=True, traj=approach_traj_cand,
                        wrist_se3=wrist_grasp_cand,
                        pregrasp_pose=pregrasp_cand,
                        grasp_pose=grasp_cand,
                        scene_info=scene_info_chosen,
                        openpose_pose=openpose_chosen,
                        timing={"candidate_idx": cand_idx,
                                "n_approach_fail": n_approach_fail,
                                "n_lift_fail": n_lift_fail,
                                "n_reorient_fail": n_reorient_fail,
                                "n_descent_fail": n_descent_fail},
                    )
                    break

                rec["timing"]["plan_s"] = round(time.time() - t0, 2)
                if chosen_cand_idx is None:
                    rec["progress"]["plan"] = (
                        f"no_grasp_passed (approach_fail={n_approach_fail}, "
                        f"lift_fail={n_lift_fail}, "
                        f"reorient_fail={n_reorient_fail}, "
                        f"descent_fail={n_descent_fail}, total={len(ik_ok)})")
                    rec["status"] = "no_grasp_with_feasible_full_chain"
                    print(f"[cycle {cycle}] No candidate passed full chain "
                          f"(approach_fail={n_approach_fail}/{len(ik_ok)}  "
                          f"lift_fail={n_lift_fail}/{len(ik_ok)}  "
                          f"reorient_fail={n_reorient_fail}/{len(ik_ok)}  "
                          f"descent_fail={n_descent_fail}/{len(ik_ok)})")
                    if args.viz:
                        vis = _viz_cand_failures(
                            cand_log, args.obj, args.port_viser, args.hand,
                            mesh_path=MESH_BASE / args.obj / "raw_mesh" /
                                       f"{args.obj}.obj",
                            vis_prev=vis)
                    raise _SoftSkip

                rec["progress"]["plan"] = "ok"
                rec["scene_info"] = scene_info_chosen
                rec["candidate_idx"] = chosen_cand_idx
                rec["reorient_pre_check"] = {
                    "n_feasible": pre_info["n_feasible"],
                    "n_candidates": pre_info["n_candidates"],
                    "chosen_grasp_idx": chosen_cand_idx,
                    "chosen_x": pre_info["chosen_x"],
                    "chosen_yaw_deg": float(np.degrees(pre_info["chosen_yaw"])),
                    "chosen_cyl_yaw_deg": float(np.degrees(
                        pre_info.get("chosen_cyl_yaw", 0.0))),
                    "n_approach_fail_before_chosen": n_approach_fail,
                    "n_lift_fail_before_chosen": n_lift_fail,
                    "n_reorient_fail_before_chosen": n_reorient_fail,
                    "n_descent_fail_before_chosen": n_descent_fail,
                }
                cyl_str = (
                    f" cyl={np.degrees(pre_info['chosen_cyl_yaw']):.0f}°"
                    if CYL_YAW_GRID_PRE is not None else "")
                print(f"    plan: {rec['timing']['plan_s']}s  cand#{chosen_cand_idx}  "
                      f"x={pre_info['chosen_x']:.3f} "
                      f"yaw={np.degrees(pre_info['chosen_yaw']):.0f}°{cyl_str}  "
                      f"(skipped approach={n_approach_fail}, lift={n_lift_fail}, "
                      f"reorient={n_reorient_fail}, descent={n_descent_fail})")

                np.save(cdir / "plan" / "traj.npy", result.traj)
                np.save(cdir / "plan" / "wrist_se3.npy", result.wrist_se3)
                np.save(cdir / "plan" / "pregrasp_pose.npy", result.pregrasp_pose)
                np.save(cdir / "plan" / "grasp_pose.npy", result.grasp_pose)
                np.save(cdir / "plan" / "lift_traj.npy", lift_traj_chosen)
                np.save(cdir / "plan" / "reorient_traj.npy", reorient_traj_chosen)
                np.save(cdir / "plan" / "descent_traj.npy", descent_traj_chosen)
                np.save(cdir / "plan" / "T_obj_in_wrist.npy", T_obj_in_wrist_chosen)
                np.save(cdir / "plan" / "T_wrist_lift.npy", T_wrist_lift_chosen)
                if T_wrist_descent_chosen is not None:
                    np.save(cdir / "plan" / "T_wrist_descent.npy",
                            T_wrist_descent_chosen)

                if args.viz:
                    # Phase-by-phase viser viz (mirrors view_reorient.py) —
                    # approach → grasp_close → lift → reorient → descent →
                    # release → hand_init → retract.
                    if vis is not None:
                        try:
                            vis.stop_viewer()
                        except Exception as ve:
                            print(f"[viz] previous viewer stop failed: {ve!r}")
                    try:
                        urdf_viz_path = URDF_BY_HAND_VIZ[args.hand]
                        urdf_fk = yourdfpy.URDF.load(str(urdf_viz_path))
                        pregrasp_h = np.asarray(pregrasp_chosen, dtype=np.float32)
                        grasp_h = np.asarray(grasp_chosen, dtype=np.float32)
                        init_hand_q = planner._init_state[6:].astype(np.float32)
                        # Override hand in planner-generated trajectories with
                        # grasp_h. plan_single_js sometimes nudges fingers for
                        # self-collision avoidance, which makes the viewer (and
                        # the real robot) look like the hand is opening
                        # mid-trajectory. We hold at grasp throughout lift/
                        # reorient/descent.
                        lift_traj_viz = lift_traj_chosen.copy()
                        lift_traj_viz[:, 6:] = grasp_h[None, :]
                        reorient_traj_viz = reorient_traj_chosen.copy()
                        reorient_traj_viz[:, 6:] = grasp_h[None, :]
                        descent_traj_viz = descent_traj_chosen.copy()
                        descent_traj_viz[:, 6:] = grasp_h[None, :]
                        # Synthesized hand-only / sequential phases.
                        grasp_qpos_top = approach_traj_chosen[-1].copy()
                        Nb = 20
                        b = np.linspace(0, 1, Nb)[:, None]
                        hand_close = (1 - b) * pregrasp_h[None, :] + b * grasp_h[None, :]
                        grasp_close_traj = np.concatenate(
                            [np.tile(grasp_qpos_top[:6][None], (Nb, 1)),
                             hand_close], axis=1)
                        Nr = 20
                        b_r = np.linspace(0, 1, Nr)[:, None]
                        hand_open = (1 - b_r) * grasp_h[None, :] + b_r * pregrasp_h[None, :]
                        arm_release = np.tile(descent_traj_chosen[-1, :6][None],
                                              (Nr, 1))
                        release_traj_viz = np.concatenate([arm_release, hand_open],
                                                           axis=1)
                        Nh = 20
                        b_h = np.linspace(0, 1, Nh)[:, None]
                        hand_to_init = (1 - b_h) * pregrasp_h[None, :] + b_h * init_hand_q[None, :]
                        arm_descent_end = np.tile(descent_traj_chosen[-1, :6][None],
                                                  (Nh, 1))
                        hand_init_traj = np.concatenate([arm_descent_end, hand_to_init],
                                                         axis=1)
                        xarm_init = planner._init_state[:6].astype(np.float32)
                        clear_view = xarm_init.copy()
                        clear_view[0] -= np.deg2rad(40.0)
                        cur_arm = descent_traj_chosen[-1, :6].astype(np.float32).copy()
                        joint_order = ([1, 2, 5, 0, 3, 4]
                                       if cur_arm[1] >= xarm_init[1]
                                       else [2, 1, 5, 0, 3, 4])
                        Nj = 15
                        arm_blocks = []
                        running_arm = cur_arm.copy()
                        for j in joint_order:
                            if abs(running_arm[j] - clear_view[j]) < 0.06:
                                continue
                            interp = np.linspace(running_arm[j], clear_view[j], Nj)
                            block = np.tile(running_arm, (Nj, 1))
                            block[:, j] = interp
                            arm_blocks.append(block)
                            running_arm[j] = clear_view[j]
                        arm_retract = (np.concatenate(arm_blocks, axis=0)
                                       if arm_blocks else running_arm[None].copy())
                        hand_held_init = np.tile(init_hand_q[None],
                                                 (len(arm_retract), 1))
                        retract_traj_viz = np.concatenate([arm_retract, hand_held_init],
                                                           axis=1)
                        # obj follows wrist via T_obj_in_wrist_chosen
                        T_obj_start_viz = T_obj_grasp_world_full
                        obj_approach = np.tile(T_obj_start_viz[None],
                                               (len(approach_traj_chosen), 1, 1))
                        obj_grasp = np.tile(T_obj_start_viz[None], (Nb, 1, 1))
                        ee_lift = _fk_ee(urdf_fk, lift_traj_viz)
                        obj_lift = ee_lift @ T_obj_in_wrist_chosen
                        ee_reorient = _fk_ee(urdf_fk, reorient_traj_viz)
                        obj_reorient = ee_reorient @ T_obj_in_wrist_chosen
                        ee_descent = _fk_ee(urdf_fk, descent_traj_viz)
                        obj_descent = ee_descent @ T_obj_in_wrist_chosen
                        obj_release = np.tile(obj_descent[-1][None],
                                              (len(release_traj_viz), 1, 1))
                        obj_hand_init = np.tile(obj_descent[-1][None],
                                                (len(hand_init_traj), 1, 1))
                        obj_retract = np.tile(obj_descent[-1][None],
                                              (len(retract_traj_viz), 1, 1))
                        mesh_viz_path = MESH_BASE / args.obj / "raw_mesh" / f"{args.obj}.obj"
                        vis = ViserViewer(port_number=args.port_viser)
                        vis.add_robot("xarm", str(urdf_viz_path))
                        if mesh_viz_path.exists():
                            vis.add_object("obj",
                                           trimesh.load(str(mesh_viz_path),
                                                        process=False),
                                           T_obj_start_viz)
                        vis.add_floor(height=0.0)
                        vis.add_traj("approach",  {"xarm": approach_traj_chosen}, {"obj": obj_approach})
                        vis.add_traj("grasp",     {"xarm": grasp_close_traj},     {"obj": obj_grasp})
                        vis.add_traj("lift",      {"xarm": lift_traj_viz},        {"obj": obj_lift})
                        vis.add_traj("reorient",  {"xarm": reorient_traj_viz},    {"obj": obj_reorient})
                        vis.add_traj("descent",   {"xarm": descent_traj_viz},     {"obj": obj_descent})
                        vis.add_traj("release",   {"xarm": release_traj_viz},     {"obj": obj_release})
                        vis.add_traj("hand_init", {"xarm": hand_init_traj},       {"obj": obj_hand_init})
                        vis.add_traj("retract",   {"xarm": retract_traj_viz},     {"obj": obj_retract})
                        vis.start_viewer(use_thread=True)
                        print(f"[viz] phase-by-phase viewer at "
                              f"http://localhost:{args.port_viser}")
                    except Exception as ve:
                        import traceback
                        traceback.print_exc()
                        print(f"[viz] phase viewer setup failed: {ve!r}")

                # 4. Recording start (arm/hand + video).
                raw_dir = str(cdir / "raw")
                rcc.stop()
                video_rel = os.path.join(
                    "AutoDex", "experiment", EXP_NAME,
                    sub, args.obj, trial_ts, "raw",
                )
                rcc.start("full", True, video_rel)   # "full" = video AVI + SHM stream (for snapshot_daemon)
                timestamp_monitor.start(os.path.join(raw_dir, "timestamps"))
                sync_generator.start(fps=VIDEO_FPS)
                video_started = True
                executor.start_recording(raw_dir)

                # 5. Execute (init → approach → pregrasp → grasp → squeeze).
                #    Lift is done separately in joint space via the planner
                #    (avoids _move_cartesian / set_servo_cartesian_aa).
                print(f"[cycle {cycle}] Execute (grasp + squeeze, no lift)...")
                t0 = time.time()
                try:
                    s_hand = executor.execute(result, skip_lift=True)
                except Exception as e:
                    rec["timing"]["execute_s"] = round(time.time() - t0, 2)
                    states = getattr(executor, "state_timestamps", []) or []
                    phase = states[-1]["state"] if states else "unknown"
                    rec["progress"]["execute"] = f"{phase}_failed: {e!r}"
                    rec["progress"]["charuco"] = "skipped"
                    rec["progress"]["reorient"] = "skipped"
                    rec["progress"]["descent"] = "skipped"
                    rec["progress"]["release"] = "skipped"
                    rec["progress"]["post_perception"] = "skipped"
                    rec["status"] = f"{phase}_failed"
                    rec["error"] = repr(e)
                    rec["fail_phase"] = phase
                    rec["fail_primitive"] = getattr(e, "where", None)
                    print(f"[cycle {cycle}] {phase.upper()} FAILED: {e!r} "
                          f"— reset_fallback only")

                    cleanup_errs: list = []
                    try:
                        executor.stop_recording()
                    except Exception as ce:
                        cleanup_errs.append(f"stop_recording: {ce!r}")
                    if video_started:
                        try:
                            _stop_video(rcc, sync_generator, timestamp_monitor)
                            video_started = False
                            rcc.start("stream", False, fps=STREAM_FPS)
                        except Exception as ce:
                            cleanup_errs.append(f"video_stop_or_stream: {ce!r}")
                    if cleanup_errs:
                        rec["cleanup_errors"] = cleanup_errs

                    try:
                        fb_log = executor.reset_fallback(result)
                        rec["reset"] = fb_log
                        rec["progress"]["retract"] = "fallback_after_lift_fail"
                    except Exception as fe:
                        rec["fallback_error"] = repr(fe)
                        rec["progress"]["retract"] = f"fallback_failed: {fe!r}"
                    raise _SoftSkip
                rec["timing"]["execute_s"] = round(time.time() - t0, 2)
                rec["progress"]["execute"] = "ok"
                if s_hand is not None:
                    np.save(cdir / "squeeze_hand.npy", s_hand)

                # Snapshot the constant object-in-wrist transform at grasp.
                # (Same value as T_obj_in_wrist_chosen computed pre-grasp.)
                T_obj_grasp = cart2se3(scene_cfg["mesh"]["target"]["pose"])
                T_obj_in_wrist = np.linalg.inv(result.wrist_se3) @ T_obj_grasp
                lift_traj = lift_traj_chosen   # already planned pre-grasp
                scene_lift = {"mesh": {}, "cuboid": dict(scene_cfg["cuboid"])}

                # 5b. Joint-space lift — replay pre-planned lift trajectory.
                #     Hand held at s_hand (squeeze controller units) throughout.
                print(f"[cycle {cycle}] Joint-space lift ({LIFT_HEIGHT_M*100:.0f}cm)...")
                t_exec = time.time()
                executor._log_state("lift")
                arm_traj_lift = lift_traj[:, :6]
                hand_traj_lift = np.tile(s_hand[None], (len(lift_traj), 1))
                executor._move_joints(arm_traj_lift, hand_traj_lift)
                rec["timing"]["lift_exec_s"] = round(time.time() - t_exec, 2)
                rec["progress"]["lift"] = "ok"
                print(f"    [lift] OK  exec={rec['timing']['lift_exec_s']}s "
                      f"(traj pre-planned)")

                # 6. Charuco lift-check via snapshot_daemon — pull one JPEG per
                #    camera over ZMQ from the SHM ring buffer. Video keeps
                #    recording the whole time (rcc mode "full" populates both
                #    AVI and SHM in parallel — paradex camera.py line 224-225),
                #    so we don't touch rcc/sync_generator at all here.
                print(f"[cycle {cycle}] Charuco lift-check (snapshot_daemon)...")
                t0 = time.time()
                label_abs = str(cdir / "label_at_lift" / "raw" / "images")
                _, snap_timing = snap_orch.snap(
                    n_expected=n_cams_total, timeout_s=3.0,
                    save_dir_local=label_abs,
                )
                n_jpg = len(glob.glob(os.path.join(label_abs, "*.jpg")))
                print(f"    [charuco] {n_jpg}/{n_cams_total} JPGs collected in "
                      f"{snap_timing['dispatch_to_collected_s']:.2f}s "
                      f"-> {label_abs}")
                auto_succ, label_info = auto_label_charuco(
                    label_abs, required_board=CHARUCO_BOARD)
                rec["charuco_snap"] = snap_timing
                rec["timing"]["charuco_s"] = round(time.time() - t0, 2)
                rec["charuco"] = label_info
                rec["charuco_success"] = bool(auto_succ)
                if label_info.get("reason"):
                    rec["progress"]["charuco"] = f"failed: {label_info['reason']}"
                    print(f"    [charuco] FAILED ({label_info['reason']})")
                else:
                    rec["progress"]["charuco"] = (
                        "pass" if auto_succ else
                        f"fail (covered {label_info['covered']}/{label_info['expected']})"
                    )
                    print(f"    [charuco] success={auto_succ}  "
                          f"covered {label_info['covered']}/{label_info['expected']}")

                if not auto_succ:
                    # Lift did not clear the table — drop the (failed) grasp
                    # and reset. No reorient / release / post-perception.
                    rec["progress"]["reorient"] = "skipped"
                    rec["progress"]["descent"] = "skipped"
                    rec["progress"]["release"] = "skipped"
                    rec["progress"]["post_perception"] = "skipped"
                    rec["status"] = "charuco_fail"
                    # Tear down THIS trial's recording (video + sync + ts).
                    try:
                        executor.stop_recording()
                    except Exception:
                        pass
                    if video_started:
                        try:
                            _stop_video(rcc, sync_generator, timestamp_monitor)
                            video_started = False
                        except Exception:
                            pass
                    rcc.start("stream", False, fps=STREAM_FPS)
                    try:
                        fb_log = executor.reset_fallback(result)
                        rec["reset"] = fb_log
                        rec["progress"]["retract"] = "fallback_after_charuco_fail"
                    except Exception as fe:
                        rec["fallback_error"] = repr(fe)
                        rec["progress"]["retract"] = f"fallback_failed: {fe!r}"
                    raise _SoftSkip

                # 7. Reorient — replay pre-planned reorient trajectory.
                print(f"[cycle {cycle}] Reorient (target tabletop "
                      f"file={tt_filename})  "
                      f"x={pre_info['chosen_x']:.3f}  "
                      f"yaw={np.degrees(pre_info['chosen_yaw']):.0f}°...")
                T_link6_now = executor.arm.get_data()["position"].copy()
                T_wrist_now = T_link6_now @ executor._link6_to_wrist
                reorient_traj = reorient_traj_chosen
                rec["reorient"] = {
                    "T_wrist_before": T_wrist_now.tolist(),
                    "T_wrist_lift_target": T_wrist_lift_chosen.tolist(),
                    "T_wrist_chosen": pre_info["T_wrist_target"].tolist(),
                    "R_target_robot": R_target_robot.tolist(),
                    "chosen_x": pre_info["chosen_x"],
                    "chosen_yaw_deg": float(np.degrees(pre_info["chosen_yaw"])),
                }

                t1 = time.time()
                executor._log_state("reorient")
                arm_traj = reorient_traj[:, :6]
                # Hand held at s_hand (squeeze) throughout — same as lift step.
                # plan_single_js can slightly open fingers for self-collision
                # avoidance; we ignore that and command s_hand directly.
                hand_traj = np.tile(s_hand[None], (len(reorient_traj), 1))
                executor._move_joints(arm_traj, hand_traj)
                rec["timing"]["reorient_exec_s"] = round(time.time() - t1, 2)
                rec["progress"]["reorient"] = (
                    f"ok x={pre_info['chosen_x']:.3f} "
                    f"yaw={np.degrees(pre_info['chosen_yaw']):.0f}°"
                )
                print(f"    [reorient] OK  exec={rec['timing']['reorient_exec_s']}s "
                      f"(traj pre-planned)")

                # 9. Descent — replay pre-planned descent trajectory.
                descend = LIFT_HEIGHT_M - RELEASE_HEIGHT_M
                print(f"[cycle {cycle}] Descent ({descend*100:.0f}cm)...")
                descent_traj = descent_traj_chosen
                t1 = time.time()
                executor._log_state("descent")
                arm_traj = descent_traj[:, :6]
                # Hand stays at s_hand (squeeze) — same reasoning as reorient.
                hand_traj = np.tile(s_hand[None], (len(descent_traj), 1))
                executor._move_joints(arm_traj, hand_traj)
                rec["timing"]["descent_s"] = round(time.time() - t1, 2)
                rec["progress"]["descent"] = "ok"

                # Snapshot planned object pose at the moment of release.
                T_wrist_release = (executor.arm.get_data()["position"]
                                   @ executor._link6_to_wrist)
                planned_obj_pose = T_wrist_release @ T_obj_in_wrist

                # 10. Release (squeeze -> grasp -> pregrasp).
                print(f"[cycle {cycle}] Release...")
                t0 = time.time()
                try:
                    executor.release(result)
                except Exception as re_e:
                    rec["progress"]["release"] = f"exception: {re_e!r}"
                    rec["release_error"] = repr(re_e)
                    print(f"    release FAILED: {re_e!r}")
                else:
                    rec["timing"]["release_s"] = round(time.time() - t0, 2)
                    rec["progress"]["release"] = "ok"

                # 11. Reset_hybrid: slow pregrasp→openpose interp internally,
                #     then sequential [1,2,0] + cuRobo wrist(3,4,5).
                t1 = time.time()
                try:
                    fb_log = executor.reset_hybrid(result, planner, scene_cfg)
                    rec["timing"]["retract_s"] = round(time.time() - t1, 2)
                    rec["reset"] = fb_log
                    rec["progress"]["retract"] = "hybrid"
                    rec["states"] = executor.state_timestamps
                    print(f"    release={rec['timing'].get('release_s', '?')}s  "
                          f"retract={rec['timing']['retract_s']}s  "
                          f"final_qpos_err={fb_log.get('final_qpos_err'):.4f}")
                except Exception as fb_e:
                    rec["timing"]["retract_s"] = round(time.time() - t1, 2)
                    rec["progress"]["retract"] = f"fallback_failed: {fb_e!r}"
                    rec["fallback_error"] = repr(fb_e)
                    rec["status"] = "reset_failed"
                    rec["progress"]["post_perception"] = "skipped"
                    print(f"    reset_fallback FAILED: {fb_e!r}")
                    raise _SoftSkip

                # 12. Stop recordings → video stop → image snap → stream.
                executor.stop_recording()
                _stop_video(rcc, sync_generator, timestamp_monitor)
                video_started = False
                time.sleep(0.5)

                image_rel = os.path.join(
                    "shared_data", "AutoDex", "experiment", EXP_NAME,
                    sub, args.obj, trial_ts, "post_drop_snap",
                )
                rcc.start("image", False, image_rel)
                rcc.stop()
                time.sleep(0.5)
                rcc.start("stream", False, fps=STREAM_FPS)

                if POST_DROP_SETTLE_S > 0:
                    time.sleep(POST_DROP_SETTLE_S)

                # 13. Post-drop perception + tabletop_after + drop_quality.
                print(f"[cycle {cycle}] Post-drop perception...")
                t0 = time.time()
                pose_world_after, post_timing = orch.trigger_init(
                    prompt=PROMPT,
                    save_capture_dir=str(cdir / "post_drop_capture"),
                    sil_iters=SIL_ITERS, sil_lr=SIL_LR,
                    timeout_s=POST_INIT_TIMEOUT_S,
                )
                rec["timing"]["post_perception_s"] = round(time.time() - t0, 2)
                if post_timing:
                    rec["timing"]["post_perception_detail"] = post_timing

                if pose_world_after is None:
                    reason = (post_timing or {}).get("reason", "perception_failed")
                    rec["progress"]["post_perception"] = f"failed: {reason}"
                    rec["status"] = "ok_post_perception_fail"
                else:
                    np.save(cdir / "pose_world_after_drop.npy", pose_world_after)
                    rec["progress"]["post_perception"] = "ok"
                    rec["status"] = "ok"

                    pose_robot_after = np.linalg.inv(c2r) @ pose_world_after
                    tb_after = classify_tabletop_pose(pose_robot_after, args.obj)
                    rec["tabletop_after"] = tb_after
                    rec["progress"]["tabletop_after"] = (
                        f"idx={tb_after['idx']} ({tb_after['rot_err_deg']:.1f}deg)"
                        if tb_after else "no_tabletop_data"
                    )
                    if tb_after:
                        rec["tabletop_hit_target"] = bool(
                            tb_after["idx"] == tt_sorted_idx
                        )
                        if tb_before:
                            rec["tabletop_changed"] = bool(
                                tb_after["idx"] != tb_before["idx"]
                            )
                        print(f"    [tabletop after]  idx={tb_after['idx']} "
                              f"({tb_after['filename']}) err={tb_after['rot_err_deg']:.1f}deg "
                              f"target={tt_filename} "
                              f"hit={rec['tabletop_hit_target']}")

                    R_a = planned_obj_pose[:3, :3]
                    R_b = pose_robot_after[:3, :3]
                    cos = (np.trace(R_a.T @ R_b) - 1.0) / 2.0
                    cos = float(np.clip(cos, -1.0, 1.0))
                    rot_err = float(np.degrees(np.arccos(cos)))
                    trans_err = float(np.linalg.norm(
                        planned_obj_pose[:3, 3] - pose_robot_after[:3, 3]
                    ))
                    z_drop = float(planned_obj_pose[2, 3] - pose_robot_after[2, 3])
                    rec["drop_quality"] = {
                        "planned_obj_pose_robot": planned_obj_pose.tolist(),
                        "post_drop_obj_pose_robot": pose_robot_after.tolist(),
                        "trans_err_m": trans_err,
                        "rot_err_deg": rot_err,
                        "z_drop_m": z_drop,
                    }
                    print(f"    drop_quality: trans={trans_err*1000:.1f}mm  "
                          f"rot={rot_err:.1f}deg  z={z_drop*1000:.1f}mm")

            except _SoftSkip:
                pass
            except Exception as e:
                rec["status"] = "aborted"
                rec["error"] = repr(e)
                cleanup_errs: list = []
                try:
                    executor.stop_recording()
                except Exception as ce:
                    cleanup_errs.append(f"stop_recording: {ce!r}")
                if video_started:
                    try:
                        _stop_video(rcc, sync_generator, timestamp_monitor)
                        video_started = False
                    except Exception as ce:
                        cleanup_errs.append(f"_stop_video: {ce!r}")
                if cleanup_errs:
                    rec["cleanup_errors"] = cleanup_errs
                rec["end"] = _now()
                _write_json(cdir / "result.json", rec)
                trials.append(rec)
                _write_json(summary_path, trials)
                print(f"[cycle {cycle}] ABORTED: {e!r}")
                raise

            # End-of-cycle safety cleanup.
            cycle_cleanup_errs: list = []
            if video_started:
                try:
                    _stop_video(rcc, sync_generator, timestamp_monitor)
                    video_started = False
                    rcc.start("stream", False, fps=STREAM_FPS)
                except Exception as ce:
                    cycle_cleanup_errs.append(f"video_cleanup: {ce!r}")
            if cycle_cleanup_errs:
                rec["cleanup_errors"] = (rec.get("cleanup_errors") or []) + cycle_cleanup_errs

            arm_data = executor.arm.get_data()
            rec["final_qpos"] = arm_data["qpos"].tolist()
            rec["final_arm_pose"] = arm_data["position"].tolist()

            rec["end"] = _now()
            _write_json(cdir / "result.json", rec)
            trials.append(rec)
            _write_json(summary_path, trials)

            n_ok = sum(1 for c in trials if c.get("status") == "ok")
            n_charuco_fail = sum(1 for c in trials if c.get("status") == "charuco_fail")
            n_hit = sum(1 for c in trials if c.get("tabletop_hit_target") is True)
            print(f"    cycle {cycle} done — ok: {n_ok}/{len(trials)}  "
                  f"charuco_fail: {n_charuco_fail}  hit_target: {n_hit}")
            cycle += 1
            time.sleep(CYCLE_SLEEP_S)

    finally:
        print(f"\n{'='*60}\nSUMMARY: {args.obj} × {len(trials)} trials  "
              f"target_tabletop={tt_filename}")
        for c in trials:
            tag = c.get("status", "?")
            extra = ""
            plan_s = (c.get("timing") or {}).get("plan_s")
            if plan_s is not None:
                extra += f"  plan={plan_s}s"
            dq = c.get("drop_quality")
            if dq:
                extra += f"  drop t={dq['trans_err_m']*1000:.0f}mm r={dq['rot_err_deg']:.0f}d"
            if c.get("tabletop_hit_target") is not None:
                extra += f"  hit={c['tabletop_hit_target']}"
            print(f"  {c.get('trial_ts', '?')}: {tag}{extra}")
        _write_json(summary_path, trials)
        print(f"  summary -> {summary_path}")
        _do_cleanup()


if __name__ == "__main__":
    main()
