#!/usr/bin/env python3
"""Camera-count ablation: success_only run_auto with k random cams + 2 perceptions.

Per trial:
    1. Charuco check at start. If all corners visible (no object on board) →
       wait for user Enter to place object then start.
    2. Pick k cameras uniformly at random from active set (24 cams).
    3. Run trigger_init TWICE with the same k cams (fresh capture each).
       Save both poses → offline ADD-S self-consistency.
    4. Plan/execute with the 1st pose. Approach collision (torque monitor)
       counts as failure per run_auto.
    5. Manual label + Enter to start next trial.

Always success_only. No --auto.

Prerequisites:
    bash scripts/init_daemons.sh start    # init_daemon on capture1-3, 5, 6

Usage:
    python src/experiment/num_camera/run_auto.py --obj brown_ramen --k 4
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import chime
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from paradex.io.robot_controller import get_arm, get_hand  # noqa: F401  (warm import)
from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.io.camera_system.signal_generator import UTGE900
from paradex.io.camera_system.timestamp_monitor import TimestampMonitor
from paradex.utils.system import network_info, get_pc_ip, get_camera_list
from paradex.calibration.utils import save_current_camparam, save_current_C2R, load_c2r

from autodex.utils.path import project_dir
from autodex.planner import GraspPlanner
from autodex.planner.obstacles import add_obstacles
from autodex.planner.visualizer import ScenePlanVisualizer
from autodex.executor.real import RealExecutor
from autodex.perception.init_orchestrator import InitOrchestrator

from src.execution.scene_cfg import pose_world_to_scene_cfg
from src.execution.label import auto_label_charuco

CHARUCO_BOARD = "1"


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
logging.getLogger("curobo").setLevel(logging.WARNING)

DEFAULT_PC_LIST = ["capture1", "capture2", "capture3", "capture5", "capture6"]
ASSETS_BASE = Path.home() / "shared_data/AutoDex/foundpose_assets"
MESH_BASE = Path.home() / "shared_data/AutoDex/object/paradex"
CAM_PARAM_ROOT = Path.home() / "shared_data/cam_param"




def _pick_k_serials(active_serials: List[str], k: int) -> List[str]:
    """Uniform random k-subset (without replacement)."""
    if k >= len(active_serials):
        return sorted(active_serials)
    return sorted(random.sample(active_serials, k))


def _perception_with_subset(
    orch: InitOrchestrator,
    *,
    subset: List[str],
    full_intr: Dict,
    full_extr: Dict,
    save_capture_dir: str,
    args,
) -> Tuple[Optional[np.ndarray], dict]:
    """Run trigger_init using only the given k camera serials.

    Daemons still publish all 24; we wait for ALL to arrive (n_expected=full)
    then IoU+sil runs on the chosen subset because orch.intrinsics_undist /
    extrinsics have been narrowed.
    """
    n_full = len(full_intr)
    orch.intrinsics_undist = {s: full_intr[s] for s in subset if s in full_intr}
    orch.extrinsics = {s: full_extr[s] for s in subset if s in full_extr}
    try:
        pose_world, timing = orch.trigger_init(
            prompt=args.prompt,
            save_capture_dir=save_capture_dir,
            sil_iters=args.sil_iters, sil_lr=args.sil_lr,
            timeout_s=args.init_timeout_s,
            n_expected_serials=n_full,   # wait for all 24 to publish
            sil_loss_threshold=float("inf"),  # k-cam ablation: don't reject
        )
    finally:
        orch.intrinsics_undist = full_intr
        orch.extrinsics = full_extr
    return pose_world, timing


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
    executor: RealExecutor,
    rcc,
    sync_generator,
    timestamp_monitor,
    full_intr: Dict,
    full_extr: Dict,
    active_serials: List[str],
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

    # ── 2. Camera subset + TWO perception runs ─────────────────────────────
    chosen = _pick_k_serials(active_serials, args.k)
    print(f"[2/6] Perception (k={args.k}, 2x trigger_init)...")
    print(f"    chosen serials: {chosen}")
    with open(os.path.join(img_dir, "chosen_serials.json"), "w") as f:
        json.dump({"k": args.k, "serials": chosen,
                   "active": sorted(active_serials)}, f, indent=2)
    timing["perception_start"] = _ts()
    timing["k"] = args.k
    timing["chosen_serials"] = chosen

    poses_two: List[Optional[np.ndarray]] = []
    perc_timings: List[dict] = []
    for run_idx in (1, 2):
        t0 = time.time()
        save_capture_dir = os.path.join(img_dir, f"init_capture_{run_idx}")
        pose_world, perc_timing = _perception_with_subset(
            orch, subset=chosen,
            full_intr=full_intr, full_extr=full_extr,
            save_capture_dir=save_capture_dir, args=args,
        )
        dt = round(time.time() - t0, 2)
        print(f"    perception #{run_idx}: {dt}s  ok={pose_world is not None}")
        perc_timings.append({"run": run_idx, "s": dt, "detail": perc_timing})
        poses_two.append(pose_world)
        if pose_world is not None:
            np.save(os.path.join(img_dir, f"pose_world_{run_idx}.npy"), pose_world)
    timing["perception_detail"] = perc_timings

    pose_world = poses_two[0]
    if pose_world is None:
        reason = (perc_timings[0].get("detail") or {}).get("reason", "perception_failed")
        print(f"    Perception #1 FAILED ({reason}) — aborting trial")
        chime.error()
        try:
            input("    [perception_failed] Fix the scene then press Enter to "
                  "continue (Ctrl-C to abort)... ")
        except KeyboardInterrupt:
            raise
        fail = {"dir_idx": dir_idx, "scene_type": args.scene, "success": False,
                "reason": reason, "k": args.k, "chosen_serials": chosen,
                "timing": timing}
        with open(os.path.join(img_dir, "result.json"), "w") as f:
            json.dump(fail, f, indent=2, default=str)
        return _stamp_end(fail)
    np.save(os.path.join(img_dir, "pose_world.npy"), pose_world)

    # ── 3. scene_cfg + plan ──────────────────────────────────────────────────
    print(f"[3/6] Planning (version={args.grasp_version}, scene={args.scene})...")
    timing["planning_start"] = _ts()
    t0 = time.time()
    c2r = load_c2r(img_dir)
    scene_cfg = pose_world_to_scene_cfg(pose_world, c2r, obj)
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
    tb_before = classify_tabletop_pose(pose_robot, obj)
    timing["tabletop_before"] = tb_before
    if tb_before is not None:
        scene_id = str(tb_before["idx"])
        pose_stem = tb_before["filename"].replace(".npy", "")
        print(f"    [tabletop] idx={tb_before['idx']} "
              f"({tb_before['filename']}) err={tb_before['rot_err_deg']:.1f}°")
        # Camera-ablation: do NOT skip scenes that already have a success —
        # we WANT to retest those validated grasps under sparse-view perception.
    else:
        scene_id = None
        pose_stem = None
        print(f"    [tabletop] UNCLASSIFIED — aborting trial "
              f"(grasp 의 scene_id 일치 보장 못함)")
        fail = {"dir_idx": dir_idx, "scene_type": args.scene, "success": False,
                "reason": "tabletop_unclassified", "k": args.k,
                "chosen_serials": chosen, "timing": timing}
        with open(os.path.join(img_dir, "result.json"), "w") as f:
            json.dump(fail, f, indent=2, default=str)
        return _stamp_end(fail)
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
    if args.grasp_version == "v7":
        _plan_scene_id = None
        _plan_scene_type_filter = None
    else:
        _plan_scene_id = scene_id
        _plan_scene_type_filter = None
    # Camera-ablation: re-test validated grasps → success_only=True with
    # skip_done=False / skip_scenes_with_success=False so we pick up
    # previously-succeeded candidates again.
    result = planner.plan(
        scene_cfg, obj, args.grasp_version,
        skip_done=False,
        success_only=True, hand=hand,
        scene_id=_plan_scene_id,
        scene_type_filter=_plan_scene_type_filter,
        skip_scenes_with_success=False,
        openpose_pose_stem=pose_stem,
        cyl_axis_local=_cyl_axis,
        cyl_yaw_grid=_cyl_grid,
    )
    timing["plan_s"] = round(time.time() - t0, 2)
    print(f"    Plan: {timing['plan_s']}s  success={result.success}")

    if not result.success:
        # n_total == 0 means load_candidate returned empty — every scene is
        # already done (skip_scenes_with_success filtered them all out).
        # Surface this so the outer cycle loop can stop.
        n_total = (result.timing or {}).get("n_total", -1)
        if n_total == 0:
            print(f"    All scenes for {obj} ({args.grasp_version}) already "
                  f"have a successful grasp — nothing left to try.")
            done = {"dir_idx": dir_idx, "scene_type": args.scene,
                    "success": None, "reason": "all_scenes_done",
                    "all_done": True, "timing": timing}
            with open(os.path.join(img_dir, "result.json"), "w") as f:
                json.dump(done, f, indent=2, default=str)
            return _stamp_end(done)
        print("    Planning FAILED — launching visualizer to inspect...")
        # Match planner.plan()'s actual candidate pool: skip_done=True and
        # skip_scenes_with_success=True so the viewer shows only what was
        # actually attempted (not the full disk pool).
        wrist_se3, _, grasp_pose, filtered, ik_failed = planner.get_candidates(
            scene_cfg, obj, args.grasp_version,
            success_only=True,
            skip_done=False, hand=hand, run_ik=True,
            scene_id=_plan_scene_id,
            scene_type_filter=_plan_scene_type_filter,
            cyl_axis_local=_cyl_axis,
            cyl_yaw_grid=_cyl_grid,
            skip_scenes_with_success=False,
        )
        fv = ScenePlanVisualizer(scene_cfg, None, port=8080, hand=hand)
        fv.add_candidates(wrist_se3, grasp_pose, filtered, ik_failed=ik_failed)
        fv.start_viewer(use_thread=True)
        _active_vis = fv
        chime.error()
        # All remaining candidates failed → treat as exhausted, stop cycle.
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

    if args.viz:
        print("    Launching visualizer (http://localhost:8080)...")
        sv = ScenePlanVisualizer(scene_cfg, result, port=8080, hand=hand)
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
    raw_dir = os.path.join(img_dir, "raw")
    rcc.start("video", True, raw_rel)
    timestamp_monitor.start(os.path.join(raw_dir, "timestamps"))
    executor.start_recording(raw_dir)
    sync_generator.start(fps=30)

    t0 = time.time()
    s_hand = executor.execute(result)              # grasp + lift (torque mon)

    # ── 5. Auto-label at LIFT via charuco ────────────────────────────────────
    timing["label_start"] = _ts()
    print(f"[5/6] Auto-label (charuco at lift)")
    try: rcc.stop()
    except Exception: pass
    label_rel = os.path.join("shared_data", "AutoDex", "experiment",
                             args.exp_name, sub, obj, dir_idx, "label", "raw")
    label_abs = os.path.join(img_dir, "label", "raw", "images")
    os.makedirs(label_abs, exist_ok=True)
    rcc.start("image", False, label_rel)
    rcc.stop()
    # Poll up to 5s for images to land on NFS.
    for _ in range(50):
        try:
            if any(Path(label_abs).iterdir()):
                break
        except FileNotFoundError:
            pass
        time.sleep(0.1)
    succ, label_info = auto_label_charuco(label_abs, required_board=CHARUCO_BOARD)
    timing["auto_label"] = label_info
    if label_info.get("reason"):
        note = label_info["reason"]
        print(f"    [auto-label] FAIL ({note}) — treating as fail")
        succ = False
    else:
        note = (f"charuco covered {label_info.get('covered')}/"
                f"{label_info.get('expected')}")
        print(f"    [auto-label] success={succ}  {note}")
    # Camera-ablation: skip place. Go straight to release + sequential retract.
    timing["execute_s"] = round(time.time() - t0, 2)
    timing["execution_states"] = executor.state_timestamps
    timestamp_monitor.stop()
    sync_generator.stop()

    # ── 6. Release & save ────────────────────────────────────────────────────
    print(f"[6/6] Releasing (slow)...")
    executor.release(result, slow_factor=2.0)

    # reset_hybrid now does the slow pregrasp→openpose interp internally,
    # then keeps hand at openpose during sequential [1,2,0] + cuRobo wrist.
    #     replan around placed object, back to XARM_INIT.
    try:
        fb_log = executor.reset_hybrid(result, planner, scene_cfg)
        timing["retract"] = fb_log
        print(f"    retract OK  final_qpos_err={fb_log.get('final_qpos_err'):.4f}")
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
        "success": succ,
        "k": args.k,
        "chosen_serials": chosen,
        "scene_info": result.scene_info,
        "candidate_idx": result.timing.get("candidate_idx") if result.timing else None,
        "tabletop_before": tb_before,
        "timing": timing,
    }
    if note is not None:
        trial_result["note"] = note
    with open(os.path.join(img_dir, "result.json"), "w") as f:
        json.dump(trial_result, f, indent=2, default=str)
    # Camera-ablation: do NOT overwrite candidate dir result.json.

    status = "SUCCESS" if succ else ("ISSUE" if succ is None else "FAIL")
    print(f"    Result: {status}  saved to {img_dir}/result.json")

    # Resume the stream so the next trial's init has live SHM frames.
    rcc.start("stream", False, fps=args.stream_fps)

    return _stamp_end(trial_result)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--k", type=int, required=True,
                        help="Number of cameras to use for perception (e.g. 2,4,8,16).")
    parser.add_argument("--grasp_version", type=str, default="selected_100")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Defaults to num_camera_k{k}.")
    parser.add_argument("--hand", type=str, default="allegro",
                        choices=["allegro", "inspire", "inspire_left"])
    parser.add_argument("--scene", type=str, default="table",
                        choices=["table", "wall", "shelf", "cluttered"])
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for k-cam selection (optional).")
    parser.add_argument("--viz", action="store_true")

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
    parser.add_argument("--prompt", type=str, default="object on the checkerboard")
    parser.add_argument("--sil_iters", type=int, default=100)
    parser.add_argument("--sil_lr", type=float, default=0.002)
    parser.add_argument("--init_timeout_s", type=float, default=120.0)
    parser.add_argument("--calib_dir", type=str, default=None,
                        help="Camera calib dir. Default: latest under ~/shared_data/cam_param/.")
    parser.add_argument("--stream_fps", type=int, default=10)
    parser.add_argument("--stream_warmup_s", type=float, default=2.0)

    args = parser.parse_args()
    if args.exp_name is None:
        args.exp_name = f"num_camera_k{args.k}"
    if args.seed is not None:
        random.seed(args.seed)

    # scene_prefix: '' (table), 'wall', 'shelf', 'cluttered' + always success_only.
    scene_prefix = args.scene if args.scene != "table" else ""
    scene_prefix = f"{scene_prefix}_success_only" if scene_prefix else "success_only"

    # Mesh / FoundPose assets sanity check.
    mesh_path = MESH_BASE / args.obj / "raw_mesh" / f"{args.obj}.obj"
    assets_root = ASSETS_BASE / args.obj
    if not mesh_path.exists():
        sys.exit(f"mesh not found: {mesh_path}")
    if not (assets_root / "object_repre/v1" / args.obj / "1/repre.pth").exists():
        sys.exit(f"repre.pth missing for {args.obj} (expected under {assets_root})")

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
    active_serials_list = sorted(active_serials)
    print(f"  {len(intrinsics_full)} cams active across {len(args.pc_list)} PCs  ({H}x{W})")
    if args.k > len(active_serials_list):
        sys.exit(f"--k {args.k} > active cams {len(active_serials_list)}")

    # Hardware init.
    rcc = remote_camera_controller("run_auto", pc_list=args.pc_list)
    sync_generator = UTGE900(**network_info["signal_generator"]["param"])
    timestamp_monitor = TimestampMonitor(**network_info["timestamp"]["param"])

    print(f"[stream] starting on {len(args.pc_list)} PCs @ {args.stream_fps} FPS...")
    rcc.start("stream", False, fps=args.stream_fps)
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

    # Snapshot full per-serial K_undist + extrinsics for k-subset masking.
    full_intr = dict(orch.intrinsics_undist)
    full_extr = dict(orch.extrinsics)

    print("[planner] warming up...")
    planner = GraspPlanner(hand=args.hand)
    print("[executor] connecting to robot...")
    executor = RealExecutor(hand_name=args.hand)

    def _cleanup():
        print("\n[cleanup] Stopping hardware...")
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
            try:
                cmd = input("Press Enter to start trial, 'q' to quit: ").strip().lower()
            except KeyboardInterrupt:
                _cleanup()
                break
            if cmd == "q":
                break

            tr = run_single_trial(
                args, scene_prefix=scene_prefix,
                orch=orch, planner=planner, executor=executor,
                rcc=rcc, sync_generator=sync_generator,
                timestamp_monitor=timestamp_monitor,
                full_intr=full_intr, full_extr=full_extr,
                active_serials=active_serials_list,
            )
            results.append(tr)
            n_succ = sum(1 for r in results if r.get("success"))
            print(f"\n    Running total: {n_succ}/{len(results)} success")
            if tr.get("all_done"):
                print(f"\n    All scenes done for {args.obj} — stopping loop.")
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
