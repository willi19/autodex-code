#!/usr/bin/env python3
"""Continuous, fixed-catalog pick-to-basket demo.

This is the deliberately small first demo loop, built from Mingi's current
distributed FoundPose/ParaDex/FR3 path rather than the legacy experiment loop.
It is designed for one object in the pick workspace at a time, selected from a
fixed *pre-onboarded* catalogue.  It never asks a human to label a trial and
does not home-reset after a miss: a verified empty grasp re-observes the same
object and replans from the arm's measured raised configuration.

Example (after pre-onboarding all listed objects):

    python src/demo/continuous_basket/run_demo.py \
      --objects banana=banana wood_organizer='wood organizer' \
      --max-successes 12

The basket release reference is either supplied manually or measured once at
startup from a standalone ArUco marker.  With a marker, use the local marker
offset to point from the marker centre to the safe release point over the
basket's open interior.  Start with an empty basket, a clear approach from
above, and an external continuous camera for the uncut take.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
# The deployment commonly keeps the latest ParaDex checkout beside AutoDex
# rather than installing it into every specialised CUDA environment. Prefer an
# explicit task-scoped override, then that standard checkout; leave normal
# installed-package resolution untouched when neither exists.
for _paradex_root in (
    os.environ.get("AUTODEX_PARADEX_ROOT"),
    str(Path.home() / "paradex"),
):
    if _paradex_root and (Path(_paradex_root).expanduser() / "paradex").is_dir():
        sys.path.insert(0, str(Path(_paradex_root).expanduser()))
        break

from paradex.calibration.utils import load_c2r, save_current_C2R, save_current_camparam
from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.utils.system import get_camera_list, get_pc_ip

from autodex.perception.init_orchestrator import InitOrchestrator
from autodex.planner import GraspPlanner
from autodex.planner.obstacles import TABLE_CUBOID, add_obstacles
from autodex.planner.planner import _to_curobo_world
from autodex.utils.conversion import cart2se3
from autodex.utils.path import get_obj_root, project_dir
from autodex.utils.symmetry import get_cyl_axis_local, get_cyl_yaw_grid

from src.demo.banana_test.run_demo import (
    ASSETS_BASE,
    CAM_PARAM_ROOT,
    DEFAULT_PC_LIST,
    _clear_camera_errors,
    _ensure_camera_lock,
    _fk_wrist,
    _place_wrist,
    _planner_robot,
    _rcc_start,
    _safe,
    _warn_if_not_streaming,
    _wrist_now,
    filter_by_place_reach,
)
from src.demo.banana_test.success_grasps import success_keys_at_pose
from src.demo.banana_test.place_target import locate_marker
from src.demo.continuous_basket.basket_marker import (
    DEFAULT_BASKET_MARKER_ID,
    release_reference_from_marker,
)
from src.demo.continuous_basket.catalog import (
    CatalogObject,
    CatalogRecognizer,
    CatalogMatch,
    parse_catalog,
    read_capture_images,
    require_catalog_runtime,
    single_object_match,
)
from src.demo.continuous_basket.camera import capture_catalog_snapshot
from src.demo.continuous_basket.policy import (
    LocalRetryPolicy,
    PoseEvidence,
    PoseVerifier,
    Verification,
    choose_success_candidates,
)
from src.demo.continuous_basket.preflight import build_report, require_ready
from src.demo.continuous_basket.tracking import LiveGoTrackSession
from src.execution.scene_cfg import pose_world_to_scene_cfg
from src.experiment.reset.tabletop_pose import classify_tabletop_pose


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(v) for v in value]
    return value


def _planner_full_qpos(planner: GraspPlanner, executor, fallback_hand: np.ndarray) -> np.ndarray:
    """Read the physical start state used by the next, non-reset plan."""
    arm = np.asarray(executor.arm.get_data()["qpos"][:planner._n_arm], dtype=np.float32)
    hand = np.asarray(getattr(executor, "_last_hand_qpos", fallback_hand), dtype=np.float32)
    full = np.concatenate([arm, hand])
    if len(full) != len(planner._init_state):
        raise RuntimeError(
            f"robot state shape {len(full)} does not match planner ({len(planner._init_state)})"
        )
    return full


def _object_paths(item: CatalogObject, grasp_version: str) -> tuple[Path, Path]:
    root = Path(get_obj_root(grasp_version))
    mesh = root / item.name / "raw_mesh" / f"{item.name}.obj"
    # FoundPose assets currently live in the ParaDex asset tree even for v8
    # grasps.  The mesh used by pose_scene_cfg follows get_obj_root() above.
    assets = ASSETS_BASE / item.name
    repre = assets / "object_repre" / "v1" / item.name / "1" / "repre.pth"
    if not mesh.is_file():
        raise FileNotFoundError(f"catalog mesh missing for {item.name}: {mesh}")
    if not repre.is_file():
        raise FileNotFoundError(
            f"FoundPose assets missing for {item.name}: {repre}; onboard before the demo"
        )
    return mesh, assets


def _setup_object(
    orch: InitOrchestrator,
    item: CatalogObject,
    args,
    intrinsics: Dict[str, Dict[str, Any]],
    extrinsics: Dict[str, np.ndarray],
    image_hw: tuple[int, int],
    pc_serials: Dict[str, list[str]],
) -> None:
    mesh, assets = _object_paths(item, args.grasp_version)
    # quality mode makes renderer/optimizer construction unnecessary.
    orch.init_object(
        obj_name=item.name, mesh_path=str(mesh), assets_root=str(assets),
        intrinsics_full=intrinsics, extrinsics_full=extrinsics, image_hw=image_hw,
        mode="live", pc_serials=pc_serials, load_silhouette=False,
    )
    # Init is sent asynchronously by ParaDex.  This bounded settling window
    # prevents a stale object model on an individual capture PC after a
    # catalogue switch, without resetting the robot or camera daemons.
    time.sleep(args.object_switch_settle_s)


def _observe_fast(orch: InitOrchestrator, item: CatalogObject, out_dir: Path,
                  args, c2r: np.ndarray,
                  robot_bounds: Optional[np.ndarray] = None) -> tuple[Optional[np.ndarray], dict]:
    """Fast-init one selected object, optionally restricted to a robot ROI.

    The basket accumulates seen objects.  Restricting the candidate *pose* to
    the pick workspace is what makes a known class at the basket distinct from
    another instance of the same class placed for the next cycle.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    masks, poses, collect_timing = orch.collect_payloads(
        prompt=item.prompt, timeout_s=args.init_timeout_s,
        save_capture_dir=str(out_dir),
    )
    rejected = []
    if robot_bounds is not None:
        lo, hi = np.asarray(robot_bounds, dtype=float).reshape(2, 3)
        kept = {}
        for serial, payload in poses.items():
            pose_world = payload.get("pose_world") if payload.get("ok") else None
            if pose_world is None:
                continue
            xyz = (np.linalg.inv(c2r) @ np.asarray(pose_world))[:3, 3]
            if np.all(xyz >= lo) and np.all(xyz <= hi):
                kept[serial] = payload
            else:
                rejected.append({"serial": serial, "xyz_robot": xyz.tolist()})
        poses = kept
    pose, select_timing = orch.refine_from_payloads(
        masks, poses, sil_iters=0, selection_mode="quality",
    )
    timing = {**collect_timing, **select_timing,
              "selection_mode": "quality", "sil_skipped": True,
              "outside_workspace": rejected}
    if pose is not None:
        np.save(out_dir / "pose_world.npy", pose)
    return pose, timing


def _candidate_order(item: CatalogObject, hand: str, version: str,
                     pose_robot: np.ndarray, arm: str,
                     strict_tabletop: bool) -> tuple[list, Optional[str], dict]:
    obj_root = get_obj_root(version)
    tabletop = classify_tabletop_pose(pose_robot, item.name, obj_root)
    stem = tabletop["filename"].replace(".npy", "") if tabletop else None
    at_pose, any_pose = success_keys_at_pose(item.name, hand, version, stem, arm=arm)
    candidates, source = choose_success_candidates(
        at_pose, any_pose, strict_tabletop=strict_tabletop,
    )
    info = dict(tabletop or {})
    info["candidate_source"] = source
    info["matched_tabletop_successes"] = len(at_pose)
    info["other_tabletop_successes"] = len(any_pose)
    return list(candidates), stem, info


def _plan_attempt(planner: GraspPlanner, executor, item: CatalogObject, candidate_order: Sequence,
                  pose_world: np.ndarray, c2r: np.ndarray, basket_xyz: np.ndarray, args):
    """Plan grasp + preflight lift/carry, starting at the measured arm pose."""
    hand = args.hand
    scene_cfg = add_obstacles(
        pose_world_to_scene_cfg(pose_world, c2r, item.name, get_obj_root(args.grasp_version)),
        "table",
    )
    pose_robot = np.linalg.inv(c2r) @ pose_world
    order, tabletop_stem, tabletop = _candidate_order(item, hand, args.grasp_version,
                                                       pose_robot, args.arm,
                                                       args.strict_tabletop_success)
    # Keep only catalog/pose-compatible successful grasps.  On a retry the
    # policy removes the candidate that just missed, while retaining the same
    # object pose and current arm state.
    allowed = [tuple(k) for k in candidate_order] if candidate_order else order
    if not allowed:
        return None, {"reason": "no_success_grasp_at_tabletop", "tabletop": tabletop_stem}, None

    T_obj = cart2se3(scene_cfg["mesh"]["target"]["pose"])
    if getattr(planner, "_ik_solver", None) is None:
        planner._init_ik_solver(_to_curobo_world({"mesh": {}, "cuboid": {"table": TABLE_CUBOID}}))
    yaws = np.arange(0, 360, args.yaw_step, dtype=float)
    reachable = filter_by_place_reach(planner, allowed, item.name, hand,
                                      args.grasp_version, T_obj, basket_xyz[:2], yaws)
    if reachable:
        allowed = reachable

    planner.set_start_state(_planner_full_qpos(planner, executor, planner._init_state[planner._n_arm:]))
    result = planner.plan(
        scene_cfg, item.name, args.grasp_version, skip_done=False,
        success_only=False, hand=hand, scene_id=None, scene_type_filter=None,
        skip_scenes_with_success=False, openpose_pose_stem=tabletop_stem,
        cyl_axis_local=get_cyl_axis_local(item.name), cyl_yaw_grid=get_cyl_yaw_grid(item.name),
        candidate_order=allowed,
    )
    if not result.success:
        return None, {"reason": "plan_failed", "planner": result.timing,
                      "tabletop": tabletop_stem, "candidate_order": allowed}, None

    # Keep this key even when the following dry-run fails: it lets the local
    # retry policy remove the bad candidate before the robot ever moves.
    selected_key = tuple(result.scene_info)

    # Preflight before squeezing: a grasp is only allowed if a raised carry to
    # the basket has a collision-checked trajectory.  This prevents the old
    # failure mode of discovering an unreachable drop only after holding it.
    grasp_end = np.asarray(result.traj[-1], dtype=np.float32)
    T_wrist_grasp = _fk_wrist(planner, grasp_end)
    T_obj_in_wrist = np.linalg.inv(T_wrist_grasp) @ T_obj

    def full(q_arm: np.ndarray) -> np.ndarray:
        return np.concatenate([np.asarray(q_arm[:planner._n_arm], dtype=np.float32),
                               np.asarray(result.grasp_pose, dtype=np.float32)])

    lift_wrist = T_wrist_grasp.copy()
    lift_wrist[2, 3] += args.lift_height
    lift_traj = planner.plan_pose_constrained(
        full(grasp_end), lift_wrist, hold_vec_weight=[1, 1, 1, 1, 1, 0],
        scene_cfg=scene_cfg, include_obj_obstacle=False,
    )
    if lift_traj is None:
        return None, {"reason": "lift_plan_failed", "tabletop": tabletop_stem,
                      "candidate": selected_key}, None
    lift_end = np.asarray(lift_traj[-1], dtype=np.float32)
    T_wrist_lift = _fk_wrist(planner, lift_end)
    obj_z_lift = float((T_wrist_lift @ T_obj_in_wrist)[2, 3])

    carry_candidates = np.array([
        _place_wrist(T_obj, T_obj_in_wrist, basket_xyz, yaw, obj_z_lift,
                     float(T_wrist_lift[2, 3]))
        for yaw in yaws
    ])
    feasible = np.asarray(planner.ik_pose_batch(carry_candidates)).reshape(-1)
    if not feasible.any():
        return None, {"reason": "basket_ik_infeasible", "tabletop": tabletop_stem,
                      "candidate": selected_key}, None
    yaw = float(yaws[np.flatnonzero(feasible)[0]])
    carry_target = _place_wrist(T_obj, T_obj_in_wrist, basket_xyz, yaw, obj_z_lift,
                                 float(T_wrist_lift[2, 3]))
    carry_traj = planner.plan_pose_constrained(
        full(lift_end), carry_target, hold_vec_weight=[0, 0, 0, 0, 0, 1],
        scene_cfg=scene_cfg, include_obj_obstacle=False,
    )
    if carry_traj is None:
        return None, {"reason": "carry_plan_failed", "tabletop": tabletop_stem,
                      "candidate": selected_key}, None
    return result, {
        "tabletop": tabletop, "tabletop_stem": tabletop_stem,
        "candidate_order": allowed, "candidate": selected_key,
        "place_yaw_deg": yaw, "scene_cfg": scene_cfg,
    }, {"lift_traj": lift_traj, "carry_traj": carry_traj,
         "object_in_wrist": T_obj_in_wrist, "place_yaw": yaw}


def _retreat_up(planner: GraspPlanner, executor, hand_qpos: np.ndarray,
                scene_cfg: dict, height: float) -> bool:
    wrist = _wrist_now(planner, executor, planner._n_arm, hand_qpos)
    wrist[2, 3] += height
    start = _planner_full_qpos(planner, executor, hand_qpos)
    traj = planner.plan_pose_constrained(
        start, wrist, hold_vec_weight=[1, 1, 1, 1, 1, 0], scene_cfg=scene_cfg,
        include_obj_obstacle=False,
    )
    if traj is None:
        return False
    cmd = np.asarray(executor._convert(hand_qpos.astype(np.float64)), dtype=np.float64)
    executor._move_joints(traj[:, :planner._n_arm], np.tile(cmd, (len(traj), 1)))
    return True


def _write_trial(run_dir: Path, record: dict) -> None:
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = run_dir / "trials" / stamp
    out.mkdir(parents=True, exist_ok=False)
    (out / "result.json").write_text(json.dumps(_jsonable(record), indent=2))


def _pose_in_bounds(pose_world: np.ndarray, c2r: np.ndarray, bounds: np.ndarray) -> bool:
    lo, hi = np.asarray(bounds, dtype=float).reshape(2, 3)
    xyz = (np.linalg.inv(c2r) @ np.asarray(pose_world, dtype=float))[:3, 3]
    return bool(np.all(xyz >= lo) and np.all(xyz <= hi))


def _track_timing(sample, *, source: str = "gotrack") -> dict:
    if sample is None:
        return {"source": source, "ok": False}
    return {"source": source, "ok": True, "frame_id": sample.frame_id,
            "n_inliers": sample.n_inliers,
            "mean_residual_mm": sample.mean_residual_mm}


def _measure_basket_reference(rcc, args, run_dir: Path, expected_serials: Iterable[str]) -> tuple[np.ndarray, dict]:
    """Capture and triangulate the standalone basket marker before arm setup."""
    marker_dir = run_dir / "basket_marker"
    image_count = capture_catalog_snapshot(
        rcc, marker_dir, min_images=args.basket_marker_min_views,
        settle_timeout_s=args.basket_marker_snapshot_timeout_s,
        expected_serials=expected_serials, require_decodable=True,
    )
    # ``locate_marker`` consumes the per-capture camera/calibration sidecars,
    # matching the legacy banana place-target path.
    save_current_C2R(str(marker_dir))
    save_current_camparam(str(marker_dir))
    info = locate_marker(
        str(marker_dir), dict_type=args.basket_marker_dict,
        marker_id=args.basket_marker_id,
    )
    reference = release_reference_from_marker(
        info["center_robot"], info["pose_robot"], args.basket_marker_offset,
    )
    record = {
        "source": "aruco_marker",
        "marker": info,
        "marker_offset_m": np.asarray(args.basket_marker_offset, dtype=float),
        "release_reference_robot": reference,
        "snapshot_images": image_count,
    }
    print(f"[basket] marker {args.basket_marker_dict} id={info['marker_id']} "
          f"({info['n_views']} views) release_robot={reference.round(4)}")
    return reference, record


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--objects", nargs="+", required=True,
                   help="fixed catalogue: object_name or object_name=YOLO-E prompt")
    p.add_argument("--hand", default="inspire", choices=["allegro", "inspire", "inspire_left"])
    p.add_argument("--arm", default="franka", choices=["xarm", "franka"])
    p.add_argument("--grasp-version", default="v8")
    p.add_argument("--strict-tabletop-success", action="store_true",
                   help="disable object-frame fallback to successes recorded at other stable poses")
    basket_source = p.add_mutually_exclusive_group()
    basket_source.add_argument("--basket-center", nargs=3, type=float, metavar=("X", "Y", "Z"),
                               help="manual robot-frame basket release reference, metres")
    basket_source.add_argument("--basket-marker-id", type=int, metavar="ID",
                               default=DEFAULT_BASKET_MARKER_ID,
                               help="standalone 6X6_1000 ArUco ID fixed to the basket "
                                    f"(default: legacy marker {DEFAULT_BASKET_MARKER_ID})")
    p.add_argument("--basket-marker-dict", default="6X6_1000",
                   help="OpenCV dictionary for --basket-marker-id")
    p.add_argument("--basket-marker-offset", nargs=3, type=float, default=[0.0, 0.0, 0.0],
                   metavar=("DX", "DY", "DZ"),
                   help="marker-local metres from marker centre to release reference")
    p.add_argument("--basket-marker-min-views", type=int, default=3,
                   help="minimum NAS-visible marker snapshot views (must be >= 3)")
    p.add_argument("--basket-marker-snapshot-timeout-s", type=float, default=15.0,
                   help="marker snapshot maximum wait before any robot connection")
    p.add_argument("--pick-workspace", nargs=6, type=float,
                   default=[0.35, -0.30, 0.00, 0.80, 0.21, 0.45],
                   metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
                   help="robot-frame box used to distinguish the next object from basket contents")
    p.add_argument("--basket-observe-radius", type=float, default=0.14,
                   help="xy half-width of post-drop pose verification region (m)")
    p.add_argument("--basket-observe-height", type=float, default=0.30,
                   help="z extent above --basket-center for post-drop verification (m)")
    p.add_argument("--max-successes", type=int, default=12)
    p.add_argument("--max-cycles", type=int, default=40,
                   help="safety cap including failed/retried objects")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--pc-list", nargs="+", default=DEFAULT_PC_LIST)
    p.add_argument("--catalog-min-views", type=int, default=2)
    p.add_argument("--catalog-min-score", type=float, default=0.25)
    p.add_argument("--catalog-gpu", type=int, default=0)
    p.add_argument("--catalog-snapshot-timeout-s", type=float, default=15.0,
                   help="maximum wait for the requested NAS-visible snapshot views; returns early when ready")
    p.add_argument("--init-timeout-s", type=float, default=10.0)
    p.add_argument("--init-command-timeout-s", type=float, default=5.0,
                   help="per-capture-PC init-daemon command deadline; fail before robot motion")
    p.add_argument("--verification-mode", choices=["gotrack", "foundpose"], default="gotrack",
                   help="gotrack keeps normal cycles under the 20s inference target; foundpose is a daemon-free fallback")
    p.add_argument("--tracking-timeout-s", type=float, default=1.5,
                   help="max wait for a post-action GoTrack pose")
    p.add_argument("--tracking-warmup-s", type=float, default=3.0,
                   help="max wait for first GoTrack pose after a FoundPose init")
    p.add_argument("--tracking-command-timeout-s", type=float, default=3.0,
                   help="per-capture-PC GoTrack daemon command deadline")
    p.add_argument("--tracking-anchor-root", default=str(
        Path(__file__).resolve().parents[3] / "autodex/perception/thirdparty/MV-GoTrack/anchor_banks"),
                   help="per-object GoTrack .npz anchor-bank directory")
    p.add_argument("--port-track-obs", type=int, default=1235)
    p.add_argument("--port-track-prior", type=int, default=1236)
    p.add_argument("--port-track-cmd", type=int, default=6892)
    p.add_argument("--track-min-cams", type=int, default=6)
    p.add_argument("--track-min-inliers", type=int, default=12)
    p.add_argument("--object-switch-settle-s", type=float, default=1.0)
    p.add_argument("--stream-fps", type=int, default=10)
    p.add_argument("--run-id", default=None,
                   help="unique result session name (default: current timestamp; prevents stale snapshots)")
    p.add_argument("--yaw-step", type=int, default=30)
    p.add_argument("--lift-height", type=float, default=0.10)
    p.add_argument("--drop-height", type=float, default=0.05)
    p.add_argument("--retreat-height", type=float, default=0.15)
    p.add_argument("--exp-name", default="continuous_basket_demo")
    args = p.parse_args()
    if (args.max_successes < 1 or args.max_retries < 1 or args.yaw_step < 1
            or args.tracking_timeout_s <= 0 or args.tracking_warmup_s <= 0
            or args.init_command_timeout_s <= 0 or args.tracking_command_timeout_s <= 0
            or args.catalog_snapshot_timeout_s <= 0
            or args.basket_marker_snapshot_timeout_s <= 0
            or args.basket_marker_min_views < 3):
        p.error("retry/count/timing arguments must be positive")

    catalogue = parse_catalog(args.objects)
    single_object_mode = len(catalogue) == 1
    # A YOLO-E class scan only has value when there are two or more known
    # classes to distinguish. The single-object banana bring-up goes straight
    # to FoundPose and therefore needs neither ultralytics nor its checkpoint.
    if not single_object_mode:
        try:
            require_catalog_runtime()
        except RuntimeError as exc:
            p.error(str(exc))
    # Do this before opening camera/robot sessions: every catalogue entry must
    # be demo-ready even when it is selected only after several successes.
    readiness = build_report(
        catalogue, object_root=Path(get_obj_root(args.grasp_version)),
        assets_base=ASSETS_BASE,
        candidate_root=Path(project_dir) / "candidates" / args.hand / args.grasp_version,
        anchor_root=Path(args.tracking_anchor_root).expanduser(),
        require_gotrack=args.verification_mode == "gotrack", arm=args.arm,
    )
    try:
        require_ready(readiness)
    except RuntimeError as exc:
        p.error(str(exc))
    pick_bounds = np.asarray(args.pick_workspace, dtype=np.float64).reshape(2, 3)
    if np.any(pick_bounds[1] <= pick_bounds[0]):
        p.error("pick-workspace max bounds must exceed min bounds")
    run_id = args.run_id or dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (Path(project_dir) / "experiment" / args.exp_name
               / f"{args.arm}_{args.hand}" / run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    save_current_C2R(str(run_dir)); save_current_camparam(str(run_dir))
    c2r = load_c2r(str(run_dir))

    calib_dir = sorted(Path(CAM_PARAM_ROOT).iterdir())[-1]
    from src.demo.banana_test.run_demo import _load_calib
    intrinsics, extrinsics, h, w = _load_calib(calib_dir)
    pc_serials = {pc: get_camera_list(pc) for pc in args.pc_list}
    active = {s for serials in pc_serials.values() for s in serials}
    intrinsics = {s: value for s, value in intrinsics.items() if s in active}
    extrinsics = {s: value for s, value in extrinsics.items() if s in active}
    pc_ips = [get_pc_ip(pc) for pc in args.pc_list]

    rcc = remote_camera_controller("continuous_basket_demo", pc_list=args.pc_list,
                                   stall_timeout=15.0)
    orch = InitOrchestrator(
        pc_list=args.pc_list, capture_ips=pc_ips,
        command_timeout_ms=round(args.init_command_timeout_s * 1000),
        command_retries=1,
    )
    tracking = None
    recognizer = None
    executor = None
    successes = 0
    current_item: Optional[CatalogObject] = None
    verifier = PoseVerifier()
    try:
        _ensure_camera_lock(rcc); _clear_camera_errors(rcc)
        _rcc_start(rcc, "stream", False, fps=args.stream_fps)
        _warn_if_not_streaming(rcc)
        if args.basket_center is not None:
            basket_xyz = np.asarray(args.basket_center, dtype=np.float64)
            basket_record = {
                "source": "manual", "release_reference_robot": basket_xyz,
            }
        else:
            basket_xyz, basket_record = _measure_basket_reference(
                rcc, args, run_dir, active,
            )
        basket_bounds = np.array([
            [basket_xyz[0] - args.basket_observe_radius,
             basket_xyz[1] - args.basket_observe_radius, 0.0],
            [basket_xyz[0] + args.basket_observe_radius,
             basket_xyz[1] + args.basket_observe_radius,
             basket_xyz[2] + args.basket_observe_height],
        ], dtype=np.float64)
        (run_dir / "catalog.json").write_text(json.dumps(_jsonable({
            "objects": [item.__dict__ for item in catalogue], "basket": basket_record,
            "pick_workspace": pick_bounds, "basket_observation_workspace": basket_bounds,
            "policy": "fast_quality_no_silhouette + local_retry_no_home_reset",
        }), indent=2))
        if not single_object_mode:
            recognizer = CatalogRecognizer(gpu=args.catalog_gpu, conf_threshold=args.catalog_min_score)
        else:
            print(f"[catalog] one-object fast path: {catalogue[0].name}; YOLO-E skipped")
        planner = GraspPlanner(hand=_planner_robot(args.arm, args.hand))
        if args.verification_mode == "gotrack":
            tracking = LiveGoTrackSession(
                pc_list=args.pc_list, capture_ips=pc_ips,
                intrinsics=intrinsics, extrinsics=extrinsics,
                anchor_root=Path(args.tracking_anchor_root).expanduser(),
                port_obs=args.port_track_obs, port_prior=args.port_track_prior,
                port_cmd=args.port_track_cmd, min_cams_per_frame=args.track_min_cams,
                min_inliers=args.track_min_inliers,
                command_timeout_ms=round(args.tracking_command_timeout_s * 1000),
                command_retries=1,
            )
        if args.arm == "franka":
            from src.execution.franka_executor import FrankaExecutor
            executor = FrankaExecutor(hand_name=args.hand)
            executor.home(clear_view=True)  # one startup home only
        else:
            from autodex.executor.real import RealExecutor
            executor = RealExecutor(hand_name=args.hand)

        for cycle in range(1, args.max_cycles + 1):
            if successes >= args.max_successes:
                break
            if tracking is not None:
                tracking.stop()
            cycle_t0 = time.perf_counter()
            if single_object_mode:
                alternatives: list[CatalogMatch] = [single_object_match(catalogue)]
                print(f"[cycle {cycle}] one-object FoundPose check: {catalogue[0].name}")
            else:
                snapshot_dir = run_dir / "catalog_snapshots" / f"{cycle:03d}"
                n_images = capture_catalog_snapshot(
                    rcc, snapshot_dir, min_images=args.catalog_min_views,
                    settle_timeout_s=args.catalog_snapshot_timeout_s,
                    expected_serials=active, require_decodable=True,
                )
                print(f"[cycle {cycle}] live snapshot: {n_images} camera images")
                images = read_capture_images(snapshot_dir)
                match, alternatives = recognizer.identify(
                    images, catalogue, min_views=args.catalog_min_views,
                    min_score=args.catalog_min_score,
                )
                if match is None:
                    print(f"[cycle {cycle}] no known object; keep stream alive and try again")
                    _write_trial(run_dir, {"cycle": cycle, "status": "no_catalog_match"})
                    continue

            # A basket can contain objects from the same known catalogue.  The
            # detector names classes, not instances, so try its ranked classes
            # until FoundPose returns one inside the *pick* workspace.
            item = None
            pose_world = None
            perception = {}
            selected_match = None
            for candidate_match in alternatives:
                candidate_item = next(x for x in catalogue if x.name == candidate_match.name)
                if current_item != candidate_item:
                    _setup_object(orch, candidate_item, args, intrinsics, extrinsics,
                                  (h, w), pc_serials)
                    current_item = candidate_item
                candidate_dir = run_dir / "init" / f"{cycle:03d}_catalog_{candidate_item.name}"
                candidate_pose, candidate_timing = _observe_fast(
                    orch, candidate_item, candidate_dir, args, c2r, pick_bounds)
                if candidate_pose is not None:
                    item, pose_world, perception = candidate_item, candidate_pose, candidate_timing
                    selected_match = candidate_match
                    print(f"[cycle {cycle}] selected {item.name} score={candidate_match.score:.2f} "
                          f"views={candidate_match.supporting_views}")
                    break
            if item is None:
                _write_trial(run_dir, {"cycle": cycle, "status": "no_pose_in_pick_workspace",
                                        "catalog": [x.__dict__ for x in alternatives]})
                continue

            if tracking is not None:
                mesh_path, _assets = _object_paths(item, args.grasp_version)
                tracking.start(obj_name=item.name, mesh_path=mesh_path,
                               init_pose_world=pose_world,
                               settle_s=args.object_switch_settle_s)
                warm = tracking.wait_for_pose(timeout_s=args.tracking_warmup_s)
                if warm is None:
                    err = tracking.worker_error or "no reliable pose from GoTrack daemons"
                    diagnostics = tracking.diagnostics()
                    tracking.stop()
                    raise RuntimeError(
                        f"GoTrack warmup failed before grasp motion: {err}. "
                        f"diagnostics={json.dumps(_jsonable(diagnostics), sort_keys=True)}. "
                        "Run gotrack_smoke.py to diagnose without moving the arm; "
                        "use --verification-mode foundpose only for the slower fallback."
                    )
                pose_world = warm.pose_world
                perception = {**perception, "tracking_warmup": _track_timing(warm)}

            attempt = 0
            drop_retries = 0
            retry: Optional[LocalRetryPolicy] = None
            reuse_observation = True  # catalogue-stage pose for attempt one
            while True:
                attempt += 1
                attempt_dir = run_dir / "init" / f"{cycle:03d}_{attempt:02d}"
                t0 = cycle_t0 if attempt == 1 else time.perf_counter()
                if not reuse_observation:
                    pose_world, perception = _observe_fast(orch, item, attempt_dir, args,
                                                           c2r, pick_bounds)
                reuse_observation = False
                if pose_world is None:
                    _write_trial(run_dir, {"cycle": cycle, "attempt": attempt,
                                            "object": item.name, "status": "perception_failed",
                                            "perception": perception})
                    break
                before_robot = np.linalg.inv(c2r) @ pose_world
                if retry is None:
                    order, _stem, _tabletop = _candidate_order(item, args.hand, args.grasp_version,
                                                                before_robot, args.arm,
                                                                args.strict_tabletop_success)
                    retry = LocalRetryPolicy(order, max_attempts=args.max_retries)
                result, plan_info, prepared = _plan_attempt(
                    planner, executor, item, retry.remaining_candidates(),
                    pose_world, c2r, basket_xyz, args,
                )
                if result is None:
                    failed_key = tuple(plan_info["candidate"]) if plan_info.get("candidate") else None
                    decision = (retry.next_after_failure(failed_key, Verification.NOT_HELD)
                                if failed_key is not None else None)
                    _write_trial(run_dir, {"cycle": cycle, "attempt": attempt, "object": item.name,
                                            "status": plan_info["reason"], "perception": perception,
                                            "plan": plan_info,
                                            "retry": decision.__dict__ if decision else None})
                    if decision is not None and decision.retry:
                        print(f"  {plan_info['reason']}: trying the next grasp in place")
                        reuse_observation = True
                        continue
                    break
                # First automatic outcome: re-observe after the physical lift.
                lift_started = time.time()
                executor.execute(result, planner=planner, scene_cfg=plan_info["scene_cfg"],
                                 lift_height=args.lift_height,
                                 lift_traj_override=prepared["lift_traj"],
                                 start_from_current=True)
                if tracking is not None:
                    lift_sample = tracking.wait_for_pose(
                        since_wall_time=lift_started, timeout_s=args.tracking_timeout_s)
                    lift_pose, lift_timing = (
                        (lift_sample.pose_world, _track_timing(lift_sample))
                        if lift_sample is not None else (None, _track_timing(None))
                    )
                else:
                    lift_pose, lift_timing = _observe_fast(
                        orch, item, attempt_dir / "lift_check", args, c2r, pick_bounds)
                after_lift = PoseEvidence(
                    tuple((np.linalg.inv(c2r) @ lift_pose)[:3, 3]) if lift_pose is not None else None,
                    float(lift_timing.get("best_quality", 0.0)),
                )
                lift_check = verifier.after_lift(PoseEvidence(tuple(before_robot[:3, 3])), after_lift)
                candidate = tuple(result.scene_info)
                if lift_check is not Verification.HELD:
                    decision = retry.next_after_failure(candidate, lift_check)
                    # Only a re-observation at the original table position is
                    # evidence that the hand is empty.  Never open or descend
                    # on a missing/ambiguous observation: it might be holding
                    # the object behind the wrist.
                    if lift_check is Verification.NOT_HELD:
                        executor.release(result)
                    record = {"cycle": cycle, "attempt": attempt, "object": item.name,
                              "status": lift_check.value, "retry": decision.__dict__,
                              "perception": perception, "lift_check": lift_timing,
                              "pipeline_s": time.perf_counter() - t0}
                    _write_trial(run_dir, record)
                    if decision.retry:
                        print(f"  grasp miss: retrying on the observed object, without home reset")
                        pose_world, perception = lift_pose, lift_timing
                        reuse_observation = True
                        continue
                    if lift_check is Verification.UNCERTAIN:
                        raise RuntimeError(
                            "lift verification is uncertain while the gripper may hold an object; "
                            "leaving the robot raised for manual safety check"
                        )
                    print(f"  {decision.reason}; keeping arm raised and returning to catalogue scan")
                    break

                # The lift was visually verified.  Carry/drop is planned from
                # measured joints, then verified at the basket after retreat.
                hand_qpos = np.asarray(getattr(executor, "_last_hand_qpos", result.grasp_pose), dtype=np.float32)
                current = _planner_full_qpos(planner, executor, hand_qpos)
                wrist = _wrist_now(planner, executor, planner._n_arm, hand_qpos)
                T_now = wrist @ prepared["object_in_wrist"]
                target = _place_wrist(cart2se3(plan_info["scene_cfg"]["mesh"]["target"]["pose"]),
                                      prepared["object_in_wrist"], basket_xyz,
                                      prepared["place_yaw"], float(T_now[2, 3]), float(wrist[2, 3]))
                carry = planner.plan_pose_constrained(
                    current, target, hold_vec_weight=[0, 0, 0, 0, 0, 1],
                    scene_cfg=plan_info["scene_cfg"], include_obj_obstacle=False)
                if carry is None:
                    carry = prepared["carry_traj"]
                cmd = np.asarray(getattr(executor, "_last_hand_action", executor._convert(hand_qpos)), dtype=np.float64)
                executor._move_joints(carry[:, :planner._n_arm], np.tile(cmd, (len(carry), 1)))
                place_wrist = _wrist_now(planner, executor, planner._n_arm, hand_qpos)
                place_wrist[2, 3] -= args.drop_height
                place_kwargs = {"grasp_wrist": place_wrist} if args.arm == "franka" else {}
                place = executor.place(result, planner=planner, scene_cfg=plan_info["scene_cfg"],
                                       lift_height=args.drop_height, **place_kwargs)
                released_at = time.time()
                if args.arm != "franka":
                    executor.release(result)
                hand_after = np.asarray(getattr(executor, "_last_hand_qpos", result.pregrasp_pose), dtype=np.float32)
                retreat_ok = _retreat_up(planner, executor, hand_after, plan_info["scene_cfg"], args.retreat_height)
                if tracking is not None:
                    drop_sample = tracking.wait_for_pose(
                        since_wall_time=released_at, timeout_s=args.tracking_timeout_s)
                    dropped_pose, drop_timing = (
                        (drop_sample.pose_world, _track_timing(drop_sample))
                        if drop_sample is not None else (None, _track_timing(None))
                    )
                else:
                    dropped_pose, drop_timing = _observe_fast(
                        orch, item, attempt_dir / "drop_check", args, c2r, basket_bounds)
                drop_evidence = PoseEvidence(
                    tuple((np.linalg.inv(c2r) @ dropped_pose)[:3, 3]) if dropped_pose is not None else None,
                    float(drop_timing.get("best_quality", 0.0)),
                )
                drop_check = verifier.after_drop(drop_evidence, tuple(basket_xyz[:2]))
                success = drop_check is Verification.IN_BASKET
                successes += int(success)
                recovery_pose = None
                recovery_timing = None
                if not success and drop_retries < args.max_retries:
                    # A dropped object that missed the basket but remains in
                    # the pick workspace is a normal next pick, not a reason
                    # to reset home or stop the take.
                    if tracking is not None:
                        if dropped_pose is not None and _pose_in_bounds(dropped_pose, c2r, pick_bounds):
                            recovery_pose, recovery_timing = dropped_pose, drop_timing
                    else:
                        recovery_pose, recovery_timing = _observe_fast(
                            orch, item, attempt_dir / "drop_recovery", args, c2r, pick_bounds)
                _write_trial(run_dir, {"cycle": cycle, "attempt": attempt, "object": item.name,
                                        "status": "success" if success else drop_check.value,
                                        "catalog": selected_match.__dict__,
                                        "alternatives": [x.__dict__ for x in alternatives],
                                        "perception": perception, "lift_check": lift_timing,
                                        "drop_check": drop_timing, "place": place,
                                        "retreat_ok": retreat_ok,
                                        "drop_recovery": recovery_timing,
                                        "pipeline_s": time.perf_counter() - t0})
                print(f"  {'SUCCESS' if success else drop_check.value}: {successes}/{args.max_successes}")
                if recovery_pose is not None:
                    drop_retries += 1
                    pose_world, perception, retry = recovery_pose, recovery_timing, None
                    reuse_observation = True
                    print("  dropped object remains in pick workspace: retrying without home reset")
                    continue
                break
    finally:
        if tracking is not None:
            tracking.close()
        if recognizer is not None:
            recognizer.close()
        if executor is not None:
            _safe("executor.shutdown", executor.shutdown)
        _safe("orch.close", orch.close)
        _safe("rcc.stop", rcc.stop)
        _safe("rcc.end", rcc.end)


if __name__ == "__main__":
    main()
