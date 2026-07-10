#!/usr/bin/env python3
"""Reorient policy: drop-style chain plan (approach → lift → reorient → descent)
seeded by **reset** candidates from
``candidates/{hand}/reset/{obj}/reorient_{h_cm}/{i}_{j}/``.

Per-cycle (mirrors ``reorient_drop.py`` structure):
    perception -> classify tabletop_before (i)
    -> if i == target_j: skip
    -> load reset seeds from cell {i}_{target_j} for the smallest reorient_{h_cm}
    -> for each seed (IK-feasible): approach -> lift -> reorient -> descent
       (first seed whose full chain plans wins)
    -> execute init→approach→pregrasp→grasp→squeeze (skip_lift)
    -> joint-space lift trajectory replay
    -> charuco lift-check via snapshot_daemon
        * fail -> reset_fallback, skip cycle
        * pass -> reorient_traj replay -> descent_traj replay
              -> release (at ``RELEASE_HEIGHT_M = h_cm/100`` above table)
              -> reset_fallback (open hand, sequential retract)
              -> post-perception -> classify tabletop_after
              -> tabletop_hit_target = (filename int of tb_after == target_j)

``reorient_{h_cm}`` is the BODex-generated descent height: ``h_cm=0`` lands the
object on the table, ``h_cm=8`` releases 8 cm above. Highest priority (smallest
h_cm) folder is selected automatically.

Prerequisites:
    bash scripts/init_daemons.sh start
    bash scripts/snapshot_daemons.sh start

Usage:
    python src/experiment/reset/reorient.py --obj donut --target_j 2 --auto
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
import torch
import trimesh
import yourdfpy
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.io.camera_system.signal_generator import UTGE900
from paradex.io.camera_system.timestamp_monitor import TimestampMonitor
from paradex.utils.system import network_info, get_pc_ip, get_camera_list
from paradex.calibration.utils import save_current_camparam, save_current_C2R, load_c2r

from autodex.utils.path import project_dir, obj_path
from autodex.utils.conversion import cart2se3
from autodex.planner import GraspPlanner
from autodex.planner.planner import (
    PlanResult, _to_curobo_world, _to_curobo_pose, _snap_joint6,
)
from autodex.planner.obstacles import TABLE_CUBOID, add_obstacles
from autodex.planner.visualizer import ScenePlanVisualizer
from autodex.executor.real import RealExecutor
from autodex.perception.init_orchestrator import InitOrchestrator
from autodex.perception.snapshot_orchestrator import SnapshotOrchestrator

from curobo.geom.types import WorldConfig

from src.execution.scene_cfg import pose_world_to_scene_cfg
from autodex.utils.symmetry import get_cyl_axis_local
from src.execution.run_auto import (
    DEFAULT_PC_LIST, ASSETS_BASE, MESH_BASE, CAM_PARAM_ROOT, _load_calib,
)
from src.execution.label import auto_label_charuco
from src.experiment.reset.tabletop_pose import classify_tabletop_pose

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
LIFT_HEIGHT_M = 0.25         # +25 cm above grasp pose
TABLE_SURFACE_Z = TABLE_CUBOID["pose"][2] + TABLE_CUBOID["dims"][2] / 2  # 0.039
EXP_NAME = "reset_test/reorient"
SCENE = "table"
PC_LIST = DEFAULT_PC_LIST
PORT_MASK = 5006
PORT_POSE = 5007
PORT_CMD = 6893
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


def _yaw_search_and_print_cmd(planner, scene_cfg, seeds, args, prev_n_ok=0):
    """Search (x, yaw) over a grid for an obj pose that improves IK count.

    Why x too: xarm6's joint 0 only rotates around vertical, so obj position
    in xy reduces (under j0 freedom) to the radial distance — sweeping x
    along y=0 explores that radius. Yaw is the obj's world-z rotation.

    Returns ``(target_yaw_deg, target_x, n_ok)`` for the best pose whose
    IK-feasible count STRICTLY exceeds ``prev_n_ok``. Prints the
    ``rotate_obj_yaw.py`` command on success. Returns ``(None, None, prev_n_ok)``
    if nothing improves.
    """
    from autodex.utils.conversion import cart2se3 as _cart2se3
    from autodex.utils.conversion import se32cart as _se32cart
    _obj_pose_now = _cart2se3(scene_cfg["mesh"]["target"]["pose"])
    _orig_wrist = seeds["wrist_se3"]
    _obj_inv = np.linalg.inv(_obj_pose_now)
    _yaw_grid = np.deg2rad(np.arange(0, 360, 30))
    _x_grid = np.arange(0.30, 0.70, 0.05)   # y fixed = 0 (j0 covers angle)
    _found_yaw_deg = None
    _found_x = None
    _found_n_ok = prev_n_ok
    for _x in _x_grid:
        for _θ in _yaw_grid:
            _c, _s = np.cos(_θ), np.sin(_θ)
            _Rz4 = np.eye(4)
            _Rz4[:3, :3] = np.array([[_c, -_s, 0], [_s, _c, 0], [0, 0, 1]])
            _new_obj = _obj_pose_now.copy()
            _new_obj[:3, :3] = _Rz4[:3, :3] @ _obj_pose_now[:3, :3]
            _new_obj[0, 3] = float(_x)
            _new_obj[1, 3] = 0.0
            # _obj_pose_now[2,3] (z) preserved via .copy().
            # Wrist follows the new obj rigidly:
            #   T_wrist_new = T_obj_new @ inv(T_obj_now) @ T_wrist_now
            _new_wrist = np.einsum("ij,jk,Nkl->Nil",
                                   _new_obj, _obj_inv, _orig_wrist)
            _rot_seeds = dict(seeds)
            _rot_seeds["wrist_se3"] = _new_wrist
            _rot_scene = dict(scene_cfg)
            _rot_scene["mesh"] = dict(scene_cfg["mesh"])
            _rot_scene["mesh"]["target"] = dict(scene_cfg["mesh"]["target"])
            _rot_scene["mesh"]["target"]["pose"] = _se32cart(_new_obj).tolist()
            try:
                _r = _ik_check_seeds(planner, _rot_scene, _rot_seeds)
                _n = int(_r["ik_success"].sum())
            except Exception:
                _n = 0
            if _n > _found_n_ok:
                _found_n_ok = _n
                _found_yaw_deg = float(np.degrees(_θ))
                _found_x = float(_x)
                if _n == len(_orig_wrist):
                    break
        if _found_n_ok == len(_orig_wrist):
            break
    if _found_yaw_deg is not None:
        # Verify approach plan_single_js succeeds at this (x, yaw) for at
        # least one IK-feasible candidate. yaw_search only proves the GRASP
        # IK reaches; the trajectory from INIT_STATE to grasp_qpos can still
        # collide / hit joint limits. Re-run _ik_check_seeds at the best
        # pose to recover IK qpos, then plan_single_js for each feasible
        # candidate and keep the suggestion only if at least one passes.
        from autodex.utils.conversion import se32cart as _se32cart_v
        _θ_best = np.deg2rad(_found_yaw_deg)
        _cb, _sb = np.cos(_θ_best), np.sin(_θ_best)
        _Rz_best = np.array([[_cb, -_sb, 0], [_sb, _cb, 0], [0, 0, 1]])
        _best_obj = _obj_pose_now.copy()
        _best_obj[:3, :3] = _Rz_best @ _obj_pose_now[:3, :3]
        _best_obj[0, 3] = float(_found_x); _best_obj[1, 3] = 0.0
        _best_wrist = np.einsum("ij,jk,Nkl->Nil",
                                 _best_obj, _obj_inv, _orig_wrist)
        _best_seeds = dict(seeds)
        _best_seeds["wrist_se3"] = _best_wrist
        _best_scene = dict(scene_cfg)
        _best_scene["mesh"] = dict(scene_cfg["mesh"])
        _best_scene["mesh"]["target"] = dict(scene_cfg["mesh"]["target"])
        _best_scene["mesh"]["target"]["pose"] = _se32cart_v(_best_obj).tolist()
        _best_ik = _ik_check_seeds(planner, _best_scene, _best_seeds)
        _ok_idx = np.where(_best_ik["ik_success"])[0]
        approach_ok = False
        for _ci in _ok_idx:
            try:
                _ok_ap, _ = planner._refine_fingers(
                    planner._init_state, _best_ik["ik_qpos"][int(_ci)])
            except Exception:
                _ok_ap = False
            if _ok_ap:
                approach_ok = True
                break
        if approach_ok:
            print(f"    [pose_search] move obj to (x={_found_x:.2f}, y=0) "
                  f"+ rotate {_found_yaw_deg:.0f}° → "
                  f"{_found_n_ok}/{len(_orig_wrist)} IK-feasible "
                  f"(prev={prev_n_ok}, approach OK)")
            print(f"    [run this]")
            print(f"      python src/execution/rotate_obj_yaw.py "
                  f"--obj {args.obj} --hand {args.hand} "
                  f"--target_yaw_deg {_found_yaw_deg:.0f} "
                  f"--target_x {_found_x:.2f}")
        else:
            print(f"    [pose_search] (x={_found_x:.2f}, yaw={_found_yaw_deg:.0f}°)"
                  f" is IK-feasible but approach trajopt failed for all "
                  f"{len(_ok_idx)} candidates — NOT suggesting.")
            _found_yaw_deg = None  # suppress as if not found
    else:
        print(f"    [pose_search] no (x, yaw) in grid improves on "
              f"prev={prev_n_ok}/{len(_orig_wrist)}")
    return _found_yaw_deg, _found_x, _found_n_ok


def _now() -> str:
    return datetime.datetime.now().isoformat()


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _stop_video(rcc, sync_generator, timestamp_monitor):
    rcc.stop()
    time.sleep(0.3)
    sync_generator.stop()
    timestamp_monitor.stop()


def _pose_int_from_filename(filename: str) -> int:
    """Parse the integer from a tabletop pose filename (e.g. ``002.npy`` → 2)."""
    return int(filename.replace(".npy", ""))


def _autoselect_h_cm(hand: str, obj: str, target_j: int | None = None) -> int | None:
    """Pick the smallest ``reorient_{h_cm}`` folder that contains at least one
    ``*_{target_j}`` cell. If ``target_j`` is None, fall back to the smallest
    ``reorient_{h_cm}`` folder regardless of cell contents."""
    reset_root = Path(project_dir) / "candidates" / hand / "reset" / obj
    if not reset_root.exists():
        return None
    cands = []
    for p in reset_root.iterdir():
        if not p.is_dir() or not p.name.startswith("reorient_"):
            continue
        try:
            h = int(p.name.split("_", 1)[1])
        except ValueError:
            continue
        if target_j is not None:
            has_target = any(
                c.is_dir() and c.name.endswith(f"_{target_j}")
                and c.name.split("_", 1)[0].isdigit()
                for c in p.iterdir()
            )
            if not has_target:
                continue
        cands.append((h, p))
    if not cands:
        return None
    return sorted(cands, key=lambda kv: kv[0])[0][0]


def _load_target_tabletop_pose(obj: str, target_j: int) -> np.ndarray:
    """Load 4x4 tabletop pose (robot frame) for filename int ``target_j``.
    Files are zero-padded 3-digit (``002.npy``)."""
    fname = f"{target_j:03d}.npy"
    p = Path(obj_path) / obj / "processed_data" / "info" / "tabletop" / fname
    if not p.exists():
        # try un-padded fallback
        p2 = Path(obj_path) / obj / "processed_data" / "info" / "tabletop" / f"{target_j}.npy"
        if p2.exists():
            p = p2
        else:
            raise FileNotFoundError(f"target tabletop pose not found: {p} (or {p2})")
    pose = np.load(p)
    if pose.shape == (3, 3):
        T = np.eye(4)
        T[:3, :3] = pose
        pose = T
    return pose


def _load_reset_seeds(hand: str, obj: str, h_cm: int, i_int: int, j_int: int,
                      T_obj_world: np.ndarray) -> dict | None:
    """Load reset grasp seeds for cell (i_int, j_int) under reorient_{h_cm}.

    Files on disk are in **object frame**; this transforms wrist_se3 to world
    via ``T_wrist_world = T_obj_world @ wrist_se3_obj``. Returns ``None`` if
    the cell directory does not exist or has no grasp subfolders.

    Also loads ``openpose_{i_int:03d}.npy`` (start-pose-matched openpose) and
    ``openpose_{j_int:03d}.npy`` (target-pose-matched openpose) per grasp.

    Output dict mirrors ``GraspPlanner.solve_ik``'s candidate-related
    fields plus ``openpose_start`` and ``openpose_target``.
    """
    cell_dir = (Path(project_dir) / "candidates" / hand / "reset" / obj
                / f"reorient_{h_cm}" / f"{i_int}_{j_int}")
    if not cell_dir.exists():
        return None
    # Order by stats.json priority desc (Laplace-smoothed success rate),
    # tie-break by numeric grasp_id asc — same scheme as
    # table_only_grasp_order_by_stats. Falls back to numeric order if
    # stats.json missing (read_grasp_stats returns (0, 0)).
    from autodex.utils.coverage import read_grasp_stats, grasp_priority_score
    # Drop dirs that lack the required candidate files (e.g. "pA_*" preview
    # dirs that only ship openpose images).
    raw_dirs = [p for p in cell_dir.iterdir()
                if p.is_dir() and (p / "wrist_se3.npy").exists()]
    if not raw_dirs:
        return None
    def _name_key(p):
        # Numeric names sort numerically; alphanumeric (e.g. "pA_15") fall
        # back to lexicographic — placed after numeric to keep historic
        # order for purely-numeric cells.
        try:
            return (0, int(p.name))
        except ValueError:
            return (1, p.name)
    grasp_dirs = sorted(
        raw_dirs,
        key=lambda p: (
            -grasp_priority_score(*read_grasp_stats(str(p))),
            _name_key(p),
        ),
    )
    wrist_obj = np.stack([np.load(g / "wrist_se3.npy")     for g in grasp_dirs])
    pregrasp  = np.stack([np.load(g / "pregrasp_pose.npy") for g in grasp_dirs])
    grasp     = np.stack([np.load(g / "grasp_pose.npy")    for g in grasp_dirs])
    op_start = []
    op_target = []
    for g in grasp_dirs:
        op_s_path = g / f"openpose_{i_int:03d}.npy"
        op_t_path = g / f"openpose_{j_int:03d}.npy"
        op_start.append(np.load(op_s_path) if op_s_path.exists() else None)
        op_target.append(np.load(op_t_path) if op_t_path.exists() else None)
    wrist_world = T_obj_world[None] @ wrist_obj
    scene_info = [{
        "grasp_idx": int(g.name),
        "cell": f"{i_int}_{j_int}",
        "h_cm": h_cm,
        "source": str(g),
    } for g in grasp_dirs]
    return {
        "wrist_se3": wrist_world,
        "pregrasp": pregrasp,
        "grasp": grasp,
        "openpose_start": op_start,
        "openpose_target": op_target,
        "scene_info": scene_info,
        "n_total": len(grasp_dirs),
    }


def _ik_check_seeds(planner: GraspPlanner, scene_cfg: dict, seeds: dict) -> dict:
    """Run IK + backward + collision filter on pre-loaded reset seeds. Mirrors
    ``GraspPlanner.solve_ik``'s post-load logic (planner.py:376-466) but
    consumes our disk-loaded seeds instead of calling ``load_candidate``.

    Returns a dict matching ``solve_ik`` output shape:
        ik_success, ik_qpos, wrist_se3, pregrasp, grasp, scene_info, n_total,
        n_backward, n_table_collision, n_valid, n_ik_success, timing
    """
    import time as _time

    wrist_se3 = seeds["wrist_se3"]
    pregrasp  = seeds["pregrasp"]
    grasp     = seeds["grasp"]
    scene_info = seeds["scene_info"]
    N = len(wrist_se3)

    t0 = _time.time()
    world_cfg_no_target = _to_curobo_world(scene_cfg)
    world_cfg_no_target["mesh"] = {}
    # Lower collision activation distance to 0 — reset candidates intentionally
    # bring fingers close to table/obj surface; default ~2mm rejects valid IK.
    planner._collision_act_dist = 0.0
    planner._ik_solver = None
    planner._init_ik_solver(world_cfg_no_target)
    t_world = _time.time() - t0

    t0 = _time.time()
    if planner._hand.startswith("inspire"):
        backward = np.zeros(N, dtype=bool)
    else:
        backward = (wrist_se3[:, :3, :3] @ planner._link6_y_in_wrist)[:, 2] < 0.3
    collision = planner._check_collision(world_cfg_no_target, wrist_se3, pregrasp)
    filtered = backward | collision
    valid = np.where(~filtered)[0]
    t_filter = _time.time() - t0

    ik_success = np.zeros(N, dtype=bool)
    ik_qpos = np.full((N, len(planner._init_state)), np.nan)

    t0 = _time.time()
    BATCH_SIZE = planner.BATCH_SIZE
    if len(valid) > 0:
        for chunk_start in range(0, len(valid), BATCH_SIZE):
            chunk_idx = valid[chunk_start : chunk_start + BATCH_SIZE]
            chunk_poses = wrist_se3[chunk_idx]
            B = len(chunk_poses)
            if B < BATCH_SIZE:
                pad = BATCH_SIZE - B
                chunk_poses = np.concatenate(
                    [chunk_poses, np.tile(chunk_poses[:1], (pad, 1, 1))], axis=0,
                )
            goal = _to_curobo_pose(chunk_poses, planner._tensor_args.device)
            B_padded = chunk_poses.shape[0]
            retract = torch.tensor(
                planner._init_state, dtype=torch.float32,
                device=planner._tensor_args.device,
            ).unsqueeze(0).repeat(B_padded, 1)
            result = planner._ik_solver.solve_batch(goal, retract_config=retract)
            succ = result.success.cpu().numpy()[:B].reshape(-1)
            q_sol = result.solution.cpu().numpy()[:B]
            if q_sol.ndim == 3:
                q_sol = q_sol[:, 0, :]
            try:
                pos_err = result.position_error.cpu().numpy()[:B].reshape(-1)
                rot_err = result.rotation_error.cpu().numpy()[:B].reshape(-1)
            except Exception:
                pos_err = [None] * B
                rot_err = [None] * B

            # Diagnostic: for fails with tiny pos_err/rot_err but succ=False,
            # rerun IK with NO obstacles in world to test if collision is the
            # cause. If succ=True without obstacles, the rejection was from
            # collision_activation_distance, not unreachability.
            need_diag = False
            for i in range(B):
                if not succ[i] and pos_err[i] is not None and pos_err[i] < 1e-3:
                    need_diag = True
            diag_succ_noworld = None
            if need_diag:
                try:
                    from curobo.geom.types import WorldConfig as _WC
                    empty_world = {"mesh": {}, "cuboid": {}}
                    saved_world_cfg = world_cfg_no_target
                    planner._ik_solver.update_world(_WC.from_dict(empty_world))
                    result2 = planner._ik_solver.solve_batch(
                        goal, retract_config=retract)
                    diag_succ_noworld = result2.success.cpu().numpy()[:B].reshape(-1)
                    planner._ik_solver.update_world(_WC.from_dict(saved_world_cfg))
                except Exception as _de:
                    print(f"      [diag noworld] failed: {_de!r}")

            for i, idx in enumerate(chunk_idx):
                wp = wrist_se3[idx][:3, 3]
                dist = float(np.linalg.norm(wp))
                if succ[i]:
                    ik_success[idx] = True
                    arm_q = q_sol[i, :6].copy()
                    arm_q[5] = _snap_joint6(arm_q[5], planner._init_state[5])
                    ik_qpos[idx, :6] = arm_q
                    ik_qpos[idx, 6:] = pregrasp[idx]
                else:
                    noworld_tag = ""
                    if diag_succ_noworld is not None:
                        noworld_tag = (f"  noworld_succ={bool(diag_succ_noworld[i])}"
                                       f" ← {'collision was cause' if diag_succ_noworld[i] else 'unreachable (not collision)'}")
                    print(f"      [ik fail] cand {idx}: wrist pos={wp.round(3)} "
                          f"|pos|={dist:.3f}m  pos_err={pos_err[i]}  "
                          f"rot_err={rot_err[i]}{noworld_tag}")
    t_ik = _time.time() - t0

    # Lift IK pre-check (z + 10 cm reachable).
    # Just need lift to go UP a bit, not all the way to LIFT_HEIGHT_M.
    # Executor handles "as far as feasible" at runtime.
    LIFT_HEIGHT_CHECK = 0.03
    ik_valid_pre = np.where(ik_success)[0]
    if len(ik_valid_pre) > 0:
        lift_poses = wrist_se3[ik_valid_pre].copy()
        lift_poses[:, 2, 3] += LIFT_HEIGHT_CHECK
        for chunk_start in range(0, len(ik_valid_pre), BATCH_SIZE):
            chunk = ik_valid_pre[chunk_start : chunk_start + BATCH_SIZE]
            chunk_poses = lift_poses[chunk_start : chunk_start + len(chunk)]
            B = len(chunk_poses)
            if B < BATCH_SIZE:
                pad = BATCH_SIZE - B
                chunk_poses = np.concatenate(
                    [chunk_poses, np.tile(chunk_poses[:1], (pad, 1, 1))], axis=0,
                )
            goal = _to_curobo_pose(chunk_poses, planner._tensor_args.device)
            lift_res = planner._ik_solver.solve_batch(goal)
            lift_succ = lift_res.success.cpu().numpy()[:B]
            for i, idx in enumerate(chunk):
                if not lift_succ[i]:
                    ik_success[idx] = False
        n_lift_fail = len(ik_valid_pre) - int(ik_success.sum())
        if n_lift_fail > 0:
            print(f"[reorient] lift IK check: {n_lift_fail} seeds failed "
                  f"(z+{LIFT_HEIGHT_CHECK}m unreachable)")

    timing = {
        "world_setup_s": round(t_world, 3),
        "filter_s": round(t_filter, 3),
        "ik_solve_s": round(t_ik, 3),
    }

    return {
        "n_total": N,
        "n_backward": int(backward.sum()),
        "n_table_collision": int(collision.sum()),
        "n_valid": int(len(valid)),
        "n_ik_success": int(ik_success.sum()),
        "ik_success": ik_success,
        "ik_qpos": ik_qpos,
        "wrist_se3": wrist_se3,
        "pregrasp": pregrasp,
        "grasp": grasp,
        "scene_info": scene_info,
        "timing": timing,
    }


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
            if qpos is not None:
                try: vis.robot_dict["xarm"].update_cfg(np.asarray(qpos))
                except Exception as ue:
                    print(f"[viz] xarm update_cfg: {ue!r}")
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
            T_obj = e.get("T_obj_at_fail")
            if T_obj is not None and "obj" in vis.obj_dict:
                fr = vis.obj_dict["obj"]["frame"]
                fr.position = T_obj[:3, 3]
                fr.wxyz = R.from_matrix(T_obj[:3, :3]).as_quat()[[3, 0, 1, 2]]
            info_md.content = (
                f"**cand #{e['cand_idx']}** — fail at **{e['stage']}**  \n"
                f"reason: `{e.get('reason')}`"
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--hand", type=str, default="inspire_left",
                        choices=["allegro", "inspire", "inspire_left"])
    parser.add_argument("--target_j", type=int, required=True,
                        help="Target tabletop pose int (folder {i}_{target_j} "
                             "and file {target_j:03d}.npy).")
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

    # Auto-select smallest reorient_{h_cm} folder containing target_j cells.
    h_cm = _autoselect_h_cm(args.hand, args.obj, args.target_j)
    if h_cm is None:
        sys.exit(
            f"no reset/reorient_*/ folder with target_j={args.target_j} cells at "
            f"{project_dir}/candidates/{args.hand}/reset/{args.obj}/"
        )
    RELEASE_HEIGHT_M = h_cm / 100.0
    print(f"[reset] selected reorient_{h_cm}  "
          f"(RELEASE_HEIGHT_M={RELEASE_HEIGHT_M:.2f}m, LIFT_HEIGHT_M={LIFT_HEIGHT_M:.2f}m)")

    # Target tabletop pose (robot frame, fixed for the whole run).
    target_tabletop_robot = _load_target_tabletop_pose(args.obj, args.target_j)
    R_target_robot = target_tabletop_robot[:3, :3]
    print(f"[target] target_j={args.target_j} "
          f"(file {args.target_j:03d}.npy or {args.target_j}.npy)")

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

    client_name = f"reorient_{os.getpid()}"
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

    print(f"[orch] init for {args.obj}...")
    orch = InitOrchestrator(
        pc_list=PC_LIST, capture_ips=pc_ips,
        port_mask=PORT_MASK, port_pose=PORT_POSE, port_cmd=PORT_CMD,
    )
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
    from curobo.util.logger import setup_curobo_logger
    setup_curobo_logger("warning")
    print("[executor] connecting to robot...")
    executor = RealExecutor(hand_name=args.hand)

    trials: list = []
    summary_path = obj_root / "summary.json"
    cycle = 0
    vis = None

    # ── Guaranteed cleanup ───────────────────────────────────────────────
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
            print(f"\n{'#'*60}\n# Cycle {cycle}  "
                  f"(target_j={args.target_j}, h_cm={h_cm})\n{'#'*60}")
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
                "target_j": args.target_j,
                "h_cm": h_cm,
                "release_height_m": RELEASE_HEIGHT_M,
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
                # 1. Perception.
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
                if not tb_before:
                    rec["progress"]["tabletop_before"] = "no_tabletop_data"
                    rec["status"] = "no_tabletop_data"
                    print("    no tabletop_data for object — skipping cycle")
                    raise _SoftSkip
                i_int = _pose_int_from_filename(tb_before["filename"])
                rec["i_int"] = i_int
                rec["cell"] = f"{i_int}_{args.target_j}"
                rec["progress"]["tabletop_before"] = (
                    f"i={i_int} (file={tb_before['filename']}, "
                    f"err={tb_before['rot_err_deg']:.1f}°)"
                )
                print(f"    [tabletop before] i={i_int} ({tb_before['filename']}) "
                      f"err={tb_before['rot_err_deg']:.1f}°  target_j={args.target_j}")

                if i_int == args.target_j:
                    rec["status"] = "already_at_target"
                    rec["progress"]["plan"] = "skipped (i == target_j)"
                    print(f"    already at target (i={i_int} == target_j) — skipping cycle")
                    raise _SoftSkip

                # 3. Load reset seeds for cell {i_int}_{target_j}.
                print(f"[cycle {cycle}] Loading reset seeds "
                      f"reorient_{h_cm}/{i_int}_{args.target_j}...")
                seeds = _load_reset_seeds(
                    args.hand, args.obj, h_cm, i_int, args.target_j,
                    pose_robot_before,
                )
                if seeds is None:
                    rec["progress"]["plan"] = f"no_cell ({i_int}_{args.target_j})"
                    rec["status"] = "no_cell"
                    print(f"    no candidates for cell {i_int}_{args.target_j} "
                          f"under reorient_{h_cm} — skipping cycle")
                    raise _SoftSkip
                print(f"    loaded {seeds['n_total']} seeds from cell "
                      f"{i_int}_{args.target_j}")
                # Cylinder freedom for approach IK — expand seeds by cyl_yaw
                # rotations around the object's symmetry axis when not (nearly)
                # vertical at the start pose.
                _start_cyl_axis = get_cyl_axis_local(args.obj)
                if _start_cyl_axis is not None:
                    _axis_world = pose_robot_before[:3, :3] @ _start_cyl_axis
                    if abs(_axis_world[2]) >= 0.95:
                        _start_cyl_grid = None
                    else:
                        _start_cyl_grid = np.linspace(
                            0, 2 * np.pi, 8, endpoint=False)
                else:
                    _start_cyl_grid = None
                if _start_cyl_grid is not None:
                    from autodex.planner.planner import _expand_candidates_cyl
                    w, p, g, op_list_pair, si = _expand_candidates_cyl(
                        seeds["wrist_se3"], seeds["pregrasp"], seeds["grasp"],
                        # zip start+target openposes into a tuple per cand so they
                        # ride along the expansion
                        list(zip(seeds["openpose_start"],
                                  seeds["openpose_target"])),
                        seeds["scene_info"], pose_robot_before,
                        _start_cyl_axis, _start_cyl_grid)
                    seeds = {
                        "wrist_se3": w, "pregrasp": p, "grasp": g,
                        "openpose_start": [t[0] for t in op_list_pair],
                        "openpose_target": [t[1] for t in op_list_pair],
                        "scene_info": si, "n_total": len(w),
                    }
                    print(f"    cyl_yaw expanded: {len(w)} candidates "
                          f"({len(_start_cyl_grid)}x)")

                # 4. Plan — IK check + candidate enumerate (approach → lift →
                #    reorient → descent) on reset seeds.
                print(f"[cycle {cycle}] Planning (scene={SCENE})...")
                t0 = time.time()
                scene_cfg = pose_world_to_scene_cfg(pose_world, c2r, args.obj)
                scene_cfg = add_obstacles(scene_cfg, SCENE)
                _write_json(cdir / "scene_cfg.json", scene_cfg)

                ik_res = _ik_check_seeds(planner, scene_cfg, seeds)
                ik_ok = list(np.where(ik_res["ik_success"])[0])
                np.random.shuffle(ik_ok)
                n_grasp_total = ik_res["n_total"]
                planner._ik_solver = None  # rebuild for next plan_obj_placement
                # Openpose lookup: seeds["openpose_start"] = openpose matching
                # the START tabletop pose (i_int). Used for approach IK so the
                # hand is wider open during approach.
                openpose_start_list = seeds.get(
                    "openpose_start", [None] * n_grasp_total)
                openpose_target_list = seeds.get(
                    "openpose_target", [None] * n_grasp_total)
                n_op_s = sum(1 for op in openpose_start_list if op is not None)
                n_op_t = sum(1 for op in openpose_target_list if op is not None)
                print(f"    openpose: start({i_int:03d})={n_op_s}/{n_grasp_total}  "
                      f"target({args.target_j:03d})={n_op_t}/{n_grasp_total}")
                print(f"    grasp IK: {len(ik_ok)}/{n_grasp_total} feasible "
                      f"(backward={ik_res['n_backward']}, "
                      f"collision={ik_res['n_table_collision']})")
                if len(ik_ok) == 0:
                    # Phase 1: search world-z yaw rotation of obj that would
                    # make at least one reset candidate IK-feasible.
                    _found_yaw_deg, _found_x, _found_n_ok = _yaw_search_and_print_cmd(
                        planner, scene_cfg, seeds, args)
                    rec["progress"]["plan"] = (
                        f"no_ik_feasible_grasp ({n_grasp_total} total)"
                        + (f"; suggested yaw={_found_yaw_deg:.0f}°"
                           if _found_yaw_deg is not None else ""))
                    rec["yaw_suggestion_deg"] = _found_yaw_deg
                    rec["status"] = "plan_failed"
                    # Viz: launch ScenePlanVisualizer showing candidates
                    # colored by IK status (green=ok, yellow=ik_fail, red=filtered).
                    try:
                        from autodex.planner.visualizer import ScenePlanVisualizer
                        _filtered = np.zeros(n_grasp_total, dtype=bool)
                        _ik_failed = ~ik_res["ik_success"]
                        fv = ScenePlanVisualizer(scene_cfg, None,
                                                  port=8080, hand=args.hand)
                        fv.add_candidates(seeds["wrist_se3"], seeds["grasp"],
                                          _filtered, ik_failed=_ik_failed)
                        fv.start_viewer(use_thread=True)
                        print(f"    [viz] http://localhost:8080  "
                              f"(yellow=IK fail, slider 0..{n_grasp_total-1})")
                    except Exception as _ve:
                        print(f"    [viz] launch failed: {_ve!r}")
                    raise _SoftSkip

                T_obj_grasp_world_full = cart2se3(scene_cfg["mesh"]["target"]["pose"])
                R_target_obj_world_pre = target_tabletop_robot[:3, :3]
                obj_target_pos_world_pre = np.array([
                    0.0, 0.0,
                    float(target_tabletop_robot[2, 3])
                        + TABLE_SURFACE_Z + LIFT_HEIGHT_M,
                ])
                scene_lift_pre = {"mesh": {}, "cuboid": dict(scene_cfg["cuboid"])}
                X_GRID_PRE = np.arange(0.35, 0.55, 0.05)
                YAW_GRID_PRE = np.linspace(0, 2 * np.pi, 8, endpoint=False)
                # Continuous-revolute objects: free DoF about object's
                # symmetry axis (from src/scene_generation/symmetry.json),
                # applied IN OBJECT FRAME:
                # Rz(world_yaw) @ R_target @ R_local(cyl).
                # When the symmetry axis ends up (nearly) world-vertical at
                # the target, R_local is degenerate with the world yaw —
                # collapse cyl grid to a single point to avoid wasted IK.
                from autodex.utils.symmetry import get_cyl_yaw_grid
                CYL_AXIS_LOCAL = get_cyl_axis_local(args.obj)
                CYL_YAW_GRID_PRE = get_cyl_yaw_grid(args.obj)
                if (CYL_AXIS_LOCAL is not None
                        and CYL_YAW_GRID_PRE is not None
                        and abs((R_target_obj_world_pre @ CYL_AXIS_LOCAL)[2])
                            >= 0.95):
                    CYL_YAW_GRID_PRE = np.array([0.0])

                world_approach = _to_curobo_world(scene_cfg)
                # Restore default collision activation distance for trajopt —
                # IK was forced to 0 above so it accepts close-to-surface
                # candidates, but plan_single_js with act_dist=0 has no
                # margin and finetune_trajopt fails. The default (per-hand,
                # e.g. inspire_left=0.005) gives the trajectory enough
                # clearance to avoid jitter-triggered collisions.
                _hand_cfg = planner.HAND_CONFIGS.get(
                    args.hand, planner.HAND_CONFIGS["allegro"])
                planner._collision_act_dist = _hand_cfg[2]
                planner._motion_gen = None
                planner._init_motion_gen(world_approach)
                planner._cached_world = world_approach

                obj_target_pos_descent = np.array([
                    0.0, 0.0,
                    float(target_tabletop_robot[2, 3])
                        + TABLE_SURFACE_Z + RELEASE_HEIGHT_M,
                ])

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
                openpose_target_chosen = None    # for release-side wider open
                scene_info_chosen = None
                wrist_grasp_chosen = None
                result = None
                n_approach_fail = 0
                n_lift_fail = 0
                n_reorient_fail = 0
                n_descent_fail = 0
                cand_log: list = []

                for cand_idx in ik_ok:
                    cand_idx = int(cand_idx)
                    wrist_grasp_cand = ik_res["wrist_se3"][cand_idx]
                    grasp_cand = ik_res["grasp"][cand_idx]
                    pregrasp_cand = ik_res["pregrasp"][cand_idx]
                    T_obj_in_wrist_cand = (
                        np.linalg.inv(wrist_grasp_cand) @ T_obj_grasp_world_full)

                    # (a) approach — use openpose_start (wider hand) as the
                    # approach-end finger config if available; else pregrasp.
                    if planner._world_structure_changed(world_approach):
                        planner._update_world(world_approach)
                        planner._cached_world = world_approach
                    approach_goal = ik_res["ik_qpos"][cand_idx].copy()
                    if openpose_start_list[cand_idx] is not None:
                        approach_goal[6:] = openpose_start_list[cand_idx]
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

                    # (c+d) reorient + descent — straight-down.
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
                    x_center = 0.5 * (float(X_GRID_PRE[0]) + float(X_GRID_PRE[-1]))
                    sorted_cands = sorted(
                        sorted_cands, key=lambda sc: abs(sc["x"] - x_center))
                    reorient_traj_cand = None
                    descent_traj_cand = None
                    pre_info_cand = None
                    descent_info_cand = None
                    last_reorient_fail = None
                    last_descent_fail = None
                    for sc in sorted_cands:
                        sx, syaw, scyl = sc["x"], sc["yaw"], sc["cyl_yaw"]
                        x1 = np.array([sx]); y1 = np.array([syaw])
                        cyl1 = (np.array([scyl])
                                if CYL_YAW_GRID_PRE is not None else None)
                        r_traj, r_info = planner.plan_obj_placement(
                            scene_lift_pre, cur_qpos_reorient, T_obj_in_wrist_cand,
                            R_target_obj_world_pre, obj_target_pos_world_pre,
                            hold_hand_qpos=pregrasp_cand,
                            x_grid=x1, yaw_grid=y1,
                            cyl_yaw_grid=cyl1,
                            cyl_axis_local=CYL_AXIS_LOCAL,
                            skip_plan=False)
                        if r_traj is None:
                            last_reorient_fail = (sc, r_info)
                            continue
                        cur_qpos_descent = r_traj[-1].copy()
                        d_traj, d_info = planner.plan_obj_placement(
                            scene_lift_pre, cur_qpos_descent, T_obj_in_wrist_cand,
                            R_target_obj_world_pre, obj_target_pos_descent,
                            hold_hand_qpos=pregrasp_cand,
                            x_grid=x1, yaw_grid=y1,
                            cyl_yaw_grid=cyl1,
                            cyl_axis_local=CYL_AXIS_LOCAL,
                            skip_plan=False)
                        if d_traj is None:
                            last_descent_fail = (sc, d_info)
                            continue
                        reorient_traj_cand = r_traj
                        descent_traj_cand = d_traj
                        pre_info_cand = r_info
                        descent_info_cand = d_info
                        break

                    if reorient_traj_cand is None:
                        if last_descent_fail is not None:
                            n_descent_fail += 1
                            sc, info_d = last_descent_fail
                            chosen_yaw_d = sc["yaw"]
                            cy_d, sy_d = np.cos(chosen_yaw_d), np.sin(chosen_yaw_d)
                            Rz_d = np.array([[cy_d, -sy_d, 0.0],
                                             [sy_d,  cy_d, 0.0],
                                             [0.0,   0.0,  1.0]])
                            if CYL_AXIS_LOCAL is not None:
                                R_cyl_d = R.from_rotvec(
                                    CYL_AXIS_LOCAL * float(sc["cyl_yaw"])
                                ).as_matrix()
                            else:
                                R_cyl_d = np.eye(3)
                            T_obj_reorient_end = np.eye(4)
                            T_obj_reorient_end[:3, :3] = (
                                Rz_d @ R_target_obj_world_pre @ R_cyl_d)
                            T_obj_reorient_end[0, 3] = float(sc["x"])
                            T_obj_reorient_end[1, 3] = 0.0
                            T_obj_reorient_end[2, 3] = (
                                float(target_tabletop_robot[2, 3])
                                + TABLE_SURFACE_Z + LIFT_HEIGHT_M)
                            cand_log.append({
                                "cand_idx": cand_idx, "stage": "descent",
                                "reason": info_d.get("reason"),
                                "T_wrist_target": info_d.get("T_wrist_target"),
                                "T_obj_at_fail": T_obj_reorient_end,
                                "last_qpos": cur_qpos_reorient.copy(),
                                "grasp_qpos": grasp_cand,
                            })
                            print(f"  cand#{cand_idx}: descent FAIL on every "
                                  f"reorient-feasible (x, yaw) "
                                  f"({len(sorted_cands)} tried)")
                        else:
                            n_reorient_fail += 1
                            cand_log.append({
                                "cand_idx": cand_idx, "stage": "reorient",
                                "reason": sorted_info.get("reason"),
                                "T_wrist_target": T_wrist_lift_cand,
                                "T_obj_at_fail": T_wrist_lift_cand @ T_obj_in_wrist_cand,
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
                    openpose_target_chosen = openpose_target_list[cand_idx]
                    wrist_grasp_chosen = wrist_grasp_cand
                    scene_info_chosen = ik_res["scene_info"][cand_idx]
                    result = PlanResult(
                        success=True, traj=approach_traj_cand,
                        wrist_se3=wrist_grasp_cand,
                        pregrasp_pose=pregrasp_cand,
                        grasp_pose=grasp_cand,
                        scene_info=scene_info_chosen,
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
                    # Only approach is yaw-sensitive — once lifted, yaw is
                    # free. So suggest a yaw rotation only when approach is
                    # contributing to the failure.
                    if n_approach_fail > 0:
                        _found_yaw_deg, _found_x, _found_n_ok = _yaw_search_and_print_cmd(
                            planner, scene_cfg, seeds, args,
                            prev_n_ok=len(ik_ok))
                        rec["yaw_suggestion_deg"] = _found_yaw_deg
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
                    "n_approach_fail_before_chosen": n_approach_fail,
                    "n_lift_fail_before_chosen": n_lift_fail,
                    "n_reorient_fail_before_chosen": n_reorient_fail,
                    "n_descent_fail_before_chosen": n_descent_fail,
                }
                print(f"    plan: {rec['timing']['plan_s']}s  cand#{chosen_cand_idx}  "
                      f"x={pre_info['chosen_x']:.3f} "
                      f"yaw={np.degrees(pre_info['chosen_yaw']):.0f}°  "
                      f"(skipped approach={n_approach_fail}, lift={n_lift_fail}, "
                      f"reorient={n_reorient_fail}, descent={n_descent_fail})")

                np.save(cdir / "plan" / "traj.npy", result.traj)
                np.save(cdir / "plan" / "wrist_se3.npy", result.wrist_se3)
                np.save(cdir / "plan" / "pregrasp_pose.npy", result.pregrasp_pose)
                np.save(cdir / "plan" / "grasp_pose.npy", result.grasp_pose)
                np.save(cdir / "plan" / "lift_traj.npy", lift_traj_chosen)
                np.save(cdir / "plan" / "reorient_traj.npy", reorient_traj_chosen)
                np.save(cdir / "plan" / "descent_traj.npy", descent_traj_chosen)

                if args.viz:
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
                        lift_traj_viz = lift_traj_chosen.copy()
                        lift_traj_viz[:, 6:] = grasp_h[None, :]
                        reorient_traj_viz = reorient_traj_chosen.copy()
                        reorient_traj_viz[:, 6:] = grasp_h[None, :]
                        descent_traj_viz = descent_traj_chosen.copy()
                        descent_traj_viz[:, 6:] = grasp_h[None, :]
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
                rcc.start("full", True, video_rel)
                timestamp_monitor.start(os.path.join(raw_dir, "timestamps"))
                sync_generator.start(fps=VIDEO_FPS)
                video_started = True
                executor.start_recording(raw_dir)

                # 5. Execute (init → approach → pregrasp → grasp → squeeze).
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

                T_obj_grasp = cart2se3(scene_cfg["mesh"]["target"]["pose"])
                T_obj_in_wrist = np.linalg.inv(result.wrist_se3) @ T_obj_grasp
                lift_traj = lift_traj_chosen
                scene_lift = {"mesh": {}, "cuboid": dict(scene_cfg["cuboid"])}

                # 5b. Joint-space lift — replay pre-planned lift trajectory.
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

                # 6. Charuco lift-check via snapshot_daemon.
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
                    rec["progress"]["reorient"] = "skipped"
                    rec["progress"]["descent"] = "skipped"
                    rec["progress"]["release"] = "skipped"
                    rec["progress"]["post_perception"] = "skipped"
                    rec["status"] = "charuco_fail"
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
                print(f"[cycle {cycle}] Reorient (target_j={args.target_j})  "
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
                hand_traj = np.tile(s_hand[None], (len(reorient_traj), 1))
                executor._move_joints(arm_traj, hand_traj)
                rec["timing"]["reorient_exec_s"] = round(time.time() - t1, 2)
                rec["progress"]["reorient"] = (
                    f"ok x={pre_info['chosen_x']:.3f} "
                    f"yaw={np.degrees(pre_info['chosen_yaw']):.0f}°"
                )
                print(f"    [reorient] OK  exec={rec['timing']['reorient_exec_s']}s")

                # 8. Descent — replay pre-planned descent trajectory.
                descend = LIFT_HEIGHT_M - RELEASE_HEIGHT_M
                print(f"[cycle {cycle}] Descent ({descend*100:.0f}cm)...")
                descent_traj = descent_traj_chosen
                t1 = time.time()
                executor._log_state("descent")
                arm_traj = descent_traj[:, :6]
                hand_traj = np.tile(s_hand[None], (len(descent_traj), 1))
                executor._move_joints(arm_traj, hand_traj)
                rec["timing"]["descent_s"] = round(time.time() - t1, 2)
                rec["progress"]["descent"] = "ok"

                T_wrist_release = (executor.arm.get_data()["position"]
                                   @ executor._link6_to_wrist)
                planned_obj_pose = T_wrist_release @ T_obj_in_wrist

                # 9. Release (squeeze -> grasp -> pregrasp).
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

                # 9b. Swap result.openpose_pose to the TARGET-side openpose so
                #     reset_hybrid's slow pregrasp→openpose interp uses the
                #     pose-correct config for releasing at the target tabletop.
                if openpose_target_chosen is not None:
                    result.openpose_pose = openpose_target_chosen

                # 10. Reset_hybrid: slow pregrasp→openpose interp internally,
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

                # 11. Stop recordings → video stop → image snap → stream.
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

                # 12. Post-drop perception + tabletop_after + drop_quality.
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
                        f"idx={tb_after['idx']} ({tb_after['rot_err_deg']:.1f}°)"
                        if tb_after else "no_tabletop_data"
                    )
                    if tb_after:
                        j_int_after = _pose_int_from_filename(tb_after["filename"])
                        rec["j_int_after"] = j_int_after
                        rec["tabletop_hit_target"] = bool(
                            j_int_after == args.target_j
                        )
                        if tb_before:
                            rec["tabletop_changed"] = bool(
                                tb_after["idx"] != tb_before["idx"]
                            )
                        print(f"    [tabletop after]  j={j_int_after} "
                              f"({tb_after['filename']}) err={tb_after['rot_err_deg']:.1f}° "
                              f"target_j={args.target_j} "
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
                          f"rot={rot_err:.1f}°  z={z_drop*1000:.1f}mm")

            except _SoftSkip:
                # Stop the whole cycle loop on first plan fail so user can
                # inspect/copy log instead of letting noise scroll past.
                print(f"\n[cycle {cycle}] plan_failed — stopping cycle loop "
                      f"(was: continue → next cycle).")
                try:
                    cmd = input("    Press Enter to retry same cycle, "
                                "'q' to quit: ").strip().lower()
                except KeyboardInterrupt:
                    cmd = "q"
                if cmd == "q":
                    break
                else:
                    continue
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
        print(f"\n{'='*60}\nSUMMARY: {args.obj} target_j={args.target_j} "
              f"h_cm={h_cm} × {len(trials)} trials")
        for c in trials:
            tag = c.get("status", "?")
            extra = ""
            plan_s = (c.get("timing") or {}).get("plan_s")
            if plan_s is not None:
                extra += f"  plan={plan_s}s"
            dq = c.get("drop_quality")
            if dq:
                extra += f"  drop t={dq['trans_err_m']*1000:.0f}mm r={dq['rot_err_deg']:.0f}°"
            if c.get("tabletop_hit_target") is not None:
                extra += f"  hit={c['tabletop_hit_target']}"
            print(f"  {c.get('trial_ts', '?')}: {tag}{extra}")
        _write_json(summary_path, trials)
        print(f"  summary -> {summary_path}")
        _do_cleanup()


if __name__ == "__main__":
    main()
