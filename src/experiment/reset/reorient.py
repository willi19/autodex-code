#!/usr/bin/env python3
"""Reorient policy: drop-style chain plan (approach → lift → reorient → descent).

The live object/tabletop pose remains v8/object_processing.  Existing reset
candidates use the legacy paradex tabletop numbering, so every reset cell is
selected through ``autodex.utils.tabletop_map``'s strict pose mapping before
being used.  The full v8 approach/lift/reorient/descent preflight remains the
final safety gate before any physical grasp.

Per-cycle (mirrors ``reorient_drop.py`` structure):
    perception -> classify tabletop_before (i)
    -> if i == target_j: skip
    -> map v8 cell {i}_{target_j} to its validated legacy reset cell
       and load its seeds for the smallest reorient_{h_cm}
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

``reorient_{h_cm}`` is the BODex-generated descent height: ``h_cm=0`` lands
the object on the table and ``h_cm=8`` releases 8 cm above it. Candidate roots
are attempted in the order ``reset_0 → reset_4 → reset_8 → reset_12``.

Prerequisites:
    bash scripts/init_daemons.sh start
    bash scripts/snapshot_daemons.sh start

Usage:
    # v8 reset collection path
    python src/experiment/reset/reorient.py --obj donut --target_j 2 --auto

    # Franka FR3 + right Inspire hand
    python src/experiment/reset/reorient.py --obj donut --target_j 2 --auto \\
        --arm franka --hand inspire
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

from autodex.utils.path import (
    project_dir, get_obj_root, get_reset_candidate_root,
    iter_reset_candidate_roots,
)
from autodex.utils.conversion import cart2se3, se32cart
from autodex.utils.robot_config import CHARUCO_BOARD_11_CENTER_XY
from autodex.planner import GraspPlanner
from autodex.planner.planner import (
    PlanResult, _to_curobo_world, _to_curobo_pose,
)
from autodex.planner.obstacles import TABLE_CUBOID, add_obstacles
from autodex.planner.visualizer import ScenePlanVisualizer
from autodex.executor.real import RealExecutor
from src.execution.franka_executor import (
    FrankaExecutor, PLACE_VERTICAL_TRAVEL_M,
    VERTICAL_STROKE_Z_TOL_M,
)
from autodex.perception.init_orchestrator import InitOrchestrator
from autodex.perception.snapshot_orchestrator import SnapshotOrchestrator

from curobo.geom.types import WorldConfig

from src.execution.scene_cfg import pose_world_to_scene_cfg
from autodex.utils.symmetry import get_cyl_axis_local, get_cyl_yaw_grid
from src.execution.run_auto import (
    DEFAULT_PC_LIST, ASSETS_BASE, CAM_PARAM_ROOT, _load_calib,
)
from src.execution.label import auto_label_charuco
from src.experiment.reset.tabletop_pose import classify_tabletop_pose
from src.demo.p2.recording import resolve_signal_generator_params

from paradex.visualization.visualizer.viser import ViserViewer

_URDF_ROOT = Path.home() / "shared_data" / "AutoDex" / "content" / "assets" / "robot"
URDF_BY_HAND_VIZ = {
    "inspire_left": _URDF_ROOT / "inspire_left_description" / "xarm_inspire_left.urdf",
    "inspire":      _URDF_ROOT / "inspire_description"      / "xarm_inspire.urdf",
    "allegro":      _URDF_ROOT / "allegro_description"      / "xarm_allegro.urdf",
}
FR3_INSPIRE_URDF_VIZ = _URDF_ROOT / "fr3_inspire_description" / "fr3_inspire.urdf"
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


def _planner_robot(arm: str, hand: str) -> str:
    """Return the GraspPlanner robot key for the selected physical arm."""
    if arm == "franka":
        if hand != "inspire":
            raise ValueError(
                "FR3 reorient currently supports the right Inspire hand only; "
                "use --arm franka --hand inspire"
            )
        return "fr3_inspire"
    return hand


def _viz_urdf(arm: str, hand: str) -> Path:
    """Full arm+hand URDF matching the planner/executor pair."""
    if arm == "franka":
        return FR3_INSPIRE_URDF_VIZ
    return URDF_BY_HAND_VIZ[hand]


def _executor_log(executor, state: str) -> None:
    """Log an execution phase on either legacy XArm or Franka executor."""
    log = getattr(executor, "_log_state", None)
    if log is None:
        log = getattr(executor, "_log")
    log(state)


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
# Board id lives in src/execution/label.py — one place to swap.
from src.execution.label import CHARUCO_BOARD  # noqa: E402


def _rcc_start(rcc, mode, sync_mode, save_path=None, fps=30):
    """Start a capture, translating the retired 'stream'/'video' modes.

    paradex's camera API dropped both: a capture arms in 'acquire' and its
    outputs are toggled as SINKS. The capture PCs reject the old names, and the
    rejection LATCHES an error on every camera that only a daemon-side reload
    clears -- so one call from a stale script poisons the next run too.
    """
    if mode == "stream":
        rcc.arm(syncMode=sync_mode, fps=fps)
        rcc.set_stream(True)
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
    # Board 11 was measured live at y=0.153 m.  Search on that centreline
    # and omit x values that fall outside the board rather than proposing an
    # unreachable-looking pose that physically lands off the board.
    _target_y = float(CHARUCO_BOARD_11_CENTER_XY[1])
    _x_grid = np.arange(0.50, 0.71, 0.05)
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
            _new_obj[1, 3] = _target_y
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
        _best_obj[0, 3] = float(_found_x); _best_obj[1, 3] = _target_y
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
            print(f"    [pose_search] move obj to "
                  f"(x={_found_x:.2f}, y={_target_y:.3f}) "
                  f"+ rotate {_found_yaw_deg:.0f}° → "
                  f"{_found_n_ok}/{len(_orig_wrist)} IK-feasible "
                  f"(prev={prev_n_ok}, approach OK)")
            print(f"    [run this]")
            print(f"      python src/execution/rotate_obj_yaw.py "
                  f"--obj {args.obj} --hand {args.hand} --arm {args.arm} "
                  f"--target_yaw_deg {_found_yaw_deg:.0f} "
                  f"--target_x {_found_x:.2f} --target_y {_target_y:.3f}")
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


def _legacy_reset_cell_indices(obj: str, i_int: int, j_int: int,
                               obj_root: str) -> tuple[int, int]:
    """Return the verified legacy reset-cell indices for one v8 transition."""
    from autodex.utils.tabletop_map import to_reset_index

    return (
        to_reset_index(obj, i_int, obj_root, allow_partial=True),
        to_reset_index(obj, j_int, obj_root, allow_partial=True),
    )


def _autoselect_h_cm(hand: str, obj: str, target_j: int | None = None,
                     *, obj_root: str, version: str = "v8") -> int | None:
    """Pick the smallest ``reorient_{h_cm}`` folder that contains at least one
    ``*_{target_j}`` cell. If ``target_j`` is None, fall back to the smallest
    ``reorient_{h_cm}`` folder regardless of cell contents.

    ``target_j`` is a v8 tabletop filename integer and is resolved to its
    verified legacy reset-cell ID before inspecting the existing pool."""
    legacy_target = None
    if target_j is not None:
        try:
            # The source index is irrelevant when selecting by target only;
            # map the target directly so standalone startup can find a cell.
            from autodex.utils.tabletop_map import to_reset_index
            legacy_target = to_reset_index(
                obj, target_j, obj_root, allow_partial=True)
        except (ValueError, KeyError):
            return None
    cands = []
    for h, root in iter_reset_candidate_roots(hand, version=version):
        p = Path(root) / obj / f"reorient_{h}"
        if not p.is_dir():
            continue
        if legacy_target is not None:
            has_target = any(
                c.is_dir() and c.name.endswith(f"_{legacy_target}")
                and c.name.split("_", 1)[0].isdigit()
                for c in p.iterdir()
            )
            if not has_target:
                continue
        cands.append((h, p))
    if not cands:
        return None
    return sorted(cands, key=lambda kv: kv[0])[0][0]


def _autoselect_h_cm_for_cell(hand: str, obj: str, i_int: int,
                               j_int: int,
                               *, obj_root: str,
                               version: str = "v8") -> int | None:
    """Smallest ``reorient_{h_cm}`` folder containing the EXACT cell
    ``{i_int}_{j_int}``.

    Cells are height-escalated per pair (``reset_escalated_manifest.json``), so
    two transitions to the same ``target_j`` can live at different heights —
    e.g. green_attached_container has ``2_0`` at reorient_0 but ``6_0`` at
    reorient_12. Selecting h from ``target_j`` alone (``_autoselect_h_cm``)
    therefore misses cells that exist at another height.
    """
    hs = _available_h_cm_for_cell(
        hand, obj, i_int, j_int, obj_root=obj_root, version=version)
    return hs[0] if hs else None


def _available_h_cm_for_cell(hand: str, obj: str, i_int: int, j_int: int,
                             *, obj_root: str,
                             version: str = "v8") -> list[int]:
    """Legacy reset heights for a strictly mapped v8 tabletop transition."""
    try:
        legacy_i, legacy_j = _legacy_reset_cell_indices(
            obj, i_int, j_int, obj_root)
    except (ValueError, KeyError):
        return []
    hs = []
    for h, root in iter_reset_candidate_roots(hand, version=version):
        p = Path(root) / obj / f"reorient_{h}"
        if not p.is_dir():
            continue
        if (p / f"{legacy_i}_{legacy_j}").is_dir():
            hs.append(h)
    return hs


def _load_target_tabletop_pose(obj: str, target_j: int,
                               obj_root: str) -> np.ndarray:
    """Load 4x4 tabletop pose (robot frame) for filename int ``target_j``.
    Files are zero-padded 3-digit (``002.npy``).

    ``obj_root`` must match the pool that produced ``target_j`` — a v8
    target_j indexes object_processing stems, which differ from the legacy
    paradex ones (see ``autodex.utils.path.get_obj_root``)."""
    if not obj_root:
        raise ValueError("obj_root is required for v8 reorientation")
    root = obj_root
    fname = f"{target_j:03d}.npy"
    p = Path(root) / obj / "processed_data" / "info" / "tabletop" / fname
    if not p.exists():
        # try un-padded fallback
        p2 = Path(root) / obj / "processed_data" / "info" / "tabletop" / f"{target_j}.npy"
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
                      T_obj_world: np.ndarray,
                      *, obj_root: str, version: str = "v8") -> dict | None:
    """Load legacy reset seeds for a verified v8 cell under reorient_{h_cm}.

    Files on disk are in **object frame**; this transforms wrist_se3 to world
    via ``T_wrist_world = T_obj_world @ wrist_se3_obj``. Returns ``None`` if
    the cell directory does not exist or has no grasp subfolders.

    Also loads the legacy-indexed start/target open poses that belong to the
    mapped reset cell.  The physical target pose is still supplied separately
    from the v8 table-top asset tree to the full-chain planner.

    Output dict mirrors ``GraspPlanner.solve_ik``'s candidate-related
    fields plus ``openpose_start`` and ``openpose_target``.
    """
    try:
        legacy_i, legacy_j = _legacy_reset_cell_indices(
            obj, i_int, j_int, obj_root)
    except (ValueError, KeyError):
        return None
    cell_dir = (Path(get_reset_candidate_root(hand, h_cm, version=version)) / obj
                / f"reorient_{h_cm}" / f"{legacy_i}_{legacy_j}")
    if not cell_dir.exists():
        return None
    # Order by stats.json priority desc (Laplace-smoothed success rate),
    # tie-break by numeric grasp_id asc. Falls back to numeric order if
    # stats.json is missing (read_grasp_stats returns (0, 0)).
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
        op_s_path = g / f"openpose_{legacy_i:03d}.npy"
        op_t_path = g / f"openpose_{legacy_j:03d}.npy"
        op_start.append(np.load(op_s_path) if op_s_path.exists() else None)
        op_target.append(np.load(op_t_path) if op_t_path.exists() else None)
    wrist_world = T_obj_world[None] @ wrist_obj
    scene_info = [{
        "grasp_idx": g.name,
        "cell": f"{legacy_i}_{legacy_j}",
        "v8_cell": f"{i_int}_{j_int}",
        "legacy_cell": f"{legacy_i}_{legacy_j}",
        "candidate_contract": "legacy_mapped_reset",
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
        "legacy_i": legacy_i,
        "legacy_j": legacy_j,
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
    if "inspire" in planner._hand:
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
                    # Reset candidates store a hand-relative wrist transform,
                    # so their 6-DoF grasp geometry can be solved on either
                    # XArm (6 joints) or FR3 (7 joints).  Do not hard-code
                    # the old XArm arm/hand split here.
                    adof = planner._n_arm
                    arm_q = q_sol[i, :adof].copy()
                    planner._snap_arm(arm_q, planner._init_state[:adof])
                    ik_qpos[idx, :adof] = arm_q
                    ik_qpos[idx, adof:] = pregrasp[idx]
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


def _franka_fk_xyz(planner: GraspPlanner, arm_traj: np.ndarray) -> np.ndarray:
    """Planner-frame wrist FK for a batched FR3 arm trajectory."""
    arm = np.asarray(arm_traj, dtype=np.float32)
    n_arm = int(planner._n_arm)
    q = np.tile(np.asarray(planner._init_state, dtype=np.float32),
                (len(arm), 1))
    q[:, :n_arm] = arm[:, :n_arm]
    kin = planner._motion_gen.kinematics.get_state(torch.tensor(
        q, dtype=torch.float32, device=planner._tensor_args.device))
    return np.asarray(kin.ee_position.detach().cpu().numpy(), dtype=np.float64)


def _plan_franka_verified_vertical_stroke(
    planner: GraspPlanner,
    start_full: np.ndarray,
    wrist_start: np.ndarray,
    wrist_end: np.ndarray,
    scene_cfg: dict,
    include_obj_obstacle: bool,
    label: str,
) -> np.ndarray:
    """Preflight one geometric 10cm vertical stroke before a real grasp.

    This mirrors FrankaExecutor's runtime primitive: solve the full stroke
    once, inspect all returned FK waypoints, and reject it before closing the
    real gripper if it bows away from the required vertical path.
    """
    start = np.asarray(wrist_start, dtype=np.float64)
    end = np.asarray(wrist_end, dtype=np.float64)
    delta = end[:3, 3] - start[:3, 3]
    if abs(abs(delta[2]) - PLACE_VERTICAL_TRAVEL_M) > VERTICAL_STROKE_Z_TOL_M:
        raise RuntimeError(
                f"{label}: expected a 10cm vertical stroke, got "
                f"delta={delta.round(5).tolist()}")
    direction = 1 if delta[2] > 0 else -1
    traj = planner.plan_pose_constrained(
        np.asarray(start_full, dtype=np.float32), end,
        hold_vec_weight=[0, 0, 0, 0, 0, 0],
        scene_cfg=scene_cfg, include_obj_obstacle=include_obj_obstacle)
    if traj is None:
        raise RuntimeError(f"{label}: 10cm vertical-stroke preflight failed")
    traj = np.asarray(traj)
    xyz = _franka_fk_xyz(planner, traj[:, :planner._n_arm])
    dz = float(direction * (xyz[-1, 2] - xyz[0, 2]))
    dz_steps = direction * np.diff(xyz[:, 2])
    if (dz < PLACE_VERTICAL_TRAVEL_M - VERTICAL_STROKE_Z_TOL_M
            or (len(dz_steps) and np.min(dz_steps) < -VERTICAL_STROKE_Z_TOL_M)):
        raise RuntimeError(
            f"{label}: invalid vertical-stroke z motion "
            f"(signed_dz={dz * 1000:.1f}mm)")
    return traj


def _preflight_franka_place_chain(
    planner: GraspPlanner,
    scene_cfg: dict,
    result: PlanResult,
    reorient_traj: np.ndarray,
    wrist_release: np.ndarray,
    obj_in_wrist: np.ndarray,
    release_hand: np.ndarray,
) -> dict:
    """Require place/down/release-up/retract feasibility before grasping."""
    wrist_release = np.asarray(wrist_release, dtype=np.float64)
    wrist_preplace = wrist_release.copy()
    wrist_preplace[2, 3] += PLACE_VERTICAL_TRAVEL_M
    held_scene = {"mesh": {}, "cuboid": dict(scene_cfg["cuboid"])}
    start_full = np.concatenate([
        np.asarray(reorient_traj[-1, :planner._n_arm], dtype=np.float32),
        np.asarray(result.grasp_pose, dtype=np.float32),
    ])
    preplace = planner.plan_pose_constrained(
        start_full, wrist_preplace, hold_vec_weight=[0, 0, 0, 0, 0, 0],
        scene_cfg=held_scene, include_obj_obstacle=False)
    if preplace is None:
        raise RuntimeError("preplace_10cm failed")
    descend_start = np.concatenate([
        np.asarray(preplace[-1, :planner._n_arm], dtype=np.float32),
        np.asarray(result.grasp_pose, dtype=np.float32),
    ])
    descend = _plan_franka_verified_vertical_stroke(
        planner, descend_start, wrist_preplace, wrist_release, held_scene,
        include_obj_obstacle=False, label="preflight place descent")

    T_obj_released = wrist_release @ obj_in_wrist
    placed_scene = dict(scene_cfg)
    placed_scene["mesh"] = dict(scene_cfg.get("mesh", {}))
    placed_scene["mesh"]["target"] = dict(scene_cfg["mesh"]["target"])
    placed_scene["mesh"]["target"]["pose"] = se32cart(T_obj_released).tolist()
    lift_start = np.concatenate([
        np.asarray(descend[-1, :planner._n_arm], dtype=np.float32),
        np.asarray(release_hand, dtype=np.float32),
    ])
    post_lift = _plan_franka_verified_vertical_stroke(
        planner, lift_start, wrist_release, wrist_preplace, placed_scene,
        include_obj_obstacle=True, label="preflight post-release lift")
    clear_view = np.asarray(planner._init_state[:planner._n_arm], dtype=np.float32).copy()
    clear_view[0] -= np.deg2rad(40.0)
    retract = planner.plan_js_to_init(
        placed_scene, post_lift[-1, :planner._n_arm],
        start_hand_qpos=np.asarray(release_hand, dtype=np.float32),
        goal_arm_qpos=clear_view)
    if retract is None:
        raise RuntimeError("preflight post-release retract failed")
    return {
        "preplace_traj": preplace,
        "descent_traj": descend,
        "post_lift_traj": post_lift,
        "retract_traj": retract,
    }


def _plan_reorient_full_chain(
    *,
    planner: GraspPlanner,
    scene_cfg: dict,
    obj: str,
    pose_robot_before: np.ndarray,
    target_tabletop_robot: np.ndarray,
    seeds: dict,
    planner_robot: str,
    release_height_m: float,
) -> dict:
    """Plan every held-object stage for one reset transition.

    This is the non-hardware part of the standalone reorient policy, factored
    so an already-live :mod:`run_auto` session can use it directly.  A result
    is returned only after one *same* reset grasp has passed approach, the full
    25 cm lift, high reorientation, and final descent.  In particular, a
    grasp-IK result alone is never enough to begin a physical grasp.
    """
    arm_dof = planner._n_arm

    # For a horizontally symmetric object, enumerate equivalent object-frame
    # yaws for the *approach* grasp too.  Carry both start/target openposes
    # through the expansion so the selected candidate stays paired with its
    # correct hand poses.
    cyl_axis_start = get_cyl_axis_local(obj)
    if cyl_axis_start is not None:
        axis_world = pose_robot_before[:3, :3] @ cyl_axis_start
        cyl_grid_start = (None if abs(axis_world[2]) >= 0.95
                          else np.linspace(0, 2 * np.pi, 8, endpoint=False))
    else:
        cyl_grid_start = None
    if cyl_grid_start is not None:
        from autodex.planner.planner import _expand_candidates_cyl
        wrist, pregrasp, grasp, openpose_pairs, scene_info = _expand_candidates_cyl(
            seeds["wrist_se3"], seeds["pregrasp"], seeds["grasp"],
            list(zip(seeds["openpose_start"], seeds["openpose_target"])),
            seeds["scene_info"], pose_robot_before, cyl_axis_start,
            cyl_grid_start,
        )
        seeds = {
            "wrist_se3": wrist,
            "pregrasp": pregrasp,
            "grasp": grasp,
            "openpose_start": [pair[0] for pair in openpose_pairs],
            "openpose_target": [pair[1] for pair in openpose_pairs],
            "scene_info": scene_info,
            "n_total": len(wrist),
        }

    # _ik_check_seeds deliberately reduces collision activation to zero for
    # fingertip-near-table reset IK.  This object is shared with run_auto, so
    # restore the normal motion margin even if reset planning fails before a
    # candidate is selected.
    hand_cfg = planner.HAND_CONFIGS.get(
        planner_robot, planner.HAND_CONFIGS["allegro"])
    try:
        ik_res = _ik_check_seeds(planner, scene_cfg, seeds)
    finally:
        planner._collision_act_dist = hand_cfg[2]
        planner._ik_solver = None
    ik_ok = [int(i) for i in np.flatnonzero(ik_res["ik_success"])]
    # _ik_check_seeds used an IK-only solver; motion generation below starts
    # from the normal trajectory collision margin restored above.
    counts = {
        "n_total": int(ik_res["n_total"]),
        "n_ik": len(ik_ok),
        "n_approach_fail": 0,
        "n_lift_fail": 0,
        "n_reorient_fail": 0,
        "n_descent_fail": 0,
        "n_place_fail": 0,
    }
    if not ik_ok:
        return {
            "success": False,
            "reason": "no_reset_grasp_ik",
            "counts": counts,
            "ik": ik_res,
        }

    openpose_start = seeds.get("openpose_start", [None] * len(seeds["wrist_se3"]))
    openpose_target = seeds.get("openpose_target", [None] * len(seeds["wrist_se3"]))
    obj_grasp = cart2se3(scene_cfg["mesh"]["target"]["pose"])
    scene_lift = {"mesh": {}, "cuboid": dict(scene_cfg["cuboid"])}

    # Restore the hand-specific trajectory collision margin after the
    # deliberately permissive surface-contact IK check.
    planner._collision_act_dist = hand_cfg[2]
    world_approach = _to_curobo_world(scene_cfg)
    planner._motion_gen = None
    planner._init_motion_gen(world_approach)
    planner._cached_world = world_approach

    target_rot = target_tabletop_robot[:3, :3]
    target_high = np.array([
        0.0,
        0.0,
        float(target_tabletop_robot[2, 3]) + TABLE_SURFACE_Z + LIFT_HEIGHT_M,
    ])
    target_release = np.array([
        0.0,
        0.0,
        float(target_tabletop_robot[2, 3]) + TABLE_SURFACE_Z + release_height_m,
    ])
    x_grid = np.arange(0.35, 0.55, 0.05)
    yaw_grid = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    cyl_axis_target = get_cyl_axis_local(obj)
    cyl_yaw_grid = get_cyl_yaw_grid(obj)
    if (cyl_axis_target is not None and cyl_yaw_grid is not None
            and abs((target_rot @ cyl_axis_target)[2]) >= 0.95):
        cyl_yaw_grid = np.array([0.0])

    for cand_idx in ik_ok:
        wrist_grasp = ik_res["wrist_se3"][cand_idx]
        pregrasp = ik_res["pregrasp"][cand_idx]
        grasp = ik_res["grasp"][cand_idx]
        obj_in_wrist = np.linalg.inv(wrist_grasp) @ obj_grasp

        if planner._world_structure_changed(world_approach):
            planner._update_world(world_approach)
            planner._cached_world = world_approach
        approach_goal = ik_res["ik_qpos"][cand_idx].copy()
        if openpose_start[cand_idx] is not None:
            approach_goal[arm_dof:] = openpose_start[cand_idx]
        approach_ok, approach_traj = planner._refine_fingers(
            planner._init_state, approach_goal)
        if not approach_ok:
            counts["n_approach_fail"] += 1
            continue

        wrist_lift = wrist_grasp.copy()
        wrist_lift[2, 3] += LIFT_HEIGHT_M
        lift_start = np.concatenate([approach_traj[-1, :arm_dof], pregrasp])
        lift_traj, lift_info = planner.plan_wrist_reorient(
            scene_lift, lift_start, wrist_lift, hold_hand_qpos=pregrasp,
            n_yaw=8)
        if lift_traj is None:
            counts["n_lift_fail"] += 1
            continue

        # Enumerate the cheap target-pose candidates first, then plan both
        # carried-object segments for each exact (x, yaw, cyl_yaw) choice.
        reorient_start = lift_traj[-1].copy()
        _, sorted_info = planner.plan_obj_placement(
            scene_lift, reorient_start, obj_in_wrist, target_rot, target_high,
            hold_hand_qpos=pregrasp, x_grid=x_grid, yaw_grid=yaw_grid,
            cyl_yaw_grid=cyl_yaw_grid, cyl_axis_local=cyl_axis_target,
            skip_plan=True)
        sorted_candidates = (sorted_info or {}).get("sorted_candidates", [])
        x_center = 0.5 * (float(x_grid[0]) + float(x_grid[-1]))
        sorted_candidates = sorted(
            sorted_candidates, key=lambda candidate: abs(candidate["x"] - x_center))
        any_reorient_path = False
        for candidate in sorted_candidates:
            one_x = np.array([candidate["x"]])
            one_yaw = np.array([candidate["yaw"]])
            one_cyl = (np.array([candidate["cyl_yaw"]])
                       if cyl_yaw_grid is not None else None)
            reorient_traj, reorient_info = planner.plan_obj_placement(
                scene_lift, reorient_start, obj_in_wrist, target_rot, target_high,
                hold_hand_qpos=pregrasp, x_grid=one_x, yaw_grid=one_yaw,
                cyl_yaw_grid=one_cyl, cyl_axis_local=cyl_axis_target,
                skip_plan=False)
            if reorient_traj is None:
                continue
            any_reorient_path = True
            descent_traj, descent_info = planner.plan_obj_placement(
                scene_lift, reorient_traj[-1].copy(), obj_in_wrist,
                target_rot, target_release, hold_hand_qpos=pregrasp,
                x_grid=one_x, yaw_grid=one_yaw, cyl_yaw_grid=one_cyl,
                cyl_axis_local=cyl_axis_target, skip_plan=False)
            if descent_traj is None:
                continue

            result = PlanResult(
                success=True,
                traj=approach_traj,
                wrist_se3=wrist_grasp,
                pregrasp_pose=pregrasp,
                grasp_pose=grasp,
                scene_info=ik_res["scene_info"][cand_idx],
                timing={"candidate_idx": cand_idx, **counts},
            )
            placement_preflight = None
            if planner_robot == "fr3_inspire":
                # Do not let a grasp get as far as the real robot until the
                # *entire* placement exit has proved feasible.  In
                # particular, a nominal object-placement descent is not
                # sufficient: the actual policy uses a vertical -10 cm
                # stroke, opens the hand, then must make a vertical +10 cm
                # stroke and clear the released object.
                release_hand = (openpose_target[cand_idx]
                                if openpose_target[cand_idx] is not None
                                else pregrasp)
                try:
                    placement_preflight = _preflight_franka_place_chain(
                        planner, scene_cfg, result, reorient_traj,
                        np.asarray(descent_info["T_wrist_target"],
                                   dtype=np.float64),
                        obj_in_wrist, release_hand)
                except Exception as exc:
                    counts["n_place_fail"] += 1
                    print("    [reorient preflight] reject candidate "
                          f"{cand_idx}: full place/release/exit chain failed "
                          f"({exc!r})")
                    continue
            return {
                "success": True,
                "result": result,
                "lift_traj": lift_traj,
                "reorient_traj": reorient_traj,
                "descent_traj": descent_traj,
                "obj_in_wrist": obj_in_wrist,
                "openpose_target": openpose_target[cand_idx],
                "pre_info": reorient_info,
                "descent_info": descent_info,
                "descent_wrist_target": np.asarray(
                    descent_info["T_wrist_target"], dtype=np.float64),
                "placement_preflight": placement_preflight,
                "counts": counts,
                "ik": ik_res,
                "release_height_m": release_height_m,
            }

        if any_reorient_path:
            counts["n_descent_fail"] += 1
        else:
            counts["n_reorient_fail"] += 1

    return {
        "success": False,
        "reason": "no_reset_grasp_with_full_chain",
        "counts": counts,
        "ik": ik_res,
    }


def reorient_from_live_scene(
    *,
    obj: str,
    hand: str,
    arm: str,
    target_j: int,
    planner: GraspPlanner,
    executor,
    rcc,
    scene_cfg: dict,
    obj_root: str,
    grasp_version: str,
    lift_label_rel: str,
    lift_label_abs: str,
    held_speed_scale: float = 0.25,
) -> dict:
    """Execute a reset transition with the already-live run_auto resources.

    The caller supplies a table-only scene built from its current perception,
    plus paths under the current run_auto trial directory for the lift Charuco
    check.  No camera controller, FoundPose orchestrator, CUDA planner or
    robot executor is constructed here.  A normal run_auto perception occurs
    once after a successful placement, which both avoids duplicate FoundPose
    work and validates the new tabletop before collection resumes.
    """
    if held_speed_scale <= 0:
        raise ValueError("held_speed_scale must be positive")
    if grasp_version != "v8":
        return {"success": False, "reason": "reorient_requires_v8_assets"}
    if arm == "franka" and hand != "inspire":
        return {"success": False, "reason": "unsupported_franka_hand"}

    pose_robot_before = cart2se3(scene_cfg["mesh"]["target"]["pose"])
    tabletop_before = classify_tabletop_pose(pose_robot_before, obj, obj_root)
    if tabletop_before is None:
        return {"success": False, "reason": "reorient_tabletop_unclassified"}
    i_int = _pose_int_from_filename(tabletop_before["filename"])
    if i_int == target_j:
        return {
            "success": True,
            "reason": "already_at_reorient_target",
            "i_int": i_int,
            "target_j": target_j,
        }

    try:
        legacy_i, legacy_j = _legacy_reset_cell_indices(
            obj, i_int, target_j, obj_root)
    except (ValueError, KeyError) as exc:
        return {
            "success": False,
            "reason": "reorient_legacy_mapping_unavailable",
            "i_int": i_int,
            "target_j": target_j,
            "mapping_error": str(exc),
        }
    print(f"[reorient] v8 cell {i_int}_{target_j} -> legacy reset cell "
          f"{legacy_i}_{legacy_j}")

    available_h_cm = _available_h_cm_for_cell(
        hand, obj, i_int, target_j, obj_root=obj_root,
        version=grasp_version)
    if not available_h_cm:
        return {
            "success": False,
            "reason": "reorient_seed_cell_missing",
            "i_int": i_int,
            "target_j": target_j,
        }
    target_pose = _load_target_tabletop_pose(obj, target_j, obj_root)
    planner_robot = _planner_robot(arm, hand)
    height_attempts = []
    plan = None
    h_cm = None
    for candidate_h_cm in available_h_cm:
        seeds = _load_reset_seeds(
            hand, obj, candidate_h_cm, i_int, target_j, pose_robot_before,
            obj_root=obj_root, version=grasp_version)
        if seeds is None:
            height_attempts.append({
                "h_cm": candidate_h_cm,
                "reason": "reorient_seeds_missing",
            })
            continue
        candidate_plan = _plan_reorient_full_chain(
            planner=planner, scene_cfg=scene_cfg, obj=obj,
            pose_robot_before=pose_robot_before,
            target_tabletop_robot=target_pose, seeds=seeds,
            planner_robot=planner_robot,
            release_height_m=candidate_h_cm / 100.0,
        )
        if candidate_plan["success"]:
            h_cm = candidate_h_cm
            plan = candidate_plan
            break
        height_attempts.append({
            "h_cm": candidate_h_cm,
            "reason": candidate_plan.get("reason", "reorient_plan_failed"),
            "counts": candidate_plan.get("counts"),
        })

    if plan is None or h_cm is None:
        return {
            "success": False,
            "reason": "no_reset_height_with_full_chain",
            "i_int": i_int,
            "target_j": target_j,
            "height_attempts": height_attempts,
        }

    result = plan["result"]
    arm_dof = getattr(executor, "arm_dof", planner._n_arm)
    move_kwargs = {"speed": held_speed_scale} if arm == "franka" else {}
    lift_kwargs = {"held_speed_scale": held_speed_scale} if arm == "franka" else {}
    try:
        rcc.stop()
    except Exception as exc:
        print(f"[reorient] rcc.stop before motion failed: {exc!r}")

    try:
        print(f"[reorient] selected v8 cell {i_int}_{target_j} via legacy "
              f"cell {legacy_i}_{legacy_j}, h={h_cm}cm; "
              "full approach/lift/reorient/place/release/exit preflight passed")
        squeeze_hand = executor.execute(
            result, planner=planner, scene_cfg=scene_cfg, skip_lift=True)
        executor.execute_lift(plan["lift_traj"], squeeze_hand, **lift_kwargs)

        # Retain the standalone policy's critical lift verification but use a
        # one-shot image capture from run_auto's existing controller instead
        # of starting SnapshotOrchestrator/a second capture session.
        rcc.start("image", False, lift_label_rel)
        rcc.stop()
        time.sleep(0.3)
        charuco_ok, charuco_info = auto_label_charuco(
            lift_label_abs, required_board=CHARUCO_BOARD)
        if charuco_ok is not True:
            try:
                executor.reset_fallback(result, planner=planner, scene_cfg=scene_cfg)
            except Exception as reset_exc:
                charuco_info = dict(charuco_info or {})
                charuco_info["reset_fallback_exception"] = repr(reset_exc)
            return {
                "success": False,
                "reason": "reorient_lift_charuco_failed",
                "i_int": i_int,
                "target_j": target_j,
                "h_cm": h_cm,
                "charuco": charuco_info,
                "plan": plan,
            }

        if arm == "franka":
            print(f"[franka] held-object speed scale: {held_speed_scale:.2f} "
                  "(reorient transfer); free-hand moves use the 25cm→10cm profile")
        reorient_hand = np.tile(squeeze_hand[None], (len(plan["reorient_traj"]), 1))
        _executor_log(executor, "reorient")
        executor._move_joints(
            plan["reorient_traj"][:, :arm_dof], reorient_hand, **move_kwargs)
        if arm == "franka":
            # Do not replay the legacy arbitrary descent trajectory on FR3.
            # The common place primitive preflights: collision-free move to
            # target+10cm -> perpendicular 10cm descent -> release. reset()
            # then performs the matching planned 10cm vertical exit.
            _executor_log(executor, "place_vertical")
            place_info = executor.place(
                result, planner=planner, scene_cfg=scene_cfg,
                grasp_wrist=plan["descent_wrist_target"],
                hand_qpos=result.grasp_pose,
                pregrasp_qpos=(plan["openpose_target"]
                               if plan["openpose_target"] is not None
                               else result.pregrasp_pose),
            )
            if not place_info.get("released", False):
                raise RuntimeError(
                    "reorient placement did not reach the 10cm vertical "
                    "release point; object remains held")
        else:
            descend_hand = np.tile(squeeze_hand[None], (len(plan["descent_traj"]), 1))
            _executor_log(executor, "descent")
            executor._move_joints(plan["descent_traj"][:, :arm_dof], descend_hand)
            executor.release(result)
        if plan["openpose_target"] is not None:
            result.openpose_pose = plan["openpose_target"]
        reset_info = executor.reset_hybrid(result, planner, scene_cfg)
    except Exception as exc:
        print(f"[reorient] execution/vertical-placement failed: {exc!r}")
        try:
            executor.reset_fallback(result, planner=planner, scene_cfg=scene_cfg)
        except Exception as reset_exc:
            return {
                "success": False,
                "reason": "reorient_execute_and_recovery_failed",
                "exception": repr(exc),
                "recovery_exception": repr(reset_exc),
                "plan": plan,
            }
        return {
            "success": False,
            "reason": "reorient_execute_failed",
            "exception": repr(exc),
            "plan": plan,
        }

    return {
        "success": True,
        "i_int": i_int,
        "target_j": target_j,
        "legacy_i": legacy_i,
        "legacy_j": legacy_j,
        "candidate_contract": "legacy_mapped_reset",
        "h_cm": h_cm,
        "charuco": charuco_info,
        "plan_counts": plan["counts"],
        "scene_info": result.scene_info,
        "place": place_info if arm == "franka" else None,
        "reset": reset_info,
    }


def _viz_cand_failures(cand_log, obj_name, port, hand, arm, mesh_path,
                       vis_prev=None):
    """Dropdown viewer: selected arm at last_qpos + floating hand at the
    failed wrist target + object at fail-time pose. Blocks on Enter."""
    if vis_prev is not None:
        try: vis_prev.stop_viewer()
        except Exception: pass
    if not cand_log:
        print("[viz] no candidate failures to show")
        return vis_prev
    try:
        urdf_path = _viz_urdf(arm, hand)
        floating_urdf_path = FLOATING_URDF_BY_HAND[hand]
        target_mesh = (trimesh.load(str(mesh_path), process=False)
                       if mesh_path.exists() else None)
        vis = ViserViewer(port_number=port)
        robot_name = "franka" if arm == "franka" else "xarm"
        vis.add_robot(robot_name, str(urdf_path))
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
                try:
                    vis.robot_dict[robot_name].update_cfg(np.asarray(qpos))
                except Exception as ue:
                    print(f"[viz] arm update_cfg: {ue!r}")
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
    parser.add_argument("--arm", choices=["xarm", "franka"], default="xarm",
                        help="physical arm; franka uses the FR3+right-Inspire "
                             "planner/executor path")
    parser.add_argument("--hand", type=str, default="inspire_left",
                        choices=["allegro", "inspire", "inspire_left"])
    parser.add_argument("--target_j", type=int, required=True,
                        help="Target tabletop pose int (folder {i}_{target_j} "
                             "and file {target_j:03d}.npy).")
    parser.add_argument("--auto", action="store_true",
                        help="Skip per-cycle Enter prompt, fully autonomous.")
    parser.add_argument("--version", type=str, default="v8",
                        help="v8 reset/tabletop asset contract (the only "
                             "supported reorientation data format).")
    parser.add_argument("--viz", action="store_true")
    parser.add_argument("--port_viser", type=int, default=8080)
    args = parser.parse_args()

    if args.arm == "franka" and args.hand != "inspire":
        parser.error("--arm franka requires --hand inspire (right Inspire hand)")
    if args.version != "v8":
        parser.error("reorientation supports only --version v8 for live tabletop assets")
    planner_robot = _planner_robot(args.arm, args.hand)

    asset_root = get_obj_root(args.version)

    # Asset sanity.
    mesh_path = Path(asset_root) / args.obj / "raw_mesh" / f"{args.obj}.obj"
    assets_root = ASSETS_BASE / args.obj
    if not mesh_path.exists():
        sys.exit(f"mesh not found: {mesh_path}")
    if not (assets_root / "object_repre/v1" / args.obj / "1/repre.pth").exists():
        sys.exit(f"repre.pth missing for {args.obj}")

    # Reset candidates use legacy tabletop IDs, but only after the target's
    # strict v8→legacy mapping has been verified.
    h_cm = _autoselect_h_cm(args.hand, args.obj, args.target_j,
                            obj_root=asset_root, version=args.version)
    if h_cm is None:
        sys.exit(
            f"no validated legacy reset cell for v8 target_j={args.target_j} under "
            f"{get_reset_candidate_root(args.hand, 0)}/... through "
            f"{get_reset_candidate_root(args.hand, 12)}/..."
        )
    RELEASE_HEIGHT_M = h_cm / 100.0
    print(f"[reset] selected reorient_{h_cm}  "
          f"(RELEASE_HEIGHT_M={RELEASE_HEIGHT_M:.3f}m, LIFT_HEIGHT_M={LIFT_HEIGHT_M:.2f}m)")

    # Target tabletop pose (robot frame, fixed for the whole run).
    target_tabletop_robot = _load_target_tabletop_pose(
        args.obj, args.target_j, asset_root)
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
    _rcc_start(rcc, "stream", False, fps=STREAM_FPS)
    time.sleep(STREAM_WARMUP_S)

    # The UTG900E can be renumbered by Linux (e.g. /dev/usbtmc5 after a USB
    # reconnect).  Reuse the normal P2 resolution rule so Franka reorient
    # does not die before planning because the stale configured node is absent.
    trigger_params, trigger_note = resolve_signal_generator_params(
        network_info["signal_generator"]["param"])
    if trigger_note:
        print(f"[video] {trigger_note}")
    sync_generator = UTGE900(**trigger_params)
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

    print(f"[planner] warming up curobo ({planner_robot})...")
    planner = GraspPlanner(hand=planner_robot)
    from curobo.util.logger import setup_curobo_logger
    setup_curobo_logger("warning")
    print(f"[executor] connecting to {args.arm}...")
    executor = (FrankaExecutor(hand_name=args.hand)
                if args.arm == "franka" else RealExecutor(hand_name=args.hand))
    # Match the normal Franka AutoDex flow: keep the arm out of FoundPose's
    # cameras, then execute() brings it to the identical FR3_INIT state from
    # which the planner generated the approach trajectory.
    if args.arm == "franka":
        executor.set_speed_profile_planner(planner)
        print("[executor] moving once to clear-view home")
        executor.home(clear_view=True)
    arm_dof = planner._n_arm

    trials: list = []
    summary_path = obj_root / "summary.json"
    cycle = 0
    vis = None
    # Standalone mode retries the same live tabletop with the next release
    # height after a full-chain preflight failure.  No robot motion occurs
    # before this promotion, so reset_0 -> reset_4 -> reset_8 -> reset_12 is
    # safe to evaluate in order.
    failed_heights_by_cell: dict[str, set[int]] = {}

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
                "obj": args.obj, "arm": args.arm, "hand": args.hand,
                "planner_robot": planner_robot, "scene": SCENE,
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
                tb_before = classify_tabletop_pose(pose_robot_before, args.obj,
                                                   asset_root)
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

                # 3. Load legacy reset seeds for the mapped v8 cell.
                # ``reset_<h>`` means h centimetres above the floor.  Select
                # only strictly mapped legacy cells in ascending release-height
                # order; a mapping failure leaves the robot stationary.
                cell_key = f"{i_int}_{args.target_j}"
                available_heights = _available_h_cm_for_cell(
                    args.hand, args.obj, i_int, args.target_j,
                    obj_root=asset_root, version=args.version)
                exhausted_heights = failed_heights_by_cell.get(cell_key, set())
                remaining_heights = [h for h in available_heights
                                     if h not in exhausted_heights]
                if not remaining_heights:
                    rec["progress"]["plan"] = f"no_cell ({i_int}_{args.target_j})"
                    rec["status"] = "reset_heights_exhausted"
                    print(f"    no remaining reset height for {cell_key}; "
                          f"available={available_heights}, failed="
                          f"{sorted(exhausted_heights)}")
                    raise _SoftSkip
                h_cm_cell = remaining_heights[0]
                if h_cm_cell != h_cm:
                    print(f"    [height] cell {cell_key}: selecting reset_"
                          f"{h_cm_cell} ({h_cm_cell}cm; prior failed="
                          f"{sorted(exhausted_heights)})")
                h_cm = h_cm_cell
                RELEASE_HEIGHT_M = h_cm / 100.0
                rec["h_cm"] = h_cm
                rec["release_height_m"] = RELEASE_HEIGHT_M
                print(f"[cycle {cycle}] Loading reset seeds "
                      f"reorient_{h_cm}/{i_int}_{args.target_j}...")
                seeds = _load_reset_seeds(
                    args.hand, args.obj, h_cm, i_int, args.target_j,
                    pose_robot_before, obj_root=asset_root,
                    version=args.version,
                )
                if seeds is None:
                    failed_heights_by_cell.setdefault(cell_key, set()).add(h_cm)
                    rec["progress"]["plan"] = f"no_cell ({i_int}_{args.target_j})"
                    rec["status"] = "no_cell"
                    print(f"    no candidates for cell {i_int}_{args.target_j} "
                          f"under reorient_{h_cm} — skipping cycle")
                    raise _SoftSkip
                print(f"    loaded {seeds['n_total']} seeds from v8 cell "
                      f"{i_int}_{args.target_j} via legacy cell "
                      f"{seeds['legacy_i']}_{seeds['legacy_j']}")
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
                scene_cfg = pose_world_to_scene_cfg(pose_world, c2r, args.obj,
                                                    asset_root)
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
                                                  port=8080, hand=planner_robot)
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
                    planner_robot, planner.HAND_CONFIGS["allegro"])
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
                n_place_fail = 0
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
                        approach_goal[arm_dof:] = openpose_start_list[cand_idx]
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
                        [approach_traj_cand[-1, :arm_dof], pregrasp_cand])
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
                        if planner_robot == "fr3_inspire":
                            # The physical FR3 path is not the candidate
                            # descent above.  It is: move to release+10cm,
                            # descend vertically, release, lift vertically,
                            # and retract.  Verify that exact sequence before
                            # allowing this candidate to reach the gripper.
                            preflight_result = PlanResult(
                                success=True, traj=approach_traj_cand,
                                wrist_se3=wrist_grasp_cand,
                                pregrasp_pose=pregrasp_cand,
                                grasp_pose=grasp_cand,
                                scene_info=ik_res["scene_info"][cand_idx],
                                timing={"candidate_idx": cand_idx},
                            )
                            release_hand_cand = (
                                openpose_target_list[cand_idx]
                                if openpose_target_list[cand_idx] is not None
                                else pregrasp_cand)
                            try:
                                _preflight_franka_place_chain(
                                    planner, scene_cfg, preflight_result,
                                    r_traj,
                                    np.asarray(d_info["T_wrist_target"],
                                               dtype=np.float64),
                                    T_obj_in_wrist_cand, release_hand_cand)
                            except Exception as exc:
                                n_place_fail += 1
                                last_descent_fail = (
                                    sc,
                                    {**(d_info or {}),
                                     "reason": f"full_place_preflight: {exc!r}"},
                                )
                                print(f"  cand#{cand_idx}: full place/release/"
                                      f"exit preflight FAIL ({exc!r})")
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
                                "n_descent_fail": n_descent_fail,
                                "n_place_fail": n_place_fail},
                    )
                    break

                rec["timing"]["plan_s"] = round(time.time() - t0, 2)
                if chosen_cand_idx is None:
                    failed_heights_by_cell.setdefault(cell_key, set()).add(h_cm)
                    rec["progress"]["plan"] = (
                        f"no_grasp_passed (approach_fail={n_approach_fail}, "
                        f"lift_fail={n_lift_fail}, "
                        f"reorient_fail={n_reorient_fail}, "
                        f"descent_fail={n_descent_fail}, "
                        f"place_fail={n_place_fail}, total={len(ik_ok)})")
                    rec["status"] = "no_grasp_with_feasible_full_chain"
                    print(f"[cycle {cycle}] No candidate passed full chain "
                          f"(approach_fail={n_approach_fail}/{len(ik_ok)}  "
                          f"lift_fail={n_lift_fail}/{len(ik_ok)}  "
                          f"reorient_fail={n_reorient_fail}/{len(ik_ok)}  "
                          f"descent_fail={n_descent_fail}/{len(ik_ok)}  "
                          f"place_fail={n_place_fail}/{len(ik_ok)})")
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
                            cand_log, args.obj, args.port_viser, args.hand, args.arm,
                            mesh_path=Path(asset_root) / args.obj / "raw_mesh" /
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
                    "n_place_fail_before_chosen": n_place_fail,
                }
                print(f"    plan: {rec['timing']['plan_s']}s  cand#{chosen_cand_idx}  "
                      f"x={pre_info['chosen_x']:.3f} "
                      f"yaw={np.degrees(pre_info['chosen_yaw']):.0f}°  "
                      f"(skipped approach={n_approach_fail}, lift={n_lift_fail}, "
                      f"reorient={n_reorient_fail}, descent={n_descent_fail}, "
                      f"place={n_place_fail})")

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
                        urdf_viz_path = _viz_urdf(args.arm, args.hand)
                        urdf_fk = yourdfpy.URDF.load(str(urdf_viz_path))
                        pregrasp_h = np.asarray(pregrasp_chosen, dtype=np.float32)
                        grasp_h = np.asarray(grasp_chosen, dtype=np.float32)
                        init_hand_q = planner._init_state[arm_dof:].astype(np.float32)
                        lift_traj_viz = lift_traj_chosen.copy()
                        lift_traj_viz[:, arm_dof:] = grasp_h[None, :]
                        reorient_traj_viz = reorient_traj_chosen.copy()
                        reorient_traj_viz[:, arm_dof:] = grasp_h[None, :]
                        descent_traj_viz = descent_traj_chosen.copy()
                        descent_traj_viz[:, arm_dof:] = grasp_h[None, :]
                        grasp_qpos_top = approach_traj_chosen[-1].copy()
                        Nb = 20
                        b = np.linspace(0, 1, Nb)[:, None]
                        hand_close = (1 - b) * pregrasp_h[None, :] + b * grasp_h[None, :]
                        grasp_close_traj = np.concatenate(
                            [np.tile(grasp_qpos_top[:arm_dof][None], (Nb, 1)),
                             hand_close], axis=1)
                        Nr = 20
                        b_r = np.linspace(0, 1, Nr)[:, None]
                        hand_open = (1 - b_r) * grasp_h[None, :] + b_r * pregrasp_h[None, :]
                        arm_release = np.tile(descent_traj_chosen[-1, :arm_dof][None],
                                              (Nr, 1))
                        release_traj_viz = np.concatenate([arm_release, hand_open],
                                                           axis=1)
                        Nh = 20
                        b_h = np.linspace(0, 1, Nh)[:, None]
                        hand_to_init = (1 - b_h) * pregrasp_h[None, :] + b_h * init_hand_q[None, :]
                        arm_descent_end = np.tile(descent_traj_chosen[-1, :arm_dof][None],
                                                  (Nh, 1))
                        hand_init_traj = np.concatenate([arm_descent_end, hand_to_init],
                                                         axis=1)
                        arm_init = planner._init_state[:arm_dof].astype(np.float32)
                        clear_view = arm_init.copy()
                        clear_view[0] -= np.deg2rad(40.0)
                        cur_arm = descent_traj_chosen[-1, :arm_dof].astype(np.float32).copy()
                        if args.arm == "franka":
                            # Visualisation only: the actual FR3 reset uses
                            # plan_js_to_init.  Avoid reusing the XArm-specific
                            # sequential-joint sketch for a 7-DoF arm.
                            arm_retract = np.linspace(cur_arm, clear_view, 30)
                        else:
                            # Preserve the legacy XArm visualisation exactly.
                            joint_order = ([1, 2, 5, 0, 3, 4]
                                           if cur_arm[1] >= arm_init[1]
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
                        mesh_viz_path = (Path(asset_root) / args.obj / "raw_mesh" /
                                         f"{args.obj}.obj")
                        vis = ViserViewer(port_number=args.port_viser)
                        viz_robot_name = "franka" if args.arm == "franka" else "xarm"
                        vis.add_robot(viz_robot_name, str(urdf_viz_path))
                        if mesh_viz_path.exists():
                            vis.add_object("obj",
                                           trimesh.load(str(mesh_viz_path),
                                                        process=False),
                                           T_obj_start_viz)
                        vis.add_floor(height=0.0)
                        vis.add_traj("approach",  {viz_robot_name: approach_traj_chosen}, {"obj": obj_approach})
                        vis.add_traj("grasp",     {viz_robot_name: grasp_close_traj},     {"obj": obj_grasp})
                        vis.add_traj("lift",      {viz_robot_name: lift_traj_viz},        {"obj": obj_lift})
                        vis.add_traj("reorient",  {viz_robot_name: reorient_traj_viz},    {"obj": obj_reorient})
                        vis.add_traj("descent",   {viz_robot_name: descent_traj_viz},     {"obj": obj_descent})
                        vis.add_traj("release",   {viz_robot_name: release_traj_viz},     {"obj": obj_release})
                        vis.add_traj("hand_init", {viz_robot_name: hand_init_traj},       {"obj": obj_hand_init})
                        vis.add_traj("retract",   {viz_robot_name: retract_traj_viz},     {"obj": obj_retract})
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
                _rcc_start(rcc, "full", True, video_rel)
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
                            _rcc_start(rcc, "stream", False, fps=STREAM_FPS)
                        except Exception as ce:
                            cleanup_errs.append(f"video_stop_or_stream: {ce!r}")
                    if cleanup_errs:
                        rec["cleanup_errors"] = cleanup_errs

                    try:
                        fb_log = executor.reset_fallback(
                            result, planner=planner, scene_cfg=scene_cfg)
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
                # Both executors expose execute_lift(): on XArm it replays the
                # dense planned path, while Franka streams it through its
                # velocity follower without inserting point-to-point shortcuts.
                executor.execute_lift(lift_traj, s_hand)
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
                    _rcc_start(rcc, "stream", False, fps=STREAM_FPS)
                    try:
                        fb_log = executor.reset_fallback(
                            result, planner=planner, scene_cfg=scene_cfg)
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
                _executor_log(executor, "reorient")
                arm_traj = reorient_traj[:, :arm_dof]
                hand_traj = np.tile(s_hand[None], (len(reorient_traj), 1))
                executor._move_joints(arm_traj, hand_traj)
                rec["timing"]["reorient_exec_s"] = round(time.time() - t1, 2)
                rec["progress"]["reorient"] = (
                    f"ok x={pre_info['chosen_x']:.3f} "
                    f"yaw={np.degrees(pre_info['chosen_yaw']):.0f}°"
                )
                print(f"    [reorient] OK  exec={rec['timing']['reorient_exec_s']}s")

                # 8. FR3 always enters the release pose from 10cm above and
                # descends vertically 10cm. The common executor preflights
                # that descent before moving; the legacy XArm replay remains
                # unchanged below.
                t1 = time.time()
                if args.arm == "franka":
                    print(f"[cycle {cycle}] Perpendicular place (10cm)...")
                    _executor_log(executor, "place_vertical")
                    place_info = executor.place(
                        result, planner=planner, scene_cfg=scene_cfg,
                        grasp_wrist=T_wrist_descent_chosen,
                        hand_qpos=grasp_chosen,
                        pregrasp_qpos=(openpose_target_chosen
                                       if openpose_target_chosen is not None
                                       else pregrasp_chosen),
                    )
                    rec["place"] = place_info
                    if not place_info.get("released", False):
                        raise RuntimeError(
                            "vertical reorient placement ended before release; "
                            "object remains held")
                    rec["timing"]["descent_s"] = round(time.time() - t1, 2)
                    rec["progress"]["descent"] = "vertical_10cm_ok"
                    rec["progress"]["release"] = "ok"
                else:
                    descend = LIFT_HEIGHT_M - RELEASE_HEIGHT_M
                    print(f"[cycle {cycle}] Descent ({descend*100:.0f}cm)...")
                    descent_traj = descent_traj_chosen
                    _executor_log(executor, "descent")
                    arm_traj = descent_traj[:, :arm_dof]
                    hand_traj = np.tile(s_hand[None], (len(descent_traj), 1))
                    executor._move_joints(arm_traj, hand_traj)
                    rec["timing"]["descent_s"] = round(time.time() - t1, 2)
                    rec["progress"]["descent"] = "ok"

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

                T_wrist_release = (executor.arm.get_data()["position"]
                                   @ executor._link6_to_wrist)
                planned_obj_pose = T_wrist_release @ T_obj_in_wrist

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
                _rcc_start(rcc, "stream", False, fps=STREAM_FPS)

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
                    tb_after = classify_tabletop_pose(pose_robot_after, args.obj,
                                                      asset_root)
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
                print(f"\n[cycle {cycle}] preflight unavailable/failed.")
                try:
                    cmd = input("    Press Enter to try the next available reset height, "
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
                    _rcc_start(rcc, "stream", False, fps=STREAM_FPS)
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
