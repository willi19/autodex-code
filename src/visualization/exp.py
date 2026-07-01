#!/usr/bin/env python3
"""Interactive viser path-finder: "can we find a path to execute an arbitrary
grasp?"

A grasp candidate lives in a scene (``scene/{shelf,wall}/{id}.json``) which
fixes a required tabletop pose ``j`` (``meta.pose_idx``) and an environment
(walls). To *execute* that grasp the object must be resting in pose ``j``. If it
currently rests in pose ``i != j`` we need a **reset route** ``i -> ... -> j``
through the reorient-transition graph (the ``reset/reorient_{h_cm}/{i}_{j}/``
cells), then the grasp's approach inside the scene world.

So per Plan:
    1. start: object at pose ``i`` placed at (x, yaw=theta) on the table.
    2. target: pose ``j`` = selected scene's ``pose_idx``.
    3. BFS a reset route i -> ... -> j on the reorient graph.
    4. plan each reset hop  (approach->grasp->lift->reorient->descent->release->retract).
    5. plan the selected grasp in its scene world (obstacles transformed to where
       the resets left the object): approach -> close -> lift.
The whole route is loaded into one viser timeline with a FEASIBLE / failed-at
readout, so you can see end-to-end whether the grasp is reachable.

Controls (viser GUI): object, hand, scene type, scene id, grasp, start pose,
initial x, initial theta. Object/hand are hot-swapped on Plan.

Usage:
    python src/visualization/exp.py                 # defaults, pick in the GUI
    python src/visualization/exp.py --obj pepsi --port 8080
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import trimesh
import yourdfpy
from scipy.spatial.transform import Rotation as R
from curobo.geom.types import WorldConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from autodex.utils.path import project_dir, obj_path, repo_dir
from autodex.utils.conversion import se32cart, cart2se3
from autodex.planner import GraspPlanner
from autodex.planner.obstacles import TABLE_CUBOID, add_obstacles
from autodex.planner.planner import (
    _to_curobo_world, _to_curobo_pose, _snap_joint6,
)
from paradex.visualization.visualizer.viser import ViserViewer


LIFT_HEIGHT_M = 0.25
GRASP_LIFT_M = 0.12           # small post-grasp lift to show the pick
TABLE_SURFACE_Z = TABLE_CUBOID["pose"][2] + TABLE_CUBOID["dims"][2] / 2  # 0.039
EE_LINK = "base_link"
GRASP_VERSION = "v7"          # scene grasp candidates (scene types auto-discovered
                              # from v7/{obj}/ — shelf, wall, box, ... generically)

_URDF_ROOT = Path.home() / "shared_data" / "AutoDex" / "content" / "assets" / "robot"
URDF_BY_HAND = {
    "inspire_left": _URDF_ROOT / "inspire_left_description" / "xarm_inspire_left.urdf",
    "inspire":      _URDF_ROOT / "inspire_description"      / "xarm_inspire.urdf",
    "allegro":      _URDF_ROOT / "allegro_description"      / "xarm_allegro.urdf",
}
# Floating-base hand URDFs (6 base joints + finger joints) — used to draw the
# target grasp hand as a static ghost at the goal wrist pose.
FLOATING_URDF_BY_HAND = {
    "inspire_left": _URDF_ROOT / "inspire_description" / "inspire_left_floating.urdf",
    "inspire":      _URDF_ROOT / "inspire_description" / "inspire_floating.urdf",
    "allegro":      _URDF_ROOT / "allegro_description" / "allegro_floating.urdf",
}
MESH_BASE = Path(obj_path)

X_GRID = np.arange(0.30, 0.71, 0.05)        # must match seed_cache.X_GRID
YAW_GRID = np.linspace(0, 2 * np.pi, 36, endpoint=False)   # 10deg, match seed_cache n_yaw=36

# Current deployment scene types only — EXCLUDE *_prev (old/stale geometry),
# reorient_* (reset-task scenes), float, table. compute_order.py counts ALL
# scene types in obj/scene/, which contaminates coverage with stale scenes.
CURRENT_SCENE_TYPES = ("shelf", "wall", "box")


# --------------------------------------------------------------------------- #
# data discovery
# --------------------------------------------------------------------------- #
def hands_with_reset() -> list[str]:
    base = Path(project_dir) / "candidates"
    return sorted(h.name for h in base.iterdir()
                  if (h / "reset").is_dir()) if base.is_dir() else []


def objects_for_hand(hand: str) -> list[str]:
    """Objects that have BOTH reset cells (for the route) and v7 grasps."""
    reset_dir = Path(project_dir) / "candidates" / hand / "reset"
    v7_dir = Path(project_dir) / "candidates" / hand / GRASP_VERSION
    if not (reset_dir.is_dir() and v7_dir.is_dir()):
        return []
    return sorted(set(p.name for p in reset_dir.iterdir() if p.is_dir())
                  & set(p.name for p in v7_dir.iterdir() if p.is_dir()))


def scene_types_for(hand: str, obj: str) -> list[str]:
    d = Path(project_dir) / "candidates" / hand / GRASP_VERSION / obj
    return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.is_dir() else []


def scene_ids_for(hand: str, obj: str, stype: str) -> list[str]:
    d = Path(project_dir) / "candidates" / hand / GRASP_VERSION / obj / stype
    return sorted((p.name for p in d.iterdir() if p.is_dir()), key=int) \
        if d.is_dir() else []


def grasp_names_for(hand: str, obj: str, stype: str, sid: str) -> list[str]:
    d = Path(project_dir) / "candidates" / hand / GRASP_VERSION / obj / stype / sid
    return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.is_dir() else []


def scene_pose_idx(obj: str, stype: str, sid: str) -> int | None:
    """Required tabletop pose for a scene (meta.pose_idx)."""
    p = Path(obj_path) / obj / "scene" / stype / f"{sid}.json"
    if not p.exists():
        return None
    return int(json.load(open(p))["meta"]["pose_idx"])


def default_start_pose(obj: str, stype: str, sid: str) -> str:
    """Pick a start pose DIFFERENT from the scene's required pose so the first
    Plan shows a reset (falls back to the target pose if it's the only one)."""
    poses = sorted(load_tabletop_poses(obj).keys())
    j = scene_pose_idx(obj, stype, sid)
    other = [p for p in poses if p != j]
    return str(other[0] if other else (j if j is not None else poses[0]))


def autoselect_h_cm(hand: str, obj: str) -> int | None:
    base = Path(project_dir) / "candidates" / hand / "reset" / obj
    if not base.is_dir():
        return None
    hs = sorted(int(p.name.split("_")[1]) for p in base.glob("reorient_*")
                if p.name.split("_")[1].isdigit())
    return hs[0] if hs else None


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def _load_T(p: Path) -> np.ndarray:
    M = np.load(p)
    if M.shape == (3, 3):
        T = np.eye(4); T[:3, :3] = M; return T
    return M


def fk_ee(urdf, joint_traj) -> np.ndarray:
    out = np.tile(np.eye(4), (len(joint_traj), 1, 1))
    for t, q in enumerate(joint_traj):
        urdf.update_cfg(q)
        out[t] = urdf.get_transform(EE_LINK, urdf.base_link)
    return out


def load_tabletop_poses(obj: str) -> dict[int, np.ndarray]:
    d = Path(obj_path) / obj / "processed_data" / "info" / "tabletop"
    return {int(f.stem): _load_T(f) for f in sorted(d.glob("*.npy"))}


def build_graph(hand: str, obj: str, h_cm: int) -> dict[int, set[int]]:
    base = Path(project_dir) / "candidates" / hand / "reset" / obj / f"reorient_{h_cm}"
    graph: dict[int, set[int]] = {}
    for cell in base.iterdir():
        if not cell.is_dir() or "_" not in cell.name:
            continue
        i_s, j_s = cell.name.split("_")
        if i_s.isdigit() and j_s.isdigit() and any(g.is_dir() for g in cell.iterdir()):
            graph.setdefault(int(i_s), set()).add(int(j_s))
    return graph


def find_reset_path(graph: dict[int, set[int]], i: int, j: int) -> list[int] | None:
    if i == j:
        return [i]
    q = deque([(i, [i])]); seen = {i}
    while q:
        n, path = q.popleft()
        for nb in sorted(graph.get(n, set())):
            if nb == j:
                return path + [nb]
            if nb not in seen:
                seen.add(nb); q.append((nb, path + [nb]))
    return None


def load_reset_seeds(hand, obj, h_cm, i, j, T_obj_world):
    cell = (Path(project_dir) / "candidates" / hand / "reset" / obj
            / f"reorient_{h_cm}" / f"{i}_{j}")
    if not cell.exists():
        return None
    gdirs = sorted((p for p in cell.iterdir() if p.is_dir()), key=lambda p: int(p.name))
    if not gdirs:
        return None
    wrist_obj = np.stack([np.load(g / "wrist_se3.npy")     for g in gdirs])
    pregrasp  = np.stack([np.load(g / "pregrasp_pose.npy") for g in gdirs])
    grasp     = np.stack([np.load(g / "grasp_pose.npy")    for g in gdirs])
    return {"wrist_se3": T_obj_world[None] @ wrist_obj, "pregrasp": pregrasp,
            "grasp": grasp, "n_total": len(gdirs)}


def ik_check_seeds(planner: GraspPlanner, scene_cfg: dict, seeds: dict) -> dict:
    """IK + backward + collision filter on world-frame seeds. Mask + ik_qpos."""
    wrist_se3, pregrasp = seeds["wrist_se3"], seeds["pregrasp"]
    N = len(wrist_se3)
    world_no_target = _to_curobo_world(scene_cfg); world_no_target["mesh"] = {}
    if planner._ik_solver is None:
        planner._init_ik_solver(world_no_target)
    else:
        planner._ik_solver.update_world(WorldConfig.from_dict(world_no_target))

    if planner._hand.startswith("inspire"):
        backward = np.zeros(N, dtype=bool)
    else:
        backward = (wrist_se3[:, :3, :3] @ planner._link6_y_in_wrist)[:, 2] < 0.3
    collision = planner._check_collision(world_no_target, wrist_se3, pregrasp)
    valid = np.where(~(backward | collision))[0]

    ik_success = np.zeros(N, dtype=bool)
    ik_qpos = np.full((N, len(planner._init_state)), np.nan)
    BS, dev = planner.BATCH_SIZE, planner._tensor_args.device
    for cs in range(0, len(valid), BS):
        idx = valid[cs:cs + BS]; poses = wrist_se3[idx]; B = len(poses)
        if B < BS:
            poses = np.concatenate([poses, np.tile(poses[:1], (BS - B, 1, 1))])
        goal = _to_curobo_pose(poses, dev)
        retract = torch.tensor(planner._init_state, dtype=torch.float32,
                               device=dev).unsqueeze(0).repeat(poses.shape[0], 1)
        res = planner._ik_solver.solve_batch(goal, retract_config=retract)
        succ = res.success.cpu().numpy()[:B]
        qsol = res.solution.cpu().numpy()[:B]
        if qsol.ndim == 3:
            qsol = qsol[:, 0, :]
        for k, ii in enumerate(idx):
            if succ[k]:
                ik_success[ii] = True
                arm = qsol[k, :6].copy(); arm[5] = _snap_joint6(arm[5], planner._init_state[5])
                ik_qpos[ii, :6] = arm; ik_qpos[ii, 6:] = pregrasp[ii]
    return {"ik_success": ik_success, "ik_qpos": ik_qpos, "wrist_se3": wrist_se3,
            "pregrasp": pregrasp, "grasp": seeds["grasp"], "n_total": N,
            "n_backward": int(backward.sum()), "n_collision": int(collision.sum())}


def _interp(q0, q1, n):
    b = np.linspace(0, 1, n)[:, None]
    return (1 - b) * q0[None] + b * q1[None]


def _ensure_motion_gen(planner, world):
    """Reuse the existing MotionGen (update world) instead of re-creating it —
    repeated _init_motion_gen leaks GPU memory across poses/transitions."""
    if planner._motion_gen is None:
        planner._init_motion_gen(world)
    elif planner._world_structure_changed(world):
        planner._update_world(world)
    else:
        planner._update_target_pose_only(world)
    planner._cached_world = world


def _arm_limits(planner):
    jl = planner._motion_gen.kinematics.get_joint_limits()
    return jl.position[0].cpu().numpy()[:6], jl.position[1].cpu().numpy()[:6]


def _unwrap_arm(arm, prev, lo, hi):
    """Per-joint continuity: pick arm[j] + 2*pi*k closest to prev[j] (within joint
    limits). cuRobo IK can return a config that's the SAME wrist pose but with a
    joint wrapped by ~2*pi -> a big visual jump even though the wrist is smooth.
    (Previously only joint 6 was unwrapped, so other joints could flip.)"""
    out = np.asarray(arm, dtype=np.float32).copy()
    for j in range(6):
        cand = out[j] - 2.0 * np.pi * np.round((out[j] - prev[j]) / (2.0 * np.pi))
        if lo[j] - 1e-6 <= cand <= hi[j] + 1e-6:
            out[j] = cand
    return out


def cartesian_move_z(planner, urdf, scene_lift, start_qpos, dz, hold_hand, n=24):
    """STRAIGHT world-z cartesian move of the wrist by signed ``dz`` (>0 up,
    <0 down), solving IK per waypoint seeded from the previous solution — so the
    held object travels in a straight vertical line. This mirrors the real
    executor (autodex/executor/real.py): lift = ``_move_cartesian(lift_pose)``
    (real.py:592) and place = ``target_pose[2,3] -= ...`` "straight down in
    world z" (real.py:626). ``plan_obj_placement`` / ``plan_wrist_reorient``
    (joint-space) curve instead, which is why those looked wrong. Returns
    (n, n_dof) joint traj, or None if any waypoint is unreachable.
    """
    world = _to_curobo_world(scene_lift); world["mesh"] = {}
    # use_cuda_graph=False: per-waypoint IK below uses solve_single + num_seeds.
    planner._init_ik_solver(world, use_cuda_graph=False)
    dev = planner._tensor_args.device
    lo, hi = _arm_limits(planner)
    GATE = 0.5            # rad; max per-waypoint joint step (continuity success metric)
    T_w0 = fk_ee(urdf, start_qpos[None])[0]
    hold = np.asarray(hold_hand, dtype=np.float32)
    traj = [np.concatenate([start_qpos[:6], hold]).astype(np.float32)]
    prev_arm = start_qpos[:6].astype(np.float32)
    for z in np.linspace(0.0, dz, n)[1:]:
        T = T_w0.copy(); T[2, 3] += float(z)
        # CONTINUITY-gated IK: 1 warm-start at prev + 31 random, pick the solution
        # CLOSEST to prev within GATE (NOT the lowest-cost best, which may flip
        # branch ~pi). If none stays within GATE -> this lift can't stay continuous
        # -> FAIL (the campaign retries / uses another grasp). The wrist path is the
        # SAME straight world-z line; only the IK-branch selection changed.
        pc = torch.tensor(np.concatenate([prev_arm, hold]), dtype=torch.float32, device=dev)
        res = planner._ik_solver.solve_single(
            _to_curobo_pose(T[None], dev), retract_config=pc[None],
            seed_config=pc[None, None], num_seeds=8, return_seeds=4)
        sols = res.solution.detach().cpu().numpy().reshape(-1, res.solution.shape[-1])
        succ = res.success.detach().cpu().numpy().reshape(-1)
        best, bd = None, GATE
        for i in range(len(sols)):
            if not succ[i]:
                continue
            c = _unwrap_arm(sols[i][:6], prev_arm, lo, hi)
            d = float(np.abs(c - prev_arm).max())
            if d < bd:
                bd, best = d, c
        if best is None:
            return None                                   # discontinuous / unreachable
        traj.append(np.concatenate([best, hold]).astype(np.float32))
        prev_arm = best.astype(np.float32)
    return np.array(traj, dtype=np.float32)


def _unclear_phase(planner, obj_T, to_config=None, name="unclear"):
    """Move from the -60deg camera-clear home -> the motion's actual START config
    (no planning, object held still). Prepended to reset/reposition so the arm leaves
    the -60deg rest pose continuously. ``to_config`` = the real start qpos (carries a
    j0 azimuth offset for swept reposition picks); defaults to INIT (reset starts there)."""
    init = planner._init_state.astype(np.float32)
    tgt = init if to_config is None else np.asarray(to_config, np.float32)
    clear6 = init[:6].copy(); clear6[0] -= np.deg2rad(60.0)
    rob = np.concatenate([_interp(clear6, tgt[:6], 20),
                          np.tile(tgt[6:][None], (20, 1))], 1)
    obj = np.tile(np.asarray(obj_T, np.float32)[None], (20, 1, 1))
    return (name, rob, obj)


def build_release_retract(planner, descent_end, pregrasp_hand, grasp_hand):
    init_hand = planner._init_state[6:].astype(np.float32)
    xarm_init = planner._init_state[:6].astype(np.float32)
    Nr = 20
    arm = np.tile(descent_end[:6][None], (Nr, 1))
    release = np.concatenate(
        [np.concatenate([arm, _interp(grasp_hand, pregrasp_hand, Nr)], 1),
         np.concatenate([arm, _interp(pregrasp_hand, init_hand, Nr)], 1)], 0)
    # retract to the -60deg "clear" pose (base rotated out of the camera view) --
    # ALL joints move together (continuous), not one-at-a-time sequential.
    clear = xarm_init.copy(); clear[0] -= np.deg2rad(60.0)
    arm_ret = _interp(descent_end[:6].astype(np.float32), clear, 40)
    retract = np.concatenate([arm_ret, np.tile(init_hand[None], (len(arm_ret), 1))], 1)
    return release, retract


# --------------------------------------------------------------------------- #
# one reset transition i -> j (table world)
# --------------------------------------------------------------------------- #
def plan_transition(planner, urdf, obj, hand, h_cm, i, j, T_obj_world_i,
                    mesh_path, tabletop_poses, release_h,
                    x_grid=X_GRID, yaw_grid=YAW_GRID):
    scene_cfg = {"mesh": {"target": {"pose": se32cart(T_obj_world_i).tolist(),
                                     "file_path": str(mesh_path)}}, "cuboid": {}}
    scene_cfg = add_obstacles(scene_cfg, "table")
    seeds = load_reset_seeds(hand, obj, h_cm, i, j, T_obj_world_i)
    if seeds is None:
        return None, "no reset cell"
    ik = ik_check_seeds(planner, scene_cfg, seeds)
    ik_ok = list(np.where(ik["ik_success"])[0]); np.random.shuffle(ik_ok)
    planner._ik_solver = None
    if not ik_ok:
        return None, f"no IK-feasible grasp ({ik['n_total']} seeds)"

    tgt = tabletop_poses[j]; R_target = tgt[:3, :3]
    z_lift = float(tgt[2, 3]) + TABLE_SURFACE_Z + LIFT_HEIGHT_M
    z_rel = float(tgt[2, 3]) + TABLE_SURFACE_Z + release_h
    scene_lift = {"mesh": {}, "cuboid": dict(scene_cfg["cuboid"])}
    world_approach = _to_curobo_world(scene_cfg)
    _ensure_motion_gen(planner, world_approach)

    for cand in ik_ok:
        cand = int(cand)
        wrist_grasp = ik["wrist_se3"][cand]
        grasp_hand = np.asarray(ik["grasp"][cand], dtype=np.float32)
        pregrasp_hand = np.asarray(ik["pregrasp"][cand], dtype=np.float32)
        T_obj_in_wrist = np.linalg.inv(wrist_grasp) @ T_obj_world_i
        if planner._world_structure_changed(world_approach):
            planner._update_world(world_approach); planner._cached_world = world_approach
        ok, approach = planner._refine_fingers(planner._init_state, ik["ik_qpos"][cand])
        if not ok:
            continue
        grasp_qpos = approach[-1].copy()
        # STRAIGHT cartesian lift up (mirrors executor real.py:592
        # _move_cartesian), not plan_wrist_reorient which curves.
        lift = cartesian_move_z(planner, urdf, scene_lift, grasp_qpos,
                                LIFT_HEIGHT_M, pregrasp_hand)
        if lift is None:
            continue
        # reorient (rotation, not a translation) stays a joint-space placement.
        reorient, rinfo = planner.plan_obj_placement(
            scene_lift, lift[-1].copy(), T_obj_in_wrist, R_target,
            np.array([0.0, float(T_obj_world_i[1, 3]), z_lift]),
            hold_hand_qpos=pregrasp_hand, x_grid=x_grid, yaw_grid=yaw_grid)
        if reorient is None:
            continue
        # STRAIGHT cartesian descent down (mirrors executor real.py:626
        # place "straight down in world z"), not plan_obj_placement which curves.
        descent = cartesian_move_z(
            planner, urdf, scene_lift, reorient[-1].copy(),
            -(z_lift - z_rel), pregrasp_hand)
        if descent is None:
            continue

        Nb = 20
        grasp_close = np.concatenate(
            [np.tile(grasp_qpos[:6][None], (Nb, 1)), _interp(pregrasp_hand, grasp_hand, Nb)], 1)
        release, retract = build_release_retract(planner, descent[-1].copy(), pregrasp_hand, grasp_hand)
        obj_app = np.tile(T_obj_world_i[None], (len(approach), 1, 1))
        obj_grp = np.tile(T_obj_world_i[None], (len(grasp_close), 1, 1))
        obj_lift = fk_ee(urdf, lift) @ T_obj_in_wrist
        obj_reo = fk_ee(urdf, reorient) @ T_obj_in_wrist
        obj_dsc = fk_ee(urdf, descent) @ T_obj_in_wrist
        T_obj_j = obj_dsc[-1].copy()
        obj_rel = np.tile(T_obj_j[None], (len(release), 1, 1))
        obj_ret = np.tile(T_obj_j[None], (len(retract), 1, 1))
        phases = [_unclear_phase(planner, obj_app[0], name="unclear"),
                  ("approach", approach, obj_app), ("grasp", grasp_close, obj_grp),
                  ("lift", lift, obj_lift), ("reorient", reorient, obj_reo),
                  ("descent", descent, obj_dsc), ("release", release, obj_rel),
                  ("retract", retract, obj_ret)]
        return {"phases": phases, "T_obj_j": T_obj_j, "retract_end": retract[-1].copy()}, "ok"
    return None, "no candidate planned full chain"


# --------------------------------------------------------------------------- #
# execute the selected grasp — physical-validity check only
# --------------------------------------------------------------------------- #
def plan_grasp_in_scene(planner, urdf, obj, hand, stype, sid, grasp_sel,
                        T_obj_world, mesh_path):
    """Plan approach->close->lift for the chosen v7 grasp with the object resting
    at its world pose.

    The shelf/wall scene is NOT imposed as obstacles here: the scene was only a
    *generation-time* constraint to make the grasps satisfy something useful
    (approach direction etc.); that constraint is not tested at runtime — neither
    on the real robot nor in this validity check. So the grasp is checked in the
    SAME table-only world as the reset; the scene only selects which candidates
    to use (via scene_id) and the target pose. We use the reference
    ``planner.solve_ik`` (its +10cm lift pre-check now passes on the open table).
    """
    scene_cfg = {"mesh": {"target": {"pose": se32cart(T_obj_world).tolist(),
                                     "file_path": str(mesh_path)}}, "cuboid": {}}
    scene_cfg = add_obstacles(scene_cfg, "table")

    ik = planner.solve_ik(scene_cfg, obj, GRASP_VERSION, hand=hand, scene_id=sid)
    planner._ik_solver = None   # rebuild for retract-compatible cartesian moves
    # load_candidate filters by scene_id only, so also filter the scene_type and
    # (optionally) a specific grasp idx.
    idx = [k for k in range(ik["n_total"])
           if ik["scene_info"][k][0] == stype and ik["ik_success"][k]
           and (grasp_sel == "auto" or ik["scene_info"][k][2] == grasp_sel)]
    if not idx:
        return None, (f"grasp IK infeasible on table "
                      f"(n_total={ik['n_total']}, n_ik_success={ik['n_ik_success']})")

    world_approach = _to_curobo_world(scene_cfg)
    _ensure_motion_gen(planner, world_approach)
    scene_lift = {"mesh": {}, "cuboid": dict(scene_cfg["cuboid"])}
    for cand in idx:
        wrist_grasp = ik["wrist_se3"][cand]
        grasp_hand = np.asarray(ik["grasp"][cand], dtype=np.float32)
        pregrasp_hand = np.asarray(ik["pregrasp"][cand], dtype=np.float32)
        T_obj_in_wrist = np.linalg.inv(wrist_grasp) @ T_obj_world
        if planner._world_structure_changed(world_approach):
            planner._update_world(world_approach); planner._cached_world = world_approach
        ok, approach = planner._refine_fingers(planner._init_state, ik["ik_qpos"][cand])
        if not ok:
            continue
        grasp_qpos = approach[-1].copy()
        Nb = 20
        grasp_close = np.concatenate(
            [np.tile(grasp_qpos[:6][None], (Nb, 1)), _interp(pregrasp_hand, grasp_hand, Nb)], 1)
        # STRAIGHT cartesian lift (mirrors executor real.py:592).
        grasp_at_close = np.concatenate([grasp_qpos[:6], grasp_hand]).astype(np.float32)
        lift = cartesian_move_z(planner, urdf, scene_lift, grasp_at_close,
                                GRASP_LIFT_M, grasp_hand)
        if lift is None:
            return None, "lift failed — grasp cannot lift the object"
        obj_app = np.tile(T_obj_world[None], (len(approach), 1, 1))
        obj_grp = np.tile(T_obj_world[None], (len(grasp_close), 1, 1))
        obj_lift = fk_ee(urdf, lift) @ T_obj_in_wrist
        phases = [("g_approach", approach, obj_app), ("g_close", grasp_close, obj_grp),
                  ("g_lift", lift, obj_lift)]
        return {"phases": phases, "grasp_name": ik["scene_info"][cand][2],
                "target_wrist": wrist_grasp, "grasp_hand": grasp_hand,
                "T_obj_target": T_obj_world}, "ok"
    return None, "grasp approach failed to plan"


# --------------------------------------------------------------------------- #
# full path: resets + grasp
# --------------------------------------------------------------------------- #
def plan_full_path(planner, urdf, obj, hand, h_cm, tabletop_poses, graph,
                   mesh_path, stype, sid, grasp_sel, start_pose, x0, theta0):
    j = int(json.load(open(Path(obj_path) / obj / "scene" / stype / f"{sid}.json"))
            ["meta"]["pose_idx"])
    i = int(start_pose)
    log = [f"start pose i={i} @ x={x0:.2f} theta={np.degrees(theta0):.0f}deg  "
           f"-> grasp scene {stype}/{sid} needs pose j={j}"]

    # start placement: pose i rotated by theta about world-z, at (x0, 0)
    T_obj = tabletop_poses[i].copy()
    Rz = R.from_euler("z", theta0).as_matrix()
    T_obj[:3, :3] = Rz @ T_obj[:3, :3]
    T_obj[0, 3], T_obj[1, 3] = float(x0), 0.0
    T_obj[2, 3] += TABLE_SURFACE_Z  # tabletop z is the rest offset; lift to table surface

    timeline, prev_retract = [], None
    path = find_reset_path(graph, i, j)
    if path is None:
        return [], T_obj, None, "FAIL: no reset route i->j on the reorient graph\n" + "\n".join(log)
    if len(path) == 1:
        log.append(f"i == j (={j}): NO RESET — object stays put, grasped in place. "
                   f"pick a different start pose to see a reset.")
    else:
        log.append(f"reset route: {' -> '.join(map(str, path))}")
        for hop, (a, b) in enumerate(zip(path[:-1], path[1:])):
            res, why = plan_transition(planner, urdf, obj, hand, h_cm, a, b, T_obj,
                                       mesh_path, tabletop_poses, h_cm / 100.0)
            if res is None:
                log.append(f"  hop {a}->{b}: FAIL ({why})")
                return timeline, T_obj, None, "FAIL at reset hop\n" + "\n".join(log)
            if prev_retract is not None:
                home = _interp(prev_retract, planner._init_state.astype(np.float32), 20)
                timeline.append((f"h{hop}_home", home, np.tile(T_obj[None], (20, 1, 1))))
            for nm, rt, ot in res["phases"]:
                timeline.append((f"r{hop}_{a}_{b}_{nm}", rt, ot))
            T_obj = res["T_obj_j"]; prev_retract = res["retract_end"]
            log.append(f"  hop {a}->{b}: ok")

    # grasp execution at the object's resulting pose
    if prev_retract is not None:
        home = _interp(prev_retract, planner._init_state.astype(np.float32), 20)
        timeline.append(("g_home", home, np.tile(T_obj[None], (20, 1, 1))))
    gres, why = plan_grasp_in_scene(planner, urdf, obj, hand, stype, sid, grasp_sel,
                                    T_obj, mesh_path)
    if gres is None:
        return timeline, T_obj, None, "FAIL at grasp\n" + "\n".join(log) + f"\n  grasp: FAIL ({why})"
    for nm, rt, ot in gres["phases"]:
        timeline.append((nm, rt, ot))
    log.append(f"  grasp: ok (candidate {gres['grasp_name']})")
    target = {"wrist": gres["target_wrist"], "grasp_hand": gres["grasp_hand"],
              "T_obj": gres["T_obj_target"]}
    return timeline, T_obj, target, "FEASIBLE\n" + "\n".join(log)


# --------------------------------------------------------------------------- #
# coverage campaign: cover every scene of one object, minimizing resets
# --------------------------------------------------------------------------- #
def compute_coverage_data(planner, obj, hand, scene_types=CURRENT_SCENE_TYPES):
    """Collision-free coverage matrix over CURRENT-deployment scenes only.

    Uses the experiment's ``load_grasp_data`` (compute_order.py) for the grasp
    pool, and builds ``valid_array`` (S scenes x G grasps; True = grasp g is
    collision-free with scene s) over ``scene_types`` ONLY (shelf/wall/box) —
    excluding *_prev / reorient_* / float / table that compute_order.py would
    include. Cached to coverage_current.{npy,json}.

    Returns dict: valid_array (S,G), scene_meta [(stype,sid,pose)],
    grasp_meta [(stype,sid,gname,pose)], or None on error.
    """
    from src.grasp_generation.order.compute_order import load_grasp_data

    cdir = Path(repo_dir) / "order" / hand / GRASP_VERSION / obj
    va_path, meta_path = cdir / "coverage_current.npy", cdir / "coverage_current.json"
    cand_root = str(Path(project_dir) / "candidates" / hand / GRASP_VERSION)
    info, wrist_obj, preg = load_grasp_data(cand_root, obj)
    if len(info) == 0:
        return None
    grasp_meta = [(g[1], g[2], g[3], scene_pose_idx(obj, g[1], g[2])) for g in info]

    if va_path.exists() and meta_path.exists():
        valid_array = np.load(va_path)
        scene_meta = [tuple(m) for m in json.load(open(meta_path))["scene_meta"]]
    else:
        scene_root = Path(obj_path) / obj / "scene"
        rows, scene_meta = [], []
        for st in scene_types:
            std = scene_root / st
            if not std.is_dir():
                continue
            for sf in sorted(std.glob("*.json")):
                scfg = json.load(open(sf))["scene"]
                obj_se3 = cart2se3(scfg["mesh"]["target"]["pose"])
                wrist_world = np.einsum("ij,ajk->aik", obj_se3, wrist_obj)
                coll = planner._check_collision(_to_curobo_world(scfg), wrist_world, preg)
                if (~coll).any():
                    rows.append(~coll)
                    scene_meta.append((st, sf.stem, scene_pose_idx(obj, st, sf.stem)))
        if not rows:
            return None
        valid_array = np.array(rows)
        cdir.mkdir(parents=True, exist_ok=True)
        np.save(va_path, valid_array)
        json.dump({"scene_meta": scene_meta}, open(meta_path, "w"), indent=2)

    # --- precomputed reachability: reach[g, xi, yi] = grasp g IK-feasible at
    #     placement (X_GRID[xi], YAW_GRID[yi]) for g's pose. Computed once, cached.
    reach_path = cdir / "reach_current.npy"
    if reach_path.exists():
        reach = np.load(reach_path)
    else:
        reach = _compute_reachability(planner, obj, wrist_obj, grasp_meta)
        cdir.mkdir(parents=True, exist_ok=True)
        np.save(reach_path, reach)

    return {"valid_array": valid_array, "scene_meta": scene_meta,
            "grasp_meta": grasp_meta, "wrist_obj": wrist_obj, "pregrasp": preg,
            "reach": reach}


def _compute_reachability(planner, obj, wrist_obj, grasp_meta):
    """reach[g, xi, yi] = IK-feasible for grasp g at placement
    (X_GRID[xi], YAW_GRID[yi]) with the object resting in g's pose. One batch-IK
    sweep over all placements — done once and cached so the campaign just looks
    up where each grasp is graspable instead of re-solving IK every run."""
    torch.manual_seed(0)               # deterministic reachability map -> predictable resettable area
    tabletop = load_tabletop_poses(obj)
    gpose = np.array([gm[3] for gm in grasp_meta])
    G = len(wrist_obj)
    reach = np.zeros((G, len(X_GRID), len(YAW_GRID)), dtype=bool)
    world = _to_curobo_world({"mesh": {}, "cuboid": {"table": TABLE_CUBOID}})
    if planner._ik_solver is None:
        planner._init_ik_solver(world)
    else:
        planner._ik_solver.update_world(WorldConfig.from_dict(world))
    dev, BS = planner._tensor_args.device, planner.BATCH_SIZE
    uposes = sorted(set(int(p) for p in gpose))
    for xi, x in enumerate(X_GRID):
        for yi, yaw in enumerate(YAW_GRID):
            c, s = np.cos(yaw), np.sin(yaw)
            Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            wrist_world = np.tile(np.eye(4), (G, 1, 1))
            for p in uposes:
                idx = np.where(gpose == p)[0]
                Tp = np.eye(4)
                Tp[:3, :3] = Rz @ tabletop[p][:3, :3]
                Tp[0, 3] = float(x); Tp[1, 3] = 0.0
                Tp[2, 3] = float(tabletop[p][2, 3]) + TABLE_SURFACE_Z
                wrist_world[idx] = Tp[None] @ wrist_obj[idx]
            for cs in range(0, G, BS):
                chunk = wrist_world[cs:cs + BS]; B = len(chunk)
                if B < BS:
                    chunk = np.concatenate([chunk, np.tile(chunk[:1], (BS - B, 1, 1))])
                goal = _to_curobo_pose(chunk, dev)
                retract = torch.tensor(planner._init_state, dtype=torch.float32,
                                       device=dev).unsqueeze(0).repeat(BS, 1)
                res = planner._ik_solver.solve_batch(goal, retract_config=retract)
                reach[cs:cs + B, xi, yi] = res.success.cpu().numpy().reshape(-1)[:B]
    planner._ik_solver = None
    return reach


def _placement_indices(obj_T, pose_T):
    """Inverse of _place_T: recover the (xi, yi) grid indices of obj_T relative to the
    tabletop pose. A yaw-changing reposition can leave the object at yaw != 0, so the
    reset transport must look up reach[g, xi, yi] at the object's ACTUAL placement, not
    assume yaw=0."""
    xi = int(np.argmin(np.abs(X_GRID - float(obj_T[0, 3]))))
    Rz = obj_T[:3, :3] @ pose_T[:3, :3].T          # Rz(yaw) = obj_R @ tab_R^T
    yaw = float(np.arctan2(Rz[1, 0], Rz[0, 0])) % (2 * np.pi)
    yi = int(np.argmin(np.abs(((YAW_GRID - yaw + np.pi) % (2 * np.pi)) - np.pi)))
    return xi, yi


def _place_T(pose_T, x, yaw):
    """Object world pose: tabletop orientation pose_T yawed by `yaw` about world-z,
    resting at (x, 0, table)."""
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T = pose_T.copy()
    T[:3, :3] = Rz @ pose_T[:3, :3]
    T[0, 3], T[1, 3] = float(x), 0.0
    T[2, 3] = float(pose_T[2, 3]) + TABLE_SURFACE_Z
    return T


def _clean_plan(planner, wrist_obj_g, pregrasp_g, current_T, mesh_path):
    """Plan INIT->grasp LIVE with the seeding METHOD (no precomputed cache):
    build_one = fixed-INIT plan; if it ARCS (wrist z-rise) do the azimuth sweep
    (rotate object {0,±30,±60,±90}, re-IK, plan, rotate j0 back) and keep the
    CLEANEST candidate. Deterministic + clean, only for the FEW grasps the campaign
    actually uses (greedy set-cover -> ~13 total). Returns (ok, traj)."""
    import src.visualization.seed_cache as sc
    lo, hi = _arm_limits(planner)

    def world_at(T):
        return _to_curobo_world({"mesh": {"target": {"pose": se32cart(np.asarray(T)),
                                                      "file_path": str(mesh_path)}},
                                 "cuboid": {"table": TABLE_CUBOID}})
    cands = sc.build_one(planner, wrist_obj_g, np.asarray(pregrasp_g, np.float32),
                         np.asarray(current_T, np.float64), world_at, lo, hi)
    if not cands:
        return False, None
    return True, np.asarray(sc.select(cands, planner, "cart"), np.float32)


def rank_placements(reach, cand, restrict=None, yaw_idxs=None):
    """Rank placements (xi, yi) by how many of the candidate grasps are reachable
    there, using the PRECOMPUTED reachability map (no live IK). ``cand`` = grasp
    indices; ``restrict`` = optional bool mask over cand (e.g. only grasps
    covering still-uncovered scenes). ``yaw_idxs`` restricts which yaw columns to
    consider — default [0] (yaw=0), because reorienting the held object to a
    non-zero yaw during the reset is unreliable; same-pose repositions can pass a
    wider set. Returns [(x, yaw, xi, yi, reachable_grasp_indices)] sorted by
    count desc (count>0)."""
    cand = np.asarray(list(cand))
    sub = cand if restrict is None else cand[restrict]
    if yaw_idxs is None:
        yaw_idxs = [0]
    out = []
    for xi, x in enumerate(X_GRID):
        for yi in yaw_idxs:
            r = sub[reach[sub, xi, yi]]
            if len(r):
                out.append((float(x), float(YAW_GRID[yi]), xi, yi, r))
    out.sort(key=lambda t: -len(t[4]))
    return out


_ATTEMPT = [0]


def _alog(what, stage, dt, ok):
    """One structured line per plan attempt -> easy to grep into a timing table."""
    _ATTEMPT[0] += 1
    print(f"[ATTEMPT {_ATTEMPT[0]:>3}] {what:<16} {stage:<10} {dt:>6.1f}s  "
          f"{'OK' if ok else 'FAIL'}", flush=True)


def plan_grasps_at_placement(planner, urdf, obj, hand, current_T, cand, wrist_obj,
                             pregrasp, grasp_meta, va, scenes_p, covered, mesh_path):
    """Efficiently cover scenes_p at a FIXED object placement (no reset between
    grasps). Batch-IK all candidate grasps once at current_T, then greedily
    _refine_fingers only the reachable grasps needed to cover scenes_p (loading
    grasp_pose.npy on demand). Returns (phases, played, failed); mutates covered.
    Avoids the per-grasp solve_ik/load_candidate reload that made the campaign
    time out."""
    scene_cfg = {"mesh": {"target": {"pose": se32cart(current_T).tolist(),
                                     "file_path": str(mesh_path)}},
                 "cuboid": {"table": TABLE_CUBOID}}
    world_approach = _to_curobo_world(scene_cfg)
    _ensure_motion_gen(planner, world_approach)
    world_no = _to_curobo_world({"mesh": {}, "cuboid": {"table": TABLE_CUBOID}})
    # use_cuda_graph=False: _clean_plan -> build_one does single-pose (batch-1) IK,
    # which a graph-captured (BATCH_SIZE=50) ik_solver rejects ("changing goal type").
    planner._init_ik_solver(world_no, use_cuda_graph=False)

    # batch IK for all candidate grasps at this placement
    dev, BS = planner._tensor_args.device, planner.BATCH_SIZE
    cand = list(cand)
    W = current_T[None] @ wrist_obj[cand]
    ik_qpos = {}
    for cs in range(0, len(cand), BS):
        chunk = W[cs:cs + BS]; B = len(chunk)
        if B < BS:
            chunk = np.concatenate([chunk, np.tile(chunk[:1], (BS - B, 1, 1))])
        goal = _to_curobo_pose(chunk, dev)
        retract = torch.tensor(planner._init_state, dtype=torch.float32,
                               device=dev).unsqueeze(0).repeat(BS, 1)
        res = planner._ik_solver.solve_batch(goal, retract_config=retract)
        succ = res.success.cpu().numpy().reshape(-1)[:B]
        sol = res.solution.cpu().numpy()
        if sol.ndim == 3:
            sol = sol[:, 0, :]
        for i in range(B):
            if succ[i]:
                g = cand[cs + i]
                arm = sol[i, :6].copy(); arm[5] = _snap_joint6(arm[5], planner._init_state[5])
                ik_qpos[g] = np.concatenate([arm, np.asarray(pregrasp[g], np.float32)])
    # keep _ik_solver alive: _clean_plan -> build_one re-IKs per sweep angle below.

    scene_lift = {"mesh": {}, "cuboid": {"table": TABLE_CUBOID}}
    cbase = Path(project_dir) / "candidates" / hand / GRASP_VERSION / obj
    phases, played, failed = [], 0, 0
    avail = [g for g in cand if g in ik_qpos]      # IK-reachable only
    while not covered[scenes_p].all() and avail:
        need = scenes_p & ~covered
        gains = [int((va[:, g] & need).sum()) for g in avail]
        bi = int(np.argmax(gains))
        if gains[bi] == 0:
            break
        g = avail.pop(bi)
        _t = time.time()
        ok, approach = _clean_plan(planner, wrist_obj[g], pregrasp[g], current_T, mesh_path)
        _alog(f"grasp g{g}", "approach", time.time() - _t, ok)
        if not ok:
            failed += 1; continue
        st, sid, gname, _ = grasp_meta[g]
        grasp_g = np.load(cbase / st / sid / gname / "grasp_pose.npy").astype(np.float32)
        pregrasp_g = np.asarray(pregrasp[g], np.float32)
        grasp_qpos = approach[-1].copy()
        Nb = 20
        grasp_close = np.concatenate(
            [np.tile(grasp_qpos[:6][None], (Nb, 1)), _interp(pregrasp_g, grasp_g, Nb)], 1)
        _t = time.time()
        lift = cartesian_move_z(planner, urdf, scene_lift,
                                np.concatenate([grasp_qpos[:6], grasp_g]).astype(np.float32),
                                GRASP_LIFT_M, grasp_g)
        _alog(f"grasp g{g}", "lift", time.time() - _t, lift is not None)
        if lift is None:
            failed += 1; continue       # can't lift -> grasp fails (no fake 1-frame stub)
        T_oiw = np.linalg.inv(current_T @ wrist_obj[g]) @ current_T
        oa = np.tile(current_T[None], (len(approach), 1, 1))
        og = np.tile(current_T[None], (len(grasp_close), 1, 1))
        ol = fk_ee(urdf, lift) @ T_oiw
        # camera-clear home: base rotated -60deg so the arm is OUT of the camera
        # view at rest. Each grasp is bracketed clear->INIT (unclear) ... INIT->clear
        # so the arm rests at -60deg between grasps AND the path stays continuous.
        # camera-clear rest pose = INIT with base rotated -60deg. Bracket to the plan's
        # ACTUAL start config (approach[0]) -- a swept grasp carries a j0 azimuth offset
        # there -- so there's no jump between the bracket and the approach.
        start6 = approach[0][:6].astype(np.float32)
        start_hand = approach[0][6:].astype(np.float32)
        clear6 = planner._init_state[:6].astype(np.float32).copy(); clear6[0] -= np.deg2rad(60.0)
        unclear = np.concatenate([_interp(clear6, start6, 20),
                                  np.tile(start_hand[None], (20, 1))], 1)
        clearp = np.concatenate([_interp(start6, clear6, 20),
                                 np.tile(start_hand[None], (20, 1))], 1)
        obj_hold = np.tile(current_T[None], (20, 1, 1))
        phases += [(f"g{g}_unclear", unclear, obj_hold),
                   (f"g{g}_approach", approach, oa), (f"g{g}_close", grasp_close, og),
                   (f"g{g}_lift", lift, ol)]
        # RETURN to INIT by moving BACKWARD along the same path (continuous), then
        # INIT->clear so the arm rests at -60deg (camera unobstructed for next trial).
        rob_back = np.concatenate([lift[::-1], grasp_close[::-1], approach[::-1]], 0).astype(np.float32)
        obj_back = np.concatenate([ol[::-1], og[::-1], oa[::-1]], 0)
        phases.append((f"g{g}_return", rob_back, obj_back))
        phases.append((f"g{g}_clear", clearp, obj_hold))
        played += 1
        covered |= (va[:, g] & scenes_p)
    return phases, played, failed


def _interp_se3(A, B, t):
    from scipy.spatial.transform import Slerp
    T = np.eye(4)
    T[:3, 3] = (1 - t) * A[:3, 3] + t * B[:3, 3]
    slerp = Slerp([0, 1], R.from_matrix(np.stack([A[:3, :3], B[:3, :3]])))
    T[:3, :3] = slerp(t).as_matrix()
    return T


def cartesian_object_path(planner, scene_lift, start_qpos, obj_wps, T_obj_in_wrist,
                          hold, n_seg=18):
    """Move a held object straight along ``obj_wps`` (list of SE3 object poses),
    solving wrist IK = obj_pose @ inv(T_obj_in_wrist) per interpolated waypoint.
    Returns joint traj or None if any waypoint is unreachable."""
    world = _to_curobo_world(scene_lift); world["mesh"] = {}
    if planner._ik_solver is None:
        planner._init_ik_solver(world)
    else:
        planner._ik_solver.update_world(WorldConfig.from_dict(world))
    dev, BS = planner._tensor_args.device, planner.BATCH_SIZE
    inv_oiw = np.linalg.inv(T_obj_in_wrist)
    lo, hi = _arm_limits(planner)
    hold = np.asarray(hold, np.float32)
    traj = [np.concatenate([start_qpos[:6], hold]).astype(np.float32)]
    prev = start_qpos[:6].astype(np.float32)
    for k in range(1, len(obj_wps)):
        for t in np.linspace(0.0, 1.0, n_seg)[1:]:
            wrist = _interp_se3(obj_wps[k - 1], obj_wps[k], float(t)) @ inv_oiw
            goal = _to_curobo_pose(np.tile(wrist[None], (BS, 1, 1)), dev)
            retract = torch.tensor(np.concatenate([prev, hold]), dtype=torch.float32,
                                   device=dev).unsqueeze(0).repeat(BS, 1)
            res = planner._ik_solver.solve_batch(goal, retract_config=retract)
            succ = res.success.cpu().numpy().reshape(-1)
            sol = res.solution.cpu().numpy()
            if sol.ndim == 3:
                sol = sol[:, 0, :]
            if not bool(succ[0]):
                planner._ik_solver = None
                return None
            arm = _unwrap_arm(sol[0, :6], prev, lo, hi)   # all joints continuous
            traj.append(np.concatenate([arm, hold]).astype(np.float32))
            prev = arm
    planner._ik_solver = None
    return np.array(traj, dtype=np.float32)


_OBJ_VERTS_CACHE: dict = {}


def _obj_verts(mesh_path):
    """Object mesh vertices (V,3), cached. process=False keeps raw geometry."""
    key = str(mesh_path)
    if key not in _OBJ_VERTS_CACHE:
        import trimesh
        m = trimesh.load(key, process=False)
        _OBJ_VERTS_CACHE[key] = np.asarray(m.vertices, np.float32)
    return _OBJ_VERTS_CACHE[key]


def _held_clears_table(obj_poses, mesh_path, tol=0.005):
    """True if the HELD object stays above the table top across obj_poses (M,4,4).
    The object is stripped from the planner's collision world while held (it moves
    with the hand), so cuRobo can tilt it INTO the table -- this post-hoc mesh check
    (lowest object vertex >= table top) rejects such plans. Same gate reorient uses."""
    v = _obj_verts(mesh_path)
    op = np.asarray(obj_poses, np.float32)
    zrow = op[:, 2, :3]                      # (M,3) world-z row of each pose
    zt = op[:, 2, 3]                         # (M,)
    vz = v @ zrow.T + zt[None, :]            # (V,M) world z of every vertex
    return float(vz.min()) >= TABLE_SURFACE_Z - tol


def plan_reposition(planner, urdf, obj, hand, current_T, target_T, transport_cands,
                    wrist_obj, pregrasp, grasp_meta, mesh_path):
    """SAME-pose reposition: PICK the object at current_T with a transport grasp,
    MOVE it to target_T in JOINT space via plan_obj_placement (the validated
    reorient mover -- NO cartesian drag, so no branch flips and ~1s not ~50s),
    RELEASE. The target is a single (x,yaw) cell = exactly target_T. A transport
    grasp must hold the object at BOTH ends; a cheap skip_plan IK pre-flight
    (~0.1s) rejects target-unreachable grasps BEFORE the costly approach.
    Returns (phases, target_T) or None."""
    _ensure_motion_gen(planner, _to_curobo_world(
        {"mesh": {"target": {"pose": se32cart(current_T).tolist(),
                             "file_path": str(mesh_path)}},
         "cuboid": {"table": TABLE_CUBOID}}))
    scene_lift = {"mesh": {}, "cuboid": {"table": TABLE_CUBOID}}
    if planner._ik_solver is None:
        planner._init_ik_solver(_to_curobo_world(scene_lift), use_cuda_graph=False)
    R_tgt = target_T[:3, :3]
    obj_tgt_pos = target_T[:3, 3].astype(np.float32)
    x_grid = np.array([float(target_T[0, 3])], np.float32)   # single cell = exact target
    yaw_grid = np.array([0.0], np.float32)
    MAX_TRIES = 12

    for tg in transport_cands[:MAX_TRIES]:
        tw = wrist_obj[tg]
        pg = np.asarray(pregrasp[tg], np.float32)
        T_oiw = np.linalg.inv(current_T @ tw) @ current_T    # object-in-wrist
        # cheap target-reach pre-flight (skip_plan ~0.1s) before the costly approach
        _, info = planner.plan_obj_placement(
            scene_lift, planner._init_state, T_oiw, R_tgt, obj_tgt_pos,
            hold_hand_qpos=pg, x_grid=x_grid, yaw_grid=yaw_grid, skip_plan=True)
        if info.get("n_feasible", 0) == 0:
            continue                                         # can't hold obj at target
        ok, approach = _clean_plan(planner, tw, pg, current_T, mesh_path)
        if not ok:
            continue                                         # pick approach failed
        grasp_qpos = approach[-1].astype(np.float32)
        move, info = planner.plan_obj_placement(
            scene_lift, grasp_qpos, T_oiw, R_tgt, obj_tgt_pos,
            hold_hand_qpos=pg, x_grid=x_grid, yaw_grid=yaw_grid)
        if move is None:
            continue                                         # joint move blocked
        obj_move = fk_ee(urdf, move) @ T_oiw                 # held-object path
        if not _held_clears_table(obj_move, mesh_path):
            continue                                         # object dips into table -> reject
        release, retract_t = build_release_retract(planner, move[-1].copy(), pg, pg)
        obj_app = np.tile(current_T[None], (len(approach), 1, 1))
        obj_rel = np.tile(target_T[None], (len(release), 1, 1))
        obj_ret = np.tile(target_T[None], (len(retract_t), 1, 1))
        phases = [_unclear_phase(planner, current_T, approach[0], "rp_unclear"),
                  ("rp_approach", approach, obj_app), ("rp_move", move, obj_move),
                  ("rp_release", release, obj_rel), ("rp_retract", retract_t, obj_ret)]
        return phases, target_T
    return None                                              # no transport grasp worked


# --------------------------------------------------------------------------- #
# interactive app
# --------------------------------------------------------------------------- #
class App:
    def __init__(self, vis, default_obj, default_hand, port):
        self.vis = vis
        self.planners: dict[str, GraspPlanner] = {}
        self.urdfs: dict[str, yourdfpy.URDF] = {}
        self.cur_obj = None
        self.cur_hand = None
        self._pending = None      # plan request enqueued by the button callback
        self._busy = False
        self._build_gui(default_obj, default_hand)

    # ---- lazy resources ----
    def planner(self, hand):
        if hand not in self.planners:
            self._status(f"warming up planner ({hand})...")
            self.planners[hand] = GraspPlanner(hand=hand)
            self.urdfs[hand] = yourdfpy.URDF.load(str(URDF_BY_HAND[hand]))
        return self.planners[hand], self.urdfs[hand]

    def _status(self, msg):
        self.status.content = f"```\n{msg}\n```"

    # ---- gui ----
    def _build_gui(self, default_obj, default_hand):
        s = self.vis.server
        hands = hands_with_reset() or ["inspire_left"]
        hand0 = default_hand if default_hand in hands else hands[0]
        objs = objects_for_hand(hand0)
        obj0 = default_obj if default_obj in objs else (objs[0] if objs else "")

        with s.gui.add_folder("Path finder"):
            self.dd_hand = s.gui.add_dropdown("hand", hands, initial_value=hand0)
            self.dd_obj = s.gui.add_dropdown("object", objs or [""], initial_value=obj0)
            self.dd_type = s.gui.add_dropdown("scene type", scene_types_for(hand0, obj0) or [""])
            self.dd_sid = s.gui.add_dropdown("scene id", scene_ids_for(hand0, obj0, self.dd_type.value) or [""])
            self.dd_grasp = s.gui.add_dropdown("grasp", ["auto"] + grasp_names_for(hand0, obj0, self.dd_type.value, self.dd_sid.value))
            poses = sorted(load_tabletop_poses(obj0).keys()) if obj0 else [0]
            start0 = default_start_pose(obj0, self.dd_type.value, self.dd_sid.value) if obj0 else "0"
            self.dd_start = s.gui.add_dropdown("start pose", [str(p) for p in poses], initial_value=start0)
            self.sl_x = s.gui.add_slider("initial x", min=0.30, max=0.55, step=0.01, initial_value=0.45)
            self.sl_theta = s.gui.add_slider("initial theta (deg)", min=-180, max=180, step=5, initial_value=0)
            self.btn = s.gui.add_button("Plan path")
            self.btn_campaign = s.gui.add_button("Cover all scenes")
            self.cb_rebuild = s.gui.add_checkbox("rebuild cache", False)
        self.status = s.gui.add_markdown("```\nready\n```")

        self.dd_hand.on_update(lambda _: self._on_hand())
        self.dd_obj.on_update(lambda _: self._on_obj())
        self.dd_type.on_update(lambda _: self._on_type())
        self.dd_sid.on_update(lambda _: self._on_sid())
        self.dd_start.on_update(lambda _: self._preview())
        self.sl_x.on_update(lambda _: self._preview())
        self.sl_theta.on_update(lambda _: self._preview())
        self.btn.on_click(lambda _: self._on_plan())
        self.btn_campaign.on_click(lambda _: self._on_campaign())

        # show the object at its initial placement right away
        if obj0:
            self._preview()

    def _start_placement(self, obj, start_pose, x0, theta0):
        """4x4 world pose of the object resting in `start_pose` at (x0, 0, table),
        rotated by theta0 about world-z."""
        T = load_tabletop_poses(obj)[int(start_pose)].copy()
        T[:3, :3] = R.from_euler("z", theta0).as_matrix() @ T[:3, :3]
        T[0, 3], T[1, 3] = float(x0), 0.0
        T[2, 3] += TABLE_SURFACE_Z
        return T

    def _preview(self):
        """Live: place the object at the current initial x / theta / start pose
        (no planning). Clears any stale trajectory so the player won't override."""
        obj, hand = self.dd_obj.value, self.dd_hand.value
        if not obj:
            return
        self._ensure_scene(obj, hand)          # planner-free (mesh + urdf only)
        self.vis.clear_traj()
        Tinit = self._start_placement(obj, self.dd_start.value,
                                      float(self.sl_x.value),
                                      np.radians(float(self.sl_theta.value)))
        self.vis.obj_dict["obj"]["transform"] = Tinit
        fh = self.vis.obj_dict["obj"]["frame"]
        fh.position = Tinit[:3, 3]
        fh.wxyz = R.from_matrix(Tinit[:3, :3]).as_quat()[[3, 0, 1, 2]]
        if hand in self.planners:              # park robot at init if warmed
            self.vis.robot_dict["xarm"].update_cfg(self.planners[hand]._init_state)
        j = scene_pose_idx(obj, self.dd_type.value, self.dd_sid.value)
        self._status(f"preview: start pose {self.dd_start.value} @ "
                     f"x={float(self.sl_x.value):.2f} "
                     f"theta={float(self.sl_theta.value):.0f}deg\n"
                     f"target pose (scene {self.dd_type.value}/{self.dd_sid.value}) = {j}\n"
                     f"press 'Plan path'")

    def _on_hand(self):
        objs = objects_for_hand(self.dd_hand.value) or [""]
        self.dd_obj.options = objs
        if self.dd_obj.value not in objs:
            self.dd_obj.value = objs[0]
        self._on_obj()

    def _on_obj(self):
        hand, obj = self.dd_hand.value, self.dd_obj.value
        types = scene_types_for(hand, obj) or [""]
        self.dd_type.options = types
        if self.dd_type.value not in types:
            self.dd_type.value = types[0]
        poses = sorted(load_tabletop_poses(obj).keys()) if obj else [0]
        self.dd_start.options = [str(p) for p in poses]
        if self.dd_start.value not in self.dd_start.options:
            self.dd_start.value = self.dd_start.options[0]
        self._on_type()

    def _on_type(self):
        ids = scene_ids_for(self.dd_hand.value, self.dd_obj.value, self.dd_type.value) or [""]
        self.dd_sid.options = ids
        if self.dd_sid.value not in ids:
            self.dd_sid.value = ids[0]
        self._on_sid()

    def _on_sid(self):
        obj, stype, sid = self.dd_obj.value, self.dd_type.value, self.dd_sid.value
        gs = ["auto"] + grasp_names_for(self.dd_hand.value, obj, stype, sid)
        self.dd_grasp.options = gs
        if self.dd_grasp.value not in gs:
            self.dd_grasp.value = gs[0]
        # if the current start pose == this scene's required pose, no reset would
        # happen (object stays put). Nudge to a pose that triggers a reset.
        j = scene_pose_idx(obj, stype, sid)
        if j is not None and self.dd_start.value == str(j):
            self.dd_start.value = default_start_pose(obj, stype, sid)

    # ---- swap meshes when object/hand change ----
    def _ensure_scene(self, obj, hand):
        if hand != self.cur_hand:
            if "xarm" in self.vis.robot_dict:
                self.vis.robot_dict["xarm"].remove(); del self.vis.robot_dict["xarm"]
            self.vis.add_robot("xarm", str(URDF_BY_HAND[hand]))
            self.cur_hand = hand
        if obj != self.cur_obj:
            if "obj" in self.vis.obj_dict:
                self.vis.obj_dict["obj"]["frame"].remove(); del self.vis.obj_dict["obj"]
            mesh = trimesh.load(str(MESH_BASE / obj / "raw_mesh" / f"{obj}.obj"), process=False)
            self.vis.add_object("obj", mesh, np.eye(4))
            self.cur_obj = obj

    # ---- Plan button: only ENQUEUE from the viser callback thread. cuRobo uses
    #      CUDA-graph capture which breaks (cudaErrorStreamCaptureInvalidated) if
    #      run off the main thread, so the actual planning is serviced from the
    #      main loop via service_pending(). ----
    def _on_plan(self):
        if self._busy:
            self._status("planning in progress..."); return
        obj, stype, sid = self.dd_obj.value, self.dd_type.value, self.dd_sid.value
        if not (obj and stype and sid):
            self._status("pick object / scene type / scene id first"); return
        self._pending = dict(
            mode="single",
            obj=obj, hand=self.dd_hand.value, stype=stype, sid=sid,
            gsel=self.dd_grasp.value, start_pose=int(self.dd_start.value),
            x0=float(self.sl_x.value), theta0=np.radians(float(self.sl_theta.value)))
        self._status("queued — planning (warmup can take ~10s)...")

    def _on_campaign(self):
        if self._busy:
            self._status("planning in progress..."); return
        obj = self.dd_obj.value
        if not obj:
            self._status("pick an object first"); return
        self._pending = dict(mode="campaign", obj=obj, hand=self.dd_hand.value,
                             x0=float(self.sl_x.value), rebuild=bool(self.cb_rebuild.value))
        self._status("queued — campaign (replay cache or build+cache)...")

    def service_pending(self):
        """Run on the MAIN thread from the app loop."""
        if self._pending is None:
            return
        req = self._pending; self._pending = None
        self._busy = True
        try:
            if req["mode"] == "campaign":
                self._run_campaign(req)
            else:
                self._run_plan(req)
        except Exception as e:
            self._status(f"ERROR: {e}")
            import traceback; traceback.print_exc()
        finally:
            self._busy = False

    def _run_plan(self, req):
        obj, hand = req["obj"], req["hand"]
        self._ensure_scene(obj, hand)
        planner, urdf = self.planner(hand)        # warms in main thread
        h_cm = autoselect_h_cm(hand, obj)
        if h_cm is None:
            self._status(f"no reset cells for {hand}/{obj}"); return
        tabletop = load_tabletop_poses(obj)
        graph = build_graph(hand, obj, h_cm)
        mesh_path = MESH_BASE / obj / "raw_mesh" / f"{obj}.obj"

        # The reset placement is randomized (closest-arm pick over a shuffled
        # candidate set), so a bad placement can leave the grasp out of reach.
        # Retry a few times until the full path is feasible.
        MAX_TRIES = 4
        timeline, status, target = [], "", None
        for attempt in range(MAX_TRIES):
            self._status(f"planning... (attempt {attempt + 1}/{MAX_TRIES})")
            timeline, T0, target, status = plan_full_path(
                planner, urdf, obj, hand, h_cm, tabletop, graph, mesh_path,
                req["stype"], req["sid"], req["gsel"],
                req["start_pose"], req["x0"], req["theta0"])
            if status.startswith("FEASIBLE"):
                break
            # no point retrying if there's simply no reset route
            if "no reset route" in status:
                break

        self.vis.clear_traj()                     # NOTE: also sets gui_playing=False
        Tinit = self._start_placement(obj, req["start_pose"], req["x0"], req["theta0"])
        self.vis.obj_dict["obj"]["transform"] = Tinit
        # target ghost (goal grasp hand + object) so you can see the path converge
        self._show_target(hand, target, MESH_BASE / obj / "raw_mesh" / f"{obj}.obj")
        if not timeline:
            self._status(status); return
        for nm, rt, ot in timeline:
            self.vis.add_traj(nm, {"xarm": rt}, {"obj": ot})
        self.vis.gui_timestep.value = 0
        self.vis.gui_playing.value = True         # re-enable playback after clear_traj
        nfr = sum(len(t[1]) for t in timeline)
        self._status(f"{status}\n\n[{len(timeline)} phases, {nfr} frames] — playing")

    def _show_target(self, hand, target, mesh_path):
        """Draw the goal grasp as a static ghost: floating hand at the grasp
        wrist (fingers = grasp config) + the object at its target pose."""
        # clear previous ghost
        if "target_hand" in self.vis.robot_dict:
            self.vis.robot_dict["target_hand"].remove()
            del self.vis.robot_dict["target_hand"]
        if "obj_target" in self.vis.obj_dict:
            self.vis.obj_dict["obj_target"]["frame"].remove()
            del self.vis.obj_dict["obj_target"]
        if target is None:
            return
        floating = FLOATING_URDF_BY_HAND[hand]
        self.vis.add_robot("target_hand", str(floating), pose=target["wrist"])
        furdf = yourdfpy.URDF.load(str(floating))
        cfg = np.zeros(len(furdf.actuated_joints), dtype=np.float32)
        gh = np.asarray(target["grasp_hand"], dtype=np.float32)
        cfg[6:6 + len(gh)] = gh                   # 6 floating-base joints stay 0
        self.vis.robot_dict["target_hand"].update_cfg(cfg)
        self.vis.add_object("obj_target",
                            trimesh.load(str(mesh_path), process=False), target["T_obj"])

    def _run_campaign(self, req):
        """Cover EVERY current-deployment scene (shelf/wall/box) with GRASPABLE
        grasps — like real grasping, where we select a grasp that actually plans.

        Per pose group (reset once): greedily pick the same-pose grasp covering
        the most still-uncovered scenes (collision-free), try to plan reset->grasp;
        if it plans, execute it and mark its scenes covered; if NOT, drop it and
        fall back to the next candidate. Repeat until the pose's scenes are
        covered or no graspable candidate remains. Resets = #pose groups."""
        obj, hand, x0 = req["obj"], req["hand"], req["x0"]
        cache_path = (Path(repo_dir) / "order" / hand / GRASP_VERSION / obj
                      / "campaign_cache.pkl")

        # --- REPLAY from cache: the flaky planning was done once, offline. No
        #     planner/GPU needed — just load the saved trajectories and play. ---
        if cache_path.exists() and not req.get("rebuild"):
            self._ensure_scene(obj, hand)                  # mesh + urdf only
            c = pickle.load(open(cache_path, "rb"))
            self.vis.clear_traj()
            self.vis.obj_dict["obj"]["transform"] = c["init_T"]
            self._show_target(hand, None, MESH_BASE / obj / "raw_mesh" / f"{obj}.obj")
            for nm, rt, ot in c["timeline"]:
                self.vis.add_traj(nm, {"xarm": rt}, {"obj": ot})
            self.vis.gui_timestep.value = 0
            self.vis.gui_playing.value = True
            self._status(c["status"] + "\n\n[REPLAYED from cache — no planning]")
            return

        self._ensure_scene(obj, hand)
        planner, urdf = self.planner(hand)

        import time as _time
        _T0 = _time.time()
        def _log(m):
            print(f"[campaign t={_time.time()-_T0:6.1f}s] {m}", flush=True)
        self._status("computing coverage matrix (shelf/wall/box)...")
        _t = _time.time(); data = compute_coverage_data(planner, obj, hand)
        _log(f"coverage_data done ({_time.time()-_t:.1f}s)")
        if data is None:
            self._status(f"no coverage data for {hand}/{obj}"); return
        va = data["valid_array"]; scene_meta = data["scene_meta"]; gmeta = data["grasp_meta"]
        S, G = va.shape

        h_cm = autoselect_h_cm(hand, obj)
        if h_cm is None:
            self._status(f"no reset cells for {hand}/{obj}"); return
        tabletop = load_tabletop_poses(obj)
        graph = build_graph(hand, obj, h_cm)
        mesh_path = MESH_BASE / obj / "raw_mesh" / f"{obj}.obj"

        wrist_obj = data["wrist_obj"]
        poses = sorted({m[2] for m in scene_meta})
        scene_pose = np.array([m[2] for m in scene_meta])
        gpose = np.array([g[3] for g in gmeta])
        covered = np.zeros(S, dtype=bool)
        timeline, resets, played, failed = [], 0, 0, 0
        current_pose, current_T = None, None
        first_T = None

        reach = data["reach"]; pregrasp = data["pregrasp"]
        repos = 0
        pose_detail = {}                                   # per-pose breakdown
        for p in poses:
            torch.cuda.empty_cache()                       # reclaim between poses
            scenes_p = (scene_pose == p)
            cand = np.array([g for g in np.where(gpose == p)[0]])
            if len(cand) == 0:
                continue
            _log(f"--- pose {p}: {int(scenes_p.sum())} scenes, {len(cand)} grasps ---")
            det = pose_detail[p] = {"scenes": int(scenes_p.sum()), "reset_tries": 0,
                                    "reset_ok": False, "g_ok": 0, "g_fail": 0,
                                    "repos": 0, "cov0": int(covered[scenes_p].sum())}
            # Ranked grasp-good x for this pose (yaw=0), from reachability map.
            x_ranked = [pl[0] for pl in rank_placements(reach, cand)]
            # --- (a) initial placement: reset to a grasp-good x (pass the RANKED
            #     LIST so the reorient picks one it can actually reach; fallback to
            #     the full grid so the reset stays valid). ---
            if current_pose is None:                       # very first: start here
                current_T = _place_T(tabletop[p], x_ranked[0] if x_ranked else x0, 0.0)
                first_T = current_T.copy(); current_pose = p
            elif current_pose != p:
                # RESET is stochastic (cuRobo trajopt seed-dependent), so RETRY a
                # few times before giving up — a failed reset drops a whole pose's
                # scenes, so reliability here dominates coverage.
                res = None
                for attempt in range(4):
                    det["reset_tries"] += 1
                    self._status(f"reset {current_pose}->{p} (try {attempt+1}/4)...")
                    _t = time.time()
                    res, _ = plan_transition(
                        planner, urdf, obj, hand, h_cm, current_pose, p, current_T,
                        mesh_path, tabletop, h_cm / 100.0,
                        x_grid=np.array(x_ranked if x_ranked else X_GRID),
                        yaw_grid=np.array([0.0]))
                    if res is None:                        # widen to full grid
                        res, _ = plan_transition(
                            planner, urdf, obj, hand, h_cm, current_pose, p, current_T,
                            mesh_path, tabletop, h_cm / 100.0,
                            x_grid=X_GRID, yaw_grid=YAW_GRID)
                    _alog(f"reset {current_pose}->{p}", f"try{attempt+1}", time.time() - _t, res is not None)
                    if res is not None:
                        break
                if res is None:
                    # RESET failed from here -> REPOSITION within the current pose
                    # to a placement from which the reset CAN plan, then retry.
                    cur_xi, cur_yi = _placement_indices(current_T, tabletop[current_pose])
                    cand_cur = np.where(gpose == current_pose)[0]
                    tcands = [int(g) for g in cand_cur if reach[g, cur_xi, cur_yi]]
                    for xi_alt, x_alt in enumerate(X_GRID):
                        if abs(x_alt - float(current_T[0, 3])) < 0.02:
                            continue
                        tcands_t = [g for g in tcands if reach[g, xi_alt, 0]]  # holds at target
                        if not tcands_t:
                            continue
                        self._status(f"reset {current_pose}->{p} failed; reposition "
                                     f"current pose to x={x_alt:.2f} then retry reset...")
                        rp = plan_reposition(
                            planner, urdf, obj, hand, current_T,
                            _place_T(tabletop[current_pose], x_alt, 0.0), tcands_t,
                            wrist_obj, pregrasp, gmeta, mesh_path)
                        if rp is None:
                            continue
                        cand_phases, cand_T = rp
                        r2 = None
                        for _a in range(3):
                            det["reset_tries"] += 1
                            r2, _ = plan_transition(
                                planner, urdf, obj, hand, h_cm, current_pose, p, cand_T,
                                mesh_path, tabletop, h_cm / 100.0,
                                x_grid=np.array(x_ranked if x_ranked else X_GRID),
                                yaw_grid=np.array([0.0]))
                            if r2 is None:
                                r2, _ = plan_transition(
                                    planner, urdf, obj, hand, h_cm, current_pose, p, cand_T,
                                    mesh_path, tabletop, h_cm / 100.0,
                                    x_grid=X_GRID, yaw_grid=YAW_GRID)
                            if r2 is not None:
                                break
                        if r2 is not None:                 # reposition enabled the reset
                            for nm, rt, ot in cand_phases:
                                timeline.append((f"repos{repos}_pre_reset_{current_pose}_{nm}", rt, ot))
                            repos += 1; det["repos"] += 1
                            current_T = cand_T; res = r2
                            break
                if res is None:
                    continue                               # truly can't reach pose
                det["reset_ok"] = True
                for nm, rt, ot in res["phases"]:
                    timeline.append((f"reset{resets}_{current_pose}_{p}_{nm}", rt, ot))
                current_T = res["T_obj_j"]; current_pose = p; resets += 1
            # --- (b) cover at this placement ---
            ph, pl, fa = plan_grasps_at_placement(
                planner, urdf, obj, hand, current_T, list(cand), wrist_obj,
                pregrasp, gmeta, va, scenes_p, covered, mesh_path)
            for nm, rt, ot in ph:
                timeline.append((f"p{p}_{nm}", rt, ot))
            played += pl; failed += fa; det["g_ok"] += pl; det["g_fail"] += fa
            # --- (c) REPOSITION for leftover scenes: MOVE the object to placements
            #     where the leftover grasp actually PLANS (not just reach-gated).
            #     Iterate candidate target placements; at each, transport there and
            #     actually try to cover (plan_grasps_at_placement). First placement
            #     that yields new coverage wins; if none do, move on. ---
            # GRASP-FIRST reposition: for the leftover scenes, pick the GRASP that
            # covers the most still-uncovered scenes AND is reachable SOMEWHERE, then
            # move the object to ONE of that grasp's reachable placements and execute
            # it. Targeted -- we do NOT rank/try every (placement, grasp) pair.
            for _round in range(2 * len(cand)):
                if covered[scenes_p].all():
                    break
                need = scenes_p & ~covered
                gains = np.array([int((va[:, g] & need).sum()) for g in cand])
                reach_any = np.array([bool(reach[g].any()) for g in cand])
                ok = (gains > 0) & reach_any
                if not ok.any():
                    break
                cand_order = [int(cand[i]) for i in np.argsort(np.where(ok, gains, -1))[::-1]
                              if ok[i]]                      # best-covering grasps first
                cur_x = float(current_T[0, 3])
                cur_xi, cur_yi = _placement_indices(current_T, tabletop[p])  # actual yaw
                tcands = [int(g) for g in cand if reach[g, cur_xi, cur_yi]]  # transport from here
                progressed = False
                for gbest in cand_order:
                    cells = np.argwhere(reach[gbest])         # (xi, yi) where gbest is graspable
                    cells = sorted(cells, key=lambda c: abs(float(X_GRID[c[0]]) - cur_x))
                    for xi, yi in cells:
                        x, yaw = float(X_GRID[xi]), float(YAW_GRID[yi])
                        if abs(x - cur_x) < 0.02 and yi == 0:
                            continue                          # already here
                        # transport grasp must hold the object at BOTH the current
                        # placement AND this target cell -- filter by the reach map
                        # (instant) so plan_reposition only tries grasps that can.
                        tcands_t = [g for g in tcands if reach[g, xi, yi]]
                        if not tcands_t:
                            continue
                        tgt_T = _place_T(tabletop[p], x, yaw)
                        # PRE-CHECK: does gbest actually PLAN at this placement?
                        # reachable (IK) != plannable (trajopt). Confirm BEFORE moving
                        # the object, else a grasp that fails to plan would strand the
                        # object at a dead cell and block recovery to a working cell.
                        _t = time.time()
                        pre_ok, _ = _clean_plan(planner, wrist_obj[gbest],
                                                np.asarray(pregrasp[gbest], np.float32),
                                                tgt_T, mesh_path)
                        _alog(f"repos-precheck g{gbest}", f"x{x:.2f}", time.time() - _t, pre_ok)
                        if not pre_ok:
                            continue                          # won't plan here -> don't move
                        self._status(f"reposition @pose {p}: grasp {gbest} -> x={x:.2f} "
                                     f"yaw={np.degrees(yaw):.0f} (covered {int(covered.sum())}/{S})")
                        _t = time.time()
                        rp = plan_reposition(planner, urdf, obj, hand, current_T,
                                             tgt_T, tcands_t,
                                             wrist_obj, pregrasp, gmeta, mesh_path)
                        _alog(f"reposition g{gbest}", f"x{x:.2f}", time.time() - _t, rp is not None)
                        if rp is None:
                            continue                          # couldn't transport here
                        # Try the grasp at the NEW placement, but DON'T commit the move
                        # yet. Only if the grasp actually covers a new scene do we keep
                        # the reposition; otherwise the object stays at current_T (we
                        # never moved it) and we try the next cell/grasp from here.
                        ph_r, new_T = rp
                        before = int(covered.sum())
                        ph, pl, fa = plan_grasps_at_placement(   # execute ONLY the target grasp
                            planner, urdf, obj, hand, new_T, [gbest], wrist_obj,
                            pregrasp, gmeta, va, scenes_p, covered, mesh_path)
                        if int(covered.sum()) > before:        # SUCCESS -> commit move + grasp
                            current_T = new_T
                            for nm, rt, ot in ph_r:
                                timeline.append((f"repos{repos}_p{p}_{nm}", rt, ot))
                            repos += 1; det["repos"] += 1
                            for nm, rt, ot in ph:
                                timeline.append((f"p{p}_r{repos}_{nm}", rt, ot))
                            played += pl; det["g_ok"] += pl
                            progressed = True
                            break
                        failed += fa; det["g_fail"] += fa     # FAIL -> discard move, retry
                    if progressed:
                        break
                if not progressed:
                    break                                     # no grasp coverable -> stop

        self.vis.clear_traj()
        T0 = first_T if first_T is not None else tabletop[poses[0]].copy()
        self.vis.obj_dict["obj"]["transform"] = T0
        self._show_target(hand, None, mesh_path)
        for nm, rt, ot in timeline:
            self.vis.add_traj(nm, {"xarm": rt}, {"obj": ot})
        self.vis.gui_timestep.value = 0
        self.vis.gui_playing.value = True
        nfr = sum(len(t[1]) for t in timeline)
        msg = (f"COVERAGE CAMPAIGN ({obj}, {GRASP_VERSION}, shelf+wall+box)\n"
               f"  scenes: {S}   covered: {int(covered.sum())} "
               f"({100*covered.sum()//S}%)\n"
               f"  graspable grasps used: {played}  (not-graspable skipped: {failed})\n"
               f"  pose groups: {len(poses)} -> {poses}\n"
               f"  resets (pose change): {resets}   repositions (same pose): {repos}\n"
               f"  (placement x/yaw from precomputed reachability map)\n"
               f"  [{nfr} frames] — playing")
        # per-pose breakdown table
        msg += "\n\npose | scenes cov | reset(try/ok) | grasp(ok/fail) | repos"
        for p in poses:
            d = pose_detail.get(p)
            if not d:
                continue
            cov_now = int(covered[scene_pose == p].sum())
            msg += (f"\n{p:>4} |  {cov_now:>2}/{d['scenes']:<2}     |  "
                    f"{d['reset_tries']}/{int(d['reset_ok'])}          |  "
                    f"{d['g_ok']}/{d['g_fail']}          |  {d['repos']}")
        if not covered.all():
            miss = [scene_meta[i] for i in np.where(~covered)[0]]
            msg += f"\nUNCOVERED {len(miss)}: {miss[:8]}{'...' if len(miss)>8 else ''}"
        # --- SAVE successful trajectories to cache so future runs just replay
        #     (no flaky planning). Rebuilt when req['rebuild'] is set. ---
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            pickle.dump({"timeline": timeline, "init_T": T0, "status": msg},
                        open(cache_path, "wb"))
            # FULL documented record for the homepage: every phase, the selected
            # grasp, frame counts, and a plain explanation of what each phase does.
            _save_campaign_detail(cache_path.with_name("campaign_detail.json"),
                                  obj, hand, timeline, T0, covered, scene_meta,
                                  scene_pose, poses, pose_detail, resets, repos, msg)
            msg += (f"\n\n[cached -> {cache_path.name}; detail -> campaign_detail.json"
                    f" (+ npz trajectories); next run replays]")
        except Exception as e:
            msg += f"\n[cache save failed: {e}]"
        self._status(msg)


_PHASE_DOC = {
    "approach": "INIT -> grasp: LIVE clean plan (build_one: fixed-INIT trajopt, azimuth"
                " sweep if it arcs, keep the lowest wrist-z-rise candidate).",
    "close": "Close the fingers from pregrasp to the grasp configuration.",
    "lift": "Lift the grasped object straight up (clearance for transport/reorient).",
    "reorient": "Reorient in the air: A_lifted -> B_lifted (joint plan_obj_placement),"
                " the object rotates from the old tabletop pose to the new one.",
    "descent": "Lower the object straight down to rest on the table in the new pose.",
    "release": "Open the fingers to release the object.",
    "retract": "Retract the arm back toward home.",
    "home": "Return to the home configuration between hops.",
    "reposition": "Reposition on the table (same pose, new x/yaw) so a grasp becomes"
                  " executable.",
    "grasp": "Grasp phase.",
}


def _phase_kind(name):
    for k in _PHASE_DOC:
        if k in name:
            return k
    return "move"


def _save_campaign_detail(path, obj, hand, timeline, init_T, covered, scene_meta,
                          scene_pose, poses, pose_detail, resets, repos, status):
    """Write a human-readable JSON describing the whole exp.py trajectory (phases,
    selected grasp ids, frame counts, explanations) + an .npz of every phase's robot
    and object trajectory, so a homepage can replay/annotate it without the planner."""
    import re
    phases = []
    arrs = {}
    for i, (nm, rt, ot) in enumerate(timeline):
        m = re.search(r"g(\d+)", nm)
        gid = int(m.group(1)) if m else None
        kind = _phase_kind(nm)
        phases.append({"i": i, "name": nm, "kind": kind, "grasp_id": gid,
                       "frames": int(len(rt)), "dof": int(np.asarray(rt).shape[1]),
                       "explanation": _PHASE_DOC.get(kind, "")})
        arrs[f"{i:03d}_{nm}_robot"] = np.asarray(rt, np.float32)
        arrs[f"{i:03d}_{nm}_object"] = np.asarray(ot, np.float32)
    grasps_used = sorted({p["grasp_id"] for p in phases if p["grasp_id"] is not None})
    detail = {
        "obj": obj, "hand": hand,
        "coverage": {"covered": int(covered.sum()), "total": int(len(covered)),
                     "uncovered": [list(map(str, scene_meta[i]))
                                   for i in np.where(~covered)[0]]},
        "resets": int(resets), "repositions": int(repos),
        "grasps_used": [int(g) for g in grasps_used],
        "n_phases": len(phases), "n_frames": int(sum(p["frames"] for p in phases)),
        "init_object_pose": np.asarray(init_T, np.float32).tolist(),
        "per_pose": {int(p): pose_detail.get(p) for p in poses},
        "phases": phases,
        "status_readout": status,
    }
    json.dump(detail, open(path, "w"), indent=2)
    np.savez_compressed(path.with_suffix(".npz"), **arrs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obj", default="pepsi")
    p.add_argument("--hand", default="inspire_left")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--plan", action="store_true",
                   help="HEADLESS: plan the whole campaign and SAVE (cache + "
                        "campaign_detail.json + .npz), then exit. No GUI. The viewer "
                        "then just LOADS the saved result (run exp.py without --plan).")
    p.add_argument("--x0", type=float, default=0.40, help="initial object x (headless plan)")
    args = p.parse_args()

    vis = ViserViewer(port_number=args.port)
    vis.add_floor(height=0.0)
    app = App(vis, args.obj, args.hand, args.port)

    if args.plan:                                          # plan-by-code: build + save, no GUI
        print(f"[exp] HEADLESS plan: {args.obj}/{args.hand} ...", flush=True)
        app._pending = dict(mode="campaign", obj=args.obj, hand=args.hand,
                            x0=float(args.x0), rebuild=True)
        app.service_pending()                             # runs _run_campaign -> saves everything
        print(f"[exp] plan saved -> order/{args.hand}/{GRASP_VERSION}/{args.obj}/"
              f"(campaign_cache.pkl, campaign_detail.json, campaign_detail.npz)", flush=True)
        return

    vis.start_viewer(use_thread=True)
    print(f"[exp] interactive path-finder at http://localhost:{args.port}")
    # all cuRobo/CUDA work happens HERE on the main thread (button only enqueues)
    try:
        while True:
            app.service_pending()
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
