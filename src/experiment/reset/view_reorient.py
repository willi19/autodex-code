#!/usr/bin/env python3
"""Viser viewer: full reorient_drop cycle (offline, no robot/cameras).

Phases:
    approach   : planner.plan() — INIT → grasp_qpos (obj + table obstacles)
    grasp      : arm fixed, hand pregrasp → grasp
    lift       : plan_wrist_reorient — same orientation, z + LIFT_HEIGHT_M
    reorient   : plan_wrist_reorient — wrist rotates so held obj matches
                 target tabletop orientation, still at lift height
    descent    : plan_wrist_reorient — z -= (LIFT - RELEASE_HEIGHT)
    release    : arm fixed, hand grasp → pregrasp

Usage:
    python src/experiment/reset/view_reorient.py --obj donut \\
        --hand inspire_left --start_xy 0.5 0.0 --target_tabletop 002
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import numpy as np
import torch
import trimesh
import yourdfpy
from scipy.spatial.transform import Rotation as R
from curobo.geom.types import WorldConfig

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from autodex.utils.path import obj_path
from autodex.utils.conversion import se32cart
from autodex.planner import GraspPlanner
from autodex.planner.obstacles import TABLE_CUBOID, add_obstacles
from autodex.planner.planner import _to_curobo_pose, _to_curobo_world
from src.execution.run_auto import MESH_BASE
from paradex.visualization.visualizer.viser import ViserViewer


LIFT_HEIGHT_M = 0.25
RELEASE_HEIGHT_M = 0.15
TABLE_SURFACE_Z = TABLE_CUBOID["pose"][2] + TABLE_CUBOID["dims"][2] / 2  # 0.039
GRASP_VERSION = "table_only"
EE_LINK = "base_link"

_URDF_ROOT = Path.home() / "shared_data" / "AutoDex" / "content" / "assets" / "robot"
URDF_BY_HAND = {
    "inspire_left": _URDF_ROOT / "inspire_left_description" / "xarm_inspire_left.urdf",
    "inspire":      _URDF_ROOT / "inspire_description"      / "xarm_inspire.urdf",
    "allegro":      _URDF_ROOT / "allegro_description"      / "xarm_allegro.urdf",
}
# Floating-base hand URDFs — used to visualize the wrist target pose
# (6 base joints + finger joints; placing the URDF root at T_wrist_target
# with base joints at 0 puts the wrist link exactly at the target).
FLOATING_URDF_BY_HAND = {
    "inspire_left": _URDF_ROOT / "inspire_description" / "inspire_left_floating.urdf",
    "inspire":      _URDF_ROOT / "inspire_description" / "inspire_floating.urdf",
    "allegro":      _URDF_ROOT / "allegro_description" / "allegro_floating.urdf",
}


def _load_T(p: Path) -> np.ndarray:
    M = np.load(p)
    if M.shape == (3, 3):
        T = np.eye(4); T[:3, :3] = M; return T
    return M


def fk_ee(urdf, joint_traj):
    out = np.tile(np.eye(4), (len(joint_traj), 1, 1))
    for t, q in enumerate(joint_traj):
        urdf.update_cfg(q)
        out[t] = urdf.get_transform(EE_LINK, urdf.base_link)
    return out


def search_reorient_placement(planner, scene_lift, lift_end_qpos, T_obj_in_wrist,
                              T_target_table, start_y, lift_z_above_table,
                              grasp_hand_arr, x_grid, yaw_grid):
    """Search (x, yaw) for an IK-feasible obj placement at lift height.

    For each (x, yaw), construct ``T_obj_target`` with the target tabletop
    orientation yawed by yaw (around obj center / world-z), position
    ``(x, start_y, lift_z)``. Compute the required ``T_wrist_target =
    T_obj_target @ inv(T_obj_in_wrist)`` and batch-IK all candidates. Pick
    the IK-feasible candidate whose arm config is closest to ``lift_end_qpos``
    in joint space, then ``plan_single_js`` from lift end to that arm config
    holding ``grasp_hand_arr``.

    Returns ``(traj or None, info_dict)`` — info dict has ``chosen_x``,
    ``chosen_yaw``, ``T_wrist_target``, ``n_feasible``, ``reason`` (on fail).
    """
    R_target_base = T_target_table[:3, :3]
    table_target_z = T_target_table[2, 3]

    candidates_T_wrist = []
    candidates_meta = []
    for x_try in x_grid:
        for yaw_try in yaw_grid:
            c, s = np.cos(yaw_try), np.sin(yaw_try)
            R_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            T_obj_target = np.eye(4)
            T_obj_target[:3, :3] = R_z @ R_target_base
            T_obj_target[0, 3] = float(x_try)
            T_obj_target[1, 3] = float(start_y)
            T_obj_target[2, 3] = float(table_target_z + lift_z_above_table)
            T_wrist = T_obj_target @ np.linalg.inv(T_obj_in_wrist)
            candidates_T_wrist.append(T_wrist)
            candidates_meta.append((float(x_try), float(yaw_try)))
    candidates_T_wrist = np.array(candidates_T_wrist)
    N = len(candidates_T_wrist)

    # IK solver uses table-only world (no held mesh — attached to robot).
    world_cfg_no_target = _to_curobo_world(scene_lift)
    world_cfg_no_target["mesh"] = {}
    if planner._ik_solver is None:
        planner._init_ik_solver(world_cfg_no_target)
    else:
        planner._ik_solver.update_world(WorldConfig.from_dict(world_cfg_no_target))

    # Batch IK over the grid
    device = planner._tensor_args.device
    ik_success_all = np.zeros(N, dtype=bool)
    ik_arm_qpos = np.full((N, 6), np.nan, dtype=np.float32)

    for chunk_start in range(0, N, planner.BATCH_SIZE):
        chunk_idx = list(range(chunk_start,
                               min(chunk_start + planner.BATCH_SIZE, N)))
        chunk_poses = candidates_T_wrist[chunk_idx]
        B = len(chunk_poses)
        if B < planner.BATCH_SIZE:
            pad = planner.BATCH_SIZE - B
            chunk_poses = np.concatenate(
                [chunk_poses, np.tile(chunk_poses[:1], (pad, 1, 1))], axis=0)
        goal = _to_curobo_pose(chunk_poses, device)
        retract = torch.tensor(
            planner._init_state, dtype=torch.float32, device=device,
        ).unsqueeze(0).repeat(planner.BATCH_SIZE, 1)
        res = planner._ik_solver.solve_batch(goal, retract_config=retract)
        succ = res.success.cpu().numpy()[:B]
        q_sol = res.solution.cpu().numpy()[:B]
        if q_sol.ndim == 3:
            q_sol = q_sol[:, 0, :]
        for i, idx in enumerate(chunk_idx):
            if succ[i]:
                ik_success_all[idx] = True
                arm_q = q_sol[i, :6].copy()
                diff = arm_q[5] - lift_end_qpos[5]
                arm_q[5] -= np.round(diff / (2 * np.pi)) * 2 * np.pi
                ik_arm_qpos[idx] = arm_q

    feasible = np.where(ik_success_all)[0]
    info = {"n_candidates": N, "n_feasible": int(len(feasible)),
            "T_wrist_target": candidates_T_wrist[0]}  # placeholder for viz
    if len(feasible) == 0:
        info["reason"] = "no_ik_feasible_in_grid"
        return None, info

    # Pick the IK-feasible candidate closest to current arm in joint space.
    dists = np.linalg.norm(ik_arm_qpos[feasible] - lift_end_qpos[:6], axis=1)
    best_local = int(np.argmin(dists))
    best_idx = int(feasible[best_local])
    chosen_x, chosen_yaw = candidates_meta[best_idx]
    chosen_arm = ik_arm_qpos[best_idx]
    info["chosen_x"] = chosen_x
    info["chosen_yaw"] = chosen_yaw
    info["best_arm_dist_rad"] = float(dists[best_local])
    info["T_wrist_target"] = candidates_T_wrist[best_idx]

    # plan_single_js from lift end to chosen arm config (hand held).
    start_full = np.asarray(lift_end_qpos, dtype=np.float32)
    goal_full = np.concatenate(
        [chosen_arm.astype(np.float32), grasp_hand_arr.astype(np.float32)])
    ok, traj = planner._refine_fingers(start_full, goal_full)
    if not ok:
        info["reason"] = "plan_single_js_failed"
        return None, info
    return traj, info


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obj", required=True)
    p.add_argument("--hand", default="inspire_left",
                   choices=["allegro", "inspire", "inspire_left"])
    p.add_argument("--start_tabletop", default=None)
    p.add_argument("--target_tabletop", default=None,
                   help="Filename stem of the target tabletop pose for the "
                        "mid-air reorient. Defaults to start_tabletop (no "
                        "rotation), which makes the reorient a no-op.")
    p.add_argument("--start_xy", type=float, nargs=2, default=[0.5, 0.0])
    p.add_argument("--port", type=int, default=8080)
    args = p.parse_args()

    tabletop_dir = Path(obj_path) / args.obj / "processed_data" / "info" / "tabletop"
    start_key = args.start_tabletop or sorted(tabletop_dir.glob("*.npy"))[0].stem
    T_start = _load_T(tabletop_dir / f"{start_key}.npy")

    T_obj_start = T_start.copy()
    T_obj_start[0, 3] = float(args.start_xy[0])
    T_obj_start[1, 3] = float(args.start_xy[1])
    T_obj_start[2, 3] += TABLE_SURFACE_Z
    print(f"[view] obj on table = {T_obj_start[:3, 3]}")

    mesh_path = MESH_BASE / args.obj / "raw_mesh" / f"{args.obj}.obj"
    if not mesh_path.exists():
        sys.exit(f"mesh not found: {mesh_path}")

    # scene_cfg WITH obj mesh + table
    scene_cfg = {
        "mesh": {"target": {"pose": se32cart(T_obj_start).tolist(),
                            "file_path": str(mesh_path)}},
        "cuboid": {},
    }
    scene_cfg = add_obstacles(scene_cfg, "table")

    print("[view] warming up cuRobo planner...")
    planner = GraspPlanner(hand=args.hand)

    # URDF + joint-limits — hoisted out of the candidate-retry loop.
    urdf_path = URDF_BY_HAND[args.hand]
    urdf_fk = yourdfpy.URDF.load(str(urdf_path))
    actuated = urdf_fk.actuated_joints
    hand_lo = np.array([j.limit.lower for j in actuated[6:]], dtype=np.float32)
    hand_hi = np.array([j.limit.upper for j in actuated[6:]], dtype=np.float32)

    # Target tabletop ORIENTATION — the placement (x, y, yaw) is searched
    # below; here we only need the base rotation.
    target_key = args.target_tabletop or start_key
    T_target_table = _load_T(tabletop_dir / f"{target_key}.npy")

    # Placement search grid — fix y at start_xy[1], search x and yaw.
    X_GRID = np.arange(0.25, 0.55, 0.05)
    YAW_GRID = np.linspace(0, 2 * np.pi, 8, endpoint=False)

    # Lift scene — table only, no held obj mesh (it's "attached" to the hand).
    scene_lift = {"mesh": {}, "cuboid": dict(scene_cfg["cuboid"])}

    # === Enumerate all IK-feasible grasp candidates upfront ===
    # solve_ik returns the full candidate list (load_candidate output) with a
    # per-candidate ik_success mask, so we know exactly how many table-scene
    # candidates exist and which are reachable, without re-running IK in a loop.
    print("[view] enumerating IK-feasible candidates...")
    ik_res = planner.solve_ik(scene_cfg, args.obj, GRASP_VERSION, hand=args.hand)
    n_total = ik_res["n_total"]
    # Match candidate scene_id to the sorted index of the start tabletop. Grasps
    # are stored under candidates/{hand}/{version}/{obj}/table/{scene_id}/ where
    # scene_id is the integer index of the tabletop pose ("0", "1", ...), while
    # tabletop poses on disk use padded names like "000.npy". Convert start_key
    # to its sorted index so the filter matches.
    sorted_tabletops = sorted(p.stem for p in tabletop_dir.glob("*.npy"))
    if start_key not in sorted_tabletops:
        sys.exit(f"start_tabletop '{start_key}' not in {sorted_tabletops}")
    start_scene_id = str(sorted_tabletops.index(start_key))
    scene_info = ik_res["scene_info"]
    ik_valid_indices = np.array(
        [i for i in np.where(ik_res["ik_success"])[0]
         if len(scene_info[i]) >= 2 and scene_info[i][1] == start_scene_id],
        dtype=int,
    )
    n_for_this_tabletop = sum(1 for s in scene_info
                              if len(s) >= 2 and s[1] == start_scene_id)
    print(f"[view] {len(ik_valid_indices)}/{n_for_this_tabletop} IK-feasible "
          f"for tabletop '{start_key}' (scene_id='{start_scene_id}')  "
          f"({n_total} total across all scene_ids)")
    if len(ik_valid_indices) == 0:
        sys.exit(f"no IK-feasible grasps for tabletop '{start_key}' "
                 f"(scene_id '{start_scene_id}') — "
                 f"available scene_ids: {sorted({s[1] for s in scene_info if len(s) >= 2})}")

    # solve_ik only inits ik_solver. _refine_fingers (plan_single_js) needs
    # motion_gen too — initialize it explicitly with the full scene (obj
    # mesh + table) so the approach plan respects the table-scene obj as
    # an obstacle. plan_wrist_reorient() will swap world to scene_lift later.
    planner._init_motion_gen(_to_curobo_world(scene_cfg))
    planner._cached_world = _to_curobo_world(scene_cfg)

    # solve_ik's solve_batch was called WITHOUT retract_config, baking that
    # goal type into cuRobo's CUDA graph. plan_wrist_reorient passes
    # retract_config and would crash with "changing goal type, cuda graph
    # reset not available" on CUDA < 12. Force re-init so the next IK call
    # recompiles the graph with retract-compatible goal type.
    planner._ik_solver = None

    # === Candidate retry loop ===
    # Iterate through every IK-feasible candidate until lift + reorient both
    # succeed. plan_single_js is called explicitly per candidate via
    # _refine_fingers so we don't re-run IK each time (cf. plan() which
    # bundles IK + planning and only returns the first success).
    chosen_try = None
    approach_traj = grasp_qpos = T_wrist_grasp = grasp_hand = None
    T_obj_in_wrist = lift_traj_full = reorient_traj = reorient_info = None
    T_wrist_reorient_goal = None
    pregrasp_hand = None

    for try_idx, cand_idx in enumerate(ik_valid_indices):
        cand_idx = int(cand_idx)
        ik_qpos = ik_res["ik_qpos"][cand_idx]
        T_wrist_grasp = ik_res["wrist_se3"][cand_idx]
        grasp_hand = ik_res["grasp"][cand_idx]
        pregrasp_pose = ik_res["pregrasp"][cand_idx]
        grasp_hand_arr = np.asarray(grasp_hand, dtype=np.float32)

        print(f"\n[try {try_idx + 1}/{len(ik_valid_indices)}] candidate idx={cand_idx} — approach plan_single_js...")
        ok, approach_traj = planner._refine_fingers(planner._init_state, ik_qpos)
        if not ok:
            print(f"  approach FAIL")
            continue
        grasp_qpos = approach_traj[-1].copy()
        T_obj_in_wrist = np.linalg.inv(T_wrist_grasp) @ T_obj_start

        # Lift (n_yaw=1, pure z-translation)
        T_wrist_lift = T_wrist_grasp.copy()
        T_wrist_lift[2, 3] += LIFT_HEIGHT_M
        cur_qpos = np.concatenate([grasp_qpos[:6], grasp_hand])
        lift_traj_full, lift_info = planner.plan_wrist_reorient(
            scene_lift, cur_qpos, T_wrist_lift,
            hold_hand_qpos=grasp_hand, n_yaw=1,
        )
        if lift_traj_full is None:
            print(f"  lift FAIL ({lift_info.get('reason')})")
            continue

        # Reorient — search (x, yaw) grid for an IK-feasible placement.
        # x/y placement of the released obj doesn't matter to the user, but
        # the obj must end up with the target tabletop orientation. Yaw
        # rotates obj around its own center (= world-z), x sweeps the arm
        # workspace; first IK-feasible combo wins.
        pregrasp_hand = np.asarray(pregrasp_pose, dtype=np.float32)
        if target_key == start_key:
            reorient_traj = lift_traj_full[-1:].copy()
            reorient_info = {"skipped": True}
            T_wrist_reorient_goal = fk_ee(urdf_fk, lift_traj_full[-1:].copy())[-1]
            chosen_try = try_idx
            print(f"  OK (reorient skipped — same tabletop)")
            break

        reorient_traj, reorient_info = search_reorient_placement(
            planner, scene_lift, lift_traj_full[-1].copy(), T_obj_in_wrist,
            T_target_table, float(args.start_xy[1]),
            TABLE_SURFACE_Z + LIFT_HEIGHT_M,
            grasp_hand_arr, X_GRID, YAW_GRID,
        )
        if reorient_traj is None:
            print(f"  reorient FAIL ({reorient_info.get('reason')}, "
                  f"feasible={reorient_info.get('n_feasible')}/"
                  f"{reorient_info.get('n_candidates')})")
            # Keep the last attempted target for the target-hand viz, even
            # though we move on to the next grasp candidate.
            if "T_wrist_target" in reorient_info:
                T_wrist_reorient_goal = reorient_info["T_wrist_target"]
            continue
        T_wrist_reorient_goal = reorient_info["T_wrist_target"]
        chosen_try = try_idx
        print(f"  OK  reorient x={reorient_info['chosen_x']:.3f}  "
              f"yaw={np.degrees(reorient_info['chosen_yaw']):.0f}°  "
              f"arm_dist={reorient_info['best_arm_dist_rad']:.3f}rad  "
              f"(feasible {reorient_info['n_feasible']}/"
              f"{reorient_info['n_candidates']})")
        break

    if chosen_try is None:
        if lift_traj_full is None:
            sys.exit(f"no candidate (out of {len(ik_valid_indices)}) succeeded approach+lift")
        print(f"\n[view] all {len(ik_valid_indices)} candidates failed reorient — "
              f"showing last attempt with no-op reorient (target hand still visible)")
        reorient_traj = lift_traj_full[-1:].copy()
        if T_wrist_reorient_goal is None:
            # No search ever returned a T_wrist_target (every lift succeeded
            # but every search hit n_feasible==0 path that already sets info).
            # Fall back to the lift end wrist as the viz reference.
            T_wrist_reorient_goal = fk_ee(urdf_fk, lift_traj_full[-1:].copy())[-1]
    print(f"\n[view] reorient target wrist pos = "
          f"{T_wrist_reorient_goal[:3, 3].round(3)}  "
          f"rpy(deg) = "
          f"{np.degrees(R.from_matrix(T_wrist_reorient_goal[:3, :3]).as_euler('XYZ')).round(1)}")

    # === GRASP_CLOSE: arm fixed, hand pregrasp → grasp ===
    Nb = 20
    b = np.linspace(0, 1, Nb)[:, None]
    hand_close = (1 - b) * pregrasp_hand[None, :] + b * grasp_hand_arr[None, :]
    grasp_close_traj = np.concatenate(
        [np.tile(grasp_qpos[:6][None], (Nb, 1)), hand_close], axis=1)

    # === DESCENT: wrist z -= (LIFT - RELEASE). Reorient_drop.py applies this
    # as a cartesian servo on link6; for viz we plan it with plan_wrist_reorient
    # (same orientation as reorient end, just z-shifted) so the motion is
    # collision-aware against the table.
    ee_after_reorient = fk_ee(urdf_fk, reorient_traj[-1:].copy())[-1]
    T_wrist_descent_goal = ee_after_reorient.copy()
    T_wrist_descent_goal[2, 3] -= (LIFT_HEIGHT_M - RELEASE_HEIGHT_M)

    cur_after_reorient = reorient_traj[-1].copy()
    # n_yaw=1 — same orientation as reorient end, just z-shift.
    descent_traj, descent_info = planner.plan_wrist_reorient(
        scene_lift, cur_after_reorient, T_wrist_descent_goal,
        hold_hand_qpos=grasp_hand_arr, n_yaw=1,
    )
    if descent_traj is None:
        print(f"[view] descent FAILED ({descent_info.get('reason')}) — "
              f"continuing with no-op trajectory")
        descent_traj = reorient_traj[-1:].copy()
    else:
        print(f"[view] descent: {descent_traj.shape}")

    # === RELEASE: arm fixed, hand grasp → pregrasp (no squeeze step).
    Nr = 20
    b_r = np.linspace(0, 1, Nr)[:, None]
    hand_open = (1 - b_r) * grasp_hand_arr[None, :] + b_r * pregrasp_hand[None, :]
    arm_release = np.tile(descent_traj[-1, :6][None], (Nr, 1))
    release_traj = np.concatenate([arm_release, hand_open], axis=1)

    # === HAND_INIT: arm fixed, hand pregrasp → init_hand (fully open).
    # Mirrors RealExecutor.reset_fallback step 1 (real.py:935-937).
    init_hand_qpos = planner._init_state[6:].astype(np.float32)
    Nh = 20
    b_h = np.linspace(0, 1, Nh)[:, None]
    hand_to_init = (1 - b_h) * pregrasp_hand[None, :] + b_h * init_hand_qpos[None, :]
    arm_at_descent_end = np.tile(descent_traj[-1, :6][None], (Nh, 1))
    hand_init_traj = np.concatenate([arm_at_descent_end, hand_to_init], axis=1)

    # === ARM_RETRACT: sequential joints to clear-view pose.
    # Mirrors RealExecutor.reset_fallback step 2 (real.py:944-951): clear_view
    # = XARM_INIT with joint 0 -60°; joint order [1, 2, 5, 0, 3, 4] (or
    # [2, 1, 5, 0, 3, 4] if joint 1 is below init). Each joint moves one at
    # a time while others stay fixed.
    xarm_init = planner._init_state[:6].astype(np.float32)
    clear_view = xarm_init.copy()
    clear_view[0] -= np.deg2rad(40.0)
    cur_arm = descent_traj[-1, :6].astype(np.float32).copy()
    joint_order = ([1, 2, 5, 0, 3, 4] if cur_arm[1] >= xarm_init[1]
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
    arm_retract = (np.concatenate(arm_blocks, axis=0) if arm_blocks
                   else running_arm[None].copy())
    hand_held_init = np.tile(init_hand_qpos[None], (len(arm_retract), 1))
    retract_traj = np.concatenate([arm_retract, hand_held_init], axis=1)

    # === viser ===
    obj_approach = np.tile(T_obj_start[None], (len(approach_traj), 1, 1))
    obj_grasp = np.tile(T_obj_start[None], (Nb, 1, 1))
    ee_lift = fk_ee(urdf_fk, lift_traj_full)
    obj_lift = ee_lift @ T_obj_in_wrist
    ee_reorient = fk_ee(urdf_fk, reorient_traj)
    obj_reorient = ee_reorient @ T_obj_in_wrist
    ee_descent = fk_ee(urdf_fk, descent_traj)
    obj_descent = ee_descent @ T_obj_in_wrist
    # After descent the hand opens / arm retracts — obj is released and
    # stays put at descent end pose for the rest of the visualization.
    obj_after_release = np.tile(obj_descent[-1][None], (1, 1, 1))
    obj_release = np.tile(obj_descent[-1][None], (len(release_traj), 1, 1))
    obj_hand_init = np.tile(obj_descent[-1][None], (len(hand_init_traj), 1, 1))
    obj_retract = np.tile(obj_descent[-1][None], (len(retract_traj), 1, 1))

    vis = ViserViewer(port_number=args.port)
    vis.add_robot("xarm", str(urdf_path))
    vis.add_object("obj", trimesh.load(str(mesh_path), process=False), T_obj_start)
    vis.add_floor(height=0.0)
    vis.add_traj("approach",  {"xarm": approach_traj},    {"obj": obj_approach})
    vis.add_traj("grasp",     {"xarm": grasp_close_traj}, {"obj": obj_grasp})
    vis.add_traj("lift",      {"xarm": lift_traj_full},   {"obj": obj_lift})
    vis.add_traj("reorient",  {"xarm": reorient_traj},    {"obj": obj_reorient})
    vis.add_traj("descent",   {"xarm": descent_traj},     {"obj": obj_descent})
    vis.add_traj("release",   {"xarm": release_traj},     {"obj": obj_release})
    vis.add_traj("hand_init", {"xarm": hand_init_traj},   {"obj": obj_hand_init})
    vis.add_traj("retract",   {"xarm": retract_traj},     {"obj": obj_retract})

    # === TARGET WRIST VIZ: floating-hand URDF placed at the reorient target,
    # finger joints at grasp_hand. Lets us see whether the IK target is even
    # reachable. Frame axes added too for orientation reference.
    floating_path = FLOATING_URDF_BY_HAND[args.hand]
    vis.add_robot("target_hand", str(floating_path), pose=T_wrist_reorient_goal)
    floating_urdf = yourdfpy.URDF.load(str(floating_path))
    # 6 floating-base joints (x,y,z,rx,ry,rz) stay at 0 — root pose above
    # already places the wrist link at T_wrist_reorient_goal; finger joints
    # share the same actuated-joint order as xarm_inspire_left URDF.
    n_actuated = len(floating_urdf.actuated_joints)
    target_cfg = np.zeros(n_actuated, dtype=np.float32)
    target_cfg[6:] = grasp_hand_arr
    vis.robot_dict["target_hand"].update_cfg(target_cfg)
    # Object would also be at the target if reorient succeeded — show it too.
    T_obj_at_target = T_wrist_reorient_goal @ T_obj_in_wrist
    vis.add_object("obj_target", trimesh.load(str(mesh_path), process=False),
                   T_obj_at_target)
    vis.add_frame("target_wrist_axes", T_wrist_reorient_goal)

    print(f"[view] viser at http://localhost:{args.port}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
