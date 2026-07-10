#!/usr/bin/env python3
"""Offline planning rehearsal — uses pose_world.npy + C2R.npy from a saved trial
to run the full reorient_drop planning chain (no perception, no robot, no
recording). Use this to iterate on sphere/clearance/planner tweaks against a
fixed scene without touching hardware.

The planning logic mirrors reorient_drop.py:626-1006 exactly (approach IK →
lift → reorient + descent search → save).

Usage:
    # latest pepsi trial, target_tabletop 001
    python src/experiment/reset/mock_plan.py --obj pepsi --target_tabletop 001

    # specific trial dir
    python src/experiment/reset/mock_plan.py \\
        --trial_dir ~/shared_data/AutoDex/experiment/reset_test/reorient_drop/inspire_left/pepsi/20260524_112431 \\
        --target_tabletop 001 --viz
"""
from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import trimesh
import yourdfpy
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from autodex.utils.path import project_dir, obj_path
from autodex.utils.conversion import cart2se3
from autodex.utils.symmetry import get_cyl_axis_local
from autodex.planner import GraspPlanner
from autodex.planner.planner import PlanResult, _to_curobo_world
from autodex.planner.obstacles import TABLE_CUBOID, add_obstacles

from src.execution.scene_cfg import pose_world_to_scene_cfg
from src.execution.run_auto import MESH_BASE, ASSETS_BASE
from paradex.visualization.visualizer.viser import ViserViewer
from paradex.calibration.utils import load_c2r

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logging.getLogger("curobo").setLevel(logging.WARNING)


GRASP_VERSION = "table_only"
LIFT_HEIGHT_M = 0.25
RELEASE_HEIGHT_M = 0.15
TABLE_SURFACE_Z = TABLE_CUBOID["pose"][2] + TABLE_CUBOID["dims"][2] / 2

_URDF_ROOT = Path.home() / "shared_data" / "AutoDex" / "content" / "assets" / "robot"
URDF_BY_HAND_VIZ = {
    "inspire_left": _URDF_ROOT / "inspire_left_description" / "xarm_inspire_left.urdf",
    "inspire":      _URDF_ROOT / "inspire_description"      / "xarm_inspire.urdf",
    "allegro":      _URDF_ROOT / "allegro_description"      / "xarm_allegro.urdf",
}


def _resolve_trial_dir(args) -> Path:
    if args.trial_dir:
        p = Path(args.trial_dir).expanduser()
        if not p.exists():
            sys.exit(f"trial dir not found: {p}")
        return p
    root = (Path.home() / "shared_data" / "AutoDex" / "experiment"
            / args.exp_name / args.hand / args.obj)
    if not root.exists():
        sys.exit(f"obj experiment dir not found: {root}")
    cands = sorted([p for p in root.iterdir() if p.is_dir()
                    and p.name[:1].isdigit()])
    if not cands:
        sys.exit(f"no trial subdirectories under {root}")
    return cands[-1]


def _load_target_tabletop(obj_name: str, key: str):
    tabletop_dir = os.path.join(obj_path, obj_name, "processed_data",
                                "info", "tabletop")
    files = sorted(glob.glob(os.path.join(tabletop_dir, "*.npy")))
    if not files:
        sys.exit(f"no tabletop poses at {tabletop_dir}")
    fname = f"{key}.npy"
    matched = next((i for i, f in enumerate(files)
                    if os.path.basename(f) == fname), None)
    if matched is None:
        avail = ", ".join(os.path.basename(f).replace(".npy", "") for f in files)
        sys.exit(f"target {fname!r} not in {tabletop_dir} — avail: [{avail}]")
    pose = np.load(files[matched])
    if pose.shape == (3, 3):
        T = np.eye(4); T[:3, :3] = pose
        pose = T
    return fname, matched, pose


def _phase_viz(args, mesh_path, urdf_viz_path, hand, planner,
               approach_traj, lift_traj, reorient_traj, descent_traj,
               wrist_grasp, pregrasp, grasp, T_obj_in_wrist,
               T_obj_grasp_world_full):
    """Phase-by-phase viser preview — mirrors reorient_drop.py viz block."""
    urdf_fk = yourdfpy.URDF.load(str(urdf_viz_path))
    pregrasp_h = np.asarray(pregrasp, dtype=np.float32)
    grasp_h = np.asarray(grasp, dtype=np.float32)
    init_hand_q = planner._init_state[6:].astype(np.float32)

    def _fk_ee(j):
        out = np.tile(np.eye(4), (len(j), 1, 1))
        for t, q in enumerate(j):
            urdf_fk.update_cfg(q)
            out[t] = urdf_fk.get_transform("base_link", urdf_fk.base_link)
        return out

    lift_v = lift_traj.copy(); lift_v[:, 6:] = grasp_h
    reor_v = reorient_traj.copy(); reor_v[:, 6:] = grasp_h
    desc_v = descent_traj.copy(); desc_v[:, 6:] = grasp_h

    Nb = 20
    b = np.linspace(0, 1, Nb)[:, None]
    grasp_close = np.concatenate(
        [np.tile(approach_traj[-1, :6][None], (Nb, 1)),
         (1 - b) * pregrasp_h[None] + b * grasp_h[None]], axis=1)
    Nr = 20
    b_r = np.linspace(0, 1, Nr)[:, None]
    release = np.concatenate(
        [np.tile(descent_traj[-1, :6][None], (Nr, 1)),
         (1 - b_r) * grasp_h[None] + b_r * pregrasp_h[None]], axis=1)

    obj_approach = np.tile(T_obj_grasp_world_full[None],
                            (len(approach_traj), 1, 1))
    obj_grasp = np.tile(T_obj_grasp_world_full[None], (Nb, 1, 1))
    ee_lift = _fk_ee(lift_v); obj_lift = ee_lift @ T_obj_in_wrist
    ee_reor = _fk_ee(reor_v); obj_reor = ee_reor @ T_obj_in_wrist
    ee_desc = _fk_ee(desc_v); obj_desc = ee_desc @ T_obj_in_wrist
    obj_rel = np.tile(obj_desc[-1][None], (len(release), 1, 1))

    vis = ViserViewer(port_number=args.port)
    vis.add_robot("xarm", str(urdf_viz_path))
    if mesh_path.exists():
        vis.add_object("obj", trimesh.load(str(mesh_path), process=False),
                       T_obj_grasp_world_full)
    vis.add_floor(height=0.0)
    vis.add_traj("approach", {"xarm": approach_traj}, {"obj": obj_approach})
    vis.add_traj("grasp",    {"xarm": grasp_close},   {"obj": obj_grasp})
    vis.add_traj("lift",     {"xarm": lift_v},        {"obj": obj_lift})
    vis.add_traj("reorient", {"xarm": reor_v},        {"obj": obj_reor})
    vis.add_traj("descent",  {"xarm": desc_v},        {"obj": obj_desc})
    vis.add_traj("release",  {"xarm": release},       {"obj": obj_rel})
    vis.start_viewer(use_thread=True)
    print(f"[viz] viser  http://localhost:{args.port}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[viz] bye")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial_dir", default=None)
    parser.add_argument("--obj", default=None,
                        help="Object name; used to find latest trial if "
                             "--trial_dir omitted; also for mesh / candidates.")
    parser.add_argument("--hand", default="inspire_left",
                        choices=["allegro", "inspire", "inspire_left"])
    parser.add_argument("--target_tabletop", required=True,
                        help="Target tabletop filename stem (e.g. '001').")
    parser.add_argument("--exp_name", default="reset_test/reorient_drop")
    parser.add_argument("--out_dir", default=None,
                        help="Where to save the mock plan. Default = "
                             "experiment/mock_plan/{hand}/{obj}/{trial_ts}_mock_{ts}/")
    parser.add_argument("--viz", action="store_true",
                        help="Open viser viewer at --port to preview the plan.")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    trial_dir = _resolve_trial_dir(args)
    obj_name = args.obj or trial_dir.parent.name
    print(f"[mock] trial = {trial_dir}")
    print(f"[mock] obj   = {obj_name}  hand = {args.hand}")

    pose_world = np.load(trial_dir / "pose_world.npy")
    c2r = load_c2r(str(trial_dir))

    tt_fn, tt_idx, target_tabletop_robot = _load_target_tabletop(
        obj_name, args.target_tabletop)
    print(f"[mock] target tabletop = {tt_fn} (sorted_idx={tt_idx})")

    out_dir = (Path(args.out_dir) if args.out_dir
               else Path(project_dir) / "experiment" / "mock_plan"
                    / args.hand / obj_name
                    / f"{trial_dir.name}_mock_{int(time.time())}")
    (out_dir / "plan").mkdir(parents=True, exist_ok=True)
    print(f"[mock] out   = {out_dir}")

    print("[mock] planner warmup...")
    planner = GraspPlanner(hand=args.hand)
    from curobo.util.logger import setup_curobo_logger
    setup_curobo_logger("warning")

    # ── Planning (mirrors reorient_drop.py:619-1006) ─────────────────────
    print("[mock] planning...")
    t0 = time.time()
    scene_cfg = pose_world_to_scene_cfg(pose_world, c2r, obj_name)
    scene_cfg = add_obstacles(scene_cfg, "table")
    with open(out_dir / "scene_cfg.json", "w") as f:
        json.dump(scene_cfg, f, indent=2, default=str)

    if planner._motion_gen is not None:
        planner._motion_gen.clear_world_cache()
        planner._motion_gen.reset_seed()

    ik_res = planner.solve_ik(scene_cfg, obj_name, GRASP_VERSION,
                               hand=args.hand, scene_id=None)
    ik_ok = list(np.where(ik_res["ik_success"])[0])
    np.random.shuffle(ik_ok)
    n_total = ik_res["n_total"]
    planner._ik_solver = None
    print(f"  grasp IK: {len(ik_ok)}/{n_total} feasible")
    if not ik_ok:
        sys.exit("no IK-feasible grasp candidate.")

    T_obj_grasp_world_full = cart2se3(scene_cfg["mesh"]["target"]["pose"])
    R_target_obj_world_pre = target_tabletop_robot[:3, :3]
    obj_target_pos_world_pre = np.array([
        0.0, 0.0,
        float(target_tabletop_robot[2, 3])
            + TABLE_SURFACE_Z + LIFT_HEIGHT_M,
    ])
    scene_lift_pre = {"mesh": {}, "cuboid": dict(scene_cfg["cuboid"])}
    X_GRID = np.arange(0.35, 0.55, 0.05)
    YAW_GRID = np.linspace(0, 2 * np.pi, 8, endpoint=False)

    CYL_AXIS_LOCAL = get_cyl_axis_local(obj_name)
    if CYL_AXIS_LOCAL is not None:
        axis_world = R_target_obj_world_pre @ CYL_AXIS_LOCAL
        if abs(axis_world[2]) >= 0.95:
            CYL_YAW_GRID = np.array([0.0])
        else:
            CYL_YAW_GRID = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    else:
        CYL_YAW_GRID = None

    world_approach = _to_curobo_world(scene_cfg)
    planner._init_motion_gen(world_approach)
    planner._cached_world = world_approach

    obj_target_pos_descent = np.array([
        0.0, 0.0,
        float(target_tabletop_robot[2, 3])
            + TABLE_SURFACE_Z + RELEASE_HEIGHT_M,
    ])

    chosen = None
    n_app = n_lift = n_reor = n_desc = 0
    for cand_idx in ik_ok:
        cand_idx = int(cand_idx)
        wrist_grasp = ik_res["wrist_se3"][cand_idx]
        grasp_q     = ik_res["grasp"][cand_idx]
        pregrasp_q  = ik_res["pregrasp"][cand_idx]
        T_obj_in_wrist = np.linalg.inv(wrist_grasp) @ T_obj_grasp_world_full

        if planner._world_structure_changed(world_approach):
            planner._update_world(world_approach)
            planner._cached_world = world_approach
        ok_ap, approach_traj = planner._refine_fingers(
            planner._init_state, ik_res["ik_qpos"][cand_idx])
        if not ok_ap:
            n_app += 1
            print(f"  cand#{cand_idx}: approach FAIL"); continue

        T_wrist_lift = wrist_grasp.copy()
        T_wrist_lift[2, 3] += LIFT_HEIGHT_M
        cur_lift = np.concatenate([approach_traj[-1, :6], pregrasp_q])
        lift_traj, lift_info = planner.plan_wrist_reorient(
            scene_lift_pre, cur_lift, T_wrist_lift,
            hold_hand_qpos=pregrasp_q, n_yaw=8)
        if lift_traj is None:
            n_lift += 1
            print(f"  cand#{cand_idx}: lift FAIL ({lift_info.get('reason')})")
            continue

        cur_reor = lift_traj[-1].copy()
        _, sorted_info = planner.plan_obj_placement(
            scene_lift_pre, cur_reor, T_obj_in_wrist,
            R_target_obj_world_pre, obj_target_pos_world_pre,
            hold_hand_qpos=pregrasp_q,
            x_grid=X_GRID, yaw_grid=YAW_GRID,
            cyl_yaw_grid=CYL_YAW_GRID, cyl_axis_local=CYL_AXIS_LOCAL,
            skip_plan=True)
        sorted_cands = sorted_info.get("sorted_candidates", [])
        x_center = 0.5 * (float(X_GRID[0]) + float(X_GRID[-1]))
        sorted_cands.sort(key=lambda sc: abs(sc["x"] - x_center))

        reor_traj = desc_traj = None
        for sc in sorted_cands:
            x1 = np.array([sc["x"]]); y1 = np.array([sc["yaw"]])
            cyl1 = (np.array([sc["cyl_yaw"]])
                    if CYL_YAW_GRID is not None else None)
            r_t, _ = planner.plan_obj_placement(
                scene_lift_pre, cur_reor, T_obj_in_wrist,
                R_target_obj_world_pre, obj_target_pos_world_pre,
                hold_hand_qpos=pregrasp_q, x_grid=x1, yaw_grid=y1,
                cyl_yaw_grid=cyl1, cyl_axis_local=CYL_AXIS_LOCAL,
                skip_plan=False)
            if r_t is None: continue
            cur_desc = r_t[-1].copy()
            d_t, _ = planner.plan_obj_placement(
                scene_lift_pre, cur_desc, T_obj_in_wrist,
                R_target_obj_world_pre, obj_target_pos_descent,
                hold_hand_qpos=pregrasp_q, x_grid=x1, yaw_grid=y1,
                cyl_yaw_grid=cyl1, cyl_axis_local=CYL_AXIS_LOCAL,
                skip_plan=False)
            if d_t is None: continue
            reor_traj, desc_traj = r_t, d_t
            break
        if reor_traj is None:
            if any(True for _ in sorted_cands): n_desc += 1
            else: n_reor += 1
            print(f"  cand#{cand_idx}: reorient/descent FAIL"); continue

        chosen = dict(cand_idx=cand_idx, approach=approach_traj,
                      lift=lift_traj, reorient=reor_traj, descent=desc_traj,
                      wrist_grasp=wrist_grasp, grasp=grasp_q,
                      pregrasp=pregrasp_q, T_obj_in_wrist=T_obj_in_wrist,
                      T_wrist_lift=T_wrist_lift)
        break

    dt = time.time() - t0
    if chosen is None:
        print(f"\n[mock] NO PLAN ({dt:.1f}s)  approach={n_app}  lift={n_lift}  "
              f"reorient={n_reor}  descent={n_desc}  total_tried={len(ik_ok)}")
        sys.exit(1)
    print(f"\n[mock] PLAN OK ({dt:.1f}s)  cand#{chosen['cand_idx']}  "
          f"skipped approach={n_app} lift={n_lift} reorient={n_reor} descent={n_desc}")

    # Save plan.
    p = out_dir / "plan"
    np.save(p / "traj.npy",          chosen["approach"])
    np.save(p / "wrist_se3.npy",     chosen["wrist_grasp"])
    np.save(p / "pregrasp_pose.npy", chosen["pregrasp"])
    np.save(p / "grasp_pose.npy",    chosen["grasp"])
    np.save(p / "lift_traj.npy",     chosen["lift"])
    np.save(p / "reorient_traj.npy", chosen["reorient"])
    np.save(p / "descent_traj.npy",  chosen["descent"])
    np.save(p / "T_obj_in_wrist.npy", chosen["T_obj_in_wrist"])
    np.save(p / "T_wrist_lift.npy",  chosen["T_wrist_lift"])
    np.save(out_dir / "pose_world.npy", pose_world)
    np.save(out_dir / "C2R.npy", c2r)
    print(f"[mock] saved → {out_dir}")

    if args.viz:
        mesh_path = MESH_BASE / obj_name / "raw_mesh" / f"{obj_name}.obj"
        urdf_viz_path = URDF_BY_HAND_VIZ[args.hand]
        _phase_viz(args, mesh_path, urdf_viz_path, args.hand, planner,
                   chosen["approach"], chosen["lift"], chosen["reorient"],
                   chosen["descent"], chosen["wrist_grasp"], chosen["pregrasp"],
                   chosen["grasp"], chosen["T_obj_in_wrist"],
                   T_obj_grasp_world_full)


if __name__ == "__main__":
    main()
