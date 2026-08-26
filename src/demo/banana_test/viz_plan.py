#!/usr/bin/env python3
"""Viser preview of the demo's WHOLE planned motion — no robot, no execution.

Plans exactly what ``run_demo.py`` would plan for the object's CURRENT pose
(same success-grasp pool, same place-reach filter, same drop yaw), then shows
all four phases in one scrubbable player:

    grasp -> lift (+10cm) -> carry to the marker at constant height -> drop 3cm

From the grasp onward the object is attached RIGIDLY to the wrist, so the
object mesh in the viewer follows FK the way it will in reality — if the object
sweeps through the table or an obstacle here, it will there too.

    # current object pose, straight off the capture PCs (daemons must be up)
    python src/demo/banana_test/viz_plan.py --obj banana

    # replay a finished trial's perceived pose instead (no cameras needed)
    python src/demo/banana_test/viz_plan.py --obj banana --trial <trial_dir>
    python src/demo/banana_test/viz_plan.py --obj banana --trial latest
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from autodex.planner import GraspPlanner
from autodex.planner.planner import _to_curobo_world
from autodex.planner.obstacles import add_obstacles, TABLE_CUBOID
from autodex.planner.visualizer import ScenePlanVisualizer
from autodex.utils.conversion import cart2se3
from autodex.utils.path import project_dir, get_obj_root
from autodex.utils.symmetry import get_cyl_axis_local, get_cyl_yaw_grid

from src.execution.scene_cfg import pose_world_to_scene_cfg
from src.experiment.reset.tabletop_pose import classify_tabletop_pose

from src.demo.banana_test.place_target import locate_marker, capture_images
from src.demo.banana_test.success_grasps import success_keys_at_pose
from src.demo.banana_test.run_demo import (
    DEFAULT_PC_LIST, DEFAULT_TARGET_CAPTURE, LIFT_HEIGHT,
    _planner_robot, _yaw_grid, _place_wrist, filter_by_place_reach,
)


def fk_wrist(planner, qpos: np.ndarray) -> np.ndarray:
    """WRIST (= cuRobo ee_link) pose for a full arm+hand config."""
    import torch
    from scipy.spatial.transform import Rotation as R
    kin = planner._motion_gen.kinematics.get_state(
        torch.tensor(np.asarray(qpos, dtype=np.float32),
                     device=planner._tensor_args.device).unsqueeze(0))
    pos = kin.ee_position[0].detach().cpu().numpy()
    q = kin.ee_quaternion[0].detach().cpu().numpy()      # wxyz
    T = np.eye(4)
    T[:3, :3] = R.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    T[:3, 3] = pos
    return T


def pose_from_live(args):
    """Trigger one FoundPose init on the capture PCs and return (pose_world, c2r)."""
    from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
    from paradex.utils.system import get_pc_ip, get_camera_list
    from paradex.calibration.utils import load_current_C2R
    from autodex.perception.init_orchestrator import InitOrchestrator
    from src.demo.banana_test.run_demo import (_load_calib, _rcc_start,
                                                ASSETS_BASE, MESH_BASE,
                                                CAM_PARAM_ROOT)

    calib_dir = sorted(CAM_PARAM_ROOT.iterdir())[-1]
    intr, extr, H, W = _load_calib(calib_dir)
    pc_serials = {pc: get_camera_list(pc) for pc in args.pc_list}
    active = {s for pc in args.pc_list for s in pc_serials[pc]}
    intr = {s: v for s, v in intr.items() if s in active}
    extr = {s: v for s, v in extr.items() if s in active}

    rcc = remote_camera_controller("banana_viz", pc_list=args.pc_list)
    _rcc_start(rcc, "stream", False, fps=args.stream_fps)
    time.sleep(args.stream_warmup_s)
    orch = InitOrchestrator(
        pc_list=args.pc_list,
        capture_ips=[get_pc_ip(pc) for pc in args.pc_list],
        port_mask=args.port_mask, port_pose=args.port_pose,
        port_cmd=args.port_cmd)
    try:
        orch.init_object(
            obj_name=args.obj,
            mesh_path=str(MESH_BASE / args.obj / "raw_mesh" / f"{args.obj}.obj"),
            assets_root=str(ASSETS_BASE / args.obj),
            intrinsics_full=intr, extrinsics_full=extr,
            image_hw=(H, W), mode="live", pc_serials=pc_serials)
        pose_world, t = orch.trigger_init(
            prompt=args.prompt, sil_iters=args.sil_iters, sil_lr=args.sil_lr,
            timeout_s=args.init_timeout_s)
    finally:
        for fn in (orch.close, rcc.stop, rcc.end):
            try:
                fn()
            except Exception:
                pass
    if pose_world is None:
        sys.exit(f"[viz] perception failed: {t}")
    return pose_world, load_current_C2R()


def pose_from_trial(trial: str, args):
    d = Path(trial)
    if trial == "latest":
        run_dir = Path(project_dir) / "experiment" / args.exp_name / \
            f"{args.arm}_{args.hand}" / args.obj
        cands = [p for p in sorted(run_dir.iterdir())
                 if (p / "pose_world.npy").exists()]
        if not cands:
            sys.exit(f"[viz] no trial with pose_world.npy under {run_dir}")
        d = cands[-1]
    print(f"[viz] trial: {d}")
    return np.load(d / "pose_world.npy"), np.load(d / "C2R.npy")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obj", default="banana")
    p.add_argument("--hand", default="inspire")
    p.add_argument("--arm", default="franka", choices=["xarm", "franka"])
    p.add_argument("--grasp_version", default="v8")
    p.add_argument("--exp_name", default="banana_demo")
    p.add_argument("--trial", default=None,
                   help="trial dir (or 'latest') to take the object pose from; "
                        "default = run perception now")
    p.add_argument("--allow_other_pose", action="store_true")
    p.add_argument("--target_capture_dir", default=DEFAULT_TARGET_CAPTURE)
    p.add_argument("--marker_id", type=int, default=None)
    p.add_argument("--marker_dict", default="6X6_1000")
    p.add_argument("--target_yaw_deg", type=float, default=None)
    p.add_argument("--yaw_step", type=int, default=10)
    p.add_argument("--drop_h", type=float, default=0.03)
    p.add_argument("--port", type=int, default=8080)
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
    args = p.parse_args()

    obj, hand = args.obj, args.hand
    planner_robot = _planner_robot(args.arm, hand)

    # ── place target ─────────────────────────────────────────────────────────
    cap = (os.path.expanduser(args.target_capture_dir)
           if args.target_capture_dir else capture_images(pc_list=args.pc_list))
    tinfo = locate_marker(cap, dict_type=args.marker_dict,
                          marker_id=args.marker_id)
    target_xyz = np.asarray(tinfo["center_robot"])
    print(f"[viz] marker id={tinfo['marker_id']} center_robot="
          f"{target_xyz.round(4)}  (from {cap})")

    # ── object pose ──────────────────────────────────────────────────────────
    if args.trial:
        pose_world, c2r = pose_from_trial(args.trial, args)
    else:
        pose_world, c2r = pose_from_live(args)

    obj_root = get_obj_root(args.grasp_version)
    scene_cfg = add_obstacles(
        pose_world_to_scene_cfg(pose_world, c2r, obj, obj_root), "table")
    pose_robot = np.linalg.inv(c2r) @ pose_world
    tb = classify_tabletop_pose(pose_robot, obj, obj_root)
    pose_stem = tb["filename"].replace(".npy", "") if tb else None
    print(f"[viz] obj pos (robot) {pose_robot[:3, 3].round(3)}  "
          f"tabletop={pose_stem}")

    at_pose, any_pose = success_keys_at_pose(obj, hand, args.grasp_version,
                                             pose_stem, arm=args.arm)
    cand_order = at_pose or (any_pose if args.allow_other_pose else [])
    if not cand_order:
        sys.exit(f"[viz] no success grasp for tabletop {pose_stem} "
                 f"(arm={args.arm}); pass --allow_other_pose")

    planner = GraspPlanner(hand=planner_robot)
    if getattr(planner, "_ik_solver", None) is None:
        planner._init_ik_solver(_to_curobo_world(
            {"mesh": {}, "cuboid": {"table": TABLE_CUBOID}}))
    T_obj_grasp = cart2se3(scene_cfg["mesh"]["target"]["pose"])
    yaws = _yaw_grid(args.target_yaw_deg, args.yaw_step)
    reach = filter_by_place_reach(planner, cand_order, obj, hand,
                                  args.grasp_version, T_obj_grasp,
                                  target_xyz[:2], yaws)
    print(f"[viz] {len(cand_order)} success grasps, {len(reach)} can reach the drop")
    if reach:
        cand_order = reach

    result = planner.plan(
        scene_cfg, obj, args.grasp_version,
        skip_done=False, success_only=False, hand=hand,
        scene_type_filter=None, skip_scenes_with_success=False,
        openpose_pose_stem=pose_stem,
        cyl_axis_local=get_cyl_axis_local(obj),
        cyl_yaw_grid=get_cyl_yaw_grid(obj),
        candidate_order=cand_order)
    if not result.success:
        sys.exit(f"[viz] grasp plan failed: {result.timing}")
    print(f"[viz] grasp plan OK  scene_info={result.scene_info}")

    adof = 7 if args.arm == "franka" else 6
    sv = ScenePlanVisualizer(scene_cfg, result, port=args.port,
                             hand=planner_robot)

    # ── phases after the grasp: object rides the wrist rigidly ───────────────
    grasp_end = np.asarray(result.traj[-1], dtype=np.float32)
    T_wrist_grasp = fk_wrist(planner, grasp_end)
    # FK-derived (not result.wrist_se3) so the object does not jump between the
    # planned grasp pose and the first lift frame.
    T_obj_in_wrist = np.linalg.inv(T_wrist_grasp) @ T_obj_grasp

    def obj_traj(traj: np.ndarray) -> np.ndarray:
        return np.array([fk_wrist(planner, q) @ T_obj_in_wrist for q in traj])

    def full(q_arm, ):
        return np.concatenate([np.asarray(q_arm[:adof], dtype=np.float32),
                               np.asarray(result.grasp_pose, dtype=np.float32)])

    # 1. lift straight up
    lift_wrist = T_wrist_grasp.copy()
    lift_wrist[2, 3] += LIFT_HEIGHT
    lift = planner.plan_pose_constrained(
        full(grasp_end), lift_wrist, hold_vec_weight=[1, 1, 1, 1, 1, 0],
        scene_cfg=scene_cfg, include_obj_obstacle=False)
    if lift is None:
        print("[viz] lift plan FAILED — showing grasp only")
    else:
        sv.add_traj("lift", {"traj_robot": lift},
                    obj_traj={"mesh_target": obj_traj(lift)})

        # 2. carry to the marker at constant height, drop yaw chosen like run_demo
        lift_end = np.asarray(lift[-1], dtype=np.float32)
        T_wrist_lift = fk_wrist(planner, lift_end)
        obj_z = float((T_wrist_lift @ T_obj_in_wrist)[2, 3])
        probe = np.array([_place_wrist(T_obj_grasp, T_obj_in_wrist, target_xyz,
                                       y, obj_z, float(T_wrist_lift[2, 3]))
                          for y in yaws])
        ok = np.asarray(planner.ik_pose_batch(probe)).reshape(-1)
        feasible = [y for y, f in zip(yaws, ok) if f]
        if not feasible:
            print("[viz] drop spot IK-infeasible at every yaw — carry/drop "
                  "phases omitted")
        else:
            place_yaw = feasible[0]
            print(f"[viz] drop yaw = {place_yaw:+.0f}deg "
                  f"({len(feasible)}/{len(yaws)} feasible)")
            carry = planner.plan_pose_constrained(
                full(lift_end),
                _place_wrist(T_obj_grasp, T_obj_in_wrist, target_xyz,
                             place_yaw, obj_z, float(T_wrist_lift[2, 3])),
                hold_vec_weight=[0, 0, 0, 0, 0, 1],     # hold z
                scene_cfg=scene_cfg, include_obj_obstacle=False)
            if carry is None:
                print("[viz] carry plan FAILED")
            else:
                sv.add_traj("carry", {"traj_robot": carry},
                            obj_traj={"mesh_target": obj_traj(carry)})
                # 3. drop: descend drop_h
                carry_end = np.asarray(carry[-1], dtype=np.float32)
                drop_wrist = fk_wrist(planner, carry_end)
                drop_wrist[2, 3] -= args.drop_h
                drop = planner.plan_pose_constrained(
                    full(carry_end), drop_wrist,
                    hold_vec_weight=[1, 1, 1, 1, 1, 0],
                    scene_cfg=scene_cfg, include_obj_obstacle=False)
                if drop is None:
                    print("[viz] drop plan FAILED")
                else:
                    sv.add_traj("drop", {"traj_robot": drop},
                                obj_traj={"mesh_target": obj_traj(drop)})

    sv.start_viewer(use_thread=True)
    print(f"[viz] http://localhost:{args.port}  — press Playing, or scrub "
          f"the timestep slider. Ctrl-C to quit.")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        sv.stop_viewer()


if __name__ == "__main__":
    main()
