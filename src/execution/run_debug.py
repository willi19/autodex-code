#!/usr/bin/env python3
"""Debug / step-through runner: FoundPose init -> plan -> viser preview -> GUI exec.

Single-trial counterpart to ``run_auto.py``. Pipeline is identical up to planning;
after planning we launch the viser scene viewer and wait for ``y`` (execute on
robot via the GUI controller) or ``q`` (skip). Use this when you want to inspect
the planned trajectory before committing to an execution.

Prerequisites:
    bash scripts/init_daemons.sh start

Usage:
    python src/execution/run_debug.py --obj attached_container
    python src/execution/run_debug.py --obj brown_ramen --hand inspire_left
    python src/execution/run_debug.py --obj brown_ramen --scene wall --wall_angle 0
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path

import chime
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.utils.system import get_pc_ip, get_camera_list
from paradex.calibration.utils import save_current_camparam, save_current_C2R, load_c2r

from autodex.utils.path import project_dir
from autodex.planner import GraspPlanner
from autodex.planner.obstacles import add_obstacles
from autodex.planner.visualizer import ScenePlanVisualizer
from autodex.executor.real import RealExecutor
from autodex.perception.init_orchestrator import InitOrchestrator

from src.execution.scene_cfg import pose_world_to_scene_cfg
from src.execution.label import auto_label_charuco
from src.execution.run_auto import (
    DEFAULT_PC_LIST, ASSETS_BASE, MESH_BASE, CAM_PARAM_ROOT, _load_calib,
)

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")
logging.getLogger("curobo").setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", type=str, required=True)
    parser.add_argument("--grasp_version", type=str, default="selected_100")
    parser.add_argument("--exp_name", type=str, default="debug")
    parser.add_argument("--hand", type=str, default="allegro",
                        choices=["allegro", "inspire", "inspire_left"])
    parser.add_argument("--scene", type=str, default="table",
                        choices=["table", "wall", "shelf", "cluttered"])
    parser.add_argument("--success_only", action="store_true")

    # scene-specific args
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
    parser.add_argument("--prompt", type=str, default="object on the checkerboard")
    parser.add_argument("--sil_iters", type=int, default=100)
    parser.add_argument("--sil_lr", type=float, default=0.002)
    parser.add_argument("--init_timeout_s", type=float, default=120.0)
    parser.add_argument("--calib_dir", type=str, default=None)
    parser.add_argument("--stream_fps", type=int, default=10)
    parser.add_argument("--stream_warmup_s", type=float, default=2.0)
    parser.add_argument("--port_viser", type=int, default=8080)
    args = parser.parse_args()

    # Mesh / assets sanity.
    mesh_path = MESH_BASE / args.obj / "raw_mesh" / f"{args.obj}.obj"
    assets_root = ASSETS_BASE / args.obj
    if not mesh_path.exists():
        sys.exit(f"mesh not found: {mesh_path}")
    if not (assets_root / "object_repre/v1" / args.obj / "1/repre.pth").exists():
        sys.exit(f"repre.pth missing for {args.obj}")

    # Calibration.
    calib_dir = Path(args.calib_dir).expanduser() if args.calib_dir else sorted(CAM_PARAM_ROOT.iterdir())[-1]
    print(f"calib: {calib_dir.name}")
    intrinsics_full, extrinsics_full, H, W = _load_calib(calib_dir)

    pc_ips = [get_pc_ip(p) for p in args.pc_list]
    pc_serials = {p: get_camera_list(p) for p in args.pc_list}
    active = {s for pc in args.pc_list for s in pc_serials[pc]}
    intrinsics_full = {s: v for s, v in intrinsics_full.items() if s in active}
    extrinsics_full = {s: v for s, v in extrinsics_full.items() if s in active}
    print(f"  {len(intrinsics_full)} cams active across {len(args.pc_list)} PCs ({H}x{W})")

    # Hardware.
    rcc = remote_camera_controller("run_debug", pc_list=args.pc_list)
    print(f"[stream] starting on {len(args.pc_list)} PCs @ {args.stream_fps} FPS...")
    rcc.start("stream", False, fps=args.stream_fps)
    if args.stream_warmup_s > 0:
        time.sleep(args.stream_warmup_s)

    # Trial dir.
    scene_prefix = args.scene if args.scene != "table" else ""
    if args.success_only:
        scene_prefix = f"{scene_prefix}_success_only" if scene_prefix else "success_only"
    sub = f"{scene_prefix}/{args.hand}" if scene_prefix else args.hand
    dir_idx = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    img_dir = os.path.join(project_dir, "experiment", args.exp_name, sub, args.obj, dir_idx)
    os.makedirs(img_dir, exist_ok=True)
    save_current_C2R(img_dir)
    save_current_camparam(img_dir)

    timing: dict = {}
    def _ts(): return datetime.datetime.now().isoformat()

    # Init orchestrator.
    print(f"[orch] init for {args.obj}...")
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

    try:
        # ── 1. Perception ────────────────────────────────────────────────────
        print(f"\n[1/4] Perception (FoundPose distributed)...")
        timing["perception_start"] = _ts()
        t0 = time.time()
        pose_world, perc_timing = orch.trigger_init(
            prompt=args.prompt,
            save_capture_dir=os.path.join(img_dir, "init_capture"),
            sil_iters=args.sil_iters, sil_lr=args.sil_lr,
            timeout_s=args.init_timeout_s,
        )
        timing["perception_s"] = round(time.time() - t0, 2)
        if perc_timing:
            timing["perception_detail"] = perc_timing
        if pose_world is None:
            reason = (perc_timing or {}).get("reason", "perception_failed")
            print(f"    Perception FAILED ({reason})")
            with open(os.path.join(img_dir, "timing.json"), "w") as f:
                json.dump(timing, f, indent=2, default=str)
            return
        print(f"    Perception: {timing['perception_s']}s")
        np.save(os.path.join(img_dir, "pose_world.npy"), pose_world)

        # ── 2. Planning ──────────────────────────────────────────────────────
        print(f"[2/4] Planning (version={args.grasp_version}, scene={args.scene})...")
        timing["planning_start"] = _ts()
        t0 = time.time()
        c2r = load_c2r(img_dir)
        scene_cfg = pose_world_to_scene_cfg(pose_world, c2r, args.obj)
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
        # Classify perceived pose against the object's tabletop poses to
        # filter candidates to the matching scene_id + pick openpose_{stem}
        # appropriately.
        from src.experiment.reset.tabletop_pose import classify_tabletop_pose
        from autodex.utils.symmetry import get_cyl_axis_local
        pose_robot = np.linalg.inv(c2r) @ pose_world
        tb_before = classify_tabletop_pose(pose_robot, args.obj)
        if tb_before is not None:
            scene_id = str(tb_before["idx"])
            pose_stem = tb_before["filename"].replace(".npy", "")
            print(f"    [tabletop] idx={tb_before['idx']} "
                  f"({tb_before['filename']}) err={tb_before['rot_err_deg']:.1f}°")
        else:
            scene_id = None
            pose_stem = None
        # Cylinder freedom — expand candidates by symmetry-axis rotations when
        # the axis ends up horizontal at the current pose (lying cylinder).
        # Symmetry-axis enumerate. Continuous-revolute → 8 angles. Discrete
        # (e.g. blue_alarm 180°) → order-N grid from pose_symmetry.json.
        from autodex.utils.symmetry import get_cyl_yaw_grid as _get_cyl_yaw_grid
        _cyl_axis = get_cyl_axis_local(args.obj)
        _cyl_grid = _get_cyl_yaw_grid(args.obj)
        planner = GraspPlanner(hand=args.hand)
        result = planner.plan(
            scene_cfg, args.obj, args.grasp_version,
            skip_done=False,
            success_only=args.success_only, hand=args.hand,
            scene_id=scene_id,
            openpose_pose_stem=pose_stem,
            cyl_axis_local=_cyl_axis,
            cyl_yaw_grid=_cyl_grid,
        )
        timing["plan_s"] = round(time.time() - t0, 2)
        print(f"    Plan: {timing['plan_s']}s  success={result.success}")

        if not result.success:
            print("    Planning FAILED — launching visualizer with candidates...")
            wrist_se3, _, grasp_pose, filtered = planner.get_candidates(
                scene_cfg, args.obj, args.grasp_version,
                success_only=args.success_only,
                skip_done=False, hand=args.hand,
            )
            fv = ScenePlanVisualizer(scene_cfg, None, port=args.port_viser, hand=args.hand)
            fv.add_candidates(wrist_se3, grasp_pose, filtered)
            fv.start_viewer(use_thread=False)  # blocks
            return

        plan_dir = os.path.join(img_dir, "plan")
        os.makedirs(plan_dir, exist_ok=True)
        np.save(os.path.join(plan_dir, "traj.npy"), result.traj)
        np.save(os.path.join(plan_dir, "wrist_se3.npy"), result.wrist_se3)
        np.save(os.path.join(plan_dir, "pregrasp_pose.npy"), result.pregrasp_pose)
        np.save(os.path.join(plan_dir, "grasp_pose.npy"), result.grasp_pose)
        if result.timing:
            with open(os.path.join(plan_dir, "timing.json"), "w") as f:
                json.dump(result.timing, f, indent=2)
        print(f"    Scene info: {result.scene_info}")

        # ── 3. Viser preview ─────────────────────────────────────────────────
        print(f"[3/4] Viser preview (http://localhost:{args.port_viser})...")
        vis = ScenePlanVisualizer(scene_cfg, result, port=args.port_viser, hand=args.hand)
        # Also overlay all candidates (red = filtered, green = valid) so we can
        # scrub through them with the "Candidate #" slider.
        cand_wrist, _, cand_grasp, cand_filtered = planner.get_candidates(
            scene_cfg, args.obj, args.grasp_version,
            success_only=args.success_only,
            skip_done=False, hand=args.hand,
        )
        vis.add_candidates(cand_wrist, cand_grasp, cand_filtered)
        vis.start_viewer(use_thread=True)
        chime.info()

        while True:
            ans = input("Press 'y' to execute on robot (GUI), 'q' to skip: ").strip().lower()
            if ans in ("y", "q"):
                break
        if ans == "q":
            print("Skipping execution.")
            with open(os.path.join(img_dir, "timing.json"), "w") as f:
                json.dump(timing, f, indent=2, default=str)
            return

        # ── 4. Execute ───────────────────────────────────────────────────────
        print(f"[4/4] Executing...")
        timing["execution_start"] = _ts()
        executor = RealExecutor(hand_name=args.hand)
        s_hand = executor.execute(result, planner=planner)  # grasp + lift (z-constrained)

        # Auto-label probe (info only — does NOT decide trial success).
        label_rel = os.path.join("shared_data", "AutoDex", "experiment",
                                 args.exp_name, sub, args.obj, dir_idx,
                                 "label_at_lift", "raw")
        label_abs = os.path.join(img_dir, "label_at_lift", "raw", "images")
        # Pause continuous stream so rcc.start("image", ...) is accepted.
        try:
            rcc.stop()
        except Exception:
            pass
        rcc.start("image", False, label_rel)
        rcc.stop()
        time.sleep(0.5)   # let capture PCs flush their PNGs to NFS
        auto_succ, label_info = auto_label_charuco(label_abs, required_board="1")
        if label_info.get("reason"):
            print(f"[auto-label] FAILED ({label_info['reason']})  dir={label_abs}")
        else:
            print(f"[auto-label] success={auto_succ}  "
                  f"board1 covered {label_info['covered']}/{label_info['expected']}  "
                  f"(missing IDs: {label_info['missing_ids']})")
        # Resume stream for any future operations (next trial would need it).
        rcc.start("stream", False, fps=args.stream_fps)

        place_info = executor.place(
            result,
            log_path=os.path.join(img_dir, "place_mcc_log.csv"),
        )
        timing["execution_states"] = executor.state_timestamps
        timing["place"] = place_info
        timing["auto_label"] = label_info

        executor.release(result)
        if s_hand is not None:
            np.save(os.path.join(img_dir, "squeeze_hand.npy"), s_hand)

        # success=None: trial just iterates, no manual/auto pass/fail decision.
        trial_result = {
            "dir_idx": dir_idx,
            "scene_type": args.scene,
            "success": None,
            "auto_label": label_info,
            "scene_info": result.scene_info,
            "candidate_idx": result.timing.get("candidate_idx") if result.timing else None,
            "timing": timing,
        }
        with open(os.path.join(img_dir, "result.json"), "w") as f:
            json.dump(trial_result, f, indent=2, default=str)

        if result.scene_info is not None and args.scene == "table":
            from autodex.utils.path import get_candidate_path
            sei = result.scene_info
            cand_result_path = os.path.join(
                get_candidate_path(args.hand), args.grasp_version, args.obj,
                sei[0], sei[1], sei[2], "result.json",
            )
            with open(cand_result_path, "w") as f:
                json.dump({"success": None, "dir_idx": dir_idx}, f)

        print(f"Result saved to {img_dir}/result.json")
        executor.shutdown()
    finally:
        try:
            orch.close()
        except Exception:
            pass
        for fn in (rcc.stop, rcc.end):
            try:
                fn()
            except Exception:
                pass


if __name__ == "__main__":
    main()
