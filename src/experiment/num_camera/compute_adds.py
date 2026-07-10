#!/usr/bin/env python3
"""ADD-S vs camera-count ablation.

For each trial:
    1. Press Enter to start (object placed on table).
    2. Capture once (all ~24 cams publish mask + FoundPose).
    3. Refine with FULL camera set → pose_full (baseline).
    4. For each k in --ks (default 2,4,8,16):
         - Pick k random serials.
         - Refine with that subset → pose_k.
         - ADD-S(pose_full, pose_k) using mesh point cloud.
    5. Save all poses + ADD-S per k to trial dir.

Skips planning / execution / video — perception only.

Prerequisites:
    bash scripts/init_daemons.sh start

Usage:
    python src/experiment/num_camera/compute_adds.py --obj donut
    python src/experiment/num_camera/compute_adds.py --obj donut --ks 2 4 8 16
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
from typing import Dict, List, Optional

import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
from paradex.utils.system import get_pc_ip, get_camera_list

from autodex.utils.path import project_dir
from autodex.perception.init_orchestrator import InitOrchestrator

from src.experiment.num_camera.run_auto import (
    DEFAULT_PC_LIST, ASSETS_BASE, MESH_BASE, CAM_PARAM_ROOT, _load_calib,
)

logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")


def add_s(pose_a: np.ndarray, pose_b: np.ndarray,
          mesh_points: np.ndarray) -> float:
    """ADD-S = mean nearest-neighbor distance between mesh transformed by 2 poses."""
    from scipy.spatial import cKDTree
    pa = (pose_a[:3, :3] @ mesh_points.T).T + pose_a[:3, 3]
    pb = (pose_b[:3, :3] @ mesh_points.T).T + pose_b[:3, 3]
    tree = cKDTree(pb)
    d, _ = tree.query(pa, k=1)
    return float(np.mean(d))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--obj", type=str, required=True)
    p.add_argument("--ks", type=int, nargs="+", default=[2, 4, 8, 16])
    p.add_argument("--n_mesh_points", type=int, default=4096)
    p.add_argument("--exp_name", type=str, default="num_camera_adds")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--pc_list", type=str, nargs="+", default=DEFAULT_PC_LIST)
    p.add_argument("--port_mask", type=int, default=5006)
    p.add_argument("--port_pose", type=int, default=5007)
    p.add_argument("--port_cmd", type=int, default=6893)
    p.add_argument("--prompt", type=str, default="object on the checkerboard")
    p.add_argument("--sil_iters", type=int, default=100)
    p.add_argument("--sil_lr", type=float, default=0.002)
    p.add_argument("--init_timeout_s", type=float, default=120.0)
    p.add_argument("--calib_dir", type=str, default=None)
    p.add_argument("--stream_fps", type=int, default=10)
    p.add_argument("--stream_warmup_s", type=float, default=2.0)
    args = p.parse_args()

    if args.seed is not None:
        random.seed(args.seed)

    mesh_path = MESH_BASE / args.obj / "raw_mesh" / f"{args.obj}.obj"
    assets_root = ASSETS_BASE / args.obj
    if not mesh_path.exists():
        sys.exit(f"mesh not found: {mesh_path}")
    if not (assets_root / "object_repre/v1" / args.obj / "1/repre.pth").exists():
        sys.exit(f"repre.pth missing for {args.obj}")

    # Mesh point sample for ADD-S.
    mesh = trimesh.load(str(mesh_path), process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    mesh_points = mesh.sample(args.n_mesh_points).astype(np.float64)
    print(f"  mesh sampled: {mesh_points.shape}")

    # Calibration.
    calib_dir = (Path(args.calib_dir).expanduser() if args.calib_dir
                 else sorted(CAM_PARAM_ROOT.iterdir())[-1])
    print(f"calib: {calib_dir.name}")
    intrinsics_full, extrinsics_full, H, W = _load_calib(calib_dir)

    pc_ips = [get_pc_ip(pc) for pc in args.pc_list]
    pc_serials = {pc: get_camera_list(pc) for pc in args.pc_list}
    active = {s for pc in args.pc_list for s in pc_serials[pc]}
    intrinsics_full = {s: v for s, v in intrinsics_full.items() if s in active}
    extrinsics_full = {s: v for s, v in extrinsics_full.items() if s in active}
    active_serials = sorted(active)
    n_full = len(active_serials)
    print(f"  {n_full} cams active across {len(args.pc_list)} PCs ({H}x{W})")
    for k in args.ks:
        if k > n_full:
            sys.exit(f"--ks contains {k} > active cams {n_full}")

    # Hardware: stream so daemons get live SHM frames.
    rcc = remote_camera_controller("compute_adds", pc_list=args.pc_list)
    print(f"[stream] starting on {len(args.pc_list)} PCs...")
    rcc.start("stream", False, fps=args.stream_fps)
    if args.stream_warmup_s > 0:
        time.sleep(args.stream_warmup_s)

    # Orchestrator.
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

    out_root = Path(project_dir) / "experiment" / args.exp_name / args.obj
    out_root.mkdir(parents=True, exist_ok=True)

    results: List[dict] = []
    trial = 0
    try:
        while True:
            trial += 1
            print(f"\n{'#'*60}\n# ADD-S trial {trial}\n{'#'*60}")
            try:
                cmd = input("Press Enter to start (q to quit): ").strip().lower()
            except KeyboardInterrupt:
                break
            if cmd == "q":
                break

            dir_idx = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            trial_dir = out_root / dir_idx
            trial_dir.mkdir(parents=True, exist_ok=True)

            # 1. Capture once.
            print("[1] capture (collect_payloads)...")
            t0 = time.time()
            masks, poses, col_timing = orch.collect_payloads(
                prompt=args.prompt,
                n_expected_serials=n_full,
                timeout_s=args.init_timeout_s,
                save_capture_dir=str(trial_dir / "init_capture"),
            )
            print(f"    collected in {time.time()-t0:.2f}s "
                  f"(masks={col_timing['n_masks_recv']}, "
                  f"poses={col_timing['n_poses_recv']})")

            # 2. Full baseline.
            print("[2] refine FULL ...")
            t0 = time.time()
            pose_full, full_timing = orch.refine_from_payloads(
                masks, poses, subset_serials=None,
                sil_iters=args.sil_iters, sil_lr=args.sil_lr,
                sil_loss_threshold=float("inf"),
                save_capture_dir=str(trial_dir / "refine_full"),
            )
            print(f"    full refine {time.time()-t0:.2f}s  "
                  f"sil_loss={full_timing.get('sil_loss')}")
            if pose_full is None:
                print(f"    FULL refine FAILED ({full_timing.get('reason')}) — skip trial")
                json.dump({"reason": full_timing.get("reason")},
                          open(trial_dir / "result.json", "w"), indent=2)
                continue
            np.save(trial_dir / "pose_full.npy", pose_full)

            # 3. Subset per k.
            adds_per_k: Dict[int, dict] = {}
            for k in args.ks:
                chosen = sorted(random.sample(active_serials, k))
                print(f"[3] refine k={k} subset={chosen}")
                t0 = time.time()
                pose_k, k_timing = orch.refine_from_payloads(
                    masks, poses, subset_serials=chosen,
                    sil_iters=args.sil_iters, sil_lr=args.sil_lr,
                    sil_loss_threshold=float("inf"),
                    save_capture_dir=str(trial_dir / f"refine_k{k}"),
                )
                dt = time.time() - t0
                entry = {"chosen": chosen, "timing": k_timing,
                         "refine_s": dt}
                if pose_k is None:
                    entry["pose"] = None
                    entry["adds_m"] = None
                    print(f"    k={k} refine FAILED ({k_timing.get('reason')})")
                else:
                    np.save(trial_dir / f"pose_k{k}.npy", pose_k)
                    adds = add_s(pose_full, pose_k, mesh_points)
                    entry["adds_m"] = adds
                    print(f"    k={k}  ADD-S = {adds*1000:.2f} mm  "
                          f"sil_loss={k_timing.get('sil_loss'):.6f}")
                adds_per_k[k] = entry

            trial_result = {
                "dir_idx": dir_idx,
                "obj": args.obj,
                "n_full": n_full,
                "n_mesh_points": args.n_mesh_points,
                "collect_timing": col_timing,
                "full_timing": full_timing,
                "k_entries": adds_per_k,
            }
            with open(trial_dir / "result.json", "w") as f:
                json.dump(trial_result, f, indent=2, default=str)
            results.append(trial_result)

    finally:
        # Summary.
        if results:
            print(f"\n{'='*60}\nSUMMARY: {args.obj}  n_trials={len(results)}")
            print(f"{'k':>3}  {'mean_ADD-S_mm':>14}  {'std_mm':>8}  {'n':>3}")
            for k in args.ks:
                vals = [r["k_entries"][k]["adds_m"] for r in results
                        if r["k_entries"][k].get("adds_m") is not None]
                if vals:
                    arr = np.array(vals) * 1000.0
                    print(f"{k:>3}  {arr.mean():>14.2f}  {arr.std():>8.2f}  {len(vals):>3}")
                else:
                    print(f"{k:>3}  {'—':>14}  {'—':>8}  {0:>3}")
            with open(out_root / "summary.json", "w") as f:
                json.dump(results, f, indent=2, default=str)
        try:
            orch.close()
        except Exception:
            pass
        for fn in (rcc.stop, rcc.end):
            try: fn()
            except Exception: pass


if __name__ == "__main__":
    main()
