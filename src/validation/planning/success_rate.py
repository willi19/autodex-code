"""
Planning Success Rate

Grid search over (pose, x_offset, z_rotation) and run full planning
(IK + plan_single_js) on each point. Measures per-trial success rate.

Usage:
    # Single object
    python src/validation/planning/success_rate.py --obj blue_vase --version selected_100

    # All objects with tabletop poses and grasp candidates
    python src/validation/planning/success_rate.py --version selected_100

    # Custom grid
    python src/validation/planning/success_rate.py --version selected_100 \
        --x_min 0.2 --x_max 0.5 --x_step 0.05 --z_step 30 --n_trials 5
"""

import os
import sys
import argparse
import json
import logging
import time
import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "paradex"))

from autodex.planner import GraspPlanner
from autodex.utils.path import obj_path, get_candidate_path
from autodex.utils.conversion import se32cart


def load_tabletop_scene(obj_name, pose_idx="000", x_offset=0.4, z_rotation=0.0):
    """Load object pose with configurable x offset and z rotation."""
    pose_dir = os.path.join(obj_path, obj_name, "processed_data", "info", "tabletop")
    pose_file = os.path.join(pose_dir, f"{pose_idx}.npy")
    if not os.path.exists(pose_file):
        available = sorted(os.listdir(pose_dir))
        raise FileNotFoundError(f"Pose {pose_idx} not found. Available: {available}")

    obj_pose = np.load(pose_file)
    obj_pose[0, 3] += x_offset

    if z_rotation != 0.0:
        c, s = np.cos(z_rotation), np.sin(z_rotation)
        Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        obj_pose[:3, :3] = Rz @ obj_pose[:3, :3]

    mesh_path = os.path.join(obj_path, obj_name, "processed_data", "mesh", "simplified.obj")

    scene_cfg = {
        "mesh": {
            "target": {
                "pose": se32cart(obj_pose).tolist(),
                "file_path": mesh_path,
            },
        },
        "cuboid": {
            "table": {
                "dims": [2, 3, 0.2],
                "pose": [1.1, 0, -0.1, 1, 0, 0, 0],
            },
        },
    }
    return scene_cfg


def get_tabletop_poses(obj_name):
    """Get all available tabletop pose indices."""
    pose_dir = os.path.join(obj_path, obj_name, "processed_data", "info", "tabletop")
    if not os.path.isdir(pose_dir):
        return []
    return sorted([f.replace(".npy", "") for f in os.listdir(pose_dir) if f.endswith(".npy")])


def get_all_objects(version, hand="allegro"):
    """Find objects that have both tabletop poses and grasp candidates."""
    cand_dir = os.path.join(get_candidate_path(hand), version)
    if not os.path.isdir(cand_dir):
        return []
    cand_objects = set(os.listdir(cand_dir))
    objects = []
    for obj_name in sorted(cand_objects):
        tabletop_dir = os.path.join(obj_path, obj_name, "processed_data", "info", "tabletop")
        if os.path.isdir(tabletop_dir) and len(os.listdir(tabletop_dir)) > 0:
            objects.append(obj_name)
    return objects


def run_planning_grid(obj_name, grasp_version, n_trials,
                      x_offsets, z_rotations_deg, save_dir=None, hand="allegro"):
    """Run planning grid search for one object. Supports resume from partial results."""
    pose_indices = get_tabletop_poses(obj_name)
    if not pose_indices:
        print(f"  No tabletop poses found for {obj_name}")
        return None

    if save_dir is None:
        save_dir = os.path.join("outputs", "planning_success_rate", obj_name)
    os.makedirs(save_dir, exist_ok=True)
    partial_path = os.path.join(save_dir, f"plan_vs_ik_{grasp_version}_partial.json")

    # Build grid
    grid_points = []
    for pose_idx in pose_indices:
        for x_off in x_offsets:
            for z_deg in z_rotations_deg:
                grid_points.append({
                    "pose_idx": pose_idx,
                    "x_offset": x_off,
                    "z_rotation_deg": z_deg,
                })

    # Resume from partial results if available
    all_results = []
    done_keys = set()
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            partial_data = json.load(f)
        all_results = partial_data.get("results", [])
        for r in all_results:
            done_keys.add((r["pose_idx"], r["x_offset"], r["z_rotation_deg"]))
        print(f"  Resuming: {len(all_results)}/{len(grid_points)} points done")

    remaining = [pt for pt in grid_points
                 if (pt["pose_idx"], pt["x_offset"], pt["z_rotation_deg"]) not in done_keys]

    n_points = len(grid_points)
    print(f"Object: {obj_name}")
    print(f"Grasp version: {grasp_version}")
    print(f"Grid: {len(pose_indices)} poses × {len(x_offsets)} x_offsets × {len(z_rotations_deg)} z_rotations = {n_points} points")
    print(f"Remaining: {len(remaining)} points")
    print(f"Trials per point: {n_trials}")
    print("=" * 60)

    setup_timing = {}
    if not remaining:
        print("  All points already done")
    else:
        logging.getLogger("curobo").setLevel(logging.ERROR)
        t_ctor = time.perf_counter()
        planner = GraspPlanner(hand=hand)
        t_ctor = time.perf_counter() - t_ctor

        # Pay the one-time, reusable CUDA/solver setup here so the first grid
        # point measures steady-state planning instead of MotionGen creation +
        # CUDA graph capture + the first joint-space trajopt compile.
        warm_pt = remaining[0]
        warm_cfg = load_tabletop_scene(obj_name, warm_pt["pose_idx"],
                                       x_offset=warm_pt["x_offset"],
                                       z_rotation=np.radians(warm_pt["z_rotation_deg"]))
        warm_info = planner.warmup(warm_cfg, warmup_js_trajopt=True)
        setup_timing = {"planner_ctor_s": round(t_ctor, 3), **warm_info["breakdown"]}
        print("Setup (one-time, reusable):")
        for k, v in setup_timing.items():
            print(f"  {k:26s} {v:7.3f}s")
        print("=" * 60)

    # plan_single_js_s is the wall clock of the whole candidate loop; the js_*
    # keys break it into cuRobo's own graph/trajopt/finetune solver time plus
    # the remaining setup overhead.
    stage_keys = ["load_candidates_s", "world_setup_s", "filter_s",
                  "ik_s", "plan_single_js_s",
                  "js_graph_s", "js_trajopt_s", "js_finetune_s",
                  "js_overhead_s"]
    # Sub-stages of plan_single_js_s — excluded from total_per_point so the
    # per-point total is not double counted.
    substage_keys = {"js_graph_s", "js_trajopt_s", "js_finetune_s", "js_overhead_s"}

    n_plan_success = sum(1 for r in all_results if r["success_rate"] == 1.0)
    n_plan_fail = sum(1 for r in all_results if r["success_rate"] < 1.0)

    pbar = tqdm(remaining, desc=f"{obj_name}", unit="pt",
                initial=len(all_results), total=n_points)
    for pt in pbar:
        pose_idx = pt["pose_idx"]
        x_off = pt["x_offset"]
        z_deg = pt["z_rotation_deg"]
        z_rad = np.radians(z_deg)

        scene_cfg = load_tabletop_scene(obj_name, pose_idx,
                                        x_offset=x_off, z_rotation=z_rad)

        successes = 0
        trial_timings = []
        ik_counts = []
        filter_counts = []  # (n_total, n_backward, n_collision, n_valid)
        first_success_result = None

        for trial in range(n_trials):
            result = planner.plan(
                scene_cfg, obj_name=obj_name,
                grasp_version=grasp_version, mode="batch",
                seed=trial, hand=hand,
            )
            trial_timings.append(result.timing)
            ik_counts.append(result.timing.get("n_ik_success", 0))
            filter_counts.append({
                "n_total": result.timing.get("n_total", 0),
                "n_backward": result.timing.get("n_backward", 0),
                "n_collision": result.timing.get("n_collision", 0),
                "n_valid": result.timing.get("n_valid", 0),
            })
            if result.success:
                successes += 1
                if first_success_result is None:
                    first_success_result = result

        # Aggregate timing
        avg_timing = {}
        for key in stage_keys:
            vals = [t.get(key, 0) for t in trial_timings if t]
            if vals:
                avg_timing[key] = round(float(np.mean(vals)), 3)

        rate = successes / n_trials
        fc = filter_counts[0]  # same across seeds (deterministic filter)

        entry = {
            "pose_idx": pose_idx,
            "x_offset": x_off,
            "z_rotation_deg": z_deg,
            "success_count": successes,
            "n_trials": n_trials,
            "success_rate": rate,
            "n_total": fc["n_total"],
            "n_backward": fc["n_backward"],
            "n_collision": fc["n_collision"],
            "n_valid": fc["n_valid"],
            "ik_counts": ik_counts,
            "ik_mean": round(float(np.mean(ik_counts)), 1),
            "avg_timing": avg_timing,
            # Why plan_single_js rejected each candidate (GRAPH_FAIL /
            # TRAJOPT_FAIL / FINETUNE_TRAJOPT_FAIL / INVALID_QUERY counts).
            "js_fail_status": [t.get("js_fail_status") for t in trial_timings if t],
            # How many IK-reachable candidates plan_single_js had to try.
            "n_plan_attempts": [t.get("n_plan_attempts", 0) for t in trial_timings if t],
        }
        all_results.append(entry)

        # Save trajectory and grasp info for first successful trial
        if first_success_result is not None:
            traj_dir = os.path.join(save_dir, "trajectories",
                                    f"{pose_idx}_x{x_off:.2f}_z{z_deg:.0f}")
            os.makedirs(traj_dir, exist_ok=True)
            np.save(os.path.join(traj_dir, "traj.npy"), first_success_result.traj)
            np.save(os.path.join(traj_dir, "wrist_se3.npy"), first_success_result.wrist_se3)
            np.save(os.path.join(traj_dir, "pregrasp.npy"), first_success_result.pregrasp_pose)
            np.save(os.path.join(traj_dir, "grasp.npy"), first_success_result.grasp_pose)

        if rate == 1.0:
            n_plan_success += 1
        else:
            n_plan_fail += 1

        pbar.set_postfix(ok=n_plan_success, fail=n_plan_fail,
                         rate=f"{rate*100:.0f}%")

        # Save partial results periodically
        if len(all_results) % 10 == 0:
            with open(partial_path, "w") as f:
                json.dump({"results": all_results}, f, indent=2, default=str)

    # Summary
    print(f"\n{'=' * 60}")
    print("PLANNING SUCCESS RATE")
    print(f"{'=' * 60}")

    rates = [r["success_rate"] for r in all_results]
    always_ok = sum(1 for r in rates if r == 1.0)
    never_ok = sum(1 for r in rates if r == 0.0)
    partial = len(rates) - always_ok - never_ok

    print(f"Grid points tested: {len(all_results)}")
    print(f"  Always succeeds:  {always_ok}")
    print(f"  Sometimes fails:  {partial}")
    print(f"  Always fails:     {never_ok}")
    print(f"Overall mean rate: {np.mean(rates)*100:.1f}%")

    if never_ok + partial > 0:
        print(f"\nPlanning failures ({never_ok + partial} points):")
        failures = [r for r in all_results if r["success_rate"] < 1.0]
        for r in sorted(failures, key=lambda x: x["success_rate"]):
            print(f"  pose={r['pose_idx']} x={r['x_offset']:.2f} z={r['z_rotation_deg']:3.0f}°  "
                  f"plan={r['success_count']}/{r['n_trials']} ({r['success_rate']*100:.0f}%)")

    # Per-stage timing breakdown
    success_results = [r for r in all_results if r["success_rate"] > 0]
    fail_results = [r for r in all_results if r["success_rate"] == 0]

    def compute_timing_stats(results, stage_keys):
        stats = {}
        for key in stage_keys:
            vals = [r["avg_timing"].get(key, 0) for r in results if r["avg_timing"]]
            if vals:
                stats[key] = {
                    "mean": round(float(np.mean(vals)), 3),
                    "total": round(float(np.sum(vals)), 3),
                    "min": round(float(np.min(vals)), 3),
                    "max": round(float(np.max(vals)), 3),
                }
        top_keys = [k for k in stage_keys if k not in substage_keys]
        totals = []
        for r in results:
            if r["avg_timing"]:
                totals.append(sum(r["avg_timing"].get(k, 0) for k in top_keys))
        if totals:
            stats["total_per_point"] = {
                "mean": round(float(np.mean(totals)), 3),
                "min": round(float(np.min(totals)), 3),
                "max": round(float(np.max(totals)), 3),
                "total": round(float(np.sum(totals)), 3),
            }
        return stats

    timing_all = compute_timing_stats(all_results, stage_keys)
    timing_success = compute_timing_stats(success_results, stage_keys) if success_results else {}
    timing_fail = compute_timing_stats(fail_results, stage_keys) if fail_results else {}

    # Save
    summary = {
        "obj_name": obj_name,
        "grasp_version": grasp_version,
        "n_trials": n_trials,
        "n_grid_points": len(all_results),
        "n_plan_always_ok": always_ok,
        "n_plan_partial": partial,
        "n_plan_always_fail": never_ok,
        "overall_mean_rate": round(float(np.mean(rates)), 3),
        "setup_timing": setup_timing,
        "timing": {
            "all": timing_all,
            "success": timing_success,
            "fail": timing_fail,
        },
        "results": all_results,
    }

    out_path = os.path.join(save_dir, f"plan_vs_ik_{grasp_version}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    # Clean up partial file
    if os.path.exists(partial_path):
        os.remove(partial_path)
    print(f"\nResults saved to: {out_path}")

    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Planning success rate grid search")
    parser.add_argument("--obj", type=str, default=None, help="Object name (omit for all)")
    parser.add_argument("--version", type=str, required=True, help="Grasp candidate version")
    parser.add_argument("--n_trials", type=int, default=5, help="Trials per point")
    parser.add_argument("--x_min", type=float, default=0.2, help="Min x offset")
    parser.add_argument("--x_max", type=float, default=0.5, help="Max x offset")
    parser.add_argument("--x_step", type=float, default=0.05, help="X offset step")
    parser.add_argument("--z_step", type=float, default=30, help="Z rotation step (degrees)")
    parser.add_argument("--save_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--hand", type=str, default="allegro", choices=["allegro", "inspire", "inspire_left", "fr3_inspire"])
    args = parser.parse_args()

    x_offsets = np.arange(args.x_min, args.x_max + 1e-6, args.x_step).round(3).tolist()
    z_rotations_deg = np.arange(0, 360, args.z_step).tolist()

    if args.obj is not None:
        objects = [args.obj]
    else:
        objects = get_all_objects(args.version, hand=args.hand)
        print(f"Found {len(objects)} objects with tabletop poses + candidates\n")

    all_summaries = {}
    for obj_name in objects:
        print(f"\n{'#' * 60}")
        print(f"# {obj_name}")
        print(f"{'#' * 60}")

        # Skip if result already exists
        save_dir = args.save_dir or os.path.join("outputs", "planning_success_rate", obj_name)
        out_path = os.path.join(save_dir, f"plan_vs_ik_{args.version}.json")
        if os.path.exists(out_path):
            print(f"  Result exists, skipping: {out_path}")
            with open(out_path) as f:
                all_summaries[obj_name] = json.load(f)
            continue

        try:
            summary = run_planning_grid(
                obj_name=obj_name,
                grasp_version=args.version,
                n_trials=args.n_trials,
                x_offsets=x_offsets,
                z_rotations_deg=z_rotations_deg,
                save_dir=args.save_dir,
                hand=args.hand,
            )
            if summary:
                all_summaries[obj_name] = summary
        except Exception as e:
            print(f"  SKIPPED: {e}")
            import traceback
            traceback.print_exc()
            all_summaries[obj_name] = {"error": str(e)}

    if len(all_summaries) > 1:
        print(f"\n{'=' * 60}")
        print("ALL OBJECTS: PLANNING SUCCESS RATE")
        print(f"{'=' * 60}")
        for obj_name, s in all_summaries.items():
            if "error" in s:
                print(f"  {obj_name}: ERROR - {s['error']}")
            else:
                print(f"  {obj_name}: grid={s['n_grid_points']}  "
                      f"ok={s['n_plan_always_ok']}  "
                      f"fail={s['n_plan_always_fail']}  "
                      f"partial={s['n_plan_partial']}  "
                      f"rate={s['overall_mean_rate']*100:.1f}%")