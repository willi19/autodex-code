#!/usr/bin/env python3
"""Onboard multiple object meshes for FoundPose Stage A in parallel.

Each object's onboarding takes ~15-20 min (798 viewpoint renders + DINOv2
feature extraction + PCA + clustering). When you need to onboard N objects
the serial wall time is N × 18 min — too long.

This script launches K of those onboardings concurrently as subprocesses.
GPU is shared (DINOv2 backbone), so don't go too parallel: 3-4 workers is
typically the sweet spot on a single 24 GB GPU.

Usage:
    # Onboard 20 objects with 4 parallel workers
    python src/process/batch_onboard_foundpose.py \\
        --objects attached_container banana wood_organizer ... \\
        --output-root outputs/foundpose_assets \\
        --workers 4

    # Or pass a file with one object per line
    python src/process/batch_onboard_foundpose.py \\
        --objects-file my_objects.txt --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List

REPO_ROOT = Path(__file__).resolve().parents[2]
GOTRACK_ROOT = REPO_ROOT / "autodex/perception/thirdparty/MV-GoTrack"
MESH_BASE = Path.home() / "shared_data/object_processing"


def _resolve_mesh(obj: str, mesh_base: Path) -> Path:
    for sub in [
        mesh_base / obj / "raw_mesh" / f"{obj}.obj",
        mesh_base / obj / "processed_data" / "mesh" / "raw.obj",
        mesh_base / obj / "processed_data" / "mesh" / "simplified.obj",
    ]:
        if sub.exists():
            return sub
    raise FileNotFoundError(f"No mesh for {obj} under {mesh_base}")


def _resolve_reference_camera(reference_intrinsics_json: Path) -> str:
    """Pick a stable reference camera id (lowest serial)."""
    with open(reference_intrinsics_json) as f:
        return sorted(json.load(f).keys())[0]


def _onboard_one(
    obj: str,
    mesh_path: Path,
    output_root: Path,
    reference_intrinsics_json: Path,
    reference_camera_id: str,
    python_bin: str,
    log_dir: Path,
    overwrite: bool,
) -> dict:
    asset_dir = output_root / obj
    repre = asset_dir / "object_repre" / "v1" / obj / "1" / "repre.pth"
    if repre.exists() and not overwrite:
        return {"obj": obj, "status": "cached", "elapsed_s": 0.0}

    cmd = [
        python_bin,
        str(GOTRACK_ROOT / "scripts/onboard_custom_mesh_for_foundpose.py"),
        "--mesh-path", str(mesh_path),
        "--object-id", "1",
        "--dataset-name", obj,
        "--output-root", str(asset_dir),
        "--reference-intrinsics-json", str(reference_intrinsics_json),
        "--reference-camera-id", reference_camera_id,
        "--reference-image-scale", "1.0",
        "--mesh-scale", "1000.0",
        "--min-num-viewpoints", "57",
        "--num-inplane-rotations", "14",
        "--ssaa-factor", "4.0",
        "--pca-components", "256",
        "--cluster-num", "2048",
    ]
    if overwrite or asset_dir.exists():
        cmd.append("--overwrite")

    env = os.environ.copy()
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("EGL_PLATFORM", "surfaceless")

    log_path = log_dir / f"{obj}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    with open(log_path, "w") as logf:
        proc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    elapsed = time.perf_counter() - t0
    return {
        "obj": obj,
        "status": "ok" if proc.returncode == 0 else "failed",
        "rc": proc.returncode,
        "elapsed_s": elapsed,
        "log": str(log_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--objects", type=str, nargs="*", default=[],
                        help="Object names, e.g. attached_container banana ...")
    parser.add_argument("--objects-file", type=str, default=None,
                        help="One object name per line.")
    parser.add_argument("--output-root", type=str,
                        default=str(REPO_ROOT / "outputs/foundpose_assets"))
    parser.add_argument("--object-root", type=str, default=str(MESH_BASE),
                        help="Object tree containing raw_mesh and processed_data (v8 defaults to object_processing).")
    parser.add_argument("--reference-intrinsics-json", type=str, default=None,
                        help="Calibration json (intrinsics) used during onboarding. "
                             "Defaults to the first usable cam_param/intrinsics.json "
                             "we can find under experiment/selected_100/allegro/.")
    parser.add_argument("--reference-camera-id", type=str, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--python-bin", type=str,
                        default=str(Path.home() / "anaconda3/envs/gotrack_cu128/bin/python"))
    parser.add_argument("--log-dir", type=str,
                        default=str(REPO_ROOT / "outputs/foundpose_onboard_logs"))
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    objects: List[str] = list(args.objects)
    if args.objects_file:
        with open(args.objects_file) as f:
            objects.extend(line.strip() for line in f if line.strip() and not line.startswith("#"))
    if not objects:
        parser.error("No objects provided (--objects or --objects-file).")

    # Resolve reference intrinsics if not given.
    ref_json = Path(args.reference_intrinsics_json) if args.reference_intrinsics_json else None
    if ref_json is None:
        exp_root = Path.home() / "shared_data/AutoDex/experiment/selected_100/allegro"
        for obj_dir in sorted(exp_root.iterdir()):
            if not obj_dir.is_dir():
                continue
            for ep in sorted(obj_dir.iterdir()):
                cand = ep / "cam_param" / "intrinsics.json"
                if cand.exists():
                    ref_json = cand
                    break
            if ref_json:
                break
    if ref_json is None or not ref_json.exists():
        parser.error("Could not find a reference intrinsics.json; pass --reference-intrinsics-json.")
    ref_cam = args.reference_camera_id or _resolve_reference_camera(ref_json)

    output_root = Path(args.output_root).resolve()
    log_dir = Path(args.log_dir).resolve()
    object_root = Path(args.object_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[batch_onboard] {len(objects)} objects, {args.workers} workers")
    print(f"[batch_onboard] ref_intrinsics={ref_json}  ref_cam={ref_cam}")
    print(f"[batch_onboard] output_root={output_root}")
    print(f"[batch_onboard] log_dir={log_dir}")

    # Resolve meshes up front; skip missing.
    jobs = []
    for obj in objects:
        try:
            mesh = _resolve_mesh(obj, object_root)
        except FileNotFoundError as e:
            print(f"[skip] {obj}: {e}")
            continue
        jobs.append((obj, mesh))
    print(f"[batch_onboard] {len(jobs)} jobs to run")

    t0 = time.perf_counter()
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(
                _onboard_one,
                obj=obj,
                mesh_path=mesh,
                output_root=output_root,
                reference_intrinsics_json=ref_json,
                reference_camera_id=ref_cam,
                python_bin=args.python_bin,
                log_dir=log_dir,
                overwrite=args.overwrite,
            ): obj
            for obj, mesh in jobs
        }
        n_done = 0
        for fut in as_completed(futures):
            obj = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:
                res = {"obj": obj, "status": "error", "error": str(exc)}
            results.append(res)
            n_done += 1
            print(f"[{n_done}/{len(jobs)}] {obj}: {res['status']}  "
                  f"({res.get('elapsed_s', 0):.1f}s)")

    total = time.perf_counter() - t0
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_cache = sum(1 for r in results if r["status"] == "cached")
    n_fail = sum(1 for r in results if r["status"] not in ("ok", "cached"))
    print(f"\n[batch_onboard] done: {n_ok} ok, {n_cache} cached, {n_fail} failed "
          f"in {total:.1f}s wall ({total / max(len(jobs), 1):.1f}s/obj avg)")

    summary_path = log_dir / "_summary.json"
    with open(summary_path, "w") as f:
        json.dump({"results": results, "wall_s": total}, f, indent=2)
    print(f"[batch_onboard] summary -> {summary_path}")


if __name__ == "__main__":
    main()
