#!/usr/bin/env python3
"""Prepare a fixed *unseen-to-the-robot* catalogue for Franka + Inspire.

This is deliberately a staged tool.  It never moves the robot unless both
``--execute`` and ``--run-robot`` are supplied.  The normal workflow is:

1. Put a scanned mesh plus object-processing outputs under ``--object-root``.
2. Run ``--stage assets --execute`` (FoundPose representation + GoTrack bank).
3. Run ``--stage candidates --execute`` (tabletop simulated candidates).
4. Run the printed collection command, which uses ``run_auto.py`` to record
   arm-specific physical successes in the same NAS candidate pool.
5. Check ``--stage audit`` until preflight is ready, then use the printed
   continuous-demo command.

"Unseen" here means unseen in real Franka grasp data, not mesh-free open-set
grasping: a mesh and object-processing outputs are still required for safe
planning and 6D pose estimation.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.demo.continuous_basket.catalog import CatalogObject, parse_catalog
from src.demo.continuous_basket.basket_marker import DEFAULT_BASKET_MARKER_ID
from src.demo.continuous_basket.preflight import (
    DEFAULT_ANCHOR_ROOT,
    DEFAULT_ASSETS_BASE,
    build_report,
)


DEFAULT_OBJECT_ROOT = Path.home() / "shared_data/object_processing"
DEFAULT_BODEX_PYTHON = Path.home() / "anaconda3/envs/bodex/bin/python"
DEFAULT_GOTRACK_PYTHON = Path.home() / "anaconda3/envs/gotrack_cu128/bin/python"
DEFAULT_PLANNER_PYTHON = Path.home() / "anaconda3/envs/planner/bin/python"
DEFAULT_SESSION_ROOT = Path.home() / "shared_data/AutoDex/experiment/franka_catalog_onboarding"
DEFAULT_CAM_PARAM_ROOT = Path.home() / "shared_data/cam_param"


@dataclass(frozen=True)
class ProcessingReadiness:
    name: str
    raw_mesh: str
    simplified_mesh: str
    collision_urdf: str
    obb: str
    tabletop_pose_count: int
    ready: bool
    missing: list[str]


def processing_readiness(item: CatalogObject, object_root: Path) -> ProcessingReadiness:
    """Check the immutable scan-processing products before generating anything."""
    base = Path(object_root) / item.name
    raw = base / "raw_mesh" / f"{item.name}.obj"
    simplified = base / "processed_data/mesh/simplified.obj"
    collision = base / "processed_data/urdf/coacd.urdf"
    obb = base / "processed_data/info/simplified.json"
    tabletop = sorted((base / "processed_data/info/tabletop").glob("*.npy"))
    missing = []
    for label, path in (("raw_mesh", raw), ("simplified_mesh", simplified),
                        ("collision_urdf", collision), ("obb", obb)):
        if not path.is_file():
            missing.append(label)
    if not tabletop:
        missing.append("tabletop_poses")
    return ProcessingReadiness(
        name=item.name, raw_mesh=str(raw), simplified_mesh=str(simplified),
        collision_urdf=str(collision), obb=str(obb),
        tabletop_pose_count=len(tabletop), ready=not missing, missing=missing,
    )


def _quat_wxyz(rotation: np.ndarray) -> list[float]:
    """Convert a 3x3 rotation matrix to the w,x,y,z convention used by scenes."""
    r = np.asarray(rotation, dtype=float).reshape(3, 3)
    trace = float(np.trace(r))
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        q = np.array([0.25 * s, (r[2, 1] - r[1, 2]) / s,
                      (r[0, 2] - r[2, 0]) / s, (r[1, 0] - r[0, 1]) / s])
    else:
        idx = int(np.argmax(np.diag(r)))
        if idx == 0:
            s = 2.0 * np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2])
            q = np.array([(r[2, 1] - r[1, 2]) / s, 0.25 * s,
                          (r[0, 1] + r[1, 0]) / s, (r[0, 2] + r[2, 0]) / s])
        elif idx == 1:
            s = 2.0 * np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2])
            q = np.array([(r[0, 2] - r[2, 0]) / s, (r[0, 1] + r[1, 0]) / s,
                          0.25 * s, (r[1, 2] + r[2, 1]) / s])
        else:
            s = 2.0 * np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1])
            q = np.array([(r[1, 0] - r[0, 1]) / s, (r[0, 2] + r[2, 0]) / s,
                          (r[1, 2] + r[2, 1]) / s, 0.25 * s])
    q /= np.linalg.norm(q)
    return [float(v) for v in q]


def table_scene_payload(item: CatalogObject, pose: np.ndarray, object_root: Path) -> dict:
    """Build one table scene without relying on generate_scene's legacy root."""
    base = Path(object_root) / item.name / "processed_data"
    return {
        "scene": {
            "mesh": {"target": {
                "scale": [1.0, 1.0, 1.0],
                "pose": [*np.asarray(pose, dtype=float)[:3, 3].tolist(),
                         *_quat_wxyz(np.asarray(pose, dtype=float)[:3, :3])],
                "file_path": str(base / "mesh/simplified.obj"),
                "urdf_path": str(base / "urdf/coacd.urdf"),
            }},
            "cuboid": {"table": {
                "dims": [2.0, 2.0, 0.2],
                "pose": [0.0, 0.0, -0.1, 1.0, 0.0, 0.0, 0.0],
            }},
        },
    }


def write_table_scenes(items: Sequence[CatalogObject], *, object_root: Path,
                       scene_root: Path, overwrite: bool) -> dict[str, list[str]]:
    """Create stable table scene IDs; refuse mismatched existing definitions."""
    ids: dict[str, list[str]] = {}
    for item in items:
        pose_paths = sorted((object_root / item.name / "processed_data/info/tabletop").glob("*.npy"))
        out_dir = scene_root / "inspire" / item.name / "table"
        out_dir.mkdir(parents=True, exist_ok=True)
        ids[item.name] = []
        for index, pose_path in enumerate(pose_paths):
            scene_id = str(index)
            destination = out_dir / f"{scene_id}.json"
            payload = table_scene_payload(item, np.load(pose_path), object_root)
            payload["meta"] = {"pose_idx": pose_path.stem, "param": {}}
            if destination.exists() and not overwrite:
                existing = json.loads(destination.read_text())
                if existing != payload:
                    raise FileExistsError(
                        f"refusing to replace a different table scene: {destination}; "
                        "pass --overwrite-scenes after checking its candidate pool"
                    )
            else:
                destination.write_text(json.dumps(payload, indent=2) + "\n")
            ids[item.name].append(scene_id)
    return ids


def _reference_camera(intrinsics_json: Path, requested: str | None) -> str:
    if requested:
        return requested
    data = json.loads(intrinsics_json.read_text())
    if not data:
        raise ValueError(f"no camera intrinsics in {intrinsics_json}")
    return sorted(data)[0]


def _latest_intrinsics(cam_param_root: Path) -> Path:
    candidates = sorted(cam_param_root.glob("*/intrinsics.json"))
    if not candidates:
        raise FileNotFoundError(f"no calibration intrinsics under {cam_param_root}")
    return candidates[-1]


def _run(command: Sequence[str], *, execute: bool) -> None:
    printable = " ".join(str(arg) for arg in command)
    print(f"$ {printable}")
    if execute:
        subprocess.run([str(arg) for arg in command], check=True)


def _asset_commands(items: Sequence[CatalogObject], args, reference_intrinsics: Path,
                    reference_camera: str) -> tuple[list[str], list[list[str]]]:
    names = [item.name for item in items]
    onboard = [
        args.gotrack_python, str(REPO_ROOT / "src/process/batch_onboard_foundpose.py"),
        "--objects", *names, "--object-root", args.object_root,
        "--output-root", args.assets_base, "--reference-intrinsics-json", str(reference_intrinsics),
        "--reference-camera-id", reference_camera, "--workers", str(args.onboard_workers),
        "--python-bin", args.gotrack_python,
    ]
    commands = [onboard]
    for item in items:
        commands.append([
            args.gotrack_python,
            str(REPO_ROOT / "autodex/perception/thirdparty/MV-GoTrack/scripts/generate_anchor_bank.py"),
            "--mesh-path", str(Path(args.object_root) / item.name / "raw_mesh" / f"{item.name}.obj"),
            "--output-path", str(Path(args.anchor_root) / f"{item.name}.npz"),
            "--num-anchors", str(args.num_anchors),
        ])
    return names, commands


def _candidate_commands(items: Sequence[CatalogObject], args, session_dir: Path) -> list[list[str]]:
    return [[
        sys.executable, str(REPO_ROOT / "src/demo/continuous_basket/generate_table_candidates.py"),
        "--objects", *[item.name for item in items], "--object-root", args.object_root,
        "--version", args.version, "--parallel", str(args.bodex_parallel),
        "--seed-num", str(args.seed_num), "--bodex-python", args.bodex_python,
        "--sim-python", args.sim_python, "--session-dir", str(session_dir / "candidates"),
        "--execute",
    ]]


def _collection_commands(items: Iterable[CatalogObject], args) -> list[list[str]]:
    commands = []
    for item in items:
        cmd = [
            args.planner_python, str(REPO_ROOT / "src/execution/run_auto.py"),
            "--obj", item.name, "--grasp_version", args.version,
            "--hand", "inspire", "--arm", "franka", "--scene", "table",
            "--candidate-scene-type", "table", "--max_trials", str(args.collection_trials),
        ]
        if args.auto_label:
            cmd.append("--auto")
        commands.append(cmd)
    return commands


def _demo_command(items: Sequence[CatalogObject], args) -> list[str]:
    command = [
        args.planner_python, str(REPO_ROOT / "src/demo/continuous_basket/run_demo.py"),
        "--objects", *[f"{item.name}={item.prompt}" for item in items],
        "--hand", "inspire", "--arm", "franka", "--grasp-version", args.version,
        "--max-successes", str(args.max_successes),
    ]
    if args.basket_marker_id is not None:
        return command + [
            "--basket-marker-id", str(args.basket_marker_id),
            "--basket-marker-dict", args.basket_marker_dict,
            "--basket-marker-offset", *[str(v) for v in args.basket_marker_offset],
        ]
    basket = list(args.basket_center) if args.basket_center else ["<X>", "<Y>", "<Z>"]
    return command + ["--basket-center", *[str(v) for v in basket]]


def _franka_successful_tabletops(item: CatalogObject, *, candidate_root: Path,
                                 scene_root: Path) -> set[str]:
    """Return stable-pose stems with a real Franka success, if recoverable.

    Older v8 candidates use the coverage map; newly generated table candidates
    have no coverage file yet, so their hand-specific table-scene metadata is
    the authoritative fallback.
    """
    try:
        coverage_path = (Path.home() / "shared_data/AutoDex/experiment/v8/coverage"
                         / f"cov_v8_cand_{item.name}.json")
        coverage_rows = json.loads(coverage_path.read_text()).get("grasps", [])
    except (OSError, ValueError, json.JSONDecodeError):
        coverage_rows = []
    coverage = {
        (str(row.get("type")), str(row.get("sid")), str(row.get("gid"))): str(row.get("pose_idx"))
        for row in coverage_rows if row.get("pose_idx") is not None
    }
    object_root = candidate_root / item.name
    stems: set[str] = set()
    for wrist in object_root.glob("**/wrist_se3.npy"):
        result_path = wrist.parent / "result.json"
        try:
            result = json.loads(result_path.read_text())
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        if result.get("success") is not True or str(result.get("arm", "xarm")) != "franka":
            continue
        try:
            scene_type, scene_id, grasp_id = wrist.relative_to(object_root).parts[:3]
        except ValueError:
            continue
        stem = coverage.get((scene_type, scene_id, grasp_id))
        if stem is None:
            scene_path = scene_root / "inspire" / item.name / scene_type / f"{scene_id}.json"
            try:
                stem = str(json.loads(scene_path.read_text())["meta"]["pose_idx"])
            except (KeyError, OSError, ValueError, json.JSONDecodeError):
                pass
        if stem:
            stems.add(stem)
    return stems


def _print_readiness(items: Sequence[CatalogObject], args) -> bool:
    processing = [processing_readiness(item, Path(args.object_root)) for item in items]
    runtime = build_report(
        items, object_root=Path(args.object_root), assets_base=Path(args.assets_base),
        candidate_root=Path.home() / "shared_data/AutoDex/candidates/inspire" / args.version,
        anchor_root=Path(args.anchor_root), require_gotrack=True, arm="franka",
    )
    by_name = {row.name: row for row in runtime}
    candidate_root = Path.home() / "shared_data/AutoDex/candidates/inspire" / args.version
    scene_root = Path.home() / "shared_data/AutoDex/scene"
    print(f"{'object':<24} {'tabletop':>8} {'candidate':>9} {'FR3 ok':>7} {'FR3 pose':>8}  status")
    ready = True
    for row in processing:
        runtime_row = by_name[row.name]
        missing = [*row.missing, *runtime_row.missing]
        pose_count = len(_franka_successful_tabletops(
            CatalogObject(row.name, row.name), candidate_root=candidate_root, scene_root=scene_root,
        ))
        if runtime_row.successful_candidate_count and pose_count < args.min_franka_poses:
            missing.append(f"successful_tabletop_poses<{args.min_franka_poses}")
        status = "READY" if not missing else "MISSING: " + ", ".join(dict.fromkeys(missing))
        print(f"{row.name:<24} {row.tabletop_pose_count:>8} {runtime_row.candidate_count:>9} "
              f"{runtime_row.successful_candidate_count:>7} {pose_count:>8}  {status}")
        ready &= not missing
    return ready


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", nargs="+", required=True,
                        help="fixed catalogue names, optionally name=YOLO-E prompt")
    parser.add_argument("--stage", choices=["audit", "assets", "candidates", "collect", "demo", "all"],
                        default="audit")
    parser.add_argument("--execute", action="store_true",
                        help="run non-motion asset/candidate work; otherwise print commands only")
    parser.add_argument("--run-robot", action="store_true",
                        help="also run a collection/demo command; requires --execute")
    parser.add_argument("--object-root", default=str(DEFAULT_OBJECT_ROOT))
    parser.add_argument("--assets-base", default=str(DEFAULT_ASSETS_BASE))
    parser.add_argument("--anchor-root", default=str(DEFAULT_ANCHOR_ROOT))
    parser.add_argument("--version", default="v8")
    parser.add_argument("--reference-intrinsics-json", default=None)
    parser.add_argument("--reference-camera-id", default=None)
    parser.add_argument("--cam-param-root", default=str(DEFAULT_CAM_PARAM_ROOT))
    parser.add_argument("--gotrack-python", default=str(DEFAULT_GOTRACK_PYTHON))
    parser.add_argument("--bodex-python", default=str(DEFAULT_BODEX_PYTHON))
    parser.add_argument("--sim-python", default=str(DEFAULT_PLANNER_PYTHON))
    parser.add_argument("--planner-python", default=str(DEFAULT_PLANNER_PYTHON))
    parser.add_argument("--onboard-workers", type=int, default=1)
    parser.add_argument("--num-anchors", type=int, default=256)
    parser.add_argument("--bodex-parallel", type=int, default=4)
    parser.add_argument("--seed-num", type=int, default=200)
    parser.add_argument("--collection-trials", type=int, default=8)
    parser.add_argument("--min-franka-poses", type=int, default=1,
                        help="require this many stable tabletop poses with Franka successes during audit")
    parser.add_argument("--auto-label", action="store_true",
                        help="use Charuco lift verification during physical collection")
    basket_source = parser.add_mutually_exclusive_group()
    basket_source.add_argument("--basket-center", type=float, nargs=3, default=None)
    basket_source.add_argument("--basket-marker-id", type=int, default=DEFAULT_BASKET_MARKER_ID,
                               help=f"standalone ArUco ID fixed to the basket (default: {DEFAULT_BASKET_MARKER_ID})")
    parser.add_argument("--basket-marker-dict", default="6X6_1000")
    parser.add_argument("--basket-marker-offset", type=float, nargs=3,
                        default=[0.0, 0.0, 0.0], metavar=("DX", "DY", "DZ"),
                        help="marker-local metres from marker centre to release reference")
    parser.add_argument("--max-successes", type=int, default=12)
    parser.add_argument("--session-dir", default=None)
    args = parser.parse_args()
    if args.run_robot and not args.execute:
        parser.error("--run-robot requires --execute")
    if min(args.onboard_workers, args.num_anchors, args.bodex_parallel, args.seed_num,
           args.collection_trials, args.max_successes, args.min_franka_poses) < 1:
        parser.error("worker/count arguments must be positive")

    items = parse_catalog(args.objects)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = Path(args.session_dir).expanduser() if args.session_dir else DEFAULT_SESSION_ROOT / stamp
    print("[audit]")
    processing_ready = _print_readiness(items, args)

    if args.stage == "audit":
        raise SystemExit(0 if processing_ready else 2)
    missing_processing = [row.name for row in (processing_readiness(item, Path(args.object_root)) for item in items)
                          if not row.ready]
    if missing_processing:
        raise SystemExit("cannot onboard incomplete object-processing assets: " + ", ".join(missing_processing))

    ref = (Path(args.reference_intrinsics_json).expanduser() if args.reference_intrinsics_json
           else _latest_intrinsics(Path(args.cam_param_root).expanduser()))
    if not ref.is_file():
        raise SystemExit(f"reference intrinsics missing: {ref}")
    ref_camera = _reference_camera(ref, args.reference_camera_id)
    manifest = {
        "objects": [item.__dict__ for item in items], "arm": "franka", "hand": "inspire",
        "version": args.version, "object_root": str(Path(args.object_root).expanduser()),
        "reference_intrinsics": str(ref), "reference_camera": ref_camera,
    }
    if args.execute:
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    if args.stage in ("assets", "all"):
        print("[assets]")
        _names, commands = _asset_commands(items, args, ref, ref_camera)
        for command in commands:
            _run(command, execute=args.execute)

    if args.stage in ("candidates", "all"):
        print("[candidates]")
        for command in _candidate_commands(items, args, session_dir):
            _run(command, execute=args.execute)

    if args.stage == "collect":
        print("[collect] Place one object at a time on the labelled table. Re-run an object in distinct stable poses.")
        for command in _collection_commands(items, args):
            _run(command, execute=args.execute and args.run_robot)

    if args.stage == "demo":
        print("[demo] This command is blocked until all objects pass the Franka preflight and basket coordinates are known.")
        command = _demo_command(items, args)
        if args.basket_center is not None or args.basket_marker_id is not None:
            _run(command, execute=args.execute and args.run_robot)
        else:
            print("$ " + " ".join(command))

    if args.stage == "all":
        print("[next: collection]")
        for command in _collection_commands(items, args):
            print("$ " + " ".join(command))
        print("[next: continuous demo]")
        print("$ " + " ".join(_demo_command(items, args)))


if __name__ == "__main__":
    main()
