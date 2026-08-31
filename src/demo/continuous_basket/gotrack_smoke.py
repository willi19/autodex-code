#!/usr/bin/env python3
"""Exercise distributed GoTrack without creating any robot motion.

Use a FoundPose world pose captured by a failed/previous demo run.  The script
sends only GoTrack ``init`` / ``start`` / ``stop`` commands to capture PCs,
publishes the initial prior pose, and waits for a fused observation.  It never
constructs a Franka or Inspire executor and never contacts the FCI daemon.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))
for _paradex_root in (os.environ.get("AUTODEX_PARADEX_ROOT"),
                      str(Path.home() / "paradex")):
    if _paradex_root and (Path(_paradex_root).expanduser() / "paradex").is_dir():
        sys.path.insert(0, str(Path(_paradex_root).expanduser()))
        break

from paradex.utils.system import get_camera_list, get_pc_ip

from autodex.utils.path import get_obj_root
from src.demo.continuous_basket.tracking import LiveGoTrackSession


DEFAULT_PC_LIST = ["capture1", "capture2", "capture3", "capture5", "capture6"]
CAM_PARAM_ROOT = Path.home() / "shared_data/cam_param"


def _load_calib(calib_dir: Path):
    """Read only the fields GoTrack needs without importing cuRobo/planning."""
    intr_raw = json.loads((calib_dir / "intrinsics.json").read_text())
    extr_raw = json.loads((calib_dir / "extrinsics.json").read_text())
    intrinsics = {
        serial: {
            "K_orig": np.asarray(data["original_intrinsics"], dtype=np.float64).reshape(3, 3),
            "K_undist": np.asarray(data["intrinsics_undistort"], dtype=np.float64).reshape(3, 3),
            "dist_params": np.asarray(data["dist_params"], dtype=np.float64).reshape(-1),
            "width": int(data["width"]), "height": int(data["height"]),
        }
        for serial, data in intr_raw.items()
    }
    extrinsics = {}
    for serial, raw in extr_raw.items():
        matrix = np.asarray(raw, dtype=np.float64).reshape(-1)
        extrinsics[serial] = (np.vstack([matrix.reshape(3, 4), [0, 0, 0, 1]])
                              if matrix.size == 12 else matrix.reshape(4, 4))
    return intrinsics, extrinsics


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object", required=True, dest="obj_name")
    parser.add_argument("--init-pose", required=True,
                        help="FoundPose pose_world.npy from a prior no-motion capture")
    parser.add_argument("--grasp-version", default="v8")
    parser.add_argument("--pc-list", nargs="+", default=DEFAULT_PC_LIST)
    parser.add_argument("--anchor-root", default=str(
        REPO_ROOT / "autodex/perception/thirdparty/MV-GoTrack/anchor_banks"))
    parser.add_argument("--warmup-s", type=float, default=15.0)
    parser.add_argument("--object-switch-settle-s", type=float, default=2.0)
    parser.add_argument("--command-timeout-s", type=float, default=3.0)
    parser.add_argument("--min-cams", type=int, default=6)
    parser.add_argument("--min-inliers", type=int, default=12)
    args = parser.parse_args()
    if min(args.warmup_s, args.object_switch_settle_s, args.command_timeout_s) <= 0:
        parser.error("timing arguments must be positive")

    initial = np.asarray(np.load(Path(args.init_pose).expanduser()), dtype=np.float64)
    if initial.shape != (4, 4) or not np.isfinite(initial).all():
        parser.error("--init-pose must contain a finite 4x4 pose_world matrix")
    mesh = (Path(get_obj_root(args.grasp_version)) / args.obj_name / "raw_mesh"
            / f"{args.obj_name}.obj")
    if not mesh.is_file():
        parser.error(f"mesh missing: {mesh}")

    calib_dirs = sorted(Path(CAM_PARAM_ROOT).iterdir())
    if not calib_dirs:
        parser.error(f"no camera calibration under {CAM_PARAM_ROOT}")
    intrinsics, extrinsics = _load_calib(calib_dirs[-1])
    active = {serial for pc in args.pc_list for serial in get_camera_list(pc)}
    intrinsics = {serial: value for serial, value in intrinsics.items() if serial in active}
    extrinsics = {serial: value for serial, value in extrinsics.items() if serial in active}
    if len(intrinsics) < args.min_cams or len(extrinsics) < args.min_cams:
        parser.error("latest calibration does not cover enough selected cameras")

    session = LiveGoTrackSession(
        pc_list=args.pc_list, capture_ips=[get_pc_ip(pc) for pc in args.pc_list],
        intrinsics=intrinsics, extrinsics=extrinsics,
        anchor_root=Path(args.anchor_root).expanduser(), min_cams_per_frame=args.min_cams,
        min_inliers=args.min_inliers,
        command_timeout_ms=round(args.command_timeout_s * 1000), command_retries=1,
    )
    try:
        session.start(obj_name=args.obj_name, mesh_path=mesh, init_pose_world=initial,
                      settle_s=args.object_switch_settle_s)
        sample = session.wait_for_pose(timeout_s=args.warmup_s)
        diagnostics = session.diagnostics()
        if sample is None:
            print("GOTRACK_SMOKE_FAILED " + json.dumps(_jsonable(diagnostics), sort_keys=True))
            raise SystemExit(2)
        print("GOTRACK_SMOKE_OK " + json.dumps(_jsonable({
            "frame_id": sample.frame_id,
            "n_inliers": sample.n_inliers,
            "mean_residual_mm": sample.mean_residual_mm,
            "pose_world": sample.pose_world,
            "diagnostics": diagnostics,
        }), sort_keys=True))
    finally:
        session.stop()


if __name__ == "__main__":
    main()
