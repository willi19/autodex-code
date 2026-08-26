#!/usr/bin/env python3
"""Locate the place target for the pick-and-place demo from an ArUco marker.

The demo's drop-off spot is marked by a single standalone ArUco marker (the
one sitting on the cutting board, NOT part of the charuco board). This module
captures one multi-camera image set, triangulates that marker's 4 corners, and
returns its center + orientation in the ROBOT frame -- that center is the
target the object has to be placed on.

The marker is identified as "the aruco id that belongs to no configured
charuco board": every board marker (ids 70..104 of the 6X6 family here) shows
up in several dictionaries at once, so filtering by id against
``boardinfo_dict`` is what separates the lone marker from the board.

    # from an already-captured image dir
    python src/execution/place_target.py \
        --capture_dir ~/shared_data/mingi_erasethis/20260826_182132

    # capture a fresh image set first
    python src/execution/place_target.py --capture

Note the 4X4_* detections that OpenCV reports on this scene are false decodes
of the board markers (they land on the same pixels) -- scanning only
``--dict 6X6_1000`` avoids them.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from paradex.calibration.utils import (load_current_C2R, save_current_C2R,
                                       save_current_camparam)
from paradex.image.aruco import boardinfo_dict
from paradex.image.image_dict import ImageDict

DEFAULT_PC_LIST = ["capture1", "capture2", "capture3", "capture5", "capture6"]
CAPTURE_ROOT_REL = "shared_data/mingi_erasethis"   # relative to $HOME, like run_auto
MIN_VIEWS = 3


def board_marker_ids() -> set:
    """Every aruco id claimed by a configured charuco board."""
    ids = set()
    for cfg in boardinfo_dict.values():
        ids.update(int(i) for i in cfg["markerIDs"])
    return ids


def marker_frame(corners_3d: np.ndarray) -> np.ndarray:
    """4x4 pose of a square marker from its 4 triangulated corners.

    Corner order out of OpenCV is CW starting top-left in image space, so
    c0->c3 and c1->c2 are two parallel edges; averaging them gives the marker
    x axis. z is the plane normal, fixed to point "up" (+z of the input frame)
    because the marker lies flat on the table.
    """
    c = np.asarray(corners_3d, dtype=np.float64).reshape(4, 3)
    x = (c[3] - c[0]) + (c[2] - c[1])
    y = (c[1] - c[0]) + (c[2] - c[3])
    x /= np.linalg.norm(x)
    y -= x * float(x @ y)
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    if z[2] < 0:                       # keep the normal pointing up
        y, z = -y, -z
    T = np.eye(4)
    T[:3, :3] = np.stack([x, y, z], axis=1)
    T[:3, 3] = c.mean(0)
    return T


def locate_marker(capture_dir: str, dict_type: str = "6X6_1000",
                  marker_id: Optional[int] = None,
                  undistort_dir: Optional[str] = None) -> dict:
    """Triangulate the standalone marker in ``capture_dir``.

    Returns a dict with the marker id, its 4 corners and center in both the
    camera-calibration ("world") frame and the robot frame, plus the marker
    pose and the measured side length (a sanity check: all four sides should
    agree to a couple of mm).
    """
    img_dict = ImageDict.from_path(capture_dir)
    # get_cammtx builds projections from intrinsics_undistort, so the 2D input
    # has to be undistorted first.
    und = img_dict.undistort(save_path=undistort_dir)
    marker_2d, marker_3d = und.triangulate_markers(dict_type=dict_type)

    board_ids = board_marker_ids()
    n_views = {int(k): len(v["2d"]) for k, v in marker_2d.items()}
    lone = {int(k): v for k, v in marker_3d.items()
            if int(k) not in board_ids and n_views[int(k)] >= MIN_VIEWS}
    if marker_id is not None:
        if marker_id not in lone:
            raise SystemExit(
                f"marker {marker_id} not triangulated from >={MIN_VIEWS} views "
                f"(candidates: { {k: n_views[k] for k in lone} })")
        mid = marker_id
    elif not lone:
        raise SystemExit(f"no standalone {dict_type} marker found in {capture_dir}")
    elif len(lone) > 1:
        raise SystemExit(f"several standalone markers found: "
                         f"{ {k: n_views[k] for k in lone} } — pass --marker_id")
    else:
        mid = next(iter(lone))

    corners_w = np.asarray(lone[mid], dtype=np.float64).reshape(4, 3)
    sides = [float(np.linalg.norm(corners_w[i] - corners_w[(i + 1) % 4]))
             for i in range(4)]

    C2R = load_current_C2R()
    R2C = np.linalg.inv(C2R)
    corners_r = (R2C @ np.hstack([corners_w, np.ones((4, 1))]).T).T[:, :3]
    return {
        "marker_id": int(mid),
        "dict_type": dict_type,
        "n_views": int(n_views[mid]),
        "corners_world": corners_w,
        "center_world": corners_w.mean(0),
        "corners_robot": corners_r,
        "center_robot": corners_r.mean(0),
        "pose_robot": marker_frame(corners_r),
        "side_len_m": sides,
        "capture_dir": str(capture_dir),
    }


def capture_images(pc_list=DEFAULT_PC_LIST, tag: str = "place_target",
                   settle_s: float = 0.3) -> str:
    """Grab one image per camera and return the capture dir (with cam_param)."""
    from paradex.io.camera_system.remote_camera_controller import remote_camera_controller

    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rel = os.path.join(CAPTURE_ROOT_REL, stamp)
    abs_dir = os.path.join(os.path.expanduser("~"), rel)
    os.makedirs(abs_dir, exist_ok=True)

    rcc = remote_camera_controller(tag, pc_list=pc_list)
    try:
        rcc.start("image", False, rel)
        rcc.stop()
        time.sleep(settle_s)
    finally:
        for fn in (rcc.stop, rcc.end):
            try:
                fn()
            except Exception:
                pass
    save_current_camparam(abs_dir)
    save_current_C2R(abs_dir)
    return abs_dir


def get_place_target(capture_dir: Optional[str] = None,
                     marker_id: Optional[int] = None,
                     dict_type: str = "6X6_1000") -> Tuple[np.ndarray, dict]:
    """Convenience entry point: ``(center_robot_xyz, info)``."""
    if capture_dir is None:
        capture_dir = capture_images()
    info = locate_marker(capture_dir, dict_type=dict_type, marker_id=marker_id)
    return info["center_robot"], info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--capture_dir", default=None,
                    help="existing capture dir (needs raw/images + cam_param)")
    ap.add_argument("--capture", action="store_true",
                    help="take a fresh image set instead of --capture_dir")
    ap.add_argument("--marker_id", type=int, default=None,
                    help="force a specific aruco id (default: the only "
                         "non-charuco-board id found)")
    ap.add_argument("--dict", dest="dict_type", default="6X6_1000")
    ap.add_argument("--pc_list", nargs="+", default=DEFAULT_PC_LIST)
    ap.add_argument("--save_json", default=None)
    args = ap.parse_args()

    if not args.capture and not args.capture_dir:
        ap.error("pass --capture_dir or --capture")
    cap = (capture_images(pc_list=args.pc_list) if args.capture
           else os.path.expanduser(args.capture_dir))
    print(f"[place_target] capture: {cap}")

    info = locate_marker(cap, dict_type=args.dict_type, marker_id=args.marker_id)
    np.set_printoptions(precision=4, suppress=True)
    print(f"  marker      : {args.dict_type} id={info['marker_id']} "
          f"({info['n_views']} views)")
    print(f"  side lengths: {np.round(info['side_len_m'], 4)} m")
    print(f"  center world: {info['center_world'].round(4)}")
    print(f"  center robot: {info['center_robot'].round(4)}")
    print(f"  pose robot  :\n{info['pose_robot'].round(4)}")

    if args.save_json:
        out = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
               for k, v in info.items()}
        Path(args.save_json).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.save_json, "w"), indent=1)
        print(f"  saved: {args.save_json}")


if __name__ == "__main__":
    main()
