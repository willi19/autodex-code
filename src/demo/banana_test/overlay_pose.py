#!/usr/bin/env python3
"""Overlay the perceived object mesh on a trial's init images.

Answers "did perception put the object where it actually was?" -- the thing the
finger/arm logs cannot tell you. Renders the object mesh at the pose the init
pipeline produced onto every camera view of that trial and writes a grid.

    # one trial
    python src/demo/banana_test/overlay_pose.py --trial <trial_dir>

    # every trial of the demo run, success + fail, into one folder
    python src/demo/banana_test/overlay_pose.py --all
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
import trimesh

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from paradex.calibration.utils import load_c2r
from paradex.image.image_dict import ImageDict

from autodex.utils.conversion import cart2se3
from autodex.utils.path import project_dir, get_obj_root
from src.execution.scene_cfg import pose_world_to_scene_cfg

MESH_BASE = os.path.expanduser("~/shared_data/AutoDex/object/paradex")


def overlay_trial(trial: str, obj: str, version: str, out_path: str,
                  alpha: float = 0.5) -> str:
    pose_world = np.load(os.path.join(trial, "pose_world.npy"))
    c2r = load_c2r(trial)
    scene_cfg = pose_world_to_scene_cfg(pose_world, c2r, obj,
                                        get_obj_root(version))
    obj_pose = cart2se3(scene_cfg["mesh"]["target"]["pose"])

    # ImageDict wants cam_param/ NEXT TO the images; the trial keeps it one
    # level up, so link it in rather than copying 36 KB per trial.
    img_dir = os.path.join(trial, "init_capture")
    if not os.path.isdir(img_dir):
        raise SystemExit(f"no init_capture/ in {trial}")
    link = os.path.join(img_dir, "cam_param")
    if not os.path.exists(link):
        os.symlink(os.path.realpath(os.path.join(trial, "cam_param")), link)

    img_dict = ImageDict.from_path(img_dir)
    # The renderer's projections come from intrinsics_undistort, so the images
    # under them have to be undistorted too.
    if not os.path.exists(os.path.join(img_dir, "images")):
        img_dict = img_dict.undistort()

    # Mesh lives in the object frame; the renderer works in CAMERA-calibration
    # ("world") frame, hence c2r @ obj_pose (obj_pose is robot frame).
    mesh_path = os.path.join(MESH_BASE, obj, "raw_mesh", f"{obj}.obj")
    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    mesh.apply_transform(c2r @ obj_pose)

    grid = img_dict.project_mesh(mesh, color=(0, 255, 0), alpha=alpha).merge()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, grid)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trial", default=None)
    ap.add_argument("--all", action="store_true",
                    help="every trial under the demo run dir")
    ap.add_argument("--obj", default="banana")
    ap.add_argument("--version", default="v8")
    ap.add_argument("--exp_name", default="banana_demo")
    ap.add_argument("--arm", default="franka")
    ap.add_argument("--hand", default="inspire")
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--alpha", type=float, default=0.5)
    args = ap.parse_args()

    run_dir = os.path.join(project_dir, "experiment", args.exp_name,
                           f"{args.arm}_{args.hand}", args.obj)
    if args.all:
        trials = sorted(d for d in glob.glob(os.path.join(run_dir, "2026*"))
                        if os.path.exists(os.path.join(d, "pose_world.npy")))
    elif args.trial:
        trials = [os.path.expanduser(args.trial)]
    else:
        ap.error("pass --trial or --all")

    out_dir = args.out_dir or os.path.join(run_dir, "pose_overlay")
    for t in trials:
        name = os.path.basename(t.rstrip("/"))
        rp = os.path.join(t, "result.json")
        tag = "unknown"
        if os.path.exists(rp):
            r = json.load(open(rp))
            tag = ("succ" if r.get("success") else
                   ("void" if r.get("success") is None else "fail"))
        out = os.path.join(out_dir, f"{name}_{tag}.png")
        try:
            overlay_trial(t, args.obj, args.version, out, alpha=args.alpha)
            print(f"  {name} [{tag}] -> {out}")
        except Exception as e:
            print(f"  {name} [{tag}] FAILED: {e!r}")


if __name__ == "__main__":
    main()
