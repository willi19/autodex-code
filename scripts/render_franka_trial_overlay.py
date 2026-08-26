#!/usr/bin/env python3
"""Render an occlusion-correct FR3 + Inspire + object diagnostic overlay.

The renderer uses nvdiffrast's depth buffer: hidden surfaces are not drawn.
It is intended for an already-recorded trial, not for commanding hardware.
Run with the planner environment, for example:

  conda run -n planner python scripts/render_franka_trial_overlay.py \
    --trial /home/robot/shared_data/AutoDex/experiment/v8/inspire/banana/20260826_140946 \
    --output /home/robot/AutoDex/outputs/franka_overlay_20260826_140946.png
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import trimesh
import yourdfpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.visualization.overlay_full_robot_thumbnail_single import (  # noqa: E402
    MultiMeshOverlayRenderer,
    load_cam_param,
)


ARM_COLORS = [
    (70, 70, 220), (40, 150, 245), (50, 210, 120), (220, 190, 40),
    (210, 90, 190), (150, 80, 50), (180, 180, 180),
]
FINGER_COLORS = {
    "thumb": (0, 140, 255), "index": (255, 200, 0),
    "middle": (100, 255, 0), "ring": (200, 0, 255),
    "little": (0, 220, 255),
}
OBJECT_COLOR = (255, 80, 80)  # blue in BGR


def _hand_action_to_qpos(action: np.ndarray) -> np.ndarray:
    """Invert Inspire controller order/units into the FR3 URDF joint order."""
    a = np.clip(np.asarray(action, dtype=np.float64), 0.0, 1000.0)
    limits = np.array([1.15, 0.55, 1.60, 1.60, 1.60, 1.60])
    # controller order: pinky, ring, middle, index, thumb pitch, thumb yaw
    return limits * np.array([
        1.0 - a[5] / 1000.0, 1.0 - a[4] / 1000.0,
        1.0 - a[3] / 1000.0, 1.0 - a[2] / 1000.0,
        1.0 - a[1] / 1000.0, 1.0 - a[0] / 1000.0,
    ])


def _color_for_geometry(name: str):
    for i in range(7):
        if name.startswith(f"fr3_link{i}_"):
            return ARM_COLORS[i]
    for finger, color in FINGER_COLORS.items():
        if f"right_{finger}_" in name:
            return color
    return (120, 120, 120)


def _load_mesh(path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))
    return mesh


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--serials", nargs="+", default=[
        "24080331", "25305463", "25322643", "25322649"],
        help="Camera serials to render into a grid.")
    parser.add_argument("--alpha", type=float, default=0.55,
                        help="Overlay opacity in [0, 1]; depth occlusion remains enabled.")
    args = parser.parse_args()

    trial = args.trial.resolve()
    raw = trial / "raw"
    arm_q = np.load(raw / "arm" / "position.npy")[-1, :7]
    hand_action = np.load(raw / "hand" / "position.npy")[-1, :6]
    hand_q = _hand_action_to_qpos(hand_action)
    q = np.concatenate([arm_q, hand_q])

    assets = Path("/home/robot/shared_data/AutoDex/content/assets/robot")
    urdf = yourdfpy.URDF.load(str(assets / "fr3_inspire_description/fr3_inspire.urdf"))
    urdf.update_cfg(dict(zip(urdf.actuated_joint_names, q)))
    c2r = np.load(trial / "C2R.npy")
    if c2r.shape == (3, 4):
        c2r = np.vstack([c2r, [0, 0, 0, 1]])

    meshes, poses, colors = [], [], []
    for name, mesh in urdf.scene.geometry.items():
        meshes.append(mesh)
        poses.append(c2r @ urdf.scene.graph.get(name)[0])
        colors.append(_color_for_geometry(name))

    obj_mesh = _load_mesh(Path("/home/robot/shared_data/object_processing") /
                          "banana/raw_mesh/banana.obj")
    meshes.append(obj_mesh)
    poses.append(np.load(trial / "pose_world.npy"))
    colors.append(OBJECT_COLOR)

    intr, extr = load_cam_param(trial / "cam_param")
    images, used = [], []
    for serial in args.serials:
        image_path = trial / "init_capture" / "images" / f"{serial}.png"
        if serial in intr and image_path.exists():
            images.append(cv2.imread(str(image_path), cv2.IMREAD_COLOR))
            used.append(serial)
    if not images:
        raise RuntimeError("None of the requested camera images are available")
    h, w = images[0].shape[:2]
    intr = {s: intr[s] for s in used}
    extr = {s: extr[s] for s in used}

    alpha = float(np.clip(args.alpha, 0.0, 1.0))
    renderer = MultiMeshOverlayRenderer(meshes, colors, [alpha] * len(meshes),
                                        intr, extr, h, w)
    overlays = renderer.render(poses, images)
    labelled = []
    for serial, image in zip(renderer.serials, overlays):
        cv2.putText(image, serial, (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.4,
                    (255, 255, 255), 3, cv2.LINE_AA)
        labelled.append(image)
    top = np.hstack(labelled[:2])
    bottom = np.hstack(labelled[2:]) if len(labelled) > 2 else top.copy()
    grid = np.vstack([top, bottom]) if len(labelled) > 2 else top
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), grid):
        raise RuntimeError(f"Could not write {args.output}")
    print(f"wrote {args.output}")
    print("arm_qpos:", np.round(arm_q, 3).tolist())
    print("hand_qpos:", np.round(hand_q, 3).tolist())


if __name__ == "__main__":
    main()
