"""Robot overlay for a single autodex_dataset trial (everything read from the trial).

Renders the robot (calibrated xarm+hand URDF) onto each camera video using the
trial's OWN C2R.npy + cam_param + synced qpos, to visually verify the C2R/URDF
spatial alignment. Two overlays per camera:

  overlay_actual_{serial}.mp4  -- robot at measured joints (arm/state + hand/state)
  overlay_cmd_{serial}.mp4     -- robot at commanded joints (arm/action_qpos + hand/action)

action_qpos.npy is used for the command (NOT action.npy, which holds the raw
cartesian wrist_se3 on lift frames).

    ~/miniconda3/envs/foundationpose/bin/python -m src.visualization.dataset.overlay_trial \
        --trial ~/shared_data/autodex_dataset/selected_100/toothbrush_holder/20260121_181907 \
        --hand allegro [--serials 24080331 25305460] [--max_serials 4]
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path.home() / "paradex"))
from paradex.calibration.utils import load_camparam
from paradex.visualization.robot import RobotModule

from src.visualization.overlay_robot_video import RobotOverlayRenderer, _label_for_link

URDF_BASE = Path.home() / "AutoDex" / "autodex" / "planner" / "src" / "curobo" / "content" / "assets" / "robot"


def load_qpos(trial, arm_file, hand_file):
    arm = np.load(trial / "arm" / arm_file)
    hand = np.load(trial / "hand" / hand_file)
    n = min(len(arm), len(hand))
    return np.concatenate([arm[:n], hand[:n]], axis=1).astype(np.float32)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trial", required=True)
    ap.add_argument("--hand", default="allegro")
    ap.add_argument("--serials", nargs="*", default=None)
    ap.add_argument("--max_serials", type=int, default=4)
    ap.add_argument("--out", default=None, help="output dir (default: {trial}/robot_overlay)")
    args = ap.parse_args()

    trial = Path(args.trial)
    out_dir = Path(args.out) if args.out else trial / "robot_overlay"
    out_dir.mkdir(parents=True, exist_ok=True)

    c2r = np.load(trial / "C2R.npy")
    intrinsic, extrinsic_from_camparam = load_camparam(str(trial))

    serials = sorted(intrinsic.keys())
    serials = [s for s in serials if (trial / "videos" / f"{s}.avi").exists()]
    if args.serials:
        serials = [s for s in serials if s in args.serials]
    else:
        serials = serials[: args.max_serials]
    if not serials:
        sys.exit("no matching camera videos")
    print(f"[overlay] {trial.name}  serials={serials}")

    # arm/hand are already on the 30fps video-frame axis (src/dataset/resync_video_axis.py),
    # so frame f of the qpos == video frame f directly.
    actual = load_qpos(trial, "state.npy", "state.npy")           # arm/state + hand/state (measured)
    cmd = load_qpos_cmd(trial)                                     # arm/action_qpos + hand/action

    # robot + mesh scene (once)
    urdf_path = str(URDF_BASE / f"{args.hand}_description" / f"xarm_{args.hand}.urdf")
    robot = RobotModule(urdf_path)
    dof = robot.get_num_joints()
    robot.update_cfg(actual[0, :dof])
    scene = robot.scene
    link_names = list(scene.geometry.keys())
    scene_meshes = [scene.geometry[ln] for ln in link_names]
    link_labels = {ln: _label_for_link(ln) for ln in link_names}

    # cam_from_robot = cam_from_world @ c2r
    render_ext = {}
    for s in serials:
        cfw = np.eye(4)
        cfw[:3, :] = extrinsic_from_camparam[s]
        render_ext[s] = (cfw @ c2r)[:3, :]

    caps = {s: cv2.VideoCapture(str(trial / "videos" / f"{s}.avi")) for s in serials}
    W = int(caps[serials[0]].get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(caps[serials[0]].get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = caps[serials[0]].get(cv2.CAP_PROP_FPS) or 30.0

    intr_sub = {s: intrinsic[s] for s in serials}
    renderer = RobotOverlayRenderer(scene_meshes, link_names, link_labels, intr_sub, render_ext, H, W)
    ordered = renderer.serials

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    wa = {s: cv2.VideoWriter(str(out_dir / f"_a_{s}.avi"), fourcc, fps, (W, H)) for s in ordered}
    wc = {s: cv2.VideoWriter(str(out_dir / f"_c_{s}.avi"), fourcc, fps, (W, H)) for s in ordered}

    # The video just dropped its last frames: video frame f == trigger f == qpos[f].
    nv = min(int(c.get(cv2.CAP_PROP_FRAME_COUNT)) for c in caps.values())
    n = min(nv, len(actual), len(cmd))
    print(f"[overlay] video {nv}, qpos {len(actual)} -> render {n}, {W}x{H}")
    for f in range(n):
        frames = []
        for s in ordered:
            ret, fr = caps[s].read()
            frames.append(fr if ret else np.zeros((H, W, 3), np.uint8))
        for seq, wr in [(actual, wa), (cmd, wc)]:
            robot.update_cfg(seq[f, :dof])
            sc = robot.scene
            link_poses = [sc.graph.get(ln)[0] for ln in link_names]
            outs = renderer.render(link_poses, frames)
            for s, img in zip(ordered, outs):
                wr[s].write(img)

    for c in caps.values():
        c.release()
    for s in ordered:
        wa[s].release(); wc[s].release()
        for tag, w in [("actual", "_a"), ("cmd", "_c")]:
            src = out_dir / f"{w}_{s}.avi"
            dst = out_dir / f"overlay_{tag}_{s}.mp4"
            os.system(f"ffmpeg -y -loglevel error -i '{src}' -c:v libx264 -pix_fmt yuv420p '{dst}' && rm '{src}'")
    print(f"[overlay] -> {out_dir}")


def load_qpos_cmd(trial):
    arm = np.load(trial / "arm" / "action_qpos.npy")
    hand = np.load(trial / "hand" / "action.npy")
    n = min(len(arm), len(hand))
    return np.concatenate([arm[:n], hand[:n]], axis=1).astype(np.float32)


if __name__ == "__main__":
    main()
