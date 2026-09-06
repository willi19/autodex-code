"""Render the intended Inspire grasp pose on one recorded P2 grasp frame.

This is an episode-analysis utility.  It deliberately does not alter the
recording, the AutoDex plan, or any labels.  For a P2 episode it:

* recovers the selected fixed-success candidate's planned ``grasp_pose``;
* finds the middle of the recorded hand-closing motion after the arm reached
  the planned approach endpoint; and
* projects the planned Franka--Inspire hand mesh into every synchronized AVI
  view at that one timestamp.

The output makes it possible to distinguish a perception/planning pose error
from an execution or object-retention failure without replaying the robot.

Example
-------
python src/demo/p2/render_grasp_pose_overlay.py \\
  --episode /home/robot/shared_data/AutoDex/experiment/v8_demo/inspire/pringles/20260902_160410
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
PARADEX_ROOT = Path.home() / "paradex"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PARADEX_ROOT) not in sys.path:
    sys.path.insert(0, str(PARADEX_ROOT))

from paradex.calibration.utils import load_camparam
from paradex.visualization.robot import RobotModule
from src.visualization.overlay_robot_video import RobotOverlayRenderer, _label_for_link


INPIRE_LIMITS = np.array([1.15, 0.55, 1.6, 1.6, 1.6, 1.6], dtype=np.float64)
HAND_GEOMETRY_PREFIXES = ("base_link.", "right_thumb_", "right_index_",
                          "right_middle_", "right_ring_", "right_little_")
FR3_INSPIRE_URDF = (
    Path.home() / "shared_data/AutoDex/content/assets/robot"
    / "fr3_inspire_description/fr3_inspire.urdf")
CANDIDATE_ROOT = Path.home() / "shared_data/AutoDex/candidates/inspire/v8"


def _read_json(path: Path) -> dict:
    with path.open() as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path}")
    return value


def _inspire_qpos_to_action(qpos: np.ndarray) -> np.ndarray:
    """Planner radians -> physical controller order used in raw hand logs."""
    q = np.asarray(qpos, dtype=np.float64).reshape(6)
    normalized = np.clip(q / INPIRE_LIMITS, 0.0, 1.0)
    planner_order = (1.0 - normalized) * 1000.0
    # [thumb_yaw, thumb_pitch, index, middle, ring, pinky] ->
    # [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
    return planner_order[[5, 4, 3, 2, 1, 0]]


def _source_candidate_hand_poses(episode: Path, record: dict) -> tuple[np.ndarray, np.ndarray, Path]:
    """Recover the exact v8 candidate used by a fixed-success replay.

    P2's plan trajectory contains the approach/pregrasp hand configuration,
    while the closed grasp configuration is retained in the candidate that
    generated the successful source episode.  Recovering it avoids rendering
    the wrong (open) hand pose in the diagnostic image.
    """
    info = record.get("scene_info")
    if not (isinstance(info, list) and len(info) >= 3
            and info[0] == "fixed-inspire" and info[1] == "v8_inspire"):
        raise ValueError(
            "This utility currently requires a fixed v8-Inspire source; "
            f"got scene_info={info!r}")

    source_episode = Path(info[2])
    source_record = _read_json(source_episode / "result.json")
    source_info = source_record.get("scene_info")
    if not (isinstance(source_info, list) and len(source_info) == 3):
        raise ValueError(f"Invalid v8 source scene_info in {source_episode}")

    candidate = CANDIDATE_ROOT / episode.parent.name
    candidate = candidate.joinpath(*(str(part) for part in source_info))
    pregrasp_file = candidate / "pregrasp_pose.npy"
    grasp_file = candidate / "grasp_pose.npy"
    if not (pregrasp_file.is_file() and grasp_file.is_file()):
        raise FileNotFoundError(
            "The selected v8 candidate is unavailable: " f"{candidate}")
    pregrasp = np.asarray(np.load(pregrasp_file), dtype=np.float64).reshape(6)
    grasp = np.asarray(np.load(grasp_file), dtype=np.float64).reshape(6)
    return pregrasp, grasp, candidate


def _select_mid_grasp_frame(episode: Path, planned_arm: np.ndarray,
                            pregrasp: np.ndarray, grasp: np.ndarray) -> dict:
    """Find the recorded video timestamp halfway through the commanded close."""
    arm_time = np.asarray(np.load(episode / "raw/robot/arm/time.npy"), dtype=np.float64)
    arm_qpos = np.asarray(np.load(episode / "raw/robot/arm/position.npy"), dtype=np.float64)
    hand_time = np.asarray(np.load(episode / "raw/robot/hand/time.npy"), dtype=np.float64)
    hand_action = np.asarray(np.load(episode / "raw/robot/hand/position.npy"), dtype=np.float64)
    video_time = np.asarray(
        np.load(episode / "raw/timestamps/timestamp.npy"), dtype=np.float64)
    if arm_qpos.ndim != 2 or arm_qpos.shape[1] != 7:
        raise ValueError(f"Expected raw Franka joint positions, got {arm_qpos.shape}")
    if hand_action.ndim != 2 or hand_action.shape[1] != 6:
        raise ValueError(f"Expected raw Inspire actions, got {hand_action.shape}")

    arm_error = np.linalg.norm(arm_qpos - planned_arm[None], axis=1)
    arm_index = int(np.argmin(arm_error))
    arm_arrival_time = float(arm_time[arm_index])

    pre_action = _inspire_qpos_to_action(pregrasp)
    grasp_action = _inspire_qpos_to_action(grasp)
    close_delta = grasp_action - pre_action
    close_norm2 = float(close_delta @ close_delta)
    if close_norm2 <= 1e-8:
        raise ValueError("Candidate pregrasp and grasp actions are identical")

    # Ignore unrelated hand activity before the arm first arrives and during
    # the later lift/transfer.  The close ramps are < 2 seconds, but the wider
    # 4 s window also covers a slow/lagging hand controller safely.
    valid = ((hand_time >= arm_arrival_time - 0.10)
             & (hand_time <= arm_arrival_time + 4.0))
    if not np.any(valid):
        raise RuntimeError("No hand samples around the planned approach endpoint")
    progress = ((hand_action - pre_action) @ close_delta) / close_norm2
    perpendicular = np.linalg.norm(
        hand_action - (pre_action + progress[:, None] * close_delta), axis=1)
    # Minimize deviation from 50%-closed; a modest perpendicular term prevents
    # an unrelated single-finger state from being selected as the midpoint.
    score = np.abs(progress - 0.5) + 0.002 * perpendicular
    score[~valid] = np.inf
    hand_index = int(np.argmin(score))
    selected_time = float(hand_time[hand_index])
    timestamp_index = int(np.argmin(np.abs(video_time - selected_time)))

    return {
        "arm_arrival_raw_index": arm_index,
        "arm_arrival_timestamp": arm_arrival_time,
        "arm_goal_l2_rad": float(arm_error[arm_index]),
        "hand_mid_grasp_raw_index": hand_index,
        "hand_mid_grasp_timestamp": selected_time,
        "hand_close_progress": float(progress[hand_index]),
        "hand_action_recorded": hand_action[hand_index].round(3).tolist(),
        "pregrasp_action_planned": pre_action.round(3).tolist(),
        "grasp_action_planned": grasp_action.round(3).tolist(),
        "timestamp_index": timestamp_index,
        "video_timestamp": float(video_time[timestamp_index]),
        "time_from_video_start_s": float(video_time[timestamp_index] - video_time[0]),
    }


def _read_video_frame(path: Path, frame_index: int) -> tuple[np.ndarray, int, float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    decoded_index = min(max(int(frame_index), 0), max(n_frames - 1, 0))
    cap.set(cv2.CAP_PROP_POS_FRAMES, decoded_index)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise RuntimeError(f"Cannot decode frame {decoded_index} from {path}")
    return frame, decoded_index, fps


def _draw_label(image: np.ndarray, serial: str, time_from_start_s: float) -> np.ndarray:
    result = image.copy()
    text = f"{serial}  |  planned grasp hand  |  t={time_from_start_s:.2f}s"
    cv2.rectangle(result, (20, 18), (20 + min(940, 13 * len(text)), 68),
                  (15, 15, 15), -1)
    cv2.putText(result, text, (32, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.78,
                (255, 255, 255), 2, cv2.LINE_AA)
    return result


def _make_grid(overlays: dict[str, np.ndarray], out_path: Path) -> None:
    """Write a compact 5x4 grid as a quick all-view diagnostic."""
    tile_w = 480
    tiles = []
    for serial in sorted(overlays):
        image = overlays[serial]
        scale = tile_w / image.shape[1]
        tile = cv2.resize(image, (tile_w, int(round(image.shape[0] * scale))),
                          interpolation=cv2.INTER_AREA)
        tiles.append(tile)
    if not tiles:
        raise RuntimeError("No overlays to grid")
    tile_h = max(tile.shape[0] for tile in tiles)
    padded = [cv2.copyMakeBorder(tile, 0, tile_h - tile.shape[0], 0, 0,
                                 cv2.BORDER_CONSTANT, value=(0, 0, 0))
              for tile in tiles]
    columns = 5
    rows = []
    for start in range(0, len(padded), columns):
        row = padded[start:start + columns]
        while len(row) < columns:
            row.append(np.zeros_like(padded[0]))
        rows.append(np.concatenate(row, axis=1))
    cv2.imwrite(str(out_path), np.concatenate(rows, axis=0))


def render_episode(episode: Path, out_dir: Path | None = None) -> Path:
    episode = episode.expanduser().resolve()
    if not episode.is_dir():
        raise FileNotFoundError(episode)
    for required in ("plan_traj.npy", "result.json", "C2R.npy", "cam_param",
                     "raw/robot/arm/position.npy", "videos/exec"):
        if not (episode / required).exists():
            raise FileNotFoundError(f"Missing {required} in {episode}")

    record = _read_json(episode / "result.json")
    plan = np.asarray(np.load(episode / "plan_traj.npy"), dtype=np.float64)
    if plan.ndim != 2 or plan.shape[0] == 0 or plan.shape[1] != 13:
        raise ValueError(f"Expected plan_traj shape (T, 13), got {plan.shape}")
    planned_arm = plan[-1, :7]
    pregrasp, grasp, candidate_dir = _source_candidate_hand_poses(episode, record)
    match = _select_mid_grasp_frame(episode, planned_arm, pregrasp, grasp)

    if out_dir is None:
        out_dir = episode / "grasp_pose_overlay"
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    intrinsic, extrinsic = load_camparam(str(episode))
    c2r = np.asarray(np.load(episode / "C2R.npy"), dtype=np.float64)
    if c2r.shape != (4, 4):
        raise ValueError(f"Expected C2R shape (4, 4), got {c2r.shape}")

    available = sorted(serial for serial in intrinsic
                       if (episode / "videos/exec" / f"{serial}.avi").is_file())
    if not available:
        raise FileNotFoundError(f"No execution AVIs in {episode / 'videos/exec'}")

    frames: dict[str, np.ndarray] = {}
    decoded = {}
    for serial in available:
        frame, decoded_index, fps = _read_video_frame(
            episode / "videos/exec" / f"{serial}.avi", match["timestamp_index"])
        frames[serial] = frame
        decoded[serial] = {"frame_index": decoded_index, "fps": fps,
                           "n_frames": None}
    first = frames[available[0]]
    h, w = first.shape[:2]
    if any(frame.shape[:2] != (h, w) for frame in frames.values()):
        raise ValueError("All P2 execution AVIs must have the same resolution")

    if not FR3_INSPIRE_URDF.is_file():
        raise FileNotFoundError(f"FR3-Inspire URDF missing: {FR3_INSPIRE_URDF}")
    robot = RobotModule(str(FR3_INSPIRE_URDF))
    expected_dof = robot.get_num_joints()
    planned_full = np.concatenate([planned_arm, grasp])
    if planned_full.shape != (expected_dof,):
        raise ValueError(
            f"Plan/candidate have {planned_full.shape[0]} DOF, URDF expects {expected_dof}")
    robot.update_cfg({name: value for name, value in
                      zip(robot.get_joint_names(), planned_full)})
    scene = robot.scene
    all_names = list(scene.geometry.keys())
    hand_names = [name for name in all_names
                  if name.startswith(HAND_GEOMETRY_PREFIXES)]
    if not hand_names:
        raise RuntimeError("No Inspire hand meshes were found in the FR3 URDF")
    hand_meshes = [scene.geometry[name] for name in hand_names]
    hand_poses = [scene.graph.get(name)[0] for name in hand_names]
    hand_labels = {name: _label_for_link(name) for name in hand_names}

    camera_from_robot = {}
    for serial in available:
        camera_from_world = np.eye(4, dtype=np.float64)
        camera_from_world[:3, :] = np.asarray(extrinsic[serial], dtype=np.float64)
        camera_from_robot[serial] = (camera_from_world @ c2r)[:3, :]
    renderer = RobotOverlayRenderer(
        hand_meshes, hand_names, hand_labels,
        {serial: intrinsic[serial] for serial in available}, camera_from_robot,
        h, w)
    rendered = renderer.render(hand_poses, [frames[serial] for serial in renderer.serials])
    overlays = {
        serial: _draw_label(image, serial, match["time_from_video_start_s"])
        for serial, image in zip(renderer.serials, rendered)
    }
    for serial, image in overlays.items():
        cv2.imwrite(str(out_dir / f"{serial}.png"), image)
    _make_grid(overlays, out_dir / "grasp_pose_overlay_grid.png")

    metadata = {
        "episode": str(episode),
        "description": (
            "Semi-transparent overlay of the planned closed Inspire hand only. "
            "The frame is selected halfway through the recorded close after the "
            "Franka reached the planned approach endpoint."),
        "candidate_dir": str(candidate_dir),
        "planned_arm_qpos_rad": planned_arm.tolist(),
        "planned_pregrasp_qpos_rad": pregrasp.tolist(),
        "planned_grasp_qpos_rad": grasp.tolist(),
        "match": match,
        "videos": decoded,
        "views": renderer.serials,
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"[grasp-overlay] {len(overlays)} views -> {out_dir}")
    print("[grasp-overlay] "
          f"frame={match['timestamp_index']} t={match['time_from_video_start_s']:.2f}s "
          f"close={match['hand_close_progress'] * 100:.1f}% "
          f"arm-goal-error={match['arm_goal_l2_rad']:.4f} rad")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", required=True, type=Path,
                        help="P2 episode directory containing videos/exec and raw robot logs")
    parser.add_argument("--out-dir", type=Path, default=None,
                        help="default: <episode>/grasp_pose_overlay")
    args = parser.parse_args()
    render_episode(args.episode, args.out_dir)


if __name__ == "__main__":
    main()
