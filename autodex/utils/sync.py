"""Robot/hand qpos synchronization to video frames.

Reads raw recordings (raw/arm/, raw/hand/, raw/timestamps/timestamp.npy) and
writes synced per-frame qpos to {exp}/arm/state.npy, arm/action.npy,
hand/state.npy, hand/action.npy (following paradex.utils.file_io convention).

All outputs are in URDF joint order (with hand remapping applied).

After preprocessing, use paradex.utils.file_io.load_robot_traj /
load_robot_target_traj to read the synced qpos.
"""
import numpy as np
from pathlib import Path
from scipy.interpolate import interp1d

# Allegro joint order remapping
# Position: JointState (ROS) order → URDF
ALLEGRO_POS_TO_URDF = [5, 2, 0, 1, 7, 15, 14, 12, 11, 13, 4, 9, 8, 6, 10, 3]
# Action: CMD order (_convert_allegro applied) → URDF
ALLEGRO_ACT_TO_URDF = list(range(4, 16)) + list(range(0, 4))

# Inspire: raw 6-dim [little, ring, middle, index, thumb2, thumb1] → RobotModule order [thumb1, thumb2, index, middle, ring, little]
INSPIRE_RAW_TO_URDF = [5, 4, 3, 2, 1, 0]
# Per-joint limits in RobotModule order (radians). Value = limit * (1 - raw/1000).
INSPIRE_LIMITS = np.array([1.15, 0.55, 1.6, 1.6, 1.6, 1.6])


def convert_inspire_raw(raw):
    """Convert raw inspire values (0-1000, N×6) to URDF radians in RobotModule joint order."""
    return INSPIRE_LIMITS * (1.0 - raw[:, INSPIRE_RAW_TO_URDF] / 1000.0)


def resample(src_time, src_values, target_time):
    """Linear interpolate src_values to target_time. Clamps outside range.

    Some captures truncate mid-write, leaving time and values off-by-one (e.g.
    time has one fewer sample than position/action). Clip both to the common
    length so interp1d gets equal-length axes.
    """
    n = min(len(src_time), len(src_values))
    src_time, src_values = src_time[:n], src_values[:n]
    t = np.clip(target_time, float(src_time[0]), float(src_time[-1]))
    return interp1d(src_time, src_values, axis=0)(t)


def load_video_times(exp_dir):
    """Load video frame timestamps. Returns (video_times, frame_ids)."""
    exp_dir = Path(exp_dir)
    ts = np.load(exp_dir / "raw" / "timestamps" / "timestamp.npy")
    fids_path = exp_dir / "raw" / "timestamps" / "frame_id.npy"
    fids = np.load(fids_path) if fids_path.exists() else np.arange(1, len(ts) + 1, dtype=int)
    return ts, fids


def precompute_synced_qpos(exp_dir, hand_type,
                           arm_time_offset=0.03, hand_time_offset=0.03,
                           overwrite=False):
    """Build per-frame qpos synced to video timestamps; save to exp_dir.

    Writes (paradex convention, URDF joint order):
      {exp}/arm/state.npy     (N, 6)  actual arm position
      {exp}/arm/action.npy    (N, 6)  commanded arm position (action_qpos)
      {exp}/hand/state.npy    (N, H)  actual hand position
      {exp}/hand/action.npy   (N, H)  commanded hand action

    Positive *_time_offset means the synced value is pulled from
    (video_time - offset) in raw timeline. Default 0.09 compensates for
    video stream being ~0.09s slower than robot stream.

    Args:
        exp_dir: experiment dir containing raw/.
        hand_type: "allegro", "inspire", or "inspire_left".
        overwrite: re-compute even if synced files exist.

    Returns:
        (arm_state, arm_action, hand_state, hand_action)
    """
    exp_dir = Path(exp_dir)
    arm_out = exp_dir / "arm"
    hand_out = exp_dir / "hand"
    arm_out.mkdir(exist_ok=True)
    hand_out.mkdir(exist_ok=True)

    paths = {
        "arm_state": arm_out / "state.npy",
        "arm_action": arm_out / "action.npy",
        "hand_state": hand_out / "state.npy",
        "hand_action": hand_out / "action.npy",
    }
    if not overwrite and all(p.exists() for p in paths.values()):
        return (np.load(paths["arm_state"]), np.load(paths["arm_action"]),
                np.load(paths["hand_state"]), np.load(paths["hand_action"]))

    video_times, _ = load_video_times(exp_dir)

    arm_dir = exp_dir / "raw" / "arm"
    hand_dir = exp_dir / "raw" / "hand"

    arm_time = np.load(arm_dir / "time.npy") + arm_time_offset
    arm_pos = np.load(arm_dir / "position.npy")
    arm_action_p = arm_dir / "action_qpos.npy"
    arm_action_raw = np.load(arm_action_p) if arm_action_p.exists() else arm_pos

    hand_time = np.load(hand_dir / "time.npy") + hand_time_offset
    hand_pos = np.load(hand_dir / "position.npy")
    hand_action_raw = np.load(hand_dir / "action.npy")

    arm_state = resample(arm_time, arm_pos, video_times)
    arm_action = resample(arm_time, arm_action_raw, video_times)
    hand_state = resample(hand_time, hand_pos, video_times)
    hand_action = resample(hand_time, hand_action_raw, video_times)

    if hand_type in ("inspire", "inspire_left"):
        hand_state = convert_inspire_raw(hand_state)
        hand_action = convert_inspire_raw(hand_action)
    elif hand_type == "allegro":
        hand_state = hand_state[:, ALLEGRO_POS_TO_URDF]
        hand_action = hand_action[:, ALLEGRO_ACT_TO_URDF]

    np.save(paths["arm_state"], arm_state)
    np.save(paths["arm_action"], arm_action)
    np.save(paths["hand_state"], hand_state)
    np.save(paths["hand_action"], hand_action)

    return arm_state, arm_action, hand_state, hand_action