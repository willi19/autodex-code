import numpy as np

# ── XArm6 ────────────────────────────────────────────────────────────────────
XARM_INIT = np.array([
    -0.21991149, -0.20245819, -1.13620934, 2.33175988, 0.31939525, 2.36492114
])

XARM_INSPIRE_INIT = XARM_INIT.copy()

# ── Allegro ──────────────────────────────────────────────────────────────────
ALLEGRO_INIT = np.array([
    0.0, 1.5707, 0.0, 0.0,
    0.0, 1.5707, 0.0, 0.0,
    0.0, 1.5707, 0.0, 0.0,
    1.24565697, 0.05513508, 0.23153956, -0.02217758
])

ALLEGRO_LINK6_TO_WRIST = np.array([
    [0, 1, 0, 0],
    [-1, 0, 0, 0],
    [0, 0, 1, 0.1552],
    [0, 0, 0, 1]
])

# ── Inspire ──────────────────────────────────────────────────────────────────
INSPIRE_INIT = np.zeros(6)  # 6 DOF, all zeros = open hand

INSPIRE_LINK6_TO_WRIST = np.array([
    [1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0.035],
    [0, 0, 0, 1]
])

# inspire_left URDF chain (link6 -> wrist -> base_link) composes Rx(π) · Rz(π) = Ry(π).
INSPIRE_LEFT_LINK6_TO_WRIST = np.array([
    [-1, 0, 0, 0],
    [ 0, 1, 0, 0],
    [ 0, 0,-1, 0.035],
    [ 0, 0, 0, 1]
])

# ── FR3 (Franka) ─────────────────────────────────────────────────────────────
# 7-DOF init = the franka HOME pose saved by paradex hand-eye calibration
# (system/current/hecalib/franka/home_qpos.npy). Executor homes here and the
# planner plans trajectories starting here, so both stay consistent with the
# real robot's calibrated home. Falls back to the last-known home values if the
# paradex file is unavailable. (Previously an IK-solved pose matching the xarm
# init wrist; switched to the real calibrated home on user request.)
import os as _os
_FR3_HOME_FILE = _os.path.expanduser(
    "~/paradex/system/current/hecalib/franka/home_qpos.npy")
try:
    FR3_INIT = np.load(_FR3_HOME_FILE).astype(np.float64)
    assert FR3_INIT.shape == (7,)
except Exception:
    FR3_INIT = np.array([-0.00315, 0.02135, 0.00298, -2.32733,
                         -0.00027, 3.95842, 0.78357])
# NOTE: inspire hand is mounted 180° reversed on the flange; that is modeled by
# the URDF flange_to_hand yaw (rotated +pi). The home qpos itself is left as the
# calibrated home — at this config the (remounted) hand is already 180°-rotated
# per the new URDF. Rotating joint7 here would make the arm compensate (undoing
# the visual) and pushes joint7 outside cuRobo's limit (usd_flip_joint_limits),
# breaking plan_single_js with INVALID_START_STATE_JOINT_LIMITS.

# fr3_link7 -> wrist (hand base_link), from the fr3_inspire URDF's fixed chain
# (fr3_joint8 -> flange_to_hand). FR3 analog of INSPIRE_LINK6_TO_WRIST; the same
# FK derivation reproduces INSPIRE_LINK6_TO_WRIST exactly for the xarm.
FR3_INSPIRE_LINK_TO_WRIST = np.array([
    [-0.70710431, -0.70710925, 0.0, 0.0],
    [-0.70710925,  0.70710431, 0.0, 0.0],
    [ 0.0,         0.0,       -1.0, 0.147],
    [ 0.0,         0.0,        0.0, 1.0]
])

# ── Defaults (allegro) ──────────────────────────────────────────────────────
INIT_STATE = np.concatenate([XARM_INIT, ALLEGRO_INIT])
LINK6_TO_WRIST = ALLEGRO_LINK6_TO_WRIST
