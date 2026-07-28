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
# 7-DOF. Solved by IK so the wrist (hand base_link) lands on the same 6D pose
# the xarm reaches at XARM_INIT — FK check reproduced it to 0.01 mm / 0.000 deg.
# Inside both the URDF limits and the real FR3 spec ranges.
FR3_INIT = np.array([
    0.65911102, -0.26389799, -1.03441095, -2.58232594, -0.48430899,
    3.98870993, 0.87985802
])

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
