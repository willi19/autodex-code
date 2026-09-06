"""Execute the pre-computed paper-lift trajectory on FR3 + inspire.

Input (NAS, written by grasp_mingi):
    ~/shared_data/grasp_mingi/paper_lift_trajectory.npz
        wrist_se3   (T, 4, 4)  wrist (= curobo ee_link "base_link") pose, robot frame
        arm_cfg     (T, 7)     FR3 joint values
        finger_cfg  (T, 6)     inspire joint values, curobo order
                               [thumb_1, thumb_2, index_1, middle_1, ring_1, little_1]
        contact     (T, 5)     per-step fingertip/table contact flags (log only)
        joint_names (6,)       hand joint names — asserted against the planner's order
        lift_height ()         wrist +z travel over the trajectory
    ~/shared_data/grasp_mingi/paper_open_wrist_se3.npy   (4, 4)  open-start wrist

The npz IS the grasp+lift motion (open hand at step 0 -> fingers closed + wrist
lifted at step T-1); it does NOT include getting there from home.

ARM SOURCE (``--arm``). The npz's ``arm_cfg`` was solved against a possibly
different URDF, so by default the ARM IS RE-SOLVED from ``wrist_se3`` on THIS
repo's ``fr3_inspire.urdf`` (and may be nudged with ``--wrist_dx/dy/dz``);
``finger_cfg`` is ALWAYS sent verbatim, never re-solved. The npz's wrist motion
is a pure +z translation of ``lift_height`` at fixed orientation.
  * ``--arm ik``  (default) approach = collision-checked plan to wrist_se3[0];
                            lift = per-waypoint IK straight up +z with the wrist
                            orientation pinned at every step (NOT
                            ``plan_pose_constrained``, which only pins the end
                            pose and lets the wrist rotate on the way up)
  * ``--arm npz``           execute ``arm_cfg`` verbatim (only safe if the URDF
                            matches the one that generated it — the FK check
                            prints how far off it is)

Speed is the executor's own ``traj_speed``, same as ``run_auto``; trajectories are
resampled to cuRobo's interpolated density so that speed means the same thing.

Sequence:
    home (FR3_INIT, hand open)
      -> approach   plan FR3_INIT -> start wrist / arm_cfg[0], fingers at finger_cfg[0]
      -> paper lift arm follows the planned/loaded path, fingers replay finger_cfg
      -> dwell
      -> release    ramp hand back open
      -> retract    plan_js_to_init(end config -> FR3_INIT)

Prereq: franka arm daemon + inspire hand controller up (same as franka_run.py).

    # offline check first (no robot, needs no daemon)
    python src/execution/paper_lift_run.py --dry_run --viz --port 8080

    # on the franka PC
    python src/execution/paper_lift_run.py --viz
    python src/execution/paper_lift_run.py --save ~/paper_lift_raw

    # step through it by hand instead (paradex RobotGUIController, hold Start to
    # advance one waypoint at a time — like handeye/capture.py under manual control)
    python src/execution/paper_lift_run.py --gui --viz
    python src/execution/paper_lift_run.py --wrist_dz 0.01     # start 1cm higher
    python src/execution/paper_lift_run.py --arm npz           # npz joints verbatim
"""
import argparse
import os
import sys
import time

import numpy as np

# `python src/execution/paper_lift_run.py` puts only src/execution on sys.path, so
# the `src.execution.franka_executor` import below would fail. Same fix as
# paradex's src/util/robot/move_robot_gui.py.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from autodex.planner import GraspPlanner
from autodex.planner.obstacles import TABLE_CUBOID
from autodex.utils.robot_config import FR3_INIT, INSPIRE_INIT
from autodex.utils.conversion import cart2se3

NPZ_DEFAULT = os.path.expanduser("~/shared_data/grasp_mingi/paper_lift_trajectory.npz")
OPEN_WRIST_DEFAULT = os.path.expanduser("~/shared_data/grasp_mingi/paper_open_wrist_se3.npy")

# curobo hand-joint order for fr3_inspire.yml (must match npz joint_names)
HAND_JOINTS = ["right_thumb_1_joint", "right_thumb_2_joint", "right_index_1_joint",
               "right_middle_1_joint", "right_ring_1_joint", "right_little_1_joint"]

URDF_PATH = os.path.expanduser(
    "~/shared_data/AutoDex/content/assets/robot/fr3_inspire_description/fr3_inspire.urdf")

# Table top in robot frame, from the planner's table cuboid.
TABLE_SURFACE_Z = TABLE_CUBOID["pose"][2] + TABLE_CUBOID["dims"][2] / 2   # 0.040
N_LIFT_IK = 60          # waypoints solved along the +z lift (see ik_seq)


# ── data ─────────────────────────────────────────────────────────────────────

def load_traj(npz_path: str, open_wrist_path: str):
    d = np.load(npz_path, allow_pickle=True)
    arm = np.asarray(d["arm_cfg"], dtype=np.float64)         # (T, 7)
    fing = np.asarray(d["finger_cfg"], dtype=np.float64)     # (T, 6)
    wrist = np.asarray(d["wrist_se3"], dtype=np.float64)     # (T, 4, 4)
    contact = np.asarray(d["contact"])                       # (T, 5)
    names = [str(s) for s in np.asarray(d["joint_names"]).ravel()]
    lift_h = float(np.asarray(d["lift_height"]).ravel()[0])

    assert arm.shape[1] == 7, f"arm_cfg must be 7-DOF (FR3), got {arm.shape}"
    assert fing.shape[1] == 6, f"finger_cfg must be 6-DOF (inspire), got {fing.shape}"
    assert len(arm) == len(fing) == len(wrist) == len(contact), "step count mismatch"
    # A silent reorder here would send thumb angles to the pinky.
    assert names == HAND_JOINTS, \
        f"npz hand joint order != planner order\n  npz:     {names}\n  planner: {HAND_JOINTS}"

    open_wrist = np.load(open_wrist_path).astype(np.float64) if os.path.exists(open_wrist_path) else None
    if open_wrist is not None:
        dw = float(np.linalg.norm(open_wrist - wrist[0]))
        print(f"[paper_lift] open_wrist vs wrist_se3[0] diff={dw:.6f}"
              + ("" if dw < 1e-6 else "   <-- differs, using wrist_se3[0] for the FK check"))

    # Hand fully open at step 0 is what home() leaves the hand at; anything else
    # means the trajectory starts from a config the robot is not in.
    if not np.allclose(fing[0], INSPIRE_INIT, atol=1e-3):
        print(f"[paper_lift] WARNING: finger_cfg[0]={fing[0].round(4)} != INSPIRE_INIT "
              f"(hand open) — approach will pre-shape the hand to it")

    q = np.concatenate([arm, fing], axis=1)                  # (T, 13) full curobo DOF
    print(f"[paper_lift] {len(q)} steps, lift_height={lift_h:.3f}m, "
          f"wrist z {wrist[0][2, 3]:.4f} -> {wrist[-1][2, 3]:.4f} "
          f"(dz={wrist[-1][2, 3] - wrist[0][2, 3]:.4f})")
    print(f"[paper_lift] arm path length={np.abs(np.diff(arm, axis=0)).sum(0).round(4)} "
          f"(norm={np.linalg.norm(arm[-1] - arm[0]):.4f} rad)")
    n_c = contact.astype(int).sum(1)
    print(f"[paper_lift] table contacts/step: {n_c.tolist()} "
          f"(fingers leave the table from step {int(np.argmax(n_c < n_c[0]))})")
    return q, wrist, contact, lift_h


def densify(q: np.ndarray, n_out: int) -> np.ndarray:
    """Linear-interp the sparse (T, dof) waypoints to n_out samples.

    The npz has 13 waypoints; ``_follow`` treats its input as a dense trajectory
    sampled at ``traj_dt`` and differentiates it for the velocity feedforward, so
    feeding 13 points would ask for a 0.13s motion at absurd joint velocities."""
    q = np.asarray(q, dtype=np.float64)
    xs = np.linspace(0.0, 1.0, len(q))
    xt = np.linspace(0.0, 1.0, int(n_out))
    return np.stack([np.interp(xt, xs, q[:, j]) for j in range(q.shape[1])], axis=1)


def n_waypoints(arm: np.ndarray, per_step: float = 0.002) -> int:
    """Waypoint count giving ~``per_step`` rad of joint motion per traj_dt tick —
    the density cuRobo's interpolated plans have, so the same executor
    ``traj_speed`` yields the same real speed."""
    length = float(np.abs(np.diff(np.asarray(arm), axis=0)).max(axis=1).sum())
    return int(max(20, round(length / per_step)))


# ── checks ───────────────────────────────────────────────────────────────────

def sanity_check(planner, scene_cfg, q: np.ndarray, wrist: np.ndarray) -> float:
    """FK arm_cfg/finger_cfg through the planner's kinematics and (a) compare the
    wrist pose with the npz's ``wrist_se3`` — catches a frame/joint-order
    mismatch BEFORE moving — and (b) report each step's clearance above the table
    model. Returns the deepest table penetration (m, 0 if none)."""
    import torch
    from curobo.types.state import JointState
    if planner._motion_gen is None:
        planner.plan_js_to_init(scene_cfg, FR3_INIT, INSPIRE_INIT, FR3_INIT)  # world init
    kin = planner._motion_gen.kinematics
    dev = planner._tensor_args.device
    js = JointState.from_position(torch.tensor(q, dtype=torch.float32, device=dev))
    st = kin.get_state(js.position)

    pos = st.ee_position.detach().cpu().numpy()
    p_err = np.linalg.norm(pos - wrist[:, :3, 3], axis=1)
    print(f"[paper_lift] FK vs wrist_se3: pos err mean={p_err.mean() * 1e3:.2f}mm "
          f"max={p_err.max() * 1e3:.2f}mm")
    if p_err.max() > 5e-3:
        print("[paper_lift] WARNING: FK disagrees with wrist_se3 by >5mm — the npz "
              "wrist poses may be in a different frame. arm_cfg is executed as-is, "
              "so this only invalidates wrist-based reasoning (not the joint path).")

    return table_clearance(planner, q, label="npz arm_cfg")


def table_clearance(planner, q: np.ndarray, label: str) -> float:
    """Min collision-sphere clearance above the table top per step (m).

    Only spheres over the table's x/y footprint count — the FR3 base sits at
    z=0, below the table top, but outside its footprint. Returns the deepest
    penetration (0 if the whole path clears)."""
    import torch
    kin = planner._motion_gen.kinematics
    st = kin.get_state(torch.tensor(np.asarray(q, dtype=np.float32),
                                    device=planner._tensor_args.device))
    sp = st.link_spheres_tensor.detach().cpu().numpy()          # (T, S, 4) xyz + r
    cx, cy, _ = TABLE_CUBOID["pose"][:3]
    dx, dy, dz = TABLE_CUBOID["dims"]
    top = TABLE_CUBOID["pose"][2] + dz / 2
    over = ((np.abs(sp[:, :, 0] - cx) <= dx / 2)
            & (np.abs(sp[:, :, 1] - cy) <= dy / 2))
    bottom = np.where(over, sp[:, :, 2] - sp[:, :, 3], np.inf)  # ignore off-table spheres
    clear = bottom.min(axis=1) - top
    if len(clear) <= 20:
        print(f"[paper_lift] [{label}] table top z={top:.3f}; per-step sphere "
              f"clearance (mm): {(clear * 1e3).round(1).tolist()}")
    else:
        print(f"[paper_lift] [{label}] table top z={top:.3f}; sphere clearance "
              f"min={clear.min() * 1e3:.1f}mm at step {int(np.argmin(clear))}/{len(clear)}, "
              f"final={clear[-1] * 1e3:.1f}mm")
    pen = float(max(0.0, -clear.min()))
    if pen > 0:
        n_bad = int((clear < 0).sum())
        print(f"[paper_lift] [{label}] NOTE: {n_bad}/{len(clear)} steps sit up to "
              f"{pen * 1e3:.1f}mm below the table model. That is expected for this "
              f"trajectory (the fingers rest ON the surface — contact is 5/5 for the "
              f"first 10 steps), but it means the real surface height must match "
              f"TABLE_CUBOID (top z={top:.3f}): if the real table is higher, the arm "
              f"presses into it and the FR3 reflex trips. Raise --wrist_dz to back off.")
    return pen


def table_scene(margin: float) -> dict:
    """Table world for PLANNING, top surface lowered by ``margin``.

    The paper-lift trajectory presses the fingers onto the table by design
    (``contact`` is 5/5 for the first 10 steps), so its start config is a world
    collision for cuRobo and every plan touching it fails outright. Lowering the
    table top by a few mm lets the planner keep avoiding the table BODY while
    tolerating the intended surface contact. Execution is unaffected."""
    import copy
    tc = copy.deepcopy(TABLE_CUBOID)
    tc["pose"] = list(tc["pose"])
    tc["pose"][2] = float(tc["pose"][2]) - float(margin)
    return {"cuboid": {"table": tc}, "mesh": {}}


def plan_with_margin(fn, margins):
    """Run a planner call against progressively lower tables until it succeeds.
    Returns (result, margin_used) or (None, None)."""
    for m in margins:
        res = fn(table_scene(m))
        if res is not None:
            return res, m
        print(f"[paper_lift]   table_margin={m * 1e3:.0f}mm failed")
    return None, None


def ik_seq(planner, poses: np.ndarray, q_start_full: np.ndarray):
    """Per-pose IK along a wrist path, each solve seeded by the previous solution.

    ``plan_pose_constrained`` only pins the END pose — the path itself is
    ``plan_single_js`` joint interpolation, so the wrist ROTATES on the way up
    even when start and goal orientations are identical. Solving IK at every
    waypoint instead keeps the commanded orientation exactly fixed along the
    whole lift (this is also how the npz's ``arm_cfg`` was built).

    Fingers are copied from ``q_start_full`` (IK only moves the arm).
    Returns (T, dof) or None if any waypoint is unreachable."""
    import torch
    from curobo.types.math import Pose as _Pose
    from scipy.spatial.transform import Rotation as _R

    dev = planner._tensor_args.device
    B = planner.BATCH_SIZE
    n_arm = planner._n_arm
    q_prev = np.asarray(q_start_full, dtype=np.float32).copy()
    out = []
    for i, T in enumerate(poses):
        xyzw = _R.from_matrix(T[:3, :3]).as_quat()
        wxyz = np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float32)
        goal = _Pose(
            position=torch.tensor(T[:3, 3].astype(np.float32), device=dev
                                  ).unsqueeze(0).repeat(B, 1),
            quaternion=torch.tensor(wxyz, device=dev).unsqueeze(0).repeat(B, 1))
        retract = torch.tensor(q_prev, device=dev).unsqueeze(0).repeat(B, 1)
        seed = torch.tensor(q_prev, device=dev).unsqueeze(0).unsqueeze(0).repeat(B, 1, 1)
        res = planner._ik_solver.solve_batch(goal, retract_config=retract,
                                             seed_config=seed)
        succ = res.success.cpu().numpy().reshape(-1)
        if not bool(succ.any()):
            print(f"[paper_lift] lift IK failed at waypoint {i}/{len(poses)} "
                  f"(z={T[2, 3]:.4f})")
            return None
        sol = res.solution.cpu().numpy()
        if sol.ndim == 3:
            sol = sol[:, 0, :]
        # Closest branch to the previous config — cuRobo IK happily returns a
        # far elbow/wrist flip that would show up as a jump mid-lift.
        cands = [sol[k, :n_arm].copy() for k in range(len(succ)) if succ[k]]
        best = min(cands, key=lambda c: float(np.linalg.norm(c - q_prev[:n_arm])))
        q_prev = q_prev.copy()
        q_prev[:n_arm] = best
        out.append(q_prev.copy())
    traj = np.asarray(out, dtype=np.float64)
    jumps = np.abs(np.diff(traj[:, :n_arm], axis=0)).max(axis=1)
    print(f"[paper_lift] lift IK: {len(traj)} waypoints, max per-step joint jump "
          f"{jumps.max():.4f} rad")
    if jumps.max() > 0.15:
        print(f"[paper_lift] WARNING: joint jump {jumps.max():.3f} rad at waypoint "
              f"{int(np.argmax(jumps))} — IK switched branch mid-lift; inspect in --viz "
              f"before running")
    return traj


def build_post_lift(planner, q_end_full: np.ndarray, post_h: float, scene_cfg: dict):
    """Extra pure +z translation AFTER the paper lift, fingers held closed.

    Kept separate from the lift instead of just raising ``lift_height``: the npz's
    ``finger_cfg`` is replayed across the lift, so a longer lift would stretch the
    finger closing over the whole travel and the grasp would complete metres too
    late. Here the hand holds ``finger_cfg[-1]`` and only the wrist moves.

    Returns (T, dof) or None."""
    if post_h <= 0:
        return None
    from autodex.planner.planner import _to_curobo_world
    if planner._ik_solver is None:
        world = _to_curobo_world(scene_cfg)
        world = dict(world)
        world["mesh"] = {}
        planner._init_ik_solver(world)
    pos, quat = fk_wrist(planner, np.asarray(q_end_full)[None])
    T = np.eye(4)
    from scipy.spatial.transform import Rotation as _R
    T[:3, :3] = _R.from_quat([quat[0][1], quat[0][2], quat[0][3], quat[0][0]]).as_matrix()
    T[:3, 3] = pos[0]
    n = int(max(10, round(post_h / 0.001)))          # ~1mm per waypoint
    poses = np.tile(T, (n, 1, 1))
    poses[:, 2, 3] = T[2, 3] + np.linspace(0.0, post_h, n)
    print(f"[paper_lift] post-lift +{post_h:.3f}m z from {T[:3, 3].round(4)}, "
          f"fingers held closed")
    return ik_seq(planner, poses, np.asarray(q_end_full, dtype=np.float32))


def build_ik_path(planner, q: np.ndarray, wrist: np.ndarray, lift_h: float,
                  offset: np.ndarray, margins, n_lift: int = N_LIFT_IK):
    """Re-solve the arm from ``wrist_se3`` on THIS URDF.

    approach: FR3_INIT -> start wrist, collision-checked motion plan (the wrist
              is free to move however it likes on the way in).
    lift:     ``n_lift`` waypoints straight up +z, orientation and x/y FIXED,
              per-waypoint IK (see ``ik_seq``) — no mid-path wrist rotation.

    The approach plan holds the fingers at ``finger_cfg[0]``; the real finger
    motion is replayed on top at execution time. Fingers closing only SHRINKS the
    hand, so the open-hand plan is the conservative one to collision-check.

    Returns (approach_traj, lift_traj, wrist_start, margin_used) or Nones.
    """
    w0 = wrist[0].copy()
    w0[:3, 3] += offset
    start_full = np.concatenate([FR3_INIT, q[0, 7:]]).astype(np.float32)
    print(f"[paper_lift] IK target wrist pos={w0[:3, 3].round(4)} "
          f"(npz {wrist[0][:3, 3].round(4)} + offset {offset.round(4)})")

    approach, m = plan_with_margin(
        lambda sc: planner.plan_pose_constrained(
            start_full, w0, hold_vec_weight=[0, 0, 0, 0, 0, 0],
            scene_cfg=sc, include_obj_obstacle=False),
        margins)
    if approach is None:
        return None, None, w0, None
    print(f"[paper_lift] approach traj {approach.shape} (table_margin={m * 1e3:.0f}mm)")

    # Straight-up wrist path, orientation and x/y untouched. Start from the
    # approach's final arm config so the lift continues from where it landed.
    lift_start = np.concatenate([approach[-1, :7], q[0, 7:]]).astype(np.float32)
    poses = np.tile(w0, (n_lift, 1, 1))
    poses[:, 2, 3] = w0[2, 3] + np.linspace(0.0, lift_h, n_lift)
    lift = ik_seq(planner, poses, lift_start)
    if lift is None:
        return approach, None, w0, m
    print(f"[paper_lift] lift +{lift_h:.3f}m z, orientation fixed "
          f"(per-waypoint IK, no plan_single_js)")
    return approach, lift, w0, m


def fk_positions(planner, q: np.ndarray) -> np.ndarray:
    """Wrist (ee_link) positions along a full-DOF trajectory, this URDF's FK."""
    return fk_wrist(planner, q)[0]


def fk_wrist(planner, q: np.ndarray):
    """(positions (T,3), quaternions wxyz (T,4)) of the wrist along a traj."""
    import torch
    kin = planner._motion_gen.kinematics
    st = kin.get_state(torch.tensor(np.asarray(q, dtype=np.float32),
                                    device=planner._tensor_args.device))
    return (st.ee_position.detach().cpu().numpy(),
            st.ee_quaternion.detach().cpu().numpy())


def report_lift_straightness(planner, lift_full: np.ndarray):
    """Confirm the executed lift really is a straight, non-rotating +z motion."""
    pos, quat = fk_wrist(planner, lift_full)
    dxy = np.linalg.norm(pos[:, :2] - pos[0, :2], axis=1)
    # angle between each step's wrist orientation and the first one
    dot = np.abs(np.clip(quat @ quat[0], -1.0, 1.0))
    ang = np.rad2deg(2.0 * np.arccos(dot))
    print(f"[paper_lift] lift check: dz={pos[-1, 2] - pos[0, 2]:.4f}m, "
          f"max xy drift={dxy.max() * 1e3:.2f}mm, "
          f"max wrist rotation={ang.max():.2f}deg")
    if ang.max() > 2.0 or dxy.max() > 3e-3:
        print("[paper_lift] WARNING: the lift is not a clean straight-up motion "
              "(see numbers above)")


def start_viz(q_full: np.ndarray, ee_pos, port: int, split: int, label: str,
              split_post=None):
    """Launch a non-blocking viser view of the whole executed path.

    Slider scrubs the concatenated approach+lift path, a play toggle animates it,
    the wrist path is drawn as a polyline (green = approach, red = paper lift) and
    the table model is shown so the fingers-on-surface contact is visible."""
    import threading
    import time as _time
    import trimesh
    import viser
    import yourdfpy

    robot = yourdfpy.URDF.load(URDF_PATH, load_meshes=True,
                               build_collision_scene_graph=False)
    aj = robot.actuated_joint_names
    n_dof = min(len(aj), q_full.shape[1])
    n = len(q_full)

    server = viser.ViserServer(port=port)
    server.scene.add_mesh_trimesh("/table", trimesh.creation.box(
        extents=np.array(TABLE_CUBOID["dims"], float)).apply_transform(
        cart2se3(np.array(TABLE_CUBOID["pose"], float))))
    if ee_pos is not None:
        ee_pos = np.asarray(ee_pos, dtype=np.float32)
        if split > 1:
            server.scene.add_spline_catmull_rom(
                "/path/approach", ee_pos[:split], color=(0.1, 0.8, 0.2), line_width=3.0)
        lift_end = n if split_post is None else split_post
        if lift_end - split > 1:
            server.scene.add_spline_catmull_rom(
                "/path/lift", ee_pos[split:lift_end], color=(0.9, 0.2, 0.2),
                line_width=4.0)
        if split_post is not None and n - split_post > 1:
            server.scene.add_spline_catmull_rom(
                "/path/post_lift", ee_pos[split_post:], color=(0.2, 0.4, 1.0),
                line_width=4.0)
        server.scene.add_point_cloud(
            "/path/pts", ee_pos, colors=np.tile(np.array([[1.0, 1.0, 0.2]]), (n, 1)),
            point_size=0.003)

    def show(k):
        robot.update_cfg({aj[i]: float(q_full[k, i]) for i in range(n_dof)})
        server.scene.add_mesh_trimesh("/robot", robot.scene.to_geometry())

    sl = server.gui.add_slider("step", min=0, max=n - 1, step=1, initial_value=0)
    seg = server.gui.add_text("segment", initial_value="approach", disabled=True)
    play = server.gui.add_checkbox("play", initial_value=False)
    rate = server.gui.add_slider("steps/frame", min=1, max=20, step=1, initial_value=4)

    def refresh(k):
        show(k)
        if k < split:
            seg.value = "approach"
        elif split_post is None or k < split_post:
            seg.value = "paper lift (fingers closing)"
        else:
            seg.value = "post-lift (grasp held, +z only)"

    sl.on_update(lambda _: refresh(int(sl.value)))
    refresh(0)

    def _loop():
        while True:
            if play.value:
                sl.value = 0 if sl.value >= n - 1 else min(n - 1, sl.value + int(rate.value))
                refresh(int(sl.value))
            _time.sleep(0.05)

    threading.Thread(target=_loop, daemon=True).start()
    lift_end = n if split_post is None else split_post
    seg_txt = (f"lift 0..{lift_end - 1}" if split == 0 else
               f"approach 0..{split - 1}, lift {split}..{lift_end - 1}")
    if split_post is not None and split_post < n:
        seg_txt += f", post-lift {split_post}..{n - 1}"
    # viser falls back to the next free port when the requested one is taken, so
    # report the port it ACTUALLY bound — not the one we asked for.
    try:
        port = server.get_port()
    except Exception:
        pass
    print(f"[paper_lift] viz on http://localhost:{port} — {n} steps ({seg_txt}) [{label}]")
    return server


# ── plan cache ───────────────────────────────────────────────────────────────

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "outputs", "paper_lift", "plan_cache")


def cache_key(args, lift_h: float) -> str:
    """Hash of everything the plan depends on. Any change (npz contents, arm mode,
    wrist offset, lift height, table margins) misses the cache and replans."""
    import hashlib
    h = hashlib.sha1()
    with open(args.npz, "rb") as f:
        h.update(f.read())
    for v in (args.arm, args.wrist_dx, args.wrist_dy, args.wrist_dz, lift_h,
              args.post_lift, args.table_margin, args.table_margin_max, N_LIFT_IK):
        h.update(repr(v).encode())
    # A URDF / robot-config change invalidates the IK solutions too.
    for p in (URDF_PATH,
              os.path.expanduser("~/shared_data/AutoDex/content/configs/robot/"
                                 "fr3_inspire.yml")):
        try:
            st = os.stat(p)
            h.update(f"{p}:{st.st_mtime_ns}:{st.st_size}".encode())
        except OSError:
            h.update(f"{p}:missing".encode())
    return h.hexdigest()[:16]


def cache_load(key: str):
    path = os.path.join(CACHE_DIR, f"{key}.npz")
    if not os.path.exists(path):
        return None
    try:
        d = np.load(path, allow_pickle=False)
    except Exception as e:
        print(f"[paper_lift] cache unreadable ({e!r}) — replanning")
        return None
    out = {k: d[k] for k in d.files}
    report = str(out.pop("report").item()) if "report" in out else ""
    print(f"[paper_lift] cache HIT {path} — planner NOT started "
          f"(--no_cache to replan)")
    for line in report.splitlines():
        if line:
            print(f"(cached) {line}")
    return out


def cache_save(key: str, data: dict, report: str):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f"{key}.npz")
    np.savez(path, report=np.array(report), **{k: v for k, v in data.items()
                                               if v is not None})
    print(f"[paper_lift] cached plan -> {path}")


# ── GUI execution (paradex RobotGUIController) ───────────────────────────────

def subsample(traj: np.ndarray, max_step: float) -> np.ndarray:
    """Keep waypoints so no joint moves more than ``max_step`` rad between them.
    The last waypoint is always kept."""
    traj = np.asarray(traj, dtype=np.float64)
    keep = [0]
    for i in range(1, len(traj)):
        if np.abs(traj[i] - traj[keep[-1]]).max() >= max_step:
            keep.append(i)
    if keep[-1] != len(traj) - 1:
        keep.append(len(traj) - 1)
    return traj[keep]


def run_gui(args, approach, lift_arm, fing, retract, hand_name: str, n_grasp: int):
    """Step the same path through paradex's ``RobotGUIController``.

    Point-to-point waypoints with a dead-man Start button (hold to advance, release
    to stop) — the same way ``src/calibration/handeye/capture.py`` replays its taught
    poses, but under manual control. Use this instead of the velocity-streaming
    ``FrankaExecutor`` when you want to inspect every step on the real robot.

    The lift is enqueued as the npz's OWN 13 steps so each waypoint carries its
    exact ``finger_cfg`` row."""
    from paradex.io.robot_controller import get_arm, get_hand
    from paradex.io.robot_controller.gui_controller import RobotGUIController
    from autodex.executor.real import _convert_inspire

    # Cartesian jogging is optional — it needs a URDF, and paradex.robot pulls in
    # pinocchio, which is not installed in every env. Without it the GUI still has
    # the waypoint queue and joint jogging, which is all this script uses.
    urdf_path = eef_link = None
    try:
        from paradex.robot.utils import get_robot_urdf_path
        from paradex.calibration.utils import EEF_LINK
        urdf_path = get_robot_urdf_path(arm_name="franka")
        eef_link = EEF_LINK.get("franka")
    except Exception as e:
        print(f"[paper_lift] cartesian jog disabled ({e!r}) — joint jog + waypoints only")

    arm = get_arm("franka")
    hand = get_hand(hand_name)
    if arm.get_data() is None:
        print("[paper_lift] no franka state — start the daemon "
              "(./cpp/franka_daemon/run_daemon.sh) and check the mode")
        return
    if hasattr(arm, "error_recovery"):
        arm.error_recovery()

    rgc = RobotGUIController(arm, hand_controller=hand,
                             urdf_path=urdf_path, eef_link=eef_link)

    open_cmd = _convert_inspire(fing[0])
    closed_cmd = _convert_inspire(fing[-1])
    n_added = 0
    if approach is not None:
        for i, qa in enumerate(subsample(approach[:, :7], args.gui_step)):
            rgc.add_waypoint(f"approach {i}", "joint", target=qa, hand_qpos=open_cmd)
            n_added += 1
    # Paper lift: the npz's own steps, fingers verbatim per step.
    lift_wp = densify(lift_arm[:n_grasp], len(fing))
    for i in range(len(fing)):
        rgc.add_waypoint(f"lift {i}/{len(fing) - 1}", "joint", target=lift_wp[i],
                         hand_qpos=_convert_inspire(fing[i]))
        n_added += 1
    # Post-lift: straight up with the grasp held.
    last = lift_wp[-1]
    for i, qa in enumerate(subsample(lift_arm[n_grasp:], args.gui_step / 4.0)):
        rgc.add_waypoint(f"post-lift {i}", "joint", target=qa, hand_qpos=closed_cmd)
        last = qa
        n_added += 1
    rgc.add_waypoint("release", "joint", target=last,
                     hand_qpos=_convert_inspire(INSPIRE_INIT))
    n_added += 1
    if retract is not None:
        for i, qa in enumerate(subsample(retract[:, :7], args.gui_step)):
            rgc.add_waypoint(f"retract {i}", "joint", target=qa, hand_qpos=open_cmd)
            n_added += 1

    print(f"[paper_lift] GUI: {n_added} waypoints queued "
          f"(approach step<={args.gui_step:.2f}rad, lift = the npz's {len(fing)} steps, "
          f"then post-lift + release + retract). HOLD Start to advance, release to stop.")
    rgc.run()
    for ctrl in (arm, hand):
        try:
            ctrl.end()
        except Exception as e:
            print(f"[paper_lift] {type(ctrl).__name__}.end() failed: {e!r}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=NPZ_DEFAULT)
    ap.add_argument("--open_wrist", default=OPEN_WRIST_DEFAULT)
    ap.add_argument("--hand", default="inspire", choices=["inspire", "inspire_left"])
    ap.add_argument("--arm", default="ik", choices=["ik", "npz"],
                    help="ik = re-solve the arm from wrist_se3 on this repo's URDF "
                         "(default); npz = execute arm_cfg verbatim")
    # Offsets applied to the npz start wrist before IK (--arm ik only). Defaults set
    # by the user for this rig; pass 0 to use the npz pose as-is.
    ap.add_argument("--wrist_dx", type=float, default=0.2,
                    help="forward offset of the grasp, m (npz x=0.4 -> 0.6)")
    ap.add_argument("--wrist_dy", type=float, default=0.0)
    ap.add_argument("--wrist_dz", type=float, default=0.025,
                    help="height offset of the grasp, m (npz z=0.1048 -> 0.1298)")
    ap.add_argument("--lift_height", type=float, default=None,
                    help="override the npz lift_height (--arm ik only)")
    ap.add_argument("--post_lift", type=float, default=0.20,
                    help="extra pure +z translation after the paper lift, m, with the "
                         "fingers held at finger_cfg[-1]. 0 disables.")
    ap.add_argument("--table_margin", type=float, default=0.005,
                    help="first table-lowering margin tried when planning; escalated "
                         "up to --table_margin_max as needed")
    ap.add_argument("--table_margin_max", type=float, default=0.04)
    ap.add_argument("--gui", action="store_true",
                    help="drive the path through paradex's RobotGUIController "
                         "(hold-to-move waypoints) instead of FrankaExecutor")
    ap.add_argument("--gui_step", type=float, default=0.1,
                    help="--gui: max joint motion (rad) per approach/retract waypoint")
    ap.add_argument("--follow_tol", type=float, default=None,
                    help="override executor.follow_tol (default: the executor's own, "
                         "same as run_auto). _follow indexes the hand by how far the "
                         "ARM has measurably progressed, so lowering it keeps the "
                         "fingers from closing ahead of the wrist.")
    ap.add_argument("--dwell", type=float, default=1.5, help="hold seconds after lift")
    ap.add_argument("--save", default=None, help="dir for arm/hand recording")
    ap.add_argument("--no_release", action="store_true", help="keep holding at the end")
    ap.add_argument("--dry_run", action="store_true", help="no robot, no daemon needed")
    ap.add_argument("--no_plan", action="store_true",
                    help="dry_run only: skip the planner (no CUDA needed)")
    ap.add_argument("--no_cache", action="store_true",
                    help="replan even when a cached plan for these exact inputs exists")
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    q, wrist, contact, lift_h = load_traj(args.npz, args.open_wrist)
    if args.lift_height is not None:
        lift_h = float(args.lift_height)
    margins = [args.table_margin]
    while margins[-1] + 0.005 <= args.table_margin_max + 1e-9:
        margins.append(round(margins[-1] + 0.005, 4))

    approach = None
    lift = None
    retract = None
    lift_arm = fing_dense = ee = None
    planner = None

    # Planning takes minutes (cuRobo init + CUDA graph build) and is fully
    # determined by the inputs, so the result — including the FK/clearance report
    # and the wrist path for the viewer — is cached. A cache hit never touches
    # cuRobo at all.
    key = None if args.no_plan else cache_key(args, lift_h)
    cached = None if (key is None or args.no_cache) else cache_load(key)

    n_grasp = None      # waypoints of lift_arm belonging to the paper lift itself
    if cached is not None:
        approach = cached.get("approach")
        lift_arm = cached["lift_arm"]
        fing_dense = cached["fing_dense"]
        retract = cached.get("retract")
        ee = cached.get("ee")
        n_grasp = int(cached["n_grasp"].item()) if "n_grasp" in cached else len(lift_arm)
    elif not (args.dry_run and args.no_plan):
        planner = GraspPlanner(hand="fr3_inspire")
        sanity_check(planner, table_scene(0.0), q, wrist)

        if args.arm == "ik":
            offset = np.array([args.wrist_dx, args.wrist_dy, args.wrist_dz], float)
            print("[paper_lift] re-solving the arm from wrist_se3 on this URDF ...")
            approach, lift, w0, m = build_ik_path(planner, q, wrist, lift_h,
                                                  offset, margins)
            if approach is None or lift is None:
                print("[paper_lift] IK path planning FAILED — try a larger --wrist_dz "
                      "or --table_margin_max, or --arm npz to run the npz joints "
                      "verbatim. Aborting.")
                return
            # ik_seq returns one waypoint per 1mm of lift; resample to the
            # density cuRobo's interpolated plans have so the executor's own
            # traj_speed gives the same real speed as run_auto.
            lift_arm = densify(lift[:, :7], n_waypoints(lift[:, :7]))
            end_full = lift[-1]
            end_arm = lift[-1, :7]
        else:
            print("[paper_lift] planning approach FR3_INIT -> arm_cfg[0] ...")
            approach, m = plan_with_margin(
                lambda sc: planner.plan_js_to_init(
                    sc, start_arm_qpos=FR3_INIT, start_hand_qpos=q[0, 7:],
                    goal_arm_qpos=q[0, :7]),
                margins)
            if approach is None:
                print("[paper_lift] approach plan FAILED — aborting (no blind move to "
                      "arm_cfg[0]). Try --arm ik.")
                return
            print(f"[paper_lift] approach traj {approach.shape} "
                  f"(table_margin={m * 1e3:.0f}mm)")
            # The npz has 13 waypoints; _follow differentiates its input as a
            # traj_dt=0.01 path, so resample to the density cuRobo would have
            # produced for the same joint-space length (~2mrad/waypoint).
            lift_arm = densify(q[:, :7], n_waypoints(q[:, :7]))
            end_full = q[-1]
            end_arm = q[-1, :7]

        # Fingers are ALWAYS the npz values (never re-solved), resampled onto the
        # arm trajectory's timeline so they close over the lift.
        fing_dense = densify(q[:, 7:], len(lift_arm))
        n_grasp = len(lift_arm)

        # Extra straight-up travel with the grasp held, appended to the same
        # segment so the executor / viewer / clearance check see one path.
        post = build_post_lift(planner, end_full, args.post_lift, table_scene(0.0))
        if post is not None:
            post_arm = densify(post[:, :7], n_waypoints(post[:, :7]))
            lift_arm = np.vstack([lift_arm, post_arm])
            fing_dense = np.vstack([fing_dense,
                                    np.tile(q[-1, 7:], (len(post_arm), 1))])
            end_arm = post[-1, :7]
        elif args.post_lift > 0:
            print("[paper_lift] post-lift IK failed — continuing without it")

        print(f"[paper_lift] lift segment {len(lift_arm)} waypoints at traj_dt=0.01 "
              f"({n_grasp} paper lift + {len(lift_arm) - n_grasp} post-lift), "
              f"followed at executor traj_speed (same as run_auto)")

        print("[paper_lift] planning retract -> FR3_INIT ...")
        retract, _ = plan_with_margin(
            lambda sc: planner.plan_js_to_init(
                sc, start_arm_qpos=end_arm, start_hand_qpos=INSPIRE_INIT),
            margins)
        if retract is None:
            print("[paper_lift] retract plan failed — will fall back to home() "
                  "(free-space blocking move)")

    # Full executed path (approach with fingers held at finger_cfg[0], then the
    # lift with the npz finger values replayed) — this is exactly what goes to
    # the robot, so it is also what the viewer shows.
    if approach is None:                              # --no_plan: npz joints only
        split = 0
        n_np = n_waypoints(q[:, :7])
        q_full = np.hstack([densify(q[:, :7], n_np), densify(q[:, 7:], n_np)])
    else:
        split = len(approach)
        q_full = np.vstack([
            np.hstack([approach[:, :7], np.tile(q[0, 7:], (len(approach), 1))]),
            np.hstack([lift_arm, fing_dense])])

    if planner is not None:
        # Freshly planned: run the checks, capture their text and the wrist path so
        # a cached run can reprint them without cuRobo.
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            table_clearance(planner, q_full, label="executed path")
            report_lift_straightness(planner, q_full[split:])
        report = buf.getvalue()
        print(report, end="")
        ee = fk_positions(planner, q_full)
        cache_save(key, {"approach": approach, "lift_arm": lift_arm,
                         "fing_dense": fing_dense, "retract": retract, "ee": ee,
                         "n_grasp": np.array(n_grasp)},
                   report)

    if args.viz:
        if ee is not None:
            print(f"[paper_lift] executed wrist path: "
                  f"start={ee[split][:3].round(4)} end={ee[-1][:3].round(4)} "
                  f"(dz={ee[-1][2] - ee[split][2]:.4f})")
        start_viz(q_full, ee, args.port, split,
                  "dry run" if args.dry_run else "confirm to execute",
                  split_post=(None if n_grasp is None or n_grasp >= len(lift_arm)
                              else split + n_grasp))

    if args.dry_run:
        print("[paper_lift] dry run — no robot commands sent")
        if args.viz:
            print("[paper_lift] Ctrl-C to exit the viewer")
            while True:
                time.sleep(1.0)
        return

    if args.viz:
        if input("[paper_lift] execute this path on the robot? [y/N] ").strip().lower() != "y":
            print("[paper_lift] aborted by user")
            return

    if args.gui:
        run_gui(args, approach, lift_arm, q[:, 7:], retract, args.hand,
                n_grasp if n_grasp is not None else len(lift_arm))
        return

    from src.execution.franka_executor import FrankaExecutor, ContactAbort

    executor = FrankaExecutor(hand_name=args.hand)
    if args.follow_tol is not None:
        executor.follow_tol = float(args.follow_tol)
    try:
        executor.home()                                  # FR3_INIT + hand open
        if args.save:
            executor.start_recording(args.save)

        # 1. approach — cuRobo traj is already dense at interpolation_dt=0.01
        executor._log("approach")
        open_cmd = executor._convert(q[0, 7:])
        executor._follow(approach[:, :7], np.tile(open_cmd, (len(approach), 1)),
                         abort_on_contact=True)

        # 2. paper grasp + lift — arm and fingers move together. The fingers press
        #    on the table by design (contact flags), so a reflex here is recovered,
        #    not treated as a wrong-pose abort.
        executor._log("paper_lift")
        hand_traj = np.array([executor._convert(f) for f in fing_dense])
        executor._follow(lift_arm, hand_traj, abort_on_contact=False)
        executor._last_hand_qpos = q[-1, 7:].copy()
        executor._log("paper_lift_done")

        if args.dwell > 0:
            print(f"[paper_lift] holding {args.dwell:.1f}s")
            time.sleep(args.dwell)

        # 3. release
        if not args.no_release:
            executor._log("release")
            executor._ramp_hand(executor._convert(INSPIRE_INIT), steps=60, step_dt=0.02)
            executor._last_hand_qpos = np.asarray(INSPIRE_INIT, dtype=np.float64).copy()
            time.sleep(0.5)

            # 4. retract
            executor._log("retract")
            if retract is not None:
                executor._follow(retract[:, :7],
                                 np.tile(executor._convert(INSPIRE_INIT), (len(retract), 1)))
            else:
                executor.home()
    except ContactAbort as e:
        print(f"[paper_lift] ABORTED: {e}")
    finally:
        if args.save:
            executor.stop_recording()
        executor.shutdown()
        print("[paper_lift] done")


if __name__ == "__main__":
    main()
