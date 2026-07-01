#!/usr/bin/env python3
"""Smoke test: GraspPlanner.plan_with_seed plumbing.

Plan one grasp the normal way (IK -> plan_single_js) to get a canonical seed,
then feed that seed back into plan_with_seed for (a) the SAME goal [sanity] and
(b) a slightly perturbed goal [does a raw, unadjusted seed already help?].
"""
import sys, os
sys.path.insert(0, "/home/mingi/AutoDex")
import numpy as np, torch, json
from pathlib import Path
import src.visualization.exp as exp
from autodex.planner import GraspPlanner
from autodex.planner.planner import _to_curobo_world, _to_curobo_pose, _snap_joint6
from autodex.utils.path import project_dir
from src.grasp_generation.order.compute_order import load_grasp_data

OBJ, HAND, POSE = "pepsi", "inspire_left", 2
planner = GraspPlanner(hand=HAND)
cand_root = str(Path(project_dir) / "candidates" / HAND / exp.GRASP_VERSION)
info, wrist_obj, preg = load_grasp_data(cand_root, OBJ)
gmeta = [(g[1], g[2], g[3], exp.scene_pose_idx(OBJ, g[1], g[2])) for g in info]
tab = exp.load_tabletop_poses(OBJ)

# pick first grasp belonging to POSE
gi = next(i for i, m in enumerate(gmeta) if m[3] == POSE)
print(f"grasp idx {gi}  pose {POSE}")

def wrist_at(x, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    Tp = np.eye(4)
    Tp[:3, :3] = Rz @ tab[POSE][:3, :3]
    Tp[0, 3] = x; Tp[2, 3] = tab[POSE][2, 3] + exp.TABLE_SURFACE_Z
    return Tp @ wrist_obj[gi]

world = _to_curobo_world({"mesh": {}, "cuboid": {"table": exp.TABLE_CUBOID}})
planner._init_motion_gen(world)
planner._init_ik_solver(world)
dev = planner._tensor_args.device

def ik(Twrist):
    goal = _to_curobo_pose(Twrist[None], dev)
    retract = torch.tensor(planner._init_state, dtype=torch.float32, device=dev).unsqueeze(0)
    r = planner._ik_solver.solve_batch(goal, retract_config=retract)
    if not bool(r.success.view(-1)[0]):
        return None
    q = r.solution.cpu().numpy().reshape(-1)[:6].copy()
    q[5] = _snap_joint6(q[5], planner._init_state[5])
    full = planner._init_state.copy(); full[:6] = q; full[6:] = preg[gi]
    return full

# canonical placement (grid cell)
Tc = wrist_at(0.45, 0.0)
qc = ik(Tc)
print("canonical IK:", None if qc is None else "ok")
ok, seed = planner._refine_fingers(planner._init_state, qc)
print(f"canonical plan_single_js: ok={ok} seed_shape={None if seed is None else seed.shape}")
assert ok

# (a) same goal, seeded
ok, traj, st = planner.plan_with_seed(qc, seed)
print(f"(a) same-goal plan_with_seed: ok={ok} solve_time={st:.3f} traj={None if traj is None else traj.shape}")

def wrap(d):  # wrap angle diffs to [-pi, pi] (avoid 2pi jumps)
    return (d + np.pi) % (2 * np.pi) - np.pi

def adjust_seed(seed, q_start, q_goal):
    """Joint-space endpoint re-anchor: pin seed[0]->q_start, seed[-1]->q_goal,
    ramp the corrections linearly across steps; keep interior shape."""
    H = len(seed)
    d0 = wrap(q_start - seed[0])
    d1 = wrap(q_goal - seed[-1])
    a = np.linspace(0, 1, H)[:, None]
    return seed + (1 - a) * d0 + a * d1

# (b) perturbed goals (off-grid): RAW seed vs ADJUSTED seed
for dx, dyaw in [(0.0, 5), (0.0, 10), (0.02, 0), (0.02, 10), (-0.02, 15)]:
    Tp = wrist_at(0.45 + dx, np.radians(dyaw))
    qp = ik(Tp)
    if qp is None:
        print(f"(b) dx={dx:+.2f} dyaw={dyaw:>3}: IK unreachable")
        continue
    ok_raw, _, _ = planner.plan_with_seed(qp, seed)
    ok_adj, _, st = planner.plan_with_seed(qp, adjust_seed(seed, planner._init_state, qp))
    print(f"(b) dx={dx:+.2f} dyaw={dyaw:>3}: RAW={ok_raw}  ADJUSTED={ok_adj} (t={st:.3f})")
