#!/usr/bin/env python3
"""Offline replay of a recorded trial in viser.

Reads ``raw/arm/position.npy`` + ``raw/hand/position.npy`` (sensor streams)
from a trial directory, syncs them onto the arm timeline, converts the hand
controller units (0-1000, controller-order) back to radians (cuRobo-order),
and feeds the 12-DOF trajectory to ``ViserViewer.add_traj`` so the user can
scrub through the actual motion and see how/when fingers collide.

Usage:
    python src/experiment/reset/replay_trial.py \\
        --trial_dir ~/shared_data/AutoDex/experiment/reset_test/reorient_drop/inspire_left/pepsi/20260524_112431

    # latest trial under <obj>
    python src/experiment/reset/replay_trial.py --obj pepsi
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import trimesh
import yourdfpy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from paradex.visualization.visualizer.viser import ViserViewer  # noqa: E402


URDF_BY_HAND = {
    "inspire_left": Path.home() / "shared_data" / "AutoDex" / "content"
                    / "assets" / "robot" / "inspire_left_description"
                    / "xarm_inspire_left.urdf",
    "inspire":      Path.home() / "shared_data" / "AutoDex" / "content"
                    / "assets" / "robot" / "inspire_description"
                    / "xarm_inspire.urdf",
    "allegro":      Path.home() / "shared_data" / "AutoDex" / "content"
                    / "assets" / "robot" / "allegro_description"
                    / "xarm_allegro.urdf",
}

ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]

# cuRobo hand-joint order (matches xarm_inspire[_left].yml cspace.joint_names).
HAND_JOINTS_INSPIRE_LEFT = [
    "left_thumb_1_joint", "left_thumb_2_joint",
    "left_index_1_joint", "left_middle_1_joint",
    "left_ring_1_joint",  "left_little_1_joint",
]

# Per-joint angle limits in cuRobo order (matches _convert_inspire).
INSPIRE_LIMITS = np.array([1.15, 0.55, 1.6, 1.6, 1.6, 1.6])

EE_LINK = "base_link"   # hand base — wrist for FK
MESH_BASE = Path.home() / "shared_data" / "AutoDex" / "object" / "paradex"

SPHERES_BY_HAND = {
    "inspire_left": Path.home() / "shared_data" / "AutoDex" / "content"
                    / "configs" / "robot" / "spheres" / "xarm_inspire_left.yml",
    "inspire":      Path.home() / "shared_data" / "AutoDex" / "content"
                    / "configs" / "robot" / "spheres" / "xarm_inspire.yml",
    "allegro":      Path.home() / "shared_data" / "AutoDex" / "content"
                    / "configs" / "robot" / "spheres" / "xarm_allegro.yml",
}
HAND_LINK_PREFIXES = ("base_link", "left_", "right_", "thumb_", "index_",
                      "middle_", "ring_", "little_", "pinky_",
                      "palm_link", "ff_", "mf_", "rf_", "th_")


def _load_spheres(hand: str, scope: str) -> dict:
    """Returns {link_name: list[(center_3, radius)]} filtered by scope."""
    import yaml
    path = SPHERES_BY_HAND[hand]
    with open(path) as f:
        data = yaml.safe_load(f)["collision_spheres"]
    out = {}
    for link, spheres in data.items():
        if scope == "hand" and not any(link.startswith(p) for p in HAND_LINK_PREFIXES):
            continue
        out[link] = [(np.array(s["center"], dtype=np.float64),
                      float(s["radius"])) for s in spheres]
    return out


def _compute_sphere_poses(urdf: yourdfpy.URDF,
                          full_q: np.ndarray,
                          spheres: dict,
                          arm_names: list[str],
                          hand_names: list[str]) -> dict:
    """For each (link, sphere_idx), return (T_steps, 4, 4) translation-only
    transforms (identity rotation; sphere is isotropic)."""
    T = len(full_q)
    items = [(link, i, c, r)
             for link, slist in spheres.items()
             for i, (c, r) in enumerate(slist)]
    poses = {f"sph_{link}_{i:02d}": np.tile(np.eye(4), (T, 1, 1))
             for link, i, _, _ in items}
    for t, q in enumerate(full_q):
        cfg = {**{arm_names[i]: float(q[i]) for i in range(6)},
               **{hand_names[i]: float(q[6 + i]) for i in range(6)}}
        urdf.update_cfg(cfg)
        link_T_cache = {}
        for link, i, c, _ in items:
            if link not in link_T_cache:
                try:
                    link_T_cache[link] = urdf.get_transform(link, urdf.base_link)
                except Exception:
                    link_T_cache[link] = np.eye(4)
            T_link = link_T_cache[link]
            world_c = T_link[:3, :3] @ c + T_link[:3, 3]
            poses[f"sph_{link}_{i:02d}"][t, :3, 3] = world_c
    radii = {f"sph_{link}_{i:02d}": r for link, i, _, r in items}
    return poses, radii


def sensor_units_to_rad(s: np.ndarray) -> np.ndarray:
    """Inverse of ``_convert_inspire``.

    Input  ``s``: (..., 6) int/float, controller-order
        [pinky, ring, middle, index, thumb_pitch, thumb_yaw], 0=closed, 1000=open.
    Output    : (..., 6) float, cuRobo-order
        [thumb_yaw, thumb_pitch, index, middle, ring, pinky], radians.
    """
    s = np.clip(np.asarray(s, dtype=np.float64), 0.0, 1000.0)
    norm = 1.0 - s / 1000.0
    q = np.zeros_like(norm)
    q[..., 0] = norm[..., 5] * INSPIRE_LIMITS[0]   # thumb_yaw
    q[..., 1] = norm[..., 4] * INSPIRE_LIMITS[1]   # thumb_pitch
    q[..., 2] = norm[..., 3] * INSPIRE_LIMITS[2]   # index
    q[..., 3] = norm[..., 2] * INSPIRE_LIMITS[3]   # middle
    q[..., 4] = norm[..., 1] * INSPIRE_LIMITS[4]   # ring
    q[..., 5] = norm[..., 0] * INSPIRE_LIMITS[5]   # pinky (little)
    return q


def _resolve_trial_dir(args) -> Path:
    if args.trial_dir:
        p = Path(args.trial_dir).expanduser()
        if not p.exists():
            sys.exit(f"trial dir not found: {p}")
        return p
    if not args.obj:
        sys.exit("provide either --trial_dir or --obj")
    root = (Path.home() / "shared_data" / "AutoDex" / "experiment"
            / args.exp_name / args.hand / args.obj)
    if not root.exists():
        sys.exit(f"obj experiment dir not found: {root}")
    cands = sorted([p for p in root.iterdir() if p.is_dir()
                    and p.name[:1].isdigit()])
    if not cands:
        sys.exit(f"no trial subdirectories under {root}")
    return cands[-1]


def _infer_obj_name(trial_dir: Path, override: str | None) -> str:
    if override:
        return override
    # Layout: .../experiment/{exp}/{hand}/{obj}/{trial_ts}
    return trial_dir.parent.name


def _fk_wrist_batch(urdf: yourdfpy.URDF,
                    full_q: np.ndarray,
                    arm_names: list[str],
                    hand_names: list[str]) -> np.ndarray:
    out = np.tile(np.eye(4), (len(full_q), 1, 1))
    for t, q in enumerate(full_q):
        cfg = {**{arm_names[i]: float(q[i]) for i in range(6)},
               **{hand_names[i]: float(q[6 + i]) for i in range(6)}}
        urdf.update_cfg(cfg)
        out[t] = urdf.get_transform(EE_LINK, urdf.base_link)
    return out


def _collision_check_traj(full_q: np.ndarray,
                          scene_cfg_path: Path,
                          hand: str,
                          grasp_step: int,
                          chunk: int = 256) -> dict:
    """Run cuRobo collision check on every step of the trajectory.

    Splits into PRE/POST grasp:
      - pre  : object IS in the world (we want to know when the hand first
               touches it — collision count is the "approach contact" signal)
      - post : object removed from world (hand+obj move rigidly; only useful
               for self-collision and obstacle collision such as the table)

    Returns dict with keys: ``d_world``, ``d_self``, ``collide`` (each (T,)).
    """
    import torch
    from curobo.geom.types import WorldConfig
    from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig

    from autodex.planner import GraspPlanner
    from autodex.planner.planner import _to_curobo_world

    with open(scene_cfg_path) as f:
        scene_cfg = json.load(f)
    world_with_obj = _to_curobo_world(scene_cfg)
    scene_no_obj = {"mesh": {}, "cuboid": dict(scene_cfg.get("cuboid", {}))}
    world_no_obj = _to_curobo_world(scene_no_obj)

    print(f"[collide] warming up cuRobo for {hand}...")
    planner = GraspPlanner(hand=hand)
    device = planner._tensor_args.device

    def _run(world_cfg, q_arr):
        cfg = RobotWorldConfig.load_from_config(
            planner._robot_cfg,
            WorldConfig.from_dict(world_cfg),
            collision_activation_distance=0.0,
            tensor_args=planner._tensor_args,
        )
        rw = RobotWorld(cfg)
        d_w = np.zeros(len(q_arr), dtype=np.float32)
        d_s = np.zeros(len(q_arr), dtype=np.float32)
        for s in range(0, len(q_arr), chunk):
            e = min(s + chunk, len(q_arr))
            q_t = torch.tensor(q_arr[s:e], dtype=torch.float32, device=device)
            dw, ds = rw.get_world_self_collision_distance_from_joints(q_t)
            d_w[s:e] = dw.detach().cpu().numpy().squeeze()
            d_s[s:e] = ds.detach().cpu().numpy().squeeze()
        return d_w, d_s

    T = len(full_q)
    d_w = np.zeros(T, dtype=np.float32)
    d_s = np.zeros(T, dtype=np.float32)
    if grasp_step > 0:
        d_w[:grasp_step], d_s[:grasp_step] = _run(
            world_with_obj, full_q[:grasp_step])
    if grasp_step < T:
        d_w[grasp_step:], d_s[grasp_step:] = _run(
            world_no_obj, full_q[grasp_step:])

    collide_world = d_w > 0
    collide_self = d_s > 0
    collide = collide_world | collide_self

    print(f"[collide] world: {collide_world.sum()}/{T} steps  "
          f"self: {collide_self.sum()}/{T}  "
          f"any: {collide.sum()}/{T}")

    # First and last collide step per category (helpful summary).
    def _span(mask, label):
        idx = np.where(mask)[0]
        if not len(idx):
            print(f"[collide] {label}: none")
            return
        d_max = float(np.max(d_w if label == "world" else d_s))
        print(f"[collide] {label}: first={idx[0]} last={idx[-1]}  "
              f"max_depth={d_max:.4f}m")
    _span(collide_world, "world")
    _span(collide_self, "self")

    return {"d_world": d_w, "d_self": d_s,
            "collide_world": collide_world, "collide_self": collide_self,
            "collide": collide}


def _detect_grasp_step(hand_rad: np.ndarray) -> int:
    """Return the first index where the thumb_pitch has closed at least 0.15 rad
    above its initial value, indicating the squeeze started. Falls back to 0
    if no such step (caller will then attach object from frame 0).
    """
    thumb_pitch = hand_rad[:, 1]
    base = thumb_pitch[0]
    closed = np.where(thumb_pitch > base + 0.15)[0]
    return int(closed[0]) if len(closed) else 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trial_dir", default=None,
                        help="Trial directory. If omitted, latest trial under "
                             "--obj is used.")
    parser.add_argument("--obj", default=None,
                        help="Object name (used to find latest trial when "
                             "--trial_dir omitted; otherwise mesh override).")
    parser.add_argument("--hand", default="inspire_left",
                        choices=["inspire_left", "inspire", "allegro"])
    parser.add_argument("--exp_name", default="reset_test/reorient_drop",
                        help="Path under experiment/ to search for trials.")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--no_action", action="store_true",
                        help="Hide the commanded-action ghost robot (default "
                             "shows both sensor and action overlaid).")
    parser.add_argument("--action_offset_y", type=float, default=0.0,
                        help="Offset the action ghost robot along world y "
                             "(meters) so the two robots don't overlap.")
    parser.add_argument("--check_collision", action="store_true",
                        help="Run cuRobo collision check on every trajectory "
                             "step (pre-grasp = object in world, post-grasp = "
                             "object removed). Prints a per-phase summary and "
                             "saves a per-step report to the trial dir.")
    parser.add_argument("--show_spheres", action="store_true",
                        help="Overlay cuRobo collision spheres on the SENSOR "
                             "robot (read from spheres/xarm_<hand>.yml) so you "
                             "can see exactly what cuRobo is checking against.")
    parser.add_argument("--sphere_links", default="hand",
                        choices=["hand", "all"],
                        help="Which sphere groups to show when --show_spheres "
                             "is set. 'hand'=palm+fingers (default), 'all' "
                             "adds the xArm arm links too (~89 spheres).")
    args = parser.parse_args()

    if args.hand != "inspire_left":
        sys.exit("only inspire_left wired up so far (hand order/limits differ "
                 "for inspire/allegro — extend HAND_JOINTS_* and conversion).")

    trial_dir = _resolve_trial_dir(args)
    obj_name = _infer_obj_name(trial_dir, args.obj)
    print(f"[replay] trial = {trial_dir}")
    print(f"[replay] obj   = {obj_name}")

    raw = trial_dir / "raw"
    arm_t       = np.load(raw / "arm"  / "time.npy")
    arm_pos     = np.load(raw / "arm"  / "position.npy")
    # NOTE: arm/action.npy is CARTESIAN (mm + axis-angle); the joint-space
    # commanded values are saved separately as arm/action_qpos.npy.
    arm_act_qp  = np.load(raw / "arm"  / "action_qpos.npy")
    hand_t      = np.load(raw / "hand" / "time.npy")
    hand_pos_raw = np.load(raw / "hand" / "position.npy")  # int 0-1000, ctrl order
    hand_act_raw = np.load(raw / "hand" / "action.npy")    # int 0-1000, ctrl order

    print(f"[replay] arm  : {arm_pos.shape} ({arm_t[-1] - arm_t[0]:.2f}s)")
    print(f"[replay] hand : {hand_pos_raw.shape} ({hand_t[-1] - hand_t[0]:.2f}s)")

    def _hand_to_arm_timeline(h_raw_all):
        h_aligned = np.zeros((len(arm_t), 6), dtype=np.float64)
        for i in range(6):
            h_aligned[:, i] = np.interp(arm_t, hand_t, h_raw_all[:, i])
        return sensor_units_to_rad(h_aligned)

    hand_rad_sensor = _hand_to_arm_timeline(hand_pos_raw)
    hand_rad_action = _hand_to_arm_timeline(hand_act_raw)

    full_q_sensor = np.concatenate([arm_pos,    hand_rad_sensor],
                                    axis=1).astype(np.float32)
    full_q_action = np.concatenate([arm_act_qp, hand_rad_action],
                                    axis=1).astype(np.float32)
    full_q = full_q_sensor   # default: collision check + obj-attach on sensor
    print(f"[replay] sensor: {full_q_sensor.shape}  action: {full_q_action.shape}")

    # Object pose — convert WORLD (camera) → robot via C2R.
    pose_world = np.load(trial_dir / "pose_world.npy")
    c2r = np.load(trial_dir / "C2R.npy")
    pose_obj_robot = np.linalg.inv(c2r) @ pose_world

    # Build T_obj_in_wrist from saved grasp-time wrist and initial object pose.
    plan_dir = trial_dir / "plan"
    wrist_se3 = np.load(plan_dir / "wrist_se3.npy")     # 4x4, robot frame
    if (plan_dir / "T_obj_in_wrist.npy").exists():
        T_obj_in_wrist = np.load(plan_dir / "T_obj_in_wrist.npy")
    else:
        T_obj_in_wrist = np.linalg.inv(wrist_se3) @ pose_obj_robot

    grasp_step = _detect_grasp_step(hand_rad_sensor)
    print(f"[replay] grasp detected at step {grasp_step} "
          f"(t={arm_t[grasp_step] - arm_t[0]:.2f}s)")

    collide_info = None
    if args.check_collision:
        scene_path = trial_dir / "scene_cfg.json"
        if not scene_path.exists():
            sys.exit(f"--check_collision needs {scene_path} (not saved this trial)")
        # Check BOTH sensor and action — if action collides in cuRobo's own
        # check, the planner generated a bad path. If action passes but
        # sensor collides, it's tracking error or sphere coverage.
        print("\n[collide] === SENSOR (raw/arm/position.npy) ===")
        c_sensor = _collision_check_traj(full_q_sensor, scene_path,
                                          args.hand, grasp_step)
        print("\n[collide] === ACTION (raw/arm/action_qpos.npy) ===")
        c_action = _collision_check_traj(full_q_action, scene_path,
                                          args.hand, grasp_step)
        out_path = trial_dir / "replay_collision.npz"
        np.savez(out_path, time=arm_t,
                 **{f"sensor_{k}": v for k, v in c_sensor.items()},
                 **{f"action_{k}": v for k, v in c_action.items()})
        print(f"\n[collide] report saved → {out_path}")
        collide_info = {"sensor": c_sensor, "action": c_action}

    # Compute wrist FK at every step (needed only AFTER grasp_step).
    urdf_path = URDF_BY_HAND[args.hand]
    urdf = yourdfpy.URDF.load(str(urdf_path))
    wrist_T = _fk_wrist_batch(urdf, full_q, ARM_JOINTS,
                              HAND_JOINTS_INSPIRE_LEFT)

    obj_poses = np.tile(pose_obj_robot[None], (len(full_q), 1, 1))
    obj_poses[grasp_step:] = wrist_T[grasp_step:] @ T_obj_in_wrist

    # Viewer.
    mesh_path = MESH_BASE / obj_name / "raw_mesh" / f"{obj_name}.obj"
    if not mesh_path.exists():
        sys.exit(f"obj mesh not found: {mesh_path}")
    obj_mesh = trimesh.load(str(mesh_path), process=False)

    vis = ViserViewer(port_number=args.port)
    vis.add_robot("xarm", str(urdf_path))
    if not args.no_action:
        action_pose = np.eye(4)
        action_pose[1, 3] = float(args.action_offset_y)
        vis.add_robot("xarm_action", str(urdf_path), pose=action_pose)
    vis.add_object("obj", obj_mesh, pose_obj_robot)
    vis.add_floor(height=0.0)

    sphere_traj = {}
    if args.show_spheres:
        spheres = _load_spheres(args.hand, args.sphere_links)
        n_spheres = sum(len(v) for v in spheres.values())
        print(f"[replay] computing {n_spheres} sphere transforms over "
              f"{len(full_q_sensor)} steps...")
        sphere_traj, sphere_radii = _compute_sphere_poses(
            urdf, full_q_sensor, spheres,
            ARM_JOINTS, HAND_JOINTS_INSPIRE_LEFT)
        for name, r in sphere_radii.items():
            sph_mesh = trimesh.creation.icosphere(subdivisions=1, radius=r)
            sph_mesh.visual.face_colors = [255, 80, 80, 110]  # transp red
            vis.add_object(name, sph_mesh, sphere_traj[name][0])
        print(f"[replay] added {n_spheres} cuRobo collision spheres")

    robot_traj = {"xarm": full_q_sensor}
    if not args.no_action:
        robot_traj["xarm_action"] = full_q_action
    vis.add_traj("replay", robot_traj,
                 {"obj": obj_poses, **sphere_traj})
    vis.start_viewer(use_thread=True)
    print(f"[replay] viser  http://localhost:{args.port}  "
          f"(grasp_step={grasp_step}/{len(full_q_sensor)}; "
          f"{'action overlay ON' if not args.no_action else 'sensor only'})")
    try:
        while True:
            import time
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[replay] bye")


if __name__ == "__main__":
    main()
