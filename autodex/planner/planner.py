import os
import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.spatial.transform import Rotation

import torch

os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'


def _snap_joint6(q: float, cur: float,
                  lo: float = -2.0 * np.pi, hi: float = 2.0 * np.pi) -> float:
    """Pick the equivalent angle (q + k*2π) inside [lo, hi] that is
    closest to ``cur``. Falls back to wrap if cur itself is outside.

    Defaults to ±2π bounds (matches widened xarm URDF limits for joint4/joint6)
    so an IK goal returned in one wrap can snap to its 2π-equivalent if that
    is closer to the start config — avoids 360° detours during trajopt.
    """
    candidates = [q + k * 2.0 * np.pi for k in (-1, 0, 1)]
    valid = [c for c in candidates if lo - 1e-6 <= c <= hi + 1e-6]
    if valid:
        return min(valid, key=lambda c: abs(c - cur))
    # cur out of range — wrap to [-π, π]
    return ((q + np.pi) % (2.0 * np.pi)) - np.pi

from curobo.util_file import load_yaml
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.geom.types import WorldConfig
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig
from curobo.rollout.rollout_base import Goal
from curobo.wrap.reacher.ik_solver import IKSolver, IKSolverConfig
from curobo.rollout.cost.pose_cost import PoseCostMetric
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.geom.sdf.world import CollisionQueryBuffer
from curobo.util.trajectory import InterpolateType
from curobo.util.logger import setup_curobo_logger
setup_curobo_logger("warning")

from autodex.utils.path import robot_configs_path, load_candidate, project_dir
from autodex.utils.conversion import se32action, cart2se3
from autodex.utils.robot_config import (
    INIT_STATE, XARM_INIT, INSPIRE_INIT, FR3_INIT,
    ALLEGRO_LINK6_TO_WRIST, INSPIRE_LINK6_TO_WRIST, INSPIRE_LEFT_LINK6_TO_WRIST,
    FR3_INSPIRE_LINK_TO_WRIST,
)


# ── Result ────────────────────────────────────────────────────────────────────

@dataclass
class PlanResult:
    success: bool
    traj: Optional[np.ndarray]        # (T, dof)
    wrist_se3: Optional[np.ndarray]   # (4, 4)
    pregrasp_pose: np.ndarray         # (16,) hand joints
    grasp_pose: np.ndarray            # (16,) hand joints
    scene_info: list
    timing: Optional[dict] = None     # per-stage timing breakdown
    openpose_pose: Optional[np.ndarray] = None  # (16,) hand joints; None if
                                                # caller didn't request openpose


# ── cuRobo format conversion (private) ───────────────────────────────────────

def _expand_candidates_cyl(wrist_se3: np.ndarray,
                            pregrasp: np.ndarray,
                            grasp: np.ndarray,
                            openpose_list,
                            scene_info: list,
                            obj_pose: np.ndarray,
                            cyl_axis_local,
                            cyl_yaw_grid):
    """For cylinder objects, expand each candidate wrist by N_cyl rotations
    around the object's symmetry axis (axis_local in object frame, axis
    passes through object origin in world).

    Returns (wrist_se3, pregrasp, grasp, openpose_list, scene_info) all
    expanded to length N * N_cyl. Finger configs and scene_info entries are
    replicated since the cylinder looks identical under cyl_yaw rotation.

    Pass-through (no expansion) when ``cyl_axis_local`` or ``cyl_yaw_grid``
    is None / single-element.
    """
    if (cyl_axis_local is None or cyl_yaw_grid is None
            or len(cyl_yaw_grid) <= 1):
        return wrist_se3, pregrasp, grasp, openpose_list, scene_info
    axis = np.asarray(cyl_axis_local, dtype=np.float64).reshape(3)
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    obj_inv = np.linalg.inv(obj_pose)
    out_w, out_p, out_g, out_op, out_si = [], [], [], [], []
    for i in range(len(wrist_se3)):
        for theta in cyl_yaw_grid:
            R_cyl = Rotation.from_rotvec(axis * float(theta)).as_matrix()
            R_4 = np.eye(4); R_4[:3, :3] = R_cyl
            out_w.append(obj_pose @ R_4 @ obj_inv @ wrist_se3[i])
            out_p.append(pregrasp[i])
            out_g.append(grasp[i])
            if openpose_list is not None:
                out_op.append(openpose_list[i])
            out_si.append(scene_info[i])
    return (np.array(out_w), np.array(out_p), np.array(out_g),
            (out_op if openpose_list is not None else None), out_si)


def _se3_to_7vec(mat: np.ndarray) -> list:
    """4x4 SE3 -> [x, y, z, qx, qy, qz, qw]."""
    t = mat[:3, 3].tolist()
    q = Rotation.from_matrix(mat[:3, :3]).as_quat().tolist()
    return t + q


def _to_curobo_world(scene_cfg: dict) -> dict:
    """scene_cfg -> cuRobo WorldConfig dict. Poses are already 7D [x,y,z,qw,qx,qy,qz]."""
    cfg = {"cuboid": {}, "mesh": {}}
    for name, info in scene_cfg.get("cuboid", {}).items():
        cfg["cuboid"][name] = {
            "dims": info["dims"],
            "pose": info["pose"],
            "color": info.get("color", [0.5, 0.5, 0.5, 1.0]),
        }
    for name, info in scene_cfg.get("mesh", {}).items():
        cfg["mesh"][name] = {
            "pose": info["pose"],
            "file_path": info["file_path"],
        }
    return cfg


def _to_curobo_pose(poses_se3: np.ndarray, device) -> Pose:
    """(B, 4, 4) -> cuRobo Pose."""
    position = torch.tensor(poses_se3[:, :3, 3], dtype=torch.float32, device=device).contiguous()
    xyzw = Rotation.from_matrix(poses_se3[:, :3, :3]).as_quat()
    wxyz = torch.tensor(xyzw[:, [3, 0, 1, 2]], dtype=torch.float32, device=device).contiguous()
    return Pose(position=position, quaternion=wxyz)


# ── Planner ───────────────────────────────────────────────────────────────────

class GraspPlanner:
    """
    scene_cfg + grasp candidates -> collision-free trajectory.

    Usage:
        planner = GraspPlanner()
        result = planner.plan(scene_cfg, obj_name="bottle", grasp_version="v1")
        if result.success:
            execute(result.traj)
    """

    BATCH_SIZE = 50
    N_CUBOIDS = 30
    N_MESHES = 5

    HAND_CONFIGS = {
        "allegro":      ("xarm_allegro.yml",      "allegro_floating.yml",      0.01,  32, InterpolateType.CUBIC),
        "inspire":      ("xarm_inspire.yml",      "inspire_floating.yml",      0.005, 32, InterpolateType.LINEAR_CUDA),
        "inspire_left": ("xarm_inspire_left.yml", "inspire_left_floating.yml", 0.005, 32, InterpolateType.LINEAR_CUDA),
        # FR3 (Franka) arm + inspire right hand. Same hand as "inspire", so the
        # floating-hand cfg and numerics are shared; only the arm differs.
        "fr3_inspire":  ("fr3_inspire.yml",       "inspire_floating.yml",      0.005, 32, InterpolateType.LINEAR_CUDA),
    }

    def __init__(self, robot_cfg_path: Optional[str] = None, hand_cfg_path: Optional[str] = None,
                 hand: str = "allegro", use_cuda_graph: bool = True):
        if robot_cfg_path is None:
            robot_file, hand_file, self._collision_act_dist, self._num_trajopt_seeds, self._interpolation_type = self.HAND_CONFIGS.get(hand, self.HAND_CONFIGS["allegro"])
            robot_cfg_path = os.path.join(robot_configs_path, robot_file)
            if hand_cfg_path is None:
                hand_cfg_path = os.path.join(robot_configs_path, hand_file)
        else:
            self._collision_act_dist = 0.01
            self._num_trajopt_seeds = 1024
            self._interpolation_type = InterpolateType.LINEAR_CUDA

        self._robot_cfg = load_yaml(robot_cfg_path)["robot_cfg"]
        self._hand_cfg = load_yaml(hand_cfg_path)["robot_cfg"]
        # Redirect curobo's robot config / asset lookups to AutoDex's content
        # dir. Without this curobo falls back to its install-internal content
        # (e.g. /home/robot/RSS_2026/planner/src/curobo/content/) which only
        # ships xarm_allegro spheres — xarm_inspire / xarm_inspire_left live
        # under shared_data/AutoDex/content/configs/robot/spheres/.
        _ext_robot = robot_configs_path
        _ext_asset = os.path.join(project_dir, "content", "assets")
        for _cfg in (self._robot_cfg, self._hand_cfg):
            _cfg.setdefault("kinematics", {})
            _cfg["kinematics"]["external_robot_configs_path"] = _ext_robot
            _cfg["kinematics"]["external_asset_path"] = _ext_asset
        self._tensor_args = TensorDeviceType()
        self._motion_gen: Optional[MotionGen] = None
        self._plan_cfg: Optional[MotionGenPlanConfig] = None
        self._ik_solver: Optional[IKSolver] = None
        # Last world_cfg loaded into motion_gen — used to skip full rebuild when
        # only mesh poses changed across plan() calls.
        self._cached_world: Optional[dict] = None

        # Init state: same arm position for all hands, hand-specific finger init.
        # FR3 is 7-DOF, so it must branch before the 6-DOF xarm cases.
        if hand == "fr3_inspire":
            self._init_state = np.concatenate([FR3_INIT, INSPIRE_INIT]).astype(np.float32)
            self._link6_to_wrist_rot = FR3_INSPIRE_LINK_TO_WRIST[:3, :3]
            self._n_arm = len(FR3_INIT)
        elif hand.startswith("inspire"):
            self._init_state = np.concatenate([XARM_INIT, INSPIRE_INIT]).astype(np.float32)
            self._link6_to_wrist_rot = INSPIRE_LINK6_TO_WRIST[:3, :3]
            self._n_arm = len(XARM_INIT)
        else:
            self._init_state = INIT_STATE.astype(np.float32)
            self._link6_to_wrist_rot = ALLEGRO_LINK6_TO_WRIST[:3, :3]
            self._n_arm = len(XARM_INIT)
        # Arm joints whose URDF limits were widened to ±2π, so an IK solution may
        # come back a full turn away from the start config. Only the xarm has
        # these (joint4 / joint6); the FR3's limits are all inside ±2π, so
        # snapping there would be a no-op at best and a limit violation at worst.
        self._wrap_joints = (3, 5) if self._n_arm == 6 else ()

        # Precompute link6 y-axis in wrist frame for backward filter
        self._link6_y_in_wrist = np.linalg.inv(self._link6_to_wrist_rot) @ np.array([0, 1, 0])
        self._hand = hand
        self._use_cuda_graph = use_cuda_graph

    def set_start_state(self, qpos: np.ndarray) -> None:
        """Use the robot's measured full configuration for the next plan.

        The historical execution loop always planned from ``_init_state`` and
        consequently had to reset home after a miss.  A continuous demo needs
        the opposite: after a raised, empty-handed miss it should re-observe
        and choose another grasp *from where the arm already is*.  Updating
        this state does not rebuild cuRobo's world or roadmap.
        """
        q = np.asarray(qpos, dtype=np.float32).reshape(-1)
        if q.shape != self._init_state.shape:
            raise ValueError(
                f"start state has {len(q)} joints, expected {len(self._init_state)}"
            )
        if not np.isfinite(q).all():
            raise ValueError("start state contains non-finite joint values")
        self._init_state = q.copy()

    def _snap_arm(self, arm_q: np.ndarray, ref) -> np.ndarray:
        """In-place ±2π snap of the wide-limit arm joints toward ``ref``.

        No-op on arms without ±2π joints (FR3). Returns ``arm_q``.
        """
        for j in self._wrap_joints:
            arm_q[j] = _snap_joint6(arm_q[j], float(ref[j]))
        return arm_q

    # ── world setup ───────────────────────────────────────────────────────────

    def _init_motion_gen(self, world_cfg: dict, use_cuda_graph: bool = True):
        config = MotionGenConfig.load_from_robot_config(
            self._robot_cfg,
            WorldConfig.from_dict(world_cfg),
            self._tensor_args,
            num_trajopt_seeds=self._num_trajopt_seeds,
            num_graph_seeds=1,
            num_ik_seeds=32,
            use_cuda_graph=self._use_cuda_graph,
            interpolation_dt=0.01,
            interpolation_type=self._interpolation_type,
            collision_cache={"obb": self.N_CUBOIDS, "mesh": self.N_MESHES},
            ik_opt_iters=200,
            grad_trajopt_iters=200,
            trajopt_tsteps=64,
            collision_activation_distance=self._collision_act_dist,
            store_debug_in_result=True,
        )
        self._motion_gen = MotionGen(config)
        self._motion_gen.warmup(enable_graph=True, warmup_js_trajopt=False)
        self._plan_cfg = MotionGenPlanConfig(
            enable_graph=True,
            enable_opt=True,
            enable_graph_attempt=2,    # was 4 — fewer GP retries per call
            max_attempts=3,            # 20 -> 5 -> 3: failing plans exhaust attempts (slow); successes hit early
            enable_finetune_trajopt=True,
            num_trajopt_seeds=32,
            num_ik_seeds=32,
            timeout=60.0,
            parallel_finetune=True,
        )

    def _update_world(self, world_cfg: dict):
        self._motion_gen.clear_world_cache()
        self._motion_gen.update_world(WorldConfig.from_dict(world_cfg))

    def _world_structure_changed(self, new_cfg: dict) -> bool:
        """True if anything other than mesh-pose entries changed vs cache.
        When False, we can in-place update only the mesh poses (~1ms) instead
        of a full clear_world_cache + update_world (~10-20ms + roadmap loss)."""
        if self._cached_world is None:
            return True
        old, new = self._cached_world, new_cfg
        if old.get("cuboid", {}) != new.get("cuboid", {}):
            return True
        om, nm = old.get("mesh", {}), new.get("mesh", {})
        if set(om) != set(nm):
            return True
        for k in om:
            if om[k].get("file_path") != nm[k].get("file_path"):
                return True
        return False

    def _update_target_pose_only(self, new_cfg: dict):
        """In-place update of every mesh's pose in the existing motion_gen world.
        Assumes structure unchanged (caller is responsible — see
        _world_structure_changed). IK solver world has empty mesh dict so no
        update needed there."""
        device = self._tensor_args.device
        for name, info in new_cfg.get("mesh", {}).items():
            p = info["pose"]   # [x, y, z, qw, qx, qy, qz]
            pos = torch.tensor(p[:3], dtype=torch.float32, device=device).unsqueeze(0)
            quat = torch.tensor(p[3:], dtype=torch.float32, device=device).unsqueeze(0)
            pose = Pose(position=pos, quaternion=quat)
            self._motion_gen.world_coll_checker.update_obstacle_pose(name, pose)

    # ── collision check ───────────────────────────────────────────────────────

    def _check_collision(self, world_cfg: dict, wrist_se3: np.ndarray, pregrasp: np.ndarray) -> np.ndarray:
        """bool array (N,): True = collides."""
        rw_config = RobotWorldConfig.load_from_config(
            self._hand_cfg,
            WorldConfig.from_dict(world_cfg),
            collision_activation_distance=0.0,
            tensor_args=self._tensor_args,
        )
        rw = RobotWorld(rw_config)
        n_dof = rw.kinematics.get_dof()

        # Build q vector matching robot's expected DOF
        if n_dof == len(pregrasp[0]) + 6:
            # Floating hand (e.g. allegro_floating): [x,y,z,roll,pitch,yaw] + joints
            q = np.array([se32action(w, g) for w, g in zip(wrist_se3, pregrasp)])
        else:
            # use_root_pose hand (e.g. inspire): [x,y,z,qw,qx,qy,qz] + joints
            from autodex.utils.conversion import se32cart
            q = np.array([np.concatenate([se32cart(w), g]) for w, g in zip(wrist_se3, pregrasp)])
            # If still doesn't match, just use joints
            if q.shape[1] != n_dof:
                q = pregrasp

        q_t = torch.tensor(q, dtype=torch.float32, device=self._tensor_args.device)
        d_world, d_self = rw.get_world_self_collision_distance_from_joints(q_t)
        world_coll = (d_world > 0).cpu().numpy()
        self_coll = (d_self > 0).cpu().numpy()
        return world_coll | self_coll

    def _check_world_collision_only(self, world_cfg: dict, wrist_se3: np.ndarray, joints: np.ndarray) -> np.ndarray:
        """bool array (N,): True = hand spheres collide with world (object/obstacles).
        Self-collision is IGNORED — caller uses this for "is hand in contact with X?" queries."""
        rw_config = RobotWorldConfig.load_from_config(
            self._hand_cfg,
            WorldConfig.from_dict(world_cfg),
            collision_activation_distance=0.0,
            tensor_args=self._tensor_args,
        )
        rw = RobotWorld(rw_config)
        n_dof = rw.kinematics.get_dof()
        if n_dof == len(joints[0]) + 6:
            q = np.array([se32action(w, g) for w, g in zip(wrist_se3, joints)])
        else:
            from autodex.utils.conversion import se32cart
            q = np.array([np.concatenate([se32cart(w), g]) for w, g in zip(wrist_se3, joints)])
            if q.shape[1] != n_dof:
                q = joints
        q_t = torch.tensor(q, dtype=torch.float32, device=self._tensor_args.device)
        d_world, _ = rw.get_world_self_collision_distance_from_joints(q_t)
        return (d_world > 0).cpu().numpy()

    def check_collision_per_sphere(self, world_cfg: dict, wrist_se3: np.ndarray, pregrasp: np.ndarray,
                                    compute_esdf: bool = False):
        """Per-sphere world collision.

        Accepts either a single (4, 4) + (J,) or batched (B, 4, 4) + (B, J).

        Args:
            compute_esdf: if True, return per-sphere signed distance instead of bool.
                          Signed distance is negative outside obstacles, positive inside.

        Returns:
            centers: (B, N_s, 3) — or (N_s, 3) if input was unbatched
            radii:   (N_s,)
            result:  (B, N_s) bool collide if compute_esdf=False, else
                     (B, N_s) float signed distance.
                     Unbatched output is (N_s,).
        """
        unbatched = wrist_se3.ndim == 2
        if unbatched:
            wrist_se3 = wrist_se3[None]
            pregrasp = pregrasp[None]

        rw_config = RobotWorldConfig.load_from_config(
            self._hand_cfg,
            WorldConfig.from_dict(world_cfg),
            collision_activation_distance=0.0,
            tensor_args=self._tensor_args,
        )
        rw = RobotWorld(rw_config)
        n_dof = rw.kinematics.get_dof()

        if n_dof == pregrasp.shape[1] + 6:
            q = np.array([se32action(w, g) for w, g in zip(wrist_se3, pregrasp)])
        else:
            from autodex.utils.conversion import se32cart
            q = np.array([np.concatenate([se32cart(w), g]) for w, g in zip(wrist_se3, pregrasp)])
            if q.shape[1] != n_dof:
                q = pregrasp

        q_t = torch.tensor(q, dtype=torch.float32, device=self._tensor_args.device)
        state = rw.get_kinematics(q_t)
        spheres = state.link_spheres_tensor.unsqueeze(1)  # (B, 1, N_s, 4)

        buffer = CollisionQueryBuffer.initialize_from_shape(
            spheres.shape, self._tensor_args, rw.world_model.collision_types
        )
        weight = torch.tensor([1.0], dtype=torch.float32, device=self._tensor_args.device)
        act = torch.tensor([0.0], dtype=torch.float32, device=self._tensor_args.device)

        # BODex fork added `contact_distance` (required, not optional in its kernel);
        # upstream cuRobo doesn't have the kwarg at all. Detect by signature.
        import inspect
        sig = inspect.signature(rw.world_model.get_sphere_distance)
        kwargs = {"sum_collisions": False}
        if compute_esdf:
            kwargs["compute_esdf"] = True
        if "contact_distance" in sig.parameters:
            kwargs["contact_distance"] = torch.zeros(
                spheres.shape[2], dtype=torch.float32, device=self._tensor_args.device,
            )
        d = rw.world_model.get_sphere_distance(spheres, buffer, weight, act, **kwargs)

        centers = spheres[:, 0, :, :3].detach().cpu().numpy()  # (B, N_s, 3)
        radii   = spheres[0, 0, :, 3].detach().cpu().numpy()   # (N_s,)
        d_flat = d.view(d.shape[0], -1).detach().cpu().numpy()
        if compute_esdf:
            result = d_flat  # signed distance: negative outside, positive inside
        else:
            result = (d_flat > 0)  # bool collide

        # Filter zero-radius placeholder spheres.
        valid = radii > 1e-6
        centers = centers[:, valid, :]
        radii = radii[valid]
        result = result[:, valid]

        if unbatched:
            return centers[0], radii, result[0]
        return centers, radii, result

    # ── IK solver ─────────────────────────────────────────────────────────────

    def _init_ik_solver(self, world_cfg: dict, use_cuda_graph: bool = True):
        config = IKSolverConfig.load_from_robot_config(
            self._robot_cfg,
            WorldConfig.from_dict(world_cfg),
            self._tensor_args,
            num_seeds=32,
            collision_cache={"obb": self.N_CUBOIDS, "mesh": self.N_MESHES},
            collision_activation_distance=self._collision_act_dist,
            use_cuda_graph=self._use_cuda_graph,
        )
        self._ik_solver = IKSolver(config)

    def solve_ik(self, scene_cfg: dict, obj_name: str, grasp_version: str,
                 seed: Optional[int] = None, hand: str = "allegro",
                 scene_id: Optional[str] = None,
                 cyl_axis_local: Optional[np.ndarray] = None,
                 cyl_yaw_grid: Optional[np.ndarray] = None,
                 scene_type_filter: Optional[str] = None,
                 skip_scenes_with_success: bool = False,
                 candidates_root: Optional[str] = None):
        """
        IK-only reachability check for all grasp candidates.

        Skips hand-object collision check (hand is supposed to be near the object).
        Only applies backward filter. IK solver handles arm-scene collision internally.

        ``scene_id`` (str) restricts loaded candidates to the matching tabletop
        scene_id (sorted index), same convention as ``planner.plan``.

        ``cyl_axis_local`` + ``cyl_yaw_grid`` (optional): for continuous-revolute
        objects (e.g. lying cylinder), expand each candidate wrist into N_cyl
        rotated variants around the object's symmetry axis. The same finger
        config (pregrasp/grasp/openpose) is shared across variants because the
        cylinder looks identical under that rotation. Multiplies candidate
        pool by ``len(cyl_yaw_grid)``.

        Returns:
            dict with per-candidate success, qpos, and timing.
        """
        import time as _time

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        t0 = _time.time()
        obj_pose = cart2se3(scene_cfg["mesh"]["target"]["pose"])
        wrist_se3, pregrasp, grasp, scene_info = load_candidate(
            obj_name, obj_pose, grasp_version, hand=hand, scene_id=scene_id,
            scene_type_filter=scene_type_filter,
            skip_scenes_with_success=skip_scenes_with_success,
            candidates_root=candidates_root)
        # Expand by cyl_yaw around object symmetry axis (cylinder objects only).
        wrist_se3, pregrasp, grasp, _, scene_info = _expand_candidates_cyl(
            wrist_se3, pregrasp, grasp, None, scene_info,
            obj_pose, cyl_axis_local, cyl_yaw_grid)
        t_load = _time.time() - t0

        t0 = _time.time()
        # IK solver uses table-only world (no target mesh — arm shouldn't collide with table)
        world_cfg_no_target = _to_curobo_world(scene_cfg)
        world_cfg_no_target["mesh"] = {}
        if self._ik_solver is None:
            self._init_ik_solver(world_cfg_no_target)
        else:
            self._ik_solver.update_world(WorldConfig.from_dict(world_cfg_no_target))
        t_world = _time.time() - t0

        # Filter: backward + hand-table collision (no object mesh — hand should be near object)
        t0 = _time.time()
        backward = np.zeros(len(wrist_se3), dtype=bool) if self._hand.startswith("inspire") else (wrist_se3[:, :3, :3] @ self._link6_y_in_wrist)[:, 2] < 0.3
        collision = self._check_collision(world_cfg_no_target, wrist_se3, pregrasp)
        filtered = backward | collision
        valid = np.where(~filtered)[0]
        t_filter = _time.time() - t0

        N = len(wrist_se3)
        ik_success = np.zeros(N, dtype=bool)
        ik_qpos = np.full((N, len(self._init_state)), np.nan)  # 6 arm + 16 fingers

        t0 = _time.time()
        if len(valid) > 0:
            # Process in fixed-size chunks for consistent CUDA graph shape
            for chunk_start in range(0, len(valid), self.BATCH_SIZE):
                chunk_idx = valid[chunk_start : chunk_start + self.BATCH_SIZE]
                chunk_poses = wrist_se3[chunk_idx]
                B = len(chunk_poses)

                if B < self.BATCH_SIZE:
                    pad = self.BATCH_SIZE - B
                    chunk_poses = np.concatenate(
                        [chunk_poses, np.tile(chunk_poses[:1], (pad, 1, 1))], axis=0)

                goal = _to_curobo_pose(chunk_poses, self._tensor_args.device)
                # Retract toward init_state so IK solutions stay near start
                # config — matches planner.plan() so the subsequent
                # plan_single_js(INIT_STATE → ik_qpos) has a short, mostly
                # collision-free distance to cover.
                B_padded = chunk_poses.shape[0]
                retract = torch.tensor(
                    self._init_state, dtype=torch.float32,
                    device=self._tensor_args.device,
                ).unsqueeze(0).repeat(B_padded, 1)
                result = self._ik_solver.solve_batch(
                    goal, retract_config=retract)
                succ = result.success.cpu().numpy()[:B]
                q_sol = result.solution.cpu().numpy()[:B]

                if q_sol.ndim == 3:
                    q_sol = q_sol[:, 0, :]

                for i, idx in enumerate(chunk_idx):
                    if succ[i]:
                        ik_success[idx] = True
                        arm_q = q_sol[i, :self._n_arm].copy()
                        # Snap the ±2π joints to the equivalent angle nearest
                        # init_state; IK can return any angle in [-2π, 2π].
                        self._snap_arm(arm_q, self._init_state)
                        ik_qpos[idx, :self._n_arm] = arm_q
                        ik_qpos[idx, self._n_arm:] = pregrasp[idx]
        t_ik = _time.time() - t0

        # Lift IK check: verify z+10cm pose is reachable — mirrors
        # planner.plan() so candidates that would hit joint limit during a
        # short lift are filtered out here.
        LIFT_HEIGHT_CHECK = 0.05
        ik_valid_pre = np.where(ik_success)[0]
        if len(ik_valid_pre) > 0:
            lift_poses = wrist_se3[ik_valid_pre].copy()
            lift_poses[:, 2, 3] += LIFT_HEIGHT_CHECK
            for chunk_start in range(0, len(ik_valid_pre), self.BATCH_SIZE):
                chunk = ik_valid_pre[chunk_start : chunk_start + self.BATCH_SIZE]
                chunk_poses = lift_poses[chunk_start : chunk_start + len(chunk)]
                B = len(chunk_poses)
                if B < self.BATCH_SIZE:
                    pad = self.BATCH_SIZE - B
                    chunk_poses = np.concatenate(
                        [chunk_poses, np.tile(chunk_poses[:1], (pad, 1, 1))],
                        axis=0)
                goal = _to_curobo_pose(chunk_poses, self._tensor_args.device)
                lift_res = self._ik_solver.solve_batch(goal)
                lift_succ = lift_res.success.cpu().numpy()[:B]
                for i, idx in enumerate(chunk):
                    if not lift_succ[i]:
                        ik_success[idx] = False
            n_lift_fail = len(ik_valid_pre) - int(ik_success.sum())
            if n_lift_fail > 0:
                print(f"[planner] solve_ik lift IK check: {n_lift_fail} "
                      f"candidates failed (z+{LIFT_HEIGHT_CHECK}m unreachable)")

        timing = {
            "load_candidates_s": round(t_load, 3),
            "world_setup_s": round(t_world, 3),
            "filter_s": round(t_filter, 3),
            "ik_solve_s": round(t_ik, 3),
        }

        return {
            "n_total": N,
            "n_backward": int(backward.sum()),
            "n_table_collision": int(collision.sum()),
            "n_valid": int(len(valid)),
            "n_ik_success": int(ik_success.sum()),
            "ik_success": ik_success,
            "ik_qpos": ik_qpos,
            "wrist_se3": wrist_se3,
            "pregrasp": pregrasp,
            "grasp": grasp,
            "scene_info": scene_info,
            "timing": timing,
        }

    # ── motion planning ───────────────────────────────────────────────────────

    def _plan_goalset(self, goal_poses_se3: np.ndarray):
        """INIT_STATE -> best among N goals. Returns (local_idx, traj) or (None, None)."""
        init_js = JointState.from_position(
            torch.tensor(self._init_state, dtype=torch.float32, device=self._tensor_args.device).unsqueeze(0)
        )
        goal = _to_curobo_pose(goal_poses_se3, self._tensor_args.device)
        goal = Pose(position=goal.position.unsqueeze(0), quaternion=goal.quaternion.unsqueeze(0))

        result = self._motion_gen.plan_goalset(start_state=init_js, goal_pose=goal, plan_config=self._plan_cfg)
        if not result.success.item():
            return None, None
        return result.goalset_index.item(), result.get_interpolated_plan().position.cpu().numpy()

    def _plan_batch(self, init_states: np.ndarray, goal_poses_se3: np.ndarray):
        """(B, dof), (B, 4, 4) -> success (B,), trajs (B, T, dof)."""
        B = len(init_states)
        # Pad to BATCH_SIZE so cuRobo gets a consistent batch size
        if B < self.BATCH_SIZE:
            pad = self.BATCH_SIZE - B
            init_states = np.concatenate([init_states, np.tile(init_states[:1], (pad, 1))], axis=0)
            goal_poses_se3 = np.concatenate([goal_poses_se3, np.tile(goal_poses_se3[:1], (pad, 1, 1))], axis=0)

        init_js = JointState.from_position(
            torch.tensor(init_states, dtype=torch.float32, device=self._tensor_args.device)
        )
        try:
            result = self._motion_gen.plan_batch(
                start_state=init_js,
                goal_pose=_to_curobo_pose(goal_poses_se3, self._tensor_args.device),
                plan_config=self._plan_cfg,
            )
        except RuntimeError:
            # cuRobo crashes when IK finds 0 solutions (internal shape mismatch)
            return np.zeros(B, dtype=bool), None
        success = result.success.cpu().numpy()[:B]
        trajs = result.optimized_plan.position.cpu().numpy()[:B] if success.any() else None
        if trajs is not None and trajs.ndim == 2:
            trajs = trajs[np.newaxis]
        return success, trajs

    def plan_wrist_reorient(self,
                             scene_cfg: dict,
                             current_qpos: np.ndarray,
                             target_wrist_se3: np.ndarray,
                             hold_hand_qpos: np.ndarray,
                             n_yaw: int = 8,
                             ) -> tuple:
        """Plan an in-air wrist reorient to ``target_wrist_se3`` (in WORLD).

        The held object is assumed yaw-symmetric around the world-z axis
        (true for tabletop classes — see ``tabletop_pose._z_aligned_geodesic``).
        Generates ``n_yaw`` candidates rotated around world-z, runs IK on all
        of them, picks the IK solution closest to ``current_qpos`` in arm
        joint space (joint-6 wrap-unrolled), then runs ``plan_single_js`` for
        the full 22-DOF trajectory holding ``hold_hand_qpos`` throughout.

        Args:
            scene_cfg: scene with the held object's mesh.target.pose updated
                       to wherever it currently is in WORLD (for visualization
                       / planner table-cuboid). The IK solver itself runs on
                       a table-only world (no held mesh as obstacle) — held
                       object collisions are the caller's responsibility (see
                       LIFT_HEIGHT_M in reorient_drop.py).
            current_qpos: (22,) arm + hand qpos right now (squeeze state).
            target_wrist_se3: (4, 4) WORLD-frame wrist target. Position is
                              held, orientation is what matters; the N yaw
                              candidates rotate this orientation around
                              world-z.
            hold_hand_qpos: (n_finger,) finger config to hold throughout
                            (e.g. the squeeze pose from ``execute``).
            n_yaw: number of yaw candidates around world-z (8 ~= 45° steps).

        Returns:
            (traj or None, info_dict)
            traj: (T, 22) interpolated joint trajectory if planning succeeded
            info: dict with n_ik_success, best_yaw_idx, best_arm_dist_rad,
                  reason (set on failure).
        """
        import time as _time

        info = {"n_yaw": n_yaw, "n_ik_success": 0, "best_yaw_idx": -1,
                "best_arm_dist_rad": float("inf")}
        t0 = _time.time()

        # 1. Ensure motion_gen + ik_solver are initialized (lazy init mirrors
        #    plan_js_to_init / solve_ik patterns).
        world_cfg = _to_curobo_world(scene_cfg)
        if self._motion_gen is None:
            self._init_motion_gen(world_cfg)
        elif self._world_structure_changed(world_cfg):
            self._update_world(world_cfg)
        else:
            self._update_target_pose_only(world_cfg)
        self._cached_world = world_cfg

        # IK solver world: table only (no held mesh — it's "attached" to robot).
        world_cfg_no_target = dict(world_cfg)
        world_cfg_no_target["mesh"] = {}
        if self._ik_solver is None:
            self._init_ik_solver(world_cfg_no_target)
        else:
            self._ik_solver.update_world(WorldConfig.from_dict(world_cfg_no_target))

        # 2. Generate N yaw candidates around world-z.
        R_target = target_wrist_se3[:3, :3]
        p_target = target_wrist_se3[:3, 3]
        candidates = np.zeros((n_yaw, 4, 4))
        for k in range(n_yaw):
            theta = 2.0 * np.pi * k / n_yaw
            c, s = np.cos(theta), np.sin(theta)
            R_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            T = np.eye(4)
            T[:3, :3] = R_z @ R_target
            T[:3, 3] = p_target
            candidates[k] = T

        # 3. Pad to BATCH_SIZE for cuRobo IK (matches solve_ik pattern).
        B = n_yaw
        if B < self.BATCH_SIZE:
            pad = self.BATCH_SIZE - B
            cand_padded = np.concatenate(
                [candidates, np.tile(candidates[:1], (pad, 1, 1))], axis=0)
        else:
            cand_padded = candidates

        goal = _to_curobo_pose(cand_padded, self._tensor_args.device)

        # Caller is responsible for assembling current_qpos with the correct
        # DOF (12 for inspire, 22 for allegro).
        cur_full = np.asarray(current_qpos, dtype=np.float32)
        if len(cur_full) != len(self._init_state):
            info["reason"] = (
                f"current_qpos DOF {len(cur_full)} != expected "
                f"{len(self._init_state)} (arm {self._n_arm} + hand "
                f"{len(self._init_state) - self._n_arm})"
            )
            return None, info
        B_padded = cand_padded.shape[0]
        # Retract toward init_state (mirrors plan() at L1043). Using cur_full
        # as retract has been observed to make cuRobo's solve_batch return
        # success=False on ALL yaw candidates even when valid solutions exist
        # (e.g. lift's IK trivially reachable from cur_full=lift_end_qpos).
        # We still pick the IK solution closest to current_qpos below, so the
        # bias toward "minimal motion" is preserved without relying on retract.
        retract = torch.tensor(
            self._init_state, dtype=torch.float32, device=self._tensor_args.device,
        ).unsqueeze(0).repeat(B_padded, 1)
        result = self._ik_solver.solve_batch(goal, retract_config=retract)
        succ = result.success.cpu().numpy()[:B]
        q_sol = result.solution.cpu().numpy()[:B]
        if q_sol.ndim == 3:
            q_sol = q_sol[:, 0, :]
        info["n_ik_success"] = int(succ.sum())
        info["ik_solve_s"] = round(_time.time() - t0, 3)

        # 4. Sort IK-feasible yaw solutions by closeness to current arm.
        cur_arm = np.asarray(current_qpos[:self._n_arm])
        feasible_arms = []   # list of (yaw_idx, arm_q, dist)
        for i in range(B):
            if not succ[i]:
                continue
            arm_q = q_sol[i, :self._n_arm].copy()
            self._snap_arm(arm_q, current_qpos)
            dist = float(np.linalg.norm(arm_q - cur_arm))
            feasible_arms.append((i, arm_q, dist))
        feasible_arms.sort(key=lambda t: t[2])

        if len(feasible_arms) == 0:
            info["reason"] = "no_ik_feasible_among_yaw_candidates"
            return None, info

        info["best_yaw_idx"] = int(feasible_arms[0][0])
        info["best_arm_dist_rad"] = float(feasible_arms[0][2])

        # 5. plan_single_js: try yaw candidates in order until one succeeds.
        t1 = _time.time()
        start_full = np.asarray(current_qpos, dtype=np.float32)
        n_hand = len(self._init_state) - self._n_arm
        hand_held = np.asarray(hold_hand_qpos, dtype=np.float32)
        if len(hand_held) != n_hand:
            info["reason"] = (
                f"hold_hand_qpos len={len(hand_held)} != expected {n_hand}"
            )
            return None, info

        n_plan_attempts = 0
        for yaw_i, arm_q, dist in feasible_arms:
            goal_full = np.concatenate(
                [arm_q.astype(np.float32), hand_held])
            n_plan_attempts += 1
            ok, traj = self._refine_fingers(start_full, goal_full)
            if ok:
                info["plan_s"] = round(_time.time() - t1, 3)
                info["plan_success"] = True
                info["chosen_yaw_idx"] = int(yaw_i)
                info["chosen_arm_dist_rad"] = float(dist)
                info["goal_qpos"] = goal_full.tolist()
                info["n_plan_attempts"] = n_plan_attempts
                return traj, info

        info["plan_s"] = round(_time.time() - t1, 3)
        info["reason"] = (
            f"plan_single_js_failed_all_{len(feasible_arms)}_yaw")
        info["n_plan_attempts"] = n_plan_attempts
        return None, info

    def plan_obj_placement(self,
                            scene_lift: dict,
                            current_qpos: np.ndarray,
                            T_obj_in_wrist: np.ndarray,
                            R_target_obj_world: np.ndarray,
                            obj_target_pos_world: np.ndarray,
                            hold_hand_qpos: np.ndarray,
                            x_grid: np.ndarray,
                            yaw_grid: np.ndarray,
                            y_grid: np.ndarray = None,
                            cyl_yaw_grid: np.ndarray = None,
                            cyl_axis_local: np.ndarray = None,
                            skip_plan: bool = False,
                            ) -> tuple:
        """Search (x, yaw) for an IK-feasible placement and plan to it.

        Replaces ``plan_wrist_reorient`` for the "set the held obj down at
        target orientation, anywhere reachable" case. For each (x, yaw) in
        the grid, builds an obj target pose with rotation
        ``Rz(yaw) @ R_target_obj_world`` at position
        ``(x, obj_target_pos_world[1], obj_target_pos_world[2])`` (y/z fixed),
        computes the required wrist pose via ``inv(T_obj_in_wrist)``, and
        batch-IKs all candidates. Picks the IK-feasible candidate whose arm
        config is closest to the current arm config in joint space, then runs
        ``plan_single_js`` from ``current_qpos`` to that arm config holding
        ``hold_hand_qpos`` throughout.

        Args:
            scene_lift: world for collision (held mesh stripped; see
                        ``plan_wrist_reorient``).
            current_qpos: (n_dof,) arm + hand qpos at search start (e.g.
                          right after lift).
            T_obj_in_wrist: (4, 4) constant obj-in-wrist transform measured
                            at grasp time.
            R_target_obj_world: (3, 3) target obj rotation in WORLD frame.
            obj_target_pos_world: (3,) target obj position in WORLD frame.
                                  Only y and z are used; x is searched.
            hold_hand_qpos: (n_hand,) finger config to hold throughout.
            x_grid: (Nx,) world-frame x values to try.
            yaw_grid: (Nyaw,) yaw rotations around obj's vertical axis.
            cyl_yaw_grid: optional (Ncyl,) rotations about the object's local
                          symmetry axis (``cyl_axis_local``) applied IN OBJECT
                          FRAME before world yaw, i.e. final rotation is
                          ``Rz(yaw) @ R_target @ R_axis(cyl_axis_local, cyl_yaw)``.
                          Use for cylinder-symmetric objects whose appearance is
                          invariant under rotation about ``cyl_axis_local``.
                          ``None`` (default) → singleton ``[0.0]`` (no extra DoF).
            cyl_axis_local: (3,) unit axis in object frame. Required when
                            ``cyl_yaw_grid`` is provided. For CYLINDER_OBJECTS
                            this is the object's local +Y axis ``[0, 1, 0]``.
            skip_plan: if True, run only the (x, yaw) IK feasibility check
                       and return ``(None, info)`` with the chosen-best info
                       set but without calling plan_single_js. Use for cheap
                       pre-flight reachability checks before committing to
                       a grasp (~0.1s vs ~1-2s with plan_single_js).

        Returns ``(traj or None, info_dict)`` with keys ``chosen_x``,
        ``chosen_yaw``, ``T_wrist_target`` (chosen or placeholder for viz),
        ``n_feasible``, ``n_candidates``, ``reason`` (on fail).
        """
        # 1. World setup — motion_gen + IK on table-only scene.
        world_cfg = _to_curobo_world(scene_lift)
        if self._motion_gen is None:
            self._init_motion_gen(world_cfg)
        elif self._world_structure_changed(world_cfg):
            self._update_world(world_cfg)
        else:
            self._update_target_pose_only(world_cfg)
        self._cached_world = world_cfg

        world_cfg_no_target = dict(world_cfg)
        world_cfg_no_target["mesh"] = {}
        if self._ik_solver is None:
            self._init_ik_solver(world_cfg_no_target)
        else:
            self._ik_solver.update_world(
                WorldConfig.from_dict(world_cfg_no_target))

        # 2. Build (x, y, yaw, cyl_yaw) candidate wrist targets.
        if y_grid is None:
            y_grid = np.array([float(obj_target_pos_world[1])])
        if cyl_yaw_grid is None:
            cyl_yaw_grid = np.array([0.0])
            R_cyl_list = [np.eye(3)]
        else:
            if cyl_axis_local is None:
                raise ValueError(
                    "cyl_axis_local required when cyl_yaw_grid is provided")
            axis = np.asarray(cyl_axis_local, dtype=np.float64).reshape(3)
            axis = axis / (np.linalg.norm(axis) + 1e-12)
            R_cyl_list = [Rotation.from_rotvec(axis * float(theta)).as_matrix()
                          for theta in cyl_yaw_grid]
        z_fixed = float(obj_target_pos_world[2])
        T_obj_in_wrist_inv = np.linalg.inv(T_obj_in_wrist)
        candidates_T_wrist = []
        candidates_meta = []
        for x_try in x_grid:
            for y_try in y_grid:
                for yaw_try in yaw_grid:
                    c, s = np.cos(yaw_try), np.sin(yaw_try)
                    R_z = np.array(
                        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
                    for cyl_idx, cyl_try in enumerate(cyl_yaw_grid):
                        R_cyl = R_cyl_list[cyl_idx]
                        T_obj_target = np.eye(4)
                        T_obj_target[:3, :3] = R_z @ R_target_obj_world @ R_cyl
                        T_obj_target[0, 3] = float(x_try)
                        T_obj_target[1, 3] = float(y_try)
                        T_obj_target[2, 3] = z_fixed
                        T_wrist = T_obj_target @ T_obj_in_wrist_inv
                        candidates_T_wrist.append(T_wrist)
                        candidates_meta.append(
                            (float(x_try), float(y_try), float(yaw_try),
                             float(cyl_try)))
        candidates_T_wrist = np.array(candidates_T_wrist)
        N = len(candidates_T_wrist)

        # 3. Batch IK over the grid.
        device = self._tensor_args.device
        ik_success_all = np.zeros(N, dtype=bool)
        ik_arm_qpos = np.full((N, self._n_arm), np.nan, dtype=np.float32)
        for chunk_start in range(0, N, self.BATCH_SIZE):
            chunk_idx = list(range(
                chunk_start, min(chunk_start + self.BATCH_SIZE, N)))
            chunk_poses = candidates_T_wrist[chunk_idx]
            B = len(chunk_poses)
            if B < self.BATCH_SIZE:
                pad = self.BATCH_SIZE - B
                chunk_poses = np.concatenate(
                    [chunk_poses, np.tile(chunk_poses[:1], (pad, 1, 1))],
                    axis=0)
            goal = _to_curobo_pose(chunk_poses, device)
            # Use current_qpos as retract seed so IK starts near the arm's
            # current config — important for descent-from-extended where
            # INIT_STATE (folded) is far from the feasible solution.
            retract_qpos = np.asarray(
                current_qpos, dtype=np.float32)[:len(self._init_state)]
            retract = torch.tensor(
                retract_qpos, dtype=torch.float32, device=device,
            ).unsqueeze(0).repeat(self.BATCH_SIZE, 1)
            res = self._ik_solver.solve_batch(
                goal, retract_config=retract)
            succ = res.success.cpu().numpy()[:B]
            q_sol = res.solution.cpu().numpy()[:B]
            if q_sol.ndim == 3:
                q_sol = q_sol[:, 0, :]
            for i, idx in enumerate(chunk_idx):
                if succ[i]:
                    ik_success_all[idx] = True
                    arm_q = q_sol[i, :self._n_arm].copy()
                    self._snap_arm(arm_q, current_qpos)
                    ik_arm_qpos[idx] = arm_q

        feasible = np.where(ik_success_all)[0]
        info = {"n_candidates": N, "n_feasible": int(len(feasible)),
                "T_wrist_target": candidates_T_wrist[0]}
        if len(feasible) == 0:
            info["reason"] = "no_ik_feasible_in_grid"
            return None, info

        # 4. Sort IK-feasible candidates by closeness to current arm.
        cur_arm = np.asarray(current_qpos[:self._n_arm])
        dists = np.linalg.norm(ik_arm_qpos[feasible] - cur_arm, axis=1)
        order = np.argsort(dists)

        if skip_plan:
            best_local = int(order[0])
            best_idx = int(feasible[best_local])
            chosen_x, chosen_y, chosen_yaw, chosen_cyl_yaw = (
                candidates_meta[best_idx])
            info["chosen_x"] = chosen_x
            info["chosen_y"] = chosen_y
            info["chosen_yaw"] = chosen_yaw
            info["chosen_cyl_yaw"] = chosen_cyl_yaw
            info["best_arm_dist_rad"] = float(dists[best_local])
            info["T_wrist_target"] = candidates_T_wrist[best_idx]
            info["chosen_arm_qpos"] = ik_arm_qpos[best_idx].tolist()
            # Sorted (closest-arm-first) candidate list for caller-driven
            # fallback (e.g. reorient+descent must both pass).
            info["sorted_candidates"] = [
                {
                    "x": candidates_meta[int(feasible[i])][0],
                    "y": candidates_meta[int(feasible[i])][1],
                    "yaw": candidates_meta[int(feasible[i])][2],
                    "cyl_yaw": candidates_meta[int(feasible[i])][3],
                    "arm_qpos": ik_arm_qpos[int(feasible[i])].tolist(),
                    "T_wrist": candidates_T_wrist[int(feasible[i])].tolist(),
                    "arm_dist_rad": float(dists[int(i)]),
                }
                for i in order
            ]
            return None, info

        # 5. Try plan_single_js on candidates in order until one succeeds.
        start_full = np.asarray(current_qpos, dtype=np.float32)
        n_hand = len(self._init_state) - self._n_arm
        hand_held = np.asarray(hold_hand_qpos, dtype=np.float32)
        if len(hand_held) != n_hand:
            info["reason"] = (
                f"hold_hand_qpos len={len(hand_held)} != expected {n_hand}")
            return None, info

        n_plan_attempts = 0
        for local_i in order:
            local_i = int(local_i)
            cand_idx = int(feasible[local_i])
            chosen_arm = ik_arm_qpos[cand_idx]
            goal_full = np.concatenate(
                [chosen_arm.astype(np.float32), hand_held])
            n_plan_attempts += 1
            ok, traj = self._refine_fingers(start_full, goal_full)
            if ok:
                chosen_x, chosen_y, chosen_yaw, chosen_cyl_yaw = (
                    candidates_meta[cand_idx])
                info["chosen_x"] = chosen_x
                info["chosen_y"] = chosen_y
                info["chosen_yaw"] = chosen_yaw
                info["chosen_cyl_yaw"] = chosen_cyl_yaw
                info["best_arm_dist_rad"] = float(dists[local_i])
                info["T_wrist_target"] = candidates_T_wrist[cand_idx]
                info["chosen_arm_qpos"] = chosen_arm.tolist()
                info["n_plan_attempts"] = n_plan_attempts
                return traj, info

        info["reason"] = (
            f"plan_single_js_failed_all_{len(feasible)}_feasible")
        info["n_plan_attempts"] = n_plan_attempts
        return None, info

    def plan_pose_constrained(
        self,
        start_full_qpos: np.ndarray,
        target_wrist_pose: np.ndarray,
        hold_vec_weight,
        scene_cfg: Optional[dict] = None,
        include_obj_obstacle: bool = False,
        debug_dump_dir: Optional[str] = None,
    ) -> Optional[np.ndarray]:
        """Plan ``start_full_qpos -> target_wrist_pose`` with a cuRobo
        ``PoseCostMetric`` constraining selected pose components.

        Args:
            start_full_qpos: (22,) current joint state — arm (6) + hand.
            target_wrist_pose: (4, 4) wrist (cuRobo ``ee_link = base_link``)
                pose in robot frame. NOTE: cuRobo's FK / IK are both wrist-
                anchored, so callers MUST convert link6 → wrist upstream
                (``link6 @ link6_to_wrist``) before passing here. Mixing
                link6 and wrist frames in this function silently introduces
                an offset of ``link6_to_wrist`` translation (~3.5cm for
                inspire_left).
            hold_vec_weight: 6-vec ``[rx, ry, rz, x, y, z]`` — 1 = hold to
                the trajectory's initial value, 0 = free. E.g. ``[1,1,1,1,1,0]``
                for a pure +z lift, ``[0,0,0,0,0,1]`` to translate xy while
                holding z.
            scene_cfg: optional; if given, rebuilds world. If None, uses
                cached world from the previous call.
            include_obj_obstacle: if False, drops the target mesh from the
                world (e.g. obj is held in hand and shouldn't self-collide).

        Returns interpolated traj ``(T, dof)`` or ``None`` on failure.
        """
        # Optional snapshot dump for offline reproduction.
        if debug_dump_dir is not None:
            import time as _time, json as _json
            os.makedirs(debug_dump_dir, exist_ok=True)
            stem = str(int(_time.time() * 1000))
            np.savez(
                os.path.join(debug_dump_dir, f"{stem}.npz"),
                start_full_qpos=np.asarray(start_full_qpos),
                target_wrist_pose=np.asarray(target_wrist_pose),
                hold_vec_weight=np.asarray(hold_vec_weight),
                include_obj_obstacle=np.asarray(include_obj_obstacle),
            )
            if scene_cfg is not None:
                with open(os.path.join(debug_dump_dir, f"{stem}_scene.json"), "w") as _f:
                    _json.dump(scene_cfg, _f, indent=2, default=str)
            print(f"    [plan_pose_constrained] snapshot → "
                  f"{debug_dump_dir}/{stem}.npz")

        if scene_cfg is not None:
            world_cfg = _to_curobo_world(scene_cfg)
            if not include_obj_obstacle:
                world_cfg = dict(world_cfg)
                world_cfg["mesh"] = {}
            if self._motion_gen is None:
                self._init_motion_gen(world_cfg)
            elif self._world_structure_changed(world_cfg):
                self._update_world(world_cfg)
            else:
                self._update_target_pose_only(world_cfg)
            self._cached_world = world_cfg
            # Also (re)init / update ik_solver with the same no-obj world.
            if self._ik_solver is None:
                self._init_ik_solver(world_cfg)
            else:
                self._ik_solver.update_world(WorldConfig.from_dict(world_cfg))
        elif self._motion_gen is None:
            raise RuntimeError(
                "plan_pose_constrained: motion_gen not initialized; "
                "pass scene_cfg on first call."
            )
        else:
            # No scene_cfg given but motion_gen already initialized.
            # Cached world still has the target mesh as obstacle, which is
            # wrong when the obj is held in hand (start state collides with
            # obj). Rebuild a no-obj world from cached.
            cached_no_obj = dict(self._cached_world)
            cached_no_obj["mesh"] = {} if not include_obj_obstacle else self._cached_world.get("mesh", {})
            if cached_no_obj != self._cached_world:
                self._update_world(cached_no_obj)
                self._cached_world = cached_no_obj

        from scipy.spatial.transform import Rotation as _R
        device = self._tensor_args.device

        start = JointState.from_position(
            torch.tensor(start_full_qpos, dtype=torch.float32,
                         device=device).unsqueeze(0)
        )

        # Compute cuRobo's own FK for the start state so we can project
        # held components of the goal pose onto the start values exactly.
        kin_state = self._motion_gen.kinematics.get_state(start.position)
        start_pos = kin_state.ee_position[0].detach().cpu().numpy()        # (3,)
        start_quat_wxyz = kin_state.ee_quaternion[0].detach().cpu().numpy() # (4,) wxyz

        goal_pos = np.array(target_wrist_pose[:3, 3], dtype=np.float32).copy()
        goal_R = np.array(target_wrist_pose[:3, :3], dtype=np.float32).copy()

        # Project HELD position axes onto start FK.
        for i, comp in enumerate([3, 4, 5]):   # x, y, z
            if hold_vec_weight[comp]:
                goal_pos[i] = float(start_pos[i])
        # Project HELD orientation. Full-hold only (all three rotation axes).
        if hold_vec_weight[0] and hold_vec_weight[1] and hold_vec_weight[2]:
            goal_R = _R.from_quat(
                [start_quat_wxyz[1], start_quat_wxyz[2],
                 start_quat_wxyz[3], start_quat_wxyz[0]]
            ).as_matrix().astype(np.float32)

        quat_xyzw = _R.from_matrix(goal_R).as_quat()
        quat_wxyz = np.array(
            [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
            dtype=np.float32,
        )
        goal_pose = Pose(
            position=torch.tensor(goal_pos,
                                   dtype=torch.float32, device=device).unsqueeze(0),
            quaternion=torch.tensor(quat_wxyz,
                                     dtype=torch.float32, device=device).unsqueeze(0),
        )

        # cuRobo PoseCostMetric API consistently IK_FAIL'd for our cases
        # (start joint 4 wrap, etc). Skipped. Approximate the hold via:
        #  - goal pose held axes projected onto start FK (above),
        #  - IK biased with start_qpos as both seed_config and retract,
        #  - MAX_IK_DELTA reject for far-branch IK solutions,
        #  - plan_single_js joint interp between near-equal qpos.
        if self._ik_solver is None:
            raise RuntimeError(
                "plan_pose_constrained: ik_solver not initialized; "
                "call plan() or solve_ik() once first."
            )
        start_arm = np.asarray(start_full_qpos[:self._n_arm], dtype=np.float32)
        start_full = np.asarray(start_full_qpos, dtype=np.float32)
        # retract / seed use full-DOF (arm+hand) to match cuRobo IK's
        # internal joint dimension.
        retract_tensor = torch.tensor(
            start_full, dtype=torch.float32, device=device
        ).unsqueeze(0).repeat(self.BATCH_SIZE, 1)
        # seed_config wants (batch, n_seeds_extra, dof). Match batch size.
        seed_tensor = torch.tensor(
            start_full, dtype=torch.float32, device=device
        ).unsqueeze(0).unsqueeze(0).repeat(self.BATCH_SIZE, 1, 1)
        #     → (BATCH_SIZE, 1, full_dof)
        ik_pos = goal_pose.position.repeat(self.BATCH_SIZE, 1)
        ik_quat = goal_pose.quaternion.repeat(self.BATCH_SIZE, 1)
        ik_goal = Pose(position=ik_pos, quaternion=ik_quat)
        ik_result = self._ik_solver.solve_batch(
            ik_goal, retract_config=retract_tensor, seed_config=seed_tensor
        )
        succ_arr = ik_result.success.cpu().numpy().reshape(-1)
        if not bool(succ_arr.any()):
            print("    [plan_pose_constrained] IK to goal FAILED")
            return None
        q_sol = ik_result.solution.cpu().numpy()
        if q_sol.ndim == 3:
            q_sol = q_sol[:, 0, :]
        # cuRobo IK can return many local minima; even with seed=start it
        # sometimes picks a far branch (joint 4/6 wrap), making the
        # subsequent plan_single_js trajectory swing wide through air.
        # Pick the successful solution that's *closest* to start in joint
        # space — that's the one whose trajopt path will be shortest.
        candidates = []
        for k in range(len(succ_arr)):
            if not succ_arr[k]:
                continue
            cand = q_sol[k, :self._n_arm].copy()
            self._snap_arm(cand, start_arm)
            candidates.append(cand)
        if not candidates:
            print("    [plan_pose_constrained] IK to goal FAILED")
            return None
        deltas = [float(np.linalg.norm(c - start_arm)) for c in candidates]
        best = int(np.argmin(deltas))
        target_arm = candidates[best]
        delta = deltas[best]
        print(f"    [plan_pose_constrained] IK delta {delta:.2f} rad "
              f"(best of {len(candidates)} feasible)")
        target_full = np.concatenate([
            target_arm.astype(np.float32),
            np.asarray(start_full_qpos[self._n_arm:], dtype=np.float32),
        ])
        ok, traj = self._refine_fingers(
            np.asarray(start_full_qpos, dtype=np.float32), target_full
        )
        if not ok:
            print("    [plan_pose_constrained] plan_single_js FAILED")
            return None
        return traj

    def ik_pose_batch(self, target_link6_poses: np.ndarray) -> np.ndarray:
        """Batched IK reachability check for ``N`` arbitrary link6 poses.

        Requires ``self._ik_solver`` already to be initialized (call ``plan()``
        or ``solve_ik()`` for the current scene at least once first; world
        update follows that solver's cached state).

        Args:
            target_link6_poses: ``(N, 4, 4)`` link6 poses in robot frame.

        Returns:
            ``(N,)`` bool array — True where IK succeeded.
        """
        if self._ik_solver is None:
            raise RuntimeError(
                "ik_pose_batch: _ik_solver not initialized. "
                "Call planner.plan() or planner.solve_ik() first."
            )
        from scipy.spatial.transform import Rotation as _R
        device = self._tensor_args.device
        N = target_link6_poses.shape[0]
        succ = np.zeros(N, dtype=bool)
        for chunk_start in range(0, N, self.BATCH_SIZE):
            chunk = target_link6_poses[chunk_start: chunk_start + self.BATCH_SIZE]
            B = len(chunk)
            if B < self.BATCH_SIZE:
                pad = self.BATCH_SIZE - B
                chunk = np.concatenate(
                    [chunk, np.tile(chunk[:1], (pad, 1, 1))], axis=0)
            positions = chunk[:, :3, 3].astype(np.float32)
            quats_xyzw = _R.from_matrix(chunk[:, :3, :3]).as_quat()
            quats_wxyz = np.concatenate(
                [quats_xyzw[:, 3:4], quats_xyzw[:, :3]], axis=1
            ).astype(np.float32)
            goal = Pose(
                position=torch.tensor(positions, dtype=torch.float32, device=device),
                quaternion=torch.tensor(quats_wxyz, dtype=torch.float32, device=device),
            )
            retract = torch.tensor(
                self._init_state, dtype=torch.float32, device=device
            ).unsqueeze(0).repeat(self.BATCH_SIZE, 1)
            result = self._ik_solver.solve_batch(goal, retract_config=retract)
            succ_chunk = result.success.cpu().numpy().reshape(-1)[:B]
            succ[chunk_start: chunk_start + B] = succ_chunk
        return succ

    def plan_js_to_init(self, scene_cfg: dict,
                        start_arm_qpos: np.ndarray,
                        start_hand_qpos: Optional[np.ndarray] = None,
                        goal_arm_qpos: Optional[np.ndarray] = None
                        ) -> Optional[np.ndarray]:
        """Plan a joint-space retract trajectory:
        ``(start_arm, start_hand) -> init_state``.

        ``start_hand_qpos`` must be in the planner's (curobo URDF) joint order
        — same order as ``plan_result.pregrasp_pose`` / ``grasp_pose``. If
        omitted, defaults to the init_state hand config (fully open). When the
        real hand is at pregrasp, pass ``plan_result.pregrasp_pose`` so the
        planner's collision check matches the actual configuration.

        Uses the existing motion_gen world if available; falls back to a full
        init/rebuild only when scene structure (cuboids, mesh keys, file paths)
        differs from the cached world. The world's `target` mesh is updated to
        wherever the caller has moved it in `scene_cfg` — typical use is to
        reflect the placed object's new resting pose.

        Returns the interpolated traj (T, dof) or None if planning failed.
        """
        world_cfg = _to_curobo_world(scene_cfg)
        if self._motion_gen is None:
            self._init_motion_gen(world_cfg)
        elif self._world_structure_changed(world_cfg):
            self._update_world(world_cfg)
        else:
            self._update_target_pose_only(world_cfg)
        self._cached_world = world_cfg

        if start_hand_qpos is None:
            start_hand_qpos = self._init_state[self._n_arm:]
        start_full = np.concatenate([
            np.asarray(start_arm_qpos[:self._n_arm], dtype=np.float32),
            np.asarray(start_hand_qpos, dtype=np.float32),
        ])
        if goal_arm_qpos is None:
            goal_arm_qpos = self._init_state[:self._n_arm]
        goal_full = np.concatenate([
            np.asarray(goal_arm_qpos[:self._n_arm], dtype=np.float32),
            self._init_state[self._n_arm:].astype(np.float32),
        ])
        ok, traj = self._refine_fingers(start_full, goal_full)
        return traj if ok else None

    def _refine_fingers(self, init_state: np.ndarray, goal_joint: np.ndarray):
        """Joint-space trajopt for full DOF (arm + fingers). Returns (ok, traj)."""
        start = JointState.from_position(
            torch.tensor(init_state, dtype=torch.float32, device=self._tensor_args.device).unsqueeze(0)
        )
        goal = JointState.from_position(
            torch.tensor(goal_joint, dtype=torch.float32, device=self._tensor_args.device).unsqueeze(0)
        )
        result = self._motion_gen.plan_single_js(start_state=start, goal_state=goal, plan_config=self._plan_cfg)
        if not result.success.item():
            if hasattr(result, 'status') and result.status is not None:
                print(f"    [plan_single_js] status={result.status} (act_dist={self._collision_act_dist})")
# Ask cuRobo directly which constraint each state violates.
            try:
                jl = self._motion_gen.kinematics.get_joint_limits()
                jl_lo = jl.position[0].cpu().numpy()
                jl_hi = jl.position[1].cpu().numpy()
                jn = list(self._motion_gen.kinematics.joint_names)
                for label, q in [("start", init_state), ("goal", goal_joint)]:
                    qt = torch.tensor(q, dtype=torch.float32,
                                      device=self._tensor_args.device).unsqueeze(0)
                    js = JointState.from_position(qt)
                    valid, status = self._motion_gen.check_start_state(js)
                    print(f"    [check] {label}: valid={valid} status={status}")
                    qa = np.asarray(q)
                    for i, qi in enumerate(qa[:len(jl_lo)]):
                        if qi < jl_lo[i] - 1e-6 or qi > jl_hi[i] + 1e-6:
                            print(f"      OOB joint[{i}] {jn[i]}: "
                                  f"q={qi:.4f} not in [{jl_lo[i]:.4f}, {jl_hi[i]:.4f}]")
            except Exception as ce:
                print(f"    [check] failed: {ce!r}")
            # Only export debug meshes when start/end state is in collision
            # (valid_query=False). Other fail modes (GRAPH_FAIL after valid
            # query, TRAJOPT_FAIL) skip export — too noisy and unhelpful.
            if (hasattr(result, 'valid_query')
                    and result.valid_query is False):
                self._export_collision_debug(goal_joint)
        if result.success.item():
            return True, result.get_interpolated_plan().position.cpu().numpy()
        return False, None

    def plan_with_seed(self, goal_qpos: np.ndarray, seed_traj: np.ndarray,
                       start_qpos: Optional[np.ndarray] = None,
                       newton_iters: Optional[int] = None):
        """Joint-space trajopt seeded by an EXTERNAL trajectory — mechanism (B).

        Bypasses MotionGen's graph/IK seeding and feeds ``seed_traj`` straight
        into ``js_trajopt_solver`` as seed #0 (the solver pads the remaining
        seeds with linear interpolation and returns the best). A near-feasible
        seed makes trajopt converge in 1 shot instead of stochastically failing.

        Args:
            goal_qpos:  (dof,) goal joint config — typically IK of the actual
                        (off-grid) grasp wrist pose, fingers = pregrasp.
            seed_traj:  (H_seed, dof) seed trajectory (already adjusted to this
                        start/goal); resampled to the solver action_horizon.
            start_qpos: (dof,) start state; defaults to INIT_STATE.

        Returns (success: bool, traj: (T, dof) | None, solve_time: float).
        """
        dev = self._tensor_args.device
        torch.manual_seed(0)          # determinism: frozen seed -> same plan every run
        solver = self._motion_gen.js_trajopt_solver
        H, dof = solver.action_horizon, len(self._init_state)
        if start_qpos is None:
            start_qpos = self._init_state

        start = JointState.from_position(
            torch.tensor(start_qpos, dtype=torch.float32, device=dev).unsqueeze(0))
        goal_js = JointState.from_position(
            torch.tensor(goal_qpos, dtype=torch.float32, device=dev).unsqueeze(0))
        goal = Goal(current_state=start, goal_state=goal_js)

        # Resample the (possibly dense) seed to the solver's action_horizon.
        seed_np = np.asarray(seed_traj, dtype=np.float32)
        xs, xt = np.linspace(0, 1, len(seed_np)), np.linspace(0, 1, H)
        seed_rs = np.stack([np.interp(xt, xs, seed_np[:, j]) for j in range(dof)], axis=1)
        seed_t = torch.tensor(seed_rs, dtype=torch.float32, device=dev).view(1, 1, H, dof)
        seed_js = JointState.from_position(seed_t)

        result = solver.solve_single(goal, seed_traj=seed_js, newton_iters=newton_iters)
        ok = bool(result.success.view(-1)[0].item())
        if not ok:
            return False, None, float(result.solve_time)
        sol = (result.interpolated_solution if result.interpolated_solution is not None
               else result.solution)
        traj = sol.position.cpu().numpy().reshape(-1, dof)
        # interpolated_solution is a fixed-size buffer padded with the final
        # state; trim to the valid length (same as MotionGen.get_interpolated_plan).
        if (result.interpolated_solution is not None
                and result.path_buffer_last_tstep is not None):
            traj = traj[: int(result.path_buffer_last_tstep[0])]
        return True, traj, float(result.solve_time)

    def plan_lift(self, grasp_qpos: np.ndarray, grasp_wrist_world: np.ndarray,
                  scene_lift: dict, lift_h: float = 0.10):
        """Plan the grasp->lift segment as a JOINT trajectory for joint-space
        execution (``executor.execute_lift``), replacing the cartesian-servo lift
        whose kinematic errors are UNRECOVERABLE on xarm (force a full restart).

        Lift goal = the grasp wrist raised by ``lift_h`` in world +z, SAME
        orientation, fingers HELD at ``grasp_qpos``'s hand block. A pure +z lift moves the
        held object straight up (away from the table), so failures here are
        kinematic (the lifted wrist unreachable for this off-grid arm config —
        joint limit / singularity) or scene-collision (shelf/wall above), NOT
        table collision. The held object must already be STRIPPED from
        ``scene_lift`` (it moves with the hand, not the world).

        Returns ``(ok, traj)``; ``ok=False`` (lifted pose IK-unreachable or trajopt
        fail) means the caller SKIPS the lift at PLAN time — no robot crash.
        """
        world_cfg = _to_curobo_world(scene_lift)
        if self._motion_gen is None:
            self._init_motion_gen(world_cfg)
        elif self._world_structure_changed(world_cfg):
            self._update_world(world_cfg)
        else:
            self._update_target_pose_only(world_cfg)
        self._cached_world = world_cfg

        dev = self._tensor_args.device
        lift_w = np.asarray(grasp_wrist_world, dtype=np.float64).copy()
        lift_w[2, 3] += lift_h
        gq = torch.tensor(grasp_qpos, dtype=torch.float32, device=dev)
        r = self._ik_solver.solve_batch(
            _to_curobo_pose(lift_w[None], dev), retract_config=gq.unsqueeze(0))
        if not bool(r.success.view(-1)[0]):
            return False, None                       # lifted wrist unreachable (joint limit)
        q_lift = np.asarray(grasp_qpos, dtype=np.float32).copy()
        arm_q = r.solution.cpu().numpy().reshape(-1)[:self._n_arm]
        self._snap_arm(arm_q, grasp_qpos)          # ±2π wrap toward the grasp config
        q_lift[:self._n_arm] = arm_q
        ok, traj = self._refine_fingers(
            np.asarray(grasp_qpos, dtype=np.float32), q_lift)
        return (ok, traj if ok else None)

    def _export_collision_debug(self, goal_joint: np.ndarray):
        """Export hand collision spheres + world meshes at goal state for
        debugging. Spheres colliding with any world mesh/cube are red, safe
        spheres are green. Each call uses a new sequence number so files
        don't overwrite."""
        try:
            import trimesh
            debug_dir = "/tmp/collision_debug"
            os.makedirs(debug_dir, exist_ok=True)
            # Sequence number per planner instance so successive fails don't
            # overwrite each other's exports.
            if not hasattr(self, "_dbg_seq"):
                self._dbg_seq = 0
            self._dbg_seq += 1
            seq = f"{self._dbg_seq:03d}"

            # Build world trimeshes (obj meshes + table-like cuboids) for
            # sphere collision check.
            world_tms = []
            if self._motion_gen.world_model is not None:
                wm = self._motion_gen.world_model
                for m in (getattr(wm, "mesh", None) or []):
                    _p = getattr(m, "pose", None)
                    pose = np.asarray(_p if _p is not None else [0, 0, 0, 1, 0, 0, 0])
                    file_path = getattr(m, "file_path", None)
                    verts, faces = m.vertices, m.faces
                    if (verts is None or faces is None) and file_path:
                        tm = trimesh.load(file_path, force="mesh")
                    else:
                        if hasattr(verts, "cpu"): verts = verts.cpu().numpy()
                        if hasattr(faces, "cpu"): faces = faces.cpu().numpy()
                        tm = trimesh.Trimesh(vertices=np.asarray(verts),
                                              faces=np.asarray(faces))
                    T = np.eye(4); T[:3, 3] = pose[:3]
                    from scipy.spatial.transform import Rotation as Rot
                    T[:3, :3] = Rot.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
                    tm.apply_transform(T)
                    world_tms.append(tm)
                for c in (getattr(wm, "cuboid", None) or []):
                    pose = np.asarray(c.pose)
                    box = trimesh.creation.box(extents=np.asarray(c.dims))
                    T = np.eye(4); T[:3, 3] = pose[:3]
                    from scipy.spatial.transform import Rotation as Rot
                    T[:3, :3] = Rot.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
                    box.apply_transform(T)
                    world_tms.append(box)

            # Get collision spheres at goal state
            q = torch.tensor(goal_joint, dtype=torch.float32, device=self._tensor_args.device).unsqueeze(0)
            kin = self._motion_gen.kinematics
            spheres = kin.get_robot_as_spheres(q)

            # Self-collision: use motion_gen's OWN self_collision_constraint
            # (rollout_fn.robot_self_collision_constraint) since that is what
            # actually rejects plans. Extract per-sphere contribution via
            # backward gradient.
            self_collide_set = set()
            try:
                mg = self._motion_gen
                qt = torch.tensor(goal_joint, dtype=torch.float32,
                                  device=self._tensor_args.device).unsqueeze(0)
                state = mg.compute_kinematics(JointState.from_position(qt))
                x_sph = state.robot_spheres.unsqueeze(1).clone().requires_grad_(True)
                sc = mg.rollout_fn.robot_self_collision_constraint
                d_self = sc.forward(x_sph)
                d_self.sum().backward()
                g = x_sph.grad[0, 0, :, :3].abs().sum(-1).cpu().numpy()
                mg_pos = x_sph[0, 0, :, :3].detach().cpu().numpy()
                for i, gi in enumerate(g):
                    if gi > 1e-9:
                        self_collide_set.add(tuple(np.round(mg_pos[i], 5)))
            except Exception as se:
                print(f"    [debug] self-collision grad failed: {se!r}")

            # World collision + per-sphere coloring. Sphere is RED if it
            # collides with world mesh OR is in self-collision set.
            margin = float(self._collision_act_dist)
            red, green, n_total, n_world, n_self = [], [], 0, 0, 0
            for sphere_batch in spheres:
                for s in sphere_batch:
                    r = float(s.radius)
                    if r <= 0:
                        continue
                    n_total += 1
                    pos = np.asarray(s.position, dtype=float)
                    world_hit = any(
                        trimesh.proximity.signed_distance(tm, pos[None])[0] > -(r + margin)
                        for tm in world_tms
                    )
                    self_hit = tuple(np.round(pos, 5)) in self_collide_set
                    if world_hit: n_world += 1
                    if self_hit: n_self += 1
                    m = trimesh.creation.icosphere(radius=r, subdivisions=2)
                    m.apply_translation(pos)
                    if world_hit or self_hit:
                        m.visual.vertex_colors = [255, 0, 0, 255]
                        red.append(m)
                    else:
                        m.visual.vertex_colors = [0, 255, 0, 80]
                        green.append(m)
            if red:
                out = os.path.join(debug_dir, f"{seq}_goal_collide.ply")
                trimesh.util.concatenate(red).export(out)
                print(f"    [debug] goal collide spheres "
                      f"(world={n_world}, self={n_self}, total_red={len(red)}/{n_total}) "
                      f"-> {out}")
            if green:
                out = os.path.join(debug_dir, f"{seq}_goal_safe.ply")
                trimesh.util.concatenate(green).export(out)

            # Robot link meshes at goal state (URDF FK via yourdfpy).
            try:
                import yourdfpy
                urdf_path_rel = self._hand_cfg.get("kinematics", {}).get("urdf_path")
                if urdf_path_rel:
                    urdf_path = os.path.join(
                        project_dir, "content", "assets", urdf_path_rel)
                    urdf = yourdfpy.URDF.load(urdf_path)
                    # Reorder cuRobo qpos into yourdfpy's actuated-joint order
                    # by matching joint names. Joints absent from cuRobo are
                    # filled with 0.
                    urdf_jn = list(urdf.actuated_joint_names)
                    curobo_jn = list(self._motion_gen.kinematics.joint_names)
                    curobo_idx = {n: i for i, n in enumerate(curobo_jn)}
                    goal_np = np.asarray(goal_joint)
                    urdf_cfg = np.array([
                        float(goal_np[curobo_idx[jn]]) if jn in curobo_idx else 0.0
                        for jn in urdf_jn
                    ], dtype=np.float32)
                    urdf.update_cfg(urdf_cfg)
                    print(f"    [debug] urdf_cfg (urdf order): "
                          f"{np.round(urdf_cfg, 3).tolist()}")
                    scene = urdf.scene
                    # Flatten Scene into a single Trimesh in world frame
                    # (Scene.dump applies per-geometry transforms first).
                    combined = trimesh.util.concatenate(
                        list(scene.dump()))
                    out = os.path.join(debug_dir, f"{seq}_goal_robot.obj")
                    combined.export(out)
                    print(f"    [debug] robot mesh at goal -> {out}")
            except Exception as ue:
                print(f"    [debug] robot mesh export failed: {ue!r}")
            # Save world meshes + cuboids
            if self._motion_gen.world_model is not None:
                wm = self._motion_gen.world_model
                # Mesh primitives. cuRobo Mesh may store file_path instead of verts/faces.
                meshes = getattr(wm, "mesh", None) or []
                for m in meshes:
                    name = getattr(m, "name", "mesh")
                    _p = getattr(m, "pose", None)
                    pose = np.asarray(_p if _p is not None else [0, 0, 0, 1, 0, 0, 0])
                    file_path = getattr(m, "file_path", None)
                    verts, faces = m.vertices, m.faces
                    if (verts is None or faces is None) and file_path:
                        tm = trimesh.load(file_path, force="mesh")
                    else:
                        if hasattr(verts, "cpu"): verts = verts.cpu().numpy()
                        if hasattr(faces, "cpu"): faces = faces.cpu().numpy()
                        tm = trimesh.Trimesh(vertices=np.asarray(verts), faces=np.asarray(faces))
                    # Apply mesh pose
                    T = np.eye(4)
                    T[:3, 3] = pose[:3]
                    from scipy.spatial.transform import Rotation as Rot
                    T[:3, :3] = Rot.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
                    tm.apply_transform(T)
                    out = os.path.join(debug_dir, f"{seq}_world_mesh_{name}.obj")
                    tm.export(out)
                    print(f"    [debug] World mesh -> {out}")
                # Cuboid primitives (table, shelf walls)
                cubes = getattr(wm, "cuboid", None) or []
                for c in cubes:
                    name = getattr(c, "name", "cube")
                    dims = np.asarray(c.dims)
                    pose = np.asarray(c.pose)  # [x,y,z,qw,qx,qy,qz]
                    box = trimesh.creation.box(extents=dims)
                    T = np.eye(4)
                    T[:3, 3] = pose[:3]
                    from scipy.spatial.transform import Rotation as Rot
                    T[:3, :3] = Rot.from_quat(pose[[4, 5, 6, 3]]).as_matrix()
                    box.apply_transform(T)
                    out = os.path.join(debug_dir, f"{seq}_world_cube_{name}.obj")
                    box.export(out)
                    print(f"    [debug] World cube -> {out}")
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"    [debug] Export failed: {e}")
        finally:
            # Prevent GPU memory accumulation across many fail-time exports.
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    # ── internal pipeline ─────────────────────────────────────────────────────

    def _find_trajectory(self, world_cfg: dict, wrist_se3: np.ndarray, pregrasp: np.ndarray, mode: str):
        """Filter candidates -> motion plan -> finger refinement. Returns (idx, traj, timing)."""
        import time as _time
        timing = {}

        t0 = _time.time()
        collision = self._check_collision(world_cfg, wrist_se3, pregrasp)
        backward = np.zeros(len(wrist_se3), dtype=bool) if self._hand.startswith("inspire") else (wrist_se3[:, :3, :3] @ self._link6_y_in_wrist)[:, 2] < 0.3
        valid = np.where(~(collision | backward))[0]
        timing["collision_check_s"] = round(_time.time() - t0, 3)

        print(f"[planner] total={len(wrist_se3)}  collision={collision.sum()}  backward={backward.sum()}  valid={len(valid)}")

        if len(valid) == 0:
            return None, None, timing

        if mode == "goalset":
            t0 = _time.time()
            local_idx, traj = self._plan_goalset(wrist_se3[valid])
            timing["arm_plan_s"] = round(_time.time() - t0, 3)
            if local_idx is None:
                return None, None, timing
            idx = valid[local_idx]
            goal = traj[-1].copy()
            goal[self._n_arm:] = pregrasp[idx]
            t0 = _time.time()
            ok, traj = self._refine_fingers(self._init_state, goal)
            timing["finger_refine_s"] = round(_time.time() - t0, 3)
            return (idx, traj, timing) if ok else (None, None, timing)

        # batch mode
        timing["arm_plan_s"] = 0.0
        timing["finger_refine_s"] = 0.0
        timing["n_batches"] = 0
        timing["n_refine_attempts"] = 0
        inits = np.tile(self._init_state, (len(valid), 1))
        for start in range(0, len(valid), self.BATCH_SIZE):
            batch = valid[start : start + self.BATCH_SIZE]

            t0 = _time.time()
            success, trajs = self._plan_batch(inits[start : start + len(batch)], wrist_se3[batch])
            timing["arm_plan_s"] += _time.time() - t0
            timing["n_batches"] += 1

            for i, idx in enumerate(batch):
                if not success[i]:
                    continue
                goal = trajs[i, -1].copy()
                goal[self._n_arm:] = pregrasp[idx]
                t0 = _time.time()
                ok, traj = self._refine_fingers(inits[start + i], goal)
                timing["finger_refine_s"] += _time.time() - t0
                timing["n_refine_attempts"] += 1
                if ok:
                    timing["arm_plan_s"] = round(timing["arm_plan_s"], 3)
                    timing["finger_refine_s"] = round(timing["finger_refine_s"], 3)
                    return idx, traj, timing

        timing["arm_plan_s"] = round(timing["arm_plan_s"], 3)
        timing["finger_refine_s"] = round(timing["finger_refine_s"], 3)
        return None, None, timing

    # ── public API ────────────────────────────────────────────────────────────

    def get_candidates(self, scene_cfg: dict, obj_name: str, grasp_version: str,
                        success_only: bool = False, skip_done: bool = False, hand: str = "allegro",
                        scene_id: Optional[str] = None, run_ik: bool = False,
                        cyl_axis_local: Optional[np.ndarray] = None,
                        cyl_yaw_grid: Optional[np.ndarray] = None,
                        scene_type_filter: Optional[str] = None,
                        skip_scenes_with_success: bool = False,
                        tabletop_pose_stem: Optional[str] = None,
                        candidate_order: Optional[list] = None):
        """
        Return all grasp candidates with collision filter applied (no motion planning).

        Args:
            run_ik: if True, also IK-solve each non-filtered candidate so the
                    caller can distinguish IK-failed from filtered-out from
                    fully-valid. Returns ``ik_failed`` mask as 5th value when
                    set; otherwise returns the original 4-tuple.

        Returns (4-tuple by default, 5-tuple if run_ik=True):
            wrist_se3  (N, 4, 4)
            pregrasp   (N, 16)
            grasp_pose (N, 16)
            filtered   (N,) bool — collision OR backward filtered
            ik_failed  (N,) bool — passed filter but IK couldn't reach
                                   (only when run_ik=True)
        """
        obj_pose = cart2se3(scene_cfg["mesh"]["target"]["pose"])
        wrist_se3, pregrasp, grasp, scene_info = load_candidate(
            obj_name, obj_pose, grasp_version,
            skip_done=skip_done, success_only=success_only, hand=hand,
            scene_id=scene_id, scene_type_filter=scene_type_filter,
            skip_scenes_with_success=skip_scenes_with_success,
            tabletop_pose_stem=tabletop_pose_stem,
            candidate_order=candidate_order)
        # Apply cyl expansion so the viewer sees the same candidate pool the
        # planner actually IKs against (otherwise "valid=N" mismatches).
        wrist_se3, pregrasp, grasp, _, scene_info = _expand_candidates_cyl(
            wrist_se3, pregrasp, grasp, None, scene_info,
            obj_pose, cyl_axis_local, cyl_yaw_grid)

        # Early return if no candidates (collision check would crash on empty).
        if len(wrist_se3) == 0:
            print(f"[planner] get_candidates: no candidates loaded (filters too tight)")
            empty_filtered = np.zeros(0, dtype=bool)
            if run_ik:
                return wrist_se3, pregrasp, grasp, empty_filtered, empty_filtered
            return wrist_se3, pregrasp, grasp, empty_filtered

        world_cfg = _to_curobo_world(scene_cfg)
        if self._motion_gen is None:
            self._init_motion_gen(world_cfg)
        else:
            self._update_world(world_cfg)

        collision = self._check_collision(world_cfg, wrist_se3, pregrasp)
        backward  = np.zeros(len(wrist_se3), dtype=bool)
        filtered  = collision | backward
        print(f"[planner] total={len(wrist_se3)}  collision={collision.sum()}  backward={backward.sum()}  valid={(~filtered).sum()}")

        if not run_ik:
            return wrist_se3, pregrasp, grasp, filtered

        # Also IK-check the non-filtered candidates — BOTH the grasp pose AND
        # the grasp+5cm lift pose (mirrors planner.plan()'s funnel). A candidate
        # only counts as "valid" if it passes both, otherwise viewer shows it
        # yellow (IK_FAIL).
        ik_failed = np.zeros(len(wrist_se3), dtype=bool)
        valid_idx = np.where(~filtered)[0]
        if len(valid_idx) > 0:
            world_no_target = dict(world_cfg)
            world_no_target["mesh"] = {}
            if self._ik_solver is None:
                self._init_ik_solver(world_no_target)
            else:
                self._ik_solver.update_world(WorldConfig.from_dict(world_no_target))

            def _run_ik(poses):
                """Returns success bool array of length len(poses)."""
                out = np.zeros(len(poses), dtype=bool)
                for cs in range(0, len(poses), self.BATCH_SIZE):
                    chunk = poses[cs : cs + self.BATCH_SIZE]
                    B = len(chunk)
                    if B < self.BATCH_SIZE:
                        pad = self.BATCH_SIZE - B
                        chunk = np.concatenate(
                            [chunk, np.tile(chunk[:1], (pad, 1, 1))], axis=0)
                    goal = _to_curobo_pose(chunk, self._tensor_args.device)
                    retract = torch.tensor(
                        self._init_state, dtype=torch.float32,
                        device=self._tensor_args.device,
                    ).unsqueeze(0).repeat(self.BATCH_SIZE, 1)
                    r = self._ik_solver.solve_batch(goal, retract_config=retract)
                    succ = r.success.cpu().numpy()
                    if succ.ndim > 1:
                        succ = succ.reshape(-1)
                    out[cs : cs + B] = succ[:B]
                return out

            # Grasp pose IK
            grasp_succ = _run_ik(wrist_se3[valid_idx])
            # Lift pose IK (z + 5cm, matches planner.plan() lift check)
            lift_poses = wrist_se3[valid_idx].copy()
            lift_poses[:, 2, 3] += 0.05
            lift_succ = _run_ik(lift_poses)
            for j, idx in enumerate(valid_idx):
                if not (grasp_succ[j] and lift_succ[j]):
                    ik_failed[idx] = True
            n_grasp_fail = int((~grasp_succ).sum())
            n_lift_fail = int((grasp_succ & ~lift_succ).sum())
            print(f"[planner] IK-fail among valid: "
                  f"{int(ik_failed.sum())}/{len(valid_idx)} "
                  f"(grasp_fail={n_grasp_fail}, lift_fail_only={n_lift_fail})")
        return wrist_se3, pregrasp, grasp, filtered, ik_failed

    def plan_all(self, scene_cfg: dict, obj_name: str, grasp_version: str,
                 stop_on_first: bool = True, hand: str = "allegro"):
        """
        Plan trajectories for all candidates (for visualization / debugging).

        Args:
            stop_on_first: If True (default), stop after first successful grasp.
                           If False, attempt planning for ALL valid candidates.

        Returns:
            wrist_se3    (N, 4, 4)
            grasp_pose   (N, 16)
            succ_mask    (N,) bool — trajectory planning success
            collision    (N,) bool — collision or backward filtered
            traj_list    list[N] of (T, dof) arrays or None
        """
        import time as _time
        t_total = _time.time()

        t0 = _time.time()
        obj_pose = cart2se3(scene_cfg["mesh"]["target"]["pose"])
        wrist_se3, pregrasp, grasp, scene_info = load_candidate(obj_name, obj_pose, grasp_version, hand=hand)
        print(f"[planner] load candidates: {_time.time() - t0:.2f}s ({len(wrist_se3)} candidates)")

        t0 = _time.time()
        world_cfg = _to_curobo_world(scene_cfg)
        if self._motion_gen is None:
            self._init_motion_gen(world_cfg)
        else:
            self._update_world(world_cfg)
        print(f"[planner] init/update motion gen: {_time.time() - t0:.2f}s")

        t0 = _time.time()
        N = len(wrist_se3)
        collision = self._check_collision(world_cfg, wrist_se3, pregrasp)
        backward = np.zeros(len(wrist_se3), dtype=bool) if self._hand.startswith("inspire") else (wrist_se3[:, :3, :3] @ self._link6_y_in_wrist)[:, 2] < 0.3
        filtered = collision | backward
        valid = np.where(~filtered)[0]
        print(f"[planner] collision check: {_time.time() - t0:.2f}s")

        print(f"[planner] total={N}  collision={collision.sum()}  backward={backward.sum()}  valid={len(valid)}")

        succ_mask = np.zeros(N, dtype=bool)
        traj_list = [None] * N

        if len(valid) == 0:
            return wrist_se3, pregrasp, grasp, succ_mask, filtered, traj_list

        inits = np.tile(self._init_state, (len(valid), 1))
        has_succ = False
        t_batch_total = 0.0
        t_refine_total = 0.0
        n_batches = 0
        n_refines = 0

        for start in range(0, len(valid), self.BATCH_SIZE):
            batch = valid[start : start + self.BATCH_SIZE]
            if has_succ:
                break

            t0 = _time.time()
            success, trajs = self._plan_batch(
                inits[start : start + len(batch)], wrist_se3[batch]
            )
            t_batch_total += _time.time() - t0
            n_batches += 1
            print(f"[planner] batch {n_batches}: {success.sum()}/{len(batch)} arm plan success ({_time.time() - t0:.2f}s)")

            if trajs is not None and trajs.ndim == 2:
                trajs = trajs[np.newaxis]

            for i, idx in enumerate(batch):
                if has_succ:
                    break
                if not success[i]:
                    continue
                goal = trajs[i, -1].copy()
                goal[self._n_arm:] = pregrasp[idx]
                t1 = _time.time()
                ok, traj = self._refine_fingers(self._init_state, goal)
                t_refine_total += _time.time() - t1
                n_refines += 1
                print(f"[planner] plan_single #{n_refines} (idx={idx}): {'ok' if ok else 'fail'} ({_time.time() - t1:.2f}s)")
                if ok:
                    succ_mask[idx] = True
                    traj_list[idx] = traj
                    has_succ = True

        print(f"[planner] timing: plan_batch={t_batch_total:.2f}s ({n_batches} calls)  plan_single={t_refine_total:.2f}s ({n_refines} calls)")
        print(f"[planner] total plan_all: {_time.time() - t_total:.2f}s")

        return wrist_se3, pregrasp, grasp, succ_mask, filtered, traj_list

    def plan(self, scene_cfg: dict, obj_name: str, grasp_version: str,
             mode: str = "batch", seed: Optional[int] = None,
             skip_done: bool = True, success_only: bool = False,
             hand: str = "allegro",
             scene_id: Optional[str] = None,
             openpose_pose_stem: Optional[str] = None,
             cyl_axis_local: Optional[np.ndarray] = None,
             cyl_yaw_grid: Optional[np.ndarray] = None,
             scene_type_filter: Optional[str] = None,
             skip_scenes_with_success: bool = False,
             tabletop_pose_stem: Optional[str] = None,
             candidate_order: Optional[list] = None,
             priority_map: Optional[dict] = None) -> PlanResult:
        """If ``openpose_pose_stem`` is given (e.g. ``"002"``), loads
        ``openpose_{stem}.npy`` per candidate and uses it as the approach-end
        finger config (instead of pregrasp). Candidates without that openpose
        file fall back to pregrasp.

        If ``cyl_axis_local`` + ``cyl_yaw_grid`` are given, expand each
        candidate by N_cyl rotations around the object's symmetry axis
        (multiplies candidate pool for cylinder objects).
        """
        import time as _time

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        # 1. Load candidates
        t0 = _time.time()
        obj_pose = cart2se3(scene_cfg["mesh"]["target"]["pose"])
        wrist_se3, pregrasp, grasp, scene_info = load_candidate(
            obj_name, obj_pose, grasp_version,
            skip_done=skip_done, success_only=success_only,
            hand=hand, scene_id=scene_id,
            scene_type_filter=scene_type_filter,
            skip_scenes_with_success=skip_scenes_with_success,
            tabletop_pose_stem=tabletop_pose_stem,
            candidate_order=candidate_order)
        if openpose_pose_stem is not None:
            from autodex.utils.path import load_openpose_for_candidates
            openpose_list = load_openpose_for_candidates(
                obj_name, scene_info, hand, grasp_version, openpose_pose_stem)
        else:
            openpose_list = [None] * len(pregrasp)
        # Expand candidates by cyl_yaw (cylinder objects only). Pregrasp/grasp/
        # openpose finger configs are replicated since the cylinder is invariant
        # under symmetry-axis rotation.
        wrist_se3, pregrasp, grasp, openpose_list, scene_info = (
            _expand_candidates_cyl(wrist_se3, pregrasp, grasp, openpose_list,
                                    scene_info, obj_pose,
                                    cyl_axis_local, cyl_yaw_grid))
        # Use openpose for the approach-end finger config; fall back to
        # pregrasp where openpose is missing.
        approach_fingers = np.array([
            (op if op is not None else pg)
            for op, pg in zip(openpose_list, pregrasp)
        ])
        t_load = _time.time() - t0

        if len(wrist_se3) == 0:
            print(f"[planner] No candidates available (all done or no success)")
            return PlanResult(
                success=False, traj=None, wrist_se3=None,
                pregrasp_pose=None, grasp_pose=None, scene_info=[],
                timing={"load_candidates_s": round(t_load, 3), "n_total": 0},
            )

        # 2. World setup (motion_gen for trajectory, ik_solver for IK)
        t0 = _time.time()
        world_cfg = _to_curobo_world(scene_cfg)
        if self._motion_gen is None:
            self._init_motion_gen(world_cfg)
        elif self._world_structure_changed(world_cfg):
            self._update_world(world_cfg)
        else:
            # Only target mesh pose changed — in-place pose update keeps roadmap.
            self._update_target_pose_only(world_cfg)
        self._cached_world = world_cfg
        world_cfg_no_target = dict(world_cfg)
        world_cfg_no_target["mesh"] = {}
        if self._ik_solver is None:
            self._init_ik_solver(world_cfg_no_target)
        else:
            self._ik_solver.update_world(WorldConfig.from_dict(world_cfg_no_target))
        t_world = _time.time() - t0

        # 3. Filter: backward + hand-table collision
        t0 = _time.time()
        backward = np.zeros(len(wrist_se3), dtype=bool) if self._hand.startswith("inspire") else (wrist_se3[:, :3, :3] @ self._link6_y_in_wrist)[:, 2] < 0.3
        collision = self._check_collision(world_cfg_no_target, wrist_se3, pregrasp)
        valid = np.where(~(backward | collision))[0]
        t_filter = _time.time() - t0

        N = len(wrist_se3)
        print(f"[planner] total={N}  backward={backward.sum()}  collision={collision.sum()}  valid={len(valid)}")

        def _fail_result(timing):
            return PlanResult(
                success=False, traj=None, wrist_se3=None,
                pregrasp_pose=pregrasp[0], grasp_pose=grasp[0], scene_info=[],
                timing=timing,
            )

        base_timing = {
            "load_candidates_s": round(t_load, 3),
            "world_setup_s": round(t_world, 3),
            "filter_s": round(t_filter, 3),
            "n_total": N,
            "n_backward": int(backward.sum()),
            "n_collision": int(collision.sum()),
            "n_valid": int(len(valid)),
        }

        if len(valid) == 0:
            return _fail_result({**base_timing, "ik_s": 0.0, "plan_single_js_s": 0.0})

        # 4. IK solve on valid candidates
        t0 = _time.time()
        ik_success = np.zeros(N, dtype=bool)
        ik_qpos = np.full((N, len(self._init_state)), np.nan)
        for chunk_start in range(0, len(valid), self.BATCH_SIZE):
            chunk_idx = valid[chunk_start : chunk_start + self.BATCH_SIZE]
            chunk_poses = wrist_se3[chunk_idx]
            B = len(chunk_poses)
            if B < self.BATCH_SIZE:
                pad = self.BATCH_SIZE - B
                chunk_poses = np.concatenate(
                    [chunk_poses, np.tile(chunk_poses[:1], (pad, 1, 1))], axis=0)
            goal = _to_curobo_pose(chunk_poses, self._tensor_args.device)
            # Retract toward init_state so IK solutions stay near start config
            B_padded = chunk_poses.shape[0]
            retract = torch.tensor(
                self._init_state, dtype=torch.float32, device=self._tensor_args.device
            ).unsqueeze(0).repeat(B_padded, 1)
            result = self._ik_solver.solve_batch(goal, retract_config=retract)
            succ = result.success.cpu().numpy()[:B]
            q_sol = result.solution.cpu().numpy()[:B]
            if q_sol.ndim == 3:
                q_sol = q_sol[:, 0, :]
            for i, idx in enumerate(chunk_idx):
                if succ[i]:
                    arm_q = q_sol[i, :self._n_arm].copy()
                    self._snap_arm(arm_q, self._init_state)
                    # Reject IK whose any arm joint sits outside ±π — those
                    # extreme configs are on a far IK branch where the
                    # constrained lift (PoseCostMetric hold xy+rotation) has
                    # no near-start solution. plan_pose_constrained reliably
                    # hits IK_FAIL for them, dropping us to cartesian.
                    if np.any(np.abs(arm_q) > np.pi):
                        continue
                    ik_success[idx] = True
                    ik_qpos[idx, :self._n_arm] = arm_q
                    ik_qpos[idx, self._n_arm:] = approach_fingers[idx]
        t_ik = _time.time() - t0

        # Lift IK check: verify the wrist can rise a bit (avoids candidates
        # already at joint limit). Only checks small +z — exact lift height
        # is handled by executor; we just need monotonic-up to be possible.
        LIFT_HEIGHT = 0.03
        ik_valid_pre = np.where(ik_success)[0]
        if len(ik_valid_pre) > 0:
            lift_poses = wrist_se3[ik_valid_pre].copy()
            lift_poses[:, 2, 3] += LIFT_HEIGHT
            for chunk_start in range(0, len(ik_valid_pre), self.BATCH_SIZE):
                chunk = ik_valid_pre[chunk_start : chunk_start + self.BATCH_SIZE]
                chunk_poses = lift_poses[chunk_start : chunk_start + len(chunk)]
                B = len(chunk_poses)
                if B < self.BATCH_SIZE:
                    pad = self.BATCH_SIZE - B
                    chunk_poses = np.concatenate(
                        [chunk_poses, np.tile(chunk_poses[:1], (pad, 1, 1))], axis=0)
                goal = _to_curobo_pose(chunk_poses, self._tensor_args.device)
                result = self._ik_solver.solve_batch(goal)
                lift_succ = result.success.cpu().numpy()[:B]
                for i, idx in enumerate(chunk):
                    if not lift_succ[i]:
                        ik_success[idx] = False
            n_lift_fail = len(ik_valid_pre) - int(ik_success.sum())
            if n_lift_fail > 0:
                print(f"[planner] Lift IK check: {n_lift_fail} candidates failed (z+{LIFT_HEIGHT}m unreachable)")

        ik_valid = np.where(ik_success)[0]
        n_ik_success = len(ik_valid)
        print(f"[planner] IK: {n_ik_success}/{len(valid)} success (after lift check)")
        base_timing["ik_s"] = round(t_ik, 3)
        base_timing["n_ik_success"] = n_ik_success
        base_timing["n_valid"] = int(len(valid))

        if n_ik_success == 0:
            return _fail_result({**base_timing, "plan_single_js_s": 0.0})

        # 5. plan_single_js for each IK-reachable candidate until success.
        # Ordering priority:
        #   priority_map > candidate_order > random shuffle.
        # priority_map: dict[(type, sid, gid) → score]. Sort ik_valid desc
        # by score so the IK-passing candidate with highest coverage tries first.
        if priority_map is not None:
            def _score(idx):
                key = tuple(str(x) for x in scene_info[idx])
                return -priority_map.get(key, 0)   # negative for desc sort
            ik_valid = np.array(sorted(ik_valid, key=_score), dtype=ik_valid.dtype)
        elif candidate_order is None:
            np.random.shuffle(ik_valid)
        t0 = _time.time()
        n_attempts = 0
        for idx in ik_valid:
            t1 = _time.time()
            ok, traj = self._refine_fingers(self._init_state, ik_qpos[idx])
            n_attempts += 1
            print(f"[planner] plan_single_js #{n_attempts} (idx={idx}): "
                  f"{'ok' if ok else 'fail'} ({_time.time() - t1:.2f}s)")
            if ok:
                t_plan = _time.time() - t0
                print(f"[planner] Selected candidate #{idx}/{N}")
                return PlanResult(
                    success=True, traj=traj, wrist_se3=wrist_se3[idx],
                    pregrasp_pose=pregrasp[idx], grasp_pose=grasp[idx],
                    scene_info=scene_info[idx],
                    timing={**base_timing, "plan_single_js_s": round(t_plan, 3),
                            "n_plan_attempts": n_attempts,
                            "candidate_idx": int(idx)},
                    openpose_pose=openpose_list[idx],
                )

        t_plan = _time.time() - t0
        return _fail_result({**base_timing, "plan_single_js_s": round(t_plan, 3),
                             "n_plan_attempts": n_attempts})
