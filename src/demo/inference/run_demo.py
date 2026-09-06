#!/usr/bin/env python3
"""One inference: pick the object up and drop it into a fixed box.

This is ``src/demo/banana_test/run_demo.py`` with everything that exists for
the placement study taken out — the ``--loc``/``--ori`` grid tags, the running
tally / summary report, the tabletop-catalogue grasp source and its reachable-
pose bookkeeping, the human label prompt, the video recording and the
repeat-trial loop.  What is left is the motion the demo is judged on:

    perceive → pick a fixed successful Inspire grasp → approach → grasp →
    lift 15 cm → rotate J0 at constant height toward the fixed box → drop →
    retreat

The proven helpers are imported from the banana runner rather than copied, so
the two demos cannot drift apart in the parts that actually touch the robot.

Example:
    python src/demo/inference/run_demo.py --obj apple --arm franka \\
        --box-xy 0.64 0.72 --execute
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
# The deployment keeps ParaDex as a sibling checkout in the specialised
# robotics environments.  Match the other demo entry points so ``--help`` and
# the real runner work without a separate editable ParaDex install.
for _paradex_root in (
    os.environ.get("AUTODEX_PARADEX_ROOT"),
    str(Path.home() / "paradex"),
):
    _path = Path(_paradex_root).expanduser() if _paradex_root else None
    if _path is not None and (_path / "paradex").is_dir():
        sys.path.insert(0, str(_path))
        break

from src.demo.inference.grasp_library import (
    DemoGrasp,
    demo_symmetry_rotations,
    load_demo_grasps,
)
from src.demo.continuous_basket.basket_marker import DEFAULT_BASKET_MARKER_ID


# Demo episodes get their own namespace so a demo run never lands in the v8
# collection tree.  The fallback still reads BOTH namespaces (see
# ``_v8_attempted_scene_infos``) to advance to the next unattempted candidate.
DEMO_VERSION = "v8_demo"
# Collection episodes written by ``src/execution/run_auto.py``.  Read-only here.
COLLECTION_VERSION = "v8"
DEFAULT_PC_LIST = ["capture1", "capture2", "capture3", "capture5", "capture6"]
# The grasp library and the planning meshes both come from the v8 tree; there
# is no catalogue selection here, so this is a constant rather than a flag.
GRASP_ASSET_VERSION = "v8"
CARRY_CLEARANCE_M = 0.05
# Robot-frame convention: +X is forward and +Y is left.  Therefore +60 deg
# is the requested point 60 deg counter-clockwise from the robot's forward
# direction (negative would be clockwise from it).
JOINT0_DROP_BEARING_DEG = 60.0
# Surveyed from the release target in the successful 20260901_013533 run.
# This is the fixed box centre in the robot frame; no marker lookup is needed
# while the box stays in this physical configuration.
DEFAULT_BOX_XY = (0.6387998711482535, 0.715814976282987)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _warn_if_sil_loss_high(timing: dict | None, nominal_max: float) -> None:
    """Say so whenever a pose was kept only because the gate was relaxed.

    A rejected pose stops the run and is impossible to miss; an accepted bad
    one is silent, and the demo then plans a grasp against a mesh that is not
    where the object is. Keep the number in front of the operator.
    """
    loss = (timing or {}).get("sil_loss")
    if loss is None or loss <= nominal_max:
        return
    print(f"[perception] WARNING sil loss {loss:.6f} > {nominal_max} — pose "
          f"accepted anyway; verify the overlay before trusting the grasp")


def _object_vertices(scene_cfg: dict) -> np.ndarray:
    """Load the planning mesh vertices in its object-local frame once."""
    import trimesh

    mesh_path = scene_cfg["mesh"]["target"]["file_path"]
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) == 0:
        raise ValueError(f"no usable vertices in planning mesh: {mesh_path}")
    return vertices


def _object_clearance(vertices: np.ndarray, T_obj: np.ndarray,
                      table_z: float) -> float:
    """Lowest mesh vertex height above the table for one object pose."""
    T_obj = np.asarray(T_obj, dtype=np.float64)
    return float(np.min(vertices @ T_obj[2, :3] + T_obj[2, 3]) - table_z)


def _carry_path_clearance(planner, traj: np.ndarray, T_obj_in_wrist: np.ndarray,
                          vertices: np.ndarray, table_z: float) -> float:
    """Minimum held-object clearance over an entire planned carry path.

    The planner removes the target mesh while it is held, correctly avoiding
    self-collision with the hand but consequently cannot see object-table
    contact.  Evaluate the known object-in-wrist transform at every FK
    waypoint to make that omitted collision condition explicit.
    """
    import torch
    from scipy.spatial.transform import Rotation

    qpos = np.asarray(traj, dtype=np.float32)
    kin = planner._motion_gen.kinematics.get_state(
        torch.tensor(qpos, dtype=torch.float32,
                     device=planner._tensor_args.device))
    wrist_xyz = kin.ee_position.detach().cpu().numpy()
    wrist_q = kin.ee_quaternion.detach().cpu().numpy()  # wxyz
    wrist_R = Rotation.from_quat(
        np.column_stack([wrist_q[:, 1], wrist_q[:, 2],
                         wrist_q[:, 3], wrist_q[:, 0]])).as_matrix()
    T_oiw = np.asarray(T_obj_in_wrist, dtype=np.float64)
    object_R = wrist_R @ T_oiw[:3, :3]
    object_xyz = wrist_xyz + np.einsum("nij,j->ni", wrist_R, T_oiw[:3, 3])
    # Only the z row of each waypoint's rotation contributes to height;
    # contracting the row index too would mix x/y into the clearance.
    min_z = (np.einsum("nj,vj->nv", object_R[:, 2, :], vertices) +
             object_xyz[:, 2:3]).min()
    return float(min_z - table_z)


def _carry_wrist_locked(T_obj_now: np.ndarray, T_obj_in_wrist: np.ndarray,
                        target_xyz: np.ndarray) -> np.ndarray:
    """Move the held object over the target without changing attitude/z."""
    T_obj_target = np.asarray(T_obj_now, dtype=np.float64).copy()
    T_obj_target[:2, 3] = np.asarray(target_xyz, dtype=np.float64)[:2]
    return T_obj_target @ np.linalg.inv(T_obj_in_wrist)


def _arc_delta(current_angle: float, target_angle: float,
               joint0: float = 0.0,
               joint0_limits: tuple[float, float] | None = None) -> float:
    """Base-Z rotation from the current bearing to the target one.

    The transfer used to be hard-wired clockwise, which is the short way round
    only while the target sits clockwise of the held object.  A target that is
    counter-clockwise of it (e.g. a +60 deg box against a +19 deg object) then
    became a 334 deg sweep that no J0 limit can hold.  Take the short way and
    only fall back to the long one when the short one leaves the joint range.
    """
    cw = -float(np.mod(current_angle - target_angle, 2.0 * np.pi))
    ccw = cw + 2.0 * np.pi
    short, long = (cw, ccw) if abs(cw) <= abs(ccw) else (ccw, cw)
    if joint0_limits is None:
        return short
    lo, hi = joint0_limits
    if lo <= joint0 + short <= hi:
        return short
    if lo <= joint0 + long <= hi:
        return long
    return short


def _joint0_limits(planner) -> tuple[float, float] | None:
    """(lo, hi) for arm joint 0 from the planner's kinematics, if available."""
    try:
        limits = planner._motion_gen.kinematics.get_joint_limits().position
        lo = float(limits[0][0])
        hi = float(limits[1][0])
    except Exception:
        return None
    return (lo, hi) if hi > lo else None


def _joint0_arc_trajectory(planner, fk_wrist, start_full_qpos: np.ndarray,
                            arm_dof: int, T_obj_in_wrist: np.ndarray,
                            target_angle_rad: float,
                            reference_xy: np.ndarray | None = None
                            ) -> tuple[np.ndarray, dict]:
    """Build a constant-height transfer that changes only arm joint 0.

    Joint 0 is the robot-base Z-axis.  FK first finds the *held object's*
    current polar angle in the robot frame, not the wrist angle.  We then
    invert that relation to form the J0 target at which the object has the
    requested fixed robot-frame polar angle.  J0 takes the shorter way round
    the base Z axis (the longer one only when the short sweep would leave the
    joint range); every other arm and hand joint stays measured/current.
    The target J0 is formed as ``measured J0 + delta`` and is never reused
    from the dry-run trajectory.
    """
    start = np.asarray(start_full_qpos, dtype=np.float32)
    if start.ndim != 1 or len(start) <= arm_dof:
        raise ValueError("start_full_qpos must contain arm and hand joints")
    T_wrist_start = fk_wrist(planner, start)
    T_obj_start = T_wrist_start @ np.asarray(T_obj_in_wrist, dtype=np.float64)
    object_angle = float(np.arctan2(T_obj_start[1, 3], T_obj_start[0, 3]))
    target_angle = float(target_angle_rad)
    measured_joint0 = float(start[0])
    delta = _arc_delta(object_angle, target_angle, measured_joint0,
                       _joint0_limits(planner))
    target_joint0 = measured_joint0 + delta
    n_waypoints = max(2, int(np.ceil(abs(delta) / np.deg2rad(1.0))) + 1)
    traj = np.tile(start, (n_waypoints, 1))
    # Deliberately the sole changing column: J0 / link 0's base yaw.
    traj[:, 0] = np.linspace(measured_joint0, target_joint0, n_waypoints,
                             dtype=np.float32)
    T_wrist_end = fk_wrist(planner, traj[-1])
    T_obj_end = T_wrist_end @ np.asarray(T_obj_in_wrist, dtype=np.float64)
    reference_error = None
    if reference_xy is not None:
        reference_xy = np.asarray(reference_xy, dtype=np.float64).reshape(2)
        reference_error = float(np.linalg.norm(T_obj_end[:2, 3] - reference_xy))
    return traj, {
        "joint": 0,
        "measured_joint0_deg": float(np.rad2deg(measured_joint0)),
        "target_joint0_deg": float(np.rad2deg(target_joint0)),
        "object_angle_deg": float(np.rad2deg(object_angle)),
        "target_angle_deg": float(np.rad2deg(target_angle)),
        "joint0_delta_deg": float(np.rad2deg(delta)),
        "direction": "clockwise" if delta < 0 else "counter_clockwise",
        "n_waypoints": n_waypoints,
        "predicted_object_xy": T_obj_end[:2, 3].tolist(),
        # Diagnostic only: J0-only motion controls bearing, not radius.
        "box_xy_error_m": reference_error,
    }


def _v8_attempted_scene_infos(obj_name: str) -> set[tuple[str, str, str]]:
    """Candidate keys already executed against this object's v8 pool.

    Both namespaces count: the collection tree written by ``run_auto`` and this
    demo's own episodes.  The candidate is the same physical grasp either way,
    so an attempt in one must not be re-served by the other.
    """
    from autodex.utils.path import project_dir

    attempted: set[tuple[str, str, str]] = set()
    for version in (COLLECTION_VERSION, DEMO_VERSION):
        root = Path(project_dir) / "experiment" / version / "inspire" / obj_name
        if not root.is_dir():
            continue
        for result_path in root.glob("*/result.json"):
            try:
                result = json.loads(result_path.read_text())
                info = result.get("scene_info")
                if isinstance(info, (list, tuple)) and len(info) == 3:
                    attempted.add(tuple(str(part) for part in info))
            except (OSError, ValueError, TypeError):
                continue
    return attempted


# Palm normal of the right Inspire hand in the wrist frame. The URDF joins
# ``wrist`` to the hand ``base_link`` with identity, so the candidate wrist
# poses ARE hand-base poses, and in that frame the four fingers extend along
# -z while flexing moves every fingertip toward +y (checked by FK on
# inspire_hand_right.urdf: d(tip)/d(+q) = [0, 0.99, 0.13]). +y is therefore
# the direction the palm faces, and the palm points at the table when its
# world z is negative.
PALM_NORMAL_WRIST = np.array([0.0, 1.0, 0.0])


def raise_floor_scene_cfg(scene_cfg: dict, floor_z: float) -> dict:
    """Copy ``scene_cfg`` with its table cuboid lifted so its top is ``floor_z``.

    The return to home happens with the object already released, so the only
    thing the planner must respect on the way back is "stay high".  Raising
    the table into a virtual floor states that directly, instead of hoping the
    home trajectory happens to arc over the setup.
    """
    import copy

    cfg = copy.deepcopy(scene_cfg)
    cuboids = cfg.get("cuboid") or {}
    for name, box in cuboids.items():
        if name != "table":
            continue
        dims = list(box["dims"])
        pose = list(box["pose"])
        pose[2] = floor_z - dims[2] / 2.0
        cuboids[name] = dict(box, dims=dims, pose=pose)
    cfg["cuboid"] = cuboids
    return cfg


def palm_down_score(wrist_world: np.ndarray) -> np.ndarray:
    """+1 when the palm looks straight down, -1 when it looks straight up."""
    normal = np.asarray(wrist_world)[..., :3, :3] @ PALM_NORMAL_WRIST
    return -normal[..., 2]


def _palm_weighted_order(scores: np.ndarray, seed: int | None,
                         exponent: float) -> np.ndarray:
    """Rank indices by a palm-down-weighted random draw (largest first).

    Weighted sampling without replacement (Efraimidis-Spirakis): draw
    ``u ~ U(0,1)`` per item and sort by ``u ** (1 / w)``.  With ``exponent``
    0 every weight is 1 and this is a plain shuffle; raising it concentrates
    the draw on palm-down grasps without ever excluding the others, so a run
    that needs an unusual grasp can still reach it.
    """
    scores = np.asarray(scores, dtype=np.float64)
    weights = np.clip((scores + 1.0) / 2.0, 1e-6, None) ** float(exponent)
    keys = np.random.default_rng(seed).random(len(scores)) ** (1.0 / weights)
    return np.argsort(-keys)


def _remaining_v8_candidate_order(obj_name: str, object_pose: np.ndarray, *,
                                  seed: int | None = None,
                                  palm_weight: float = 0.0):
    """Return v8 candidates not recorded by an earlier v8 demo episode."""
    from autodex.utils.path import load_candidate

    wrist, _pregrasp, _grasp, all_info = load_candidate(
        obj_name, object_pose, "v8", hand="inspire", shuffle=False,
        skip_done=False, success_only=False)
    attempted = _v8_attempted_scene_infos(obj_name)
    keep = [i for i, info in enumerate(all_info)
            if tuple(str(part) for part in info) not in attempted]
    if palm_weight > 0 and len(keep) and len(wrist):
        # load_candidate already returns ``obj_pose @ wrist_se3``, so these are
        # world poses and can be scored directly.
        scores = palm_down_score(wrist[keep])
        keep = [keep[i] for i in _palm_weighted_order(scores, seed, palm_weight)]
    remaining = [tuple(str(part) for part in all_info[i]) for i in keep]
    return remaining, len(all_info), len(attempted)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--obj", required=True, help="object name on the table")
    parser.add_argument("--arm", choices=["xarm", "franka"], default="franka")
    parser.add_argument("--execute", action="store_true",
                        help="drive the robot; without it the run stops after "
                             "the dry run and reports what it would do")
    parser.add_argument("--grasp-order", choices=["shuffle", "library"],
                        default="shuffle",
                        help="order the success library is planned in. "
                             "'library' keeps the fixed newest-first priority, "
                             "which replays the same grasp every run — including "
                             "one that just failed physically. 'shuffle' (default) "
                             "permutes it per run so a failed grasp is unlikely "
                             "to be retried first")
    parser.add_argument("--grasp-seed", type=int, default=None,
                        help="seed for --grasp-order shuffle; omit for a fresh "
                             "permutation each run. The seed actually used is "
                             "recorded in result.json")
    parser.add_argument("--palm-down-weight", type=float, default=3.0,
                        help="how strongly a palm-down grasp is favoured when "
                             "the library and the v8 pool are ordered. 0 = "
                             "unbiased shuffle; larger concentrates the draw on "
                             "grasps whose palm looks at the table, without "
                             "ever removing the others")
    parser.add_argument("--return-floor", type=float, default=0.15,
                        help="virtual floor height (m, robot frame) used ONLY "
                             "for the post-release return to home: the table "
                             "obstacle is raised to this height so the arm "
                             "comes back above it. 0 keeps the real table")
    parser.add_argument("--max-v8-attempts", type=int, default=0,
                        help="cap on v8-fallback attempts after the "
                             "success-library budget is spent. 0 (default) "
                             "keeps drawing the next candidate until the pool "
                             "is empty: an approach or lift failure drops that "
                             "candidate and the next one is planned, never "
                             "ending the trial while candidates remain")
    parser.add_argument("--max-grasp-attempts", type=int, default=4,
                        help="how many candidates the SUCCESS LIBRARY may spend "
                             "before the run hands over to the v8 pool. Each "
                             "attempt plans one approach AND its lift/transfer; "
                             "a candidate whose lift does not plan is dropped "
                             "and the next one tried")
    parser.add_argument("--symmetry-samples", type=int, default=8,
                        help="number of approach variants for a continuously "
                             "symmetric object (default: 8; discrete "
                             "symmetries are always enumerated exactly)")
    parser.add_argument("--perception", choices=["foundpose", "da3-fpose"],
                        default="foundpose",
                        help="foundpose = the init_daemon template pipeline "
                             "(default). da3-fpose = the legacy SAM3 + DA3 "
                             "depth + FoundationPose pipeline, which registers "
                             "against depth instead of matching RGB templates")
    parser.add_argument("--prompt", default="object on the checkerboard")
    parser.add_argument("--sil-loss-max", type=float, default=0.003,
                        help="reject a refined pose whose silhouette loss "
                             "exceeds this (orchestrator default: 0.003)")
    parser.add_argument("--ignore-sil-loss", action="store_true",
                        help="accept the refined pose whatever its silhouette "
                             "loss. The pose may be wrong — only for objects "
                             "the optimiser cannot fit (thin symmetric "
                             "cylinders) and never unattended with --execute")
    parser.add_argument("--sil-iters", type=int, default=100)
    parser.add_argument("--sil-lr", type=float, default=0.002)
    parser.add_argument("--init-timeout-s", type=float, default=60.0)
    # drop target
    parser.add_argument("--drop-target", choices=["fixed-box", "marker"],
                        default="fixed-box",
                        help="fixed-box = no marker capture; carry to a fixed "
                             "robot-frame box center (default). marker retains "
                             "the legacy ArUco target flow")
    parser.add_argument("--box-xy", type=float, nargs=2, metavar=("X", "Y"),
                        default=DEFAULT_BOX_XY,
                        help="fixed box center in robot frame (m); default is "
                             f"the current surveyed box center {DEFAULT_BOX_XY}")
    parser.add_argument("--target-capture-dir", default=None,
                        help="reuse this image capture for the marker instead "
                            "of taking a fresh one")
    parser.add_argument("--marker-id", type=int, default=DEFAULT_BASKET_MARKER_ID)
    parser.add_argument("--marker-dict", default="6X6_1000")
    parser.add_argument("--target-yaw-deg", type=float, default=None,
                        help="force this world-z yaw for the drop (default: "
                             "the IK-feasible yaw closest to as-picked)")
    parser.add_argument("--yaw-step", type=int, default=10,
                        help="yaw sweep resolution (deg) for the drop yaw")
    parser.add_argument("--lift-height", type=float, default=None,
                        help="lift this far above the grasp (m); default is "
                             "0.15 for fixed-box, otherwise the banana "
                             "runner's LIFT_HEIGHT")
    parser.add_argument("--drop-h", type=float, default=0.03,
                        help="descend this far below the carry height before "
                             "releasing (m); never lowered onto the table")
    parser.add_argument("--drop-mode", choices=["controlled", "direct"],
                        default="controlled",
                        help="controlled = short pose-constrained descent; "
                             "direct = release immediately at carry height")
    parser.add_argument("--carry-orientation", choices=["locked", "free"],
                        default="locked",
                        help="Cartesian-transfer only: locked = preserve the "
                             "picked object attitude and height (default); free "
                             "= allow the legacy reorientation")
    parser.add_argument("--transfer-mode", choices=["joint0-arc", "cartesian"],
                        default="joint0-arc",
                        help="joint0-arc = after lifting, rotate only arm joint "
                             "0/link 0 about base Z (default); cartesian = use "
                             "the legacy collision-checked carry planner")
    parser.add_argument("--joint0-drop-bearing-deg", type=float,
                        default=JOINT0_DROP_BEARING_DEG,
                        help="joint0-arc release bearing in robot frame: +X "
                             "forward is 0°, positive is counter-clockwise "
                             "(default: +60°)")
    parser.add_argument("--carry-clearance", type=float,
                        default=CARRY_CLEARANCE_M,
                        help="minimum object-mesh clearance above the 0.04 m "
                             "table during carry (m; default: 0.05)")
    parser.add_argument("--retreat-h", type=float, default=0.15,
                        help="how far straight up the wrist climbs after "
                             "releasing, before the retract home (m)")
    # cameras / daemons
    parser.add_argument("--pc-list", nargs="+", default=DEFAULT_PC_LIST)
    parser.add_argument("--port-mask", type=int, default=5006)
    parser.add_argument("--port-pose", type=int, default=5007)
    parser.add_argument("--port-cmd", type=int, default=6893)
    parser.add_argument("--stream-fps", type=int, default=10)
    parser.add_argument("--stream-warmup-s", type=float, default=2.0)
    parser.add_argument("--calib-dir", default=None)
    return parser.parse_args(argv)


def run_once(args, *, orch, planner, executor, rcc, target_xyz: np.ndarray,
             run_dir: Path, grasps: list[DemoGrasp], pipeline=None,
             semantic_router=None, execution_recorder=None) -> dict:
    """Perceive once, then pick the object and drop it into the target box."""
    import time

    from paradex.calibration.utils import (load_c2r, save_current_C2R,
                                           save_current_camparam)

    from autodex.planner.obstacles import TABLE_CUBOID, TABLE_SURFACE_Z, add_obstacles
    from autodex.planner.planner import _to_curobo_world
    from autodex.utils.conversion import cart2se3
    from autodex.utils.path import get_obj_root
    from src.demo.inference.fixed_inspire_planning import (
        filter_fixed_inspire_by_place_reach,
        plan_fixed_inspire_grasp,
    )

    from src.demo.banana_test.run_demo import (
        LIFT_HEIGHT,
        _fk_wrist,
        _place_wrist,
        _rcc_start,
        _safe,
        _stop_with_timeout,
        _wrist_now,
        _yaw_grid,
    )
    from src.execution.scene_cfg import pose_world_to_scene_cfg

    lift_height = (0.15 if args.drop_target == "fixed-box" else LIFT_HEIGHT)
    if args.lift_height is not None:
        lift_height = args.lift_height
    drop_mode = "direct" if args.drop_target == "fixed-box" else args.drop_mode
    joint0_target_bearing_deg = float(args.joint0_drop_bearing_deg)
    joint0_target_angle = np.deg2rad(joint0_target_bearing_deg)
    adof = getattr(executor, "arm_dof", 6)
    img_dir = run_dir
    trial_clock = time.perf_counter()
    record: dict = {"object": args.obj, "arm": args.arm, "execute": args.execute,
                    "perception_backend": args.perception,
                    "drop_target": args.drop_target,
                    "transfer_mode": args.transfer_mode,
                    "joint0_target_bearing_deg": joint0_target_bearing_deg,
                    "fixed_box_xy": list(args.box_xy),
                    "started_at": dt.datetime.now().isoformat(timespec="seconds"),
                    "target_xyz": np.asarray(target_xyz).tolist(),
                    "timing": {"started_at": dt.datetime.now().isoformat(
                        timespec="seconds")}}

    def _phase(name: str, started_at: float, **detail) -> None:
        record["timing"][name] = {
            "total_s": round(time.perf_counter() - started_at, 3),
            **{key: value for key, value in detail.items() if value is not None},
        }

    def _finalize_timing() -> None:
        record["timing"]["total_s"] = round(time.perf_counter() - trial_clock, 3)
    save_current_C2R(str(img_dir))
    save_current_camparam(str(img_dir))

    # ── 1. perception ────────────────────────────────────────────────────────
    perception_clock = time.perf_counter()
    sil_loss_max = float("inf") if args.ignore_sil_loss else args.sil_loss_max
    if args.perception == "da3-fpose":
        print("[1/5] SAM3 + DA3 depth + FoundationPose (legacy pipeline)...")
        if args.ignore_sil_loss:
            # Say it rather than letting the operator believe the flag applied:
            # PerceptionPipeline.run gates at a hardcoded 0.003.
            print("[perception] NOTE --ignore-sil-loss has no effect on this "
                  "backend; its silhouette gate is fixed at 0.003")
        # This backend reads a capture directory rather than a live stream, so
        # take one image round with the stream stopped and put the stream back
        # before anything else in the trial runs.
        capture_dir = img_dir / "da3_capture"
        capture_dir.mkdir(parents=True, exist_ok=True)
        rel = os.path.join(
            os.path.relpath(str(capture_dir), str(Path.home())), "raw")
        t0 = time.time()
        _stop_with_timeout("rcc.stop", rcc.stop)
        rcc.start("image", False, rel)
        rcc.stop()
        time.sleep(0.3)
        _rcc_start(rcc, "stream", False, fps=args.stream_fps)
        save_current_camparam(str(capture_dir))
        save_current_C2R(str(capture_dir))
        # run() returns (pose, timing), but its early-out paths return a bare
        # (None, None) or even a 3-tuple. Normalise instead of unpacking blind.
        result_tuple = pipeline.run(
            str(capture_dir), prompt=args.prompt,
            sil_iters=args.sil_iters, sil_lr=args.sil_lr)
        pose_world = result_tuple[0] if result_tuple else None
        perception = result_tuple[1] if len(result_tuple) > 1 else None
        if perception is None:
            perception = {"reason": "da3_fpose_pipeline_failed"}
        perception = dict(perception, backend="da3_foundationpose")
    else:
        print("[1/5] Init pipeline (FoundPose distributed)...")
        if args.ignore_sil_loss:
            print("[perception] silhouette-loss gate DISABLED — the pose is used "
                  "however badly the mesh fits")
        t0 = time.time()
        if semantic_router is None:
            pose_world, perception = orch.trigger_init(
                prompt=args.prompt, save_capture_dir=str(img_dir / "init_capture"),
                sil_iters=args.sil_iters, sil_lr=args.sil_lr,
                timeout_s=args.init_timeout_s, sil_loss_threshold=sil_loss_max,
            )
        else:
            # P2 follows the exact same FoundPose collection and the exact
            # same cross-view IoU/silhouette refinement as the normal runner.
            # Splitting the two lets Qwen start after three SAM3 crops while
            # the capture PCs finish FoundPose, without sharing robot-GPU time
            # with silhouette rendering.
            request_id = int(time.time() * 1000) & 0x7fffffff
            semantic_router.begin(request_id, img_dir / "semantic")
            masks, poses, capture_timing = orch.collect_payloads(
                prompt=args.prompt, request_id=request_id,
                save_capture_dir=str(img_dir / "init_capture"),
                timeout_s=args.init_timeout_s,
                run_info_extra=semantic_router.run_info(),
            )
            semantic = semantic_router.wait(
                request_id, timeout_s=semantic_router.timeout_s)
            pose_world, refine_timing = orch.refine_from_payloads(
                masks, poses, sil_iters=args.sil_iters, sil_lr=args.sil_lr,
                sil_loss_threshold=sil_loss_max,
            )
            perception = dict(capture_timing)
            perception.update(refine_timing)
            record["semantic"] = semantic
    record["perception"] = perception
    record["perception_s"] = round(time.time() - t0, 2)
    semantic_timing = record.get("semantic") or {}
    _phase(
        "perception", perception_clock,
        capture_and_foundpose_s=(perception or {}).get("dispatch_to_collected_s"),
        cross_view_iou_s=(perception or {}).get("iou_select_s"),
        silhouette_refine_s=(perception or {}).get("sil_refine_s"),
        vlm_inference_s=semantic_timing.get("model_inference_s"),
        vlm_total_s=semantic_timing.get("semantic_total_s"),
    )
    if pose_world is None:
        reason = (perception or {}).get("reason", "perception_failed")
        _finalize_timing()
        return dict(record, success=False, reason=reason)
    if semantic_router is not None:
        semantic = record["semantic"]
        if semantic.get("status") != "ok":
            _finalize_timing()
            return dict(record, success=False, reason=semantic.get(
                "status", "semantic_vlm_error"))
        joint0_target_bearing_deg = float(semantic["bearing_deg"])
        joint0_target_angle = np.deg2rad(joint0_target_bearing_deg)
        record["joint0_target_bearing_deg"] = joint0_target_bearing_deg
        record["semantic_destination"] = {
            "basket": semantic["basket"],
            "semantic_class": semantic["semantic_class"],
        }
    _warn_if_sil_loss_high(perception, args.sil_loss_max)
    np.save(img_dir / "pose_world.npy", pose_world)

    # ── 2. scene + grasp ─────────────────────────────────────────────────────
    planning_clock = time.perf_counter()
    print("[2/5] Planning the approach from the fixed success library...")
    c2r = load_c2r(str(img_dir))
    obj_root = get_obj_root(GRASP_ASSET_VERSION)
    # The table has to be in the planning world: without it every trajectory
    # is collision-checked against the object alone and may sweep the arm
    # through the table.
    scene_cfg = add_obstacles(
        pose_world_to_scene_cfg(pose_world, c2r, args.obj, obj_root), "table")
    record["object_pose_robot"] = (np.linalg.inv(c2r) @ pose_world).tolist()

    T_obj_grasp = cart2se3(scene_cfg["mesh"]["target"]["pose"])
    yaws = _yaw_grid(args.target_yaw_deg, args.yaw_step)
    carry_yaws = [0.0] if args.carry_orientation == "locked" else yaws
    symmetry_rotations, symmetry_info = demo_symmetry_rotations(
        args.obj, obj_root=obj_root, n_continuous=args.symmetry_samples)
    record["symmetry"] = _jsonable(symmetry_info)
    print(f"    object at {np.round(np.asarray(T_obj_grasp)[:3, 3], 3)} (robot frame)")
    print(f"    [symmetry] {symmetry_info['n_variants']} approach variants "
          f"({symmetry_info['source']})")

    if getattr(planner, "_ik_solver", None) is None:
        # plan() builds the solver as a side effect, but target screening runs
        # before any plan on this run, so build a table-only one here.
        planner._init_ik_solver(_to_curobo_world(
            {"mesh": {}, "cuboid": {"table": TABLE_CUBOID}}))

    # A J0-only transfer has no Cartesian target IK: its feasibility is the
    # final FK check below, after the lift posture is known.  Do not discard a
    # grasp here using the legacy Cartesian placement screen.
    if args.transfer_mode == "joint0-arc":
        reach = []
        candidates = list(grasps)
        print("    [library] J0-only transfer: skipping Cartesian place-IK "
              "screen; checking the actual lifted pose by FK")
    else:
        # Screen the library against the target first: a grasp that cannot put
        # the object down where it has to go is not worth planning an approach.
        reach = filter_fixed_inspire_by_place_reach(
            planner, grasps, T_obj_grasp, np.asarray(target_xyz)[:2], carry_yaws,
            symmetry_rotations=symmetry_rotations)
        candidates = reach or list(grasps)
        if not reach:
            print("    [library] no target-reachable candidate in the IK screen — "
                  "trying the full library")
    # The planner walks ``candidates`` in order and ``candidate_order=[]``
    # disables its own reshuffle, so a fixed order replays the identical grasp
    # on every run: a grasp that was planned fine but failed physically (donut)
    # is picked first again next time. Permute per run instead, after the
    # place-reach screen so shuffling never smuggles an unreachable grasp in.
    grasp_seed = args.grasp_seed
    if args.grasp_order == "shuffle":
        if grasp_seed is None:
            grasp_seed = int.from_bytes(os.urandom(4), "big")
        # Score each library grasp by its best palm-down orientation over the
        # symmetry variants, since the planner may pick any of them.
        sym4 = np.tile(np.eye(4), (len(symmetry_rotations), 1, 1))
        sym4[:, :3, :3] = np.asarray(symmetry_rotations, dtype=np.float64)
        wrist_world = np.stack([T_obj_grasp @ sym4 @ g.wrist_obj
                                for g in candidates])          # (n_grasp, n_sym, 4, 4)
        palm_scores = palm_down_score(wrist_world).max(axis=1)
        order = _palm_weighted_order(palm_scores, grasp_seed,
                                     args.palm_down_weight)
        candidates = [candidates[i] for i in order]
        palm_scores = palm_scores[order]
        print(f"    [library] order drawn with palm-down weight "
              f"{args.palm_down_weight} (seed={grasp_seed}); "
              f"first grasp palm-down score {palm_scores[0]:+.2f}, "
              f"library range {palm_scores.min():+.2f}..{palm_scores.max():+.2f}")
    else:
        grasp_seed = None
        palm_scores = None
    print(f"    [library] {len(grasps)} success grasps; "
          f"{len(reach)} also reach the target")
    record["candidates"] = {
        "grasp_order": args.grasp_order,
        "grasp_seed": grasp_seed,
        "palm_down_weight": args.palm_down_weight,
        "palm_down_scores": (None if palm_scores is None
                             else [round(float(v), 3) for v in palm_scores]),
        "n_source_grasps": len(grasps),
        "n_symmetry_variants": len(symmetry_rotations),
        "n_approach_candidates": len(grasps) * len(symmetry_rotations),
        "n_place_reachable": (None if args.transfer_mode == "joint0-arc"
                              else len(reach)),
    }

    # ── plan approach AND lift together, per candidate ───────────────────────
    # An approach that cannot then be lifted is not a usable grasp. Validating
    # the lift only after committing to one candidate threw away the whole
    # trial (reason=lift_plan_failed) while every other candidate -- the rest
    # of the fixed library and the entire v8 pool -- was still untried. Treat a
    # lift/transfer failure exactly like an approach failure: drop that
    # candidate and plan the next one.
    vertices = _object_vertices(scene_cfg)
    planned_lift = lift_height
    carry_weights = ([1, 1, 1, 0, 0, 1]
                     if args.carry_orientation == "locked"
                     else [0, 0, 0, 0, 0, 1])

    def _lift_and_transfer(plan):
        """Plan lift + transfer for one approach. Returns (bundle, attempts).

        ``bundle`` is None when this grasp cannot be lifted or carried, which
        is the signal to move on to the next candidate.
        """
        attempts = []
        grasp_end = np.asarray(plan.traj[-1], dtype=np.float32)
        T_wrist_grasp = _fk_wrist(planner, grasp_end)
        # FK-derived (not plan.wrist_se3) so the lift starts exactly where the
        # planner thinks the executed trajectory ends.
        T_oiw = np.linalg.inv(T_wrist_grasp) @ T_obj_grasp

        def _full(q_arm):
            return np.concatenate([np.asarray(q_arm[:adof], dtype=np.float32),
                                   np.asarray(plan.grasp_pose, dtype=np.float32)])

        grasp_clearance = _object_clearance(
            vertices, T_wrist_grasp @ T_oiw, TABLE_SURFACE_Z)
        # One exact lift per candidate, never an escalating height: the
        # fixed-box default is exactly 15 cm.
        lift_wrist = T_wrist_grasp.copy()
        lift_wrist[2, 3] += planned_lift
        trial_lift = planner.plan_pose_constrained(
            _full(grasp_end), lift_wrist, hold_vec_weight=[1, 1, 1, 1, 1, 0],
            scene_cfg=scene_cfg, include_obj_obstacle=False)
        if trial_lift is None:
            attempts.append({"lift_m": planned_lift, "reason": "lift_plan_failed"})
            return None, attempts

        lift_end = np.asarray(trial_lift[-1], dtype=np.float32)
        T_wrist_lift = _fk_wrist(planner, lift_end)
        T_obj_lift = T_wrist_lift @ T_oiw
        common = {"T_oiw": T_oiw, "grasp_clearance_m": grasp_clearance,
                  "lift_traj": trial_lift}

        if args.transfer_mode == "joint0-arc":
            trial_carry, transfer_info = _joint0_arc_trajectory(
                planner, _fk_wrist, _full(lift_end), adof, T_oiw,
                joint0_target_angle, np.asarray(target_xyz)[:2])
            path_clearance = _carry_path_clearance(
                planner, trial_carry, T_oiw, vertices, TABLE_SURFACE_Z)
            lift_clearance = _object_clearance(
                vertices, T_obj_lift, TABLE_SURFACE_Z)
            transfer_info["min_clearance_m"] = path_clearance
            entry = {"lift_m": planned_lift, "joint0_transfer": transfer_info,
                     "grasp_clearance_m": grasp_clearance,
                     "lift_clearance_m": lift_clearance,
                     "min_clearance_m": path_clearance}
            # Three numbers from one geometry: at grasp, after the lift, and
            # over the arc. A carry value far off the lift value means the
            # path, not the grasp, is what fails the clearance gate.
            print(f"    [clearance] grasp {grasp_clearance * 100:+.1f}cm  "
                  f"lift {lift_clearance * 100:+.1f}cm  "
                  f"carry-path {path_clearance * 100:+.1f}cm  "
                  f"(need {args.carry_clearance * 100:.0f}cm)")
            if path_clearance < args.carry_clearance:
                attempts.append(dict(entry, reason="carry_clearance_too_low"))
                return None, attempts
            attempts.append(dict(entry, reason="selected"))
            return dict(common, carry_traj=trial_carry, place_yaw=None,
                        joint0_transfer=transfer_info), attempts

        if args.carry_orientation == "locked":
            carry_targets = [(0.0, _carry_wrist_locked(
                T_obj_lift, T_oiw, np.asarray(target_xyz)))]
        else:
            obj_z_lift = float(T_obj_lift[2, 3])
            carry_targets = [(yaw, _place_wrist(
                T_obj_grasp, T_oiw, np.asarray(target_xyz), yaw,
                obj_z_lift, float(T_wrist_lift[2, 3]))) for yaw in carry_yaws]
        for yaw, carry_wrist in carry_targets:
            trial_carry = planner.plan_pose_constrained(
                _full(lift_end), carry_wrist, hold_vec_weight=carry_weights,
                scene_cfg=scene_cfg, include_obj_obstacle=False)
            if trial_carry is None:
                continue
            path_clearance = _carry_path_clearance(
                planner, trial_carry, T_oiw, vertices, TABLE_SURFACE_Z)
            entry = {"lift_m": planned_lift, "carry_yaw_deg": yaw,
                     "min_clearance_m": path_clearance}
            if path_clearance >= args.carry_clearance:
                attempts.append(dict(entry, reason="selected"))
                return dict(common, carry_traj=trial_carry, place_yaw=yaw,
                            joint0_transfer=None), attempts
            attempts.append(dict(entry, reason="carry_clearance_too_low"))
        return None, attempts

    excluded_episodes: set[str] = set()
    excluded_v8: set[tuple] = set()
    v8_order_all = None
    grasp_attempts: list[dict] = []
    plan_s_total = 0.0
    result = None
    grasp_source = None
    validated = None

    fixed_spent = 0
    v8_spent = 0
    library_closed = False
    v8_closed = False
    # 0 = keep drawing from the v8 pool until it is empty. A candidate whose
    # approach or lift fails is dropped and the NEXT one is planned; the run
    # only gives up once both sources are actually out of candidates.
    v8_budget = args.max_v8_attempts if args.max_v8_attempts > 0 else None

    attempt = 0
    while not (library_closed and v8_closed):
        attempt += 1
        t0 = time.time()
        # The success library only gets ``max_grasp_attempts`` of the budget.
        # It can keep returning approach plans whose lift then fails, which
        # burned every attempt before the fallback was ever consulted.
        if fixed_spent >= args.max_grasp_attempts:
            library_closed = True
        pool = ([g for g in candidates if str(g.episode) not in excluded_episodes]
                if not library_closed else [])
        result = None
        fixed_timing = None
        if library_closed:
            fixed_timing = {"reason": "fixed_budget_spent"}
        elif pool:
            fixed_result = plan_fixed_inspire_grasp(
                planner, scene_cfg, pool, symmetry_rotations=symmetry_rotations)
            fixed_timing = _jsonable(fixed_result.timing)
            if fixed_result.success:
                result, grasp_source = fixed_result, "fixed_success"
                fixed_spent += 1
            else:
                # Re-planning the identical library pool yields the identical
                # failure, so stop consulting it and hand the run to the pool.
                library_closed = True
                print("    [library] no approach plan from the success library "
                      "— switching to the candidate pool")
        elif candidates:
            fixed_timing = {"reason": "fixed_library_exhausted"}
            library_closed = True
        else:
            # No physical success for this object yet: there is nothing to
            # replay, so the candidate pool is the only source.
            fixed_timing = {"reason": "no_success_library"}
            library_closed = True
        if result is None and not v8_closed:
            if v8_order_all is None:
                v8_order_all, v8_total, v8_attempted = _remaining_v8_candidate_order(
                    args.obj, T_obj_grasp, seed=grasp_seed,
                    palm_weight=args.palm_down_weight)
                record["v8_fallback"] = {
                    "palm_down_weight": args.palm_down_weight,
                    "n_total": v8_total,
                    "n_previously_attempted": v8_attempted,
                    "n_remaining": len(v8_order_all),
                }
            order = [c for c in v8_order_all if tuple(c) not in excluded_v8]
            if not order:
                v8_closed = True
            elif v8_budget is not None and v8_spent >= v8_budget:
                v8_closed = True
            else:
                from autodex.utils.symmetry import get_cyl_axis_local, get_cyl_yaw_grid

                print(f"    [v8 fallback] planning from {len(order)} remaining "
                      f"{GRASP_ASSET_VERSION} candidates")
                v8_result = planner.plan(
                    scene_cfg, args.obj, GRASP_ASSET_VERSION, skip_done=True,
                    success_only=False, hand="inspire", scene_id=None,
                    scene_type_filter=None, skip_scenes_with_success=False,
                    cyl_axis_local=get_cyl_axis_local(args.obj),
                    cyl_yaw_grid=get_cyl_yaw_grid(args.obj),
                    candidate_order=order,
                )
                if v8_result.success:
                    result, grasp_source = v8_result, "v8_fallback"
                    v8_spent += 1
                else:
                    # No approach plans out of the whole remaining pool: another
                    # draw from the same pool cannot do better.
                    v8_closed = True
                    print("    [v8 fallback] no approach plan from the remaining "
                          "pool — nothing left to try")
                record.setdefault("grasp_plan", {})["v8_fallback"] = _jsonable(
                    v8_result.timing)
        plan_s_total += time.time() - t0
        record.setdefault("grasp_plan", {})["fixed_success"] = fixed_timing

        if result is None:
            grasp_attempts.append({"attempt": attempt,
                                   "reason": "no_approach_plan"})
            continue

        print(f"    [attempt {attempt}] approach OK ({grasp_source}) "
              f"source={result.scene_info} — checking lift + transfer")
        validated, safety_attempts = _lift_and_transfer(result)
        grasp_attempts.append({
            "attempt": attempt,
            "source": grasp_source,
            "scene_info": _jsonable(result.scene_info),
            "lift_carry": _jsonable(safety_attempts),
            "reason": "selected" if validated else "lift_or_carry_failed",
        })
        if validated is not None:
            break

        # This grasp's approach plans but its lift/transfer does not. Drop it
        # and let the next iteration reach the next candidate.
        info = [str(part) for part in (result.scene_info or [])]
        if grasp_source == "fixed_success" and len(info) >= 3:
            excluded_episodes.add(info[2])
            print(f"    [attempt {attempt}] lift/transfer failed — dropping this "
                  f"success-library episode and trying the next candidate")
        elif grasp_source == "v8_fallback" and len(info) >= 3:
            excluded_v8.add(tuple(info[:3]))
            print(f"    [attempt {attempt}] lift/transfer failed — dropping this "
                  f"{GRASP_ASSET_VERSION} candidate and trying the next")
        else:
            # Nothing identifies this candidate, so it cannot be excluded and
            # the same one would be replanned forever.
            print(f"    [attempt {attempt}] lift/transfer failed on an "
                  f"unidentifiable candidate — closing this source")
            if grasp_source == "fixed_success":
                library_closed = True
            else:
                v8_closed = True
        result = None

    record["grasp_attempts"] = grasp_attempts
    record["grasp_plan_s"] = round(plan_s_total, 2)
    record["n_grasp_attempts"] = len(grasp_attempts)
    if validated is None:
        exhausted = result is None and grasp_attempts and \
            grasp_attempts[-1].get("reason") == "no_approach_plan"
        reason = ("all_fixed_and_v8_approaches_failed" if exhausted
                  else "lift_or_carry_plan_failed_for_every_candidate")
        print(f"    no candidate produced an approach + lift + transfer "
              f"({len(grasp_attempts)} attempted) — refusing to grasp")
        _phase("planning", planning_clock,
               candidate_search_s=record["grasp_plan_s"],
               grasp_attempts=len(grasp_attempts))
        _finalize_timing()
        return dict(record, success=False, reason=reason)

    record["grasp_source"] = grasp_source
    lift_traj = validated["lift_traj"]
    carry_traj = validated["carry_traj"]
    place_yaw = validated["place_yaw"]
    selected_joint0_transfer = validated["joint0_transfer"]
    T_oiw = validated["T_oiw"]
    record["carry_safety"] = {
        "transfer_mode": args.transfer_mode,
        "orientation": ("base_yaw_from_joint0"
                        if args.transfer_mode == "joint0-arc"
                        else args.carry_orientation),
        "table_surface_z": TABLE_SURFACE_Z,
        "required_clearance_m": args.carry_clearance,
        "grasp_clearance_m": validated["grasp_clearance_m"],
        "attempts": _jsonable(grasp_attempts[-1]["lift_carry"]),
    }
    record["effective_lift_height_m"] = planned_lift
    if args.transfer_mode == "joint0-arc":
        print("    lift + J0-only arc OK: "
              f"ΔJ0={selected_joint0_transfer['joint0_delta_deg']:.1f}°, "
              f"release bearing={joint0_target_bearing_deg:.1f}°, "
              f"object clearance ≥ {args.carry_clearance * 100:.0f}cm")
    else:
        print(f"    lift + carry OK: {args.carry_orientation} orientation, "
              f"object clearance ≥ {args.carry_clearance * 100:.0f}cm")
    record["place_yaw_deg"] = place_yaw
    if selected_joint0_transfer is not None:
        record["joint0_transfer_plan"] = selected_joint0_transfer

    np.save(img_dir / "plan_traj.npy", np.asarray(result.traj))
    np.save(img_dir / "grasp_wrist_se3.npy", result.wrist_se3)
    record["scene_info"] = result.scene_info
    _phase("planning", planning_clock,
           candidate_search_s=record["grasp_plan_s"],
           grasp_attempts=len(grasp_attempts))
    if not args.execute:
        print("[4/5] --execute not given; stopping after the dry run")
        _finalize_timing()
        return dict(record, success=None, reason="dry_run_only")

    # ── 4. grasp + lift ──────────────────────────────────────────────────────
    print("[4/5] Executing (grasp + lift)...")
    execution_clock = time.perf_counter()
    # Keep the physical task separate from the non-task return sequence.  The
    # latter is important for a robot demo's throughput, but must not inflate
    # a pick/place execution number.
    task_timing: dict = {}
    reset_timing: dict = {"performed": False, "total_s": 0.0}
    # Camera/video mode changes are neither grasp/drop motion nor reset motion.
    # Keep them explicit: on multi-PC AVI recording they can take seconds and
    # otherwise look like an unexplained stationary interval after release.
    recording_transition_timing: dict = {
        "enabled": execution_recorder is not None,
    }
    task_started_at: float | None = None
    task_finished_at: float | None = None

    def _close_task_timing() -> None:
        nonlocal task_finished_at
        if task_started_at is None or task_finished_at is not None:
            return
        task_finished_at = time.perf_counter()
        task_timing["total_s"] = round(task_finished_at - task_started_at, 3)

    def _save_execution_timing() -> None:
        _close_task_timing()
        _phase("execution", execution_clock,
               task=dict(task_timing), reset=dict(reset_timing),
               recording_transition=dict(recording_transition_timing))

    def _stop_execution_recording() -> None:
        if execution_recorder is None:
            return
        try:
            record["recording"] = _jsonable(execution_recorder.stop())
        except Exception as exc:
            record.setdefault("recording", {})["stop_error"] = repr(exc)
            print(f"[p2-video] stop failed: {exc!r}")

    recording_start_t0 = time.perf_counter()
    if execution_recorder is None:
        _safe("rcc.stop", rcc.stop)
    else:
        record["recording"] = _jsonable(execution_recorder.start())
    recording_transition_timing["start_s"] = round(
        time.perf_counter() - recording_start_t0, 3)
    task_started_at = time.perf_counter()
    t0 = time.perf_counter()
    try:
        s_hand = executor.execute(result, planner=planner, scene_cfg=scene_cfg,
                                  lift_height=lift_height,
                                  lift_traj_override=lift_traj)
    except Exception as exc:
        print(f"    execute FAILED: {exc!r}")
        task_timing["grasp_lift_s"] = round(time.perf_counter() - t0, 3)
        _close_task_timing()
        reset_clock = time.perf_counter()
        _stop_execution_recording()
        _stop_with_timeout("rcc.stop", rcc.stop)
        _safe("reset_fallback", executor.reset_fallback, result)
        _rcc_start(rcc, "stream", False, fps=args.stream_fps)
        reset_timing.update({
            "performed": True,
            "mode": "reset_fallback_after_execute_exception",
            "total_s": round(time.perf_counter() - reset_clock, 3),
        })
        _save_execution_timing()
        _finalize_timing()
        return dict(record, success=False, reason="execute_exception",
                    exception=repr(exc))
    record["grasp_exec_s"] = round(time.perf_counter() - t0, 2)
    task_timing["grasp_lift_s"] = record["grasp_exec_s"]

    # ── 5. carry over the target and drop ────────────────────────────────────
    drop_label = (f"controlled {args.drop_h * 100:.0f}cm descend"
                  if drop_mode == "controlled" else "direct release")
    if args.transfer_mode == "joint0-arc":
        print("[5/5] Rotate J0 at constant height to robot bearing "
              f"{joint0_target_bearing_deg:.1f}° + {drop_label}...")
    else:
        print(f"[5/5] Carry to target (constant height) + {drop_label} at "
              f"{np.round(np.asarray(target_xyz), 3)}...")
    # Take one post-lift measurement and use this exact configuration both for
    # clearance and as the origin of the J0 target.  This avoids using the
    # nominal dry-run J0 when the physical lift ended a little differently.
    current_arm_qpos = np.asarray(
        executor.arm.get_data()["qpos"][:adof], dtype=np.float32)
    current_hand_qpos = np.asarray(
        getattr(executor, "_last_hand_qpos", result.pregrasp_pose),
        dtype=np.float32)
    start_full = np.concatenate([current_arm_qpos, current_hand_qpos])
    T_wrist_now = _fk_wrist(planner, start_full)
    T_obj_now = T_wrist_now @ T_oiw
    current_clearance = _object_clearance(vertices, T_obj_now, TABLE_SURFACE_Z)
    record["carry_safety"]["actual_lift_clearance_m"] = current_clearance
    if current_clearance < args.carry_clearance:
        print("    actual post-lift object clearance "
              f"{current_clearance * 100:.1f}cm < required "
              f"{args.carry_clearance * 100:.1f}cm — not carrying")
        _stop_execution_recording()
        _rcc_start(rcc, "stream", False, fps=args.stream_fps)
        _save_execution_timing()
        _finalize_timing()
        return dict(record, success=False, reason="carry_clearance_after_lift_too_low",
                    scene_info=result.scene_info,
                    action="object_held_for_operator_check")
    if args.transfer_mode == "joint0-arc":
        # ``start_full`` was measured immediately after the actual lift.
        traj_repose, runtime_transfer = _joint0_arc_trajectory(
            planner, _fk_wrist, start_full, adof, T_oiw,
            joint0_target_angle, np.asarray(target_xyz)[:2])
        record["joint0_transfer_runtime"] = runtime_transfer
    else:
        if args.carry_orientation == "locked":
            T_wrist_target = _carry_wrist_locked(
                T_obj_now, T_oiw, np.asarray(target_xyz))
        else:
            T_wrist_target = _place_wrist(
                T_obj_grasp, T_oiw, np.asarray(target_xyz), place_yaw,
                float(T_obj_now[2, 3]), float(T_wrist_now[2, 3]))
        traj_repose = planner.plan_pose_constrained(
            start_full, T_wrist_target, hold_vec_weight=carry_weights,
            scene_cfg=scene_cfg, include_obj_obstacle=False)
        if traj_repose is None:
            print("    carry re-plan failed — not carrying the held object")
            _stop_execution_recording()
            _rcc_start(rcc, "stream", False, fps=args.stream_fps)
            _save_execution_timing()
            _finalize_timing()
            return dict(record, success=False, reason="carry_replan_failed",
                        scene_info=result.scene_info,
                        action="object_held_for_operator_check")
    runtime_clearance = _carry_path_clearance(
        planner, traj_repose, T_oiw, vertices, TABLE_SURFACE_Z)
    record["carry_safety"]["runtime_min_clearance_m"] = runtime_clearance
    if runtime_clearance < args.carry_clearance:
        print("    carry re-plan would lower the object to "
              f"{runtime_clearance * 100:.1f}cm (< "
              f"{args.carry_clearance * 100:.1f}cm) — not carrying")
        _stop_execution_recording()
        _rcc_start(rcc, "stream", False, fps=args.stream_fps)
        _save_execution_timing()
        _finalize_timing()
        return dict(record, success=False, reason="carry_replan_clearance_too_low",
                    scene_info=result.scene_info,
                    action="object_held_for_operator_check")
    transfer_t0 = time.perf_counter()
    executor._move_joints(traj_repose[:, :adof],
                          np.tile(s_hand, (len(traj_repose), 1)))
    task_timing["transfer_s"] = round(time.perf_counter() - transfer_t0, 3)
    if args.transfer_mode == "joint0-arc":
        print("    J0-only arc OK "
              f"(object bearing={runtime_transfer['target_angle_deg']:.1f}°, "
              f"ΔJ0={runtime_transfer['joint0_delta_deg']:.1f}°, "
              f"object clearance ≥ {runtime_clearance * 100:.1f}cm)")
    else:
        print(f"    carry OK (object clearance ≥ {runtime_clearance * 100:.1f}cm)")

    release_t0 = time.perf_counter()
    if drop_mode == "direct":
        # Deliberately no target wrist below the carry pose and no place():
        # this mode opens the hand immediately above the target.  The
        # grasp/carry paths are unchanged and collision-checked; only the
        # pose-constrained placement descent is omitted.
        # The transfer is finished and the object hangs over the box, so open
        # all the way rather than stopping at pregrasp: pregrasp fingers still
        # cup the object and can carry it along or catch its rim on the way up.
        release_kwargs = {"open_to_init": True} if args.arm == "franka" else {}
        executor.release(result, **release_kwargs)
        place_info = {"mode": "direct_release_from_carry", "descended": 0.0,
                      "target": args.drop_target,
                      "hand_opened_to": ("fully_open" if release_kwargs
                                         else "pregrasp"),
                      "drop_height_ignored_m": float(args.drop_h)}
        print("    direct release from carry height (no placement descent)"
              + (" — hand opened fully" if release_kwargs else ""))
    else:
        # The object is carried at the lift height and let go ``--drop-h``
        # below it — NOT lowered back onto the table. The target is built from
        # cuRobo FK and handed over explicitly; use_current_wrist would rebuild
        # it from the measured frame, which carries the 107mm ee_link offset.
        T_place = _wrist_now(planner, executor, adof, result.grasp_pose)
        T_place[2, 3] -= args.drop_h
        place_kwargs = {"grasp_wrist": T_place} if args.arm == "franka" else {}
        place_info = executor.place(result, planner=planner, scene_cfg=scene_cfg,
                                    lift_height=args.drop_h, **place_kwargs)
        print(f"    place: {place_info}")
        if args.arm != "franka":
            _safe("release", executor.release, result)
    record["place"] = _jsonable(place_info)
    task_timing["release_s"] = round(time.perf_counter() - release_t0, 3)
    _close_task_timing()

    # P2's execution AVI is the task evidence: grasp/lift, transfer, and
    # release.  Flush it before any retreat or home motion, then return to the
    # non-recording state for reset.  Do *not* restart the live stream here:
    # resetting does not consume camera frames, and waiting for five PCs to
    # restart otherwise creates a visible stationary interval after release.
    # The stream comes back after reset, still before the next trial's prompt.
    # Timing still reports reset separately, and reset remains absent from
    # ``raw/exec/videos`` and the robot-state take.
    if execution_recorder is not None:
        recording_stop_t0 = time.perf_counter()
        _stop_execution_recording()
        recording_transition_timing["stop_after_release_s"] = round(
            time.perf_counter() - recording_stop_t0, 3)

    # Straight up and OUT before anything else moves. Retracting from where
    # place() left the wrist sweeps the hand through the object and the setup
    # around it. Climb --retreat-h from HERE, and hold the fingers where
    # release left them: opening them at this height scrapes the table.
    retreat_t0 = time.perf_counter()
    reset_timing["performed"] = True
    hand_now = np.asarray(getattr(executor, "_last_hand_qpos",
                                  result.pregrasp_pose), dtype=np.float32)
    T_up = _wrist_now(planner, executor, adof, hand_now).copy()
    T_up[2, 3] += args.retreat_h
    up_traj = planner.plan_pose_constrained(
        np.concatenate([np.asarray(executor.arm.get_data()["qpos"][:adof],
                                   dtype=np.float32), hand_now]),
        T_up, hold_vec_weight=[1, 1, 1, 1, 1, 0],
        scene_cfg=scene_cfg, include_obj_obstacle=False)
    if up_traj is not None:
        # _move_joints expects the hand columns in CONTROLLER UNITS (0-1000).
        # Handing it the planner's radians clips to ~0 = fingers slammed shut
        # on the object that was just put down.
        hand_cmd = np.asarray(executor._convert(hand_now.astype(np.float64)),
                              dtype=np.float64)
        executor._move_joints(up_traj[:, :adof],
                              np.tile(hand_cmd, (len(up_traj), 1)))
        record["retreat_up"] = "ok"
        print(f"    retreat up {args.retreat_h * 100:.0f}cm OK")
    else:
        record["retreat_up"] = "plan_failed"
        print("    retreat-up plan FAILED — reset() will handle it")
    reset_timing["retreat_up_s"] = round(time.perf_counter() - retreat_t0, 3)

    # Return home over a virtual floor first. The hand hangs up to ~23cm below
    # the wrist, so a raised floor can put the START state in collision; when
    # that happens the same reset is retried against the real table.
    retract_steps = []
    if args.return_floor > 0:
        return_cfg = raise_floor_scene_cfg(scene_cfg, args.return_floor)
        record["return_floor_m"] = args.return_floor
        retract_steps.append(
            (f"reset(floor={args.return_floor:.2f}m)",
             lambda: executor.reset(result, planner, return_cfg)))
    retract_steps += [
        ("reset", lambda: executor.reset(result, planner, scene_cfg)),
        ("reset_hybrid", lambda: executor.reset_hybrid(result, planner, scene_cfg)),
        ("reset_fallback", lambda: executor.reset_fallback(result)),
    ]
    home_reset_t0 = time.perf_counter()
    for name, fn in retract_steps:
        try:
            record["retract"] = _jsonable(fn())
            record["retract_step"] = name
            print(f"    retract via {name} OK")
            break
        except Exception as exc:
            print(f"    retract step {name} failed: {exc!r}")
    reset_timing["home_reset_s"] = round(time.perf_counter() - home_reset_t0, 3)
    reset_timing["total_s"] = round(time.perf_counter() - retreat_t0, 3)

    if s_hand is not None:
        np.save(img_dir / "squeeze_hand.npy", np.asarray(s_hand))
    # The camera stream is unnecessary during non-recorded reset.  Restore it
    # only once arm motion is finished so stream startup cannot stall the arm
    # between the release and its retreat.  This is also the normal restoration
    # point for the no-video path, which stopped the stream before task motion.
    stream_restore_t0 = time.perf_counter()
    _stop_with_timeout("rcc stream restart",
                       lambda: _rcc_start(rcc, "stream", False,
                                          fps=args.stream_fps))
    recording_transition_timing["stream_restore_after_reset_s"] = round(
        time.perf_counter() - stream_restore_t0, 3)
    _save_execution_timing()
    _finalize_timing()
    return dict(record, success=True, reason="picked_and_dropped")


def main(*, argv: list[str] | None = None, semantic_router_factory=None,
         execution_recorder_factory=None) -> None:
    import time

    from paradex.io.camera_system.remote_camera_controller import remote_camera_controller
    from paradex.utils.system import get_camera_list, get_pc_ip

    from autodex.perception.init_orchestrator import InitOrchestrator
    from autodex.planner import GraspPlanner
    from autodex.utils.path import get_obj_root, project_dir

    from src.demo.banana_test.place_target import capture_images, locate_marker
    from src.execution.scene_cfg import check_mesh_frame_match
    from src.demo.banana_test.run_demo import (
        ASSETS_BASE,
        CAM_PARAM_ROOT,
        _clear_camera_errors,
        _ensure_camera_lock,
        _load_calib,
        _planner_robot,
        _rcc_start,
        _safe,
        _stop_with_timeout,
        _warn_if_not_streaming,
        quiet_curobo,
    )

    args = parse_args(argv)
    if args.symmetry_samples < 1:
        raise SystemExit("--symmetry-samples must be positive")
    if args.carry_clearance < 0:
        raise SystemExit("--carry-clearance must be non-negative")
    if args.max_v8_attempts < 0:
        raise SystemExit("--max-v8-attempts must be non-negative")

    # FoundPose estimates the pose of THIS mesh and the planner places the mesh
    # under ``get_obj_root(GRASP_ASSET_VERSION)``.  Resolving both from the same
    # version keeps them in one frame: MESH_BASE is the legacy paradex tree,
    # which ships a crude primitive for six objects (paper_bowl, paper_cup,
    # pepper_tuna_light, tea_case, tennis_ball, tissue_box) where v8 has a real
    # scan, and a pose estimated on the primitive offsets every grasp.
    obj_asset_root = Path(get_obj_root(GRASP_ASSET_VERSION))
    mesh_path = obj_asset_root / args.obj / "raw_mesh" / f"{args.obj}.obj"
    assets_root = ASSETS_BASE / args.obj
    if not mesh_path.is_file():
        raise SystemExit(f"mesh not found: {mesh_path}")
    _frame_ok, _frame_msg = check_mesh_frame_match(
        args.obj, str(mesh_path), str(obj_asset_root))
    if not _frame_ok:
        raise SystemExit(f"[mesh_frame] {_frame_msg}")
    print(f"[mesh_frame] {_frame_msg}")
    if (args.perception == "foundpose"
            and not (assets_root / "object_repre/v1" / args.obj / "1/repre.pth").is_file()):
        raise SystemExit(f"FoundPose representation missing under {assets_root}")

    grasps = load_demo_grasps(args.obj)
    counts = {source: sum(g.source == source for g in grasps)
              for source in ("v8_inspire", "selected_100_inspire")}
    if grasps:
        print(f"[library] {len(grasps)} fixed Inspire successes: {counts}")
    else:
        # An object with no physical success yet is exactly what the v8
        # candidate fallback is for.  Refusing to start here would make the
        # first attempt on a new object impossible.
        print(f"[library] no Inspire success recorded for {args.obj!r} in "
              f"selected_100_inspire + experiment/{{{COLLECTION_VERSION},"
              f"{DEMO_VERSION}}}/inspire — planning straight from the "
              f"{GRASP_ASSET_VERSION} candidate pool")

    calib_dir = (Path(args.calib_dir).expanduser() if args.calib_dir
                 else sorted(CAM_PARAM_ROOT.iterdir())[-1])
    intrinsics, extrinsics, height, width = _load_calib(calib_dir)
    pc_ips = [get_pc_ip(pc) for pc in args.pc_list]
    pc_serials = {pc: get_camera_list(pc) for pc in args.pc_list}
    active = {serial for pc in args.pc_list for serial in pc_serials[pc]}
    intrinsics = {s: v for s, v in intrinsics.items() if s in active}
    extrinsics = {s: v for s, v in extrinsics.items() if s in active}
    print(f"calib: {calib_dir.name}  ({len(intrinsics)} cams, {height}x{width})")

    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = (Path(project_dir) / "experiment" / DEMO_VERSION / "inspire"
               / args.obj / stamp)
    run_dir.mkdir(parents=True, exist_ok=False)

    rcc = remote_camera_controller("fixed_inspire_demo", pc_list=args.pc_list,
                                   stall_timeout=15.0)
    if not _ensure_camera_lock(rcc):
        _safe("rcc.end", rcc.end)
        raise SystemExit("camera daemons are owned by another controller; "
                         "no robot motion was sent")
    if not _clear_camera_errors(rcc):
        _safe("rcc.end", rcc.end)
        raise SystemExit("capture cameras remain in an error state; "
                         "no robot motion was sent")

    # ── fixed box / legacy marker target ────────────────────────────────────
    if args.drop_target == "fixed-box":
        # The box is intentionally not perceived: only its surveyed robot-frame
        # center is needed because fixed-box mode keeps the lifted object's z
        # and attitude and releases immediately above this XY location.
        target_info = {
            "target_type": "fixed_box",
            "center_robot": np.array([args.box_xy[0], args.box_xy[1], 0.04]),
            "release_target_robot": np.array([args.box_xy[0], args.box_xy[1], 0.04]),
        }
        print(f"[target] fixed box center_robot="
              f"{np.asarray(target_info['center_robot']).round(4)} "
              "(marker capture skipped)")
        if args.transfer_mode == "joint0-arc":
            print("[target] J0-only release is controlled by robot-frame "
                  f"bearing {args.joint0_drop_bearing_deg:.1f}° "
                  "(+X forward, +CCW; J0 takes the shorter sweep), not "
                  "box-center XY")
        with open(run_dir / "place_target.json", "w") as f:
            json.dump(_jsonable(target_info), f, indent=1)
    else:
        try:
            capture = (str(Path(args.target_capture_dir).expanduser())
                       if args.target_capture_dir
                       else capture_images(pc_list=args.pc_list, rcc=rcc))
            target_info = locate_marker(capture, dict_type=args.marker_dict,
                                        marker_id=args.marker_id)
            print(f"[target] {args.marker_dict} id={target_info['marker_id']} "
                  f"({target_info['n_views']} views) center_robot="
                  f"{np.asarray(target_info['center_robot']).round(4)}")
            with open(run_dir / "place_target.json", "w") as f:
                json.dump(_jsonable(target_info), f, indent=1)
        except Exception as exc:
            _safe("rcc.end", rcc.end)
            raise SystemExit(f"place-marker setup failed; no robot motion was "
                             f"sent: {exc!r}") from None

    print(f"[stream] start @ {args.stream_fps} FPS...")
    try:
        _rcc_start(rcc, "stream", False, fps=args.stream_fps)
    except Exception as exc:
        _safe("rcc.end", rcc.end)
        raise SystemExit(f"could not start camera stream; no robot motion was "
                         f"sent: {exc!r}") from None
    time.sleep(args.stream_warmup_s)
    if not _warn_if_not_streaming(rcc):
        _safe("rcc.stop", rcc.stop)
        _safe("rcc.end", rcc.end)
        raise SystemExit("camera stream did not start; no robot motion was sent")

    orch = None
    pipeline = None
    executor = None
    semantic_router = None
    execution_recorder = None
    record: dict = {}
    try:
        if args.perception == "da3-fpose":
            # The legacy pipeline owns its own SAM3/FPose daemons and reads
            # capture directories, so no InitOrchestrator is created and the
            # init_daemons are not needed (nor can they share the GPU).
            from src.execution_prev.daemon.perception_pipeline import PerceptionPipeline
            from src.execution_prev.run_perception import FPOSE_HOSTS, SAM3_HOSTS

            print(f"[perception] SAM3 {len(SAM3_HOSTS)} host(s) + FoundationPose "
                  f"{len(FPOSE_HOSTS)} host(s), depth=da3")
            pipeline = PerceptionPipeline(
                sam3_hosts=SAM3_HOSTS, fpose_hosts=FPOSE_HOSTS,
                obj_name=args.obj, depth_method="da3")
        else:
            print(f"[orch] init for {args.obj}...")
            orch = InitOrchestrator(pc_list=args.pc_list, capture_ips=pc_ips,
                                    port_mask=args.port_mask,
                                    port_pose=args.port_pose,
                                    port_cmd=args.port_cmd)
            orch.init_object(obj_name=args.obj, mesh_path=str(mesh_path),
                             assets_root=str(assets_root),
                             intrinsics_full=intrinsics, extrinsics_full=extrinsics,
                             image_hw=(height, width), mode="live",
                             pc_serials=pc_serials)
            if semantic_router_factory is not None:
                semantic_router = semantic_router_factory(
                    args=args, capture_ips=pc_ips, pc_serials=pc_serials,
                    run_dir=run_dir)
                # Loading before the robot moves means the three crop capture
                # path is the only semantic latency in a real trial.
                semantic_router.preload()
        planner_robot = _planner_robot(args.arm, "inspire")
        print(f"[planner] warmup ({planner_robot})...")
        planner = GraspPlanner(hand=planner_robot)
        quiet_curobo()
        print(f"[executor] connect ({args.arm})...")
        if args.arm == "franka":
            from src.execution.franka_executor import FrankaExecutor
            executor = FrankaExecutor(hand_name="inspire")
            executor.set_speed_profile_planner(planner)
        else:
            from autodex.executor.real import RealExecutor
            executor = RealExecutor(hand_name="inspire")
        if execution_recorder_factory is not None:
            execution_recorder = execution_recorder_factory(
                args=args, rcc=rcc, executor=executor, run_dir=run_dir,
                pc_list=args.pc_list, pc_serials=pc_serials,
                project_root=Path(project_dir),
            )
    except Exception as exc:
        if orch is not None:
            _safe("orch.close", orch.close)
        if semantic_router is not None:
            _safe("semantic_router.close", semantic_router.close)
        _safe("rcc.stop", rcc.stop)
        _safe("rcc.end", rcc.end)
        raise SystemExit(f"demo setup failed before robot motion: {exc!r}") from None

    # The first movement is always camera-clear home, so the arm neither
    # occludes the object nor starts from an unknown configuration.
    if args.execute:
        print("[executor] moving once to clear-view home")
        try:
            executor.home(clear_view=True)
        except Exception as exc:
            _safe("executor.shutdown", executor.shutdown)
            _safe("orch.close", orch.close)
            _safe("semantic_router.close", semantic_router.close)
            _safe("rcc.stop", rcc.stop)
            _safe("rcc.end", rcc.end)
            raise SystemExit(f"initial clear-view home failed ({exc!r}); "
                             f"inspect the robot before retrying") from None

    try:
        record = run_once(args, orch=orch, planner=planner, executor=executor,
                          rcc=rcc, pipeline=pipeline,
                          target_xyz=np.asarray(target_info["center_robot"]),
                          run_dir=run_dir, grasps=grasps,
                          semantic_router=semantic_router,
                          execution_recorder=execution_recorder)
    except KeyboardInterrupt:
        record = {"success": False, "reason": "interrupted"}
        print("\n[interrupted]")
    except Exception as exc:
        # Never turn an unknown post-motion error into an automatic
        # release/home command: the hand may still hold the object. Stop with
        # the arm where it is for an operator check.
        record = {"success": False, "reason": "fatal", "error": repr(exc),
                  "action": "stopped_without_robot_reset"}
        print("\n[FATAL] the demo raised an unexpected error; stopping without "
              f"an automatic robot reset:\n        {exc!r}")
    finally:
        # A fatal post-motion error can bypass ``run_once``'s normal stop
        # paths.  Flush the P2 AVI before serialising ``result.json`` so its
        # embedded recording manifest always reflects the completed take.
        if execution_recorder is not None:
            stopped_recording = _safe("execution_recorder.stop",
                                      execution_recorder.stop)
            if stopped_recording is not None and isinstance(record, dict):
                record["recording"] = _jsonable(stopped_recording)
        with open(run_dir / "result.json", "w") as f:
            json.dump(_jsonable(record), f, indent=2, default=str)
        status = ("SUCCESS" if record.get("success") else
                  ("DRY-RUN" if record.get("success") is None
                   else record.get("reason", "FAIL")))
        print(f"\nRESULT: {status}   saved to {run_dir}/result.json")
        _safe("executor.shutdown", executor.shutdown)
        if orch is not None:
            _safe("orch.close", orch.close)
        if semantic_router is not None:
            _safe("semantic_router.close", semantic_router.close)
        if pipeline is not None:
            _safe("pipeline.close", pipeline.close)
        for fn, name in ((rcc.stop, "rcc.stop"), (rcc.end, "rcc.end")):
            _stop_with_timeout(name, fn)


if __name__ == "__main__":
    main()
