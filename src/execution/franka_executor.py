"""Franka FR3 + inspire grasp executor — a faithful port of the proven xarm
``autodex/executor/real.py`` (RealExecutor), with ONLY the arm-driver bits
swapped for the FR3:

  * 7-DOF arm, homed via FR3_INIT; clear-view = FR3_INIT with j0 rotated -40°
    (same convention as real.py: ``clear_view_arm[0] -= deg2rad(40)``) so the
    arm is out of the cameras' view of the object during perception.
  * Trajectory following: the FR3 daemon's ``move()`` BLOCKS (no continuous
    position servo like the xarm), so a dense cuRobo trajectory is followed by
    STREAMING joint velocities (``set_joint_velocity``) — the arm stays ON the
    collision-free path and moves continuously (a downsampled point-to-point
    would swing off-path and ram). Free-space single-config moves (home /
    clear-view) use a plain blocking ``move()`` (safe, no obstacles).
  * Lift = joint-space ``planner.plan_pose_constrained`` (NOT cartesian — the
    xarm lesson; FR3 cartesian servo times out at limits/singularities).
  * Reset = ``planner.plan_js_to_init`` retract to clear-view, avoiding the
    placed object (mirrors real.py ``reset``).
  * Place = FR3 NATIVE cartesian impedance + direct ``wrench`` contact (FR3 has
    link-side torque sensors — no learned tau model / current admittance).
  * Errors: ``error_recovery`` / ``is_error`` replace xarm ``clear_error`` /
    ``get_err_warn_code``.

Execution sequence (identical to real.py):
    execute:  init(FR3_INIT) -> approach -> pregrasp -> grasp -> squeeze -> lift
    release:  open hand -> (reset retracts to clear-view)
"""
import datetime
import time
from typing import Optional

import numpy as np

from autodex.planner import PlanResult
from autodex.utils.robot_config import FR3_INIT, INSPIRE_INIT, FR3_INSPIRE_LINK_TO_WRIST
from autodex.utils.conversion import cart2se3, se32cart
from autodex.executor.real import _convert_inspire   # rad -> 0-1000 controller units

CLEAR_VIEW_J0_DEG = -40.0   # real.py: clear_view_arm[0] -= deg2rad(40)

# The final part of every free-hand move is deliberately slowed.  These are
# wrist-to-target-wrist distances in the robot base frame, not distances to an
# object centre (whose useful clearance varies with the asset's size).
APPROACH_SLOWDOWN_FAR_M = 0.25
APPROACH_SLOWDOWN_NEAR_M = 0.10
APPROACH_SLOWDOWN_MIN_SCALE = 0.25
DEFAULT_HELD_SPEED_SCALE = 0.25
# A placement always passes through a point exactly 10 cm above its release
# pose: approach that point first, descend vertically 10 cm, release, then
# lift vertically 10 cm before any lateral retreat.  These are planned joint
# trajectories, never blind Cartesian commands.
PLACE_VERTICAL_TRAVEL_M = 0.10
POST_RELEASE_LIFT_HEIGHT_M = PLACE_VERTICAL_TRAVEL_M
# FK validation bound for the planned 10cm vertical strokes.  The start/end
# wrists are constructed at identical x/y; intermediate joint-space motion is
# left to cuRobo's collision-checked plan rather than rejected by a separate
# Cartesian lateral-deviation threshold.
VERTICAL_STROKE_Z_TOL_M = 0.005


class ContactAbort(RuntimeError):
    """Raised when the FR3 collision reflex trips mid-approach so the caller can
    abort the trial instead of continuing into grasp at the wrong pose."""


class FrankaExecutor:
    def __init__(self, hand_name: str = "inspire", dt: float = 0.01,
                 squeeze_level: int = 2, arm_speed_scale: float = 1,
                 ctrl_dt: float = 0.02, joint_vmax: float = 1.2,
                 pos_kp: float = 4.0, follow_tol: float = 0.04,
                 vel_smooth: float = 0.6, traj_dt: float = 0.01,
                 traj_speed: float = 1, max_lead: float = 0.12,
                 land_tol: float = 0.02, follow_timeout_s: float = 90.0,
                 follow_log_every_s: float = 2.0, accel_max: float = 4.0,
                 held_speed_scale: float = DEFAULT_HELD_SPEED_SCALE):
        assert hand_name in ("inspire", "inspire_left"), \
            f"franka executor supports inspire hands only, got {hand_name}"
        self.dt = dt
        self.hand_name = hand_name
        self.squeeze_level = squeeze_level
        self.arm_speed_scale = arm_speed_scale   # free-space blocking-move speed (home)
        self.ctrl_dt = ctrl_dt                   # velocity-stream command period (s)
        self.joint_vmax = joint_vmax             # per-joint velocity clip (rad/s)
        self.pos_kp = pos_kp                     # velocity gain toward the target (1/s)
        self.follow_tol = follow_tol             # advance target once arm within this (rad)
        self.vel_smooth = vel_smooth             # EMA on the commanded velocity (0 = off)
        self.traj_dt = traj_dt                   # cuRobo interpolation_dt of the planned traj
        self.traj_speed = traj_speed             # playback scale of the planned traj timing
        self.max_lead = max_lead                 # reference may not run this far ahead (rad)
        self.land_tol = land_tol                 # skip the final blocking move within this (rad)
        self.follow_timeout_s = follow_timeout_s # hard wall-clock cap on one _follow
        self.follow_log_every_s = follow_log_every_s   # progress print period
        self.accel_max = accel_max               # slew limit on the command (rad/s^2)
        if not np.isfinite(held_speed_scale) or held_speed_scale <= 0:
            raise ValueError("held_speed_scale must be a positive finite value")
        # Absolute cap for every arm motion from squeeze through release.  It
        # is deliberately enforced in _follow rather than relying on each
        # caller to remember a speed keyword.
        self.held_speed_scale = float(held_speed_scale)
        self._holding_object = False
        # Bound by the runners once their shared GraspPlanner is available.
        # It supplies endpoint FK for generic joint trajectories (reset,
        # clear-view, recovery), which otherwise have no wrist target.
        self._speed_profile_planner = None
        # Captured at release in the same FK frame as planner targets. It is
        # the closest reliable proxy for the placed object while the hand
        # begins its retreat, and remains active until clear-view is reached.
        self._last_release_wrist_reference = None
        # Created only after a complete placement preflight succeeds. It holds
        # the already-validated post-release retract following the immediate
        # 10cm vertical exit, so reset() never needs to invent a lateral move
        # beside a newly placed object.
        self._pending_post_release_retract = None
        self._arm_init = np.asarray(FR3_INIT, dtype=np.float64)          # 7-DOF home
        self._clear_view = self._arm_init.copy()
        self._clear_view[0] += np.deg2rad(CLEAR_VIEW_J0_DEG)
        self._hand_init = np.asarray(INSPIRE_INIT, dtype=np.float64)
        # Finger config (planner order, radians) the hand was last commanded to.
        # reset() plans its retract from this, so it must track reality.
        self._last_hand_qpos = self._hand_init.copy()
        # Last command actually sent to the hand (controller units) — the start
        # point for _ramp_hand, so a ramp never jumps from a stale value.
        self._last_hand_action = _convert_inspire(self._hand_init)
        self._link6_to_wrist = np.asarray(FR3_INSPIRE_LINK_TO_WRIST, dtype=np.float64)
        self._convert = _convert_inspire
        self.state_timestamps = []
        # run_auto drives both arms through the same trial code; it reads this
        # to slice arm vs hand columns instead of hard-coding the xarm's 6.
        self.arm_dof = 7
        # set by _follow when a descent stopped on the wrench threshold — place()
        # reports it in place_info (run_auto's early-contact check reads it).
        self._last_stop_on_contact = False

        from paradex.io.robot_controller import get_arm, get_hand
        self.arm = get_arm("franka")
        self.hand = get_hand(hand_name)

        # A previous run killed mid-motion (watchdog SIGKILL) leaves the daemon's
        # velocity stream ACTIVE, and that stream holds g_robot_mutex — every
        # command below (set_collision_behavior in particular) would then block
        # until the ZMQ timeout. Close it first.
        try:
            self.arm.stop_streaming()
        except Exception as e:
            print(f"[franka] initial stop_streaming failed: {e!r}")

        # Clear any leftover error/reflex from a previous run (a hard reflex can
        # leave the FR3 in "Other" mode where commands are rejected + the daemon
        # is sluggish, so set_collision_behavior below can ZMQ-timeout).
        for _ in range(3):
            try:
                if self.arm.is_error():
                    self.arm.error_recovery()
            except Exception:
                pass
            time.sleep(0.3)
        # FR3 detects contact from its LINK-SIDE TORQUE SENSORS. Loosen thresholds
        # so free-space inertia doesn't reflex-stop; real hard contact still trips.
        # Retry — the first call after an error can time out before the daemon
        # settles.
        for attempt in range(3):
            try:
                self.arm.set_collision_behavior(
                    [30.0] * 7, [60.0] * 7, [30.0] * 6, [60.0] * 6)
                break
            except Exception as e:
                print(f"[franka] set_collision_behavior attempt {attempt+1} failed: {e!r}")
                time.sleep(0.5)
        else:
            # Continuing without the configured collision thresholds is unsafe,
            # and a failed call is commonly the first symptom of a lost
            # libfranka TCP session.  Fail here instead of turning a missing
            # state sample into an unrelated TypeError in home().
            raise RuntimeError(
                "FR3 is not accepting commands: set_collision_behavior failed "
                "three times. Check the franka daemon, robot network/FCI state, "
                "then clear any Desk fault before retrying.")

    # ── low-level ────────────────────────────────────────────────────────────

    def _log(self, state: str):
        self.state_timestamps.append(
            {"state": state, "time": datetime.datetime.now().isoformat()})
        # printed too: without it a stage that blocks (a planner call, a blocking
        # move) is indistinguishable from a hang
        print(f"[franka] >>> {state}", flush=True)

    @staticmethod
    def _clip_hand(action: np.ndarray) -> np.ndarray:
        """Clamp an inspire command to the controller's 0-1000 range.

        The squeeze / reverse-squeeze ramps EXTRAPOLATE past grasp
        (``g*(1+i/5) - pg*(i/5)``), which is fine for allegro (``_convert``
        returns radians) but overshoots the range for inspire, whose
        ``_convert`` returns controller units. A negative command is not
        rejected downstream: ``data2bytes`` sends ``v & 0xff`` / ``(v>>8) & 0xff``
        (only ``-1`` is special-cased), so e.g. -25 reaches the hand as 65511
        and the finger jumps to a garbage angle — which then gets HELD for the
        whole lift via the ``hold`` array."""
        return np.clip(np.asarray(action, dtype=np.float64), 0.0, 1000.0)

    def _move_hand(self, action: np.ndarray):
        action = self._clip_hand(action)
        self.hand.move(action)
        self._last_hand_action = action
        time.sleep(self.dt)

    def set_speed_profile_planner(self, planner) -> None:
        """Bind the live planner used to FK generic trajectory endpoints.

        Cartesian plans already provide an explicit wrist target.  A reset or
        clear-view joint trajectory does not, so retaining this shared planner
        lets those paths use the same 25--10 cm speed profile without rebuilding
        a kinematics model or adding a second GPU context.
        """
        self._speed_profile_planner = planner

    def _trajectory_wrist_target(self, arm_qpos: np.ndarray,
                                 warn: bool = True) -> Optional[np.ndarray]:
        """Return the bound planner's end-effector position for ``arm_qpos``.

        The cuRobo ee link is the planner wrist convention used by all pose
        plans.  Finger joints do not affect that fixed link, so the planner's
        init hand state is sufficient for FK of a generic 7-DoF arm endpoint.
        Returning ``None`` preserves the original motion behaviour if a caller
        intentionally uses this executor without a planner.
        """
        planner = self._speed_profile_planner
        motion_gen = getattr(planner, "_motion_gen", None)
        if planner is None or motion_gen is None:
            return None
        try:
            n_arm = int(getattr(planner, "_n_arm", self.arm_dof))
            init_state = np.asarray(planner._init_state, dtype=np.float32).copy()
            arm = np.asarray(arm_qpos, dtype=np.float32).reshape(-1)
            if init_state.ndim != 1 or len(init_state) < n_arm or len(arm) < n_arm:
                return None
            init_state[:n_arm] = arm[:n_arm]
            import torch
            kin = motion_gen.kinematics.get_state(
                torch.tensor(init_state, dtype=torch.float32,
                             device=planner._tensor_args.device).unsqueeze(0))
            xyz = np.asarray(kin.ee_position[0].detach().cpu().numpy(),
                             dtype=np.float64)
            if xyz.shape != (3,) or not np.isfinite(xyz).all():
                return None
            target = np.eye(4, dtype=np.float64)
            target[:3, 3] = xyz
            return target
        except Exception as exc:
            # A speed profile must never turn a valid recovery trajectory into
            # a failed motion merely because optional endpoint FK is absent.
            if warn:
                print(f"[franka] endpoint FK unavailable; using base speed: {exc!r}")
            return None

    def _validate_vertical_stroke(self, arm_traj: np.ndarray,
                                  direction: int, label: str) -> None:
        """Require a planned 10cm stroke to be geometrically vertical.

        The start/end poses are constructed at identical x/y. Sample planner
        FK over the returned path to verify the commanded signed z travel and
        prevent an upward/backtracking stroke; collision clearance itself is
        provided by cuRobo's planned trajectory.
        """
        if direction not in (-1, 1):
            raise ValueError("vertical stroke direction must be -1 or +1")
        planner = self._speed_profile_planner
        motion_gen = getattr(planner, "_motion_gen", None)
        if planner is None or motion_gen is None:
            raise RuntimeError(
                f"{label}: planner FK unavailable; refusing unverified "
                "vertical placement stroke")
        arm = np.asarray(arm_traj, dtype=np.float32)
        if arm.ndim != 2 or len(arm) < 2 or arm.shape[1] < self.arm_dof:
            raise RuntimeError(f"{label}: invalid arm trajectory for vertical check")
        try:
            n_arm = int(getattr(planner, "_n_arm", self.arm_dof))
            q = np.tile(np.asarray(planner._init_state, dtype=np.float32),
                        (len(arm), 1))
            q[:, :n_arm] = arm[:, :n_arm]
            import torch
            kin = motion_gen.kinematics.get_state(torch.tensor(
                q, dtype=torch.float32, device=planner._tensor_args.device))
            xyz = np.asarray(kin.ee_position.detach().cpu().numpy(),
                             dtype=np.float64)
        except Exception as exc:
            raise RuntimeError(
                f"{label}: planner FK failed; refusing unverified vertical "
                f"placement stroke ({exc!r})") from exc
        if xyz.shape != (len(arm), 3) or not np.isfinite(xyz).all():
            raise RuntimeError(f"{label}: invalid planner FK samples")
        signed_steps = direction * np.diff(xyz[:, 2])
        signed_travel = float(direction * (xyz[-1, 2] - xyz[0, 2]))
        if (signed_travel < PLACE_VERTICAL_TRAVEL_M - VERTICAL_STROKE_Z_TOL_M
                or (len(signed_steps)
                    and np.min(signed_steps) < -VERTICAL_STROKE_Z_TOL_M)):
            raise RuntimeError(
                f"{label}: invalid vertical-stroke z motion "
                f"(signed_dz={signed_travel * 1000:.1f}mm); "
                "object remains held")

    def _plan_verified_vertical_stroke(
            self, planner, start_full: np.ndarray,
            wrist_start: np.ndarray, wrist_end: np.ndarray,
            scene_cfg: dict, include_obj_obstacle: bool, label: str,
            debug_dump_dir: Optional[str] = None) -> np.ndarray:
        """Plan and validate one 10cm vertical placement stroke.

        The whole stroke is solved once and then played at the held-object
        0.25x cap.  FK samples every returned waypoint: a bowed joint-space
        path is rejected during planning rather than being slowed down by
        splitting the same 10cm motion into many artificial sub-trajectories.
        """
        start = np.asarray(wrist_start, dtype=np.float64)
        end = np.asarray(wrist_end, dtype=np.float64)
        if start.shape != (4, 4) or end.shape != (4, 4):
            raise ValueError(f"{label}: wrist poses must be 4x4")
        delta = end[:3, 3] - start[:3, 3]
        if abs(abs(delta[2]) - PLACE_VERTICAL_TRAVEL_M) > VERTICAL_STROKE_Z_TOL_M:
            raise RuntimeError(
                f"{label}: expected a 10cm vertical stroke, got "
                f"delta={delta.round(5).tolist()}")
        direction = 1 if delta[2] > 0 else -1
        traj = planner.plan_pose_constrained(
            np.asarray(start_full, dtype=np.float32), end,
            hold_vec_weight=[0, 0, 0, 0, 0, 0],
            scene_cfg=scene_cfg, include_obj_obstacle=include_obj_obstacle,
            debug_dump_dir=debug_dump_dir)
        if traj is None:
            raise RuntimeError(f"{label}: 10cm vertical-stroke preflight failed")
        traj = np.asarray(traj)
        self._validate_vertical_stroke(traj[:, :7], direction, label)
        return traj

    def _record_release_wrist_reference(self) -> None:
        """Remember the wrist location beside the object just released.

        A reset starts at this point and initially moves *away* from the
        object, so endpoint-only slowdown would otherwise start at 1.0x. The
        cached reference keeps the same distance profile active while exiting
        the object-clearance zone.
        """
        try:
            state = self.arm.get_data()
            qpos = None if state is None else state.get("qpos")
            ref = (None if qpos is None else
                   self._trajectory_wrist_target(qpos, warn=False))
            if ref is None and state is not None:
                link6 = np.asarray(state.get("position"), dtype=np.float64)
                if link6.shape == (4, 4) and np.isfinite(link6).all():
                    ref = link6 @ self._link6_to_wrist
            if ref is not None:
                self._last_release_wrist_reference = np.asarray(
                    ref, dtype=np.float64).copy()
        except Exception as exc:
            print(f"[franka] release proximity reference unavailable: {exc!r}")

    def _ramp_hand(self, target: np.ndarray, steps: int = 25, step_dt: float = 0.01):
        """Move the hand to ``target`` as a ramp from where it was last
        commanded. A single command is a max-speed slam on the inspire
        controller (``setspeed`` 1000), which is what makes the fingers snap
        open — the target itself is fine, the step is not."""
        target = self._clip_hand(target)
        start = self._clip_hand(self._last_hand_action)
        if np.max(np.abs(target - start)) < 1.0:      # already there
            self._move_hand(target)
            return
        for i in range(1, steps + 1):
            t = i / steps
            self._move_hand(start * (1.0 - t) + target * t)
            time.sleep(step_dt)

    def _move_to(self, target_qpos: np.ndarray, speed_scale: Optional[float] = None,
                 threshold: float = 0.1, what: str = "move",
                 slowdown_wrist_target: Optional[np.ndarray] = None):
        """Move safely to a free-space joint target and verify arrival.

        The distance profile is object-proximity based, never destination based.
        We therefore stream a dense path only while a release/object reference
        is active (or an explicit object reference is supplied); a normal
        clear-view destination by itself does not trigger a slowdown.
        """
        ss = self.arm_speed_scale if speed_scale is None else speed_scale
        target = np.asarray(target_qpos, dtype=np.float64)[:7]
        try:
            object_refs = ([] if self._last_release_wrist_reference is None
                           else [self._last_release_wrist_reference])
            if slowdown_wrist_target is not None:
                object_refs.append(slowdown_wrist_target)
            if not object_refs:
                self.arm.move(target, is_servo=False, speed_scale=ss)
            else:
                state = self.arm.get_data()
                current = None if state is None else state.get("qpos")
                if current is None:
                    raise RuntimeError("FR3 state is unavailable before move")
                current = np.asarray(current, dtype=np.float64)[:7]
                # This is the same free-space joint interpolation as the
                # blocking daemon command, but made dense for live profiling.
                # Keep the nominal finite-difference velocity below joint_vmax.
                delta = float(np.max(np.abs(target - current)))
                step = max(self.joint_vmax * self.traj_dt * 0.75, 1e-4)
                n_waypoints = max(2, int(np.ceil(delta / step)) + 1)
                self._follow(
                    np.linspace(current, target, n_waypoints), speed=ss,
                    slowdown_wrist_references=object_refs)
        except Exception as e:
            raise RuntimeError(
                f"{what}: FR3 move command failed ({e}). The daemon lost or "
                "cannot use its libfranka connection; verify robot power/network, "
                "Desk Execution mode, unlocked joints, and active FCI.") from e
        if self.arm.is_error():
            self.arm.error_recovery()
        # The FR3 state PUB lags the blocking move() return by a moment, so a qpos
        # read RIGHT after move() can be stale (pre-move) → false "not at target".
        # Poll until the state settles within threshold (or time out).
        err = None
        for _ in range(30):
            time.sleep(0.05)
            data = self.arm.get_data()
            qpos = None if data is None else data.get("qpos")
            if qpos is None:
                raise RuntimeError(
                    f"{what}: FR3 state is unavailable after move. The daemon is "
                    "not receiving robot state (check its libfranka connection).")
            err = float(np.linalg.norm(np.asarray(qpos, dtype=np.float64) - target))
            if err < threshold:
                return
        raise RuntimeError(
            f"{what}: arm not at target (err={err:.3f}); "
            f"qpos={self.arm.get_data()['qpos'].round(3)}")

    @staticmethod
    def _approach_speed_scale(wrist_distance_m: float) -> float:
        """Return the final approach/descent speed scale for a wrist target.

        The profile is 1.0 at/above 25 cm, decreases linearly to 0.25 over
        10--25 cm, and remains 0.25 inside 10 cm.  It scales trajectory time
        playback and feedforward velocity together; joint and acceleration
        caps still apply afterwards.
        """
        distance = float(wrist_distance_m)
        if not np.isfinite(distance) or distance >= APPROACH_SLOWDOWN_FAR_M:
            return 1.0
        if distance <= APPROACH_SLOWDOWN_NEAR_M:
            return APPROACH_SLOWDOWN_MIN_SCALE
        progress = ((distance - APPROACH_SLOWDOWN_NEAR_M)
                    / (APPROACH_SLOWDOWN_FAR_M - APPROACH_SLOWDOWN_NEAR_M))
        return (APPROACH_SLOWDOWN_MIN_SCALE
                + (1.0 - APPROACH_SLOWDOWN_MIN_SCALE) * progress)

    def _follow(self, arm_traj: np.ndarray, hand_traj: Optional[np.ndarray] = None,
                speed: Optional[float] = None, abort_on_contact: bool = False,
                stop_wrench_z: Optional[float] = None,
                slowdown_wrist_target: Optional[np.ndarray] = None,
                slowdown_wrist_references=None):
        """Follow a DENSE joint trajectory SMOOTHLY (no per-waypoint stop) by
        STREAMING joint velocities — continuous motion along the cuRobo path.

        Per control tick we command the finite-difference velocity to the next
        chunk of the trajectory; because commands are sent back-to-back the arm
        never decelerates to zero between waypoints (unlike blocking ``move()``).
        ``traj_speed`` compresses the trajectory's own timing to move faster.
        ``slowdown_wrist_target`` and optional ``slowdown_wrist_references``
        are *object-proximity* wrist poses in the planner frame, not generic
        motion destinations. The current wrist is recomputed from the live
        measured joints with the same cuRobo FK every control tick, so a
        physical FR3 link-6/URDF tool-frame offset cannot postpone the final
        slowdown. The closest object reference controls the 1.0x at 25 cm to
        0.25x at 10 cm profile. This covers grasp approach and retreat from the
        object just released. While an object is held, the independent
        held-object safety cap takes precedence: no arm motion may exceed 0.25x.

        Reference tracking, NOT waypoint chasing. The old loop gated the target
        on arrival (advance only once the arm is within ``follow_tol``), which
        makes the arm pull its own reference: it decelerates as it closes on
        traj[k], the target then jumps ahead, it accelerates again. That
        stop-go crawl — at a fraction of the planned speed, regardless of
        ``joint_vmax`` — is the chatter. Instead:
          * the reference advances with TIME along the planned traj
            (``traj_dt`` per waypoint, scaled by ``traj_speed``), interpolated
            between waypoints, so it is continuous;
          * it is CLAMPED: if the reference would sit more than ``max_lead``
            ahead of the measured qpos it stops advancing until the arm catches
            up. This keeps the old guarantee (never drive at a far waypoint,
            never cut a corner off the collision-free path) without gating the
            velocity;
          * command = the trajectory's own velocity (feedforward) + ``pos_kp`` *
            tracking error, so steady-state motion is set by the plan, not by
            the error;
          * EMA low-pass (``vel_smooth``), and ``ctrl_dt`` 50 Hz feeding the
            daemon's 1 kHz stream (10 rad/s² rate limit).

        Safety: velocities clipped to ``joint_vmax``; on a reflex (is_error) we
        zero-velocity + recover (+ abort if requested); a target that stops
        advancing for ``STALL_TICKS`` breaks the loop. NOTE: ``duration_ms`` is
        parsed but NEVER used by franka_daemon (no watchdog) — the last commanded
        velocity is held until the next command, so this loop is the only thing
        stopping the arm. At the end we decelerate to zero and land on the final
        config with a blocking ``move()`` ONLY when the residual exceeds
        ``land_tol`` (skipping it avoids a stream→position mode switch, which is
        what makes the clunk between segments)."""
        traj = np.atleast_2d(np.asarray(arm_traj, dtype=np.float64))[:, :7]
        self._last_stop_on_contact = False
        n = len(traj)
        if n == 0:
            return
        speed = self.traj_speed if speed is None else float(speed)
        if not np.isfinite(speed) or speed <= 0:
            raise ValueError("trajectory speed must be a positive finite value")
        slow_targets = []
        if slowdown_wrist_target is not None:
            target = np.asarray(slowdown_wrist_target, dtype=np.float64)
            if target.shape != (4, 4) or not np.isfinite(target).all():
                raise ValueError("slowdown_wrist_target must be a finite 4x4 pose")
            slow_targets.append(("object_0", target))
        if slowdown_wrist_references is not None:
            for i, reference in enumerate(slowdown_wrist_references):
                ref = np.asarray(reference, dtype=np.float64)
                if ref.shape != (4, 4) or not np.isfinite(ref).all():
                    raise ValueError(
                        "slowdown_wrist_references must contain finite 4x4 poses")
                slow_targets.append((f"release_{i}", ref))
        idx = 0.0                                       # float reference index into traj
        k_arm = 0                                       # index the ARM has actually reached
        # A held-object motion is capped independently of endpoint distance.
        # For a free hand, the 25--10 cm profile may reduce the final motion to
        # one quarter of its requested base speed.  Size the progress guard for
        # the slowest valid rate in either case.
        held_cap = self.held_speed_scale if self._holding_object else None
        min_effective_speed = (
            min(speed, held_cap) if held_cap is not None else
            speed * (APPROACH_SLOWDOWN_MIN_SCALE if slow_targets else 1.0)
        )
        base_d_idx = speed * self.ctrl_dt / self.traj_dt
        max_ticks = int(20 * n / max(
            min_effective_speed * self.ctrl_dt / self.traj_dt, 1e-3)) + 500
        # WALL-CLOCK cap. max_ticks alone is not a usable bound: n is the length
        # of the cuRobo INTERPOLATED traj (thousands of waypoints), so 20*n ticks
        # is tens of minutes of silent spinning if the arm crawls but never stops
        # (the stall check only fires when the reference stops advancing).
        t_start = time.time()
        deadline = t_start + self.follow_timeout_s
        t_last_print = t_start
        idx_at_print, t_prev_print = 0.0, t_start
        ticks = 0
        dq_prev = np.zeros(7)                           # EMA state (starts from rest)
        last_idx, last_prog_tick = 0.0, 0
        STALL_TICKS = int(2.5 / self.ctrl_dt)           # break if no progress ~2.5s
        wrist_distance_m = None
        wrist_distance_frame = None
        wrist_distance_reference = None
        profile_scale = 1.0
        effective_speed = speed
        if held_cap is not None:
            print(f"[franka] held-object speed cap: {held_cap:.2f}x", flush=True)
        try:
            while ticks < max_ticks:
                ticks += 1
                now = time.time()
                if now > deadline:
                    print(f"[franka] follow TIMEOUT after {self.follow_timeout_s:.0f}s "
                          f"at waypoint {int(idx)}/{n} — stopping")
                    break
                arm_state = self.arm.get_data()
                cur = np.asarray(arm_state["qpos"][:7], dtype=np.float64)
                profile_scale = 1.0
                wrist_distance_m = None
                wrist_distance_frame = None
                wrist_distance_reference = None
                if slow_targets:
                    # ``PlanResult.wrist_se3`` is cuRobo's ee_link pose, while
                    # the physical daemon reports FR3's O_T_EE.  Those frames
                    # have a fixed mounted-tool offset (~107 mm on this rig),
                    # so subtracting them directly makes a true final 10 cm
                    # approach look much farther away.  FK measured joints in
                    # cuRobo instead; only fall back to the physical frame for
                    # planner-less callers.
                    wrist_now = self._trajectory_wrist_target(cur, warn=False)
                    if wrist_now is not None:
                        wrist_distance_frame = "planner_fk"
                    else:
                        link6_pose = np.asarray(arm_state.get("position"),
                                                dtype=np.float64)
                        if (link6_pose.shape == (4, 4)
                                and np.isfinite(link6_pose).all()):
                            wrist_now = link6_pose @ self._link6_to_wrist
                            wrist_distance_frame = "physical_fallback"
                    if wrist_now is not None:
                        wrist_distance_reference, wrist_distance_m = min(
                            ((label, float(np.linalg.norm(
                                wrist_now[:3, 3] - ref[:3, 3])))
                             for label, ref in slow_targets),
                            key=lambda item: item[1])
                        profile_scale = self._approach_speed_scale(wrist_distance_m)
                # The object-held policy is an absolute cap, not a second
                # multiplier.  Thus every carry/lift/reorient/descent is at
                # most 0.25x, while the distance rule controls every free-hand
                # approach/retract/clear-view move exactly as specified.
                effective_speed = (min(speed, held_cap)
                                   if held_cap is not None
                                   else speed * profile_scale)
                d_idx = effective_speed * self.ctrl_dt / self.traj_dt
                if now - t_last_print >= self.follow_log_every_s:
                    t_last_print = now
                    # ref_rate vs the nominal d_idx/ctrl_dt tells you whether the
                    # max_lead clamp is throttling (ref_rate well below nominal =
                    # the arm cannot keep up = stop-go). Tune traj_speed to match.
                    slowdown_text = ("" if wrist_distance_m is None else
                                     f" wrist={wrist_distance_m * 100:.1f}cm "
                                     f"profile={profile_scale:.2f} "
                                     f"frame={wrist_distance_frame} "
                                     f"ref={wrist_distance_reference}")
                    held_text = ("" if held_cap is None else
                                 f" held_cap={held_cap:.2f}")
                    print(f"  [follow] {now - t_start:5.1f}s  ref {int(idx)}/{n}  "
                          f"arm {k_arm}/{n}  err={np.linalg.norm(traj[int(idx)] - cur):.3f}  "
                          f"ref_rate={(idx - idx_at_print) / (now - t_prev_print):.0f}/s "
                          f"(nominal {d_idx / self.ctrl_dt:.0f}/s){slowdown_text}"
                          f"{held_text}",
                          flush=True)
                    idx_at_print, t_prev_print = idx, now
                # Advance the reference in TIME, but only while it stays within
                # max_lead of the arm (else the arm has fallen behind — e.g. the
                # plan is faster than joint_vmax — and we wait for it).
                nxt = min(idx + d_idx, float(n - 1))
                if np.linalg.norm(traj[int(nxt)] - cur) <= self.max_lead:
                    idx = nxt
                k = int(idx)
                frac = idx - k
                # The arm TRACKS the reference, so it sits behind it. Index the
                # hand by where the arm ACTUALLY is, not by the reference —
                # otherwise the fingers run ahead of the arm and close before
                # the hand has reached the object.
                while (k_arm < n - 1
                       and np.linalg.norm(traj[k_arm] - cur) < self.follow_tol):
                    k_arm += 1
                ref = (traj[k] if k >= n - 1
                       else traj[k] * (1.0 - frac) + traj[k + 1] * frac)
                err = ref - cur
                if idx >= n - 1 and np.linalg.norm(traj[-1] - cur) < self.follow_tol:
                    break                               # reached final waypoint
                # STALL: the reference hasn't advanced in a while -> the arm is
                # stuck (reflex, unreachable target). Stop instead of spinning.
                if idx > last_idx + 1e-9:
                    last_idx, last_prog_tick = idx, ticks
                elif ticks - last_prog_tick > STALL_TICKS:
                    print(f"[franka] follow stalled at waypoint {k}/{n} "
                          f"(err={np.linalg.norm(err):.3f}) — stopping")
                    break
                # Feedforward = the planned trajectory's own joint velocity at
                # the reference; P term only corrects the tracking error.
                v_ff = ((traj[k + 1] - traj[k]) / self.traj_dt * effective_speed
                        if k < n - 1 else np.zeros(7))
                dq = v_ff + self.pos_kp * err
                m = float(np.max(np.abs(dq)))
                if m > self.joint_vmax:
                    dq = dq * (self.joint_vmax / m)     # proportional cap (keep direction)
                # EMA low-pass — the cap is applied first so the blend of two
                # capped commands stays within the cap.
                dq = self.vel_smooth * dq_prev + (1.0 - self.vel_smooth) * dq
                # HARD acceleration limit. The EMA only softens a step, it does
                # not bound it, and the daemon's own limiter is 10 rad/s^2 — so
                # any jump in the command reaches the joints as a jolt. Bound the
                # per-tick change instead.
                d_max = self.accel_max * self.ctrl_dt
                dq = dq_prev + np.clip(dq - dq_prev, -d_max, d_max)
                dq_prev = dq
                self.arm.set_joint_velocity(dq, duration_ms=int(self.ctrl_dt * 1000 * 2.0))
                if hand_traj is not None:
                    hand_cmd = self._clip_hand(
                        hand_traj[min(k_arm, len(hand_traj) - 1)])
                    self.hand.move(hand_cmd)
                    self._last_hand_action = hand_cmd
                time.sleep(self.ctrl_dt)
                if stop_wrench_z is not None:
                    wz = float(self.arm.get_data()["wrench"][2])
                    if abs(wz) > stop_wrench_z:
                        print(f"[franka] contact wrench_z={wz:.1f}N — stop descent")
                        self._last_stop_on_contact = True
                        return                          # stop here (no final land)
                if self.arm.is_error():
                    self.arm.set_joint_velocity(np.zeros(7), duration_ms=50)
                    self.arm.error_recovery()
                    if abort_on_contact:
                        raise ContactAbort(
                            f"collision reflex during motion near waypoint {k}/{n}")
                    print("[franka] arm reflex during motion — recovered")
        finally:
            print(f"  [follow] loop exit at ref {int(idx)}/{n} "
                  f"({time.time() - t_start:.1f}s) — decelerating", flush=True)
            self.arm.set_joint_velocity(np.zeros(7), duration_ms=100)   # decelerate
            time.sleep(0.08)
            # END THE STREAM. The daemon's velocity stream holds g_robot_mutex
            # for the whole robot_.control() call, and commands that do NOT stop
            # the stream first (set_collision_behavior, set_load, set_ee) then
            # block on that mutex until the ZMQ timeout. Zero velocity is not
            # enough — streaming_active_ stays true.
            try:
                print("  [follow] stop_streaming ...", flush=True)
                self.arm.stop_streaming()
                print("  [follow] stop_streaming done", flush=True)
            except Exception as e:
                print(f"[franka] stop_streaming failed: {e!r}", flush=True)
        # Finish the hand on the trajectory's LAST config. The loop exits on the
        # reference index but the hand follows the arm's measured index, so a
        # normal exit can leave k_arm short of the end — without this the fingers
        # stop at a mid-trajectory shape. Deliberately placed after the try
        # block: a ContactAbort must NOT drive the fingers any further.
        if hand_traj is not None:
            self._ramp_hand(hand_traj[-1])
        # Land on the final config only if the stream left a real residual —
        # a blocking move() tears down the velocity stream and starts a position
        # generator, and that mode switch is the jolt at each segment boundary.
        resid = float(np.linalg.norm(
            np.asarray(self.arm.get_data()["qpos"][:7], dtype=np.float64) - traj[-1]))
        if resid > self.land_tol:
            # BLOCKING: the daemon's servo_joint_position runs up to 25s (Python
            # waits up to 30s), and it prints nothing — so bracket it.
            print(f"  [follow] landing blocking move (resid={resid:.3f}) ...",
                  flush=True)
            t_land = time.time()
            # Apply the current endpoint profile / held-object cap to the
            # occasional final blocking correction too.  Without this, a
            # correctly slowed move could finish its last residual at base
            # speed.
            landing_speed = min(
                self.arm_speed_scale,
                max(float(effective_speed), 1e-3),
            )
            self.arm.move(traj[-1], is_servo=False, speed_scale=landing_speed)
            print(f"  [follow] landed ({time.time() - t_land:.1f}s)", flush=True)
        else:
            print(f"  [follow] done (resid={resid:.3f}, no landing move)", flush=True)

    # ── public API ─────────────────────────────────────────────────────────

    def start_recording(self, save_dir: str):
        import os
        os.makedirs(save_dir, exist_ok=True)
        self.hand.start(os.path.join(save_dir, "hand"))
        self.arm.start(os.path.join(save_dir, "arm"))

    def stop_recording(self):
        for ctrl in (self.arm, self.hand):
            if (getattr(ctrl, "save_path", None) is not None
                    or getattr(ctrl, "capture_path", None) is not None):
                ctrl.stop()

    def home(self, clear_view: bool = False):
        """Move to FR3_INIT (approach-ready) or the clear-view pose (arm out of
        the cameras' view of the object) and open the hand. Free-space blocking
        move. Call ``home(clear_view=True)`` BEFORE perception so the arm does
        not occlude the object."""
        self._log("clear_view" if clear_view else "init")
        self._move_hand(self._convert(self._hand_init))
        self._last_hand_qpos = self._hand_init.copy()
        # ``home`` explicitly opens the hand before moving; no held-object
        # speed cap remains after that command.
        self._holding_object = False
        target = self._clear_view if clear_view else self._arm_init
        self._move_to(target, what="home")
        self._last_release_wrist_reference = None
        self._pending_post_release_retract = None

    def execute(self, plan_result: PlanResult, planner=None, scene_cfg=None,
                lift_height: float = 0.10, skip_lift: bool = False,
                debug_dump_dir: Optional[str] = None,
                lift_traj_override: Optional[np.ndarray] = None,
                start_from_current: bool = False,
                held_speed_scale: float = 1.0):
        """init -> approach -> pregrasp -> grasp -> squeeze -> lift.
        (mirrors real.py execute). Returns the squeezed hand action or None.

        ``debug_dump_dir`` / ``lift_traj_override`` exist so run_auto can drive
        this executor with the same call it makes for the xarm: the override is
        the lift trajectory the viz already planned (so what the user previewed
        is what runs), and the dump dir goes to plan_pose_constrained.

        ``start_from_current`` is for a continuous loop whose planner was
        seeded with measured joints. It skips the legacy return to ``FR3_INIT``
        so a verified empty-grasp retry stays local to the object."""
        if not plan_result.success:
            print("[franka] plan failed — nothing to execute")
            return None
        if held_speed_scale <= 0:
            raise ValueError("held_speed_scale must be positive")
        if planner is not None:
            self.set_speed_profile_planner(planner)

        self.state_timestamps = []
        traj = np.asarray(plan_result.traj)                 # (T, 13) = 7 arm + 6 hand
        pg_hand = self._convert(plan_result.pregrasp_pose)
        g_hand = self._convert(plan_result.grasp_pose)

        # 1. Legacy trials start from FR3_INIT. Continuous retries have a
        # trajectory from the measured state and must not silently undo that.
        self._log("current_start" if start_from_current else "init")
        if not start_from_current:
            self._move_to(self._arm_init, what="execute-init")

        # 2. Approach — stream the planned arm path; hand follows the plan's hand
        #    columns.  Slow only this final approach based on the live
        #    wrist-to-grasp-wrist distance; lift/transfer/reset retain their
        #    configured global speed.  Abort (don't grasp) if the reflex trips.
        self._log("approach")
        print("[franka] approach speed profile: 1.00x at >=25cm, linear to "
              "0.25x at 10cm, then 0.25x", flush=True)
        hand_traj = np.array([self._convert(traj[i, 7:]) for i in range(len(traj))])
        self._follow(traj[:, :7], hand_traj, abort_on_contact=True,
                     slowdown_wrist_target=plan_result.wrist_se3)

        # 3. Pregrasp
        self._log("pregrasp")
        self._move_hand(pg_hand)
        self._last_hand_qpos = np.asarray(plan_result.pregrasp_pose, dtype=np.float64)

        # 4. Grasp — ramp pregrasp -> grasp for a controlled close.
        self._log("grasp")
        for i in range(1, 51):
            t = i / 50.0
            self._move_hand(pg_hand * (1 - t) + g_hand * t)
            time.sleep(0.01)

        # 5. Squeeze
        self._log("squeeze")
        s_hand = g_hand
        for i in range(self.squeeze_level * 5):
            # clipped here (not only inside _move_hand) because s_hand is
            # RETURNED and then held for the whole lift / place descent — an
            # out-of-range value would persist through both.
            s_hand = self._clip_hand(g_hand * (1 + i / 5.0) - pg_hand * (i / 5.0))
            self._move_hand(s_hand)
            time.sleep(0.02)
        # squeeze overshoots grasp; grasp_pose is the closest planner-representable
        # config, so that is what reset should plan the retract from.
        self._last_hand_qpos = np.asarray(plan_result.grasp_pose, dtype=np.float64)
        self._holding_object = True

        if skip_lift:
            self._log("squeeze_done")
            return s_hand

        # 6. Lift — JOINT-SPACE via plan_pose_constrained (NOT cartesian). Wrist
        #    target = current wrist + z. Hold hand at squeeze during lift.
        self._log("lift")
        if planner is not None:
            # Target = the PLANNED grasp wrist (base_link) pose + z. Use the known
            # collision-free grasp pose directly — NOT O_T_EE @ link6_to_wrist,
            # which double-applies the hand offset and drops the target into the
            # table.
            wrist_lift = np.asarray(plan_result.wrist_se3, dtype=np.float64).copy()
            wrist_lift[2, 3] += lift_height
            start_full = np.concatenate([
                np.asarray(self.arm.get_data()["qpos"][:7], dtype=np.float32),
                np.asarray(plan_result.grasp_pose, dtype=np.float32)])
            if lift_traj_override is not None:
                print(f"[franka] using precomputed lift traj "
                      f"{np.shape(lift_traj_override)}", flush=True)
                traj_lift = np.asarray(lift_traj_override)
            else:
                print("[franka] planning constrained lift ...", flush=True)
                traj_lift = planner.plan_pose_constrained(
                    start_full, wrist_lift, hold_vec_weight=[1, 1, 1, 1, 1, 0],
                    scene_cfg=scene_cfg, include_obj_obstacle=False,
                    debug_dump_dir=debug_dump_dir)
            if traj_lift is not None:
                hold = np.tile(s_hand, (len(traj_lift), 1))
                print(f"[franka] held-object speed scale: {held_speed_scale:.2f} "
                      "(lift)", flush=True)
                self._follow(traj_lift[:, :7], hold, speed=held_speed_scale)
            else:
                print("[franka] constrained lift failed — holding (no cartesian fallback)")
        else:
            print("[franka] no planner given — skipping lift")
        self._log("lift_done")
        return s_hand

    def execute_lift(self, lift_traj, hold_hand, held_speed_scale: float = 1.0):
        """Joint-space lift: follow a pre-planned qpos trajectory (mirrors real.py)."""
        if held_speed_scale <= 0:
            raise ValueError("held_speed_scale must be positive")
        self._log("lift")
        # This public helper is only valid after a grasp closure.  Mark it
        # explicitly so standalone recovery callers receive the same cap.
        self._holding_object = True
        hold = np.tile(np.asarray(hold_hand, dtype=float), (len(lift_traj), 1))
        self._follow(np.asarray(lift_traj)[:, :7], hold, speed=held_speed_scale)
        self._log("lift_done")

    def _release_ramp(self, pg_hand, g_hand, slow_factor: float = 1.0,
                      open_to_init: bool = False):
        """Reverse squeeze -> grasp -> pregrasp, then STOP (real.py
        ``_release_auto``). Opening ends at PREGRASP: that is the last config
        the plan guarantees is collision-free around the object, and reset then
        retracts with the hand held there.

        ``open_to_init=True`` continues pregrasp -> hand_init as a further ramp.
        Never send that as a single command — the inspire controller runs at
        ``setspeed`` 1000, so one step is a max-speed slam."""
        sl = self.squeeze_level
        # reverse squeeze (same step rate as the squeeze ramp)
        for i in range(sl * 5):
            s_hand = g_hand * (sl - i / 5) - pg_hand * (sl - 1 - i / 5)
            self._move_hand(s_hand)
            time.sleep(0.02 * slow_factor)
        # interpolated open ramp grasp -> pregrasp (mirrors the close ramp)
        n_open_steps = 50
        for i in range(1, n_open_steps + 1):
            t = i / n_open_steps
            self._move_hand(g_hand * (1 - t) + pg_hand * t)
            time.sleep(0.01 * slow_factor)
        if not open_to_init:
            return
        # continue pregrasp -> hand_init so the fingers actually clear the object
        init_hand = self._convert(self._hand_init)
        for i in range(1, n_open_steps + 1):
            t = i / n_open_steps
            self._move_hand(pg_hand * (1 - t) + init_hand * t)
            time.sleep(0.01 * slow_factor)

    def place(self, plan_result: Optional[PlanResult] = None, planner=None,
              scene_cfg=None, grasp_wrist=None, hand_qpos=None,
              pregrasp_qpos=None, lift_height: float = PLACE_VERTICAL_TRAVEL_M,
              debug_dump_dir: Optional[str] = None,
              use_current_wrist: bool = False,
              z_force_thresh: float = 12.0,
              held_speed_scale: float = 1.0) -> dict:
        """Place through a mandatory 10 cm vertical down/up clearance.

        Target = the PLANNED grasp wrist (base_link) pose (``grasp_wrist`` =
        plan_result.wrist_se3) — i.e. exactly the object's resting spot on the
        table, which was collision-free at grasp time. Before descending, the
        wrist is planned to exactly 10 cm above that target. The second planned
        segment preserves wrist x/y/orientation and lowers exactly 10 cm; only
        after it finishes may the hand release. The matching 10 cm
        post-release lift and the following retract are also preflighted here
        and the lift runs immediately after release.

        We do NOT compute a target from ``O_T_EE @ link6_to_wrist`` (that
        double-applied the hand offset and drove the goal into the table).
        All arm movement is joint-space (``plan_pose_constrained`` + velocity
        follow); FR3 Cartesian commands are intentionally not used.

        ``plan_result`` is first so run_auto's ``executor.place(result, ...)``
        call works unchanged for both arms; the explicit ``grasp_wrist`` /
        ``hand_qpos`` / ``pregrasp_qpos`` keywords still win when given.

        The release wrist must always be explicit. Inferring it from the live
        carry height is unsafe when the carry height differs from 10 cm.

        Returns a place_info dict (``descended`` / ``target`` /
        ``stopped_on_contact``) in the same shape as real.py's place, which
        run_auto reads for its early-contact check."""
        self._log("place")
        self._pending_post_release_retract = None
        if held_speed_scale <= 0:
            raise ValueError("held_speed_scale must be positive")
        if planner is not None:
            self.set_speed_profile_planner(planner)
        if plan_result is not None:
            if grasp_wrist is None:
                grasp_wrist = plan_result.wrist_se3
            if hand_qpos is None:
                hand_qpos = plan_result.grasp_pose
            if pregrasp_qpos is None:
                pregrasp_qpos = plan_result.pregrasp_pose

        def _release():
            """Ramp squeeze -> grasp -> pregrasp and STOP there. Falls back to a
            single hand_init command only when the caller gave no finger configs."""
            if pregrasp_qpos is None or hand_qpos is None:
                self._ramp_hand(self._convert(self._hand_init))
                self._last_hand_qpos = self._hand_init.copy()
            else:
                self._release_ramp(self._convert(np.asarray(pregrasp_qpos, dtype=np.float64)),
                                   self._convert(np.asarray(hand_qpos, dtype=np.float64)))
                # opened only as far as pregrasp — reset plans its retract from
                # exactly this config and holds the fingers there
                self._last_hand_qpos = np.asarray(pregrasp_qpos, dtype=np.float64)
            self._record_release_wrist_reference()
            self._holding_object = False

        if planner is None or grasp_wrist is None:
            raise RuntimeError(
                "place requires planner and an explicit release wrist; "
                "refusing an unplanned release")
        if use_current_wrist:
            raise RuntimeError(
                "use_current_wrist is unsafe for fixed 10cm placement; "
                "pass the explicit table-height grasp_wrist instead")
        wrist_low = np.asarray(grasp_wrist, dtype=np.float64).copy()  # release pose

        # Pre-place point: every release is approached from exactly 10 cm
        # above, irrespective of the grasp lift or transfer height.
        wrist_high = wrist_low.copy()
        wrist_high[2, 3] += PLACE_VERTICAL_TRAVEL_M
        hand = (np.asarray(hand_qpos, dtype=np.float32) if hand_qpos is not None
                else np.zeros(6, dtype=np.float32))
        start_full = np.concatenate([
            np.asarray(self.arm.get_data()["qpos"][:7], dtype=np.float32), hand])

        # Preflight BOTH arm segments before the arm starts toward the table.
        # ``preplace`` may take any collision-free route while holding the
        # object; ``descend`` is constrained to retain its x/y/orientation so
        # the final 10 cm motion is perpendicular to the table.
        print(f"[franka] planning pre-place point (+{PLACE_VERTICAL_TRAVEL_M * 100:.0f}cm) ...",
              flush=True)
        preplace_traj = planner.plan_pose_constrained(
            start_full, wrist_high, hold_vec_weight=[0, 0, 0, 0, 0, 0],
            scene_cfg=scene_cfg, include_obj_obstacle=False,
            debug_dump_dir=debug_dump_dir)
        if preplace_traj is None:
            raise RuntimeError(
                "pre-place (+10cm) preflight failed; object remains held")
        descend_start = np.concatenate([preplace_traj[-1, :7], hand])
        print(f"[franka] planning perpendicular place descent (-{PLACE_VERTICAL_TRAVEL_M * 100:.0f}cm) ...",
              flush=True)
        descend_traj = self._plan_verified_vertical_stroke(
            planner, descend_start, wrist_high, wrist_low, scene_cfg,
            include_obj_obstacle=False, label="place descent",
            debug_dump_dir=debug_dump_dir)

        # Before releasing, also prove that the open hand can rise straight
        # back by 10cm and subsequently retract around the object at its new
        # resting pose. No part of this release-side chain is planned after the
        # object has been let go.
        if plan_result is None or scene_cfg is None:
            raise RuntimeError(
                "place requires plan_result and scene_cfg to preflight the "
                "post-release 10cm lift")
        T_obj_grasp = cart2se3(scene_cfg["mesh"]["target"]["pose"])
        T_obj_in_wrist = np.linalg.inv(plan_result.wrist_se3) @ T_obj_grasp
        released = wrist_low @ T_obj_in_wrist
        placed_scene = dict(scene_cfg)
        placed_scene["mesh"] = dict(scene_cfg.get("mesh", {}))
        placed_scene["mesh"]["target"] = dict(scene_cfg["mesh"]["target"])
        placed_scene["mesh"]["target"]["pose"] = se32cart(released).tolist()
        post_release_hand = (
            np.asarray(pregrasp_qpos, dtype=np.float32)
            if pregrasp_qpos is not None else self._hand_init.astype(np.float32))
        post_start = np.concatenate([
            np.asarray(descend_traj[-1, :7], dtype=np.float32),
            post_release_hand,
        ])
        print(f"[franka] planning post-release perpendicular lift (+{PLACE_VERTICAL_TRAVEL_M * 100:.0f}cm) ...",
              flush=True)
        post_lift_traj = self._plan_verified_vertical_stroke(
            planner, post_start, wrist_low, wrist_high, placed_scene,
            include_obj_obstacle=True, label="post-release place lift",
            debug_dump_dir=debug_dump_dir)
        print("[franka] planning retract after post-release lift ...", flush=True)
        retract_traj = planner.plan_js_to_init(
            placed_scene, post_lift_traj[-1, :7],
            start_hand_qpos=post_release_hand,
            goal_arm_qpos=self._clear_view[:7])
        if retract_traj is None:
            raise RuntimeError(
                "post-release retract preflight failed; object remains held")

        print(f"[franka] pre-place held-object cap: {self.held_speed_scale:.2f}x",
              flush=True)
        self._follow(preplace_traj[:, :7], speed=1.0)
        z_start = float((self.arm.get_data()["position"]
                         @ self._link6_to_wrist)[2, 3])
        print(f"[franka] perpendicular place descent: -{PLACE_VERTICAL_TRAVEL_M * 100:.0f}cm "
              f"(held cap {self.held_speed_scale:.2f}x)", flush=True)
        self._follow(descend_traj[:, :7], speed=1.0,
                     stop_wrench_z=z_force_thresh,
                     slowdown_wrist_target=wrist_low)
        contact = bool(self._last_stop_on_contact)
        z_end = float((self.arm.get_data()["position"]
                       @ self._link6_to_wrist)[2, 3])
        descended = float(abs(z_start - z_end))
        early_contact = (contact
                         and PLACE_VERTICAL_TRAVEL_M - descended > 0.005)
        if early_contact:
            # Do not release an object that has not reached its verified
            # placement point. The caller's safe recovery path keeps it held.
            self._log("place_contact_abort")
            return {"descended": descended,
                    "target": PLACE_VERTICAL_TRAVEL_M,
                    "stopped_on_contact": True,
                    "released": False,
                    "mode": "early_contact"}
        _release()                                            # squeeze -> grasp -> pregrasp
        release_refs = ([] if self._last_release_wrist_reference is None
                        else [self._last_release_wrist_reference])
        print(f"[franka] post-release perpendicular lift: +{PLACE_VERTICAL_TRAVEL_M * 100:.0f}cm",
              flush=True)
        post_hand = np.tile(self._convert(post_release_hand),
                            (len(post_lift_traj), 1))
        self._follow(post_lift_traj[:, :7], post_hand,
                     slowdown_wrist_references=release_refs)
        self._pending_post_release_retract = {
            "retract_traj": np.asarray(retract_traj),
            "post_release_lift_n_waypoints": int(len(post_lift_traj)),
        }
        self._log("place_done")
        return {"descended": descended,
                "target": PLACE_VERTICAL_TRAVEL_M,
                "stopped_on_contact": contact,
                "released": True,
                "preplace_n_waypoints": int(len(preplace_traj)),
                "descent_n_waypoints": int(len(descend_traj)),
                "post_release_lift_n_waypoints": int(len(post_lift_traj)),
                "retract_n_waypoints": int(len(retract_traj)),
                "mode": "current_wrist" if use_current_wrist else "grasp_wrist"}

    def release(self, plan_result: Optional[PlanResult] = None,
                slow_factor: float = 1.0, open_to_init: bool = False):
        """Open the hand as a ramp, squeeze -> grasp -> pregrasp, and stop there
        (real.py ``release``). Without a plan_result there are no finger configs
        to ramp between, so fall back to hand_init. Arm retract is done by
        ``reset``.

        ``open_to_init=True`` keeps ramping past pregrasp to the fully open
        hand. Pregrasp is only guaranteed clear of the object where it was
        planned; a caller that has already carried the object clear (a drop
        over the box, say) wants the fingers all the way out of the way."""
        self._log("release")
        if plan_result is not None and plan_result.pregrasp_pose is not None:
            self._release_ramp(self._convert(np.asarray(plan_result.pregrasp_pose, dtype=np.float64)),
                               self._convert(np.asarray(plan_result.grasp_pose, dtype=np.float64)),
                               slow_factor=slow_factor, open_to_init=open_to_init)
            # Track where the fingers actually ended: reset() and the retreat
            # plan their motion from this value.
            self._last_hand_qpos = (self._hand_init.copy() if open_to_init else
                                    np.asarray(plan_result.pregrasp_pose, dtype=np.float64))
        else:
            self._ramp_hand(self._convert(self._hand_init))
            self._last_hand_qpos = self._hand_init.copy()
        self._record_release_wrist_reference()
        self._holding_object = False
        # An externally initiated release has no matching preflighted
        # placement chain. reset() will build a fresh safe lift/retract plan.
        self._pending_post_release_retract = None

    def reset(self, plan_result: PlanResult, planner, scene_cfg: dict):
        """Leave a placed object via a verified lift-then-retract chain.

        The hand must first lift vertically away from the released object and
        only then travel toward clear-view.  Both segments are planned against
        the placed object *before either segment moves*, preventing a fallback
        joint-space retract from sweeping sideways through the object.
        """
        self._log("reset")
        self.set_speed_profile_planner(planner)
        if self._holding_object:
            raise RuntimeError(
                "reset was requested while the object is still held; refusing "
                "a post-release path or an implicit drop")
        pending = self._pending_post_release_retract
        if pending is not None:
            # ``place()`` planned this retract before release, then executed
            # the matching +10cm vertical exit immediately after release.
            # Only the now-clear lateral route remains.
            retract = np.asarray(pending["retract_traj"])
            release_refs = ([] if self._last_release_wrist_reference is None
                            else [self._last_release_wrist_reference])
            print("[franka] using preflighted post-place retract ...", flush=True)
            hand_traj = np.array([self._convert(retract[i, 7:])
                                  for i in range(len(retract))])
            self._follow(retract[:, :7], hand_traj,
                         slowdown_wrist_references=release_refs)
            self._last_hand_qpos = self._hand_init.copy()
            self._last_release_wrist_reference = None
            self._pending_post_release_retract = None
            self._log("reset_done")
            final_qpos = np.asarray(self.arm.get_data()["qpos"][:7], dtype=np.float64)
            return {
                "mode": "preflighted_place_lift_then_retract",
                "lift_height_m": POST_RELEASE_LIFT_HEIGHT_M,
                "lift_n_waypoints": int(pending["post_release_lift_n_waypoints"]),
                "n_waypoints": int(len(retract)),
                "final_qpos_err": float(np.linalg.norm(
                    final_qpos - self._clear_view[:7])),
            }
        if plan_result is None or not plan_result.success or planner is None:
            # This method is reached after the hand has opened.  A direct
            # clear-view motion from here can sweep through the object, so a
            # missing planner/result is a safety stop rather than permission
            # to use the historical unplanned home move.
            raise RuntimeError(
                "post-release reset requires a successful plan and planner; "
                "refusing an unverified lateral clear-view motion")

        # snapshot released object pose (robot frame) under rigid grasp
        T_obj_grasp = cart2se3(scene_cfg["mesh"]["target"]["pose"])
        T_obj_in_wrist = np.linalg.inv(plan_result.wrist_se3) @ T_obj_grasp
        T_wrist_now = self.arm.get_data()["position"] @ self._link6_to_wrist
        released = T_wrist_now @ T_obj_in_wrist

        new_scene = dict(scene_cfg)
        new_scene["mesh"] = dict(scene_cfg.get("mesh", {}))
        new_scene["mesh"]["target"] = dict(scene_cfg["mesh"]["target"])
        new_scene["mesh"]["target"]["pose"] = se32cart(released).tolist()

        cur_qpos = np.asarray(self.arm.get_data()["qpos"], dtype=np.float32)
        # Start from the finger config the hand was actually left in — pregrasp
        # after release, still pregrasp/grasp after a contact abort. Hard-coding
        # pregrasp made the planner check a hand shape the robot was not in.
        start_hand = np.asarray(self._last_hand_qpos, dtype=np.float64)
        start_full = np.concatenate([
            cur_qpos[:7], np.asarray(start_hand, dtype=np.float32)])

        # ── Preflight the complete post-release path before moving ─────────
        # The target mesh remains an obstacle: it is no longer held, and the
        # first segment is explicitly the clearance move that protects it.
        wrist_lift = self._trajectory_wrist_target(cur_qpos)
        if wrist_lift is None:
            raise RuntimeError(
                "reset lift preflight unavailable: planner FK is not ready; "
                "will not issue an unverified lateral retract")
        wrist_start = wrist_lift.copy()
        wrist_lift = wrist_lift.copy()
        wrist_lift[2, 3] += POST_RELEASE_LIFT_HEIGHT_M
        print(f"[franka] planning post-release vertical lift "
              f"(+{POST_RELEASE_LIFT_HEIGHT_M * 100:.0f}cm) ...", flush=True)
        lift_traj = self._plan_verified_vertical_stroke(
            planner, start_full, wrist_start, wrist_lift, new_scene,
            include_obj_obstacle=True, label="post-release reset lift")

        print("[franka] planning reset retract from lifted hand ...", flush=True)
        retract = planner.plan_js_to_init(
            new_scene, lift_traj[-1, :7], start_hand_qpos=start_hand,
            goal_arm_qpos=self._clear_view[:7])
        if retract is None:
            raise RuntimeError(
                "post-release retract preflight failed after a valid lift; "
                "refusing to move sideways near the object")

        # ── Both segments passed: execute the vertical clearance first ──────
        lift_hand = np.tile(self._convert(start_hand), (len(lift_traj), 1))
        print(f"[franka] post-release lift: +{POST_RELEASE_LIFT_HEIGHT_M * 100:.0f}cm",
              flush=True)
        release_refs = ([] if self._last_release_wrist_reference is None
                        else [self._last_release_wrist_reference])
        self._follow(
            lift_traj[:, :7], lift_hand,
            slowdown_wrist_references=release_refs)

        # Follow the planned hand columns during the now-clear retract. The
        # planner checks their gradual opening against the placed object rather
        # than snapping fingers open at the release site.
        hand_traj = np.array([self._convert(retract[i, 7:])
                              for i in range(len(retract))])
        self._follow(
            retract[:, :7], hand_traj,
            slowdown_wrist_references=release_refs)
        self._last_hand_qpos = self._hand_init.copy()
        self._last_release_wrist_reference = None
        self._log("reset_done")
        final_qpos = np.asarray(self.arm.get_data()["qpos"][:7], dtype=np.float64)
        return {
            "mode": "post_release_lift_then_plan_js_to_init",
            "lift_height_m": POST_RELEASE_LIFT_HEIGHT_M,
            "lift_n_waypoints": int(len(lift_traj)),
            "n_waypoints": int(len(retract)),
            "final_qpos_err": float(np.linalg.norm(
                final_qpos - self._clear_view[:7])),
        }

    # ── run_auto compatibility ───────────────────────────────────────────────
    # run_auto drives the xarm through reset -> reset_hybrid -> reset_fallback.
    # Every rung uses the same preflighted lift-then-retract route: after
    # release, an unplanned direct clear-view move is unsafe.

    def reset_hybrid(self, plan_result: PlanResult, planner=None,
                     scene_cfg: dict = None) -> dict:
        """xarm's reset_hybrid equivalent — same retract as ``reset``."""
        return self.reset(plan_result, planner, scene_cfg)

    def reset_fallback(self, plan_result: Optional[PlanResult] = None,
                       planner=None, scene_cfg: Optional[dict] = None) -> dict:
        """Safety recovery after a failed execution or reset.

        Once the hand is open, only the fully preflighted vertical-lift and
        retract path in :meth:`reset` is allowed.  If its inputs/path are not
        available, leave the arm in place and raise instead of attempting the
        old unplanned clear-view motion.
        """
        self._log("reset_fallback")
        if self._holding_object:
            # A failed placement preflight/contact happens before release. The
            # old fallback opened the hand unconditionally here, turning a
            # rejected 10cm vertical path into a drop from the carry height.
            # Keep the object clamped and require an operator or a dedicated
            # held-object recovery plan; never convert a planning failure into
            # an unverified release.
            raise RuntimeError(
                "reset_fallback: object is still held; refusing to release or "
                "drop it after an unverified placement/recovery path")
        try:
            self.release(plan_result)
        except Exception as e:
            print(f"[franka] reset_fallback release failed: {e!r}")
        if planner is None or scene_cfg is None:
            raise RuntimeError(
                "reset_fallback has no planner/scene; refusing unverified "
                "post-release clear-view motion")
        try:
            log = self.reset(plan_result, planner, scene_cfg)
        except Exception as exc:
            raise RuntimeError(
                "reset_fallback could not preflight a safe post-release "
                "lift-and-retract path; arm left in place") from exc
        log["mode"] = "reset_fallback_verified_lift_then_retract"
        return log

    def _move_joints(self, arm_traj, hand_traj=None, speed: Optional[float] = None,
                     slowdown_wrist_target: Optional[np.ndarray] = None,
                     **_ignored):
        """xarm RealExecutor's dense-trajectory follower, mapped onto _follow.
        run_auto's reposition path calls this directly. The optional slowdown
        target is an object-proximity reference; arbitrary trajectory endpoints
        are deliberately not used for speed scaling."""
        arm = np.asarray(arm_traj)[:, :7]
        release_refs = ([] if self._last_release_wrist_reference is None
                        else [self._last_release_wrist_reference])
        self._follow(arm,
                     None if hand_traj is None else np.asarray(hand_traj),
                     speed=speed,
                     slowdown_wrist_target=slowdown_wrist_target,
                     slowdown_wrist_references=release_refs)

    def shutdown(self):
        # leave the daemon in a non-streaming state so the NEXT process's
        # commands aren't blocked on g_robot_mutex
        try:
            self.arm.stop_streaming()
        except Exception:
            pass
        for ctrl in (self.arm, self.hand):
            try:
                ctrl.end()
            except Exception:
                pass
