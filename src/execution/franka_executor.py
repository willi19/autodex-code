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


class ContactAbort(RuntimeError):
    """Raised when the FR3 collision reflex trips mid-approach so the caller can
    abort the trial instead of continuing into grasp at the wrong pose."""


class FrankaExecutor:
    def __init__(self, hand_name: str = "inspire", dt: float = 0.01,
                 squeeze_level: int = 2, arm_speed_scale: float = 0.3,
                 ctrl_dt: float = 0.02, joint_vmax: float = 0.35,
                 pos_kp: float = 4.0, follow_tol: float = 0.04,
                 vel_smooth: float = 0.6, traj_dt: float = 0.01,
                 traj_speed: float = 0.25, max_lead: float = 0.12,
                 land_tol: float = 0.02, follow_timeout_s: float = 90.0,
                 follow_log_every_s: float = 2.0, accel_max: float = 0.6):
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
                 threshold: float = 0.1, what: str = "move"):
        """Free-space single-config move via a blocking ``move()`` — safe where
        there are no obstacles (home / clear-view / init). Verifies arrival."""
        ss = self.arm_speed_scale if speed_scale is None else speed_scale
        target = np.asarray(target_qpos, dtype=np.float64)[:7]
        try:
            self.arm.move(target, is_servo=False, speed_scale=ss)
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

    def _follow(self, arm_traj: np.ndarray, hand_traj: Optional[np.ndarray] = None,
                speed: Optional[float] = None, abort_on_contact: bool = False,
                stop_wrench_z: Optional[float] = None):
        """Follow a DENSE joint trajectory SMOOTHLY (no per-waypoint stop) by
        STREAMING joint velocities — continuous motion along the cuRobo path.

        Per control tick we command the finite-difference velocity to the next
        chunk of the trajectory; because commands are sent back-to-back the arm
        never decelerates to zero between waypoints (unlike blocking ``move()``).
        ``traj_speed`` compresses the trajectory's own timing to move faster.

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
        idx = 0.0                                       # float reference index into traj
        k_arm = 0                                       # index the ARM has actually reached
        d_idx = speed * self.ctrl_dt / self.traj_dt     # waypoints advanced per tick
        max_ticks = int(20 * n / max(d_idx, 1e-3)) + 500  # safety bound (no infinite loop)
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
        try:
            while ticks < max_ticks:
                ticks += 1
                now = time.time()
                if now > deadline:
                    print(f"[franka] follow TIMEOUT after {self.follow_timeout_s:.0f}s "
                          f"at waypoint {int(idx)}/{n} — stopping")
                    break
                cur = np.asarray(self.arm.get_data()["qpos"][:7], dtype=np.float64)
                if now - t_last_print >= self.follow_log_every_s:
                    t_last_print = now
                    # ref_rate vs the nominal d_idx/ctrl_dt tells you whether the
                    # max_lead clamp is throttling (ref_rate well below nominal =
                    # the arm cannot keep up = stop-go). Tune traj_speed to match.
                    print(f"  [follow] {now - t_start:5.1f}s  ref {int(idx)}/{n}  "
                          f"arm {k_arm}/{n}  err={np.linalg.norm(traj[int(idx)] - cur):.3f}  "
                          f"ref_rate={(idx - idx_at_print) / (now - t_prev_print):.0f}/s "
                          f"(nominal {d_idx / self.ctrl_dt:.0f}/s)", flush=True)
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
                v_ff = ((traj[k + 1] - traj[k]) / self.traj_dt * speed
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
            self.arm.move(traj[-1], is_servo=False, speed_scale=self.arm_speed_scale)
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
        target = self._clear_view if clear_view else self._arm_init
        self._move_to(target, what="home")

    def execute(self, plan_result: PlanResult, planner=None, scene_cfg=None,
                lift_height: float = 0.10, skip_lift: bool = False,
                debug_dump_dir: Optional[str] = None,
                lift_traj_override: Optional[np.ndarray] = None):
        """init -> approach -> pregrasp -> grasp -> squeeze -> lift.
        (mirrors real.py execute). Returns the squeezed hand action or None.

        ``debug_dump_dir`` / ``lift_traj_override`` exist so run_auto can drive
        this executor with the same call it makes for the xarm: the override is
        the lift trajectory the viz already planned (so what the user previewed
        is what runs), and the dump dir goes to plan_pose_constrained."""
        if not plan_result.success:
            print("[franka] plan failed — nothing to execute")
            return None

        self.state_timestamps = []
        traj = np.asarray(plan_result.traj)                 # (T, 13) = 7 arm + 6 hand
        pg_hand = self._convert(plan_result.pregrasp_pose)
        g_hand = self._convert(plan_result.grasp_pose)

        # 1. Init: move to FR3_INIT (the trajectory's start). Free-space.
        self._log("init")
        self._move_to(self._arm_init, what="execute-init")

        # 2. Approach — stream the planned arm path; hand follows the plan's hand
        #    columns. Abort (don't grasp) if the reflex trips en route.
        self._log("approach")
        hand_traj = np.array([self._convert(traj[i, 7:]) for i in range(len(traj))])
        self._follow(traj[:, :7], hand_traj, abort_on_contact=True)

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
                self._follow(traj_lift[:, :7], hold)
            else:
                print("[franka] constrained lift failed — holding (no cartesian fallback)")
        else:
            print("[franka] no planner given — skipping lift")
        self._log("lift_done")
        return s_hand

    def execute_lift(self, lift_traj, hold_hand):
        """Joint-space lift: follow a pre-planned qpos trajectory (mirrors real.py)."""
        self._log("lift")
        hold = np.tile(np.asarray(hold_hand, dtype=float), (len(lift_traj), 1))
        self._follow(np.asarray(lift_traj)[:, :7], hold)
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
              pregrasp_qpos=None, lift_height: float = 0.10,
              debug_dump_dir: Optional[str] = None,
              use_current_wrist: bool = False,
              z_force_thresh: float = 12.0) -> dict:
        """Descend the held object back to where it was grasped and release.

        Target = the PLANNED grasp wrist (base_link) pose (``grasp_wrist`` =
        plan_result.wrist_se3) — i.e. exactly the object's resting spot on the
        table, which was collision-free at grasp time. We do NOT compute a target
        from ``O_T_EE @ link6_to_wrist`` (that double-applied the hand offset and
        drove the goal into the table -> world-collision plan failure). JOINT-SPACE
        (plan_pose_constrained + velocity follow), NO cartesian (FR3 rejects it
        from the singular lifted pose). Stop on table reaction (``wrench[2]``).

        ``plan_result`` is first so run_auto's ``executor.place(result, ...)``
        call works unchanged for both arms; the explicit ``grasp_wrist`` /
        ``hand_qpos`` / ``pregrasp_qpos`` keywords still win when given.

        ``use_current_wrist=True`` descends ``lift_height`` from where the wrist
        IS instead of returning to the planned grasp pose — needed when the arm
        was repositioned after the lift (run_auto's reposition mode), where the
        original grasp wrist is no longer above the object.

        Returns a place_info dict (``descended`` / ``target`` /
        ``stopped_on_contact``) in the same shape as real.py's place, which
        run_auto reads for its early-contact check."""
        self._log("place")
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

        if planner is None or (grasp_wrist is None and not use_current_wrist):
            _release()                                        # can't descend — just release
            self._log("place_done")
            return {"descended": 0.0, "target": 0.0, "stopped_on_contact": False,
                    "mode": "release_only"}
        if use_current_wrist:
            # descend lift_height from HERE (arm was moved after the lift)
            wrist_low = self.arm.get_data()["position"] @ self._link6_to_wrist
            wrist_low = np.asarray(wrist_low, dtype=np.float64).copy()
            wrist_low[2, 3] -= float(lift_height)
        else:
            wrist_low = np.asarray(grasp_wrist, dtype=np.float64).copy()  # grasp pose = on table
        z_start = float((self.arm.get_data()["position"]
                         @ self._link6_to_wrist)[2, 3])
        z_target = float(wrist_low[2, 3])
        hand = (np.asarray(hand_qpos, dtype=np.float32) if hand_qpos is not None
                else np.zeros(6, dtype=np.float32))
        start_full = np.concatenate([
            np.asarray(self.arm.get_data()["qpos"][:7], dtype=np.float32), hand])
        print("[franka] planning place descend ...", flush=True)
        traj = planner.plan_pose_constrained(
            start_full, wrist_low, hold_vec_weight=[1, 1, 1, 1, 1, 0],
            scene_cfg=scene_cfg, include_obj_obstacle=False,
            debug_dump_dir=debug_dump_dir)
        if traj is None:
            print("[franka] place descend plan failed — releasing in place")
            _release()
            self._log("place_done")
            return {"descended": 0.0, "target": abs(z_start - z_target),
                    "stopped_on_contact": False, "mode": "plan_failed"}
        # hold the hand (still squeezed) during descent; stop on table contact.
        self._follow(traj[:, :7], stop_wrench_z=z_force_thresh)
        contact = bool(self._last_stop_on_contact)
        z_end = float((self.arm.get_data()["position"]
                       @ self._link6_to_wrist)[2, 3])
        _release()                                            # squeeze -> grasp -> pregrasp
        self._log("place_done")
        return {"descended": float(abs(z_start - z_end)),
                "target": float(abs(z_start - z_target)),
                "stopped_on_contact": contact,
                "mode": "current_wrist" if use_current_wrist else "grasp_wrist"}

    def release(self, plan_result: Optional[PlanResult] = None,
                slow_factor: float = 1.0):
        """Open the hand as a ramp, squeeze -> grasp -> pregrasp, and stop there
        (real.py ``release``). Without a plan_result there are no finger configs
        to ramp between, so fall back to hand_init. Arm retract is done by
        ``reset``."""
        self._log("release")
        if plan_result is not None and plan_result.pregrasp_pose is not None:
            self._release_ramp(self._convert(np.asarray(plan_result.pregrasp_pose, dtype=np.float64)),
                               self._convert(np.asarray(plan_result.grasp_pose, dtype=np.float64)),
                               slow_factor=slow_factor)
            self._last_hand_qpos = np.asarray(plan_result.pregrasp_pose, dtype=np.float64)
        else:
            self._ramp_hand(self._convert(self._hand_init))
            self._last_hand_qpos = self._hand_init.copy()

    def reset(self, plan_result: PlanResult, planner, scene_cfg: dict):
        """Retract to clear-view avoiding the placed object (mirrors real.py
        reset): snapshot the released object pose, re-plan a joint-space retract
        with plan_js_to_init to the clear-view config, and stream it."""
        self._log("reset")
        if not plan_result.success or planner is None:
            self.home(clear_view=True)
            return {"mode": "home_direct", "reason": "no_plan_or_planner"}

        # snapshot released object pose (robot frame) under rigid grasp
        T_obj_grasp = cart2se3(scene_cfg["mesh"]["target"]["pose"])
        T_obj_in_wrist = np.linalg.inv(plan_result.wrist_se3) @ T_obj_grasp
        T_wrist_now = self.arm.get_data()["position"] @ self._link6_to_wrist
        released = T_wrist_now @ T_obj_in_wrist

        new_scene = dict(scene_cfg)
        new_scene["mesh"] = dict(scene_cfg.get("mesh", {}))
        new_scene["mesh"]["target"] = dict(scene_cfg["mesh"]["target"])
        new_scene["mesh"]["target"]["pose"] = se32cart(released).tolist()

        cur_qpos = self.arm.get_data()["qpos"]
        # Start the retract plan from the finger config the hand was actually
        # left in — pregrasp after place(), still pregrasp/grasp after a contact
        # abort. Hard-coding pregrasp made the planner check collisions for a
        # hand shape the robot was not necessarily in.
        start_hand = np.asarray(self._last_hand_qpos, dtype=np.float64)
        print("[franka] planning reset retract ...", flush=True)
        retract = planner.plan_js_to_init(
            new_scene, cur_qpos, start_hand_qpos=start_hand,
            goal_arm_qpos=self._clear_view[:7])
        if retract is None:
            print("[franka] reset re-plan failed — direct clear-view move")
            self.home(clear_view=True)
            return {"mode": "home_direct", "reason": "replan_failed"}
        # Follow the PLANNED hand columns too: plan_js_to_init goes from the
        # hand's current config (pregrasp, where release stopped) to the fully
        # open init config, so the fingers open along the retract on a path the
        # planner checked against the placed object — instead of snapping open
        # in place. _follow ends with a ramp onto the final (fully open) config.
        hand_traj = np.array([self._convert(retract[i, 7:])
                              for i in range(len(retract))])
        self._follow(retract[:, :7], hand_traj)
        self._last_hand_qpos = self._hand_init.copy()
        self._log("reset_done")
        return {"mode": "plan_js_to_init", "n_waypoints": int(len(retract))}

    # ── run_auto compatibility ───────────────────────────────────────────────
    # run_auto drives the xarm through reset -> reset_hybrid -> reset_fallback,
    # each a further fallback when the previous one raises. The FR3 has ONE
    # retract path (plan_js_to_init, with a direct clear-view move built in when
    # the plan fails), so both extra rungs map onto it rather than duplicating a
    # second planner strategy that was never validated on this arm.

    def reset_hybrid(self, plan_result: PlanResult, planner=None,
                     scene_cfg: dict = None) -> dict:
        """xarm's reset_hybrid equivalent — same retract as ``reset``."""
        return self.reset(plan_result, planner, scene_cfg)

    def reset_fallback(self, plan_result: Optional[PlanResult] = None) -> dict:
        """Last-resort recovery: open the hand, then a free-space blocking move
        to clear-view. No planning — this is what runs when execute() itself
        raised, so the arm may be anywhere along the approach."""
        self._log("reset_fallback")
        try:
            self.release(plan_result)
        except Exception as e:
            print(f"[franka] reset_fallback release failed: {e!r}")
        self.home(clear_view=True)
        return {"mode": "reset_fallback"}

    def _move_joints(self, arm_traj, hand_traj=None, **_ignored):
        """xarm RealExecutor's dense-trajectory follower, mapped onto _follow.
        run_auto's reposition path calls this directly."""
        self._follow(np.asarray(arm_traj)[:, :7],
                     None if hand_traj is None else np.asarray(hand_traj))

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
