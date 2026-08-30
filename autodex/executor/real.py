"""
Real-world grasp executor for xArm + Allegro hand.

Autonomous (no GUI) trajectory execution.

Execution sequence (matches RSS2026 reference: planner/inference/train/run_auto_v2.py):
    execute:  init(joint0) -> approach(traj) -> pregrasp -> grasp -> squeeze -> lift -> place
    release:  reverse_squeeze -> grasp -> pregrasp -> hand_init -> arm_return

Usage:
    executor = RealExecutor()
    executor.execute(plan_result)
    executor.release(plan_result)
    executor.shutdown()
"""
import datetime
import os
import sys
import time
from typing import Optional
import numpy as np
from scipy.spatial.transform import Rotation

from autodex.planner import PlanResult
from autodex.utils.robot_config import (
    XARM_INIT, XARM_INSPIRE_INIT,
    ALLEGRO_INIT, ALLEGRO_LINK6_TO_WRIST,
    INSPIRE_INIT, INSPIRE_LINK6_TO_WRIST, INSPIRE_LEFT_LINK6_TO_WRIST,
)

# Per-hand config: (init_joints, link6_to_wrist, convert_fn)
def _convert_allegro(hand_pose: np.ndarray) -> np.ndarray:
    """Reorder Allegro joints: move last 4 (thumb) to front."""
    if hand_pose.ndim == 1:
        out = hand_pose.copy()
        out[:4] = hand_pose[12:]
        out[4:] = hand_pose[:12]
    else:
        out = hand_pose.copy()
        out[:, :4] = hand_pose[:, 12:]
        out[:, 4:] = hand_pose[:, :12]
    return out

def _convert_inspire(hand_pose: np.ndarray) -> np.ndarray:
    """Convert inspire qpos (radians) to controller action (0-1000).

    qpos order:   [thumb_yaw, thumb_pitch, index, middle, ring, pinky]
    action order:  [pinky, ring, middle, index, thumb_pitch, thumb_yaw]
    """
    limits = np.array([1.15, 0.55, 1.6, 1.6, 1.6, 1.6])
    if hand_pose.ndim == 1:
        q = hand_pose[:6]
        normalized = np.clip(q / limits, 0.0, 1.0)
        action_float = (1.0 - normalized) * 1000.0
        action = np.zeros(6, dtype=np.float64)
        action[0] = np.clip(action_float[5], 0, 1000)  # pinky
        action[1] = np.clip(action_float[4], 0, 1000)  # ring
        action[2] = np.clip(action_float[3], 0, 1000)  # middle
        action[3] = np.clip(action_float[2], 0, 1000)  # index
        action[4] = np.clip(action_float[1], 0, 1000)  # thumb_pitch
        action[5] = np.clip(action_float[0], 0, 1000)  # thumb_yaw
    else:
        q = hand_pose[:, :6]
        normalized = np.clip(q / limits, 0.0, 1.0)
        action_float = (1.0 - normalized) * 1000.0
        action = np.zeros_like(hand_pose)
        action[:, 0] = np.clip(action_float[:, 5], 0, 1000)
        action[:, 1] = np.clip(action_float[:, 4], 0, 1000)
        action[:, 2] = np.clip(action_float[:, 3], 0, 1000)
        action[:, 3] = np.clip(action_float[:, 2], 0, 1000)
        action[:, 4] = np.clip(action_float[:, 1], 0, 1000)
        action[:, 5] = np.clip(action_float[:, 0], 0, 1000)
    return action

HAND_CONFIG = {
    "allegro": {
        "init": ALLEGRO_INIT,
        "link6_to_wrist": ALLEGRO_LINK6_TO_WRIST,
        "convert": _convert_allegro,
        "xarm_init": XARM_INIT,
    },
    "inspire": {
        "init": INSPIRE_INIT,
        "link6_to_wrist": INSPIRE_LINK6_TO_WRIST,
        "convert": _convert_inspire,
        "xarm_init": XARM_INSPIRE_INIT,
    },
    "inspire_left": {
        "init": INSPIRE_INIT,
        "link6_to_wrist": INSPIRE_LEFT_LINK6_TO_WRIST,
        "convert": _convert_inspire,
        "xarm_init": XARM_INSPIRE_INIT,
    },
}


# ── Contact monitor (shared by place / execute / reset) ──────────────────────

# Tau conversion constants used by mcc_minimal's stream pipeline. Multiplies
# raw _joints_torque (current in xArm's reported units) to Nm.
KT = np.array([0.067, 0.067, 0.0573, 0.0573, 0.056, 0.056])
GEAR = np.full(6, 100.0)
# Per-joint baseline noise (Nm) — from mcc DEADBAND_J.
DEADBAND_J = np.array([3.0, 3.0, 3.0, 1.0, 2.0, 0.5])


class ContactDetected(RuntimeError):
    """Raised by motion primitives when a ContactMonitor fires.
    Propagates out of execute() / reset() so the caller can abort the trial
    cleanly instead of continuing into pregrasp/grasp at the wrong pose."""
    def __init__(self, where: str, tau_dev, ratio):
        self.where = where
        self.tau_dev = tau_dev
        self.ratio = ratio
        super().__init__(f"contact during {where}: tau_dev={tau_dev.round(2)} "
                         f"ratio={ratio.round(2)}")


class ContactMonitor:
    """Torque-based contact detection using a learned tau_model.

    Usage:
        m = ContactMonitor(xarm_handle, model_path,
                           watch_joints=(1, 2), thresh_nm=10.0,
                           sustained_ticks=8, startup_blank_s=0.5)
        m.warmup(seconds=1.0)              # call when arm is static at start pose
        while moving:
            ...                            # send servo command
            if m.tick():                   # returns True on contact
                break
    """

    def __init__(self, xarm_handle, model_path,
                 watch_joints=(1, 2), thresh_nm=10.0,
                 sustained_ticks: int = 8, startup_blank_s: float = 0.5,
                 dt: float = 0.01,
                 filter_alpha: float = 0.1, qdot_alpha: float = 0.1):
        from pathlib import Path
        import torch  # noqa: F401  (deferred — only loaded if monitor used)
        from autodex.executor.tau_model import load_model
        if model_path is None:
            model_path = str(Path.home() / "shared_data" / "AutoDex"
                             / "weights" / "tau_model" / "inspire_left.pt")
        # Allow tighter / looser thresholds per joint by scaling the
        # deadband. thresh_nm applies to watch_joints (assumed shared scale).
        self._model = load_model(model_path)
        self._xarm = xarm_handle
        self._watch = list(watch_joints)
        # Per-joint threshold: scaled deadband so off-watch joints don't
        # accidentally count if caller widens the watch set later.
        scale_per_joint = np.maximum(DEADBAND_J, 1e-6)
        # On watched joints, set threshold to user-given thresh_nm. Others
        # default to DEADBAND_J * (thresh_nm / DEADBAND_J[watch_joints[0]]).
        ref_db = DEADBAND_J[self._watch[0]]
        self._thresh = scale_per_joint * (thresh_nm / ref_db)
        self._sustained_req = sustained_ticks
        self._blank = startup_blank_s
        self._dt = dt
        self._filter_alpha = filter_alpha
        self._qdot_alpha = qdot_alpha
        self._tau_filt = np.zeros(6)
        self._qdot_smooth = np.zeros(6)
        self._q_last = None
        self._t_last = None
        self._baseline = np.zeros(6)
        self._t0 = None
        self._sustained = 0
        self._last_dev = np.zeros(6)
        self._last_ratio = np.zeros(6)

    def _read(self):
        _, q_deg = self._xarm.get_servo_angle()
        q = np.deg2rad(np.asarray(q_deg[:6], dtype=np.float64))
        I = np.asarray(self._xarm._arm._joints_torque[:6], dtype=np.float64)
        tau = I * KT * GEAR
        return q, tau

    def _predict_tau_ext(self):
        import torch
        from autodex.executor.tau_model import build_input
        q, tau_motor = self._read()
        t_now = time.time()
        if self._q_last is not None and self._t_last is not None:
            dt = max(t_now - self._t_last, 1e-4)
            qdot = (q - self._q_last) / dt
        else:
            qdot = np.zeros(6)
        self._q_last, self._t_last = q.copy(), t_now
        self._qdot_smooth = (self._qdot_alpha * qdot
                             + (1 - self._qdot_alpha) * self._qdot_smooth)
        x = build_input(
            q[None, :], self._qdot_smooth[None, :],
            use_sincos=self._model.use_sincos,
            use_qdot=self._model.use_qdot,
            use_sign_qdot=getattr(self._model, "use_sign_qdot", False),
        )[0].astype(np.float32)
        with torch.no_grad():
            tau_hat = self._model.predict_full(torch.from_numpy(x)).numpy()
        tau_ext = tau_hat - tau_motor
        self._tau_filt = (self._filter_alpha * tau_ext
                          + (1 - self._filter_alpha) * self._tau_filt)
        return self._tau_filt

    def warmup(self, seconds: float = 1.0):
        """Hold the arm static at its current pose and capture baseline."""
        t0 = time.time()
        while time.time() - t0 < seconds:
            self._predict_tau_ext()
            time.sleep(self._dt)
        self._baseline = self._tau_filt.copy()
        self._t0 = time.time()
        self._sustained = 0

    def tick(self) -> bool:
        """Update reading once, return True iff contact sustained over watched
        joints AND the startup blank period has elapsed."""
        if self._t0 is None:
            return False
        self._predict_tau_ext()
        tau_dev = self._tau_filt - self._baseline
        ratio = np.abs(tau_dev) / np.maximum(self._thresh, 1e-6)
        self._last_dev = tau_dev
        self._last_ratio = ratio
        t = time.time() - self._t0
        watched = ratio[self._watch]
        crossed = bool(np.any(watched > 1.0))
        if crossed and t > self._blank:
            self._sustained += 1
        else:
            self._sustained = 0
        return self._sustained >= self._sustained_req

    @property
    def last_dev(self):
        return self._last_dev

    @property
    def last_ratio(self):
        return self._last_ratio


class RealExecutor:
    def __init__(
        self,
        arm_name: str = "xarm",
        hand_name: str = "allegro",
        dt: float = 0.01,
        squeeze_level: int = 2,
    ):
        if hand_name not in HAND_CONFIG:
            raise ValueError(f"Unknown hand: {hand_name}. Choose from {list(HAND_CONFIG)}")
        self.dt = dt
        self.squeeze_level = squeeze_level
        self.hand_name = hand_name

        hcfg = HAND_CONFIG[hand_name]
        self._convert = hcfg["convert"]
        self._hand_init = hcfg["init"]
        self._link6_to_wrist = hcfg["link6_to_wrist"]
        self._xarm_init = hcfg["xarm_init"]

        from paradex.io.robot_controller import get_arm, get_hand
        self.arm = get_arm(arm_name)
        self.hand = get_hand(hand_name)

        # Disable xArm's internal collision sensitivity once for the whole
        # session. Otherwise the firmware can self-stop on small torque
        # spikes (esp. during sequential / approach motions) and silently
        # ignore subsequent servo commands — arm appears to "stall" mid
        # trajectory. Our ContactMonitor handles real collision detection.
        try:
            self.arm.arm.set_report_tau_or_i(1)
            self.arm.arm.set_collision_sensitivity(0)
        except Exception as _e:
            print(f"[executor] could not disable xarm collision sensitivity: {_e!r}")

        # Safety velocity limits
        self.joint_vel_limit = 0.05
        self.cart_vel_limit = 0.002
        self.rot_vel_limit = 0.01
        self.hand_vel_limit = 0.03

    # ── low-level motion primitives ──────────────────────────────────────

    def _safe_joint_step(self, current, target, vel_limit=None):
        delta = target - current
        limit = vel_limit if vel_limit is not None else self.joint_vel_limit
        norm = np.linalg.norm(delta)
        if norm > limit:
            delta = delta / norm * limit
        return current + delta

    def _move_joints(self, arm_traj, hand_traj=None, threshold=0.02,
                     monitor: "Optional[ContactMonitor]" = None):
        for i in range(len(arm_traj)):
            target_arm = arm_traj[i]
            target_hand = hand_traj[i] if hand_traj is not None else None
            if target_hand is not None:
                self.hand.move(target_hand)
            stall_count = 0
            prev_qpos = None
            recovered = False
            for _ in range(500):
                cur = self.arm.get_data()["qpos"]
                if prev_qpos is not None and np.linalg.norm(cur - prev_qpos) < 1e-4:
                    stall_count += 1
                    if stall_count >= 50 and not recovered:
                        print("[executor] stall detected, clearing error...")
                        self.arm.clear_error()
                        recovered = True
                        stall_count = 0
                    elif stall_count >= 100:
                        print("[executor] stall after recovery, aborting")
                        break
                else:
                    stall_count = 0
                prev_qpos = cur.copy()
                nxt = self._safe_joint_step(cur, target_arm)
                self.arm.move(nxt, is_servo=True)
                time.sleep(self.dt)
                if monitor is not None and monitor.tick():
                    raise ContactDetected("_move_joints",
                                          monitor.last_dev, monitor.last_ratio)
                if np.linalg.norm(self.arm.get_data()["qpos"] - target_arm) < threshold:
                    break

    def _move_hand(self, target):
        self.hand.move(target)
        time.sleep(self.dt)

    def _move_cartesian(self, target_pose, threshold_t=0.002, threshold_r=0.02,
                        vel_scale=1.0, stop_on_stall=False,
                        stall_window=30, stall_progress_ratio=0.3,
                        monitor: "Optional[ContactMonitor]" = None):
        """Stall detection is window-based + ratio to commanded velocity:
        over the last `stall_window` ticks the arm should advance roughly
        (cart_vel_limit * vel_scale * stall_window) meters in free motion.
        If actual progress < `stall_progress_ratio` of that expected, we count
        as stalled — robust to both reading latency and xarm yielding on contact.

        stop_on_stall=True breaks immediately on stall (placing mode — stop on
        contact, don't clear_error or retry).
        """
        from collections import deque

        target_rot = Rotation.from_matrix(target_pose[:3, :3])
        pos_history = deque(maxlen=stall_window)
        expected_progress = self.cart_vel_limit * vel_scale * stall_window
        stall_thresh = expected_progress * stall_progress_ratio
        stalled = False
        recovered = False
        recover_count = 0
        for _ in range(500):
            cur = self.arm.get_data()["position"].copy()
            cur_pos = cur[:3, 3].copy()
            pos_history.append(cur_pos)
            # Stall = full window collected and progress < expected*ratio.
            if len(pos_history) == stall_window:
                progress = np.linalg.norm(pos_history[-1] - pos_history[0])
                stalled = (progress < stall_thresh)
            if stalled:
                if stop_on_stall:
                    print(f"[executor] stall detected (window {stall_window} ticks, "
                          f"progress {progress*1000:.2f}mm) — stopping (placing mode)")
                    break
                if not recovered:
                    print("[executor] stall detected, clearing error...")
                    self.arm.clear_error()
                    recovered = True
                    pos_history.clear()
                    stalled = False
                else:
                    recover_count += 1
                    if recover_count >= stall_window:
                        print("[executor] stall after recovery, aborting")
                        break
            prev_pos = cur_pos
            t_delta = target_pose[:3, 3] - cur[:3, 3]
            t_dist = np.linalg.norm(t_delta)
            vel = self.cart_vel_limit * vel_scale
            if t_dist > vel:
                t_delta = t_delta / t_dist * vel
            cur[:3, 3] += t_delta
            cur_rot = Rotation.from_matrix(cur[:3, :3])
            r_delta = (target_rot * cur_rot.inv()).as_rotvec()
            r_dist = np.linalg.norm(r_delta)
            if r_dist > self.rot_vel_limit:
                r_delta = r_delta / r_dist * self.rot_vel_limit
            if r_dist > 0.001:
                cur[:3, :3] = (Rotation.from_rotvec(r_delta) * cur_rot).as_matrix()
            self.arm.move(cur, is_servo=True)
            time.sleep(self.dt)
            if monitor is not None and monitor.tick():
                raise ContactDetected("_move_cartesian",
                                      monitor.last_dev, monitor.last_ratio)
            actual = self.arm.get_data()["position"]
            if (np.linalg.norm(actual[:3, 3] - target_pose[:3, 3]) < threshold_t
                    and np.linalg.norm((target_rot * Rotation.from_matrix(actual[:3, :3]).inv()).as_rotvec()) < threshold_r):
                break

    def _move_joint_sequential(self, target_qpos, joint_order, threshold=0.06,
                               vel_limit: float = 0.06,
                               first_vel_limit: "Optional[float]" = None,
                               monitor: "Optional[ContactMonitor]" = None):
        """``first_vel_limit`` (if set) overrides vel_limit for the FIRST
        joint in joint_order — useful for slowing only the initial motion
        that moves the held object away from a placed scene."""
        current_target = self.arm.get_data()["qpos"].copy()
        for step_i, j in enumerate(joint_order):
            vel = (first_vel_limit if (step_i == 0 and first_vel_limit is not None)
                   else vel_limit)
            current_target[j] = target_qpos[j]
            stall_count = 0
            prev_qpos = None
            recovered = False
            j_start = float(self.arm.get_data()["qpos"][j])
            iter_count = 0
            converged = False
            for _ in range(500):
                iter_count += 1
                cur = self.arm.get_data()["qpos"]
                if prev_qpos is not None and np.linalg.norm(cur - prev_qpos) < 1e-4:
                    stall_count += 1
                    if stall_count >= 50 and not recovered:
                        try:
                            err, warn = self.arm.arm.get_err_warn_code()
                        except Exception as _ee:
                            err, warn = ("?", repr(_ee))
                        print(f"[executor] joint {j} stall at qpos={cur.round(3)} "
                              f"target={current_target.round(3)}  "
                              f"xarm err={err} warn={warn} — clearing...")
                        self.arm.clear_error()
                        recovered = True
                        stall_count = 0
                    elif stall_count >= 100:
                        try:
                            err, warn = self.arm.arm.get_err_warn_code()
                        except Exception as _ee:
                            err, warn = ("?", repr(_ee))
                        print(f"[executor] joint {j} stall after recovery at "
                              f"qpos={cur.round(3)} target={current_target.round(3)}  "
                              f"xarm err={err} warn={warn} — skipping")
                        break
                else:
                    stall_count = 0
                prev_qpos = cur.copy()
                nxt = self._safe_joint_step(cur, current_target, vel_limit=vel)
                self.arm.move(nxt, is_servo=True)
                time.sleep(self.dt)
                if monitor is not None and monitor.tick():
                    raise ContactDetected(f"_move_joint_sequential (joint {j})",
                                          monitor.last_dev, monitor.last_ratio)
                if np.abs(self.arm.get_data()["qpos"][j] - target_qpos[j]) < threshold:
                    converged = True
                    break
            # Post-loop diagnostic: ALWAYS report if joint didn't converge,
            # even when stall_count never reached 50 (slow-motion / partial
            # progress case).
            j_end = float(self.arm.get_data()["qpos"][j])
            j_err = abs(j_end - target_qpos[j])
            if not converged:
                try:
                    err, warn = self.arm.arm.get_err_warn_code()
                except Exception as _ee:
                    err, warn = ("?", repr(_ee))
                print(f"[executor] joint {j} did NOT converge in {iter_count} iters: "
                      f"start={j_start:.3f} end={j_end:.3f} target={target_qpos[j]:.3f}  "
                      f"err={j_err:.3f} rad ({np.degrees(j_err):.1f}°)  "
                      f"xarm err={err} warn={warn}")

    # ── public API ────────────────────────────────────────────────────────

    def start_recording(self, save_dir: str):
        import os
        os.makedirs(save_dir, exist_ok=True)
        self.hand.start(os.path.join(save_dir, "hand"))
        self.arm.start(os.path.join(save_dir, "arm"))

    def stop_recording(self):
        # Idempotent: each controller's .stop() crashes if its save-path attr is
        # None (recording never started / already stopped). Guard each.
        # xarm uses `save_path`; inspire/allegro use `capture_path` — without
        # the second check, hand.stop() never fired and only the last cycle's
        # data persisted (via hand.end() at process shutdown).
        for ctrl in (self.arm, self.hand):
            if (getattr(ctrl, "save_path", None) is not None
                    or getattr(ctrl, "capture_path", None) is not None):
                ctrl.stop()

    def _log_state(self, state):
        ts = datetime.datetime.now().isoformat()
        self.state_timestamps.append({"state": state, "time": ts})

    def _make_monitor(self, thresh_nm: float = 15.0, model_path: str = None,
                      watch_joints=(1, 2),
                      sustained_ticks: int = 100,
                      startup_blank_s: float = 0.5) -> "ContactMonitor":
        """Construct a ContactMonitor for the current arm. Caller should call
        monitor.warmup(...) once the arm is static at the desired baseline pose.
        Defaults: 15 Nm threshold, 1s sustained — robust against motion-induced
        tau spikes; real collisions still fire (ratio >> 1 instantly)."""
        xarm_handle = self.arm.arm   # raw XArmAPI
        # Ensure the report mode is set so _joints_torque is populated.
        try:
            xarm_handle.set_report_tau_or_i(1)
            xarm_handle.set_collision_sensitivity(0)
        except Exception:
            pass
        return ContactMonitor(
            xarm_handle, model_path,
            watch_joints=watch_joints, thresh_nm=thresh_nm,
            sustained_ticks=sustained_ticks,
            startup_blank_s=startup_blank_s, dt=self.dt,
        )

    def execute(self, plan_result: PlanResult, lift_height: float = 0.10,
                skip_lift: bool = False, planner=None,
                scene_cfg=None, debug_dump_dir=None,
                lift_traj_override=None, start_from_current: bool = False):
        """
        Execute: init -> approach -> pregrasp -> grasp -> squeeze -> lift.
        State timestamps stored in self.state_timestamps.
        Returns the squeezed hand pose.

        Place (descend) is now a separate `place(plan_result, ...)` call so
        callers can do work (e.g. capture label image) while the object is
        held up.

        ``skip_lift=True`` stops after the squeeze step (no lift). Use this
        when the caller wants to perform a joint-space lift via the planner
        (avoids ``_move_cartesian`` / ``set_servo_cartesian_aa`` kinematic-
        error spam at extreme wrist orientations).

        ``start_from_current`` preserves the measured raised state for a
        continuous retry whose planner was seeded with those joints.
        """
        if not plan_result.success:
            print("Planning failed — nothing to execute.")
            return None

        self.state_timestamps = []
        traj = plan_result.traj
        pg_hand = self._convert(plan_result.pregrasp_pose)
        g_hand = self._convert(plan_result.grasp_pose)
        wrist_ee = plan_result.wrist_se3 @ np.linalg.inv(self._link6_to_wrist)

        sl = self.squeeze_level

        # 1. Legacy trials return to init first. A continuous retry has a
        # trajectory from the physical raised state, so returning home here
        # would discard the local replanning benefit.
        self._log_state("current_start" if start_from_current else "init")
        if not start_from_current:
            order = [1, 2, 5, 0, 3, 4]
            if self.arm.get_data()["qpos"][1] < self._xarm_init[1]:
                order = [2, 1, 5, 0, 3, 4]
            self._move_joint_sequential(self._xarm_init[:6], order, threshold=0.06)
            init_err = float(np.linalg.norm(self.arm.get_data()["qpos"]
                                            - self._xarm_init[:6]))
            if init_err > 0.1:
                raise RuntimeError(
                    f"execute(): init step finished with err={init_err:.3f} > 0.1 "
                    f"— arm not at XARM_INIT, refusing to approach. "
                    f"final_qpos={self.arm.get_data()['qpos'].round(3)}"
                )
        # Threshold raised 50→70 Nm because inertia spikes (joint 2
        # shoulder ~50-60Nm during free-space motion) were aborting valid
        # approaches.
        monitor = self._make_monitor(thresh_nm=70.0, sustained_ticks=50)
        print("[executor] warming up approach contact monitor (1s static)...")
        monitor.warmup(seconds=1.0)
        print(f"[executor] approach baseline tau = {monitor._baseline.round(2)}  "
              f"(thresh=70Nm, sustained=0.5s)")

        # 2. Approach trajectory (contact-monitored).
        self._log_state("approach")
        hand_traj = np.array([self._convert(traj[i, 6:]) for i in range(len(traj))])
        self._move_joints(traj[:, :6], hand_traj, monitor=monitor)

        # 3. Pregrasp
        self._log_state("pregrasp")
        self._move_hand(pg_hand)

        # 4. Grasp — interpolated ramp pregrasp → grasp for slower close.
        self._log_state("grasp")
        n_grasp_steps = 50
        for i in range(1, n_grasp_steps + 1):
            t = i / n_grasp_steps
            self._move_hand(pg_hand * (1 - t) + g_hand * t)
            time.sleep(0.01)

        # 5. Squeeze (2× slower than before: sleep 0.01 → 0.02)
        self._log_state("squeeze")
        for i in range(sl * 5):
            s_hand = g_hand * (1 + i / 5) - pg_hand * (i / 5)
            self._move_hand(s_hand)
            time.sleep(0.02)

        if skip_lift:
            self._log_state("squeeze_done")
            return s_hand

        # 6. Lift — no contact monitor: arm is now carrying the object so the
        #    empty-arm baseline is invalid for tau_dev. place() does its own
        #    baseline at the lifted pose.
        self._log_state("lift")
        # Straight +z lift. plan_pose_constrained needs WRIST target;
        # _move_cartesian needs LINK6 target. Build both.
        link6_now = self.arm.get_data()["position"].copy()
        link6_lift_pose = link6_now.copy()
        link6_lift_pose[2, 3] += lift_height
        wrist_lift_pose = (link6_now @ self._link6_to_wrist)
        wrist_lift_pose[2, 3] += lift_height
        if lift_traj_override is not None:
            # Use precomputed trajectory (e.g. the one viz showed). Ensures
            # what the user sees in viz == what the robot actually executes.
            print(f"[execute] using precomputed lift_traj "
                  f"shape={lift_traj_override.shape}")
            arm_lift = lift_traj_override[:, :6]
            hand_lift = np.tile(s_hand, (len(lift_traj_override), 1))
            self._move_joints(arm_lift, hand_lift)
        elif planner is not None:
            start_full = np.concatenate([
                np.asarray(self.arm.get_data()["qpos"][:6], dtype=np.float32),
                np.asarray(plan_result.grasp_pose, dtype=np.float32),
            ])
            traj_lift = planner.plan_pose_constrained(
                start_full, wrist_lift_pose,
                hold_vec_weight=[1, 1, 1, 1, 1, 0],
                scene_cfg=scene_cfg,
                include_obj_obstacle=False,
                debug_dump_dir=debug_dump_dir,
            )
            if traj_lift is not None:
                arm_lift = traj_lift[:, :6]
                # Hold hand at the squeeze pose during lift (planner traj's
                # hand portion = grasp_pose which is LESS closed than s_hand
                # and would open fingers mid-lift → drop obj).
                hand_lift = np.tile(s_hand, (len(traj_lift), 1))
                self._move_joints(arm_lift, hand_lift)
            else:
                print("[execute] constrained lift failed, falling back to cartesian")
                self._move_cartesian(link6_lift_pose, vel_scale=1/1.5)
        else:
            self._move_cartesian(link6_lift_pose, vel_scale=1/1.5)

        self._log_state("lift_done")
        return s_hand

    def execute_lift(self, lift_traj, hold_hand):
        """JOINT-SPACE lift: follow a pre-PLANNED qpos trajectory with
        ``_move_joints`` instead of ``_move_cartesian``. The cartesian servo
        (``set_servo_cartesian_aa``) throws kinematic errors at borderline wrist
        configs that are UNRECOVERABLE on xarm (force a full program/robot/camera
        restart). Feeding a planned joint trajectory sidesteps the cartesian IK
        entirely, and any infeasibility is caught at PLAN time (skip) rather than
        crashing the robot mid-execution.

        Args:
            lift_traj: (T, dof) planned qpos trajectory (e.g. from the planner's
                       grasp->lift segment). Only the arm joints (``[:, :6]``) are
                       commanded.
            hold_hand: hand command (controller units, e.g. the squeezed ``s_hand``
                       from ``execute(skip_lift=True)``) held constant throughout.
        """
        self._log_state("lift")
        arm_traj = np.asarray(lift_traj)[:, :6]
        hand_traj = np.tile(np.asarray(hold_hand, dtype=float), (len(arm_traj), 1))
        self._move_joints(arm_traj, hand_traj)        # no monitor: arm carries the object
        self._log_state("lift_done")

    def _place_planned(self, plan_result: PlanResult, planner, scene_cfg,
                       target_descend: float, mcc_model_path: str,
                       debug_dump_dir: str = None) -> "Optional[dict]":
        """Descend by replaying a cuRobo straight-line trajectory. Mirror of lift.

        ``lift`` plans ``wrist z + h`` with ``plan_pose_constrained`` and
        replays it in joint space; the descent is the same motion with the sign
        flipped. Doing it this way instead of streaming Cartesian setpoints
        means the straight line is *checked before the arm moves* (a plan that
        cannot be made returns None here) rather than being left to the
        controller's internal IK, which is free to bend it.

        Contact stop is preserved: ``_move_joints`` ticks the ContactMonitor
        between waypoints and raises ``ContactDetected``, so the arm freezes
        where it touched instead of pushing through.

        Returns the same dict ``place()`` does, or None if no trajectory could
        be planned (caller falls back to the Cartesian admittance descent).
        """
        link6_now = self.arm.get_data()["position"].copy()
        wrist_place_pose = link6_now @ self._link6_to_wrist
        wrist_place_pose[2, 3] -= target_descend
        start_z = float(link6_now[2, 3])

        start_full = np.concatenate([
            np.asarray(self.arm.get_data()["qpos"][:6], dtype=np.float32),
            np.asarray(plan_result.grasp_pose, dtype=np.float32),
        ])
        traj = planner.plan_pose_constrained(
            start_full, wrist_place_pose,
            hold_vec_weight=[1, 1, 1, 1, 1, 0],     # same as lift: hold pose, free z
            scene_cfg=scene_cfg,
            include_obj_obstacle=False,
            debug_dump_dir=debug_dump_dir,
        )
        if traj is None:
            print("[place] constrained descent plan failed — "
                  "falling back to cartesian admittance")
            return None
        print(f"[place] planned straight descent, {len(traj)} waypoints")

        mon = ContactMonitor(self.arm.arm, mcc_model_path,
                             watch_joints=(1, 2), thresh_nm=10.0)
        mon.warmup(seconds=1.0)

        arm_traj = traj[:, :6]

        contact = False
        try:
            # No hand trajectory: the squeeze pose set during execute() stays
            # commanded (the hand controller re-sends its last action at 100 Hz),
            # so the grip is held to the end of the descent. Feeding the
            # planner's hand columns instead would command grasp_pose, which is
            # less closed and would open the fingers mid-descent.
            self._move_joints(arm_traj, None, monitor=mon)
        except ContactDetected as exc:
            contact = True
            print(f"[place] CONTACT — {exc}")

        final_z = float(self.arm.get_data()["position"][2, 3])
        descended = start_z - final_z
        print(f"[place] descended {descended*1000:.1f}mm of target "
              f"{target_descend*1000:.0f}mm  (contact={contact})")
        return {"descended": float(descended),
                "stopped_on_contact": bool(contact),
                "target": float(target_descend),
                "mode": "planned"}

    def place(self, plan_result: PlanResult, lift_height: float = 0.10,
              overshoot: float = 0.0,
              mcc_model_path: str = None,
              descend_time_s: float = 4.0,
              total_time_s: float = 6.4,
              planner=None, scene_cfg=None, debug_dump_dir: str = None,
              log_path: str = None) -> dict:
        """Descend with mcc_minimal admittance control. Target z = lift_pose -
        (lift_height + overshoot) — i.e. with overshoot=0, the arm targets the
        original grasp z (where the object came from). Overshoot can be set >0
        to bias the motion downward past the original z if contact_stop is
        unreliable, but with the tau-model contact check this is usually 0.

        paradex's XArmController control_loop stays alive (so its recording
        keeps running); mcc only computes q_ref and writes to xarm_ctrl.action,
        which paradex sends. On contact (tau_ext > threshold) we freeze and break."""
        from pathlib import Path

        if not plan_result.success:
            return {"descended": 0.0, "stopped_on_contact": False, "target": 0.0}

        if mcc_model_path is None:
            mcc_model_path = str(Path.home() / "shared_data" / "AutoDex"
                                 / "weights" / "tau_model" / "inspire_left.pt")

        self._log_state("place")
        target_descend = lift_height + overshoot

        # Preferred path: plan the straight descent with cuRobo and replay it in
        # joint space — the mirror of lift. Only if that cannot be planned do we
        # fall through to streaming Cartesian setpoints below, the same way lift
        # falls back to _move_cartesian.
        if planner is not None:
            planned = self._place_planned(
                plan_result, planner, scene_cfg, target_descend,
                mcc_model_path, debug_dump_dir=debug_dump_dir)
            if planned is not None:
                return planned

        start_pose = self.arm.get_data()["position"].copy()   # 4x4 homo, link6 in world
        current_pos = start_pose.copy()
        target_pose = start_pose.copy()
        target_pose[2, 3] -= target_descend                   # straight down in world z

        # Adapter: paradex's control thread keeps running (so its recording
        # continues), but mcc's writes are redirected to xarm_ctrl.action and
        # paradex sends them. Reads delegate to the raw XArmAPI handle.
        xarm_ctrl = self.arm
        xarm_handle = xarm_ctrl.arm   # raw XArmAPI

        # mcc-needed handle setup (one-shot, harmless to paradex).
        xarm_handle.set_report_tau_or_i(1)
        xarm_handle.set_collision_sensitivity(0)

        # Tau model (copied from mcc_minimal/fit_tau_model.py; self-contained,
        # only depends on numpy + torch).
        import torch  # noqa: E402
        from autodex.executor.tau_model import load_model, build_input

        print(f"[place] loading mcc model: {mcc_model_path}")
        model = load_model(mcc_model_path)

        # Contact-stop loop: use the learned tau model only to estimate tau_ext;
        # on contact (tau_ext > threshold), freeze q_des at current pose (paradex
        # holds it) and break. No yield, no bounce.
        DT = 0.01
        FILTER_ALPHA = 0.1
        QDOT_SMOOTH_ALPHA = 0.1
        WARMUP_SEC = 1.0
        # baseline noise per joint (Nm) — from mcc DEADBAND_J. Kept for ref;
        # current contact check uses a flat 20 Nm threshold from lift baseline.
        DEADBAND_J = np.array([3.0, 3.0, 3.0, 1.0, 2.0, 0.5])
        CONTACT_THRESH = np.full(6, 10.0)
        # Same constants mcc uses to convert _joints_torque (raw current in
        # whatever units xarm reports) to Nm. tau_motor = I * KT * GEAR.
        KT = np.array([0.067, 0.067, 0.0573, 0.0573, 0.056, 0.056])
        GEAR = np.full(6, 100.0)

        def _read():
            _, q_deg = xarm_handle.get_servo_angle()
            q = np.deg2rad(np.asarray(q_deg[:6], dtype=np.float64))
            # tau_motor: raw _joints_torque × KT × GEAR → Nm. Matches what the
            # MLP was trained against in mcc's stream source path.
            I = np.asarray(xarm_handle._arm._joints_torque[:6], dtype=np.float64)
            tau = I * KT * GEAR
            return q, tau

        def _push_pose(pose4x4):
            """Push 4x4 link6 pose. paradex's control_loop sees non-(6,) shape
            and routes through set_servo_cartesian_aa — internal IK tracks
            current pose continuously, no arbitrary elbow flip."""
            with xarm_ctrl.lock:
                xarm_ctrl.action = pose4x4.astype(np.float64)
                xarm_ctrl.is_servo = True

        # Hold start_pose during warmup.
        _push_pose(start_pose)

        # Warmup: prime tau_filt at hold pose.
        tau_filt = np.zeros(6)
        qdot_smooth = np.zeros(6)
        q_last, t_last = None, None
        t_warm0 = time.time()
        while time.time() - t_warm0 < WARMUP_SEC:
            q, tau_motor = _read()
            t_now = time.time()
            if q_last is not None and t_last is not None:
                dt = max(t_now - t_last, 1e-4)
                qdot = (q - q_last) / dt
            else:
                qdot = np.zeros(6)
            q_last, t_last = q.copy(), t_now
            qdot_smooth = QDOT_SMOOTH_ALPHA * qdot + (1 - QDOT_SMOOTH_ALPHA) * qdot_smooth
            x = build_input(q[None, :], qdot_smooth[None, :],
                            use_sincos=model.use_sincos,
                            use_qdot=model.use_qdot,
                            use_sign_qdot=getattr(model, "use_sign_qdot", False))[0].astype(np.float32)
            with torch.no_grad():
                tau_hat = model.predict_full(torch.from_numpy(x)).numpy()
            tau_ext = tau_hat - tau_motor
            tau_filt = FILTER_ALPHA * tau_ext + (1 - FILTER_ALPHA) * tau_filt
            _push_pose(start_pose)
            time.sleep(DT)
        # Pose-dependent baseline (MLP residual + any reading offset). Subtract
        # this from later tau_filt so the contact check only sees DEVIATIONS.
        tau_baseline = tau_filt.copy()
        print(f"[place] warmup done. baseline tau_filt = {tau_baseline.round(2)}  (subtracted from now on)")
        print(f"[place] contact threshold per joint = {CONTACT_THRESH.round(2)}")

        # Descend loop with contact stop.
        log = [] if log_path else None
        contact = False
        contact_t = None
        sustained = 0
        last_print_t = -1.0
        t0 = time.time()
        next_t = t0
        while time.time() - t0 < total_time_s:
            now = time.time()
            if now < next_t:
                time.sleep(max(0.0, next_t - now))
            next_t += DT
            t = time.time() - t0

            q, tau_motor = _read()
            t_now = time.time()
            dt = max(t_now - t_last, 1e-4)
            qdot = (q - q_last) / dt
            q_last, t_last = q.copy(), t_now
            qdot_smooth = QDOT_SMOOTH_ALPHA * qdot + (1 - QDOT_SMOOTH_ALPHA) * qdot_smooth

            x = build_input(q[None, :], qdot_smooth[None, :],
                            use_sincos=model.use_sincos,
                            use_qdot=model.use_qdot,
                            use_sign_qdot=getattr(model, "use_sign_qdot", False))[0].astype(np.float32)
            with torch.no_grad():
                tau_hat = model.predict_full(torch.from_numpy(x)).numpy()
            tau_ext = tau_hat - tau_motor
            tau_filt = FILTER_ALPHA * tau_ext + (1 - FILTER_ALPHA) * tau_filt

            # Contact check: only on joints 2 and 3 (shoulder/elbow — most
            # informative for downward contact). Skip first STARTUP_BLANK_S to
            # avoid warmup->descend dynamics spike. Require SUSTAINED_TICKS
            # consecutive ticks above threshold so single-tick noise doesn't fire.
            STARTUP_BLANK_S = 0.5
            SUSTAINED_TICKS = 8
            CONTACT_JOINTS = (1, 2)   # 0-indexed: joints 2 and 3
            tau_dev = tau_filt - tau_baseline
            ratio = np.abs(tau_dev) / np.maximum(CONTACT_THRESH, 1e-6)
            ratio_watch = ratio[list(CONTACT_JOINTS)]
            crossed = bool(np.any(ratio_watch > 1.0))
            if crossed and t > STARTUP_BLANK_S:
                sustained += 1
            else:
                sustained = 0
            # Periodic dump every 0.2s so torque evolution is visible. One
            # self-overwriting line: at 5 dumps/s a 6.4s descent is 30+ lines
            # of scrollback that bury the result of the trial.
            if t - last_print_t >= 0.2:
                last_print_t = t
                line = (f"[place] t={t:5.2f}s  tau_dev={tau_dev.round(2)}  "
                        f"ratio={ratio.round(2)}")
                sys.stdout.write("\r" + line.ljust(110))
                sys.stdout.flush()

            if (not contact) and (sustained >= SUSTAINED_TICKS):
                contact = True
                contact_t = t
                sys.stdout.write("\n")
                print(f"[place] CONTACT at t={t:.2f}s (watching joints {[i+1 for i in CONTACT_JOINTS]})")
                print(f"  tau_dev = {tau_dev.round(2)}")
                print(f"  ratio   = {ratio.round(2)}")
                print(f"  thresh  = {CONTACT_THRESH.round(2)}")
                # Freeze at current actual pose; break.
                cur_pose = self.arm.get_data()["position"].copy()
                _push_pose(cur_pose)
                if log is not None:
                    log.append((t, *q, *tau_dev, 1))
                break

            # Cartesian lerp: translate z toward target, rotation held constant.
            alpha = min(1.0, max(0.0, t / descend_time_s))
            pose_des = start_pose.copy()
            pose_des[2, 3] = (1 - alpha) * start_pose[2, 3] + alpha * target_pose[2, 3]
            _push_pose(pose_des)

            if log is not None:
                log.append((t, *q, *tau_dev, 0))

        if not contact:
            sys.stdout.write("\n")
            print(f"[place] no contact within {total_time_s}s — reached target. final pose held.")

        # Optional CSV log.
        if log_path and log:
            import csv as _csv
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "w", newline="") as f:
                w = _csv.writer(f)
                w.writerow(["t"] + [f"q{i}" for i in range(6)] +
                           [f"tau_ext{i}" for i in range(6)] + ["contact"])
                w.writerows(log)
            print(f"[place] log -> {log_path}")

        # Read final pose for descended-distance reporting.
        try:
            _, final_pos_xarm = xarm_handle.get_position(is_radian=True)
        except Exception:
            final_pos_xarm = None

        # Compute descended distance from final pose.
        if final_pos_xarm is not None:
            final_z = final_pos_xarm[2] / 1000.0  # mm -> m
            descended = current_pos[2, 3] - final_z
        else:
            descended = float("nan")
        print(f"[place] descended {descended*1000:.1f}mm of target {target_descend*1000:.0f}mm "
              f"({'contact stop' if contact else 'reached target window'})")

        # paradex thread never stopped; it's been forwarding mcc's q_ref the
        # whole time, so no re-init needed. Just leave its action at last q_ref.

        self._log_state("place_done")
        return {"descended": float(descended), "stopped_on_contact": bool(contact),
                "contact_t_s": float(contact_t) if contact_t is not None else None,
                "target": float(target_descend)}

    def release(self, plan_result: PlanResult, slow_factor: float = 1.0):
        """Release object and return arm to init pose.

        slow_factor > 1.0 stretches the open ramp (e.g. 4.0 → 2s instead of 0.5s).
        """
        if not plan_result.success:
            return

        pg_hand = self._convert(plan_result.pregrasp_pose)
        g_hand = self._convert(plan_result.grasp_pose)
        self._release_auto(pg_hand, g_hand, slow_factor=slow_factor)

    def _release_auto(self, pg_hand, g_hand, slow_factor: float = 1.0):
        """Reverse squeeze -> grasp -> pregrasp, then STOP.
        Hand opening to hand_init and arm retract back to init are intentionally
        skipped — user resets those manually after inspecting the placed object."""
        sl = self.squeeze_level

        # Reverse squeeze (matches squeeze ramp speed: 0.02s per step).
        for i in range(sl * 5):
            s_hand = g_hand * (sl - i / 5) - pg_hand * (sl - 1 - i / 5)
            self._move_hand(s_hand)
            time.sleep(0.02 * slow_factor)

        # Interpolated open ramp grasp → pregrasp (mirrors close ramp).
        n_open_steps = 50
        for i in range(1, n_open_steps + 1):
            t = i / n_open_steps
            self._move_hand(g_hand * (1 - t) + pg_hand * t)
            time.sleep(0.01 * slow_factor)

    def reset(self, plan_result: PlanResult,
              planner, scene_cfg: dict) -> dict:
        """Automated reset AFTER release. Required: planner + scene_cfg.
          0. Snapshot placed object pose from CURRENT wrist (rigid-grasp).
             place() may stop early on contact, so the actual resting pose
             can differ from the planned grasp pose.
          1. Re-plan retract with the placed object as obstacle. Start hand
             config = pregrasp (real hand state after release). Goal = init
             state. Planner gradually opens fingers along a collision-free
             path away from the object. Raises if planning fails.
          2. Execute traj (arm + planner-generated hand portion).
          3. Final joint-0 unwind to land exactly on XARM_INIT."""
        t_start = time.time()
        log = {"start": datetime.datetime.now().isoformat(), "steps": {}}
        if not plan_result.success:
            log["skipped"] = True
            return log

        from autodex.utils.conversion import cart2se3, se32cart

        # 0. Snapshot released object pose (robot frame) under rigid-grasp.
        T_obj_grasp = cart2se3(scene_cfg["mesh"]["target"]["pose"])
        T_obj_in_wrist = np.linalg.inv(plan_result.wrist_se3) @ T_obj_grasp
        T_wrist_now = self.arm.get_data()["position"] @ self._link6_to_wrist
        released_obj_pose = T_wrist_now @ T_obj_in_wrist
        log["T_wrist_grasp"] = plan_result.wrist_se3.tolist()
        log["T_wrist_at_reset_start"] = T_wrist_now.tolist()
        log["T_obj_in_wrist"] = T_obj_in_wrist.tolist()
        log["released_obj_pose_robot"] = released_obj_pose.tolist()

        # 1. Open hand pregrasp → openpose first (mirrors reset_hybrid step 0).
        #    With fingers open, plan_js_to_init's trajopt has more clearance
        #    around the placed obj, reducing TRAJOPT_FAIL rate.
        op_raw = getattr(plan_result, "openpose_pose", None)
        pg_raw = getattr(plan_result, "pregrasp_pose", None)
        self._log_state("hand_open")
        if op_raw is not None and pg_raw is not None:
            pg = np.asarray(pg_raw, dtype=np.float64)
            op = np.asarray(op_raw, dtype=np.float64)
            for i in range(11):
                a = i / 10.0
                qpos = (1.0 - a) * pg + a * op
                self._move_hand(self._convert(qpos))
                time.sleep(0.05)
            hold_hand_raw = op
        else:
            hold_hand_raw = (np.asarray(plan_result.pregrasp_pose, dtype=np.float64)
                              if plan_result.pregrasp_pose is not None
                              else self._hand_init)

        # 2. Re-plan retract — hand at openpose (clearer from obj).
        self._log_state("arm_retract")
        t1 = time.time()
        new_scene = dict(scene_cfg)
        new_scene["mesh"] = dict(scene_cfg.get("mesh", {}))
        new_scene["mesh"]["target"] = dict(scene_cfg["mesh"]["target"])
        new_scene["mesh"]["target"]["pose"] = se32cart(released_obj_pose).tolist()
        cur_qpos = self.arm.get_data()["qpos"]
        clear_view_arm = self._xarm_init.copy()
        clear_view_arm[0] -= np.deg2rad(40.0)
        t_plan0 = time.time()
        retract_traj = planner.plan_js_to_init(
            new_scene, cur_qpos,
            start_hand_qpos=hold_hand_raw,
            goal_arm_qpos=clear_view_arm[:6],
        )
        log["steps"]["replan_s"] = round(time.time() - t_plan0, 2)
        if retract_traj is None:
            log["retract_mode"] = "replan_failed"
            raise RuntimeError(
                "reset(): plan_js_to_init returned None — retract not safe. "
                "Inspect placed object pose / scene_cfg."
            )
        log["retract_mode"] = "replanned"

        # 2. Execute: arm + planner-generated hand portion. No contact monitor
        #    here — retract trajectory is high-acceleration and tau_model's
        #    qdot extrapolation outside training distribution gives false
        #    positives that cut the traj short mid-flight.
        arm_traj = retract_traj[:, :6]
        hand_traj = np.array([self._convert(retract_traj[i, 6:])
                              for i in range(len(retract_traj))])
        self._move_joints(arm_traj, hand_traj)
        log["steps"]["arm_retract_s"] = round(time.time() - t1, 2)

        # 3. Verify final pose — RAISE if arm didn't actually reach
        #    clear_view (stall, partial traj, etc.) so the caller doesn't
        #    silently start the next cycle from a bad pose.
        final_qpos = self.arm.get_data()["qpos"]
        err = float(np.linalg.norm(final_qpos - clear_view_arm[:6]))
        log["final_qpos_err"] = err
        self._log_state("reset_done")
        log["total_s"] = round(time.time() - t_start, 2)
        if err > 0.1:
            raise RuntimeError(
                f"reset(): retract finished with final_qpos_err={err:.3f} > 0.1. "
                f"final_qpos={final_qpos.round(3)}  clear_view={clear_view_arm[:6].round(3)}"
            )
        return log

    def reset_hybrid(self, plan_result: PlanResult,
                      planner, scene_cfg: dict) -> dict:
        """Hybrid retract: sequential [1, 2, 0] first (moves arm base/shoulder/
        elbow away from the placed object — these are large, safe motions),
        then cuRobo plans the remaining wrist joints (3, 4, 5) to clear_view
        with collision-free self/world check.

        Steps:
          0. Open hand to hand_init.
          1. Snapshot placed object pose (from current wrist + rigid grasp).
          2. Sequential move on joints [1, 2, 0] to their clear_view values.
          3. cuRobo plan_js_to_init from current full qpos → clear_view arm
             goal. Hand stays at hand_init throughout. Collision world is
             scene_cfg with the placed object's snapshot pose as obstacle.
          4. Execute the planned trajectory.
        """
        t_start = time.time()
        log = {"start": datetime.datetime.now().isoformat(), "steps": {}}
        if not plan_result.success:
            log["skipped"] = True
            return log

        from autodex.utils.conversion import cart2se3, se32cart

        # 0. Slowly open fingers pregrasp → openpose (10 linear steps). Keep
        #    openpose for the rest of retract (skip hand_init). If openpose
        #    not given, fall back to single jump to hand_init (legacy path).
        op_raw = getattr(plan_result, "openpose_pose", None)
        pg_raw = getattr(plan_result, "pregrasp_pose", None)
        self._log_state("hand_open")
        t1 = time.time()
        if op_raw is not None and pg_raw is not None:
            pg = np.asarray(pg_raw, dtype=np.float64)
            op = np.asarray(op_raw, dtype=np.float64)
            for i in range(11):
                a = i / 10.0
                qpos = (1.0 - a) * pg + a * op
                self._move_hand(self._convert(qpos))
                time.sleep(0.05)
            hold_hand_raw = op
        else:
            hold_hand_raw = self._hand_init
            self._move_hand(self._convert(hold_hand_raw))
            time.sleep(0.3)
        log["steps"]["hand_open_s"] = round(time.time() - t1, 2)

        # 1. Snapshot placed object pose under rigid grasp assumption.
        T_obj_grasp = cart2se3(scene_cfg["mesh"]["target"]["pose"])
        T_obj_in_wrist = np.linalg.inv(plan_result.wrist_se3) @ T_obj_grasp
        T_wrist_now = self.arm.get_data()["position"] @ self._link6_to_wrist
        released_obj_pose = T_wrist_now @ T_obj_in_wrist
        log["released_obj_pose_robot"] = released_obj_pose.tolist()

        # 2. Sequential on joints [1, 2, 0] only — coarse arm motion away from
        #    the just-placed object, before wrist replanning.
        clear_view = self._xarm_init.copy()
        clear_view[0] -= np.deg2rad(40.0)
        self._log_state("seq_base_shoulder")
        t1 = time.time()
        coarse_order = [1, 2, 0]
        if self.arm.get_data()["qpos"][1] < self._xarm_init[1]:
            coarse_order = [2, 1, 0]
        self._move_joint_sequential(clear_view[:6], coarse_order,
                                     threshold=0.06,
                                     first_vel_limit=0.02)
        log["steps"]["seq_arm_s"] = round(time.time() - t1, 2)

        # 3. cuRobo plan_js for remaining wrist joints (3, 4, 5). plan_js_to_init
        #    plans the full 22-DOF trajectory but joints 0/1/2 are already at
        #    clear_view so only 3/4/5 actually change. Self/world collision
        #    handled by cuRobo trajopt.
        self._log_state("plan_wrist")
        t1 = time.time()
        new_scene = dict(scene_cfg)
        new_scene["mesh"] = dict(scene_cfg.get("mesh", {}))
        new_scene["mesh"]["target"] = dict(scene_cfg["mesh"]["target"])
        new_scene["mesh"]["target"]["pose"] = se32cart(released_obj_pose).tolist()
        cur_qpos = self.arm.get_data()["qpos"]
        wrist_traj = planner.plan_js_to_init(
            new_scene, cur_qpos,
            start_hand_qpos=hold_hand_raw,
            goal_arm_qpos=clear_view[:6],
        )
        log["steps"]["wrist_plan_s"] = round(time.time() - t1, 2)
        if wrist_traj is None:
            log["wrist_plan_mode"] = "failed"
            log["retract_mode"] = "hybrid_wrist_failed"
            self._log_state("reset_done")
            log["total_s"] = round(time.time() - t_start, 2)
            raise RuntimeError(
                "reset_hybrid(): plan_js_to_init returned None — wrist retract "
                "not safe. Inspect placed object pose / scene_cfg."
            )
        log["wrist_plan_mode"] = "planned"
        t1 = time.time()
        arm_traj = wrist_traj[:, :6]
        hand_traj = np.array([self._convert(wrist_traj[i, 6:])
                              for i in range(len(wrist_traj))])
        self._move_joints(arm_traj, hand_traj)
        log["steps"]["wrist_exec_s"] = round(time.time() - t1, 2)

        final_qpos = self.arm.get_data()["qpos"]
        err = float(np.linalg.norm(final_qpos - clear_view[:6]))
        log["final_qpos_err"] = err
        log["retract_mode"] = "hybrid"
        self._log_state("reset_done")
        log["total_s"] = round(time.time() - t_start, 2)
        if err > 0.1:
            print(f"[reset_hybrid] WARNING: final_qpos_err={err:.3f} > 0.1  "
                  f"final={final_qpos.round(3)}  target={clear_view[:6].round(3)}")
        return log

    def reset_fallback(self, plan_result: PlanResult) -> dict:
        """Reset path for failed grasps (approach contact or charuco fail).
        Open hand to hand_init, then sequentially move arm to clear_view
        (joint 0 -60° from XARM_INIT) via [1, 2, 5, 0, 3, 4] (mirror if
        joint 1 below init). No planner involvement."""
        t_start = time.time()
        log = {"start": datetime.datetime.now().isoformat(), "steps": {}}
        if not plan_result.success:
            log["skipped"] = True
            return log

        init_hand = self._convert(self._hand_init)

        # 1. Open hand to hand_init.
        self._log_state("hand_init")
        t1 = time.time()
        self._move_hand(init_hand)
        time.sleep(0.5)
        log["steps"]["hand_open_s"] = round(time.time() - t1, 2)

        # 2. Sequential arm retract to clear-view pose (joint 0 -60° from
        #    XARM_INIT). No contact monitor — sequential motion has high
        #    per-joint acceleration that breaks the tau_model baseline.
        self._log_state("clear_view")
        t1 = time.time()
        clear_view = self._xarm_init.copy()
        clear_view[0] -= np.deg2rad(40.0)
        execute_order = [1, 2, 5, 0, 3, 4]
        if self.arm.get_data()["qpos"][1] < self._xarm_init[1]:
            execute_order = [2, 1, 5, 0, 3, 4]
        self._move_joint_sequential(clear_view[:6], execute_order,
                                     threshold=0.06,
                                     first_vel_limit=0.02)
        log["steps"]["arm_retract_s"] = round(time.time() - t1, 2)

        final_qpos = self.arm.get_data()["qpos"]
        err = float(np.linalg.norm(final_qpos - clear_view[:6]))
        log["final_qpos_err"] = err
        log["retract_mode"] = "fallback_sequential"
        self._log_state("reset_done")
        log["total_s"] = round(time.time() - t_start, 2)
        if err > 0.1:
            # Don't abort — fixed-trajectory sequential retract. Caller decides
            # what to do with a partial result via log["final_qpos_err"].
            print(f"[reset_fallback] WARNING: final_qpos_err={err:.3f} > 0.1  "
                  f"final={final_qpos.round(3)}  target={clear_view[:6].round(3)}")
        return log

    def shutdown(self):
        self.arm.end()
        self.hand.end()
