"""Move the FR3 to its calibrated home (FR3_INIT) — arm only, no hand needed.

Use this to park the arm before bolting the inspire hand on the flange: the
flange (fr3_link8) is then at the exact pose the planner/executor assume, so
the mount orientation can be checked against the URDF convention.

    python src/execution/franka_home.py                 # -> FR3_INIT
    python src/execution/franka_home.py --clear_view    # -> FR3_INIT with j0 -40deg
    python src/execution/franka_home.py --print_only    # read qpos, do not move
    python src/execution/franka_home.py --with_hand     # also open the hand

Prereq: franka daemon up (~/paradex/cpp/franka_daemon/run_daemon.sh),
Desk in Execution mode with joints unlocked + FCI activated.
"""
import argparse
import os
import sys
import time

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np

from autodex.utils.robot_config import FR3_INIT, FR3_INSPIRE_LINK_TO_WRIST

CLEAR_VIEW_J0_DEG = -40.0   # same convention as FrankaExecutor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clear_view", action="store_true",
                    help="go to the clear-view pose (FR3_INIT with j0 -40deg)")
    ap.add_argument("--print_only", action="store_true", help="read state, do not move")
    ap.add_argument("--with_hand", action="store_true",
                    help="also connect the inspire hand and open it (hand must be mounted)")
    ap.add_argument("--hand", default="inspire", choices=["inspire", "inspire_left"])
    ap.add_argument("--speed", type=float, default=0.2, help="blocking-move speed scale")
    args = ap.parse_args()

    target = np.asarray(FR3_INIT, dtype=np.float64).copy()
    if args.clear_view:
        target[0] += np.deg2rad(CLEAR_VIEW_J0_DEG)

    print(f"FR3_INIT  = {np.round(FR3_INIT, 5).tolist()}")
    print(f"target    = {np.round(target, 5).tolist()}"
          f"{'  (clear_view)' if args.clear_view else ''}")
    print("link7->wrist (URDF fixed chain, hand base_link):")
    print(np.round(FR3_INSPIRE_LINK_TO_WRIST, 4))

    from paradex.io.robot_controller import get_arm, get_hand
    arm = get_arm("franka")
    hand = None
    try:
        # A previous run killed mid-motion leaves the daemon's velocity stream
        # active (it holds the robot mutex) — close it before commanding.
        try:
            arm.stop_streaming()
        except Exception as e:
            print(f"[franka] stop_streaming: {e!r}")
        for _ in range(3):
            try:
                if arm.is_error():
                    arm.error_recovery()
            except Exception:
                pass
            time.sleep(0.3)

        data = arm.get_data()
        qpos = None if data is None else data.get("qpos")
        if qpos is None:
            raise RuntimeError(
                "FR3 state is unavailable: the local franka_daemon is not "
                "running or it has no libfranka connection to the robot. Start "
                "~/paradex/cpp/franka_daemon/run_daemon.sh, then verify Franka "
                "Desk is in Execution mode with FCI active.")
        cur = np.asarray(qpos, dtype=np.float64)
        print(f"current   = {np.round(cur, 5).tolist()}")
        print(f"|target - current| = {float(np.linalg.norm(cur - target)):.4f} rad")
        if args.print_only:
            return

        if args.with_hand:
            from autodex.utils.robot_config import INSPIRE_INIT
            from autodex.executor.real import _convert_inspire
            hand = get_hand(args.hand)
            hand.move(_convert_inspire(np.asarray(INSPIRE_INIT, dtype=np.float64)))

        for attempt in range(3):
            try:
                arm.set_collision_behavior([30.0] * 7, [60.0] * 7,
                                           [30.0] * 6, [60.0] * 6)
                break
            except Exception as e:
                print(f"[franka] set_collision_behavior attempt {attempt+1}: {e!r}")
                time.sleep(0.5)

        print("[franka] moving...")
        arm.move(target, is_servo=False, speed_scale=args.speed)
        if arm.is_error():
            arm.error_recovery()
        err = None
        for _ in range(30):
            time.sleep(0.05)
            err = float(np.linalg.norm(arm.get_data()["qpos"] - target))
            if err < 0.1:
                break
        print(f"[franka] done. qpos = {np.round(arm.get_data()['qpos'], 5).tolist()} "
              f"(err={err:.4f})")
    finally:
        # leave the daemon non-streaming so the NEXT process's commands aren't
        # blocked on g_robot_mutex (same teardown as FrankaExecutor.shutdown)
        try:
            arm.stop_streaming()
        except Exception:
            pass
        for ctrl in (arm, hand):
            if ctrl is None:
                continue
            try:
                ctrl.end()
            except Exception:
                pass


if __name__ == "__main__":
    main()
