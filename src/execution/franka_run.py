"""FR3 + inspire runner — now a thin forwarder onto ``run_auto.py --arm franka``.

The two runners used to be separate programs: this file ran a stripped grasp
loop (random candidate, no coverage, no candidate bookkeeping) while
``run_auto.py`` carried the real experiment logic. The only genuine difference
between them was the ARM, so the franka path moved into run_auto behind
``--arm franka`` and this file just forwards to it.

Everything run_auto supports now applies to the FR3 as well: coverage-ranked
candidate selection, tabletop-pose filtering, reorient bookkeeping, plan/
artifacts, candidate result write-back, viz.

    python src/execution/franka_run.py --obj attached_container --auto
    # identical to:
    python src/execution/run_auto.py --obj attached_container --auto \
        --arm franka --hand inspire --grasp_version v8

Defaults injected here (unless you pass them yourself):
    --arm franka   --hand inspire   --grasp_version v8

DROPPED franka-only flags (they had no run_auto equivalent):
    --n_trials       -> --max_trials
    --no_charuco     -> omit --auto for manual labelling
    --no_precheck / --no_video / --lift_height / --watchdog_s -> gone

Prereqs unchanged:
    ~/paradex/cpp/franka_daemon/run_daemon.sh      (franka PC)
    bash scripts/init_daemons.sh start             (capture PCs)
"""
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.execution.run_auto import main as run_auto_main

# (flag, value) defaults that make this entry point mean "the franka one".
_FRANKA_DEFAULTS = [
    ("--arm", "franka"),
    ("--hand", "inspire"),
    ("--grasp_version", "v8"),
]


def main():
    argv = sys.argv[1:]
    for flag, value in _FRANKA_DEFAULTS:
        if flag not in argv:
            argv += [flag, value]
    dropped = [a for a in argv if a.split("=")[0] in (
        "--n_trials", "--no_charuco", "--no_precheck", "--no_video",
        "--lift_height", "--watchdog_s")]
    if dropped:
        raise SystemExit(
            f"franka_run.py no longer accepts {dropped} — it forwards to "
            f"run_auto.py now. Use --max_trials for --n_trials, drop --auto "
            f"for manual labelling; see the module docstring.")
    sys.argv = [sys.argv[0]] + argv
    print(f"[franka_run] forwarding to run_auto: {' '.join(argv)}")
    run_auto_main()


if __name__ == "__main__":
    main()
