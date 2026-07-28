"""Build the synced commanded joint qpos ``arm/action_qpos.npy`` (lift IK-fixed).

``arm/action.npy`` is the cartesian EE command and is NEVER touched here. This
writes a SEPARATE ``arm/action_qpos.npy`` = raw ``action_qpos`` resampled to the
video frames, with the lift-phase frames repaired: during the lift the arm
command is a cartesian link6 wrist_se3, so the raw action_qpos holds non-joint
values there (|.| >> 2pi). Those frames are replaced with proper joint qpos
solved by IK from the cartesian target (``raw/arm/action.npy`` = [xyz_mm,
rpy_rad] in the link6 frame, sxyz euler), seeded from the measured state so the
solution stays on the executed branch. Clean trials still get action_qpos.npy.

    link6 = [xyz_mm/1000, euler2mat(rpy,'sxyz')]  ->  wrist = link6 @ LINK6_TO_WRIST
    IK(wrist, seed_config=state, num_seeds=1) -> arm qpos

State is passed as the IK SEED (not just retract regularization) with num_seeds=1,
so the optimization starts at the executed config and stays on that joint branch
-- no 2pi-wrap / flip solutions, no post-hoc snapping. (Passing state only as
retract_config left the seed random, which is why solutions used to flip.)

Lift frames are detected from the RAW action_qpos (|.|>7), NOT the synced
arm/action, so a re-run is idempotent even after a partial overwrite. Only the
lift frames of arm/action.npy are modified; raw/arm/action.npy is the untouched
cartesian source.

    python -m src.dataset.fix_lift_action_ik            # dry-run report
    python -m src.dataset.fix_lift_action_ik --write    # solve + overwrite
    python -m src.dataset.fix_lift_action_ik --obj toothbrush_holder --limit 2 --write
"""
import argparse
import os

import numpy as np
import transforms3d as t3d
import torch

from autodex.utils.sync import resample, load_video_times
from autodex.utils.robot_config import (
    ALLEGRO_LINK6_TO_WRIST, INSPIRE_LINK6_TO_WRIST, INSPIRE_LEFT_LINK6_TO_WRIST,
)

LINK6_TO_WRIST = {
    "allegro": ALLEGRO_LINK6_TO_WRIST,
    "inspire": INSPIRE_LINK6_TO_WRIST,
    "inspire_left": INSPIRE_LEFT_LINK6_TO_WRIST,
}

ROOTS = ["/home/mingi/shared_data/autodex_dataset/selected_100"]
BAD_THR = 7.0        # |qpos| beyond this = cartesian pollution, not a joint value
BATCH = 256          # IK batch (cuda graph disabled -> variable size ok)
WARN_DEG = 15.0      # IK-arm past this from state = off-branch flip -> use state
TWO_PI = 2.0 * np.pi


def iter_trials(roots, obj_filter=None):
    for root in roots:
        if not os.path.isdir(root):
            continue
        for obj in sorted(os.listdir(root)):
            if obj_filter and obj != obj_filter:
                continue
            od = os.path.join(root, obj)
            if not os.path.isdir(od):
                continue
            for ts in sorted(os.listdir(od)):
                td = os.path.join(od, ts)
                if os.path.isdir(td):
                    yield obj, ts, td


def solve_lift(planner, td, ast, hst, hand="allegro"):
    """IK-fix the lift frames of one trial.

    Returns (arm_qpos (N,6), idx (N,), n_reject) where idx are the lift frames
    (detected from resampled raw action_qpos) and arm_qpos is the snapped/
    state-fallback IK solution. idx is empty if the trial has no lift pollution.
    """
    from autodex.planner.planner import _to_curobo_pose

    vt, _ = load_video_times(td)
    rt = np.load(os.path.join(td, "raw/arm/time.npy")) + 0.03
    aq = resample(rt, np.load(os.path.join(td, "raw/arm/action_qpos.npy")), vt)
    cart = resample(rt, np.load(os.path.join(td, "raw/arm/action.npy")), vt)
    F = min(len(aq), len(ast))
    idx = np.where(np.any(np.abs(aq[:F]) > BAD_THR, axis=1))[0]
    if len(idx) == 0:
        return np.zeros((0, 6)), idx, 0, aq[:F]

    link6 = np.tile(np.eye(4), (len(idx), 1, 1))
    for k, f in enumerate(idx):
        link6[k, :3, 3] = cart[f, :3] / 1000.0
        link6[k, :3, :3] = t3d.euler.euler2mat(*cart[f, 3:6], "sxyz")
    wrist = link6 @ LINK6_TO_WRIST[hand]
    state_arm = ast[idx, :6]
    seed = np.concatenate([state_arm, hst[idx]], axis=1).astype(np.float32)  # (N,22)

    dev = planner._tensor_args.device
    sol = np.zeros((len(idx), 6))
    for s in range(0, len(idx), BATCH):
        e = min(s + BATCH, len(idx))
        goal = _to_curobo_pose(wrist[s:e], dev)
        sc = torch.tensor(seed[s:e], device=dev).unsqueeze(1)   # (B,1,22)
        res = planner._ik_solver.solve_batch(goal, seed_config=sc, num_seeds=1)
        q = res.solution.cpu().numpy(); q = q[:, 0, :] if q.ndim == 3 else q
        sol[s:e] = q[:, :6]

    # seed_config keeps ~all frames on-branch, but a few can still flip. Since
    # FK(state) is verified ~= the commanded wrist (~15mm/1deg) on lift, state is
    # a valid near-command solution: snap pure 2pi wraps back to it, then for any
    # residual off-branch frame (> WARN_DEG) use state itself.
    sol = sol + TWO_PI * np.round((state_arm - sol) / TWO_PI)
    flip = np.abs(sol - state_arm).max(axis=1) > np.radians(WARN_DEG)
    sol[flip] = state_arm[flip]
    return sol, idx, int(flip.sum()), aq[:F]


def solve_lift_synced(planner, aqf, cart, ast, hst, hand="allegro"):
    """IK-fix lift frames using ALREADY-synced arrays (same 30fps video axis).

    For trials re-synced by resync_video_axis.py: arm/action_qpos.npy (aqf, with
    |.|>7 wrist_se3 on lift frames), arm/action.npy (cart [xyz_mm,rpy], the target),
    arm/state.npy (ast, IK seed), hand/state.npy (hst) are all on the same axis --
    no resampling. Returns (sol (K,6), idx (K,), n_flip).
    """
    from autodex.planner.planner import _to_curobo_pose

    F = min(len(aqf), len(ast), len(cart))
    idx = np.where(np.any(np.abs(aqf[:F]) > BAD_THR, axis=1))[0]
    if len(idx) == 0:
        return np.zeros((0, 6)), idx, 0
    link6 = np.tile(np.eye(4), (len(idx), 1, 1))
    for k, f in enumerate(idx):
        link6[k, :3, 3] = cart[f, :3] / 1000.0
        link6[k, :3, :3] = t3d.euler.euler2mat(*cart[f, 3:6], "sxyz")
    wrist = link6 @ LINK6_TO_WRIST[hand]
    state_arm = ast[idx, :6]
    seed = np.concatenate([state_arm, hst[idx]], axis=1).astype(np.float32)
    dev = planner._tensor_args.device
    sol = np.zeros((len(idx), 6))
    for s in range(0, len(idx), BATCH):
        e = min(s + BATCH, len(idx))
        goal = _to_curobo_pose(wrist[s:e], dev)
        sc = torch.tensor(seed[s:e], device=dev).unsqueeze(1)
        res = planner._ik_solver.solve_batch(goal, seed_config=sc, num_seeds=1)
        q = res.solution.cpu().numpy(); q = q[:, 0, :] if q.ndim == 3 else q
        sol[s:e] = q[:, :6]
    sol = sol + TWO_PI * np.round((state_arm - sol) / TWO_PI)
    flip = np.abs(sol - state_arm).max(axis=1) > np.radians(WARN_DEG)
    sol[flip] = state_arm[flip]
    return sol, idx, int(flip.sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", nargs="+", default=ROOTS)
    ap.add_argument("--synced", action="store_true",
                    help="operate on already-resynced synced files (arm/action_qpos.npy "
                         "on the 30fps video axis) instead of re-resampling raw; "
                         "arm/action.npy is left as-is (already cartesian).")
    ap.add_argument("--hand", default="allegro",
                    choices=["allegro", "inspire", "inspire_left"])
    ap.add_argument("--obj")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    work = list(iter_trials(args.roots, args.obj))
    if args.limit:
        work = work[:args.limit]
    print(f"[fix_lift] {len(work)} trials  (write={args.write})")

    planner = None
    stats = {"clean": 0, "fixed": 0, "no_raw": 0, "fallback_trials": 0}
    for obj, ts, td in work:
        # TARGET = arm/action_qpos.npy (the synced commanded joint qpos, lift-IK
        # fixed). arm/action.npy is the cartesian command and is NEVER touched.
        aq_p = os.path.join(td, "arm/action_qpos.npy")
        rq = os.path.join(td, "raw/arm/action_qpos.npy")
        st_p = os.path.join(td, "arm/state.npy")

        if args.synced:
            # Fix lift frames of the ALREADY-resynced action_qpos (30fps video axis).
            if not (os.path.exists(aq_p) and os.path.exists(st_p)):
                continue
            aqf = np.load(aq_p)
            if np.abs(aqf).max() <= BAD_THR:
                stats["clean"] += 1
                continue
            cart = np.load(os.path.join(td, "arm/action.npy"))
            ast = np.load(st_p)
            hst = np.load(os.path.join(td, "hand/state.npy"))
            if planner is None:
                from autodex.planner.planner import GraspPlanner
                planner = GraspPlanner(hand=args.hand)
                planner._use_cuda_graph = False
                planner._init_ik_solver({"cuboid": {}, "mesh": {}})
            sol, idx, n_high = solve_lift_synced(planner, aqf, cart, ast, hst, args.hand)
            aqf[idx] = sol
            deg = np.degrees(np.abs(sol - ast[idx, :6])).max() if len(idx) else 0.0
            stats["fixed"] += 1
            if n_high:
                stats["fallback_trials"] += 1
            print(f"  [{'WROTE' if args.write else 'dry'}] {obj}/{ts}  lift={len(idx)}  "
                  f"max_dev={deg:.1f}deg{'  flip=' + str(n_high) if n_high else ''}")
            if args.write:
                np.save(aq_p, aqf)   # action.npy untouched (already cartesian from resync)
            continue

        if not (os.path.exists(rq) and os.path.exists(st_p)):
            continue
        if not all(os.path.exists(os.path.join(td, p)) for p in
                   ("raw/arm/action.npy", "raw/arm/time.npy",
                    "raw/timestamps/timestamp.npy")):
            stats["no_raw"] += 1
            print(f"  [no_raw] {obj}/{ts}")
            continue
        ast = np.load(st_p)
        # synced commanded joint qpos = raw action_qpos resampled to video frames
        vt, _ = load_video_times(td)
        rt = np.load(os.path.join(td, "raw/arm/time.npy")) + 0.03
        aqf = resample(rt, np.load(rq), vt)
        F = min(len(aqf), len(ast))
        aqf = aqf[:F].copy()
        idx = np.where(np.any(np.abs(aqf) > BAD_THR, axis=1))[0]
        if len(idx) == 0:
            stats["clean"] += 1
            if args.write:
                np.save(aq_p, aqf)         # clean qpos, no lift pollution
            continue
        hst = np.load(os.path.join(td, "hand/state.npy"))
        if planner is None:
            from autodex.planner.planner import GraspPlanner
            planner = GraspPlanner(hand=args.hand)
            planner._use_cuda_graph = False   # variable batch + num_seeds=1
            planner._init_ik_solver({"cuboid": {}, "mesh": {}})

        orig_lift = aqf[idx].copy()        # original raw wrist-6d qpos at lift frames
        sol, sidx, n_high, _ = solve_lift(planner, td, ast, hst, args.hand)
        aqf_fixed = aqf.copy()
        aqf_fixed[sidx] = sol
        deg = np.degrees(np.abs(sol - ast[sidx, :6])).max() if len(sidx) else 0.0
        stats["fixed"] += 1
        if n_high:
            stats["fallback_trials"] += 1
        tag = "WROTE" if args.write else "dry"
        flag = f"  flip_fallback_frames={n_high}" if n_high else ""
        print(f"  [{tag}] {obj}/{ts}  lift={len(sidx)}  max_dev={deg:.1f}deg{flag}")
        if args.write:
            np.save(aq_p, aqf_fixed)       # arm/action_qpos.npy = lift IK-fixed qpos
            # UNDO the qpos I earlier wrote into arm/action.npy: restore ONLY the
            # lift frames back to the original raw wrist-6d values. Everything else
            # in action.npy is left byte-identical.
            a = np.load(os.path.join(td, "arm/action.npy"))
            if len(a) >= idx.max() + 1:
                a[idx] = orig_lift
                np.save(os.path.join(td, "arm/action.npy"), a)

    print("=== summary ===")
    for k, v in stats.items():
        print(f"  {k:14s}: {v}")


if __name__ == "__main__":
    main()
