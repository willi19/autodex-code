"""Parse a recorded execution trajectory into motion stages.

The legacy captures store one continuous servo stream (raw/arm, raw/hand) with
no stage boundaries. This segments that stream into stages by *motion state*:
whether the arm end-effector is moving and whether the hand is moving at each
timestep. Stages run in a fixed order, so contiguous motion-state blocks are
labelled by best-effort ordering.

Streams (each with its own ``time.npy``, ~100 Hz):
    raw/arm/action.npy      (N,6)  commanded EE cart [x,y,z mm, r,p,y rad]
    raw/arm/action_qpos.npy (N,6)  commanded joint qpos
    raw/hand/action.npy     (M,16) commanded hand joints

Motion-state legend per merged segment:
    A- : arm moving, hand still      (init clear / lift / place / return)
    AH : arm + hand moving           (approach)
    -H : hand moving, arm still      (pregrasp / grasp / squeeze / release / reset)
    -- : both still                  (hold / settle)

NOTE: the executor version behind a given capture is not recorded, so exact
stage names (esp. within hand-only and arm-only blocks) are best-effort. The
segments + per-segment features are the reliable output for downstream
outlier/characteristic analysis.

Usage:
    python -m src.dataset.parse_stages --trial <trial_dir> [--json out.json]
    python -m src.dataset.parse_stages --obj apple            # first trial of obj
"""

import argparse
import json
import os

import numpy as np

DATASET_ROOT = "/home/mingi/shared_data/autodex_dataset/selected_100"
SRC_ROOT = "/home/mingi/shared_data/RSS2026_Mingi/experiment/selected_100"

# Motion thresholds (displacement over WINDOW samples ~ 100 ms).
WINDOW = 10
EE_MOVE_MM = 3.0        # EE translation speed to count as "arm moving"
HAND_MOVE_RAD = 0.05    # hand joint speed to count as "hand moving"
MIN_SEG = 20            # merge segments shorter than this (samples) into neighbour
GAP_CLOSE = 15          # bridge motion gaps shorter than this (samples)


def _load_streams(trial_dir):
    a = os.path.join(trial_dir, "raw", "arm")
    h = os.path.join(trial_dir, "raw", "hand")
    arm = {k: np.load(os.path.join(a, k + ".npy")) for k in
           ("time", "action", "action_qpos", "position")}
    hand = {k: np.load(os.path.join(h, k + ".npy")) for k in
            ("time", "action", "position")}
    return arm, hand


def _resample(t_src, v, t_dst):
    """Nearest-past sample of v (indexed by t_src) at each t_dst."""
    idx = np.searchsorted(t_src, t_dst).clip(0, len(t_src) - 1)
    return v[idx]


def _speed(x, w=WINDOW):
    """Displacement magnitude over a trailing window of w samples."""
    n = len(x)
    out = np.zeros(n)
    for i in range(n):
        out[i] = np.linalg.norm(x[i] - x[max(0, i - w)])
    return out


def _smooth_bool(mask, gap_close=GAP_CLOSE, min_seg=MIN_SEG):
    """Close short gaps then drop short runs, so tiny fragments merge away."""
    m = mask.copy()
    # close short False gaps between True runs
    i = 0
    while i < len(m):
        if not m[i]:
            j = i
            while j < len(m) and not m[j]:
                j += 1
            if 0 < i and j < len(m) and (j - i) < gap_close:
                m[i:j] = True
            i = j
        else:
            i += 1
    # drop short True runs
    i = 0
    while i < len(m):
        if m[i]:
            j = i
            while j < len(m) and m[j]:
                j += 1
            if (j - i) < min_seg:
                m[i:j] = False
            i = j
        else:
            i += 1
    return m


def parse_stages(trial_dir):
    arm, hand = _load_streams(trial_dir)
    ta = arm["time"]
    ee = arm["action"][:, :3]           # EE xyz (mm)
    qpos = arm["action_qpos"]
    hand_a = _resample(hand["time"], hand["action"], ta)  # onto arm timebase

    arm_moving = _smooth_bool(_speed(ee) > EE_MOVE_MM)
    hand_moving = _smooth_bool(_speed(hand_a) > HAND_MOVE_RAD)

    # State per sample, then run-length encode into merged segments.
    state = np.where(arm_moving, np.where(hand_moving, 3, 2),
                     np.where(hand_moving, 1, 0))  # 0=--,1=-H,2=A-,3=AH
    names = {0: "--", 1: "-H", 2: "A-", 3: "AH"}

    segs = []
    s = 0
    for i in range(1, len(state) + 1):
        if i == len(state) or state[i] != state[s]:
            i0, i1 = s, i - 1
            segs.append({
                "state": names[int(state[s])],
                "i0": int(i0), "i1": int(i1), "n": int(i1 - i0 + 1),
                "t0": float(ta[i0] - ta[0]), "t1": float(ta[i1] - ta[0]),
                "ee_dxyz_mm": [round(float(x), 1) for x in (ee[i1] - ee[i0])],
                "ee_dz_mm": round(float(ee[i1, 2] - ee[i0, 2]), 1),
                "ee_path_mm": round(float(np.abs(np.diff(ee[i0:i1 + 1], axis=0)).sum()), 1),
                "qpos_dmax": round(float(np.abs(qpos[i1] - qpos[i0]).max()), 3),
                "hand_dmax": round(float(np.abs(hand_a[i1] - hand_a[i0]).max()), 3),
            })
            s = i

    # Best-effort ordered labels over the moving segments only.
    _label(segs)
    return {
        "trial": trial_dir,
        "n_arm": int(len(ta)), "n_hand": int(len(hand["time"])),
        "duration_s": round(float(ta[-1] - ta[0]), 2),
        "segments": segs,
    }


# Fixed nominal stage order for moving blocks (hold segments stay unlabeled).
_ARM_ONLY_ORDER = ["init_clear", "lift", "place", "return"]
_HAND_ONLY_ORDER = ["close", "open"]  # close=pregrasp+grasp+squeeze, open=release+reset


def _label(segs):
    arm_i = hand_i = 0
    for seg in segs:
        st = seg["state"]
        if st == "AH":
            seg["stage"] = "approach"
        elif st == "A-":
            seg["stage"] = _ARM_ONLY_ORDER[min(arm_i, len(_ARM_ONLY_ORDER) - 1)]
            arm_i += 1
        elif st == "-H":
            seg["stage"] = _HAND_ONLY_ORDER[min(hand_i, len(_HAND_ONLY_ORDER) - 1)]
            hand_i += 1
        else:
            seg["stage"] = "hold"


def _resolve_trial(args):
    if args.trial:
        return args.trial
    root = args.root
    obj_dir = os.path.join(root, args.obj)
    ts = args.ts or sorted(d for d in os.listdir(obj_dir)
                           if os.path.isdir(os.path.join(obj_dir, d)))[0]
    return os.path.join(obj_dir, ts)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trial", help="explicit trial dir (has raw/arm, raw/hand)")
    ap.add_argument("--obj", help="object name (uses first/ --ts trial under --root)")
    ap.add_argument("--ts", help="timestamp under --obj")
    ap.add_argument("--root", default=SRC_ROOT, help="root holding {obj}/{ts}/raw")
    ap.add_argument("--json", help="write full result to this path")
    ap.add_argument("--all", action="store_true", help="only print merged moving segments")
    args = ap.parse_args()

    trial = _resolve_trial(args)
    res = parse_stages(trial)
    print(f"trial: {res['trial']}")
    print(f"arm N={res['n_arm']}  hand N={res['n_hand']}  dur={res['duration_s']}s\n")
    print(f"{'stage':12s} {'st':3s} {'idx':>12s} {'t(s)':>13s} {'n':>5s}  "
          f"{'eeZΔmm':>7s} {'eePathmm':>8s} {'handΔ':>6s}")
    for s in res["segments"]:
        if args.all and s["state"] == "--":
            continue
        print(f"{s['stage']:12s} {s['state']:3s} [{s['i0']:5d}:{s['i1']:5d}] "
              f"[{s['t0']:5.1f}:{s['t1']:5.1f}] {s['n']:5d}  "
              f"{s['ee_dz_mm']:7.0f} {s['ee_path_mm']:8.0f} {s['hand_dmax']:6.2f}")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(res, f, indent=1)
        print(f"\n-> {args.json}")


if __name__ == "__main__":
    main()
