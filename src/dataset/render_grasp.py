"""Render each trial's grasp candidate as a turntable video, into the trial dir.

Reuses ``src/visualization/turntable_grasp.py`` rendering. For each dataset
trial we read ``result.json`` scene_info = [scene, scene_id, grasp], resolve the
matching RSS_2026 candidate (objects live under different version dirs, e.g.
``selected_100`` vs ``tselected_100``), point the renderer's SELECTED_DIR there,
and write ``{trial}/grasp_turntable.mp4``.

Run in the ``mingi`` conda env (needs paradex + open3d + EGL):
    ~/miniconda3/envs/mingi/bin/python -m src.dataset.render_grasp --trial <dir>
    ~/miniconda3/envs/mingi/bin/python -m src.dataset.render_grasp            # all
"""

import argparse
import glob
import json
import os
import types

import src.visualization.turntable_grasp as tg

DATASET_ROOTS = [
    "/home/mingi/shared_data/autodex_dataset/selected_100",
    "/home/mingi/shared_data/autodex_dataset/selected_100_wireout",
]
CAND_GLOB = "/home/mingi/RSS_2026/candidates/*/{obj}/{scene}/{sid}/{gid}"
# result.json (scene_info) lives with the source captures, not the dataset trial.
SRC_ROOT = "/home/mingi/shared_data/RSS2026_Mingi/experiment/selected_100"
OUT_NAME = "grasp_turntable.mp4"


def render_args(width=960, height=540, frames=60, fps=30, fov=45.0,
                elevation=25.0, padding=1.3):
    return types.SimpleNamespace(
        width=width, height=height, frames=frames, fps=fps, fov=fov,
        elevation=elevation, padding=padding,
    )


def resolve_selected_dir(obj, scene, sid, gid):
    """Version root (.../candidates/{version}) whose grasp dir has grasp_pose."""
    for hit in sorted(glob.glob(CAND_GLOB.format(obj=obj, scene=scene, sid=sid, gid=gid))):
        if os.path.exists(os.path.join(hit, "grasp_pose.npy")):
            # SELECTED_DIR must be the version root so {version}/{obj}/... resolves.
            version_root = hit.split(f"/{obj}/")[0]
            return version_root
    return None


def render_trial(trial_dir, args, overwrite=False):
    out = os.path.join(trial_dir, OUT_NAME)
    if os.path.exists(out) and not overwrite:
        return "skip_exist"
    obj = os.path.basename(os.path.dirname(trial_dir))
    ts = os.path.basename(trial_dir)
    rj = os.path.join(trial_dir, "result.json")
    if not os.path.exists(rj):
        rj = os.path.join(SRC_ROOT, obj, ts, "result.json")  # source has scene_info
    if not os.path.exists(rj):
        return "no_result"
    si = json.load(open(rj)).get("scene_info")
    if not si or len(si) != 3:
        return "bad_scene_info"
    scene, sid, gid = si
    sel = resolve_selected_dir(obj, scene, sid, gid)
    if sel is None:
        return "no_candidate"
    tg.SELECTED_DIR = sel
    ok = tg.render_single_grasp(obj, scene, sid, gid, out, args)
    return "ok" if ok else "render_fail"


def iter_trials(roots):
    for root in roots:
        if not os.path.isdir(root):
            continue
        for obj in sorted(os.listdir(root)):
            obj_dir = os.path.join(root, obj)
            if not os.path.isdir(obj_dir):
                continue
            for ts in sorted(os.listdir(obj_dir)):
                td = os.path.join(obj_dir, ts)
                if os.path.isdir(td):
                    yield td


def main():
    global SRC_ROOT, CAND_GLOB
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trial", help="single trial dir")
    ap.add_argument("--roots", nargs="+", default=DATASET_ROOTS)
    ap.add_argument("--hand", default="allegro")
    ap.add_argument("--src_root", default=SRC_ROOT,
                    help="source captures holding result.json (scene_info)")
    ap.add_argument("--cand_glob", default=CAND_GLOB,
                    help="glob for the grasp candidate dir; must contain "
                         "{obj}/{scene}/{sid}/{gid}. For inspire point at "
                         "~/shared_data/AutoDex/candidates/inspire/*/...")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    args = ap.parse_args()

    SRC_ROOT = args.src_root
    CAND_GLOB = args.cand_glob
    tg.CURRENT_HAND = args.hand
    tg.OBJ_ROOT = None
    r_args = render_args(width=args.width, height=args.height, frames=args.frames)

    trials = [args.trial] if args.trial else list(iter_trials(args.roots))
    print(f"rendering {len(trials)} trial(s)")
    stats = {}
    for td in trials:
        res = render_trial(td, r_args, overwrite=args.overwrite)
        stats[res] = stats.get(res, 0) + 1
        if res not in ("ok", "skip_exist"):
            print(f"  {res}: {td}")
    print("summary:", stats)


if __name__ == "__main__":
    main()
