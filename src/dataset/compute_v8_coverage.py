"""Precompute collision-free scene coverage for a v8 candidate pool.

For each candidate grasp (object-frame wrist) and each hand-specific deployment
scene (wall/shelf/box at a given tabletop pose_idx), test whether the grasp is
collision-free (hand vs obstacles+table). A grasp "covers" a scene iff it is
collision-free there — NO IK (per project convention: covers = collision-free
only; IK gating happens later at runtime selection).

Coverage is per tabletop pose_idx (same tabletop setting): a grasp only covers
scenes that share its pose_idx. A grasp's pose_idx is that of the scene it was
generated under; if that scene json is missing it falls back to the pose_idx
where the grasp is collision-free in the most scenes.

Output (A-format, read by autodex/utils/coverage.py):
    {project_dir}/experiment/{version}/coverage/cov_{version}_cand_{obj}.json
    {
      "object": obj,
      "scenes": [{"type","sid","pose_idx"}, ...],          # global scene list
      "grasps": [{"type","sid","gid","pose_idx","covers":[scene_idx,...]}, ...]
    }

Usage:
    python src/dataset/compute_v8_coverage.py --obj servingbowl_small --hand inspire
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

# Run as a script: sys.path[0] is src/dataset/, so `src.*` is not importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from autodex.utils.path import (get_candidate_path, get_scene_dir, project_dir,
                                get_obj_root)
from autodex.utils.conversion import cart2se3
from autodex.planner import GraspPlanner
from autodex.planner.planner import _to_curobo_world
from src.execution.scene_cfg import find_planning_mesh

SCENE_TYPES = ("wall", "shelf", "box")


def load_pool(obj, hand, version):
    """Walk candidates/{hand}/{version}/{obj}/{type}/{sid}/{gid}/ → grasp pool."""
    root = os.path.join(get_candidate_path(hand), version, obj)
    metas, wrists, pregs = [], [], []
    for st in sorted(os.listdir(root)):
        std = os.path.join(root, st)
        if st not in SCENE_TYPES or not os.path.isdir(std):
            continue
        for sid in sorted(os.listdir(std)):
            sd = os.path.join(std, sid)
            if not os.path.isdir(sd):
                continue
            for gid in sorted(os.listdir(sd)):
                gd = os.path.join(sd, gid)
                w = os.path.join(gd, "wrist_se3.npy")
                if not os.path.exists(w):
                    continue
                metas.append((st, sid, gid))
                wrists.append(np.load(w))
                pregs.append(np.load(os.path.join(gd, "pregrasp_pose.npy")))
    return metas, np.array(wrists), np.array(pregs)


_HOME_RE = re.compile(r"^/home/[^/]+/")


def _localize(p):
    """Rewrite a path authored under another user's home onto this host."""
    if isinstance(p, str) and _HOME_RE.match(p):
        return _HOME_RE.sub(os.path.expanduser("~") + "/", p, count=1)
    return p


def load_scenes(obj, hand, obj_root=None):
    """Hand-specific deployment scenes with a valid pose_idx (skip reorient_*).

    Scene JSONs were authored on another machine, so absolute paths inside are
    rewritten onto this host. The object mesh is additionally repointed at the
    planner's mesh for ``obj_root`` — coverage has to predict what the planner
    will actually collide-check, and a v8 pool plans against object_processing.
    """
    root = get_scene_dir(hand, obj)
    scenes = []  # (type, sid, pose_idx, scene_cfg)
    planning_mesh = find_planning_mesh(obj, obj_root)
    for st in SCENE_TYPES:
        std = os.path.join(root, st)
        if not os.path.isdir(std):
            continue
        for f in sorted(glob.glob(os.path.join(std, "*.json"))):
            d = json.load(open(f))
            pose_idx = str(d.get("meta", {}).get("pose_idx"))
            if pose_idx in ("None", ""):
                continue
            scfg = d["scene"]
            for name, m in (scfg.get("mesh") or {}).items():
                for k in ("file_path", "urdf_path"):
                    if k in m:
                        m[k] = _localize(m[k])
                if name == "target":
                    m["file_path"] = planning_mesh
            scenes.append((st, os.path.basename(f)[:-5], pose_idx, scfg))
    return scenes


def out_path_for(obj, version):
    return os.path.join(project_dir, "experiment", version, "coverage",
                        # `_cand` = full candidate pool (what
                        # autodex/utils/coverage.py reads). The bare
                        # `cov_{version}_{obj}.json` is the EXECUTED-grasp file
                        # (v7 has both: 1364 candidates vs 31 executed).
                        f"cov_{version}_cand_{obj}.json")


def extract_if_needed(obj, hand, version):
    """Unpack `{obj}.tar.gz` if the candidate dir is not on disk yet.

    v8 candidates ship as tarballs; a pool that was never extracted looks
    exactly like an object with no candidates, which the runner then reads as
    "everything already covered".
    """
    root = os.path.join(get_candidate_path(hand), version, obj)
    if os.path.isdir(root):
        return True
    tarball = root + ".tar.gz"
    if not os.path.exists(tarball):
        return False
    import tarfile
    print(f"[cov] extracting {os.path.basename(tarball)}")
    with tarfile.open(tarball) as tf:
        tf.extractall(os.path.dirname(root))
    return os.path.isdir(root)


def compute_one(obj, hand, version, planner, verbose=True):
    """Coverage for one object with an ALREADY-BUILT planner.

    The planner is the expensive part (~20 s of warmup), so a batch run builds
    it once and passes it in rather than paying that per object.
    """
    metas, wrist_obj, preg = load_pool(obj, hand, version)
    obj_root = get_obj_root(version)
    scenes = load_scenes(obj, hand, obj_root)
    G, S = len(metas), len(scenes)
    print(f"[cov] {obj}: {G} grasps, {S} scenes")
    if G == 0 or S == 0:
        print(f"[cov] {obj}: SKIP (empty pool or scenes)")
        return None

    # scene pose_idx per candidate sid (generating scene) → grasp's home pose
    scene_pose = {(st, sid): pose for (st, sid, pose, _) in scenes}
    args = argparse.Namespace(obj=obj, hand=hand, version=version)

    # collision-free mask per scene: free[s] = bool (G,)
    free = np.zeros((S, G), dtype=bool)
    for si, (st, sid, pose_idx, scfg) in enumerate(scenes):
        obj_se3 = cart2se3(scfg["mesh"]["target"]["pose"])
        wrist_world = np.einsum("ij,ajk->aik", obj_se3, wrist_obj)
        coll = planner._check_collision(_to_curobo_world(scfg), wrist_world, preg)
        free[si] = ~coll
        if verbose:
            print(f"[cov]   scene {st}/{sid} (pose {pose_idx}): "
                  f"{int(free[si].sum())}/{G} collision-free")

    # scene indices grouped by pose_idx
    scenes_by_pose = defaultdict(list)
    for si, (_, _, pose_idx, _) in enumerate(scenes):
        scenes_by_pose[pose_idx].append(si)

    grasps_json = []
    for gi, (gt, gsid, ggid) in enumerate(metas):
        home = scene_pose.get((gt, gsid))
        if home is None:
            # fallback: pose where this grasp is collision-free in most scenes
            best, best_n = None, -1
            for pose, sidxs in scenes_by_pose.items():
                n = int(sum(free[s, gi] for s in sidxs))
                if n > best_n:
                    best, best_n = pose, n
            home = best
        covers = [s for s in scenes_by_pose.get(home, []) if free[s, gi]]
        grasps_json.append({
            "type": gt, "sid": gsid, "gid": ggid,
            "pose_idx": home, "covers": covers,
        })

    scenes_json = [{"type": st, "sid": sid, "pose_idx": pose}
                   for (st, sid, pose, _) in scenes]
    out = {"object": obj, "scenes": scenes_json, "grasps": grasps_json}

    out_path = out_path_for(obj, version)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=1)

    # summary
    per_pose_cov = defaultdict(set)
    for g in grasps_json:
        per_pose_cov[g["pose_idx"]].update(g["covers"])
    print(f"[cov] wrote {out_path}")
    for pose, sidxs in sorted(scenes_by_pose.items()):
        print(f"[cov]   pose {pose}: {len(per_pose_cov[pose])}/{len(sidxs)} "
              f"scenes coverable by \u2265\u00b91 grasp".replace("\u00b9", ""))
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", help="single object; omit with --all")
    ap.add_argument("--objects", nargs="+", default=None,
                    help="explicit object list (planner is built once for all)")
    ap.add_argument("--all", action="store_true",
                    help="every object with candidates (or a tarball) for this hand/version")
    ap.add_argument("--hand", default="inspire")
    ap.add_argument("--version", default="v8")
    ap.add_argument("--overwrite", action="store_true",
                    help="recompute objects that already have a coverage json")
    ap.add_argument("--quiet", action="store_true",
                    help="drop the per-scene lines (implied by --all)")
    args = ap.parse_args()

    if not args.obj and not args.all and not args.objects:
        ap.error("pass --obj NAME, --objects A B C, or --all")

    root = os.path.join(get_candidate_path(args.hand), args.version)
    if args.all:
        names = set()
        for e in sorted(os.listdir(root)):
            if e.endswith(".tar.gz"):
                names.add(e[: -len(".tar.gz")])
            elif os.path.isdir(os.path.join(root, e)):
                names.add(e)
        objs = sorted(names)
    elif args.objects:
        objs = list(args.objects)
    else:
        objs = [args.obj]

    todo = [o for o in objs
            if args.overwrite or not os.path.exists(out_path_for(o, args.version))]
    print(f"[cov] {len(objs)} objects, {len(todo)} to compute "
          f"({len(objs) - len(todo)} already have coverage)")
    if not todo:
        return

    planner = GraspPlanner(hand=args.hand)     # built once, reused for all
    done, failed, skipped = [], [], []
    for i, obj in enumerate(todo, 1):
        print(f"\n[cov] ---- {i}/{len(todo)}  {obj} ----")
        try:
            if not extract_if_needed(obj, args.hand, args.version):
                print(f"[cov] {obj}: SKIP (no candidate dir and no tarball)")
                skipped.append(obj)
                continue
            r = compute_one(obj, args.hand, args.version, planner,
                            verbose=not (args.all or args.quiet))
            (done if r else skipped).append(obj)
        except Exception as exc:
            print(f"[cov] {obj}: FAILED {exc!r}")
            failed.append(obj)

    print(f"\n[cov] done={len(done)} skipped={len(skipped)} failed={len(failed)}")
    if skipped:
        print(f"[cov] skipped: {skipped}")
    if failed:
        print(f"[cov] failed : {failed}")


if __name__ == "__main__":
    main()
