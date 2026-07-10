"""For each unrun scene under candidates/{hand}/{version}/{obj}/, print the
required initial object pose (from the scene json's target mesh) and the
nearest tabletop pose class.

The tabletop class tells you which existing {obj}/processed_data/info/tabletop
preset you should physically place the object in before running run_auto.

Usage:
    python src/visualization/scene_required_pose.py --hand inspire_left --version v7
    python src/visualization/scene_required_pose.py --hand inspire_left --version v7 \\
        --obj icecream_scoop --scene_type shelf
    python src/visualization/scene_required_pose.py ... --only-unrun
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from autodex.utils.path import obj_path, get_candidate_path, get_scene_dir
from autodex.utils.conversion import cart2se3
from src.experiment.reset.tabletop_pose import classify_tabletop_pose


def _scene_status(hand, version, obj, scene_type, sid) -> str:
    base = Path(get_candidate_path(hand)) / version / obj / scene_type / sid
    if not base.is_dir():
        return "unrun"
    has_result = False
    for g in base.iterdir():
        if not g.is_dir():
            continue
        rj = g / "result.json"
        if not rj.exists():
            continue
        has_result = True
        try:
            if json.load(open(rj)).get("success", False):
                return "success"
        except Exception:
            pass
    return "failed" if has_result else "unrun"


def _list_scenes(hand, version, obj, scene_type=None):
    """Yield (scene_type, sid) under candidates/{hand}/{version}/{obj}/."""
    root = Path(get_candidate_path(hand)) / version / obj
    if not root.is_dir():
        return
    sts = ([scene_type] if scene_type
           else sorted(d.name for d in root.iterdir() if d.is_dir()))
    for st in sts:
        st_dir = root / st
        if not st_dir.is_dir():
            continue
        for sid_dir in sorted(st_dir.iterdir(),
                              key=lambda p: int(p.name) if p.name.isdigit() else p.name):
            if sid_dir.is_dir():
                yield st, sid_dir.name


def _quat_wxyz_to_rpy_deg(q):
    """wxyz -> roll/pitch/yaw in degrees."""
    from scipy.spatial.transform import Rotation as R
    return R.from_quat([q[1], q[2], q[3], q[0]]).as_euler("xyz", degrees=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="inspire_left")
    ap.add_argument("--version", default="v7")
    ap.add_argument("--obj", default=None)
    ap.add_argument("--scene_type", default=None)
    ap.add_argument("--only-unrun", action="store_true",
                    help="skip success/failed scenes")
    args = ap.parse_args()

    cand_root = Path(get_candidate_path(args.hand)) / args.version
    objs = ([args.obj] if args.obj
            else sorted(p.name for p in cand_root.iterdir() if p.is_dir()))

    for obj in objs:
        scenes = list(_list_scenes(args.hand, args.version, obj, args.scene_type))
        if not scenes:
            continue
        rows = []
        for st, sid in scenes:
            status = _scene_status(args.hand, args.version, obj, st, sid)
            if args.only_unrun and status != "unrun":
                continue
            sj = Path(get_scene_dir(args.hand, obj, st)) / f"{sid}.json"
            if not sj.is_file():
                rows.append((st, sid, status, "no_json", None, None))
                continue
            try:
                cfg = json.load(open(sj))
                pose_xyzwxyz = cfg["scene"]["mesh"]["target"]["pose"]
            except Exception:
                rows.append((st, sid, status, "bad_json", None, None))
                continue
            T = cart2se3(pose_xyzwxyz)
            rpy = _quat_wxyz_to_rpy_deg(pose_xyzwxyz[3:7])
            tb = classify_tabletop_pose(T, obj)
            tb_label = (f"{tb['filename']} ({tb['rot_err_deg']:.1f}°)"
                        if tb is not None else "—")
            rows.append((st, sid, status,
                         f"xyz=[{T[0,3]:.3f},{T[1,3]:.3f},{T[2,3]:.3f}]",
                         f"rpy=[{rpy[0]:6.1f},{rpy[1]:6.1f},{rpy[2]:6.1f}]",
                         tb_label))
        if not rows:
            continue
        print(f"\n=== {obj} ({len(rows)} scenes) ===")
        print(f"{'scene':>12}  {'sid':>3}  {'status':>7}  {'pose':>40}  "
              f"{'rpy(deg)':>30}  tabletop")
        for st, sid, status, pose_s, rpy_s, tb_s in rows:
            rpy_s = rpy_s or ""
            tb_s = tb_s or ""
            print(f"{st:>12}  {sid:>3}  {status:>7}  {pose_s:>40}  "
                  f"{rpy_s:>30}  {tb_s}")


if __name__ == "__main__":
    main()
