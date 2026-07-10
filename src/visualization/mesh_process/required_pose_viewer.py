"""Visualize, per object, each tabletop pose preset alongside how many
unrun scenes need it. Useful for deciding which physical orientation to
place the object in before running run_auto.

Renders a row of object meshes — one per tabletop pose file under
{obj}/processed_data/info/tabletop/{N}.npy. Each row also labels how many
unrun (success-still-missing) scenes match that tabletop class.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from autodex.utils.path import obj_path, get_candidate_path, get_scene_dir
from autodex.utils.conversion import cart2se3
from src.experiment.reset.tabletop_pose import classify_tabletop_pose
from src.visualization.mesh_process import scene_grid_viewer as sgv
from src.visualization.mesh_process.scene_grid_viewer import (
    load_mesh, parse_pose, clear_all,
)


def _scene_status(hand, version, obj, st, sid):
    base = Path(get_candidate_path(hand)) / version / obj / st / sid
    if not base.is_dir():
        return "unrun"
    has = False
    for g in base.iterdir():
        if not g.is_dir():
            continue
        rj = g / "result.json"
        if not rj.exists():
            continue
        has = True
        try:
            if json.load(open(rj)).get("success", False):
                return "success"
        except Exception:
            pass
    return "failed" if has else "unrun"


def _unrun_tabletop_counts(hand, version, obj):
    """Returns Counter mapping tabletop filename -> n unrun scenes needing it."""
    root = Path(get_candidate_path(hand)) / version / obj
    if not root.is_dir():
        return Counter()
    cnt = Counter()
    for st_dir in root.iterdir():
        if not st_dir.is_dir():
            continue
        for sid_dir in st_dir.iterdir():
            if not sid_dir.is_dir():
                continue
            if _scene_status(hand, version, obj, st_dir.name, sid_dir.name) != "unrun":
                continue
            sj = Path(get_scene_dir(hand, obj, st_dir.name)) / f"{sid_dir.name}.json"
            if not sj.is_file():
                continue
            try:
                T = cart2se3(json.load(open(sj))["scene"]["mesh"]["target"]["pose"])
            except Exception:
                continue
            tb = classify_tabletop_pose(T, obj)
            if tb is None:
                continue
            cnt[tb["filename"]] += 1
    return cnt


_URDF_BY_HAND = {
    "inspire_left": Path.home() / "shared_data/AutoDex/content/assets/robot"
                                  "/inspire_left_description/xarm_inspire_left.urdf",
    "inspire":      Path.home() / "shared_data/AutoDex/content/assets/robot"
                                  "/inspire_description/xarm_inspire.urdf",
    "allegro":      Path.home() / "shared_data/AutoDex/content/assets/robot"
                                  "/allegro_description/xarm_allegro.urdf",
}

# Place each tabletop instance at this x (robot front) so xarm can "see" it.
OBJ_BASE_X = 0.45
TABLE_Z = 0.0  # tabletop sits at z=0 in robot frame (matches scene jsons)


_label_handles: list = []
# Robot-frame x where obj is placed (typical workspace center).
OBJ_X_ROBOT_FRAME = 0.45


def _clear_labels():
    for h in _label_handles:
        try: h.remove()
        except Exception: pass
    _label_handles.clear()


def render_tabletops(vis, hand, version, obj):
    """Initial set-up: load xarm, build sorted tabletop list, render index 0."""
    from autodex.utils.robot_config import INIT_STATE

    clear_all()
    _clear_labels()

    # Robot at INIT pose so user sees how reachable each tabletop is.
    urdf_path = _URDF_BY_HAND.get(hand)
    if urdf_path is not None and urdf_path.is_file():
        try:
            if "xarm" in getattr(vis, "robot_dict", {}):
                vis.robot_dict["xarm"].update_cfg(np.asarray(INIT_STATE))
            else:
                vis.add_robot("xarm", str(urdf_path))
                vis.robot_dict["xarm"].update_cfg(np.asarray(INIT_STATE))
        except Exception as e:
            print(f"[xarm load] {e!r}")

    simple_path = os.path.join(obj_path, obj, "processed_data", "mesh",
                               "simplified.obj")
    mesh = load_mesh(simple_path)

    tt_dir = os.path.join(obj_path, obj, "processed_data", "info", "tabletop")
    tt_files = sorted(glob.glob(os.path.join(tt_dir, "*.npy")))
    if not tt_files:
        print(f"[{obj}] no tabletop poses")
        return

    counts = _unrun_tabletop_counts(hand, version, obj)
    print(f"[{obj}] unrun scenes per tabletop: {dict(counts)}")

    # Sort tabletops by unrun count desc (best first).
    entries = []
    for tt_path in tt_files:
        fname = os.path.basename(tt_path)
        T = np.load(tt_path).astype(np.float64)
        if T.shape == (3, 3):
            tmp = np.eye(4); tmp[:3, :3] = T; T = tmp
        entries.append((counts.get(fname, 0), fname, T))
    entries.sort(key=lambda e: -e[0])

    bbox = mesh.bounds[1] - mesh.bounds[0]
    return {"obj": obj, "mesh": mesh, "bbox": bbox,
            "entries": entries, "counts": counts}


def render_one(vis, state, idx):
    """Render obj at robot-frame x=OBJ_X_ROBOT_FRAME with tabletop[idx] orientation."""
    n, fname, T = state["entries"][idx]
    mesh = state["mesh"]
    bbox = state["bbox"]

    # Remove just the tabletop instance (keep xarm + strip).
    for name in ("tt_current",):
        if name in getattr(vis, "obj_dict", {}):
            try: vis.obj_dict[name]["frame"].remove()
            except Exception: pass
            vis.obj_dict.pop(name, None)
    _clear_labels()

    T_world = T.copy()
    T_world[0, 3] += OBJ_X_ROBOT_FRAME

    if n == 0:
        rgb = (170, 170, 170)
    else:
        t = min(1.0, n / 10.0)
        rgb = (int(255 * t + 200 * (1 - t)),
               int(120 * (1 - t)),
               int(60 * (1 - t)))

    m = mesh.copy()
    m.visual.face_colors = np.tile(
        np.array([*rgb, 255], dtype=np.uint8), (len(m.faces), 1))
    vis.add_object("tt_current", m, obj_T=T_world)

    try:
        lh = vis.server.scene.add_label(
            "/labels/current",
            text=f"{fname.replace('.npy', '')}  ({n} unrun  rank {idx+1}/"
                 f"{len(state['entries'])})",
            position=(T_world[0, 3],
                      T_world[1, 3],
                      T_world[2, 3] + float(bbox.max()) + 0.06),
        )
        _label_handles.append(lh)
    except Exception:
        pass


def add_strip(vis):
    """Floor strip + workspace footprint at robot base."""
    if "strip" in getattr(vis, "obj_dict", {}):
        return
    strip = trimesh.creation.box(extents=[1.0, 1.5, 0.005])
    strip.visual.face_colors = np.tile(
        np.array([225, 225, 230, 220], dtype=np.uint8),
        (len(strip.faces), 1))
    T = np.eye(4)
    T[0, 3] = OBJ_X_ROBOT_FRAME
    T[2, 3] = -0.003
    vis.add_object("strip", strip, obj_T=T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="inspire_left")
    ap.add_argument("--version", default="v7")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    from paradex.visualization.visualizer.viser import ViserViewer

    cand_root = Path(get_candidate_path(args.hand)) / args.version
    objs = sorted(p.name for p in cand_root.iterdir()
                  if p.is_dir() and
                  (Path(obj_path) / p.name / "processed_data" /
                   "info" / "tabletop").is_dir())
    if not objs:
        print(f"No objects with tabletop presets under {cand_root}")
        return

    vis = ViserViewer(port_number=args.port)
    sgv.vis = vis
    add_strip(vis)

    state = {"obj": None, "entries": [], "mesh": None, "bbox": None,
             "counts": {}}

    with vis.server.gui.add_folder("Required tabletop poses"):
        obj_dd = vis.server.gui.add_dropdown(
            "Object", options=tuple(objs), initial_value=objs[0])
        rank_slider = vis.server.gui.add_slider(
            "Rank (best→worst)", min=0, max=0, step=1, initial_value=0)
        vis.server.gui.add_markdown(
            "Color intensity (orange→red) = unrun count. **Gray** = none.")

    def _load(obj_name):
        s = render_tabletops(vis, args.hand, args.version, obj_name)
        if s is None:
            return
        state.update(s)
        add_strip(vis)
        rank_slider.max = max(0, len(state["entries"]) - 1)
        rank_slider.value = 0
        render_one(vis, state, 0)

    @obj_dd.on_update
    def _(_):
        _load(obj_dd.value)

    @rank_slider.on_update
    def _(_):
        if state["entries"]:
            render_one(vis, state, int(rank_slider.value))

    _load(obj_dd.value)
    vis.start_viewer()


if __name__ == "__main__":
    main()
