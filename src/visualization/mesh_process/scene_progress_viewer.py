"""
Scene-progress grid viewer: for a given (hand, version, obj, scene_type) shows
every scene laid out in a grid and colors the target object by run status:
    green  = scene has a successful candidate result.json
    red    = scene has at least one result.json but none successful
    gray   = no result.json at all (unrun)

Mirrors src/visualization/mesh_process/scene_grid_viewer.py — uses the same
procedural scene builders, just swaps the target-mesh color per status.

Usage:
    python src/visualization/mesh_process/scene_progress_viewer.py \\
        --hand inspire_left --version v7
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from autodex.utils.path import obj_path, get_candidate_path, get_scene_dir

# Reuse helpers from scene_grid_viewer.
from src.visualization.mesh_process import scene_grid_viewer as sgv
from src.visualization.mesh_process.scene_grid_viewer import (
    COLORS, load_mesh, parse_pose, add_obb, clear_all,
)


# ── Status colors (RGB 0–1) ──────────────────────────────────────────────────
STATUS_COLOR = {
    "success": (0.05, 0.85, 0.05),   # bright green
    "failed":  (1.00, 0.05, 0.05),   # bright red
    "unrun":   (0.50, 0.50, 0.50),   # gray
}

# Visual override: shrink wall/shelf-back/sides/up height for clearer view.
MAX_OBSTACLE_HEIGHT = 0.3


# ── Scene status from candidate result.json ──────────────────────────────────

def _scene_status(hand: str, version: str, obj: str,
                  scene_type: str, scene_id: str) -> str:
    """Returns 'success' | 'failed' | 'unrun' for a given scene."""
    base = Path(get_candidate_path(hand)) / version / obj
    if scene_type:
        base = base / scene_type / scene_id
    else:
        base = base / scene_id
    if not base.is_dir():
        return "unrun"
    has_result = False
    for grasp_dir in base.iterdir():
        if not grasp_dir.is_dir():
            continue
        rj = grasp_dir / "result.json"
        if not rj.exists():
            continue
        has_result = True
        try:
            with open(rj) as f:
                if json.load(f).get("success", False):
                    return "success"
        except Exception:
            continue
    return "failed" if has_result else "unrun"


def _list_scenes(hand: str, version: str, obj_name: str, scene_type: str):
    """Scene ids from candidates/{hand}/{version}/{obj}/{scene_type}/.
    Each entry is (sid, scene_json_path_or_None) — json may be missing for
    objects whose scene dir is named differently (e.g. shelf vs shelf_prev)."""
    cand_dir = Path(get_candidate_path(hand)) / version / obj_name / scene_type
    if not cand_dir.is_dir():
        return []
    sids = sorted([p.name for p in cand_dir.iterdir() if p.is_dir()],
                  key=lambda s: int(s) if s.isdigit() else s)
    out = []
    for sid in sids:
        json_path = os.path.join(get_scene_dir(hand, obj_name, scene_type),
                                 f"{sid}.json")
        if not os.path.isfile(json_path):
            json_path = None
        out.append((sid, json_path))
    return out


def _build_scene(obj_name, scene_type, cfg, obb_info):
    """Use json's `scene` dict directly — already has exact cuboid dims/pose."""
    return cfg.get("scene", {})


# ── Load + render with status colors ─────────────────────────────────────────

def load_grid(vis, hand, version, obj_name, scene_type):
    clear_all()

    scenes = _list_scenes(hand, version, obj_name, scene_type)
    if not scenes:
        print(f"No scenes for {obj_name}/{scene_type}")
        return

    info_path = os.path.join(obj_path, obj_name, "processed_data",
                             "info", "simplified.json")
    with open(info_path) as f:
        obb_info = json.load(f)
    obb_extents = np.array(obb_info['obb'])
    obb_transform = np.array(obb_info['obb_transform'])
    axis_len = float(np.max(obb_extents)) * 0.6

    # Use simplified.obj (no material) so vis.change_color applies uniformly.
    simple_path = os.path.join(obj_path, obj_name, "processed_data",
                               "mesh", "simplified.obj")
    raw_mesh = load_mesh(simple_path)

    margin = 0.1
    grid_spacing = float(np.linalg.norm(obb_extents)) * 3.0 + margin
    # Also account for obstacle cuboid extents (shelf walls extend far past
    # the OBB) so adjacent scenes don't overlap.
    max_cuboid_dim = 0.0
    for _, sp in scenes:
        if sp is None:
            continue
        try:
            with open(sp) as f:
                c = json.load(f)
            for name, ent in c.get("scene", {}).get("cuboid", {}).items():
                if name == "table":
                    continue
                max_cuboid_dim = max(max_cuboid_dim, *ent["dims"][:2])
        except Exception:
            continue
    grid_spacing = max(grid_spacing, max_cuboid_dim * 1.2 + margin)
    grid_cols = int(np.ceil(np.sqrt(len(scenes))))

    counts = {"success": 0, "failed": 0, "unrun": 0}
    print(f"Loading {len(scenes)} {scene_type} scenes in {grid_cols}x grid...")

    for idx, (sid, scene_path) in enumerate(scenes):
        if scene_path is None:
            print(f"  [skip] {obj_name}/{scene_type}/{sid}: scene json missing")
            continue
        with open(scene_path) as f:
            cfg = json.load(f)
        scene = _build_scene(obj_name, scene_type, cfg, obb_info)
        if scene is None:
            continue
        status = _scene_status(hand, version, obj_name, scene_type, sid)
        counts[status] += 1
        target_color = STATUS_COLOR[status]

        row = idx // grid_cols
        col = idx % grid_cols
        offset = np.array([
            (col - grid_cols // 2) * grid_spacing,
            (row - grid_cols // 2) * grid_spacing,
            0.0,
        ])

        for mesh_name, mesh_info in scene.get("mesh", {}).items():
            mesh = raw_mesh if mesh_name == "target" else trimesh.load(
                mesh_info["file_path"], force="mesh")
            pose = parse_pose(mesh_info["pose"])
            pose[:3, 3] += offset
            name = f"s{sid}_{mesh_name}"
            vis.add_object(name, mesh, obj_T=pose)
            if mesh_name == "target":
                fp = f"/objects/{name}_frame"
                add_obb(fp, obb_extents, obb_transform, axis_len)

        for cuboid_name, cuboid_info in scene.get("cuboid", {}).items():
            dims = list(cuboid_info["dims"])
            pose = parse_pose(cuboid_info["pose"])
            # Shrink tall obstacles (wall / shelf walls) for clearer view.
            # Keep BOTTOM of cuboid at original z so it stays on the table.
            if cuboid_name != "table" and dims[2] > MAX_OBSTACLE_HEIGHT:
                bot_z = pose[2, 3] - dims[2] / 2
                dims[2] = MAX_OBSTACLE_HEIGHT
                pose[2, 3] = bot_z + dims[2] / 2
            box = trimesh.creation.box(extents=dims)
            # Set face_colors so viser renders correct color (mesh_handle.color
            # alone is overridden by trimesh box's default visual).
            if cuboid_name == "table":
                rgba = (240, 240, 245, 230)
            else:
                rgba = (int(target_color[0] * 255),
                        int(target_color[1] * 255),
                        int(target_color[2] * 255), 153)
            box.visual.face_colors = np.tile(
                np.array(rgba, dtype=np.uint8), (len(box.faces), 1))
            pose[:3, 3] += offset
            name = f"s{sid}_{cuboid_name}"
            vis.add_object(name, box, obj_T=pose)

    print(f"[{obj_name}/{scene_type}] "
          f"success={counts['success']}  "
          f"failed={counts['failed']}  "
          f"unrun={counts['unrun']}  total={len(scenes)}")


# ── GUI ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="inspire_left")
    ap.add_argument("--version", default="v7")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    from paradex.visualization.visualizer.viser import ViserViewer

    cand_root = Path(get_candidate_path(args.hand)) / args.version
    objs = sorted(p.name for p in cand_root.iterdir()
                  if p.is_dir() and Path(get_scene_dir(args.hand, p.name)).is_dir())
    if not objs:
        print(f"No objects under {cand_root}")
        return
    print(f"Found {len(objs)} objects with candidates+scenes")

    vis = ViserViewer(port_number=args.port)
    # scene_grid_viewer.clear_all / parse_pose use module-level `vis` global.
    sgv.vis = vis

    with vis.server.gui.add_folder("Scene Progress"):
        obj_dd = vis.server.gui.add_dropdown(
            "Object", options=tuple(objs), initial_value=objs[0])
        st_dd = vis.server.gui.add_dropdown(
            "Scene Type", options=("",), initial_value="")
        load_btn = vis.server.gui.add_button("Load Grid")
        legend = vis.server.gui.add_markdown(
            "**green** = success  •  **red** = tried, all fail  •  **gray** = unrun")

    def _update_scene_types():
        obj_name = obj_dd.value
        # Scene types = those that exist under candidates/{hand}/{version}/{obj}/
        # (not all dirs under {obj}/scene/ — many are unrelated to v7).
        cand_obj = Path(get_candidate_path(args.hand)) / args.version / obj_name
        if cand_obj.is_dir():
            types = sorted(d.name for d in cand_obj.iterdir() if d.is_dir())
        else:
            types = []
        st_dd.options = tuple(types) if types else ("",)
        if types:
            st_dd.value = types[0]

    @obj_dd.on_update
    def _(_):
        _update_scene_types()

    @load_btn.on_click
    def _(_):
        load_grid(vis, args.hand, args.version, obj_dd.value, st_dd.value)

    _update_scene_types()
    load_grid(vis, args.hand, args.version, obj_dd.value, st_dd.value)

    vis.start_viewer()


if __name__ == "__main__":
    main()
