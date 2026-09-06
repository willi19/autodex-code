"""Overlay two scenes in viser for visual comparison.
e.g., reorient_8/1_2 vs reorient_8/2_1 to inspect mirror asymmetry.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation as R

from autodex.utils.path import get_obj_root


def _pos(p): return np.array(p[:3], dtype=np.float64)
def _wxyz(p): return np.array(p[3:7], dtype=np.float64)


def add_scene(server, prefix, scene_json_path, mesh_rgba, cuboid_rgba):
    with open(scene_json_path) as f:
        scene = json.load(f)
    for name, entry in scene["scene"].get("mesh", {}).items():
        mesh = trimesh.load(entry["file_path"], process=False, force="mesh")
        rgba = np.array(mesh_rgba, dtype=np.uint8)
        mesh.visual.face_colors = np.tile(rgba, (len(mesh.faces), 1))
        pose = entry["pose"]
        server.scene.add_mesh_trimesh(f"{prefix}/mesh/{name}", mesh,
                                       position=_pos(pose), wxyz=_wxyz(pose))
    for name, entry in scene["scene"].get("cuboid", {}).items():
        box = trimesh.creation.box(extents=entry["dims"])
        rgba = np.array(cuboid_rgba, dtype=np.uint8)
        box.visual.face_colors = np.tile(rgba, (len(box.faces), 1))
        pose = entry["pose"]
        server.scene.add_mesh_trimesh(f"{prefix}/cuboid/{name}", box,
                                       position=_pos(pose), wxyz=_wxyz(pose))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", required=True)
    parser.add_argument("--scene_type", default="reorient_8")
    parser.add_argument("--scene_a", required=True, help="e.g., 1_2")
    parser.add_argument("--scene_b", required=True, help="e.g., 2_1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--version", default="v8",
                        help="v8 tabletop asset contract (only supported value)")
    parser.add_argument("--obj_root", type=Path, default=None,
                        help="optional v8 object-root override")
    args = parser.parse_args()
    if args.version != "v8":
        parser.error("overlay_two_scenes supports only --version v8; legacy assets are not used")
    args.obj_root = args.obj_root or Path(get_obj_root(args.version))

    base = args.obj_root / args.obj / "scene" / args.scene_type
    server = viser.ViserServer(port=args.port)
    server.scene.world_axes.visible = True

    add_scene(server, f"/A_{args.scene_a}",
              base / f"{args.scene_a}.json",
              mesh_rgba=(200, 60, 60, 180),       # red mesh
              cuboid_rgba=(220, 80, 80, 90))      # red obstacle
    add_scene(server, f"/B_{args.scene_b}",
              base / f"{args.scene_b}.json",
              mesh_rgba=(60, 80, 200, 180),       # blue mesh
              cuboid_rgba=(80, 100, 220, 90))     # blue obstacle

    print(f"[viz] http://localhost:{args.port}  (red={args.scene_a}, blue={args.scene_b})")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
