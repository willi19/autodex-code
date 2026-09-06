"""Interactive reorient scene browser.

Dropdowns: object, scene_type (reorient_0 / reorient_4 / reorient_8).
Slider/dropdown: scene file within that obj/scene_type.
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation as R

from autodex.utils.path import get_obj_root


SAMPLE_SPHERE_MARGIN = 0.0  # matches BODex after types.py:266 patched to 0


def _near_cuboid(p, dims, cpose, margin: float = SAMPLE_SPHERE_MARGIN):
    """True if sample point's `margin`-radius sphere intersects the cuboid.
    Matches BODex collision check: SDF < margin → invalid."""
    cT_t = np.array(cpose[:3], dtype=float)
    cT_r = R.from_quat([cpose[4], cpose[5], cpose[6], cpose[3]]).as_matrix()
    p_local = (p - cT_t) @ cT_r
    half = np.array([dims[0] / 2, dims[1] / 2, dims[2] / 2])
    q = np.abs(p_local) - half
    # signed distance to box surface (neg inside, 0 surface, pos outside)
    sdf = np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(np.max(q, axis=-1), 0.0)
    return sdf < margin


def _compute_samples(scene: dict, n_samples: int = 1024, inflate: float = 0.05):
    """Reproduce BODex's exact sampling pipeline:
      1. Load object mesh, apply pose transform
      2. Inflate: vertices += inflate * vertex_normals
      3. sample_surface_even(inflated_mesh, n_samples)
      4. Tile if fewer than n_samples
      5. Point-in-cuboid check against scene obstacles
    Returns:
      pts:   (N, 3) sampled offset points in world
      valid: (N,) bool — passed collision filter
    """
    mi = scene["scene"]["mesh"]["target"]
    mesh = trimesh.load(mi["file_path"], process=False, force="mesh")
    pose = mi["pose"]
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    T[:3, :3] = R.from_quat([pose[4], pose[5], pose[6], pose[3]]).as_matrix()
    mesh.apply_transform(T)
    mesh.vertices = mesh.vertices + inflate * mesh.vertex_normals  # BODex's inflation

    pts, fi = trimesh.sample.sample_surface_even(mesh, n_samples)
    pts = np.asarray(pts)
    if len(pts) == 0:
        return dict(pts=np.zeros((0, 3)), valid=np.zeros(0, bool))
    # tile to n_samples (BODex does this in types.py)
    import math
    repeat_num = math.ceil(n_samples / pts.shape[0])
    pts = np.tile(pts, (repeat_num, 1))[:n_samples]

    valid = np.ones(len(pts), bool)
    for name, info in scene["scene"].get("cuboid", {}).items():
        valid &= ~_near_cuboid(pts, info["dims"], info["pose"])
    return dict(pts=pts, valid=valid)


def _quat_wxyz(pose7):
    return np.array(pose7[3:7], dtype=float)


def _pos(pose7):
    return np.array(pose7[0:3], dtype=float)


def _color_for(name: str):
    if name.startswith("table"):
        return (220, 80, 80, 120)
    if name.startswith("pillar"):
        return (80, 120, 220, 160)
    return (150, 150, 150, 160)


def _list_dirs(p: Path):
    if not p.is_dir():
        return []
    return sorted(d.name for d in p.iterdir() if d.is_dir())


def _list_jsons(p: Path):
    if not p.is_dir():
        return []
    return sorted(f.name for f in p.iterdir() if f.suffix == ".json")


class SceneBrowser:
    def __init__(self, obj_root: Path, scene_type_glob: str = "reorient_*", port: int = 8080):
        self.obj_root = Path(obj_root)
        self.scene_type_glob = scene_type_glob
        self.server = viser.ViserServer(port=port)
        self.server.scene.world_axes.visible = True
        self._handles = []

        # Find objects that have at least one matching scene_type dir
        all_objs = _list_dirs(self.obj_root)
        objs = []
        for o in all_objs:
            for st in _list_dirs(self.obj_root / o / "scene"):
                if Path(st).match(scene_type_glob):
                    objs.append(o); break
        if not objs:
            raise RuntimeError(f"No objects with {scene_type_glob} under {self.obj_root}")

        with self.server.gui.add_folder("Browse"):
            self.obj_sel = self.server.gui.add_dropdown("Object", options=tuple(objs), initial_value=objs[0])
            self.st_sel = self.server.gui.add_dropdown("Scene type", options=("",), initial_value="")
            self.scene_sel = self.server.gui.add_dropdown("Scene file", options=("",), initial_value="")
            self.meta_text = self.server.gui.add_text("Meta", initial_value="", disabled=True)
        with self.server.gui.add_folder("Surface samples"):
            self.show_samples = self.server.gui.add_checkbox("Show", initial_value=True)
            self.n_samples = self.server.gui.add_slider("N", min=128, max=4096, step=128, initial_value=1024)
            self.inflate = self.server.gui.add_slider("inflate (m)", min=0.0, max=0.15, step=0.005, initial_value=0.05)
            self.sample_stats = self.server.gui.add_text("Stats", initial_value="", disabled=True)
        for h in [self.show_samples, self.n_samples, self.inflate]:
            @h.on_update
            def _(_): self._on_scene_change()

        @self.obj_sel.on_update
        def _(_): self._on_obj_change()

        @self.st_sel.on_update
        def _(_): self._on_st_change()

        @self.scene_sel.on_update
        def _(_): self._on_scene_change()

        self._on_obj_change()
        print(f"[viz] viser on http://localhost:{port}")

    def _on_obj_change(self):
        obj = self.obj_sel.value
        scene_types = [
            st for st in _list_dirs(self.obj_root / obj / "scene")
            if Path(st).match(self.scene_type_glob)
        ]
        if not scene_types:
            self.st_sel.options = ("",)
            return
        self.st_sel.options = tuple(scene_types)
        self.st_sel.value = scene_types[0]
        self._on_st_change()

    def _on_st_change(self):
        obj = self.obj_sel.value
        st = self.st_sel.value
        if not st:
            return
        scenes = _list_jsons(self.obj_root / obj / "scene" / st)
        if not scenes:
            self.scene_sel.options = ("",)
            return
        self.scene_sel.options = tuple(scenes)
        self.scene_sel.value = scenes[0]
        self._on_scene_change()

    def _on_scene_change(self):
        obj = self.obj_sel.value
        st = self.st_sel.value
        sn = self.scene_sel.value
        if not sn:
            return
        self._render(self.obj_root / obj / "scene" / st / sn)

    def _clear(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def _render(self, scene_path: Path):
        self._clear()
        with open(scene_path) as f:
            scene = json.load(f)

        meta = scene.get("meta", {})
        self.meta_text.value = json.dumps(meta)

        for name, entry in scene["scene"].get("mesh", {}).items():
            mesh = trimesh.load(entry["file_path"], process=False, force="mesh")
            scale = entry.get("scale", [1, 1, 1])
            if not np.allclose(scale, 1.0):
                mesh.apply_scale(scale)
            pose = entry["pose"]
            h = self.server.scene.add_mesh_trimesh(
                f"/mesh/{name}", mesh,
                position=_pos(pose), wxyz=_quat_wxyz(pose),
            )
            self._handles.append(h)

        for name, entry in scene["scene"].get("cuboid", {}).items():
            box = trimesh.creation.box(extents=entry["dims"])
            rgba = _color_for(name)
            box.visual.face_colors = np.tile(np.array(rgba, dtype=np.uint8),
                                              (len(box.faces), 1))
            pose = entry["pose"]
            h = self.server.scene.add_mesh_trimesh(
                f"/cuboid/{name}", box,
                position=_pos(pose), wxyz=_quat_wxyz(pose),
            )
            self._handles.append(h)

        if getattr(self, "show_samples", None) is not None and self.show_samples.value:
            s = _compute_samples(scene,
                                  n_samples=int(self.n_samples.value),
                                  inflate=float(self.inflate.value))
            pts = s["pts"]
            n = len(pts)
            colors = np.zeros((n, 3), dtype=np.uint8)
            colors[~s["valid"]] = [220, 60, 60]   # invalid (inside cuboid)
            colors[s["valid"]] = [60, 200, 60]    # valid
            ph = self.server.scene.add_point_cloud(
                "/samples/offset_points", points=pts.astype(np.float32),
                colors=colors, point_size=0.004,
            )
            self._handles.append(ph)
            inval = int((~s["valid"]).sum())
            val = int(s["valid"].sum())
            self.sample_stats.value = f"N={n}  invalid(red)={inval}  valid(green)={val}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="v8",
                        help="v8 tabletop asset contract (only supported value)")
    parser.add_argument("--obj_root", type=Path, default=None,
                        help="optional v8 object-root override")
    parser.add_argument("--scene_type", default="reorient_*",
                        help="glob pattern for scene_type dirs (default reorient_*)")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--scene", type=Path, default=None,
                        help="single scene file (legacy mode)")
    args = parser.parse_args()
    if args.version != "v8":
        parser.error("viz_scene supports only --version v8; legacy assets are not used")
    args.obj_root = args.obj_root or Path(get_obj_root(args.version))

    if args.scene:
        # legacy single-scene mode (no browser)
        server = viser.ViserServer(port=args.port)
        server.scene.world_axes.visible = True
        with open(args.scene) as f:
            scene = json.load(f)
        for name, entry in scene["scene"].get("mesh", {}).items():
            mesh = trimesh.load(entry["file_path"], process=False, force="mesh")
            server.scene.add_mesh_trimesh(
                f"/mesh/{name}", mesh,
                position=_pos(entry["pose"]), wxyz=_quat_wxyz(entry["pose"]),
            )
        for name, entry in scene["scene"].get("cuboid", {}).items():
            box = trimesh.creation.box(extents=entry["dims"])
            rgba = _color_for(name)
            box.visual.face_colors = np.tile(np.array(rgba, dtype=np.uint8),
                                              (len(box.faces), 1))
            server.scene.add_mesh_trimesh(
                f"/cuboid/{name}", box,
                position=_pos(entry["pose"]), wxyz=_quat_wxyz(entry["pose"]),
            )
        print(f"[viz] viser on http://localhost:{args.port}")
    else:
        SceneBrowser(args.obj_root, args.scene_type, args.port)

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
