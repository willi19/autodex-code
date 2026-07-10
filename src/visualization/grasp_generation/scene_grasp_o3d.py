"""Interactive Open3D GUI: view an object's scenes + its grasp candidates together.

For a selected (hand, version, object, scene_type, scene_id) this shows the
scene (object mesh at its tabletop pose + table/obstacle cuboids from the scene
JSON) and overlays the grasp-candidate hands for that scene. A slider controls
how many candidate hands are overlaid at once.

This is the Open3D-GUI counterpart of the Viser viewer
``src/visualization/grasp_generation/view_bodex.py`` — same data sources, but a
standalone desktop window (mouse orbit/zoom) instead of a web viewer.

Needs a display (run on the workstation, not headless).

Usage:
    python src/visualization/grasp_generation/scene_grasp_o3d.py
    python src/visualization/grasp_generation/scene_grasp_o3d.py --hand allegro --version selected_100 --obj apple
"""

import os
import json
import argparse
import colorsys

import numpy as np
import trimesh
import yourdfpy
import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from scipy.spatial.transform import Rotation as Rot

from autodex.utils.path import obj_path, repo_dir, get_scene_dir

REPO_ROOT = repo_dir
CANDIDATES_ROOT = os.path.join(REPO_ROOT, "candidates")

# Hand URDFs (BODex forked-cuRobo assets) — same set as view_bodex.py.
HAND_URDFS = {
    "allegro": os.path.join(
        REPO_ROOT, "src", "grasp_generation", "BODex", "src", "curobo",
        "content", "assets", "robot", "allegro_description",
        "allegro_hand_description_right.urdf",
    ),
    "inspire": os.path.join(
        REPO_ROOT, "src", "grasp_generation", "BODex", "src", "curobo",
        "content", "assets", "robot", "inspire_description",
        "inspire_hand_right.urdf",
    ),
    "inspire_left": os.path.join(
        REPO_ROOT, "src", "grasp_generation", "BODex", "src", "curobo",
        "content", "assets", "robot", "inspire_description",
        "inspire_hand_left.urdf",
    ),
    "inspire_f1": os.path.join(
        REPO_ROOT, "src", "grasp_generation", "BODex", "src", "curobo",
        "content", "assets", "robot", "inspire_f1_description",
        "inspire_f1_hand_right.urdf",
    ),
    "inspire_f1_left": os.path.join(
        REPO_ROOT, "src", "grasp_generation", "BODex", "src", "curobo",
        "content", "assets", "robot", "inspire_f1_left_description",
        "inspire_f1_hand_left.urdf",
    ),
}

COLOR_OBJECT = (0.70, 0.70, 0.72)
COLOR_TABLE = (0.94, 0.94, 0.96)
COLOR_OBSTACLE = (0.47, 0.53, 0.60)
OBSTACLE_OPACITY = 0.4


def cart2se3(pose):
    """[x, y, z, qw, qx, qy, qz] -> 4x4 SE3."""
    T = np.eye(4)
    T[:3, 3] = pose[:3]
    qw, qx, qy, qz = pose[3:7]
    T[:3, :3] = Rot.from_quat([qx, qy, qz, qw]).as_matrix()
    return T


def list_dirs(path):
    if not os.path.isdir(path):
        return []
    return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))


def sort_numeric(names):
    return sorted(names, key=lambda x: (0, int(x)) if x.isdigit() else (1, x))


def trimesh_to_o3d(mesh, color=None):
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    o3d_mesh.compute_vertex_normals()
    if color is not None:
        o3d_mesh.paint_uniform_color(color)
    return o3d_mesh


def distinct_colors(n):
    """n visually distinct RGB colors (0-1) via evenly spaced hues."""
    return [colorsys.hsv_to_rgb(i / max(n, 1), 0.65, 0.95) for i in range(n)]


class SceneGraspViewer:
    def __init__(self, window, init):
        self.window = window
        self._urdf_cache = {}
        self._pitch_mult = init.pitch_mult

        em = window.theme.font_size

        # --- 3D scene widget ---
        self.scene_widget = gui.SceneWidget()
        self.scene_widget.scene = rendering.Open3DScene(window.renderer)
        self.scene_widget.scene.set_background([1, 1, 1, 1])
        self.scene_widget.scene.scene.set_sun_light(
            [0.5, -0.5, -1.0], [1, 1, 1], 75000
        )
        self.scene_widget.scene.scene.enable_sun_light(True)
        window.add_child(self.scene_widget)

        # --- control panel ---
        self.panel = gui.Vert(0.4 * em, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em, 0.5 * em))

        self.hand_cb = gui.Combobox()
        self.version_cb = gui.Combobox()
        self.obj_cb = gui.Combobox()
        self.scene_type_cb = gui.Combobox()
        self.scene_id_cb = gui.Combobox()

        self.hand_cb.set_on_selection_changed(lambda *_: self._on_hand())
        self.version_cb.set_on_selection_changed(lambda *_: self._on_version())
        self.obj_cb.set_on_selection_changed(lambda *_: self._on_object())
        self.scene_type_cb.set_on_selection_changed(lambda *_: self._on_scene_type())
        self.scene_id_cb.set_on_selection_changed(lambda *_: self._on_scene_id())

        self.num_slider = gui.Slider(gui.Slider.INT)
        self.num_slider.set_limits(1, 1)
        self.num_slider.int_value = 1
        self.num_slider.set_on_value_changed(lambda *_: self._render())

        self.show_obstacle = gui.Checkbox("Show obstacles")
        self.show_obstacle.checked = True
        self.show_obstacle.set_on_checked(lambda *_: self._render())

        self.show_table = gui.Checkbox("Show table")
        self.show_table.checked = True
        self.show_table.set_on_checked(lambda *_: self._render())

        self.show_grasps = gui.Checkbox("Show grasps")
        self.show_grasps.checked = True
        self.show_grasps.set_on_checked(lambda *_: self._render())

        self.color_by_index = gui.Checkbox("Color hands by index")
        self.color_by_index.checked = True
        self.color_by_index.set_on_checked(lambda *_: self._render())

        self.grid_mode = gui.Checkbox("Grid: all scenes of type")
        self.grid_mode.checked = False
        self.grid_mode.set_on_checked(lambda *_: self._render(reset_camera=True))

        self.status = gui.Label("")
        self._label_handles = []

        def row(label, widget):
            h = gui.Horiz(0.4 * em)
            lbl = gui.Label(label)
            lbl.text_color = gui.Color(0.2, 0.2, 0.2)
            h.add_child(lbl)
            h.add_stretch()
            h.add_child(widget)
            return h

        self.panel.add_child(row("Hand", self.hand_cb))
        self.panel.add_child(row("Version", self.version_cb))
        self.panel.add_child(row("Object", self.obj_cb))
        self.panel.add_child(row("Scene type", self.scene_type_cb))
        self.panel.add_child(row("Scene id", self.scene_id_cb))
        self.panel.add_fixed(0.3 * em)
        self.panel.add_child(gui.Label("# grasps overlaid"))
        self.panel.add_child(self.num_slider)
        self.panel.add_fixed(0.3 * em)
        self.panel.add_child(self.show_obstacle)
        self.panel.add_child(self.show_table)
        self.panel.add_child(self.show_grasps)
        self.panel.add_child(self.color_by_index)
        self.panel.add_child(self.grid_mode)
        self.panel.add_fixed(0.5 * em)
        self.panel.add_child(self.status)
        window.add_child(self.panel)

        window.set_on_layout(self._on_layout)

        # --- populate top-level dropdowns ---
        hands = [h for h in HAND_URDFS if os.path.isdir(os.path.join(CANDIDATES_ROOT, h))]
        for h in hands:
            self.hand_cb.add_item(h)
        if init.hand in hands:
            self.hand_cb.selected_text = init.hand
        self._init = init
        self._first = True
        self._on_hand()

    # ---- layout ----
    def _on_layout(self, ctx):
        r = self.window.content_rect
        panel_w = 22 * self.window.theme.font_size
        self.scene_widget.frame = gui.Rect(r.x, r.y, r.width - panel_w, r.height)
        self.panel.frame = gui.Rect(r.get_right() - panel_w, r.y, panel_w, r.height)

    # ---- cascading dropdown handlers ----
    def _cur(self, cb):
        return cb.selected_text

    def _on_hand(self):
        hand = self._cur(self.hand_cb)
        self.version_cb.clear_items()
        versions = list_dirs(os.path.join(CANDIDATES_ROOT, hand))
        for v in versions:
            self.version_cb.add_item(v)
        if getattr(self, "_first", False) and self._init.version in versions:
            self.version_cb.selected_text = self._init.version
        self._on_version()

    def _on_version(self):
        hand, version = self._cur(self.hand_cb), self._cur(self.version_cb)
        self.obj_cb.clear_items()
        objs = list_dirs(os.path.join(CANDIDATES_ROOT, hand, version))
        for o in objs:
            self.obj_cb.add_item(o)
        if getattr(self, "_first", False) and self._init.obj in objs:
            self.obj_cb.selected_text = self._init.obj
        self._on_object()

    def _on_object(self):
        hand, version, obj = self._cur(self.hand_cb), self._cur(self.version_cb), self._cur(self.obj_cb)
        self.scene_type_cb.clear_items()
        # scene types that have both candidates AND a scene JSON dir,
        # excluding stale "*_prev" geometry (see project_stale_shelf_candidates).
        cand_types = [t for t in list_dirs(os.path.join(CANDIDATES_ROOT, hand, version, obj))
                      if not t.endswith("_prev")]
        json_types = set(list_dirs(get_scene_dir(hand, obj)))
        types = [t for t in cand_types if t in json_types] or cand_types
        for t in types:
            self.scene_type_cb.add_item(t)
        self._on_scene_type()

    def _on_scene_type(self):
        hand, version, obj = self._cur(self.hand_cb), self._cur(self.version_cb), self._cur(self.obj_cb)
        st = self._cur(self.scene_type_cb)
        self.scene_id_cb.clear_items()
        ids = sort_numeric(list_dirs(os.path.join(CANDIDATES_ROOT, hand, version, obj, st)))
        for i in ids:
            self.scene_id_cb.add_item(i)
        self._on_scene_id()

    def _on_scene_id(self):
        self._first = False
        self._load_grasps()
        self._render(reset_camera=True)

    # ---- data ----
    def _scene_dir(self):
        return os.path.join(
            CANDIDATES_ROOT, self._cur(self.hand_cb), self._cur(self.version_cb),
            self._cur(self.obj_cb), self._cur(self.scene_type_cb), self._cur(self.scene_id_cb),
        )

    def _load_grasps(self):
        self.grasp_dirs = sort_numeric(list_dirs(self._scene_dir()))
        n = max(len(self.grasp_dirs), 1)
        self.num_slider.set_limits(1, n)
        # default to showing all grasps in the scene
        self.num_slider.int_value = n

    def _get_urdf(self, hand):
        if hand not in self._urdf_cache:
            self._urdf_cache[hand] = yourdfpy.URDF.load(
                HAND_URDFS[hand], build_scene_graph=True, load_meshes=True,
                build_collision_scene_graph=False,
            )
        return self._urdf_cache[hand]

    def _robot_trimesh(self, hand, joint_angles, world_T):
        urdf = self._get_urdf(hand)
        jn = list(urdf.actuated_joint_names)
        cfg = {n: float(joint_angles[i]) for i, n in enumerate(jn)}
        urdf.update_cfg(cfg)
        parts = []
        for name, geom in urdf.scene.geometry.items():
            T = np.asarray(urdf.scene.graph.get(name)[0])
            m = geom.copy()
            m.apply_transform(T)
            parts.append(m)
        combined = trimesh.util.concatenate(parts)
        combined.apply_transform(world_T)
        return combined

    def _load_object_mesh(self, obj):
        simp = os.path.join(obj_path, obj, "processed_data", "mesh", "simplified.obj")
        raw = os.path.join(obj_path, obj, "raw_mesh", f"{obj}.obj")
        path = simp if os.path.exists(simp) else raw
        return trimesh.load(path, force="mesh")

    # ---- rendering ----
    def _build_scene_geoms(self, hand, obj, st, sid, n_grasps, prefix):
        """Build (name, o3d_geom, material) list for one scene at the origin.

        Geometry is in world frame (object at its tabletop pose); the caller
        applies any grid offset. Returns (geoms, center) or (None, None)."""
        scene_json = os.path.join(get_scene_dir(hand, obj, st), f"{sid}.json")
        if not os.path.exists(scene_json):
            return None, None
        with open(scene_json) as f:
            cfg = json.load(f)
        obj_pose = cart2se3(cfg["scene"]["mesh"]["target"]["pose"])

        geoms = []

        mesh_w = self._load_object_mesh(obj).copy()
        mesh_w.apply_transform(obj_pose)
        mat_obj = rendering.MaterialRecord()
        mat_obj.shader = "defaultLit"
        geoms.append((f"{prefix}object", trimesh_to_o3d(mesh_w, COLOR_OBJECT), mat_obj))

        for name, info in cfg["scene"].get("cuboid", {}).items():
            is_table = name == "table"
            if is_table and not self.show_table.checked:
                continue
            if not is_table and not self.show_obstacle.checked:
                continue
            box = trimesh.creation.box(extents=info["dims"])
            box.apply_transform(cart2se3(info["pose"]))
            color = COLOR_TABLE if is_table else COLOR_OBSTACLE
            mat = rendering.MaterialRecord()
            if is_table:
                mat.shader = "defaultLit"
            else:
                mat.shader = "defaultLitTransparency"
                mat.base_color = [*color, OBSTACLE_OPACITY]
            geoms.append((f"{prefix}cuboid_{name}", trimesh_to_o3d(box, color), mat))

        if not self.show_grasps.checked:
            return geoms, obj_pose[:3, 3]

        grasp_dirs = sort_numeric(list_dirs(
            os.path.join(CANDIDATES_ROOT, hand, self._cur(self.version_cb), obj, st, sid)))
        sel = grasp_dirs[:n_grasps]
        colors = distinct_colors(len(sel)) if self.color_by_index.checked else None
        for i, gd in enumerate(sel):
            gp = os.path.join(CANDIDATES_ROOT, hand, self._cur(self.version_cb), obj, st, sid, gd)
            wrist_se3 = np.load(os.path.join(gp, "wrist_se3.npy"))
            grasp_pose = np.load(os.path.join(gp, "grasp_pose.npy")).flatten()
            rmesh = self._robot_trimesh(hand, grasp_pose, obj_pose @ wrist_se3)
            color = colors[i] if colors is not None else (0.6, 0.5, 0.85)
            mat = rendering.MaterialRecord()
            mat.shader = "defaultLit"
            geoms.append((f"{prefix}hand_{i}", trimesh_to_o3d(rmesh, color), mat))

        return geoms, obj_pose[:3, 3]

    def _aabb_union(self, geoms):
        bounds = geoms[0][1].get_axis_aligned_bounding_box()
        for _, g, _ in geoms[1:]:
            bounds += g.get_axis_aligned_bounding_box()
        return bounds

    def _clear_labels(self):
        for h in self._label_handles:
            self.scene_widget.remove_3d_label(h)
        self._label_handles = []

    def _render(self, reset_camera=False):
        scene = self.scene_widget.scene
        scene.clear_geometry()
        self._clear_labels()

        hand = self._cur(self.hand_cb)
        obj = self._cur(self.obj_cb)
        st = self._cur(self.scene_type_cb)
        if not all([hand, obj, st]):
            return

        n = int(self.num_slider.int_value)

        if self.grid_mode.checked:
            self._render_grid(hand, obj, st, n, reset_camera)
            return

        sid = self._cur(self.scene_id_cb)
        if not sid:
            return
        geoms, _ = self._build_scene_geoms(hand, obj, st, sid, n, "")
        if geoms is None:
            self.status.text = f"Scene JSON missing: {st}/{sid}"
            return
        for name, g, mat in geoms:
            scene.add_geometry(name, g, mat)

        self.status.text = (
            f"{obj} | {hand}/{self._cur(self.version_cb)}\n"
            f"{st}/{sid}  —  {min(n, len(self.grasp_dirs))}/{len(self.grasp_dirs)} grasps"
        )
        if reset_camera:
            bounds = self._aabb_union(geoms)
            self.scene_widget.setup_camera(60.0, bounds, bounds.get_center())

    def _render_grid(self, hand, obj, st, n, reset_camera):
        scene = self.scene_widget.scene
        sids = sort_numeric(list_dirs(os.path.join(CANDIDATES_ROOT, hand, self._cur(self.version_cb), obj, st)))
        if not sids:
            self.status.text = f"No scenes for {obj}/{st}"
            return

        # Build every tile at the origin first, then size the grid from the
        # largest tile's horizontal extent before applying offsets.
        tiles = []
        for sid in sids:
            geoms, center = self._build_scene_geoms(hand, obj, st, sid, n, f"{sid}_")
            if geoms is not None:
                tiles.append((sid, geoms, self._aabb_union(geoms)))
        if not tiles:
            self.status.text = f"No scene JSON for {obj}/{st}"
            return

        max_ext = 0.0
        for _, _, aabb in tiles:
            ext = aabb.get_extent()
            max_ext = max(max_ext, float(ext[0]), float(ext[1]))
        pitch = max_ext * self._pitch_mult

        cols = int(np.ceil(np.sqrt(len(tiles))))
        all_bounds = None
        for idx, (sid, geoms, aabb) in enumerate(tiles):
            r, c = idx // cols, idx % cols
            grid_pt = np.array([
                (c - (cols - 1) / 2) * pitch,
                -(r - (cols - 1) / 2) * pitch,
                0.0,
            ])
            # Center each tile's xy on its grid point (scenes differ in object
            # pose / wall placement), keep z so the table stays at the same level.
            center = aabb.get_center()
            off = grid_pt - np.array([center[0], center[1], 0.0])
            for name, g, mat in geoms:
                g.translate(off)
                scene.add_geometry(name, g, mat)
            h = self.scene_widget.add_3d_label(aabb.get_center() + off, sid)
            self._label_handles.append(h)
            b = self._aabb_union(geoms)
            if all_bounds is None:
                all_bounds = b
            else:
                all_bounds += b

        self.status.text = (
            f"{obj} | {hand}/{self._cur(self.version_cb)}\n"
            f"{st}  —  {len(tiles)} scenes (grid), {n} grasps each"
        )
        if reset_camera and all_bounds is not None:
            self.scene_widget.setup_camera(60.0, all_bounds, all_bounds.get_center())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", default="allegro", choices=list(HAND_URDFS))
    parser.add_argument("--version", default="v7")
    parser.add_argument("--obj", default="apple")
    parser.add_argument("--pitch-mult", type=float, default=1.05,
                        help="Grid spacing = (largest tile width) * pitch_mult (default 1.05)")
    args = parser.parse_args()

    gui.Application.instance.initialize()
    window = gui.Application.instance.create_window(
        "Scene + Grasp Viewer", 1500, 950
    )
    SceneGraspViewer(window, args)
    gui.Application.instance.run()


if __name__ == "__main__":
    main()
