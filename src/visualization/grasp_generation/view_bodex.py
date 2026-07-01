"""Browse BODex / candidate grasp outputs interactively.

Usage:
    python src/visualization/grasp_generation/view_bodex.py
"""

import os
import json
import argparse
import numpy as np
import trimesh
import yaml
from scipy.spatial.transform import Rotation as Rot
from shapely import geometry as geom

from paradex.visualization.visualizer.viser import ViserViewer
from autodex.utils.path import obj_path as DEFAULT_OBJ_PATH

obj_path = DEFAULT_OBJ_PATH  # rebound from CLI in __main__
from autodex.utils.conversion import cart2se3

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BODEX_OUTPUT_ROOT = os.path.join(REPO_ROOT, "bodex_outputs")
CANDIDATES_ROOT = os.path.join(REPO_ROOT, "candidates")

SOURCE_ROOTS = {
    "bodex_outputs": BODEX_OUTPUT_ROOT,
    "candidates": CANDIDATES_ROOT,
}

SCENE_RGBA = {
    "table":    [240, 240, 245, 230],   # near-opaque white
    "obstacle": [119, 136, 153, 77],    # semi-transparent slate gray
}

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

HAND_SPHERES = {
    "allegro": os.path.join(REPO_ROOT, "src", "grasp_generation", "BODex", "src",
                             "curobo", "content", "configs", "robot", "spheres", "allegro.yml"),
    "inspire": os.path.join(REPO_ROOT, "src", "grasp_generation", "BODex", "src",
                             "curobo", "content", "configs", "robot", "spheres", "inspire.yml"),
    "inspire_left": os.path.join(REPO_ROOT, "src", "grasp_generation", "BODex", "src",
                                  "curobo", "content", "configs", "robot", "spheres", "inspire.yml"),
    "inspire_f1": os.path.join(REPO_ROOT, "src", "grasp_generation", "BODex", "src",
                                "curobo", "content", "configs", "robot", "spheres", "inspire_f1.yml"),
    "inspire_f1_left": os.path.join(REPO_ROOT, "src", "grasp_generation", "BODex", "src",
                                     "curobo", "content", "configs", "robot", "spheres", "inspire_f1_left.yml"),
}

THICK = 0.02
VIZ_MARGIN = 0.01  # extra gap so procedural walls don't clip the object mesh

# --- procedural scene builders (from Visualization/paper/overview/recorder_box.py) ---

def _mat4_to_pose(T):
    """SE3 -> [x, y, z, qw, qx, qy, qz]"""
    q = Rot.from_matrix(T[:3, :3]).as_quat()  # xyzw
    return [T[0, 3], T[1, 3], T[2, 3], q[3], q[0], q[1], q[2]]


def _rotz(theta):
    c, s = np.cos(theta), np.sin(theta)
    R = np.eye(4)
    R[0, 0] = c; R[0, 1] = -s
    R[1, 0] = s; R[1, 1] = c
    return R


def _build_wall_scene(obj_name, tabletop_pose, meta):
    """Single wall behind object."""
    gap = meta.get("gap", 0.03) + VIZ_MARGIN
    # tabletop_pose already includes z_rotation_deg from BODex scene generation.
    pose = tabletop_pose

    # OBB
    obb_path = os.path.join(obj_path, obj_name, "processed_data", "info", "simplified.json")
    with open(obb_path) as f:
        obb_info = json.load(f)
    obb_tf = np.array(obb_info["obb_transform"])
    ext = np.array(obb_info["obb"]) / 2.0
    axes = pose[:3, :3] @ obb_tf[:3, :3]
    tw = pose[:3, 3]

    corners = np.array([axes @ (np.array([sx, sy, sz]) * ext) + tw
                        for sx in [-1, 1] for sy in [-1, 1] for sz in [-1, 1]])

    min_y = corners[:, 1].min()
    max_z = corners[:, 2].max()
    wall_h = max_z + 0.1
    wall_y = min_y - gap

    cuboids = {
        "table": {"dims": [6.0, 6.0, 0.2], "pose": [0, 0, -0.1, 1, 0, 0, 0]},
        "wall": {"dims": [0.3, THICK, wall_h], "pose": [0, wall_y - THICK/2, wall_h/2, 1, 0, 0, 0]},
    }
    return pose, cuboids


def _build_box_scene(obj_name, tabletop_pose, meta):
    """Tight-fitting box around object."""
    height_offset = meta.get("height_offset", 0.1)

    mesh_path = os.path.join(obj_path, obj_name, "processed_data", "mesh", "simplified.obj")
    mesh = trimesh.load(mesh_path, force="mesh")
    verts_w = (tabletop_pose[:3, :3] @ mesh.vertices.T).T + tabletop_pose[:3, 3]

    # XY minimum rotated rectangle
    poly = geom.MultiPoint(verts_w[:, :2]).convex_hull
    rect = poly.minimum_rotated_rectangle
    rect_pts = np.array(rect.exterior.coords)[:4]
    cx, cy = rect.centroid.coords[0]
    edge = rect_pts[1] - rect_pts[0]
    yaw = np.arctan2(edge[1], edge[0])
    width = np.linalg.norm(rect_pts[1] - rect_pts[0]) + 2 * VIZ_MARGIN
    depth = np.linalg.norm(rect_pts[2] - rect_pts[1]) + 2 * VIZ_MARGIN

    max_z = verts_w[:, 2].max()
    wall_h = max(max_z - height_offset, 0.05)
    hw, hd = width / 2, depth / 2

    T_box = np.eye(4)
    T_box[:3, :3] = _rotz(yaw)[:3, :3]
    T_box[:3, 3] = [cx, cy, wall_h / 2]

    def wall_pose(lx, ly):
        T = np.eye(4); T[:3, 3] = [lx, ly, 0]
        return _mat4_to_pose(T_box @ T)

    cuboids = {
        "table": {"dims": [6.0, 6.0, 0.2], "pose": [0, 0, -0.1, 1, 0, 0, 0]},
        "box_front": {"dims": [THICK, depth, wall_h], "pose": wall_pose(hw + THICK/2, 0)},
        "box_back":  {"dims": [THICK, depth, wall_h], "pose": wall_pose(-hw - THICK/2, 0)},
        "box_right": {"dims": [width + 2*THICK, THICK, wall_h], "pose": wall_pose(0, hd + THICK/2)},
        "box_left":  {"dims": [width + 2*THICK, THICK, wall_h], "pose": wall_pose(0, -hd - THICK/2)},
    }
    return tabletop_pose, cuboids


def _build_shelf_scene(obj_name, tabletop_pose, meta):
    """Shelf with back/sides/top walls."""
    gap = meta.get("gap", 0.03)
    up = meta.get("up", True)
    side = meta.get("side", True)
    back = meta.get("back", True)
    # tabletop_pose already includes z_rotation_deg from BODex scene generation.
    pose = tabletop_pose

    obb_path = os.path.join(obj_path, obj_name, "processed_data", "info", "simplified.json")
    with open(obb_path) as f:
        obb_info = json.load(f)
    obb_tf = np.array(obb_info["obb_transform"])
    ext = np.array(obb_info["obb"]) / 2.0
    axes = pose[:3, :3] @ obb_tf[:3, :3]
    tw = pose[:3, 3]

    corners = np.array([axes @ (np.array([sx, sy, sz]) * ext) + tw
                        for sx in [-1, 1] for sy in [-1, 1] for sz in [-1, 1]])

    min_x, max_x = corners[:, 0].min(), corners[:, 0].max()
    min_y, max_y = corners[:, 1].min(), corners[:, 1].max()
    max_z = corners[:, 2].max()

    inner_y_back = min_y - gap
    inner_y_front = max_y + gap
    inner_x_min = min_x - gap
    inner_x_max = max_x + gap
    inner_z_top = max_z + gap
    wall_h = inner_z_top
    full_width = (inner_x_max - inner_x_min) + 2 * THICK
    full_depth = (inner_y_front - inner_y_back) + THICK

    cuboids = {"table": {"dims": [6.0, 6.0, 0.2], "pose": [0, 0, -0.1, 1, 0, 0, 0]}}

    if back:
        cuboids["back"] = {
            "dims": [full_width, THICK, wall_h],
            "pose": [(inner_x_min + inner_x_max) / 2, inner_y_back - THICK/2, wall_h/2, 1, 0, 0, 0],
        }
    if side:
        cuboids["side_pos"] = {
            "dims": [THICK, full_depth - THICK, wall_h],
            "pose": [inner_x_max + THICK/2, (inner_y_back - THICK/2 + inner_y_front)/2, wall_h/2, 1, 0, 0, 0],
        }
        cuboids["side_neg"] = {
            "dims": [THICK, full_depth - THICK, wall_h],
            "pose": [inner_x_min - THICK/2, (inner_y_back - THICK/2 + inner_y_front)/2, wall_h/2, 1, 0, 0, 0],
        }
    if up:
        cuboids["up"] = {
            "dims": [full_width, full_depth, THICK],
            "pose": [(inner_x_min + inner_x_max) / 2, (inner_y_back - THICK + inner_y_front)/2, inner_z_top + THICK/2, 1, 0, 0, 0],
        }

    return pose, cuboids


SCENE_BUILDERS = {
    "wall": _build_wall_scene,
    "box": _build_box_scene,
    "shelf": _build_shelf_scene,
}


class BODexBrowser(ViserViewer):
    def __init__(self):
        super().__init__()

        self.obj_root = obj_path
        self.obj_pose = np.eye(4)

        self.current_source = None
        self.current_hand = None
        self.current_version = None
        self.current_obj = None
        self.current_scene_type = None
        self.current_scene_idx = None

        self.all_grasp_dirs = []
        self.gui_playing.value = True

        source_list = sorted(s for s in SOURCE_ROOTS if os.path.isdir(SOURCE_ROOTS[s]))

        with self.server.gui.add_folder("Grasp Viewer"):
            self.source_selector = self.server.gui.add_dropdown(
                "Source", options=source_list,
                initial_value=source_list[0] if source_list else "",
            )
            self.hand_selector = self.server.gui.add_dropdown(
                "Hand", options=[], initial_value="",
            )
            self.version_selector = self.server.gui.add_dropdown(
                "Version", options=[], initial_value="",
            )
            self.obj_selector = self.server.gui.add_dropdown(
                "Object", options=[], initial_value="",
            )
            self.scene_type_selector = self.server.gui.add_dropdown(
                "Scene Type", options=[], initial_value="",
            )
            self.scene_idx_selector = self.server.gui.add_dropdown(
                "Scene", options=[], initial_value="",
            )
            self.grasp_idx_slider = self.server.gui.add_slider(
                "Grasp Index", min=0, max=1, step=1, initial_value=0,
            )
            self.show_trajectory = self.server.gui.add_checkbox(
                "Show Trajectory", initial_value=False,
            )
            self.show_spheres = self.server.gui.add_checkbox(
                "Show Collision Spheres", initial_value=False,
            )
            self.metric_text = self.server.gui.add_text(
                "Metrics", initial_value="No grasp loaded", disabled=True,
            )

        self._on_source_change()

        @self.source_selector.on_update
        def _(event):
            self._on_source_change()

        @self.hand_selector.on_update
        def _(event):
            self._on_hand_change()

        @self.version_selector.on_update
        def _(event):
            self._on_version_change()

        @self.obj_selector.on_update
        def _(event):
            self._on_object_change()

        @self.scene_type_selector.on_update
        def _(event):
            self._on_scene_type_change()

        @self.scene_idx_selector.on_update
        def _(event):
            self._on_scene_idx_change()

        @self.grasp_idx_slider.on_update
        def _(event):
            self._load_current_grasp()

        @self.show_trajectory.on_update
        def _(event):
            self._load_current_grasp()

        @self.show_spheres.on_update
        def _(event):
            self._update_spheres()

        self.squeeze_num = 10

        # Pre-load all hand URDFs (hidden initially)
        for hand_name, urdf in HAND_URDFS.items():
            if os.path.exists(urdf):
                self.add_robot(hand_name, urdf)
                self.robot_dict[hand_name].set_visibility(False)

    # --- helpers ---

    def _list_dirs(self, path):
        if not os.path.isdir(path):
            return []
        return sorted(d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d)))

    def _source_root(self):
        return SOURCE_ROOTS.get(self.current_source, BODEX_OUTPUT_ROOT)

    def _data_root(self):
        """Root for current source/hand/version: {source}/{hand}/{version}"""
        if not all([self.current_source, self.current_hand, self.current_version]):
            return ""
        return os.path.join(self._source_root(), self.current_hand, self.current_version)

    def _get_hand_urdf(self):
        return HAND_URDFS.get(self.current_hand, HAND_URDFS["allegro"])

    # --- clear ---

    def clear_scene(self):
        for name in list(self.obj_dict.keys()):
            self.obj_dict[name]["frame"].remove()
            self.obj_dict[name]["handle"].remove()
            del self.obj_dict[name]
        # Remove cuboid handles added via add_mesh_simple
        for h in getattr(self, "_cuboid_handles", []):
            h.remove()
        self._cuboid_handles = []

    # --- cascading updates ---

    def _on_source_change(self):
        self.current_source = self.source_selector.value
        hands = self._list_dirs(self._source_root())
        self.hand_selector.options = hands if hands else ["(none)"]
        if hands:
            self.hand_selector.value = hands[0]
        self._on_hand_change()

    def _on_hand_change(self):
        self.current_hand = self.hand_selector.value
        hand_path = os.path.join(self._source_root(), self.current_hand) if self.current_hand else ""
        versions = self._list_dirs(hand_path)
        self.version_selector.options = versions if versions else ["(none)"]
        if versions:
            self.version_selector.value = versions[0]
        self._on_version_change()

    def _on_version_change(self):
        self.current_version = self.version_selector.value
        objs = self._list_dirs(self._data_root())
        self.obj_selector.options = objs if objs else ["(none)"]
        if objs:
            self.obj_selector.value = objs[0]
            self._on_object_change()

    def _on_object_change(self):
        self.current_obj = self.obj_selector.value
        # Scene types from candidate data (what scenes have grasps)
        candidate_scene_types = self._list_dirs(os.path.join(self._data_root(), self.current_obj))
        # Also check object scene data for JSON availability
        obj_scene_types = self._list_dirs(os.path.join(self.obj_root, self.current_obj, "scene"))
        # Use candidate scene types (what actually has data), but only if scene JSON exists
        scene_types = [s for s in candidate_scene_types if s in obj_scene_types]
        if not scene_types:
            scene_types = candidate_scene_types  # fallback: show even without scene json
        self.scene_type_selector.options = scene_types if scene_types else ["(none)"]
        if scene_types:
            self.scene_type_selector.value = scene_types[0]
            self._on_scene_type_change()

    def _on_scene_type_change(self):
        self.current_scene_type = self.scene_type_selector.value
        if not self.current_scene_type or self.current_scene_type == "(none)":
            return
        # Scene IDs from candidate data
        scene_dir = os.path.join(self._data_root(), self.current_obj, self.current_scene_type)
        scenes = self._list_dirs(scene_dir)
        scenes = sorted(scenes, key=lambda x: (0, int(x)) if x.isdigit() else (1, x))
        self.scene_idx_selector.options = scenes if scenes else ["(none)"]
        if scenes:
            self.scene_idx_selector.value = scenes[0]
            self._on_scene_idx_change()

    def _on_scene_idx_change(self):
        self.current_scene_idx = self.scene_idx_selector.value
        if not self.current_scene_idx or self.current_scene_idx == "(none)":
            return

        self._load_scene()

        # Collect grasp dirs for current hand/version/obj/scene
        scene_path = os.path.join(
            self._data_root(), self.current_obj,
            self.current_scene_type, self.current_scene_idx,
        )
        self.all_grasp_dirs = self._list_dirs(scene_path)
        self.all_grasp_dirs = sorted(
            self.all_grasp_dirs,
            key=lambda x: (0, int(x)) if x.isdigit() else (1, x),
        )

        if not self.all_grasp_dirs:
            self.grasp_idx_slider.disabled = True
            return

        self.grasp_idx_slider.disabled = False
        self.grasp_idx_slider.max = len(self.all_grasp_dirs) - 1
        self.grasp_idx_slider.value = 0
        self._load_current_grasp()

    # --- scene loading ---

    def _load_scene(self):
        self.clear_scene()
        scene_json_path = os.path.join(
            self.obj_root, self.current_obj, "scene",
            self.current_scene_type, f"{self.current_scene_idx}.json",
        )

        if not os.path.exists(scene_json_path):
            print(f"[Warning] Scene json not found: {scene_json_path}")
            return

        with open(scene_json_path, "r") as f:
            cfg = json.load(f)

        scene = cfg["scene"]
        meta = cfg.get("meta", {}).get("param", {})

        # Get tabletop pose from scene JSON
        target_info = scene["mesh"]["target"]
        tabletop_pose = cart2se3(target_info["pose"])

        # Use scene JSON directly to match what sim/cuRobo evaluate.
        obj_pose = tabletop_pose
        cuboids = scene.get("cuboid", {})

        self.obj_pose = obj_pose

        # Load object mesh — prefer simplified (small) over raw_mesh (can be 100MB+).
        simplified_mesh_path = os.path.join(
            self.obj_root, self.current_obj, "processed_data", "mesh", "simplified.obj"
        )
        raw_mesh_path = os.path.join(
            self.obj_root, self.current_obj, "raw_mesh", f"{self.current_obj}.obj"
        )
        if os.path.exists(simplified_mesh_path):
            mesh = trimesh.load(simplified_mesh_path, force="mesh")
        elif os.path.exists(raw_mesh_path):
            mesh = trimesh.load(raw_mesh_path, process=False)
        else:
            mesh = trimesh.load(target_info["file_path"], force="mesh")
        self.add_object("target", mesh, obj_T=obj_pose)

        # Render cuboids via add_mesh_simple (supports opacity)
        self._cuboid_handles = []
        for name, info in cuboids.items():
            box = trimesh.creation.box(extents=info["dims"])
            pose_arr = np.asarray(info["pose"])
            if len(pose_arr) == 7:
                cpose = np.eye(4)
                cpose[:3, 3] = pose_arr[:3]
                cquat = pose_arr[3:]  # wxyz
                cpose[:3, :3] = Rot.from_quat([cquat[1], cquat[2], cquat[3], cquat[0]]).as_matrix()
            else:
                cpose = cart2se3(pose_arr)
            rgba = SCENE_RGBA["table"] if name == "table" else SCENE_RGBA["obstacle"]
            handle = self.server.scene.add_mesh_simple(
                name=f"/objects/{name}",
                vertices=np.array(box.vertices, dtype=np.float32),
                faces=np.array(box.faces, dtype=np.uint32),
                color=rgba[:3],
                opacity=rgba[3] / 255.0,
                flat_shading=True,
                side="double",
                wxyz=Rot.from_matrix(cpose[:3, :3]).as_quat()[[3, 0, 1, 2]],
                position=cpose[:3, 3],
            )
            self._cuboid_handles.append(handle)

    # --- grasp loading ---

    def _update_robot_pose(self, hand_name, pose):
        """Update pose of a pre-loaded robot."""
        robot = self.robot_dict.get(hand_name)
        if robot is None:
            return
        robot._visual_root_frame.position = pose[:3, 3]
        robot._visual_root_frame.wxyz = Rot.from_matrix(pose[:3, :3]).as_quat()[[3, 0, 1, 2]]
        with open("/tmp/view_bodex_debug.log", "a") as f:
            f.write(f"[{hand_name}] obj={getattr(self,'current_obj','?')} scene={getattr(self,'current_scene_type','?')}/{getattr(self,'current_scene_idx','?')} grasp={self.grasp_idx_slider.value if hasattr(self,'grasp_idx_slider') else '?'}\n")
            f.write(f"  root_frame.position = {pose[:3,3].tolist()}\n")
            f.write(f"  root_frame.wxyz     = {robot._visual_root_frame.wxyz.tolist()}\n")
            f.write(f"  full pose:\n{pose}\n")
            f.write(f"  obj_pose:\n{self.obj_pose}\n\n")

    def _load_current_grasp(self):
        if not self.all_grasp_dirs:
            return

        self.clear_traj()

        grasp_idx = self.grasp_idx_slider.value
        grasp_dir = self.all_grasp_dirs[grasp_idx]
        grasp_path = os.path.join(
            self._data_root(), self.current_obj,
            self.current_scene_type, self.current_scene_idx, grasp_dir,
        )

        if not os.path.isdir(grasp_path):
            return

        wrist_se3 = np.load(os.path.join(grasp_path, "wrist_se3.npy"))
        pregrasp_pose = np.load(os.path.join(grasp_path, "pregrasp_pose.npy"))
        grasp_pose = np.load(os.path.join(grasp_path, "grasp_pose.npy"))

        robot_T = self.obj_pose @ wrist_se3

        # Show only current hand, hide others
        for hand_name in HAND_URDFS:
            if hand_name in self.robot_dict:
                if hand_name == self.current_hand:
                    self._update_robot_pose(hand_name, robot_T)
                    self.robot_dict[hand_name].set_visibility(True)
                else:
                    self.robot_dict[hand_name].set_visibility(False)

        if self.show_trajectory.value:
            traj_list = [pregrasp_pose, grasp_pose]
            for i in range(self.squeeze_num):
                traj_list.append(grasp_pose * (i + 2) - pregrasp_pose * (i + 1))
            traj_dict = {self.current_hand: np.stack(traj_list)}
            self.add_traj("robot_traj", traj_dict)
        else:
            traj_dict = {self.current_hand: np.stack([grasp_pose])}
            self.add_traj("robot_traj", traj_dict)

        # Metrics
        info = f"{self.current_hand} | {self.current_version} | Grasp {grasp_dir}"
        bodex_info_path = os.path.join(grasp_path, "bodex_info.npy")
        result_json_path = os.path.join(grasp_path, "result.json")
        if os.path.exists(bodex_info_path):
            bi = np.load(bodex_info_path, allow_pickle=True).item()
            ge = bi.get("grasp_error", np.array([0]))
            de = bi.get("dist_error", np.array([0]))
            info += f" | grasp_err={ge.max():.3f} | dist_err={de.max():.3f}"
        elif os.path.exists(result_json_path):
            with open(result_json_path, "r") as f:
                result = json.load(f)
            info += f" | result={result.get('result', 'N/A')}"
        self.metric_text.value = info

        self._cached_grasp_pose = grasp_pose
        self._cached_robot_T = robot_T
        self._update_spheres()

    def _update_spheres(self):
        # Remove existing
        if not hasattr(self, "_sphere_handles"):
            self._sphere_handles = []
        for h in self._sphere_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._sphere_handles = []

        if not self.show_spheres.value:
            return
        if not hasattr(self, "_cached_grasp_pose"):
            return

        sphere_yml = HAND_SPHERES.get(self.current_hand)
        if not sphere_yml or not os.path.exists(sphere_yml):
            return
        with open(sphere_yml) as f:
            sd = yaml.safe_load(f)
        sd = sd.get("collision_spheres", sd)

        robot = self.robot_dict.get(self.current_hand)
        if robot is None:
            return
        # Apply grasp_pose to URDF FK
        urdf_mod = robot._urdf if hasattr(robot, "_urdf") else robot
        urdf = urdf_mod.urdf if hasattr(urdf_mod, "urdf") else urdf_mod
        joint_names = [j.name for j in urdf.actuated_joints]
        cfg = {n: float(v) for n, v in zip(joint_names, self._cached_grasp_pose)}
        urdf.update_cfg(cfg)

        base_link = urdf.base_link
        robot_T = self._cached_robot_T

        link_colors_list = [
            (255, 50, 50), (50, 255, 50), (50, 50, 255),
            (255, 255, 50), (255, 50, 255), (50, 255, 255),
            (255, 150, 50), (150, 50, 255), (50, 150, 50),
            (200, 100, 100), (100, 200, 100), (100, 100, 200),
        ]

        idx = 0
        for li, (link_name, spheres) in enumerate(sd.items()):
            if not spheres:
                continue
            try:
                T_link = urdf.get_transform(link_name, base_link)
            except Exception:
                continue
            T_world = robot_T @ T_link
            color = link_colors_list[li % len(link_colors_list)]
            for sp in spheres:
                center_local = np.array(sp["center"])
                radius = float(sp["radius"])
                center_world = T_world[:3, :3] @ center_local + T_world[:3, 3]
                h = self.server.scene.add_icosphere(
                    f"/spheres/{idx}",
                    radius=radius,
                    color=color,
                    position=center_world,
                    opacity=0.5,
                )
                self._sphere_handles.append(h)
                idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--obj_path", default=DEFAULT_OBJ_PATH,
                        help="Object root dir (default: paradex from autodex.utils.path)")
    args = parser.parse_args()

    globals()["obj_path"] = args.obj_path

    vis = BODexBrowser()
    vis.start_viewer()