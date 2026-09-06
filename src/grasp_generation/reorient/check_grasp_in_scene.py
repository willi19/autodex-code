"""Place each successful grasp from scene A into scene B and visualize
to see where the hand collides with scene B obstacles.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh
import viser
from scipy.spatial.transform import Rotation as R
import yourdfpy

from autodex.utils.path import get_obj_root

HAND_URDF = "/home/mingi/AutoDex/src/grasp_generation/BODex/src/curobo/content/assets/robot/inspire_description/inspire_hand_left.urdf"


def xyzquat_to_T(p):
    T = np.eye(4); T[:3, 3] = p[:3]
    T[:3, :3] = R.from_quat([p[4], p[5], p[6], p[3]]).as_matrix()
    return T


def add_scene(server, prefix, scene_json, mesh_rgba, cuboid_rgba):
    for name, entry in scene_json["scene"].get("mesh", {}).items():
        m = trimesh.load(entry["file_path"], process=False, force="mesh")
        rgba = np.array(mesh_rgba, dtype=np.uint8)
        m.visual.face_colors = np.tile(rgba, (len(m.faces), 1))
        pose = entry["pose"]
        server.scene.add_mesh_trimesh(f"{prefix}/mesh/{name}", m,
                                       position=np.array(pose[:3]),
                                       wxyz=np.array(pose[3:7]))
    for name, entry in scene_json["scene"].get("cuboid", {}).items():
        box = trimesh.creation.box(extents=entry["dims"])
        rgba = np.array(cuboid_rgba, dtype=np.uint8)
        box.visual.face_colors = np.tile(rgba, (len(box.faces), 1))
        pose = entry["pose"]
        server.scene.add_mesh_trimesh(f"{prefix}/cuboid/{name}", box,
                                       position=np.array(pose[:3]),
                                       wxyz=np.array(pose[3:7]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", required=True)
    parser.add_argument("--scene_type", default="reorient_8")
    parser.add_argument("--scene_grasps", required=True, help="e.g., 1_2 (where successful grasps came from)")
    parser.add_argument("--scene_view", required=True, help="e.g., 2_1 (scene to display)")
    parser.add_argument("--version", default="reset_8",
                        help="BODex reset experiment to inspect; suffix is release height in cm")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--obj_root", type=Path, default=None,
                        help="optional v8 object-root override")
    args = parser.parse_args()
    if not args.version.startswith("reset_"):
        parser.error("--version must be a reset_<height_mm> experiment")
    args.obj_root = args.obj_root or Path(get_obj_root("v8"))

    # Load grasps from scene_grasps that passed sim_filter
    bo_root = Path('/home/mingi/AutoDex/bodex_outputs/inspire_left')
    candidates = []
    sd = (bo_root / args.version / args.obj /
          f'reorient_{args.scene_type.split("_")[1]}' / args.scene_grasps)
    if sd.is_dir():
        for seed in sd.iterdir():
            f = seed / 'sim_eval.json'
            if f.exists():
                try:
                    if json.load(open(f)).get("success"):
                        candidates.append({
                            "ver": args.version, "seed": seed.name,
                            "wrist_se3": np.load(seed / 'wrist_se3.npy'),
                            "grasp": np.load(seed / 'grasp_pose.npy'),
                            "pregrasp": np.load(seed / 'pregrasp_pose.npy'),
                        })
                except Exception:
                    pass
    print(f"Loaded {len(candidates)} successful grasps from {args.scene_grasps}")
    if not candidates:
        return

    # Load scene_view JSON
    scene_view_json = json.load(open(args.obj_root / args.obj / "scene" / args.scene_type / f"{args.scene_view}.json"))
    obj_T_view = xyzquat_to_T(scene_view_json["scene"]["mesh"]["target"]["pose"])

    server = viser.ViserServer(port=args.port)
    server.scene.world_axes.visible = True

    # Display scene_view (object + obstacles)
    add_scene(server, "/scene_view", scene_view_json,
              mesh_rgba=(120, 180, 120, 200),       # green mesh (canonical view)
              cuboid_rgba=(220, 80, 80, 100))       # red obstacles

    # Load URDF
    urdf = yourdfpy.URDF.load(HAND_URDF)

    # GUI: slider over grasps
    with server.gui.add_folder("Grasp"):
        slider = server.gui.add_slider("Grasp idx", min=0, max=len(candidates)-1, step=1, initial_value=0)
        info = server.gui.add_text("Info", initial_value="", disabled=True)

    handles = []
    def render(k):
        # remove previous handles
        for h in handles: h.remove()
        handles.clear()
        g = candidates[k]
        # wrist in world = obj_T_view @ wrist_se3 (object-frame -> world)
        wrist_world = obj_T_view @ g["wrist_se3"]
        # Set joint angles
        cfg = g["grasp"]  # 6 dof for inspire_left (or 12 with mimic?). Use whatever URDF expects
        joint_names = list(urdf.actuated_joint_names)
        if len(cfg) == len(joint_names):
            urdf.update_cfg({n: float(cfg[i]) for i, n in enumerate(joint_names)})
        else:
            # try first N
            try:
                urdf.update_cfg({n: float(cfg[i]) for i, n in enumerate(joint_names[:len(cfg)])})
            except Exception:
                pass
        # Iterate URDF links, add each mesh
        for link_name, link in urdf.link_map.items():
            T_link = urdf.get_transform(link_name)
            T_world = wrist_world @ T_link
            for vis in link.visuals:
                if vis.geometry.mesh is None: continue
                mesh_file = vis.geometry.mesh.filename
                if not Path(mesh_file).is_absolute():
                    mesh_file = str(Path(HAND_URDF).parent / mesh_file)
                try:
                    m = trimesh.load(mesh_file, process=False, force="mesh")
                except Exception:
                    continue
                rgba = np.array([80, 80, 200, 200], dtype=np.uint8)
                m.visual.face_colors = np.tile(rgba, (len(m.faces), 1))
                q_xyzw = R.from_matrix(T_world[:3, :3]).as_quat()
                wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
                h = server.scene.add_mesh_trimesh(
                    f"/hand/{link_name}/{Path(mesh_file).stem}", m,
                    position=T_world[:3, 3], wxyz=wxyz,
                )
                handles.append(h)
        info.value = f"{g['ver']}/seed={g['seed']}"

    render(0)
    @slider.on_update
    def _(_): render(int(slider.value))

    print(f"[viz] http://localhost:{args.port}")
    print(f"      mesh(green) = {args.obj} at {args.scene_view}'s pose")
    print(f"      obstacles(red) = scene {args.scene_view}'s table_i, table_j, pillars")
    print(f"      hand(blue) = grasps from {args.scene_grasps} transferred to view {args.scene_view}")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
