"""Dump mujoco's actual hand pose for one seed (headless), export to obj for visual diff.

Usage:
    python src/grasp_generation/sim_filter/dump_mujoco_pose.py \
        --hand inspire_f1 --obj Jp_Water --scene shelf --scene_id 37 --seed 0 \
        --obj_root /home/mingi/shared_data/AutoDex/object/robothome --port 8081
"""
import os
import sys
import argparse
import numpy as np
import mujoco
import trimesh
import time

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from autodex.simulator.hand_object import MjHO
from autodex.utils.conversion import se32cart, cart2se3
from autodex.utils.path import get_scene_dir

# Inline copies to avoid importing run_sim_filter (which pulls cuRobo and uses GPU).
HAND_PATHS = {
    "allegro": {
        "path": os.path.join(REPO_ROOT, "src", "grasp_generation", "sim_filter", "assets", "hand", "allegro", "right_hand.xml"),
        "weld_body": "world",
    },
    "inspire": {
        "path": os.path.join(REPO_ROOT, "src", "grasp_generation", "BODex", "src", "curobo", "content", "assets", "robot", "inspire_description", "inspire_hand_right.urdf"),
        "weld_body": "wrist",
    },
    "inspire_left": {
        "path": os.path.join(REPO_ROOT, "src", "grasp_generation", "BODex", "src", "curobo", "content", "assets", "robot", "inspire_description", "inspire_hand_left.urdf"),
        "weld_body": "wrist",
    },
    "inspire_f1": {
        "path": os.path.join(REPO_ROOT, "src", "grasp_generation", "BODex", "src", "curobo", "content", "assets", "robot", "inspire_f1_description", "inspire_f1_hand_right.urdf"),
        "weld_body": "base_link",
    },
}

INSPIRE_MIMIC_MAP = [
    None, None, (1, 0.60, 0), (1, 0.80, 0),
    None, (2, 1.05, 0), None, (3, 1.05, 0),
    None, (4, 1.05, 0), None, (5, 1.18, 0),
]
INSPIRE_F1_MIMIC_MAP = [
    None, None, (1, 1.2953, 0), (1, 1.1610, 0),
    None, (2, 1.1545, 0), None, (3, 1.1545, 0),
    None, (4, 1.1545, 0), None, (5, 1.1545, 0),
]


def _expand_mimic_joints(joints, mimic_map):
    if mimic_map is None:
        return joints
    expanded = []
    act_idx = 0
    for entry in mimic_map:
        if entry is None:
            expanded.append(joints[act_idx]); act_idx += 1
        else:
            parent_idx, mult, offset = entry
            expanded.append(joints[parent_idx] * mult + offset)
    return np.array(expanded)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="inspire_f1")
    ap.add_argument("--obj", required=True)
    ap.add_argument("--scene", default="shelf")
    ap.add_argument("--scene_id", default="37")
    ap.add_argument("--seed", default="0")
    ap.add_argument("--version", default="v3")
    ap.add_argument("--obj_root", default="/home/mingi/shared_data/AutoDex/object/robothome")
    ap.add_argument("--source", default="bodex_outputs", choices=["bodex_outputs", "candidates"],
                    help="Where to load seed from")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--phase", default="pregrasp", choices=["pregrasp", "grasp"])
    args = ap.parse_args()

    seed_dir = os.path.join(REPO_ROOT, args.source, args.hand, args.version,
                             args.obj, args.scene, args.scene_id, args.seed)
    print(f"seed_dir: {seed_dir}")

    # Load BODex grasp data
    wrist_local = np.load(os.path.join(seed_dir, "wrist_se3.npy"))
    pregrasp = np.load(os.path.join(seed_dir, "pregrasp_pose.npy"))
    grasp_active = np.load(os.path.join(seed_dir, "grasp_pose.npy"))

    # Apply scene object pose
    scene_json = os.path.join(get_scene_dir(args.hand, args.obj, args.scene), f"{args.scene_id}.json")
    import json
    scene_cfg = json.load(open(scene_json))["scene"]
    obj_se3 = cart2se3(scene_cfg["mesh"]["target"]["pose"])
    wrist_world = obj_se3 @ wrist_local

    if args.hand == "inspire_f1":
        mimic_map = INSPIRE_F1_MIMIC_MAP
    elif args.hand in ("inspire", "inspire_left"):
        mimic_map = INSPIRE_MIMIC_MAP
    else:
        mimic_map = None

    pregrasp_full = _expand_mimic_joints(pregrasp, mimic_map) if mimic_map else pregrasp
    grasp_full = _expand_mimic_joints(grasp_active, mimic_map) if mimic_map else grasp_active

    # Init mujoco
    hand_cfg = HAND_PATHS[args.hand]
    mj = MjHO(args.obj, hand_cfg["path"], weld_body_name=hand_cfg["weld_body"],
              obj_mass=0.1, debug_viewer=False, obj_root_dir=args.obj_root)

    # Build qpos: wrist_freejoint (xyz+quat wxyz) + finger12 + obj_freejoint (xyz+quat)
    wrist_cart = se32cart(wrist_world)  # [xyz + quat wxyz]
    finger = pregrasp_full if args.phase == "pregrasp" else grasp_full
    qpos_hand = np.concatenate([wrist_cart, finger])
    obj_pose7 = np.array([obj_se3[0, 3], obj_se3[1, 3], obj_se3[2, 3], 1, 0, 0, 0])
    # Use object identity quat then set proper. Easier: use cart of obj_se3
    from scipy.spatial.transform import Rotation as Rot
    obj_quat = Rot.from_matrix(obj_se3[:3, :3]).as_quat()[[3, 0, 1, 2]]  # wxyz
    obj_pose7 = np.concatenate([obj_se3[:3, 3], obj_quat])

    mj.reset_pose_qpos(qpos_hand, obj_pose7)
    mujoco.mj_forward(mj.model, mj.data)

    # Dump every body world pose
    print(f"\n=== Mujoco link poses ({args.phase}) ===")
    bodies = []
    for i in range(mj.model.nbody):
        b = mj.model.body(i)
        if mj.hand_prefix not in b.name:
            continue
        name = b.name.replace(mj.hand_prefix, "")
        pos = mj.data.xpos[i].copy()
        rotmat = mj.data.xmat[i].reshape(3, 3).copy()
        bodies.append((name, pos, rotmat))
        print(f"  {name:30s} pos={pos}")
        if name == "base_link":
            print(f"    base_link rotmat:\n{rotmat}")

    # Build a combined trimesh by transforming each link's STL by its world pose
    urdf_dir = os.path.dirname(hand_cfg["path"])
    mesh_lookup = {}
    for root, _, files in os.walk(urdf_dir):
        for f in files:
            if f.lower().endswith((".obj", ".stl")):
                mesh_lookup.setdefault(f, os.path.join(root, f))

    # Map link name to STL filename. Most are <name>.STL
    combined_meshes = []
    for name, pos, rotmat in bodies:
        # Try common file names
        candidates = [f"{name}.STL", f"{name}.stl", f"{name}.obj"]
        stl_path = None
        for c in candidates:
            if c in mesh_lookup:
                stl_path = mesh_lookup[c]
                break
        if stl_path is None:
            continue
        try:
            m = trimesh.load(stl_path, force="mesh")
        except Exception:
            continue
        T = np.eye(4)
        T[:3, :3] = rotmat
        T[:3, 3] = pos
        m = m.copy()
        m.apply_transform(T)
        combined_meshes.append(m)
        print(f"  loaded mesh: {os.path.basename(stl_path)} for {name}")

    # Export combined
    if combined_meshes:
        combined = trimesh.util.concatenate(combined_meshes)
        out = "/tmp/mujoco_hand_pose.obj"
        combined.export(out)
        print(f"\nExported combined hand mesh -> {out}")
    else:
        print("\nNo meshes found!")
        return

    # Also export object
    obj_simp = os.path.join(args.obj_root, args.obj, "processed_data", "mesh", "simplified.obj")
    obj_trimesh = trimesh.load(obj_simp, force="mesh")
    Tobj = np.eye(4)
    Tobj[:3, :3] = obj_se3[:3, :3]
    Tobj[:3, 3] = obj_se3[:3, 3]
    obj_trimesh.apply_transform(Tobj)
    obj_trimesh.export("/tmp/mujoco_object_pose.obj")
    print(f"Exported object -> /tmp/mujoco_object_pose.obj")

    # Viser
    import viser
    server = viser.ViserServer(port=args.port)
    server.scene.add_mesh_simple(
        "/hand_mujoco",
        vertices=np.asarray(combined.vertices),
        faces=np.asarray(combined.faces, dtype=np.uint32),
        color=(220, 100, 80), opacity=0.9, side="double", flat_shading=False,
    )
    server.scene.add_mesh_simple(
        "/object",
        vertices=np.asarray(obj_trimesh.vertices),
        faces=np.asarray(obj_trimesh.faces, dtype=np.uint32),
        color=(80, 220, 80), opacity=0.9, side="double", flat_shading=False,
    )

    # Add BODex contact_point (in object frame → world via obj_se3) as colored spheres,
    # and the corresponding mujoco force_sensor body position too. Connect with a line.
    bi_path = os.path.join(seed_dir, "bodex_info.npy")
    if os.path.exists(bi_path):
        bi = np.load(bi_path, allow_pickle=True).item()
        cps_obj = bi["contact_point"][0, :, :3]  # 5 finger, in object frame
        cps_world = (obj_se3[:3, :3] @ cps_obj.T).T + obj_se3[:3, 3]
        # (link_name, sphere_idx) per finger, matching the manip yml `contact_points_name`.
        if args.hand == "inspire_f1":
            finger_specs = [("thumb_force_sensor", 11), ("index_force_sensor", 14),
                            ("middle_force_sensor", 14), ("ring_force_sensor", 14),
                            ("little_force_sensor", 14)]
            sphere_yml_path = f"{REPO_ROOT}/src/grasp_generation/BODex/src/curobo/content/configs/robot/spheres/inspire_f1.yml"
        elif args.hand == "inspire":
            finger_specs = [("right_thumb_4", 0), ("right_index_2", 2),
                            ("right_middle_2", 2), ("right_ring_2", 2), ("right_little_2", 2)]
            sphere_yml_path = f"{REPO_ROOT}/src/grasp_generation/BODex/src/curobo/content/configs/robot/spheres/inspire.yml"
        elif args.hand == "inspire_left":
            finger_specs = [("left_thumb_4", 0), ("left_index_2", 2),
                            ("left_middle_2", 2), ("left_ring_2", 2), ("left_little_2", 2)]
            sphere_yml_path = f"{REPO_ROOT}/src/grasp_generation/BODex/src/curobo/content/configs/robot/spheres/inspire_left.yml"
        else:
            finger_specs = []; sphere_yml_path = None

        # Load sphere centers
        sphere_local = {}
        if sphere_yml_path and os.path.exists(sphere_yml_path):
            import yaml
            sd = yaml.safe_load(open(sphere_yml_path))
            sd = sd.get("collision_spheres", sd)
            for link, idx in finger_specs:
                if link in sd and idx < len(sd[link]):
                    sphere_local[(link, idx)] = np.array(sd[link][idx]["center"])

        finger_names = [s[0] for s in finger_specs]
        finger_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]

        for i, (fname, color) in enumerate(zip(finger_names, finger_colors)):
            # BODex's intended contact point (where finger SHOULD touch obj)
            server.scene.add_icosphere(
                f"/bodex_contact/{fname}",
                radius=0.012, color=color,
                position=cps_world[i], opacity=0.9,
            )
            # mujoco's actual sphere position (link world pose @ sphere local center)
            bid = mujoco.mj_name2id(mj.model, mujoco.mjtObj.mjOBJ_BODY, mj.hand_prefix + fname)
            if bid < 0:
                continue
            link_pos = mj.data.xpos[bid].copy()
            link_R = mj.data.xmat[bid].reshape(3, 3).copy()
            sphere_idx = finger_specs[i][1]
            local_center = sphere_local.get((fname, sphere_idx), np.zeros(3))
            mujoco_pos = link_R @ local_center + link_pos
            server.scene.add_icosphere(
                f"/mujoco_finger/{fname}",
                radius=0.008, color=color,
                position=mujoco_pos, opacity=0.4,
            )
            # Line connecting them
            server.scene.add_spline_catmull_rom(
                f"/diff_line/{fname}",
                positions=np.array([cps_world[i], mujoco_pos]),
                color=color, line_width=4.0,
            )
            print(f"  bodex_contact[{fname}] = {cps_world[i]}, mujoco_pos = {mujoco_pos}, "
                  f"delta = {np.linalg.norm(cps_world[i] - mujoco_pos):.4f}m")

    # Add scene cuboids (table, shelf walls, etc.) directly from scene_cfg.
    for cube_name, cube_info in scene_cfg.get("cuboid", {}).items():
        dims = np.asarray(cube_info["dims"])
        pose = np.asarray(cube_info["pose"])  # [x,y,z,rx,ry,rz] cart format
        T = cart2se3(pose)
        box = trimesh.creation.box(extents=dims)
        box.apply_transform(T)
        server.scene.add_mesh_simple(
            f"/scene/{cube_name}",
            vertices=np.asarray(box.vertices),
            faces=np.asarray(box.faces, dtype=np.uint32),
            color=(120, 160, 220), opacity=0.4, side="double", flat_shading=True,
        )
        print(f"  added scene cuboid: {cube_name} dims={dims}")
    print(f"\nViser at http://localhost:{args.port}")
    print("Ctrl-C to exit")
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
