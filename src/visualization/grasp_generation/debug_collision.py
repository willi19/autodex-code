"""Export and visualize cuRobo collision spheres + world mesh for one grasp seed.

Usage:
    python src/visualization/grasp_generation/debug_collision.py \
        --hand inspire_f1 --obj Jp_Water --scene shelf --scene_id 37 --seed 0 \
        --obj_root /home/mingi/shared_data/AutoDex/object/robothome
"""
import os
import sys
import json
import argparse
import numpy as np
import trimesh
import viser
import time
from scipy.spatial.transform import Rotation as Rot

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, REPO_ROOT)

from autodex.planner.planner import GraspPlanner, _to_curobo_world
from autodex.utils.conversion import cart2se3
from autodex.utils.path import get_scene_dir

HAND_CFGS = {
    "inspire_f1": "autodex/planner/src/curobo/content/configs/robot/inspire_f1_floating.yml",
    "inspire": "autodex/planner/src/curobo/content/configs/robot/inspire_floating.yml",
    "inspire_left": "autodex/planner/src/curobo/content/configs/robot/inspire_left_floating.yml",
    "allegro": "autodex/planner/src/curobo/content/configs/robot/allegro_floating.yml",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="inspire_f1")
    ap.add_argument("--obj", required=True)
    ap.add_argument("--scene", default="shelf")
    ap.add_argument("--scene_id", default="37")
    ap.add_argument("--seed", default="0")
    ap.add_argument("--version", default="v3")
    ap.add_argument("--obj_root", default="/home/mingi/shared_data/AutoDex/object/robothome")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    scene_json = os.path.join(get_scene_dir(args.hand, args.obj, args.scene), f"{args.scene_id}.json")
    seed_dir = os.path.join(REPO_ROOT, "bodex_outputs", args.hand, args.version,
                             args.obj, args.scene, args.scene_id, args.seed)
    print(f"scene_json: {scene_json}")
    print(f"seed_dir:   {seed_dir}")

    scene_cfg = json.load(open(scene_json))["scene"]
    world_cfg = _to_curobo_world(scene_cfg)
    obj_se3 = cart2se3(scene_cfg["mesh"]["target"]["pose"])

    wrist_local = np.load(os.path.join(seed_dir, "wrist_se3.npy"))
    pregrasp = np.load(os.path.join(seed_dir, "pregrasp_pose.npy"))
    wrist_world = obj_se3 @ wrist_local

    hand_cfg_path = os.path.join(REPO_ROOT, HAND_CFGS[args.hand])
    planner = GraspPlanner(robot_cfg_path=hand_cfg_path, hand_cfg_path=hand_cfg_path)
    planner._init_motion_gen(world_cfg)

    xyz = wrist_world[:3, 3]
    R = wrist_world[:3, :3]
    # Try different conventions; default to intrinsic XYZ. Override via env var.
    convention = os.environ.get("RPY_CONV", "XYZ")
    rpy = Rot.from_matrix(R).as_euler(convention)
    if os.environ.get("FLIP", "") == "z":
        rpy[2] = -rpy[2]
    elif os.environ.get("FLIP", "") == "y":
        rpy[1] = -rpy[1]
    elif os.environ.get("FLIP", "") == "x":
        rpy[0] = -rpy[0]
    goal_joint = np.concatenate([xyz, rpy, pregrasp])
    print(f"  RPY_CONV={convention} → rpy={rpy}")
    # Sanity-check: reconstruct and compare
    R_back = Rot.from_euler(convention, rpy).as_matrix()
    print(f"  rot reconstruct delta = {np.linalg.norm(R - R_back):.6f}")

    planner._export_collision_debug(goal_joint)

    # Re-extract spheres for per-link rendering in viser (color by link).
    import torch
    q = torch.tensor(goal_joint, dtype=torch.float32, device=planner._tensor_args.device).unsqueeze(0)
    kin = planner._motion_gen.kinematics
    spheres_batch = kin.get_robot_as_spheres(q)  # List[List[Sphere]] (batch, link)

    # Visualize via viser
    server = viser.ViserServer(port=args.port)
    debug_dir = "/tmp/collision_debug"

    # Per-sphere with rotating colors so link groups are distinguishable.
    palette = [
        (255, 80, 80), (80, 255, 80), (80, 120, 255),
        (255, 220, 60), (255, 80, 220), (60, 220, 220),
        (255, 160, 60), (160, 80, 255), (80, 200, 120),
        (220, 120, 120), (120, 220, 120), (120, 120, 220),
    ]
    sidx = 0
    for li, sphere_list in enumerate(spheres_batch[0] if isinstance(spheres_batch[0], list) else spheres_batch):
        color = palette[li % len(palette)]
        for s in sphere_list if isinstance(sphere_list, list) else [sphere_list]:
            if hasattr(s, "position"):
                pos = np.asarray(s.position)
                rad = float(s.radius)
            else:
                pos = np.asarray(s[:3]); rad = float(s[3])
            if rad <= 0: continue
            server.scene.add_icosphere(
                f"/spheres/{sidx}",
                radius=rad, color=color,
                position=pos, opacity=0.45,
            )
            sidx += 1
    print(f"  added {sidx} collision spheres")

    for fn in sorted(os.listdir(debug_dir)):
        if not fn.endswith(".obj") or "hand_spheres" in fn:
            continue
        path = os.path.join(debug_dir, fn)
        m = trimesh.load(path, force="mesh")
        if not isinstance(m, trimesh.Trimesh):
            continue
        color = (100, 200, 255)
        opacity = 0.4
        server.scene.add_mesh_simple(
            f"/{fn[:-4]}",
            vertices=np.asarray(m.vertices),
            faces=np.asarray(m.faces, dtype=np.uint32),
            color=color,
            opacity=opacity,
            flat_shading=False,
            side="double",
        )
        print(f"  added {fn}: V={len(m.vertices)} F={len(m.faces)}")

    # Also load object mesh
    simp = os.path.join(args.obj_root, args.obj, "processed_data", "mesh", "simplified.obj")
    if os.path.exists(simp):
        m = trimesh.load(simp, force="mesh")
        verts = (obj_se3[:3, :3] @ m.vertices.T).T + obj_se3[:3, 3]
        server.scene.add_mesh_simple(
            "/object",
            vertices=verts,
            faces=np.asarray(m.faces, dtype=np.uint32),
            color=(50, 220, 50),
            opacity=0.8,
            flat_shading=False,
            side="double",
        )
        print(f"  added object: V={len(m.vertices)} F={len(m.faces)}")

    print(f"\nViser at http://localhost:{args.port}")
    print("Ctrl-C to exit")
    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
