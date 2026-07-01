"""Visualize robot hand mesh + collision spheres side by side.

Usage:
    python src/visualization/grasp_generation/view_collision_spheres.py --hand allegro
    python src/visualization/grasp_generation/view_collision_spheres.py --hand inspire
"""

import os
import argparse
import numpy as np
import trimesh
import yaml
import viser

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
BODEX_ASSETS = os.path.join(REPO_ROOT, "src", "grasp_generation", "BODex", "src", "curobo", "content")

HAND_CONFIGS = {
    "allegro": {
        "urdf": os.path.join(BODEX_ASSETS, "assets", "robot", "allegro_description", "allegro_hand_description_right.urdf"),
        "spheres": os.path.join(BODEX_ASSETS, "configs", "robot", "spheres", "allegro.yml"),
    },
    "inspire": {
        "urdf": os.path.join(BODEX_ASSETS, "assets", "robot", "inspire_description", "inspire_hand_right.urdf"),
        "spheres": os.path.join(BODEX_ASSETS, "configs", "robot", "spheres", "inspire.yml"),
    },
    "inspire_f1": {
        "urdf": os.path.join(BODEX_ASSETS, "assets", "robot", "inspire_f1_description", "inspire_f1_hand_right.urdf"),
        "spheres": os.path.join(BODEX_ASSETS, "configs", "robot", "spheres", "inspire_f1.yml"),
    },
    # full xarm6 + inspire-left robot: the spheres file uses full robot link names
    # (link_base, link1..6, hand links), so FK needs the full robot URDF.
    "inspire_left": {
        "urdf": os.path.expanduser("~/shared_data/AutoDex/content/assets/robot/inspire_left_description/xarm_inspire_left.urdf"),
        "spheres": os.path.expanduser("~/shared_data/AutoDex/content/configs/robot/spheres/xarm_inspire_left.yml"),
    },
}

# Distinct colors per link group
LINK_COLORS = [
    (1.0, 0.2, 0.2),  # red
    (0.2, 1.0, 0.2),  # green
    (0.2, 0.2, 1.0),  # blue
    (1.0, 1.0, 0.2),  # yellow
    (1.0, 0.2, 1.0),  # magenta
    (0.2, 1.0, 1.0),  # cyan
    (1.0, 0.6, 0.2),  # orange
    (0.6, 0.2, 1.0),  # purple
    (0.2, 0.6, 0.2),  # dark green
    (0.8, 0.4, 0.4),  # pink
]


def load_urdf_meshes(urdf_path):
    """Parse URDF and load visual meshes with their transforms via FK at zero config."""
    import yourdfpy
    urdf_dir = os.path.dirname(urdf_path)
    robot = yourdfpy.URDF.load(urdf_path, build_collision_scene_graph=False, load_collision_meshes=False)

    meshes = {}
    scene = robot.scene
    for name, geom in scene.geometry.items():
        T = scene.graph.get(name)[0]
        meshes[name] = {"mesh": geom, "transform": T}
    return meshes


def load_spheres(spheres_path):
    """Load collision spheres from YAML."""
    with open(spheres_path) as f:
        data = yaml.safe_load(f)
    return data.get("collision_spheres", {})


def build_fk_transforms(urdf_path):
    """Get per-link transforms at zero joint config."""
    import yourdfpy
    robot = yourdfpy.URDF.load(urdf_path, build_collision_scene_graph=False, load_collision_meshes=False)
    # zero config
    cfg = np.zeros(len(robot.actuated_joint_names))
    robot.update_cfg(cfg)

    transforms = {}
    for link_name in robot.link_map.keys():
        try:
            T = robot.get_transform(link_name)
            transforms[link_name] = T
        except Exception:
            transforms[link_name] = np.eye(4)
    return transforms


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", choices=list(HAND_CONFIGS.keys()), default="inspire")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    cfg = HAND_CONFIGS[args.hand]
    spheres_data = load_spheres(cfg["spheres"])
    link_transforms = build_fk_transforms(cfg["urdf"])

    server = viser.ViserServer(port=args.port)
    print(f"Viewer at http://localhost:{args.port}")

    # Add hand mesh via URDF
    urdf_path = cfg["urdf"]
    urdf_dir = os.path.dirname(urdf_path)

    # Read URDF and render meshes
    import yourdfpy
    robot = yourdfpy.URDF.load(urdf_path, build_collision_scene_graph=False, load_collision_meshes=False)
    zero_cfg = np.zeros(len(robot.actuated_joint_names))
    robot.update_cfg(zero_cfg)

    scene = robot.scene
    for geom_name, geom in scene.geometry.items():
        T = scene.graph.get(geom_name)[0]
        if isinstance(geom, trimesh.Trimesh):
            server.scene.add_mesh_trimesh(
                f"/mesh/{geom_name}", geom.apply_transform(T),
            )

    # Add collision spheres
    link_names = list(spheres_data.keys())
    for li, link_name in enumerate(link_names):
        color = LINK_COLORS[li % len(LINK_COLORS)]
        T = link_transforms.get(link_name, np.eye(4))

        for si, sphere in enumerate(spheres_data[link_name]):
            center = np.array(sphere["center"])
            radius = sphere["radius"]
            if radius < 1e-6:
                continue

            # Transform sphere center to world frame
            center_world = (T[:3, :3] @ center) + T[:3, 3]

            server.scene.add_icosphere(
                f"/spheres/{link_name}/{si}",
                radius=radius,
                color=color,
                position=center_world,
            )


    # Legend
    print(f"\nCollision spheres for {args.hand}:")
    for li, link_name in enumerate(link_names):
        n = len(spheres_data[link_name])
        color = LINK_COLORS[li % len(LINK_COLORS)]
        print(f"  {link_name}: {n} spheres (color: {color})")

    print("Ctrl-C to exit...")
    import time
    while True:
        time.sleep(1)