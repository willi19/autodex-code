"""Hardware-free grasp-plan test: place an object at a reachable spot on the
table (no perception) and run the planner for a given arm+hand.

Verifies the candidate pool + planner produce a reachable, collision-free grasp
trajectory for the object at a tabletop pose — e.g. franka(fr3)+inspire on v8
candidates.

    python src/execution/plan_test.py --obj servingbowl_small --hand fr3_inspire \
        --version v8 --pose_idx 003 --x 0.5 [--viz --port 8080]
"""
import argparse
import os

import numpy as np

from autodex.planner import GraspPlanner
from autodex.planner.obstacles import TABLE_CUBOID
from autodex.utils.path import obj_path
from autodex.utils.conversion import se32cart, cart2se3

TABLE_SURFACE_Z = TABLE_CUBOID["pose"][2] + TABLE_CUBOID["dims"][2] / 2  # 0.039


def place_T(pose_T: np.ndarray, x: float, yaw: float = 0.0) -> np.ndarray:
    """Object world pose: tabletop orientation yawed about z, resting at
    (x, 0, table_surface). Mirrors exp.py:_place_T."""
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T = pose_T.copy()
    T[:3, :3] = Rz @ pose_T[:3, :3]
    T[0, 3], T[1, 3] = float(x), 0.0
    T[2, 3] = float(pose_T[2, 3]) + TABLE_SURFACE_Z
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="servingbowl_small")
    ap.add_argument("--hand", default="fr3_inspire")
    ap.add_argument("--version", default="v8")
    ap.add_argument("--pose_idx", default="003")
    ap.add_argument("--x", type=float, default=0.5)
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    tt_path = os.path.join(obj_path, args.obj, "processed_data", "info",
                           "tabletop", f"{args.pose_idx}.npy")
    pose_T = np.load(tt_path)
    T = place_T(pose_T, args.x, args.yaw)
    mesh_path = os.path.join(obj_path, args.obj, "processed_data", "mesh", "simplified.obj")

    scene_cfg = {
        "mesh": {"target": {"pose": se32cart(T).tolist(), "file_path": mesh_path}},
        "cuboid": {"table": TABLE_CUBOID},
    }

    print(f"[plan_test] {args.obj} hand={args.hand} version={args.version} "
          f"pose_idx={args.pose_idx} x={args.x}")
    planner = GraspPlanner(hand=args.hand)
    res = planner.plan(scene_cfg, args.obj, args.version, hand=args.hand)

    print(f"[plan_test] success={res.success}")
    print(f"[plan_test] timing={res.timing}")
    if res.success:
        print(f"[plan_test] selected scene_info={res.scene_info}")
        print(f"[plan_test] traj shape={None if res.traj is None else np.asarray(res.traj).shape}")

    if args.viz and res.success:
        import trimesh
        import yourdfpy
        import viser
        import time as _time

        traj = np.asarray(res.traj)                     # (T, 13) = 7 arm + 6 hand
        urdf_path = os.path.expanduser(
            "~/shared_data/AutoDex/content/assets/robot/"
            "fr3_inspire_description/fr3_inspire.urdf")
        robot = yourdfpy.URDF.load(urdf_path, load_meshes=True,
                                   build_collision_scene_graph=False)
        aj = robot.actuated_joint_names
        n = min(len(aj), traj.shape[1])
        print(f"[plan_test] URDF actuated joints={len(aj)}, traj dof={traj.shape[1]}")

        obj_mesh = trimesh.load(mesh_path, force="mesh")
        obj_mesh.apply_transform(T)

        server = viser.ViserServer(port=args.port)
        server.scene.add_mesh_trimesh("/table", trimesh.creation.box(
            extents=np.array(TABLE_CUBOID["dims"], float)).apply_transform(
            cart2se3(np.array(TABLE_CUBOID["pose"], float))))
        server.scene.add_mesh_trimesh("/obj", obj_mesh)

        def show(k):
            robot.update_cfg({aj[i]: float(traj[k, i]) for i in range(n)})
            server.scene.add_mesh_trimesh("/robot", robot.scene.to_geometry())

        sl = server.gui.add_slider("waypoint", min=0, max=traj.shape[0] - 1,
                                   step=1, initial_value=traj.shape[0] - 1)
        sl.on_update(lambda _: show(int(sl.value)))
        show(traj.shape[0] - 1)     # start at grasp pose
        print(f"[plan_test] viz on http://localhost:{args.port} "
              f"({traj.shape[0]} waypoints; slider scrubs INIT->grasp)")
        while True:
            _time.sleep(1.0)


if __name__ == "__main__":
    main()
