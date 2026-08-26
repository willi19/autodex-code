"""Offline plan visualizer: reload a franka trial's perceived object pose,
re-plan for fr3_inspire, and render the PLANNED trajectory in the scene (object
mesh + table + robot along the path) so you can see whether the PLAN itself
collides — vs the real robot drifting off it during velocity streaming.

    python src/execution/franka_plan_viz.py                 # latest trial
    python src/execution/franka_plan_viz.py --trial <dir> --port 8090
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import trimesh
import yourdfpy
import viser

from autodex.planner import GraspPlanner
from autodex.planner.planner import _to_curobo_world  # noqa: F401
from autodex.utils.conversion import cart2se3
from autodex.utils.path import obj_path, project_dir, get_obj_root
from src.execution.scene_cfg import pose_world_to_scene_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="servingbowl_small")
    ap.add_argument("--hand", default="fr3_inspire")
    ap.add_argument("--version", default="v8")
    ap.add_argument("--trial", default=None, help="trial dir; default = latest")
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args()

    exp = Path(project_dir) / "experiment" / args.version / "inspire" / args.obj
    trial = Path(args.trial) if args.trial else \
        sorted(exp.iterdir(), key=lambda p: p.name)[-1]
    print(f"[viz] trial: {trial}")
    pose_world = np.load(trial / "pose_world.npy")
    c2r = np.load(trial / "C2R.npy")
    scene_cfg = pose_world_to_scene_cfg(pose_world, c2r, args.obj,
                                        get_obj_root(args.version))
    obj_T = cart2se3(np.array(scene_cfg["mesh"]["target"]["pose"], float))

    planner = GraspPlanner(hand=args.hand)
    res = planner.plan(scene_cfg, args.obj, args.version, hand=args.hand)
    print(f"[viz] plan success={res.success}  timing={res.timing}")
    if not res.success:
        print("[viz] plan failed — nothing to show")
        return
    traj = np.asarray(res.traj)                              # (T, 13)

    mesh_path = os.path.join(obj_path, args.obj, "processed_data", "mesh", "simplified.obj")
    obj_mesh = trimesh.load(mesh_path, force="mesh").apply_transform(obj_T)

    urdf = yourdfpy.URDF.load(os.path.expanduser(
        "~/shared_data/AutoDex/content/assets/robot/"
        "fr3_inspire_description/fr3_inspire.urdf"),
        load_meshes=True, build_collision_scene_graph=False)
    aj = urdf.actuated_joint_names
    ncol = min(len(aj), traj.shape[1])

    server = viser.ViserServer(port=args.port)
    # object (RED so a robot-object overlap is obvious)
    om = obj_mesh.copy()
    om.visual = trimesh.visual.ColorVisuals(
        om, vertex_colors=np.tile([220, 60, 60, 255], (len(om.vertices), 1)))
    server.scene.add_mesh_trimesh("/obj", om)
    # obstacles (table + any wall/shelf/box cuboids)
    for name, c in scene_cfg.get("cuboid", {}).items():
        box = trimesh.creation.box(extents=np.array(c["dims"], float)).apply_transform(
            cart2se3(np.array(c["pose"], float)))
        box.visual = trimesh.visual.ColorVisuals(
            box, vertex_colors=np.tile([150, 160, 175, 90], (len(box.vertices), 1)))
        server.scene.add_mesh_trimesh(f"/obs/{name}", box)

    def show(k):
        urdf.update_cfg({aj[i]: float(traj[k, i]) for i in range(ncol)})
        server.scene.add_mesh_trimesh("/robot", urdf.scene.to_geometry())

    sl = server.gui.add_slider("waypoint", min=0, max=traj.shape[0] - 1,
                               step=1, initial_value=0)
    sl.on_update(lambda _: show(int(sl.value)))
    show(0)   # start at INIT so you can scrub INIT -> grasp and watch for overlap
    print(f"[viz] http://localhost:{args.port}  — scrub the slider; red=object. "
          f"If the robot passes THROUGH red, the PLAN collides; if the plan is "
          f"clean but the real arm hit it, that's velocity-stream drift.")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
