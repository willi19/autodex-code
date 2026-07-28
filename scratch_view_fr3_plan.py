#!/usr/bin/env python3
"""Play back a saved FR3 reorient plan (trajectory.npz + meta.json).

Uses viser's own ViserUrdf, which handles the FR3 textured .dae meshes that
paradex's ViserRobotModule chokes on (it assumes vertex_colors).

The carried object follows the hand during lift/rotate/place; before that it
sits at T_obj_start, after release at T_obj_end.

Usage:
  python scratch_view_fr3_plan.py \
      --plan_dir outputs/reset_plans/fr3_inspire/apple/reorient_0/0_65/x0.40_tz000/2 \
      --port 8090
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh
import viser
import viser.extras
import yourdfpy
from scipy.spatial.transform import Rotation as R

from autodex.utils.path import obj_path, project_dir

URDF_REL = ("fr3_inspire_description", "fr3_inspire.urdf")
EE_LINK = "base_link"
HELD_PHASES = {"lift", "rotate", "place"}
TABLE_POSE_XYZ = [1.1, 0.0, -0.1]
TABLE_DIMS = [2.0, 3.0, 0.2]


def wxyz(Rm):
    q = R.from_matrix(Rm).as_quat()
    return np.array([q[3], q[0], q[1], q[2]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan_dir", required=True)
    ap.add_argument("--port", type=int, default=8090)
    args = ap.parse_args()

    pd = Path(args.plan_dir)
    meta = json.loads((pd / "meta.json").read_text())
    npz = np.load(pd / "trajectory.npz")
    phases = [p for p in meta["phase_names"] if p in npz]

    # flatten frames, remembering which phase each came from
    frames, frame_phase = [], []
    for ph in phases:
        tr = np.asarray(npz[ph])
        for q in tr:
            frames.append(q)
            frame_phase.append(ph)
    frames = np.asarray(frames)
    print(f"[view] {meta['obj_name']} {meta['i']}->{meta['j']} h={meta['h_cm']}cm "
          f"seed={meta['seed_id']} | {len(frames)} frames over {len(phases)} phases")

    urdf_path = Path(project_dir) / "content" / "assets" / "robot" / URDF_REL[0] / URDF_REL[1]
    urdf = yourdfpy.URDF.load(str(urdf_path), build_scene_graph=True, load_meshes=True)

    T_obj_start = np.array(meta["T_obj_start"])
    T_obj_end = np.array(meta["T_obj_end"])
    W_inv = np.linalg.inv(np.array(meta["wrist_se3_obj"]))  # T_obj = T_wrist @ inv(wrist_se3_obj)

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_frame("/world", axes_length=0.15, axes_radius=0.004)
    server.scene.add_box("/table", dimensions=tuple(TABLE_DIMS),
                         position=tuple(TABLE_POSE_XYZ), color=(190, 190, 190))

    vurdf = viser.extras.ViserUrdf(server, urdf, root_node_name="/robot")

    mesh_file = Path(obj_path) / meta["obj_name"] / "processed_data" / "mesh" / "simplified.obj"
    obj_mesh = trimesh.load(str(mesh_file), force="mesh")
    obj_node = server.scene.add_mesh_simple(
        "/object", vertices=np.asarray(obj_mesh.vertices),
        faces=np.asarray(obj_mesh.faces), color=(235, 140, 60))

    # ghosts of start/end object pose
    for nm, T, col in [("/ghost_start", T_obj_start, (120, 180, 255)),
                       ("/ghost_end", T_obj_end, (140, 235, 140))]:
        server.scene.add_mesh_simple(
            nm, vertices=np.asarray(obj_mesh.vertices), faces=np.asarray(obj_mesh.faces),
            color=col, opacity=0.25, position=T[:3, 3], wxyz=wxyz(T[:3, :3]))

    g_frame = server.gui.add_slider("frame", min=0, max=len(frames) - 1, step=1,
                                    initial_value=0)
    g_phase = server.gui.add_text("phase", initial_value=frame_phase[0], disabled=True)
    g_play = server.gui.add_checkbox("play", False)
    g_speed = server.gui.add_slider("fps", min=5, max=120, step=5, initial_value=40)

    names = urdf.actuated_joint_names

    def show(idx: int):
        q = frames[idx]
        vurdf.update_cfg(np.asarray(q[:len(names)], dtype=float))
        ph = frame_phase[idx]
        g_phase.value = ph
        urdf.update_cfg(np.asarray(q[:len(names)], dtype=float))
        T_w = urdf.get_transform(EE_LINK, urdf.base_link)
        if ph in HELD_PHASES or ph == "grasp_close":
            T_o = T_w @ W_inv
        elif ph in ("approach",):
            T_o = T_obj_start
        else:
            T_o = T_obj_end
        obj_node.position = T_o[:3, 3]
        obj_node.wxyz = wxyz(T_o[:3, :3])

    @g_frame.on_update
    def _(_):
        show(int(g_frame.value))

    show(0)
    print(f"[view] http://localhost:{args.port}")
    while True:
        if g_play.value:
            nxt = (int(g_frame.value) + 1) % len(frames)
            g_frame.value = nxt
            show(nxt)
            time.sleep(1.0 / float(g_speed.value))
        else:
            time.sleep(0.05)


if __name__ == "__main__":
    main()
