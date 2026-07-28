#!/usr/bin/env python3
"""Show FR3+inspire-right at FR3_INIT next to xarm+inspire-right at XARM_INIT.

The two wrists (hand base_link) land on the same 6D pose by construction —
this is to eyeball the ARM posture (elbow config) that IK picked for the FR3,
plus the table, before using FR3_INIT for real capture.

Run: python scratch_view_fr3_init.py --port 8083
"""
import argparse
import time

import numpy as np
import viser
import viser.extras
import yourdfpy

from autodex.utils.robot_config import FR3_INIT, XARM_INIT, INSPIRE_INIT

A = "autodex/planner/src/curobo/content/assets/robot"
FR3_URDF = f"{A}/fr3_inspire_description/fr3_inspire.urdf"
XARM_URDF = f"{A}/inspire_description/xarm_inspire.urdf"

TABLE_POSE_XYZ = [1.1, 0.0, -0.1]
TABLE_DIMS = [2.0, 3.0, 0.2]


def cfg_for(urdf, arm_names, arm_q, hand_q=None):
    cfg = {n: 0.0 for n in urdf.actuated_joint_names}
    for n, v in zip(arm_names, arm_q):
        if n in cfg:
            cfg[n] = float(v)
    return np.array([cfg[n] for n in urdf.actuated_joint_names])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8083)
    args = ap.parse_args()

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")
    server.scene.add_frame("/world", axes_length=0.2, axes_radius=0.005)

    # table (same cuboid the planner uses)
    server.scene.add_box("/table", dimensions=tuple(TABLE_DIMS),
                         position=tuple(TABLE_POSE_XYZ),
                         color=(200, 200, 200))

    fr3 = yourdfpy.URDF.load(FR3_URDF, build_scene_graph=True, load_meshes=True)
    xarm = yourdfpy.URDF.load(XARM_URDF, build_scene_graph=True, load_meshes=True)

    g_fr3 = server.scene.add_frame("/fr3", show_axes=False)
    g_xarm = server.scene.add_frame("/xarm", show_axes=False)

    v_fr3 = viser.extras.ViserUrdf(server, fr3, root_node_name="/fr3/urdf")
    v_xarm = viser.extras.ViserUrdf(server, xarm, root_node_name="/xarm/urdf")

    v_fr3.update_cfg(cfg_for(fr3, [f"fr3_joint{i}" for i in range(1, 8)], FR3_INIT))
    v_xarm.update_cfg(cfg_for(xarm, [f"joint{i}" for i in range(1, 7)], XARM_INIT))

    # wrist frames — these should coincide
    fr3.update_cfg(cfg_for(fr3, [f"fr3_joint{i}" for i in range(1, 8)], FR3_INIT))
    xarm.update_cfg(cfg_for(xarm, [f"joint{i}" for i in range(1, 7)], XARM_INIT))
    T_f = fr3.get_transform("base_link", "base")
    T_x = xarm.get_transform("base_link", "link_base")
    for name, T in [("/wrist_fr3", T_f), ("/wrist_xarm", T_x)]:
        from scipy.spatial.transform import Rotation as R
        q = R.from_matrix(T[:3, :3]).as_quat()
        server.scene.add_frame(name, axes_length=0.15, axes_radius=0.006,
                               position=T[:3, 3],
                               wxyz=np.array([q[3], q[0], q[1], q[2]]))

    c_fr3 = server.gui.add_checkbox("show FR3 (FR3_INIT)", True)
    c_xarm = server.gui.add_checkbox("show xarm (XARM_INIT)", True)

    @c_fr3.on_update
    def _(_):
        g_fr3.visible = c_fr3.value

    @c_xarm.on_update
    def _(_):
        g_xarm.visible = c_xarm.value

    print(f"FR3 wrist  xyz: {np.round(T_f[:3,3],5)}")
    print(f"xarm wrist xyz: {np.round(T_x[:3,3],5)}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
