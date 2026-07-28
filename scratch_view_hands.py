#!/usr/bin/env python3
"""Show inspire LEFT and RIGHT hands together with xyz axes so we can decide
the FR3 flange->hand mount for the right hand.

Reference: on FR3 the LEFT hand mounts via `flange_to_hand`:
    rpy = (-3.14159, 0, 0.7854), xyz = (0, 0, 0.04)   [parent fr3_link8]
We display that frame too, so the right-hand equivalent can be read off.

Run:  python scratch_view_hands.py --port 8080
"""
import argparse
import time
import numpy as np
import viser
import viser.extras
import yourdfpy
from scipy.spatial.transform import Rotation as R

ROBOT_ROOT = "autodex/planner/src/curobo/content/assets/robot/inspire_description"
RIGHT_URDF = f"{ROBOT_ROOT}/inspire_hand_right.urdf"
LEFT_URDF = f"{ROBOT_ROOT}/inspire_hand_left.urdf"

# Known FR3 left-hand mount (fr3_link8 -> base_link)
LEFT_MOUNT_RPY = (-3.14159, 0.0, 0.7854)
LEFT_MOUNT_XYZ = (0.0, 0.0, 0.04)


def wxyz(rpy):
    q = R.from_euler("xyz", rpy).as_quat()  # xyzw -> wxyz
    return np.array([q[3], q[0], q[1], q[2]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    server = viser.ViserServer(port=args.port)
    server.scene.set_up_direction("+z")

    # World axes
    server.scene.add_frame("/world", axes_length=0.15, axes_radius=0.005)

    # Group containers (toggle visibility on these)
    right_grp = server.scene.add_frame("/right", show_axes=False)
    left_grp = server.scene.add_frame("/left", show_axes=False,
                                      position=np.array([0.0, 0.15, 0.0]))

    # Right hand + base_link xyz axes
    right_urdf = yourdfpy.URDF.load(RIGHT_URDF, build_scene_graph=True,
                                    load_meshes=True)
    v_right = viser.extras.ViserUrdf(server, right_urdf,
                                     root_node_name="/right/urdf")
    v_right.update_cfg(np.zeros(len(right_urdf.actuated_joint_names)))
    server.scene.add_frame("/right/base_axes", axes_length=0.10,
                           axes_radius=0.004)

    # Left hand + base_link xyz axes
    left_urdf = yourdfpy.URDF.load(LEFT_URDF, build_scene_graph=True,
                                   load_meshes=True)
    v_left = viser.extras.ViserUrdf(server, left_urdf,
                                    root_node_name="/left/urdf")
    v_left.update_cfg(np.zeros(len(left_urdf.actuated_joint_names)))
    server.scene.add_frame("/left/base_axes", axes_length=0.10,
                           axes_radius=0.004)

    # FR3 left-hand mount reference frame (where LEFT base_link sits vs flange)
    mount = server.scene.add_frame(
        "/fr3_left_mount", axes_length=0.10, axes_radius=0.005,
        position=np.array(LEFT_MOUNT_XYZ), wxyz=wxyz(LEFT_MOUNT_RPY))

    # ── GUI ──
    g_right = server.gui.add_checkbox("show RIGHT hand", True)
    g_left = server.gui.add_checkbox("show LEFT hand", True)
    g_mount = server.gui.add_checkbox("show FR3 left-mount frame", True)
    g_yoff = server.gui.add_slider("LEFT y-offset (separate<->overlay)",
                                   min=-0.3, max=0.3, step=0.01,
                                   initial_value=0.15)

    @g_right.on_update
    def _(_):
        right_grp.visible = g_right.value

    @g_left.on_update
    def _(_):
        left_grp.visible = g_left.value

    @g_mount.on_update
    def _(_):
        mount.visible = g_mount.value

    @g_yoff.on_update
    def _(_):
        left_grp.position = np.array([0.0, g_yoff.value, 0.0])

    print(f"viser up on http://localhost:{args.port}")
    print(f"RIGHT actuated joints: {right_urdf.actuated_joint_names}")
    print(f"LEFT  actuated joints: {left_urdf.actuated_joint_names}")
    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
