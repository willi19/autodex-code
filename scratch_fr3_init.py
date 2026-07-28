#!/usr/bin/env python3
"""Derive FR3_INIT (7 arm joints) + FR3 link->wrist transform for inspire right.

FR3_INIT: take the wrist (hand base_link) 6D pose that the xarm reaches at
XARM_INIT, then solve FR3 IK for that same pose. Both robots use
ee_link="base_link", so the target means the same hand pose in the robot base
frame.

FR3_INSPIRE_LINK_TO_WRIST: fixed transform from the last arm link (fr3_link7)
to the wrist (base_link), the FR3 analog of INSPIRE_LINK6_TO_WRIST (xarm link6
-> wrist). Used by GraspPlanner's backward filter.
"""
import numpy as np
import torch
import yourdfpy
from scipy.spatial.transform import Rotation as R

from autodex.utils.robot_config import XARM_INIT, INSPIRE_INIT, INSPIRE_LINK6_TO_WRIST

A = "autodex/planner/src/curobo/content/assets/robot"
XARM_URDF = f"{A}/inspire_description/xarm_inspire.urdf"
FR3_URDF = f"{A}/fr3_inspire_description/fr3_inspire.urdf"


def fk_pose(urdf, cfg_map, frame, root):
    """Pose of `frame` expressed in `root` (verified convention below)."""
    urdf.update_cfg(cfg_map)
    return urdf.get_transform(frame, root)


def main():
    # ---- 1. xarm wrist pose at XARM_INIT -----------------------------------
    xu = yourdfpy.URDF.load(XARM_URDF, build_scene_graph=True, load_meshes=False)
    xcfg = {n: 0.0 for n in xu.actuated_joint_names}
    for n, v in zip([f"joint{i}" for i in range(1, 7)], XARM_INIT):
        if n in xcfg:
            xcfg[n] = float(v)
    # xarm arm joints may be named joint1..6; verify we set 6 of them
    n_set = sum(1 for n in [f"joint{i}" for i in range(1, 7)] if n in xcfg)
    T_x_wrist = fk_pose(xu, xcfg, "base_link", "link_base")
    print(f"xarm arm joints matched: {n_set}/6  (actuated: {xu.actuated_joint_names[:8]})")
    print("xarm wrist (base_link) in link_base:")
    print("  xyz:", np.round(T_x_wrist[:3, 3], 5))
    print("  rpy:", np.round(R.from_matrix(T_x_wrist[:3, :3]).as_euler("xyz"), 5))

    # sanity: link6 -> wrist from URDF should match INSPIRE_LINK6_TO_WRIST
    T_l6_wrist = np.linalg.inv(fk_pose(xu, xcfg, "link6", "link_base")) @ T_x_wrist
    print("\nURDF link6->wrist vs INSPIRE_LINK6_TO_WRIST const:",
          np.allclose(T_l6_wrist, INSPIRE_LINK6_TO_WRIST, atol=1e-3))
    print(np.round(T_l6_wrist, 4))

    # ---- 2. FR3 link7 -> wrist (fixed) -------------------------------------
    fu = yourdfpy.URDF.load(FR3_URDF, build_scene_graph=True, load_meshes=False)
    fcfg = {n: 0.0 for n in fu.actuated_joint_names}
    T_l7_wrist = np.linalg.inv(fk_pose(fu, fcfg, "fr3_link7", "base")) @ \
        fk_pose(fu, fcfg, "base_link", "base")
    print("\nFR3_INSPIRE_LINK_TO_WRIST (fr3_link7 -> base_link):")
    print(np.round(T_l7_wrist, 6))

    # ---- 3. FR3 IK for the xarm wrist pose ---------------------------------
    import os
    from autodex.planner.planner import GraspPlanner, robot_configs_path
    from curobo.types.math import Pose

    p = GraspPlanner(robot_cfg_path=os.path.join(robot_configs_path, "fr3_inspire.yml"),
                     hand_cfg_path=os.path.join(robot_configs_path, "inspire_floating.yml"))
    world = {"cuboid": {"table": {"dims": [2.0, 3.0, 0.2],
                                  "pose": [1.1, 0.0, -0.1, 1.0, 0.0, 0.0, 0.0]}}, "mesh": {}}
    p._init_ik_solver(world)

    q = R.from_matrix(T_x_wrist[:3, :3]).as_quat()  # xyzw
    pos = torch.tensor([T_x_wrist[:3, 3]], device="cuda:0", dtype=torch.float32)
    quat = torch.tensor([[q[3], q[0], q[1], q[2]]], device="cuda:0", dtype=torch.float32)
    res = p._ik_solver.solve_batch(Pose(pos, quat))
    ok = bool(res.success[0][0])
    sol = res.solution[0][0].cpu().numpy()
    print(f"\nFR3 IK for xarm wrist pose: success={ok}")
    if not ok:
        print("  -> no exact solution; consider relaxing/target nearby pose")
        return
    arm = sol[:7]
    print("  FR3_INIT (7 arm joints):", np.round(arm, 6).tolist())

    # verify by FK
    for n, v in zip([f"fr3_joint{i}" for i in range(1, 8)], arm):
        fcfg[n] = float(v)
    T_chk = fk_pose(fu, fcfg, "base_link", "base")
    dp = np.linalg.norm(T_chk[:3, 3] - T_x_wrist[:3, 3])
    dr = np.degrees(np.linalg.norm(
        R.from_matrix(T_chk[:3, :3] @ T_x_wrist[:3, :3].T).as_rotvec()))
    print(f"  FK check vs target: pos err {dp*1000:.2f} mm, rot err {dr:.3f} deg")


if __name__ == "__main__":
    main()
