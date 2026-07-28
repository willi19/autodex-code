#!/usr/bin/env python3
"""Build the FR3 + inspire RIGHT hand robot description / curobo configs.

Composition (both halves come from already-validated sources — nothing mirrored
by hand):
  arm    : fr3_inspire_left.urdf          (fr3_link0..8, fr3_joint1..8)
  hand   : xarm_inspire.urdf              (base_link + right_* subtree, true
                                           right-hand geometry + mimic)
  mount  : flange_to_hand, parent fr3_link8 -> child base_link
           LEFT is rpy=(-3.14159,0,0.7854) xyz=(0,0,0.04); the right hand
           attaches the same way but rotated 180 deg about the hand's own z:
           R_right = R_left @ Rz(pi)  ->  rpy=(3.14159, 0, -2.35619)
  spheres: fr3_link0..7 from spheres/fr3_inspire_left.yml
           base_link + right_* from spheres/xarm_inspire.yml

Writes into the repo, then (with --deploy) copies to the NAS content dir that
GraspPlanner actually loads from (robot_configs_path / project_dir assets).
"""
import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation as R

REPO = Path("/home/mingi/AutoDex")
ASSETS = REPO / "autodex/planner/src/curobo/content/assets/robot"
CONFIGS = REPO / "autodex/planner/src/curobo/content/configs/robot"

LEFT_URDF = ASSETS / "fr3_inspire_left_description/fr3_inspire_left.urdf"
XARM_URDF = ASSETS / "inspire_description/xarm_inspire.urdf"
HAND_MESH_SRC = ASSETS / "inspire_description/meshes"
ARM_MESH_SRC = ASSETS / "fr3_inspire_left_description/meshes/robot_arms"

DEST_DESC = ASSETS / "fr3_inspire_description"
DEST_URDF = DEST_DESC / "fr3_inspire.urdf"

LEFT_MOUNT_RPY = (-3.14159, 0.0, 0.7854)
MOUNT_XYZ = (0.0, 0.0, 0.04)

HAND_LINKS = ["base_link"] + [
    f"right_{f}_{i}" for f, n in
    [("thumb", 4), ("index", 2), ("middle", 2), ("ring", 2), ("little", 2)]
    for i in range(1, n + 1)
]


def right_mount_rpy():
    """R_right = R_left @ Rz(pi)  (180 deg about the hand's own z)."""
    rr = R.from_euler("xyz", LEFT_MOUNT_RPY) * R.from_euler("z", np.pi)
    return tuple(rr.as_euler("xyz"))


def build_urdf():
    left = ET.parse(LEFT_URDF).getroot()
    xarm = ET.parse(XARM_URDF).getroot()

    robot = ET.Element("robot", {"name": "fr3_inspire"})

    # --- arm half: keep everything that is not the left hand / mount / tcp ---
    drop_link = lambda n: (n == "base_link" or n.startswith("left_") or n == "hand_tcp")
    drop_joint = lambda n: (n.startswith("left_") or n in ("flange_to_hand", "hand_tcp_joint"))
    n_arm_l = n_arm_j = 0
    for el in left:
        if el.tag == "link" and drop_link(el.get("name", "")):
            continue
        if el.tag == "joint" and drop_joint(el.get("name", "")):
            continue
        if el.tag in ("gazebo", "transmission"):
            continue
        robot.append(el)
        n_arm_l += el.tag == "link"
        n_arm_j += el.tag == "joint"

    # --- mount ---
    rpy = right_mount_rpy()
    j = ET.SubElement(robot, "joint", {"name": "flange_to_hand", "type": "fixed"})
    ET.SubElement(j, "parent", {"link": "fr3_link8"})
    ET.SubElement(j, "child", {"link": "base_link"})
    ET.SubElement(j, "origin", {"rpy": "%.5f %.5f %.5f" % rpy,
                                "xyz": "%g %g %g" % MOUNT_XYZ})

    # --- hand half: right subtree from xarm_inspire, mesh paths rewritten ---
    n_h_l = n_h_j = 0
    for el in xarm:
        name = el.get("name", "")
        if el.tag == "link" and name in HAND_LINKS:
            for m in el.iter("mesh"):
                fn = m.get("filename", "")
                if fn.startswith("meshes/"):
                    m.set("filename", "./meshes/robot_ee/inspire_right/" + fn[len("meshes/"):])
            robot.append(el)
            n_h_l += 1
        elif el.tag == "joint" and name.startswith("right_"):
            robot.append(el)
            n_h_j += 1

    # tcp for parity with the left description
    ET.SubElement(robot, "link", {"name": "hand_tcp"})
    jt = ET.SubElement(robot, "joint", {"name": "hand_tcp_joint", "type": "fixed"})
    ET.SubElement(jt, "origin", {"rpy": "0 0 0", "xyz": "0 0 -0.12"})
    ET.SubElement(jt, "parent", {"link": "base_link"})
    ET.SubElement(jt, "child", {"link": "hand_tcp"})

    DEST_DESC.mkdir(parents=True, exist_ok=True)
    ET.indent(robot, space="  ")
    ET.ElementTree(robot).write(DEST_URDF, encoding="utf-8", xml_declaration=True)
    print(f"urdf   -> {DEST_URDF}")
    print(f"         arm: {n_arm_l} links / {n_arm_j} joints | "
          f"hand: {n_h_l} links / {n_h_j} joints | mount rpy={tuple(round(v,5) for v in rpy)}")


def copy_meshes():
    arm_dst = DEST_DESC / "meshes/robot_arms"
    hand_dst = DEST_DESC / "meshes/robot_ee/inspire_right"
    if arm_dst.exists():
        shutil.rmtree(arm_dst)
    shutil.copytree(ARM_MESH_SRC, arm_dst)
    hand_dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for stl in list(HAND_MESH_SRC.glob("right_*.STL")) + [HAND_MESH_SRC / "base_link.STL"]:
        shutil.copy2(stl, hand_dst / stl.name)
        n += 1
    print(f"meshes -> {arm_dst} (arm) + {hand_dst} ({n} hand STL)")


def build_spheres():
    arm = yaml.safe_load((CONFIGS / "spheres/fr3_inspire_left.yml").read_text())
    hand = yaml.safe_load((CONFIGS / "spheres/xarm_inspire.yml").read_text())
    out = {"collision_spheres": {}}
    a_src = arm["collision_spheres"] if "collision_spheres" in arm else arm
    h_src = hand["collision_spheres"] if "collision_spheres" in hand else hand
    for k, v in a_src.items():
        if k.startswith("fr3_link"):
            out["collision_spheres"][k] = v
    for k, v in h_src.items():
        if k == "base_link" or k.startswith("right_"):
            out["collision_spheres"][k] = v
    dst = CONFIGS / "spheres/fr3_inspire.yml"
    dst.write_text(yaml.safe_dump(out, sort_keys=False))
    arm_n = sum(1 for k in out["collision_spheres"] if k.startswith("fr3_"))
    hand_n = len(out["collision_spheres"]) - arm_n
    print(f"spheres-> {dst}  ({arm_n} arm links + {hand_n} hand links)")


def build_yml():
    cfg = yaml.safe_load((CONFIGS / "fr3_inspire_left.yml").read_text())
    k = cfg["robot_cfg"]["kinematics"]
    k["urdf_path"] = "robot/fr3_inspire_description/fr3_inspire.urdf"
    k["asset_root_path"] = "robot/fr3_inspire_description"
    k["collision_spheres"] = "spheres/fr3_inspire.yml"

    def swap(x):
        if isinstance(x, str):
            return x.replace("left_", "right_")
        if isinstance(x, list):
            return [swap(i) for i in x]
        if isinstance(x, dict):
            return {swap(a): swap(b) for a, b in x.items()}
        return x

    for f in ("link_names", "collision_link_names", "mesh_link_names",
              "self_collision_ignore", "self_collision_buffer"):
        k[f] = swap(k[f])
    k["cspace"]["joint_names"] = swap(k["cspace"]["joint_names"])
    dst = CONFIGS / "fr3_inspire.yml"
    dst.write_text("## FR3 + Inspire RIGHT hand (generated)\n"
                   + yaml.safe_dump(cfg, sort_keys=False))
    print(f"yml    -> {dst}")
    print(f"         joints: {k['cspace']['joint_names']}")


def deploy():
    nas_cfg = Path.home() / "shared_data/AutoDex/content/configs/robot"
    nas_ast = Path.home() / "shared_data/AutoDex/content/assets/robot"
    shutil.copy2(CONFIGS / "fr3_inspire.yml", nas_cfg / "fr3_inspire.yml")
    (nas_cfg / "spheres").mkdir(parents=True, exist_ok=True)
    shutil.copy2(CONFIGS / "spheres/fr3_inspire.yml", nas_cfg / "spheres/fr3_inspire.yml")
    dst = nas_ast / "fr3_inspire_description"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(DEST_DESC, dst)
    print(f"deploy -> {nas_cfg}/fr3_inspire.yml + spheres/ + {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--deploy", action="store_true", help="also copy to NAS content dir")
    a = ap.parse_args()
    build_urdf()
    copy_meshes()
    build_spheres()
    build_yml()
    if a.deploy:
        deploy()
