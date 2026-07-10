import os
import glob
import json
import numpy as np
import trimesh
import transforms3d
from scipy.spatial.transform import Rotation as R
import shapely.geometry as geom

from paradex.utils.path import shared_dir
from paradex.visualization.visualizer.viser import ViserViewer
from paradex.robot.robot_wrapper import RobotWrapper
from paradex.calibration.utils import load_current_C2R
from rsslib.conversion import se32cart, cart2se3
from rsslib.path import candidate_path, obj_path, urdf_path, project_dir
from autodex.utils.path import get_scene_dir

def parse_allegro(allegro_traj):
    ret_allegro_traj = np.zeros((allegro_traj.shape[0], 16))
    ret_allegro_traj[:, :4] = allegro_traj[:, 4:8]
    ret_allegro_traj[:, 4:8] = allegro_traj[:, 12:16]
    ret_allegro_traj[:, 8:12] = allegro_traj[:, :4]
    # ret_allegro_traj[:, 12:16] = allegro_traj[:, 8:12]
    # ret_allegro_traj[:, 12:16] = allegro_traj[:, 8:12]
    ret_allegro_traj[:, 12:16] = allegro_traj[:, 8:12]

    # ret_allegro_traj[:, :12] = allegro_traj[:, :12]
    return ret_allegro_traj

class Renderer(ViserViewer):
    def __init__(self, version, obj_name, exp_idx, obj_T):
        super().__init__()
        exp_dir = os.path.join(
            project_dir, "experiment", version, obj_name, exp_idx
        )
        xarm_traj = np.load(os.path.join(exp_dir, "raw","arm",  "position.npy"))[:7500]
        hand_traj = np.load(os.path.join(exp_dir, "raw", "hand", "position.npy"))[:7500]
        hand_traj = parse_allegro(hand_traj)
        print(xarm_traj.shape, hand_traj.shape)
        self.add_robot("asdf", os.path.join(urdf_path, "xarm_allegro.urdf"))

        obj_mesh = trimesh.load(os.path.join(obj_path, obj_name, "raw_mesh", f"{obj_name}.obj"))
        c2r = load_current_C2R()
        pose_path = os.path.join(
            exp_dir, "outputs",
            f"{obj_name}_pose", "optimized_pose_world.txt"
        )
        
        obj_init_T = np.loadtxt(pose_path).reshape(4, 4)
        obj_init_T = np.linalg.inv(c2r) @ obj_init_T

        xarm = RobotWrapper(os.path.join(urdf_path, "xarm_allegro.urdf"))
        print(xarm.link_names)
        wrist_se3 = xarm.compute_forward_kinematics(
            np.concatenate([xarm_traj[809], hand_traj[809]]), ["base_link"]
        )["base_link"]

        wrist_se32 = xarm.compute_forward_kinematics(
            np.concatenate([xarm_traj[-1], hand_traj[-1]]), ["base_link"]
        )["base_link"]

        offset = np.linalg.inv(wrist_se3) @ obj_init_T
        obj_final_T = wrist_se32 @ offset
        obj_final_T[2, 3] += 0.02
        obj_final_T[0, 3] += 0.01
        self.add_trimesh("object", obj_mesh, obj_init_T)
        
        self.add_traj("robot_traj", {"asdf": np.concatenate([xarm_traj, hand_traj], axis=1)})
        
        self.add_floor(0.0)
        
if __name__ == "__main__":
    version = "fourcam"
    obj_name = "attached_container"
    exp_idx = "20260130_022138"

    scene_type = "box"
    scene_idx = "3"
    grasp_idx = "95"

    scene = json.load(open(os.path.join(get_scene_dir("allegro", obj_name, scene_type), f"{scene_idx}.json")))["scene"]
    obj_T = cart2se3(np.array(scene["mesh"]["target"]["pose"]))
    
    vis = Renderer(version, obj_name, exp_idx, obj_T)
    vis.start_viewer()