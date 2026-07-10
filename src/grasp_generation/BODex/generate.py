# Standard Library
import time
import logging
from typing import Dict
import os

# Third Party
import torch
import numpy as np
import argparse
import tqdm

# CuRobo
from curobo.geom.sdf.world import WorldConfig
from curobo.wrap.reacher.grasp_solver import GraspSolver, GraspSolverConfig
from curobo.util.world_cfg_generator import get_world_config_dataloader
from curobo.util.logger import setup_logger
from curobo.util_file import (
    get_manip_configs_path,
    join_path,
    load_yaml,
)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import random

import transforms3d as t3d

def cart2se3(cart):
    """7D [x,y,z, qw,qx,qy,qz] -> 4x4 SE3 matrix."""
    ret = np.eye(4)
    ret[:3, 3] = cart[0:3]
    ret[:3, :3] = t3d.quaternions.quat2mat(cart[3:7])
    return ret

# Resolve repo root (BODex lives at src/grasp_generation/BODex/)
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))


def _setup_logging(output_dir: str, log_dir: str) -> logging.Logger:
    """File + console logger. Log file: {log_dir}/generate.log"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("bodex")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(os.path.join(log_dir, "generate.log"), mode="a")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
        ch = logging.StreamHandler()
        ch.setFormatter(fmt)
        logger.addHandler(ch)
    return logger


def save_bodex_output(output_dir: str, save_data: Dict, seed_offset: int = 0):
    batch_size = save_data["robot_pose"].shape[0]
    for b in tqdm.tqdm(range(batch_size), desc="Saving BODex outputs"):
        output_path = os.path.join(output_dir, save_data['save_prefix'][b])
        os.makedirs(output_path, exist_ok=True)

        num_seed = save_data["robot_pose"].shape[1]
        obj_name = save_data['manip_name'][b]

        for ns in tqdm.tqdm(range(num_seed), desc=f"Saving seeds for batch {b}"):
            seed_id = seed_offset + ns
            if os.path.exists(os.path.join(output_path, str(seed_id))):
                continue
            os.makedirs(os.path.join(output_path, str(seed_id)), exist_ok=True)

            wrist_se3 = cart2se3(save_data["robot_pose"][b, ns, 0, :7])
            pregrasp_pose = save_data["robot_pose"][b, ns, 0, 7:]
            grasp_pose = save_data["robot_pose"][b, ns, 1, 7:]

            obj_se3 = cart2se3(save_data['world_cfg'][b]['mesh'][obj_name]['pose'])
            bodex_info = {
                "contact_point": save_data["contact_point"][b, ns],
                "contact_frame": save_data["contact_frame"][b, ns],
                "contact_force": save_data["contact_force"][b, ns],
                "grasp_error": save_data["grasp_error"][b, ns],
                "dist_error": save_data["dist_error"][b, ns],
                "success": save_data["success"][b, ns],
            }

            np.save(os.path.join(output_path, str(seed_id), "wrist_se3.npy"), np.linalg.inv(obj_se3) @ wrist_se3)
            np.save(os.path.join(output_path, str(seed_id), "pregrasp_pose.npy"), pregrasp_pose)
            np.save(os.path.join(output_path, str(seed_id), "grasp_pose.npy"), grasp_pose)
            np.save(os.path.join(output_path, str(seed_id), "bodex_info.npy"), bodex_info)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--manip_cfg_file", type=str, default="fc_leap.yml")
    parser.add_argument("-w", "--parallel_world", type=int, default=20)
    parser.add_argument("-o", "--output_dir", type=str, default=None,
                        help="Output directory (default: bodex_outputs/{exp_name})")
    parser.add_argument("--obj_list_file", type=str, default=None,
                        help="Text file with object names (one per line). Overrides config obj_list.")
    parser.add_argument("--obj_root_dir", type=str, default=None,
                        help="Override object root dir (default: ~/shared_data/RSS2026_Mingi/object/paradex)")
    parser.add_argument("--seed_offset", type=int, default=0,
                        help="Start index for saved seed dirs (default 0). Use to append additional seeds without overwriting existing ones.")
    parser.add_argument("--seed_num", type=int, default=None,
                        help="Override seed_num from config (number of grasp seeds per scene).")
    parser.add_argument("--exp_name", type=str, default=None,
                        help="Override exp_name from config (output subdir under bodex_outputs/{robot}/).")
    parser.add_argument("--scene_type", type=str, nargs="+", default=None,
                        help="Override scene_type list from config (restrict to these scene types).")
    parser.add_argument("--scene_filter_file", type=str, default=None,
                        help="JSON file mapping {scene_type: [scene_id, ...]} to restrict dataset to specific scenes. "
                             "Used by adaptive orchestrator for per-scene retry.")
    parser.add_argument("--task_f", type=float, nargs=3, default=None,
                        help="Override grasp_cfg.task_dict.f (primary wrench direction, world frame). "
                             "E.g. --task_f 0 0 -1 for gravity.")
    parser.add_argument("--task_gamma", type=float, default=None,
                        help="Override grasp_cfg.task_dict.gamma (robustness cone half-angle deg). "
                             "E.g. --task_gamma 30 for narrow cone around f.")
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed (numpy / torch / random).")

    setup_logger("warn")

    args = parser.parse_args()
    manip_config_data = load_yaml(join_path(get_manip_configs_path(), args.manip_cfg_file))

    if args.seed_num is not None:
        manip_config_data["seed_num"] = args.seed_num
    if args.exp_name is not None:
        manip_config_data["exp_name"] = args.exp_name
    if args.scene_type is not None:
        manip_config_data["world"]["scene_type"] = args.scene_type
    if args.task_f is not None:
        manip_config_data.setdefault("grasp_cfg", {}).setdefault("task_dict", {})["f"] = list(args.task_f)
    if args.task_gamma is not None:
        manip_config_data.setdefault("grasp_cfg", {}).setdefault("task_dict", {})["gamma"] = args.task_gamma

    # Override obj_list from file if provided
    if args.obj_list_file:
        with open(args.obj_list_file) as f:
            obj_list = [line.strip() for line in f if line.strip() and not line.startswith("#")]
        manip_config_data["world"]["obj_list"] = obj_list

    exp_name = manip_config_data["exp_name"]
    robot_name = manip_config_data["robot_file"].replace(".yml", "")  # e.g. "allegro", "inspire"

    # Output under repo root: bodex_outputs/{robot}/{version}
    save_dir = args.output_dir or os.path.join(REPO_ROOT, "bodex_outputs", robot_name, exp_name)
    log_dir = os.path.join(REPO_ROOT, "logging", "grasp_generation")
    logger = _setup_logging(save_dir, log_dir)

    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)

    scene_filter = None
    if args.scene_filter_file is not None:
        import json as _json
        with open(args.scene_filter_file) as _f:
            scene_filter = {st: set(ids) for st, ids in _json.load(_f).items()}

    # output_dir for skip check: save_dir already includes exp_name,
    # but ParadexDataset prepends version again, so pass parent dir
    save_dir_parent = os.path.dirname(save_dir)
    world_generator = get_world_config_dataloader(
        manip_config_data["world"], args.parallel_world,
        manip_config_data["seed_num"], exp_name,
        seed_offset=args.seed_offset,
        output_dir=save_dir_parent,
        obj_root_dir=args.obj_root_dir,
        scene_filter=scene_filter,
        hand=robot_name,
    )

    logger.info(f"START config={args.manip_cfg_file} exp={exp_name} parallel={args.parallel_world} output={save_dir}")

    tst = time.time()
    grasp_solver = None
    n_scenes = 0

    for world_info_dict in tqdm.tqdm(world_generator):
        sst = time.time()
        obj_names = world_info_dict["manip_name"]
        n_scenes += len(obj_names)

        if grasp_solver is None:
            grasp_config = GraspSolverConfig.load_from_robot_config(
                world_model=world_info_dict["world_cfg"],
                manip_name_list=obj_names,
                manip_config_data=manip_config_data,
                obj_gravity_center=world_info_dict["obj_gravity_center"],
                obj_obb_length=world_info_dict["obj_obb_length"],
                use_cuda_graph=False,
                store_debug=False,
            )
            grasp_solver = GraspSolver(grasp_config)
            world_info_dict["world_model"] = grasp_solver.world_coll_checker.world_model
        else:
            world_info_dict["world_model"] = [
                WorldConfig.from_dict(world_cfg) for world_cfg in world_info_dict["world_cfg"]
            ]
            grasp_solver.update_world(
                world_info_dict["world_model"],
                world_info_dict["obj_gravity_center"],
                world_info_dict["obj_obb_length"],
                obj_names,
            )

        result = grasp_solver.solve_batch_env(return_seeds=grasp_solver.num_seeds)

        n_success = result.success.sum().item()
        n_total = result.success.numel()
        elapsed = time.time() - sst

        world_info_dict["robot_pose"] = result.solution.detach().cpu().numpy()
        world_info_dict["contact_point"] = result.contact_point.detach().cpu().numpy()
        world_info_dict["contact_frame"] = result.contact_frame.detach().cpu().numpy()
        world_info_dict["contact_force"] = result.contact_force.detach().cpu().numpy()
        world_info_dict["grasp_error"] = result.grasp_error.detach().cpu().numpy()
        world_info_dict["dist_error"] = result.dist_error.detach().cpu().numpy()
        world_info_dict["success"] = result.success.detach().cpu().numpy()

        save_bodex_output(save_dir, world_info_dict, seed_offset=args.seed_offset)

        logger.info(f"BATCH objects={obj_names} success={n_success}/{n_total} time={elapsed:.1f}s")

    total_time = time.time() - tst
    logger.info(f"DONE scenes={n_scenes} total_time={total_time:.1f}s")
