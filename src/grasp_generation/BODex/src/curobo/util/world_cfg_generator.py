import numpy as np
import transforms3d
from glob import glob
import os
import json

from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate

from curobo.util.logger import log_warn
from curobo.util_file import load_json, load_scene_cfg, join_path, get_assets_path

from rsslib.path import bodex_path

def scenecfg2worldcfg(scene_cfg):
    world_cfg = {}
    for obj_name, obj_cfg in scene_cfg["scene"].items():
        if obj_cfg["type"] == "rigid_object":
            if "mesh" not in world_cfg:
                world_cfg["mesh"] = {}
            world_cfg["mesh"][scene_cfg["scene_id"] + obj_name] = {
                "scale": obj_cfg["scale"],
                "pose": obj_cfg["pose"],
                "file_path": obj_cfg["file_path"],
                "urdf_path": obj_cfg["urdf_path"],
            }
        elif obj_cfg["type"] == "plane":
            if "cuboid" not in world_cfg:
                world_cfg["cuboid"] = {}
            assert obj_cfg["pose"][3] == 1
            world_cfg["cuboid"]["table"] = {
                "dims": [5.0, 5.0, 0.2],
                "pose": [0.0, 0.0, -0.1, 1, 0, 0, 0.0],
            }
        else:
            raise NotImplementedError("Unsupported object type")
    return world_cfg
    
class WorldConfigDataset(Dataset):

    def __init__(self, type, template_path, start, end):
        assert type == "scene_cfg"
        scene_cfg_path = join_path(get_assets_path(), template_path)
        self.scene_path_lst = np.random.permutation(sorted(glob(scene_cfg_path)))[start:end]
        log_warn(
            f"From {scene_cfg_path} get {len(self.scene_path_lst)} scene cfgs. Start: {start}, End: {end}."
        )
        return

    def __len__(self):
        return len(self.scene_path_lst)

    def __getitem__(self, index):
        scene_path = self.scene_path_lst[index]
        scene_cfg = load_scene_cfg(scene_path)
        scene_id = scene_cfg["scene_id"]

        obj_name = scene_cfg["task"]["obj_name"]
        obj_cfg = scene_cfg["scene"][obj_name]
        obj_scale = obj_cfg["scale"]
        obj_pose = obj_cfg["pose"]

        json_data = load_json(obj_cfg["info_path"])
        obj_rot = transforms3d.quaternions.quat2mat(obj_pose[3:])
        gravity_center = obj_pose[:3] + obj_rot @ json_data["gravity_center"] * obj_scale
        obb_length = np.linalg.norm(obj_scale * json_data["obb"]) / 2
        return {
            "scene_path": scene_path,
            "world_cfg": scenecfg2worldcfg(scene_cfg),
            "manip_name": scene_id + obj_name,
            "obj_gravity_center": gravity_center,
            "obj_obb_length": obb_length,
            "save_prefix": f"{scene_id}_",
        }

class ParadexDataset(Dataset):
    def __init__(self, obj_list=[], scene_type_list=[], batch_size=1, version="", seed_offset=0, output_dir=None, obj_root_dir=None, scene_filter=None, hand="allegro"):
        """
        scene_filter: optional dict {scene_type: set(scene_id_no_ext)}. When set,
        restricts the dataset to scenes whose filename (without .json) is in
        scene_filter[scene_type]. Used by the adaptive orchestrator for
        per-scene retry batching.

        hand: scenes are hand-specific (adaptive gap differs per hand) and read
        from get_scene_dir(hand, obj, scene_type). obj_root_dir still supplies
        the hand-agnostic mesh/OBB info (processed_data/).
        """
        from autodex.utils.path import obj_path as _autodex_obj_path, get_scene_dir
        self._get_scene_dir = get_scene_dir
        self.hand = hand
        self.obj_root_dir = obj_root_dir or _autodex_obj_path
        self.output_dir = output_dir

        self.obj_list = obj_list if len(obj_list) != 0 else os.listdir(self.obj_root_dir)
        self.scene_list = []

        scene_cnt = 0
        for obj_name in self.obj_list:
            obj_dir = os.path.join(self.obj_root_dir, obj_name)
            obj_info_dict = load_json(os.path.join(obj_dir, "processed_data", "info", "simplified.json"))

            obb_length = np.linalg.norm(obj_info_dict["obb"]) / 2
            gravity_center = np.array(obj_info_dict["gravity_center"])

            for scene_type in scene_type_list:
                scene_type_dir = self._get_scene_dir(self.hand, obj_name, scene_type)
                if not os.path.isdir(scene_type_dir):
                    continue
                scene_list = os.listdir(scene_type_dir)
                allowed = scene_filter.get(scene_type) if scene_filter else None

                for scene_name in scene_list:
                    if allowed is not None and scene_name.split('.')[0] not in allowed:
                        continue
                    save_path = os.path.join(self.output_dir or bodex_path, version, obj_name, scene_type, scene_name.split('.')[0])
                    skip = True
                    for i in range(batch_size):
                        if not os.path.exists(os.path.join(save_path, f"{seed_offset + i}", "grasp_pose.npy")):
                            # print(f"Not exist: {os.path.join(save_path, f'{seed_offset + i}', 'grasp_pose.npy')}")
                            skip = False
                            break
                    if skip:
                        # print(f"Skip existing scene: {save_path}")
                        continue
                    scene_path = os.path.join(scene_type_dir, scene_name)
                    scene_cfg = json.load(open(scene_path, "r"))

                    pose = np.eye(4)
                    cart = scene_cfg["scene"]["mesh"]["target"]["pose"]
                    pose[:3, :3] = transforms3d.quaternions.quat2mat(cart[3:])
                    pose[:3, 3] = cart[:3]

                    gravity_center_transformed = (pose[:3, :3] @ gravity_center) + pose[:3, 3]

                    mesh_list = list(scene_cfg["scene"]["mesh"].keys())
                    for mesh_name in mesh_list:
                        scene_cfg["scene"]["mesh"][obj_name+"_" + mesh_name] = scene_cfg["scene"]["mesh"].pop(mesh_name)
                        
                    self.scene_list.append({
                        "scene_path": obj_name,
                        "world_cfg": scene_cfg["scene"],
                        "manip_name": obj_name + "_target",
                        "obj_gravity_center": gravity_center_transformed,
                        "obj_obb_length": obb_length,
                        "save_prefix": f"{obj_name}/{scene_type}/{scene_name.split('.')[0]}",
                        # "info": scene_cfg['meta'],
                        "type": scene_type,
                    })
                    scene_cnt += 1

    def __len__(self):
        return len(self.scene_list)

    def __getitem__(self, index):
        return self.scene_list[index]
    
def _world_config_collate_fn(list_data):
    world_cfg_lst = [i.pop("world_cfg") for i in list_data]
    ret_data = default_collate(list_data)
    if world_cfg_lst is not None:
        ret_data["world_cfg"] = world_cfg_lst
    return ret_data


def get_world_config_dataloader(configs, batch_size, seed_num, version="", seed_offset=0, output_dir=None, obj_root_dir=None, scene_filter=None, hand="allegro"):
    if configs["type"] == "scene_cfg":
        dataset = WorldConfigDataset(**configs)

    elif configs["type"] == "paradex":
        dataset = ParadexDataset(configs.get("obj_list", []), configs.get("scene_type", []), seed_num, version, seed_offset, output_dir=output_dir, obj_root_dir=obj_root_dir, scene_filter=scene_filter, hand=hand)

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, collate_fn=_world_config_collate_fn
    )
    return dataloader
