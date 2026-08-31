"""Sim filter: test BODex grasps in MuJoCo, extract passing candidates.

Usage:
    python src/grasp_generation/sim_filter/run_sim_filter.py --hand allegro --version v3 --obj attached_container
    python src/grasp_generation/sim_filter/run_sim_filter.py --hand allegro --version v3 --obj attached_container --viewer
"""

import os
import json
import shutil
import argparse
import numpy as np
from tqdm import tqdm
import transforms3d.quaternions as tq

import mujoco
from autodex.simulator.hand_object import MjHO
from autodex.simulator.rot_util import np_get_delta_qpos
from autodex.utils.conversion import se32cart, cart2se3
from autodex.utils.path import get_candidate_path, get_scene_dir
from autodex.planner.planner import GraspPlanner, _to_curobo_world

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))

FORCE_DIRECTIONS = np.array([
    [-1,0,0, 0,0,0], [1,0,0, 0,0,0],
    [0,-1,0, 0,0,0], [0,1,0, 0,0,0],
    [0,0,-1, 0,0,0], [0,0,1, 0,0,0],
], dtype=float)

OBJ_MASS = 0.1
FORCE_SCALE = 1.0
TRANS_THRE = 0.05
ANGLE_THRE = 15.0
MAX_PENE = 0.01

HAND_PATHS = {
    "allegro": {
        "path": os.path.join(REPO_ROOT, "src", "grasp_generation", "sim_filter", "assets", "hand", "allegro", "right_hand.xml"),
        "weld_body": "world",
    },
    "inspire": {
        "path": os.path.join(REPO_ROOT, "src", "grasp_generation", "BODex", "src", "curobo", "content", "assets", "robot", "inspire_description", "inspire_hand_right.urdf"),
        "weld_body": "wrist",
    },
    "inspire_left": {
        "path": os.path.join(REPO_ROOT, "src", "grasp_generation", "BODex", "src", "curobo", "content", "assets", "robot", "inspire_description", "inspire_hand_left.urdf"),
        "weld_body": "wrist",
    },
    "inspire_f1": {
        "path": os.path.join(REPO_ROOT, "src", "grasp_generation", "BODex", "src", "curobo", "content", "assets", "robot", "inspire_f1_description", "inspire_f1_hand_right.urdf"),
        "weld_body": "base_link",
    },
    "inspire_f1_left": {
        "path": os.path.join(REPO_ROOT, "src", "grasp_generation", "BODex", "src", "curobo", "content", "assets", "robot", "inspire_f1_left_description", "inspire_f1_hand_left.urdf"),
        "weld_body": "base_link",
    },
}

# R_delta (same as RSS_2026 base.py)
_q_delta = np.array([0, 1, 0, 1], dtype=np.float64)
_q_delta /= np.linalg.norm(_q_delta)
R_DELTA = tq.quat2mat(_q_delta)


def _expand_mimic_joints(joints, mimic_map):
    """Expand actuated joints to include mimic joints.
    mimic_map: list of (parent_idx, multiplier, offset) for each joint in order.
    None means it's an actuated joint (use as-is from input).
    """
    if mimic_map is None:
        return joints
    expanded = []
    act_idx = 0
    for entry in mimic_map:
        if entry is None:
            expanded.append(joints[act_idx])
            act_idx += 1
        else:
            parent_idx, mult, offset = entry
            expanded.append(joints[parent_idx] * mult + offset)
    return np.array(expanded)


# Inspire: 6 actuated → 12 total (with mimic)
# Joint order in URDF: thumb1, thumb2, thumb3(mimic thumb2*0.6), thumb4(mimic thumb2*0.8),
#   index1, index2(mimic index1*1.05), middle1, middle2(mimic middle1*1.05),
#   ring1, ring2(mimic ring1*1.05), little1, little2(mimic little1*1.18)
INSPIRE_MIMIC_MAP = [
    None,           # thumb1 (act 0)
    None,           # thumb2 (act 1)
    (1, 0.60, 0),   # thumb3 = thumb2 * 0.6
    (1, 0.80, 0),   # thumb4 = thumb2 * 0.8
    None,           # index1 (act 2)
    (2, 1.05, 0),   # index2 = index1 * 1.05
    None,           # middle1 (act 3)
    (3, 1.05, 0),   # middle2 = middle1 * 1.05
    None,           # ring1 (act 4)
    (4, 1.05, 0),   # ring2 = ring1 * 1.05
    None,           # little1 (act 5)
    (5, 1.18, 0),   # little2 = little1 * 1.18
]

# Inspire F1 (RH56F1): same 12-joint URDF order as inspire, different multipliers.
# URDF order: thumb1, thumb2, thumb3(mimic thumb2*1.2953), thumb4(mimic thumb2*1.1610),
#   index1, index2(mimic index1*1.1545), middle1, middle2(mimic middle1*1.1545),
#   ring1, ring2(mimic ring1*1.1545), little1, little2(mimic little1*1.1545)
INSPIRE_F1_MIMIC_MAP = [
    None,             # thumb1 (act 0)
    None,             # thumb2 (act 1)
    (1, 1.2953, 0),   # thumb3 = thumb2 * 1.2953
    (1, 1.1610, 0),   # thumb4 = thumb2 * 1.1610
    None,             # index1 (act 2)
    (2, 1.1545, 0),   # index2 = index1 * 1.1545
    None,             # middle1 (act 3)
    (3, 1.1545, 0),   # middle2 = middle1 * 1.1545
    None,             # ring1 (act 4)
    (4, 1.1545, 0),   # ring2 = ring1 * 1.1545
    None,             # little1 (act 5)
    (5, 1.1545, 0),   # little2 = little1 * 1.1545
]


def eval_single_grasp(mj, wrist_se3, pregrasp_joints, grasp_joints, mimic_map=None, apply_r_delta=True, gravity_dir=None):
    """Returns (success, traj_data).

    gravity_dir: 3-vector — direction (in object canonical frame, which is also
      MuJoCo world frame here since obj is placed at identity pose) along which
      gravity acts on the object at its tabletop_pose. Defaults to -world-z
      (object at canonical orientation = upright).
    """
    # Expand mimic joints if needed
    pregrasp_joints = _expand_mimic_joints(pregrasp_joints, mimic_map)
    grasp_joints = _expand_mimic_joints(grasp_joints, mimic_map)

    # Coordinate transform (R_DELTA only for allegro — its XML palm has quat="0 1 0 1")
    w = wrist_se3.copy()
    if apply_r_delta:
        w[:3, :3] = w[:3, :3] @ np.linalg.inv(R_DELTA)
    wrist_cart = se32cart(w)

    squeeze_joints = grasp_joints * 2 - pregrasp_joints
    pregrasp_qpos = np.concatenate([wrist_cart, pregrasp_joints])
    grasp_qpos = np.concatenate([wrist_cart, grasp_joints])
    squeeze_qpos = np.concatenate([wrist_cart, squeeze_joints])
    obj_pose = np.array([0, 0, 0, 1, 0, 0, 0], dtype=float)

    traj = {"robot_qpos": [], "object_pose": [], "phase": [], "contacts": []}

    def _rec(phase):
        traj["robot_qpos"].append(mj.data.qpos[:-7].tolist())
        traj["object_pose"].append(mj.get_obj_pose().tolist())
        traj["phase"].append(phase)
        traj["contacts"].append(mj.get_contact_info())

    # 1. Reset + pregrasp
    mj.reset_pose_qpos(pregrasp_qpos, obj_pose)
    _rec("pregrasp")

    # 2. Pregrasp → Grasp (record each step)
    from autodex.simulator.rot_util import interplote_pose, interplote_qpos
    pose_interp = interplote_pose(pregrasp_qpos[:7], grasp_qpos[:7], 10)
    ctrl_interp = interplote_qpos(mj._qpos2ctrl(pregrasp_qpos), mj._qpos2ctrl(grasp_qpos), 10)
    for j in range(10):
        mj.data.mocap_pos[0] = pose_interp[j, :3]
        mj.data.mocap_quat[0] = pose_interp[j, 3:7]
        mj.data.ctrl[:] = ctrl_interp[j]
        mujoco.mj_forward(mj.model, mj.data)
        mj.control_hand_step(step_inner=10)
        _rec("grasp")

    # 3. Grasp → Squeeze (record each step)
    pose_interp = interplote_pose(grasp_qpos[:7], squeeze_qpos[:7], 10)
    ctrl_interp = interplote_qpos(mj._qpos2ctrl(grasp_qpos), mj._qpos2ctrl(squeeze_qpos), 10)
    for j in range(10):
        mj.data.mocap_pos[0] = pose_interp[j, :3]
        mj.data.mocap_quat[0] = pose_interp[j, 3:7]
        mj.data.ctrl[:] = ctrl_interp[j]
        mujoco.mj_forward(mj.model, mj.data)
        mj.control_hand_step(step_inner=10)
        _rec("squeeze")

    # 4. Force test — single direction = gravity in object frame at tabletop_pose
    g = np.asarray(gravity_dir if gravity_dir is not None else [0, 0, -1], dtype=float)
    fdir = np.zeros(6)
    fdir[:3] = g
    mj.reset_pose_qpos(pregrasp_qpos, obj_pose)
    mj.control_hand_with_interp(pregrasp_qpos, grasp_qpos)
    mj.control_hand_with_interp(grasp_qpos, squeeze_qpos)

    mj.set_ext_force_on_obj(fdir * FORCE_SCALE * OBJ_MASS)
    pre_obj = mj.get_obj_pose().copy()
    for _ in range(50):
        mj.control_hand_step(step_inner=10)
        _rec("force_gravity")
        cur_obj = mj.get_obj_pose()
        dp, da = np_get_delta_qpos(pre_obj, cur_obj)
        if dp > TRANS_THRE or da > ANGLE_THRE:
            mj.set_ext_force_on_obj(np.zeros(6))
            return False, traj
    mj.set_ext_force_on_obj(np.zeros(6))

    return True, traj


def run_sim_filter(hand, version, obj_name, bodex_root, candidate_root, viewer=False, coll_only=False, sim_only=False, obj_root_dir=None):
    bodex_obj_dir = os.path.join(bodex_root, obj_name)
    if not os.path.isdir(bodex_obj_dir):
        print(f"  No BODex outputs for {obj_name}")
        return 0, 0

    hand_cfg = HAND_PATHS.get(hand)
    if hand_cfg is None:
        print(f"  No hand config for {hand}")
        return 0, 0

    if obj_root_dir is not None:
        obj_data_path = obj_root_dir
    else:
        from autodex.utils.path import obj_path as obj_data_path

    # Group seeds by scene
    scene_seeds = {}  # (scene_type, scene_id) -> [(seed, seed_dir), ...]
    for scene_type in sorted(os.listdir(bodex_obj_dir)):
        std = os.path.join(bodex_obj_dir, scene_type)
        if not os.path.isdir(std):
            continue
        for scene_id in sorted(os.listdir(std)):
            sid = os.path.join(std, scene_id)
            if not os.path.isdir(sid):
                continue
            for seed in sorted(os.listdir(sid)):
                sd = os.path.join(sid, seed)
                if os.path.isdir(sd):
                    scene_seeds.setdefault((scene_type, scene_id), []).append((seed, sd))

    # --- Step 1: Scene collision check (cuRobo, GPU, fast) ---
    coll_valid = {}  # seed_dir -> bool (True = collision-free)

    if sim_only:
        # Load cached coll_valid from disk, skip cuRobo
        for (scene_type, scene_id), seeds in scene_seeds.items():
            for seed, sd in seeds:
                coll_file = os.path.join(sd, "coll_valid.npy")
                if os.path.exists(coll_file):
                    coll_valid[sd] = bool(np.load(coll_file))
                elif os.path.exists(os.path.join(sd, "sim_eval.json")):
                    coll_valid[sd] = True
                else:
                    coll_valid[sd] = False
    else:
        print(f"  [1/2] Collision check...")
        # Use local planner configs (shared_data NAS may be read-only)
        _local_configs = os.path.join(REPO_ROOT, "autodex", "planner", "src", "curobo", "content", "configs", "robot")
        hand_coll_cfgs = {
            "allegro": "allegro_floating.yml",
            "inspire": "inspire_floating.yml",
            "inspire_left": "inspire_left_floating.yml",
            "inspire_f1": "inspire_f1_floating.yml",
            "inspire_f1_left": "inspire_f1_left_floating.yml",
        }
        hand_coll_cfg = os.path.join(_local_configs, hand_coll_cfgs.get(hand, "inspire_floating.yml"))
        planner = GraspPlanner(hand_cfg_path=hand_coll_cfg)

        n_coll_pass = 0
        n_coll_total = 0

        for (scene_type, scene_id), seeds in tqdm(scene_seeds.items(), desc=f"  coll check", leave=False):
            scene_json = os.path.join(get_scene_dir(hand, obj_name, scene_type), f"{scene_id}.json")
            if not os.path.exists(scene_json):
                for seed, sd in seeds:
                    coll_valid[sd] = False
                    n_coll_total += 1
                continue

            scene_cfg = json.load(open(scene_json))["scene"]
            obj_se3 = cart2se3(scene_cfg["mesh"]["target"]["pose"])
            world_cfg = _to_curobo_world(scene_cfg)

            wrist_list = []
            pregrasp_list = []
            seed_dirs = []
            for seed, sd in seeds:
                coll_file = os.path.join(sd, "coll_valid.npy")
                if os.path.exists(coll_file):
                    cv = bool(np.load(coll_file))
                    coll_valid[sd] = cv
                    n_coll_total += 1
                    if cv:
                        n_coll_pass += 1
                    continue
                if os.path.exists(os.path.join(sd, "sim_eval.json")):
                    coll_valid[sd] = True
                    n_coll_total += 1
                    continue
                wrist_se3 = np.load(os.path.join(sd, "wrist_se3.npy"))
                pregrasp = np.load(os.path.join(sd, "pregrasp_pose.npy"))
                if not np.isfinite(wrist_se3).all() or not np.isfinite(pregrasp).all():
                    coll_valid[sd] = False
                    np.save(os.path.join(sd, "coll_valid.npy"), False)
                    n_coll_total += 1
                    continue
                wrist_world = obj_se3 @ wrist_se3
                wrist_list.append(wrist_world)
                pregrasp_list.append(pregrasp)
                seed_dirs.append(sd)

            if not wrist_list:
                continue

            wrist_arr = np.array(wrist_list)
            pregrasp_arr = np.array(pregrasp_list)

            try:
                collided = planner._check_collision(world_cfg, wrist_arr, pregrasp_arr)
            except Exception:
                import traceback; traceback.print_exc()
                collided = np.ones(len(wrist_arr), dtype=bool)

            for i, sd in enumerate(seed_dirs):
                cv = not collided[i]
                coll_valid[sd] = cv
                np.save(os.path.join(sd, "coll_valid.npy"), cv)
                n_coll_total += 1
                if cv:
                    n_coll_pass += 1

        print(f"  Collision: {n_coll_pass}/{n_coll_total} passed")

    # --- Step 1.5: Squeeze contact check + per-scene gravity_dir ---
    # Skip seeds whose squeeze pose doesn't make contact with the object —
    # if fingers don't touch object at squeeze, force closure is impossible.
    contact_valid = {}        # seed_dir -> bool (True = touches object at squeeze)
    gravity_dir_per_scene = {}  # (scene_type, scene_id) -> 3-vector in obj canonical frame
    if not sim_only:
        print(f"  [1.5/2] Squeeze contact check...")
        n_contact_pass = 0
        n_contact_total = 0
        for (scene_type, scene_id), seeds in tqdm(scene_seeds.items(), desc="  squeeze check", leave=False):
            scene_json = os.path.join(get_scene_dir(hand, obj_name, scene_type), f"{scene_id}.json")
            if not os.path.exists(scene_json):
                for seed, sd in seeds:
                    contact_valid[sd] = False
                continue
            scene_cfg = json.load(open(scene_json))["scene"]

            obj_se3 = cart2se3(scene_cfg["mesh"]["target"]["pose"])
            R = obj_se3[:3, :3]
            # world -z expressed in object canonical frame (object placed at identity in MuJoCo)
            gravity_dir_per_scene[(scene_type, scene_id)] = R.T @ np.array([0.0, 0.0, -1.0])

            obj_only_cfg = {"mesh": {"target": scene_cfg["mesh"]["target"]}, "cuboid": {}}
            obj_only_world = _to_curobo_world(obj_only_cfg)

            wrist_list, squeeze_list, sds = [], [], []
            for seed, sd in seeds:
                if not coll_valid.get(sd, False):
                    contact_valid[sd] = False
                    continue
                try:
                    wrist_se3 = np.load(os.path.join(sd, "wrist_se3.npy"))
                    pregrasp = np.load(os.path.join(sd, "pregrasp_pose.npy"))
                    grasp = np.load(os.path.join(sd, "grasp_pose.npy"))
                except FileNotFoundError:
                    contact_valid[sd] = False
                    continue
                squeeze = grasp * 2 - pregrasp
                wrist_list.append(obj_se3 @ wrist_se3)
                squeeze_list.append(squeeze)
                sds.append(sd)

            if not wrist_list:
                continue
            try:
                # World-only collision (ignores self-collision) — True means hand spheres touch object
                collided = planner._check_world_collision_only(obj_only_world, np.array(wrist_list), np.array(squeeze_list))
            except Exception:
                import traceback; traceback.print_exc()
                collided = np.zeros(len(wrist_list), dtype=bool)
            for i, sd in enumerate(sds):
                cv = bool(collided[i])
                contact_valid[sd] = cv
                n_contact_total += 1
                if cv:
                    n_contact_pass += 1
        print(f"  Squeeze contact: {n_contact_pass}/{n_contact_total} passed")
    else:
        # sim_only: assume all collision-pass seeds also have contact (skip filter)
        for sd in coll_valid:
            contact_valid[sd] = coll_valid[sd]

    if coll_only:
        return n_coll_total, n_coll_pass

    # --- Step 2: MuJoCo sim eval (only collision-free seeds) ---
    print(f"  [2/2] MuJoCo sim eval...")
    mj = MjHO(obj_name, hand_cfg["path"], weld_body_name=hand_cfg["weld_body"],
              obj_mass=OBJ_MASS, debug_viewer=viewer,
              obj_root_dir=obj_root_dir)

    all_seeds = []
    for (scene_type, scene_id), seeds in scene_seeds.items():
        for seed, sd in seeds:
            all_seeds.append((scene_type, scene_id, seed, sd))

    total = 0
    success_count = 0
    coll_skip = 0
    pbar = tqdm(total=len(all_seeds), desc=f"  {obj_name}")

    for scene_type, scene_id, seed, seed_dir in all_seeds:
        result_path = os.path.join(seed_dir, "sim_eval.json")

        # Skip already evaluated
        if os.path.exists(result_path):
            total += 1
            r = json.load(open(result_path))
            if r["success"]:
                success_count += 1
            pbar.update(1)
            pbar.set_postfix_str(f"succ={success_count} rate={success_count/total*100:.1f}%")
            continue

        # Skip collision-failed seeds (save result as fail)
        if not coll_valid.get(seed_dir, False):
            coll_skip += 1
            total += 1
            with open(result_path, "w") as f:
                json.dump({"success": False, "hand": hand, "version": version,
                           "reason": "scene_collision"}, f)
            pbar.update(1)
            pbar.set_postfix_str(f"succ={success_count} rate={success_count/total*100:.1f}%")
            continue

        # Skip seeds whose squeeze pose has no object contact (force closure impossible)
        if not contact_valid.get(seed_dir, False):
            total += 1
            with open(result_path, "w") as f:
                json.dump({"success": False, "hand": hand, "version": version,
                           "reason": "no_object_contact"}, f)
            pbar.update(1)
            pbar.set_postfix_str(f"succ={success_count} rate={success_count/total*100:.1f}%")
            continue

        wrist_se3 = np.load(os.path.join(seed_dir, "wrist_se3.npy"))
        pregrasp = np.load(os.path.join(seed_dir, "pregrasp_pose.npy"))
        grasp = np.load(os.path.join(seed_dir, "grasp_pose.npy"))

        if hand in ("inspire", "inspire_left"):
            mimic_map = INSPIRE_MIMIC_MAP
        elif hand in ("inspire_f1", "inspire_f1_left"):
            mimic_map = INSPIRE_F1_MIMIC_MAP
        else:
            mimic_map = None
        apply_r_delta = (hand == "allegro")  # R_DELTA is allegro-specific
        gravity_dir = gravity_dir_per_scene.get((scene_type, scene_id))
        try:
            succ, traj_data = eval_single_grasp(mj, wrist_se3, pregrasp, grasp,
                                                mimic_map=mimic_map,
                                                apply_r_delta=apply_r_delta,
                                                gravity_dir=gravity_dir)
        except Exception as e:
            import traceback
            traceback.print_exc()
            succ, traj_data = False, None

        with open(result_path, "w") as f:
            json.dump({"success": bool(succ), "hand": hand, "version": version}, f)

        if traj_data is not None:
            with open(os.path.join(seed_dir, "sim_traj.json"), "w") as f:
                json.dump(traj_data, f)

        total += 1
        if succ:
            success_count += 1
            cand_dir = os.path.join(candidate_root, obj_name, scene_type, scene_id, seed)
            os.makedirs(cand_dir, exist_ok=True)
            for fname in ["wrist_se3.npy", "pregrasp_pose.npy", "grasp_pose.npy", "bodex_info.npy"]:
                src = os.path.join(seed_dir, fname)
                if os.path.exists(src):
                    shutil.copy2(src, cand_dir)

        pbar.update(1)
        pbar.set_postfix_str(f"succ={success_count} rate={success_count/total*100:.1f}%")

    pbar.close()
    mj.close()
    return total, success_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", type=str, default="allegro")
    parser.add_argument("--version", type=str, default="v3")
    parser.add_argument("--obj", type=str, default=None)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (object-level)")
    parser.add_argument("--obj_root_dir", type=str, default=None,
                        help="Override object root dir (default: paradex from autodex.utils.path)")
    parser.add_argument("--candidate-root", type=str, default=None,
                        help="Candidate base directory. Defaults to the NAS runtime path for --hand.")
    parser.add_argument("--obj_list_file", type=str, default=None,
                        help="Object list file (default: src/grasp_generation/obj_list.txt)")
    args = parser.parse_args()

    if args.obj:
        obj_list = [args.obj]
    else:
        obj_list_file = args.obj_list_file or os.path.join(
            REPO_ROOT, "src", "grasp_generation", "obj_list.txt"
        )
        with open(obj_list_file) as f:
            obj_list = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    bodex_root = os.path.join(REPO_ROOT, "bodex_outputs", args.hand, args.version)
    # Candidate results are consumed by the robot runner from the NAS. Keeping
    # them under the source checkout made a newly generated pool invisible to
    # preflight and caused an avoidable manual copy step.
    candidate_base = (os.path.expanduser(args.candidate_root)
                      if args.candidate_root else get_candidate_path(args.hand))
    candidate_root = os.path.join(candidate_base, args.version)

    print(f"Hand: {args.hand}, Version: {args.version}, Workers: {args.workers}")

    if args.workers > 1 and not args.viewer:
        # Stage 1: Collision check for ALL objects (GPU, sequential)
        print("\n=== Stage 1: Collision check (all objects) ===")
        for obj_name in obj_list:
            run_sim_filter(args.hand, args.version, obj_name,
                           bodex_root, candidate_root, coll_only=True,
                           obj_root_dir=args.obj_root_dir)

        # Stage 2: MuJoCo sim eval (CPU, parallel)
        print(f"\n=== Stage 2: MuJoCo sim eval ({args.workers} workers) ===")
        from multiprocessing import Pool

        def _run_mujoco(obj_name):
            total, succ = run_sim_filter(args.hand, args.version, obj_name,
                                          bodex_root, candidate_root, sim_only=True,
                                          obj_root_dir=args.obj_root_dir)
            print(f"  {obj_name}: {succ}/{total} passed")
            return obj_name, total, succ

        with Pool(args.workers) as pool:
            results = pool.map(_run_mujoco, obj_list)
        total_all = sum(r[1] for r in results)
        succ_all = sum(r[2] for r in results)
        print(f"\nTotal: {succ_all}/{total_all} passed")
    else:
        for obj_name in obj_list:
            print(f"\n{obj_name}:")
            total, succ = run_sim_filter(args.hand, args.version, obj_name,
                                          bodex_root, candidate_root, viewer=args.viewer,
                                          obj_root_dir=args.obj_root_dir)
            print(f"  {succ}/{total} passed")
