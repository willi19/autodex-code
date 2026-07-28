"""Given a FIXED wrist 6D pose + an object, synthesize a hand grasp pose
(finger joints only) with BODex and check whether it is force-closure.

This reuses the full BODex grasp-synthesis pipeline (cuRobo QP force-closure
energy, GJK contact queries) but freezes the floating wrist so ONLY the finger
joints are optimized. The wrist stays exactly where you put it.

How the wrist is frozen:
  The decision vector is [tx,ty,tz, qw,qx,qy,qz, finger_joints...]. The Newton
  optimizer scales each step by `line_scale`; zeroing its first 7 entries makes
  the translation+rotation step size 0, so the wrist never moves.  (See
  newton_base._create_box_line_search / base_scale.)

Run in the `bodex` conda env, from the BODex dir:
    cd ~/AutoDex/src/grasp_generation/BODex
    CUDA_VISIBLE_DEVICES=0 /home/mingi/miniconda3/envs/bodex/bin/python \
        solve_fixed_wrist.py --obj apple --num_seeds 32
    # supply your own wrist (object frame 4x4 or 7d [x,y,z,qw,qx,qy,qz]):
    #   --wrist_npy /path/to/wrist_se3.npy
    # exact Ferrari-Canny Q1 instead of the QP residual:
    #   --metric ch_q1
"""
import argparse
import os
import tempfile

import numpy as np
import torch
import transforms3d as t3d

from curobo.types.base import TensorDeviceType
from curobo.geom.sdf.world import WorldConfig
from curobo.wrap.reacher.grasp_solver import GraspSolver, GraspSolverConfig
from curobo.util.world_cfg_generator import get_world_config_dataloader
from curobo.util.logger import setup_logger
from curobo.util_file import get_manip_configs_path, join_path, load_yaml


def cart2se3(cart):
    T = np.eye(4)
    T[:3, 3] = cart[:3]
    T[:3, :3] = t3d.quaternions.quat2mat(cart[3:7])
    return T


def se3_to_cart(T):
    """4x4 -> [x,y,z, qw,qx,qy,qz] (wxyz quaternion, BODex convention)."""
    q = t3d.quaternions.mat2quat(T[:3, :3])  # returns wxyz
    return np.concatenate([T[:3, 3], q])


def load_wrist_cart_object_frame(path, n_finger):
    """Load a wrist file, return 7d [x,y,z,qw,qx,qy,qz] in OBJECT frame."""
    arr = np.load(path)
    if arr.shape == (4, 4):
        return se3_to_cart(arr)
    arr = arr.reshape(-1)
    if arr.shape[0] == 7:
        return arr
    if arr.shape[0] == 7 + n_finger:  # full dof seed; take wrist part
        return arr[:7]
    raise ValueError(f"Unrecognized wrist file shape {arr.shape}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("-c", "--manip_cfg", default="sim_inspire/paradex_box.yml")
    ap.add_argument("--obj", required=True, help="object name (must exist in the hand's scene dir)")
    ap.add_argument("--scene_type", nargs="+", default=None, help="override scene_type list")
    ap.add_argument("--wrist_npy", default=None,
                    help="object-frame wrist (4x4 or 7d). If omitted, a wrist is sampled on the object surface.")
    ap.add_argument("--num_seeds", type=int, default=32)
    ap.add_argument("--finger_jitter", type=float, default=0.3,
                    help="uniform +/- jitter (rad) applied to finger seeds for diversity")
    ap.add_argument("--metric", default="qp", choices=["qp", "ch_q1", "dfc"],
                    help="force-closure metric. qp=QP residual (default), ch_q1=Ferrari-Canny Q1")
    ap.add_argument("--fc_thresh", type=float, default=0.1,
                    help="grasp_error below this = force closure (QP metric). BODex uses 0.1 strong / 0.2 ok.")
    ap.add_argument("--out", default=None, help="dir to save best grasp (default: ./fixed_wrist_out/{obj})")
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    setup_logger("warn")
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    tensor_args = TensorDeviceType()

    manip_config_data = load_yaml(join_path(get_manip_configs_path(), args.manip_cfg))
    manip_config_data["world"]["obj_list"] = [args.obj]
    if args.scene_type is not None:
        manip_config_data["world"]["scene_type"] = args.scene_type
    robot_name = manip_config_data["robot_file"].replace(".yml", "")

    # --- build ONE world (object + scene) --------------------------------
    tmp_out = tempfile.mkdtemp(prefix="fixed_wrist_")
    world_generator = get_world_config_dataloader(
        manip_config_data["world"], 1, manip_config_data["seed_num"],
        manip_config_data["exp_name"], seed_offset=0, output_dir=tmp_out,
        obj_root_dir=None, scene_filter=None, hand=robot_name,
    )
    world_info = None
    for w in world_generator:
        world_info = w
        break
    if world_info is None:
        raise RuntimeError(f"No scene found for obj={args.obj} hand={robot_name}. "
                           f"Check ~/shared_data/AutoDex/scene/{robot_name}/{args.obj}/")
    obj_names = world_info["manip_name"]
    obj_name = obj_names[0]
    print(f"[world] obj={obj_name} scene ready; n_env={len(obj_names)}")

    # --- build solver -----------------------------------------------------
    grasp_config = GraspSolverConfig.load_from_robot_config(
        world_model=world_info["world_cfg"],
        manip_name_list=obj_names,
        manip_config_data=manip_config_data,
        obj_gravity_center=world_info["obj_gravity_center"],
        obj_obb_length=world_info["obj_obb_length"],
        metric_type=args.metric,
        use_cuda_graph=False,
        store_debug=False,
    )
    grasp_solver = GraspSolver(grasp_config)
    dof = grasp_solver.dof
    n_finger = dof - 7
    print(f"[solver] dof={dof} (7 wrist + {n_finger} finger joints)  metric={args.metric}")

    # --- FREEZE the wrist: zero translation+rotation step scale -----------
    ls = grasp_solver.solver.newton_optimizer.line_scale
    ls[..., :7] = 0.0
    print(f"[freeze] line_scale[:7] zeroed -> wrist held fixed, only {n_finger} finger joints optimized")

    # --- determine the fixed wrist (object frame -> scene frame) ----------
    obj_pose_cart = world_info["world_cfg"][0]["mesh"][obj_name]["pose"]  # 7d in scene frame
    obj_se3 = cart2se3(obj_pose_cart)
    if args.wrist_npy is not None:
        wrist_cart_obj = load_wrist_cart_object_frame(args.wrist_npy, n_finger)
        wrist_se3_scene = obj_se3 @ cart2se3(wrist_cart_obj)
        wrist_cart_scene = se3_to_cart(wrist_se3_scene)
        print(f"[wrist] loaded from {args.wrist_npy} (object frame) -> scene frame")
    else:
        # sample a plausible on-surface wrist from the built-in seed generator
        seed0 = grasp_solver.generate_seed(num_seeds=1, batch=1, use_nn_seed=False)
        wrist_cart_scene = seed0.view(-1, dof)[0, :7].detach().cpu().numpy()
        print("[wrist] none supplied -> sampled one on the object surface")

    # --- build seed_config: fixed wrist + jittered finger inits -----------
    base_q = np.asarray(manip_config_data["seeder_cfg"]["q"], dtype=np.float32)  # (n_finger,)
    assert base_q.shape[0] == n_finger, f"seeder q dof {base_q.shape[0]} != {n_finger}"
    seeds = np.zeros((1, args.num_seeds, dof), dtype=np.float32)
    seeds[0, :, :7] = wrist_cart_scene
    fj = args.finger_jitter
    jitter = np.random.uniform(-fj, fj, size=(args.num_seeds, n_finger)).astype(np.float32)
    jitter[0] = 0.0  # keep one clean seed
    seeds[0, :, 7:] = base_q[None] + jitter
    seed_config = tensor_args.to_device(seeds)

    # --- solve ------------------------------------------------------------
    result = grasp_solver.solve_batch_env(
        seed_config=seed_config, num_seeds=args.num_seeds, return_seeds=args.num_seeds,
    )

    solution = result.solution.detach().cpu().numpy()       # [1, S, stages, dof]
    grasp_error = result.grasp_error.detach().cpu().numpy() # [1, S, m]
    ge_per_seed = grasp_error[0].max(axis=-1)               # [S]  (worst target wrench)

    # verify wrist really stayed put
    wrist_moved = np.abs(solution[0, :, 1, :7] - wrist_cart_scene[None]).max()

    best = int(np.argmin(ge_per_seed))
    best_ge = float(ge_per_seed[best])
    grasp_pose = solution[0, best, 1, 7:]      # finger joints at grasp (squeezed)
    pregrasp_pose = solution[0, best, 0, 7:]   # finger joints at pregrasp (open)

    if args.metric == "ch_q1":
        q1 = 2.0 - best_ge
        fc = q1 > 0.0
        verdict = f"Q1={q1:+.4f}  -> force closure {'SATISFIED' if fc else 'NOT satisfied'}"
    else:
        fc = best_ge < args.fc_thresh
        verdict = (f"grasp_error={best_ge:.4f} (thresh {args.fc_thresh}) -> "
                   f"force closure {'SATISFIED' if fc else 'NOT satisfied'}"
                   f"  [strong<0.1, ok<0.2]")

    n_fc = int((ge_per_seed < args.fc_thresh).sum())
    print("\n================ RESULT ================")
    print(f"object            : {obj_name}")
    print(f"wrist drift (max)  : {wrist_moved:.2e}  (should be ~0 -> wrist stayed fixed)")
    print(f"seeds solved       : {args.num_seeds}  |  force-closure seeds (<{args.fc_thresh}): {n_fc}")
    print(f"best seed          : #{best}")
    print(f"grasp_pose (fingers): {np.array2string(grasp_pose, precision=3)}")
    print(f"VERDICT            : {verdict}")
    print("========================================")

    # --- save best -------------------------------------------------------
    out_dir = args.out or os.path.join(os.path.dirname(__file__), "fixed_wrist_out", obj_name)
    os.makedirs(out_dir, exist_ok=True)
    wrist_se3_obj = np.linalg.inv(obj_se3) @ cart2se3(solution[0, best, 1, :7])
    np.save(os.path.join(out_dir, "wrist_se3.npy"), wrist_se3_obj)
    np.save(os.path.join(out_dir, "grasp_pose.npy"), grasp_pose)
    np.save(os.path.join(out_dir, "pregrasp_pose.npy"), pregrasp_pose)
    np.save(os.path.join(out_dir, "bodex_info.npy"), {
        "contact_point": result.contact_point.detach().cpu().numpy()[0, best],
        "contact_frame": result.contact_frame.detach().cpu().numpy()[0, best],
        "contact_force": result.contact_force.detach().cpu().numpy()[0, best],
        "grasp_error": best_ge,
        "force_closure": bool(fc),
        "metric": args.metric,
    })
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
