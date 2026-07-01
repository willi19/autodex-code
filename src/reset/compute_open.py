"""Stage 1 only: open fingers via gradient descent.

For each (tabletop_pose, grasp), starting from pregrasp finger qpos at the
native wrist pose, run gradient descent on the finger qpos (wrist held fixed,
clipped to lower joint limit). Loss = Σ ESDF over hand spheres (signed
distance, negative outside; want more negative = farther from obstacles).

Save trajectory (qpos per iter), per-sphere positions, signed distances,
bbox membership over iterations.

Output npz schema is compatible with view.py (K=1 direction axis, T=n_iter+1
iteration axis), so the viewer's sweep_t slider scrubs through optimization
iterations.

Usage:
    python src/reset/compute_open.py --hand inspire_left --version v3
"""

import os
import sys
import argparse
import numpy as np
import torch
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "paradex"))

from autodex.planner.planner import GraspPlanner
from autodex.utils.path import obj_path as DEFAULT_OBJ_ROOT, load_candidate
from autodex.utils.conversion import se32action, se32cart

from common import (
    BBOX_EXPAND,
    get_all_objects, save_npz,
    list_tabletop_poses, load_tabletop_pose,
    load_obb, bbox_world_frame, points_inside_bbox, build_world_cfg,
)

from curobo.geom.types import WorldConfig
from curobo.wrap.model.robot_world import RobotWorld, RobotWorldConfig
from curobo.geom.sdf.world import CollisionQueryBuffer


N_ITER = 600
LR = 0.04
K_SWEEP = 30

# Sphere index ranges per link, in inspire_left_floating collision_link_names order.
# base_link [0,15), thumb_1 [15,16), thumb_2 [16,18), thumb_3 [18,20), thumb_4 [20,22),
# index_1 [22,25), index_2 [25,28), middle_1 [28,31), middle_2 [31,34),
# ring_1 [34,37), ring_2 [37,40), little_1 [40,43), little_2 [43,46).
FINGER_SPHERE_RANGES = {
    "thumb":  (15, 22),
    "index":  (22, 28),
    "middle": (28, 34),
    "ring":   (34, 40),
    "little": (40, 46),
}
# Joint dim within the 6-finger qpos.
THUMB_DIMS = (0, 1)
NON_THUMB_FINGERS = (
    ("index",  2),
    ("middle", 3),
    ("ring",   4),
    ("little", 5),
)


def _make_kwargs(rw, n_spheres, device, compute_esdf=True, return_loss=False):
    import inspect
    sig = inspect.signature(rw.world_model.get_sphere_distance)
    kwargs = {"sum_collisions": False, "compute_esdf": compute_esdf}
    if "return_loss" in sig.parameters:
        kwargs["return_loss"] = return_loss
    if "contact_distance" in sig.parameters:
        kwargs["contact_distance"] = torch.zeros(n_spheres, dtype=torch.float32, device=device)
    return kwargs


def _eval_spheres(rw, q, planner, weight, act, sphere_kwargs_factory):
    """Forward kinematics + distance for q. Returns (spheres, d). d shape (B, 1, N_s)."""
    state = rw.get_kinematics(q)
    spheres = state.link_spheres_tensor.unsqueeze(1)  # (B, 1, N_s, 4)
    buffer = CollisionQueryBuffer.initialize_from_shape(
        spheres.shape, planner._tensor_args, rw.world_model.collision_types
    )
    n_spheres = spheres.shape[2]
    kwargs = sphere_kwargs_factory(n_spheres)
    d = rw.world_model.get_sphere_distance(spheres, buffer, weight, act, **kwargs)
    return spheres, d


def gradient_open_per_scene(planner, world_cfg, wrist_se3, pregrasp, n_iter, lr,
                             finger_lower=0.0):
    """Per-scene gradient descent on ALL finger qpos to maximize obstacle clearance.

    Loss = -Σ ESDF (cuRobo returns ESDF with negative=outside; its backward
    returns the collision-cost gradient which is the NEGATION of the true ESDF
    gradient, so we flip the sign on the forward loss to recover proper
    gradient descent on ESDF = push spheres further outside obstacles).

    Variable: finger qpos starting at pregrasp. Wrist held at grasp pose.
    Clipped to lower joint limit (default 0.0 for inspire family).

    Returns:
        qpos_traj: (M, n_iter+1, J)
        centers:   (M, n_iter+1, N_valid, 3)
        esdf:      (M, n_iter+1, N_valid)
        radii:     (N_valid,)
    """
    M = wrist_se3.shape[0]
    J = pregrasp.shape[1]
    device = planner._tensor_args.device

    rw_config = RobotWorldConfig.load_from_config(
        planner._hand_cfg,
        WorldConfig.from_dict(world_cfg),
        collision_activation_distance=0.0,
        tensor_args=planner._tensor_args,
    )
    rw = RobotWorld(rw_config)
    n_dof = rw.kinematics.get_dof()

    if n_dof == J + 6:
        q_init_np = np.array([se32action(w, g) for w, g in zip(wrist_se3, pregrasp)])
    else:
        q_init_np = np.array([np.concatenate([se32cart(w), g]) for w, g in zip(wrist_se3, pregrasp)])
        if q_init_np.shape[1] != n_dof:
            q_init_np = pregrasp

    q_init = torch.tensor(q_init_np, dtype=torch.float32, device=device)
    wrist_part = q_init[:, :-J].contiguous()
    finger = q_init[:, -J:].clone().detach().requires_grad_(True)

    weight = torch.tensor([1.0], dtype=torch.float32, device=device)
    act = torch.tensor([0.0], dtype=torch.float32, device=device)
    sphere_kwargs_grad = lambda n_s: _make_kwargs(rw, n_s, device, return_loss=True)

    qpos_traj = np.zeros((M, n_iter + 1, J))
    qpos_traj[:, 0] = pregrasp
    centers_traj = None
    esdf_traj = None
    radii = None

    for t in range(n_iter + 1):
        q = torch.cat([wrist_part, finger], dim=1)
        spheres, d = _eval_spheres(rw, q, planner, weight, act, sphere_kwargs_grad)

        if radii is None:
            radii_full = spheres[0, 0, :, 3].detach().cpu().numpy()
            valid = radii_full > 1e-6
            radii = radii_full[valid]
            centers_traj = np.zeros((M, n_iter + 1, int(valid.sum()), 3))
            esdf_traj = np.zeros((M, n_iter + 1, int(valid.sum())))
        else:
            valid = (spheres[0, 0, :, 3].detach().cpu().numpy() > 1e-6)

        centers_np = spheres[:, 0, :, :3].detach().cpu().numpy()
        centers_traj[:, t] = centers_np[:, valid, :]
        esdf_traj[:, t] = d.view(M, -1).detach().cpu().numpy()[:, valid]

        if t == n_iter:
            break

        # ESDF: d is negative outside obstacles. Per-finger max-loss: take the
        # worst (largest d = closest to obstacle) sphere within each finger group
        # and minimize that. Sum across fingers so each joint receives gradient
        # only from its own worst sphere — fingertip can't be sacrificed for a
        # proximal gain.
        d_bn = d[:, 0, :]                                 # (M, N_s)
        loss = 0
        for _name, (s_lo, s_hi) in FINGER_SPHERE_RANGES.items():
            loss = loss + d_bn[:, s_lo:s_hi].max(dim=-1).values.sum()
        loss.backward()
        with torch.no_grad():
            finger -= lr * finger.grad
            finger.clamp_(min=finger_lower)
            finger.grad.zero_()

        qpos_traj[:, t + 1] = finger.detach().cpu().numpy()

    return qpos_traj, centers_traj, esdf_traj, radii


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", default=None)
    parser.add_argument("--version", default="v3")
    parser.add_argument("--hand", default="inspire_left")
    parser.add_argument("--x_offset", type=float, default=0.0)
    parser.add_argument("--z_rotation", type=float, default=0.0)
    parser.add_argument("--obj_root", default=DEFAULT_OBJ_ROOT)
    parser.add_argument("--candidates_root",
                        default=os.path.join(os.path.expanduser("~"), "AutoDex", "candidates"))
    args = parser.parse_args()

    import autodex.utils.path as _autopath
    _autopath.get_candidate_path = lambda hand: os.path.join(args.candidates_root, hand)

    z_rad = np.radians(args.z_rotation)

    objs = [args.obj] if args.obj else get_all_objects(args.hand, args.version, args.obj_root)
    if not objs:
        print("No objects to process.")
        return

    print(f"[open-gradient] hand={args.hand} ver={args.version} N_obj={len(objs)} "
          f"n_iter={N_ITER} lr={LR}")
    print("=" * 70)

    planner = GraspPlanner(hand=args.hand)

    for oi, obj_name in enumerate(objs):
        print(f"\n[{oi+1}/{len(objs)}] {obj_name}")
        try:
            wrist_obj, pregrasp, _, scene_info = load_candidate(
                obj_name, np.eye(4), args.version,
                shuffle=False, skip_done=False, hand=args.hand,
            )
            if len(wrist_obj) == 0:
                print("  [skip] no candidates")
                continue
            N = len(wrist_obj)
            T = N_ITER + 1

            pose_ids = list_tabletop_poses(obj_name, args.obj_root)
            if not pose_ids:
                print("  [skip] no tabletop poses")
                continue
            P = len(pose_ids)

            obb_tf, half_ext = load_obb(obj_name, args.obj_root)
            half_ext_expanded = half_ext + BBOX_EXPAND

            pose_se3 = np.zeros((P, 4, 4))
            for pi, p_id in enumerate(pose_ids):
                pose_se3[pi] = load_tabletop_pose(
                    obj_name, p_id, args.obj_root, args.x_offset, z_rad,
                )

            # ── Pass 1: filter (pose, grasp) pairs where the pregrasp itself
            # already collides with table/object. Those are dropped.
            print(f"  filtering initial collisions over {P}×{N} pairs...")
            valid_mask = np.zeros((P, N), dtype=bool)
            for pi in range(P):
                pose = pose_se3[pi]
                world_cfg = build_world_cfg(obj_name, pose, args.obj_root)
                wrist_world_p = np.einsum('ij,njk->nik', pose, wrist_obj)
                coll = planner._check_collision(world_cfg, wrist_world_p, pregrasp)
                valid_mask[pi] = ~coll
            M = int(valid_mask.sum())
            print(f"  valid pregrasp pairs: {M}/{P*N}")
            if M == 0:
                print("  [skip] no valid (pose, grasp) pairs")
                continue

            # ── Pass 2: gradient open on the valid pairs, batched per pose.
            centers_list = []
            collide_list = []
            in_bbox_list = []
            min_clearance_list = []
            wrist_list = []
            qpos_traj_list = []
            pair_pose_idx_list = []
            pair_grasp_idx_list = []
            radii = None

            for pi in range(P):
                valid_g = np.where(valid_mask[pi])[0]
                if len(valid_g) == 0:
                    continue
                pose = pose_se3[pi]
                world_cfg = build_world_cfg(obj_name, pose, args.obj_root)
                wrist_world_sub = np.einsum('ij,njk->nik', pose, wrist_obj[valid_g])

                print(f"  pose {pi+1}/{P}  {pose_ids[pi]}  ({len(valid_g)} valid grasps)")
                qpos_sub, centers_sub, esdf_sub, radii_sub = gradient_open_per_scene(
                    planner, world_cfg, wrist_world_sub, pregrasp[valid_g],
                    n_iter=N_ITER, lr=LR,
                )
                if radii is None:
                    radii = radii_sub
                clearance_sub = (-esdf_sub) - radii[None, None, :]      # (M', T, N_s)
                collide_sub = clearance_sub < 0
                min_clearance_sub = clearance_sub.min(axis=2)           # (M', T)
                bbox_world = bbox_world_frame(pose, obb_tf)
                in_bbox_sub = np.zeros_like(collide_sub)
                for m in range(len(valid_g)):
                    in_bbox_sub[m] = points_inside_bbox(
                        centers_sub[m], bbox_world, half_ext_expanded,
                    )

                centers_list.append(centers_sub)
                collide_list.append(collide_sub)
                in_bbox_list.append(in_bbox_sub)
                min_clearance_list.append(min_clearance_sub)
                wrist_list.append(wrist_world_sub)
                qpos_traj_list.append(qpos_sub)
                pair_pose_idx_list.extend([pi] * len(valid_g))
                pair_grasp_idx_list.extend(valid_g.tolist())

            centers_all = np.concatenate(centers_list, axis=0)
            collide_all = np.concatenate(collide_list, axis=0)
            in_bbox_all = np.concatenate(in_bbox_list, axis=0)
            min_clearance_all = np.concatenate(min_clearance_list, axis=0)
            wrist_world_all = np.concatenate(wrist_list, axis=0)
            qpos_traj_all = np.concatenate(qpos_traj_list, axis=0)
            pair_pose_idx = np.array(pair_pose_idx_list, dtype=int)
            pair_grasp_idx = np.array(pair_grasp_idx_list, dtype=int)

            valid_final = min_clearance_all[:, -1] > 0
            init_avg = min_clearance_all[:, 0].mean() * 1000
            final_avg = min_clearance_all[:, -1].mean() * 1000
            print(f"  → clearance avg: init {init_avg:.1f}mm → final {final_avg:.1f}mm  "
                  f"(final-safe {int(valid_final.sum())}/{M})")

            print(f"  → final-safe {int(valid_final.sum())}/{M}")

            # Sanitize version for filename (allow nested versions like "reset/0").
            ver_safe = args.version.replace("/", "_")
            save_dir = os.path.join("outputs", "reset", obj_name)
            os.makedirs(save_dir, exist_ok=True)
            out_path = os.path.join(save_dir, f"open_{ver_safe}.npz")
            meta = {
                "obj": obj_name, "method": "open",
                "hand": args.hand, "version": args.version,
                "x_offset": args.x_offset, "z_rotation": args.z_rotation,
                "obj_root": args.obj_root,
                "n_iter": N_ITER, "lr": LR,
            }
            np.savez_compressed(
                out_path,
                centers=centers_all.astype(np.float32),                  # (M, T, N_s, 3)
                radii=radii.astype(np.float32),
                collide=collide_all,                                     # (M, T, N_s)
                in_bbox=in_bbox_all,                                     # (M, T, N_s)
                pose_se3=pose_se3.astype(np.float32),                    # (P, 4, 4)
                pose_ids=np.array(pose_ids),                             # (P,)
                wrist_world=wrist_world_all.astype(np.float32),          # (M, 4, 4)
                qpos_traj=qpos_traj_all.astype(np.float32),              # (M, T, J)
                t_values=np.linspace(0.0, 1.0, T).astype(np.float32),
                half_ext_expanded=half_ext_expanded.astype(np.float32),
                pair_pose_idx=pair_pose_idx,                             # (M,)
                pair_grasp_idx=pair_grasp_idx,                           # (M,)
                valid_final=valid_final,                                 # (M,)
                min_clearance=min_clearance_all.astype(np.float32),      # (M, T)
                pregrasp=pregrasp.astype(np.float32),
                scene_info=np.array(scene_info, dtype=object),
                meta=np.array([meta], dtype=object),
            )
            print(f"  saved: {out_path}")

            # Save per-pair openpose into the candidate directory:
            # {candidates_root}/{hand}/{version}/{obj}/{scene_type}/{scene_id}/{grasp_idx}/openpose_{pose_id}.npy
            cand_obj_root = os.path.join(args.candidates_root, args.hand,
                                          args.version, obj_name)
            n_saved = 0
            n_unsafe = 0
            for m in range(qpos_traj_all.shape[0]):
                pi = int(pair_pose_idx[m])
                gi = int(pair_grasp_idx[m])
                scene_type, scene_id, grasp_idx = scene_info[gi]
                pose_id = pose_ids[pi]
                final_qpos = qpos_traj_all[m, -1, :].astype(np.float32)  # (J,)
                grasp_dir = os.path.join(cand_obj_root, scene_type,
                                          str(scene_id), str(grasp_idx))
                if not os.path.isdir(grasp_dir):
                    continue
                out_npy = os.path.join(grasp_dir, f"openpose_{pose_id}.npy")
                np.save(out_npy, final_qpos)
                n_saved += 1
                if not bool(valid_final[m]):
                    n_unsafe += 1
            print(f"  openpose saved: {n_saved} files "
                  f"(of which {n_unsafe} have residual collision) → {cand_obj_root}")
        except Exception as ex:
            print(f"  [error] {ex}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
