"""Alpha-blend open: pregrasp → zeros (fully open) at fixed wrist pose.

For each (tabletop_pose, grasp) pair: search alpha in [0,1] (pregrasp → 0).
Pick the largest alpha with contiguous collision-free range from 0. Saves
final qpos as openpose_{pose_id}.npy (same convention as compute_open.py).

Usage:
    python src/reset/compute_open_alpha.py --hand inspire_left --version v7 --obj pepsi \
        --candidates_root /home/mingi/shared_data/AutoDex/candidates
"""
import os
import sys
import argparse
import numpy as np

from autodex.planner.planner import GraspPlanner
from autodex.utils.path import obj_path as DEFAULT_OBJ_ROOT, load_candidate

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    BBOX_EXPAND,
    get_all_objects,
    list_tabletop_poses, load_tabletop_pose,
    load_obb, bbox_world_frame, points_inside_bbox, build_world_cfg,
)


N_STEPS = 21  # alpha grid: [0, 0.05, 0.10, ..., 1.0]


def alpha_blend_open_per_scene(planner, world_cfg, wrist_world, pregrasp,
                                n_steps=N_STEPS):
    """Per-pair: pregrasp * (1-α) + 0 * α at fixed wrist. Pick max α* in the
    contiguous-safe prefix from α=0. Returns:
        qpos_traj: (M, n_steps, J)  ← α index axis
        centers:   (M, n_steps, N_s, 3)
        esdf:      (M, n_steps, N_s)
        radii:     (N_s,)
    """
    M, J = pregrasp.shape
    alphas = np.linspace(0.0, 1.0, n_steps).astype(np.float32)

    # Build (M, n_steps, J) grid
    qpos_grid = pregrasp[:, None, :] * (1.0 - alphas[None, :, None])

    # Batched per-sphere ESDF eval
    wrist_b = np.broadcast_to(wrist_world[:, None, :, :], (M, n_steps, 4, 4)).reshape(-1, 4, 4)
    qpos_b = qpos_grid.reshape(-1, J)
    centers, radii, esdf = planner.check_collision_per_sphere(
        world_cfg, wrist_b, qpos_b, compute_esdf=True,
    )
    N_s = radii.shape[0]
    centers_t = centers.reshape(M, n_steps, N_s, 3)
    esdf_t = esdf.reshape(M, n_steps, N_s)
    return qpos_grid, centers_t, esdf_t, radii, alphas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default=None)
    ap.add_argument("--version", default="v7")
    ap.add_argument("--hand", default="inspire_left")
    ap.add_argument("--x_offset", type=float, default=0.0)
    ap.add_argument("--z_rotation", type=float, default=0.0)
    ap.add_argument("--obj_root", default=DEFAULT_OBJ_ROOT)
    ap.add_argument("--candidates_root",
                    default=os.path.join(os.path.expanduser("~"), "AutoDex", "candidates"))
    args = ap.parse_args()

    import autodex.utils.path as _autopath
    _autopath.get_candidate_path = lambda hand: os.path.join(args.candidates_root, hand)

    z_rad = np.radians(args.z_rotation)

    objs = [args.obj] if args.obj else get_all_objects(args.hand, args.version, args.obj_root)
    if not objs:
        print("No objects to process."); return

    print(f"[open-alpha] hand={args.hand} ver={args.version} N_obj={len(objs)} "
          f"n_steps={N_STEPS}")
    print("=" * 70)

    planner = GraspPlanner(hand=args.hand)

    for oi, obj_name in enumerate(objs):
        print(f"\n[{oi+1}/{len(objs)}] {obj_name}")
        wrist_obj, pregrasp, _, scene_info = load_candidate(
            obj_name, np.eye(4), args.version,
            shuffle=False, skip_done=False, hand=args.hand,
        )
        if len(wrist_obj) == 0:
            print("  [skip] no candidates"); continue
        N = len(wrist_obj)

        pose_ids = list_tabletop_poses(obj_name, args.obj_root)
        if not pose_ids:
            print("  [skip] no tabletop poses"); continue
        P = len(pose_ids)

        obb_tf, half_ext = load_obb(obj_name, args.obj_root)
        half_ext_expanded = half_ext + BBOX_EXPAND

        pose_se3 = np.zeros((P, 4, 4))
        for pi, p_id in enumerate(pose_ids):
            pose_se3[pi] = load_tabletop_pose(
                obj_name, p_id, args.obj_root, args.x_offset, z_rad,
            )

        # Filter: drop pregrasp pairs with initial collision.
        valid_mask = np.zeros((P, N), dtype=bool)
        for pi in range(P):
            pose = pose_se3[pi]
            world_cfg = build_world_cfg(obj_name, pose, args.obj_root)
            wrist_world_p = np.einsum('ij,njk->nik', pose, wrist_obj)
            coll = planner._check_collision(world_cfg, wrist_world_p, pregrasp)
            valid_mask[pi] = ~coll
        M_total = int(valid_mask.sum())
        print(f"  valid pregrasp pairs: {M_total}/{P*N}")
        if M_total == 0:
            continue

        centers_list, in_bbox_list, min_clearance_list = [], [], []
        wrist_list, qpos_chosen_list, alpha_chosen_list = [], [], []
        pair_pose_idx_list, pair_grasp_idx_list = [], []
        radii = None
        all_alphas = None

        for pi in range(P):
            valid_g = np.where(valid_mask[pi])[0]
            if len(valid_g) == 0:
                continue
            pose = pose_se3[pi]
            world_cfg = build_world_cfg(obj_name, pose, args.obj_root)
            wrist_world_sub = np.einsum('ij,njk->nik', pose, wrist_obj[valid_g])

            qpos_grid, centers_sub, esdf_sub, radii_sub, alphas = alpha_blend_open_per_scene(
                planner, world_cfg, wrist_world_sub, pregrasp[valid_g],
            )
            if radii is None:
                radii = radii_sub
                all_alphas = alphas

            clearance = (-esdf_sub) - radii[None, None, :]  # (M', K, N_s)
            min_clearance = clearance.min(axis=2)            # (M', K)
            safe = min_clearance > 0                          # (M', K)

            # Largest α* in contiguous-safe prefix from α=0
            M_sub = len(valid_g)
            chosen_idx = np.zeros(M_sub, dtype=int)
            for m in range(M_sub):
                end = 0
                while end < N_STEPS and safe[m, end]:
                    end += 1
                if end == 0:
                    chosen_idx[m] = 0
                else:
                    # within safe prefix, pick LARGEST alpha (most open)
                    chosen_idx[m] = end - 1

            qpos_chosen = qpos_grid[np.arange(M_sub), chosen_idx]  # (M', J)
            alpha_chosen = alphas[chosen_idx]                      # (M',)

            bbox_world = bbox_world_frame(pose, obb_tf)
            in_bbox_sub = np.zeros_like(clearance, dtype=bool)
            for m in range(M_sub):
                in_bbox_sub[m] = points_inside_bbox(centers_sub[m], bbox_world, half_ext_expanded)

            centers_list.append(centers_sub)
            in_bbox_list.append(in_bbox_sub)
            min_clearance_list.append(min_clearance)
            wrist_list.append(wrist_world_sub)
            qpos_chosen_list.append(qpos_chosen)
            alpha_chosen_list.append(alpha_chosen)
            pair_pose_idx_list.extend([pi] * M_sub)
            pair_grasp_idx_list.extend(valid_g.tolist())

            print(f"  pose {pi+1}/{P}  {pose_ids[pi]}  M={M_sub}  "
                  f"α̅={alpha_chosen.mean():.2f}  min_clr̅={min_clearance[np.arange(M_sub), chosen_idx].mean()*1000:.1f}mm")

        centers_all = np.concatenate(centers_list, axis=0)
        in_bbox_all = np.concatenate(in_bbox_list, axis=0)
        min_clearance_all = np.concatenate(min_clearance_list, axis=0)
        wrist_world_all = np.concatenate(wrist_list, axis=0)
        qpos_chosen_all = np.concatenate(qpos_chosen_list, axis=0)
        alpha_chosen_all = np.concatenate(alpha_chosen_list, axis=0)
        pair_pose_idx = np.array(pair_pose_idx_list, dtype=int)
        pair_grasp_idx = np.array(pair_grasp_idx_list, dtype=int)

        init_min = min_clearance_all[:, 0].mean() * 1000
        chosen_min = min_clearance_all[np.arange(len(alpha_chosen_all)),
                                        (alpha_chosen_all * (N_STEPS - 1)).astype(int)].mean() * 1000
        print(f"  → clearance avg: init {init_min:.1f}mm → chosen {chosen_min:.1f}mm  "
              f"(α̅={alpha_chosen_all.mean():.2f})")

        ver_safe = args.version.replace("/", "_")
        save_dir = os.path.join("outputs", "reset", obj_name)
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, f"open_alpha_{ver_safe}.npz")
        np.savez_compressed(
            out_path,
            centers=centers_all.astype(np.float32),
            radii=radii.astype(np.float32),
            in_bbox=in_bbox_all,
            pose_se3=pose_se3.astype(np.float32),
            pose_ids=np.array(pose_ids),
            wrist_world=wrist_world_all.astype(np.float32),
            qpos_chosen=qpos_chosen_all.astype(np.float32),
            alpha_chosen=alpha_chosen_all.astype(np.float32),
            min_clearance=min_clearance_all.astype(np.float32),
            pair_pose_idx=pair_pose_idx,
            pair_grasp_idx=pair_grasp_idx,
            pregrasp=pregrasp.astype(np.float32),
            scene_info=np.array(scene_info, dtype=object),
            half_ext_expanded=half_ext_expanded.astype(np.float32),
            alphas=all_alphas,
        )
        print(f"  saved: {out_path}")

        # Save openpose npy per pair (alpha-blend final qpos)
        cand_obj_root = os.path.join(args.candidates_root, args.hand,
                                      args.version, obj_name)
        n_saved = 0
        for m in range(qpos_chosen_all.shape[0]):
            pi = int(pair_pose_idx[m]); gi = int(pair_grasp_idx[m])
            scene_type, scene_id, grasp_idx = scene_info[gi]
            pose_id = pose_ids[pi]
            grasp_dir = os.path.join(cand_obj_root, scene_type, str(scene_id), str(grasp_idx))
            if not os.path.isdir(grasp_dir):
                continue
            np.save(os.path.join(grasp_dir, f"openpose_{pose_id}.npy"),
                    qpos_chosen_all[m].astype(np.float32))
            n_saved += 1
        print(f"  openpose saved: {n_saved} files → {cand_obj_root}")


if __name__ == "__main__":
    main()
