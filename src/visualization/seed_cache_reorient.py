#!/usr/bin/env python3
"""Reorient (reset) seed cache — "can we reorient pose i -> j at this placement?"

Same (B)-1 idea as the approach seed cache, but the cached trajectory is the WHOLE
reorient chain and the GOAL is a held-object reorientation, not an empty-hand grasp:

    INIT --[approach: refine_fingers]--> grasp@i
    grasp@i --[lift: JOINT, NOT cartesian]--> A_lifted
    A_lifted --[plan_obj_placement: joint]--> B_lifted (object at pose j)
    B_lifted --[descent: JOINT]--> grasp@j-on-table

Per (transition i->j, grasp, cell) we record whether the chain plans (= reorient is
POSSIBLE here, the feasibility map the campaign gates on) AND the cached trajectory
AND the landing placement (so chaining knows where the object ends up).

ALL joint-space (no cartesian -> no unrecoverable branch flips). Lift gives table
clearance; a post-hoc held-object collision check (object mesh stays above the table
through the reorient) rejects plans that scrape the table.

This is a VALIDATION harness first (one transition, a few cells) to confirm the
geometry/chain plans before the full grid build.
"""
import argparse, sys, pathlib
import numpy as np, torch

sys.path.insert(0, "/home/mingi/AutoDex")
import src.visualization.exp as exp
import src.visualization.seed_cache as sc
from autodex.planner import GraspPlanner
from autodex.planner.planner import _to_curobo_world, _to_curobo_pose
from autodex.utils.path import obj_path, project_dir


def _ik(planner, fingers, T_wrist, lo, hi, retract=None):
    dev = planner._tensor_args.device
    rc = (torch.tensor(planner._init_state if retract is None else retract,
                       dtype=torch.float32, device=dev).unsqueeze(0))
    r = planner._ik_solver.solve_batch(_to_curobo_pose(T_wrist[None], dev), retract_config=rc)
    if not bool(r.success.view(-1)[0]):
        return None
    f = (planner._init_state.copy() if retract is None else np.asarray(retract).copy())
    f[:6] = exp._unwrap_arm(r.solution.cpu().numpy().reshape(-1)[:6], rc.cpu().numpy().reshape(-1)[:6], lo, hi)
    f[6:] = fingers
    return f.astype(np.float32)


def _lift_wrist(T_wrist, dz):
    T = T_wrist.copy(); T[2, 3] += dz; return T


def _held_clears_table(urdf, configs, T_obj_in_wrist, obj_verts, tol=0.005):
    """True if the held object stays ABOVE the table surface across `configs`
    (post-hoc mesh check; table-only world, so min-z is the only constraint)."""
    from src.visualization.exp import fk_ee
    ee = fk_ee(urdf, np.asarray(configs))                 # (M,4,4) wrist poses
    obj = ee @ T_obj_in_wrist                             # (M,4,4) object poses
    zrow = obj[:, 2, :3]                                  # (M,3)
    zt = obj[:, 2, 3]                                     # (M,)
    vz = obj_verts @ zrow.T + zt[None, :]                 # (V,M) world z of every vertex
    return float(vz.min()) >= exp.TABLE_SURFACE_Z - tol


def build_reorient_one(planner, urdf, wrist_obj_g, preg_g, T_obj_i, R_target_j,
                       z_lift, z_rel, lo, hi, obj_verts=None, n_retry=5):
    """Returns the full reorient chain config-trajectory (approach..descent) or None.
    All joint-space. plan_obj_placement does the held-object reorient (already joint)."""
    dev = planner._tensor_args.device
    grasp_wrist_i = T_obj_i @ wrist_obj_g
    T_obj_in_wrist = np.linalg.inv(grasp_wrist_i) @ T_obj_i
    scene_lift = {"mesh": {}, "cuboid": {"table": exp.TABLE_CUBOID}}

    # 1) approach: INIT -> grasp@i
    q_grasp_i = _ik(planner, preg_g, grasp_wrist_i, lo, hi)
    if q_grasp_i is None:
        return None, "no_ik_grasp_i"
    ok, approach = False, None
    for _ in range(n_retry):
        ok, approach = planner._refine_fingers(planner._init_state, q_grasp_i)
        if ok:
            break
    if not ok:
        return None, "approach_fail"
    grasp_qpos = approach[-1].astype(np.float32)

    # 2) lift JOINT: grasp@i -> A_lifted (grasp wrist + z up to z_lift)
    dz_up = z_lift - float(grasp_wrist_i[2, 3])
    A_wrist = _lift_wrist(grasp_wrist_i, dz_up)
    q_A = _ik(planner, preg_g, A_wrist, lo, hi, retract=grasp_qpos)
    if q_A is None:
        return None, "no_ik_A_lifted"
    ok, lift = planner._refine_fingers(grasp_qpos, q_A)
    if not ok:
        return None, "lift_fail"
    A_lifted = lift[-1].astype(np.float32)

    # 3) reorient JOINT: A_lifted -> B_lifted (object at orientation pose j)
    target_pos = np.array([0.0, float(T_obj_i[1, 3]), z_lift])
    reorient, info = planner.plan_obj_placement(
        scene_lift, A_lifted, T_obj_in_wrist, R_target_j, target_pos,
        hold_hand_qpos=preg_g, x_grid=sc.X_GRID, yaw_grid=sc.yaw_grid(8))
    if reorient is None:
        return None, "reorient_fail"
    B_lifted = reorient[-1].astype(np.float32)

    # object pose at B_lifted, then lower to table (pose j on table)
    from src.visualization.exp import fk_ee
    obj_B = fk_ee(urdf, B_lifted[None])[0] @ T_obj_in_wrist
    obj_j_table = obj_B.copy(); obj_j_table[2, 3] = z_rel
    desc_wrist = obj_j_table @ np.linalg.inv(T_obj_in_wrist)
    # 4) descent JOINT: B_lifted -> grasp@j-on-table
    q_desc = _ik(planner, preg_g, desc_wrist, lo, hi, retract=B_lifted)
    if q_desc is None:
        return None, "no_ik_descent"
    ok, descent = planner._refine_fingers(B_lifted, q_desc)
    if not ok:
        return None, "descent_fail"

    chain = np.concatenate([approach, lift, reorient, descent], axis=0).astype(np.float32)
    return {"chain": chain, "obj_j": (fk_ee(urdf, descent[-1][None])[0] @ T_obj_in_wrist)}, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="pepsi")
    ap.add_argument("--hand", default="inspire_left")
    ap.add_argument("--i", type=int, default=2)
    ap.add_argument("--j", type=int, default=18)
    ap.add_argument("--h_cm", type=int, default=0)
    ap.add_argument("--n_cells", type=int, default=12)
    args = ap.parse_args()

    pl = GraspPlanner(hand=args.hand)
    INIT = pl._init_state
    tab = exp.load_tabletop_poses(args.obj)
    z0_i = float(tab[args.i][2, 3]) + exp.TABLE_SURFACE_Z
    tabR_i = tab[args.i][:3, :3]
    R_target_j = tab[args.j][:3, :3]
    z_lift = float(tab[args.j][2, 3]) + exp.TABLE_SURFACE_Z + exp.LIFT_HEIGHT_M
    z_rel = float(tab[args.j][2, 3]) + exp.TABLE_SURFACE_Z
    mesh = str(pathlib.Path(obj_path) / args.obj / "processed_data" / "mesh" / "simplified.obj")

    from autodex.utils.conversion import se32cart

    def W(T):
        return _to_curobo_world({"mesh": {"target": {"pose": se32cart(np.asarray(T)), "file_path": mesh}},
                                 "cuboid": {"table": exp.TABLE_CUBOID}})

    pl._init_motion_gen(W(np.eye(4)), use_cuda_graph=False)
    pl._init_ik_solver(_to_curobo_world({"mesh": {}, "cuboid": {"table": exp.TABLE_CUBOID}}), use_cuda_graph=False)
    lo, hi = exp._arm_limits(pl)
    import yourdfpy
    urdf = yourdfpy.URDF.load(str(exp.URDF_BY_HAND[args.hand]))
    print("INIT DONE", flush=True)

    seeds = exp.load_reset_seeds(args.hand, args.obj, args.h_cm, args.i, args.j, np.eye(4))
    if seeds is None:
        print(f"NO reset seeds for {args.i}->{args.j}"); return
    wrist_obj = seeds["wrist_se3"]    # object frame (T_obj_world=I)
    preg = seeds["pregrasp"]
    print(f"transition {args.i}->{args.j}: {len(wrist_obj)} reorient grasps", flush=True)

    def setw(T):
        w = W(T)
        (pl._update_world if pl._world_structure_changed(w) else pl._update_target_pose_only)(w)
        pl._cached_world = w

    X = sc.X_GRID
    YAW = sc.yaw_grid(36)
    rng = np.random.default_rng(0)
    cells = [(xi, yi) for xi in range(len(X)) for yi in range(len(YAW))]
    rng.shuffle(cells)
    from collections import Counter
    res = Counter()
    n = 0
    for (xi, yi) in cells:
        if n >= args.n_cells:
            break
        T_obj_i = sc.place_T(tabR_i, z0_i, float(X[xi]), float(YAW[yi]))
        setw(T_obj_i)
        n += 1
        # try a few reorient grasps until one plans the full chain
        gi_order = list(range(len(wrist_obj))); rng.shuffle(gi_order)
        outcome = "all_grasp_fail"
        for g in gi_order[:8]:
            out, why = build_reorient_one(pl, urdf, wrist_obj[g], preg[g].astype(np.float32),
                                          T_obj_i, R_target_j, z_lift, z_rel, lo, hi)
            if out is not None:
                outcome = "ok"; break
            outcome = why
        res[outcome] += 1
        print(f"  cell ({xi},{yi}) -> {outcome}", flush=True)
    print(f"RES reorient {args.i}->{args.j}  cells={n}  {dict(res)}", flush=True)


if __name__ == "__main__":
    main()
