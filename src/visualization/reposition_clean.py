#!/usr/bin/env python3
"""Reposition (move object A -> B on the table), CLEAN joint-space version.

The old plan_reposition dragged the held object with cartesian_object_path: a
per-waypoint cartesian IK that branch-flips, is slow (~50s) and fails ("could not
pad goal pose"). Measured: pick succeeds in 5-14s, then the cartesian drag eats
the rest and fails -> churns every candidate.

This does it the way reorient already works (validated 12/12): pick a transport
grasp, approach with the robust _clean_plan, then move the object with
plan_obj_placement -- JOINT space, no cartesian, no branch flips. Target is a
single (x,yaw) cell = the exact target pose, so it plans straight to it.

    INIT --[approach: _clean_plan]--> grasp@A
    grasp@A --[plan_obj_placement -> exact target]--> object@B (held)
    --[release + retract]

Validate: total time, max per-step joint jump (continuity), object stays above
the table through the move. Saves the trajectory to npz for the viewer.

    python src/visualization/reposition_clean.py --obj pepsi --pose 2 \
        --x0 0.40 --x1 0.50 --yaw1 10
"""
import argparse, sys, time, pathlib
import numpy as np

sys.path.insert(0, "/home/mingi/AutoDex")
import src.visualization.exp as exp
from autodex.planner import GraspPlanner
from autodex.planner.planner import _to_curobo_world
from autodex.utils.path import obj_path


def reposition(planner, urdf, obj, hand, current_T, target_T,
               transport_cands, wrist_obj, pregrasp, mesh_path, max_tries=8):
    """Pick one transport grasp and move the object current_T -> target_T in
    joint space (plan_obj_placement). Returns (phases, picked_grasp) or None.

    phases: list of (name, robot_qpos (T,n), obj_T (T,4,4)).
    """
    from autodex.utils.conversion import se32cart
    scene_lift = {"mesh": {}, "cuboid": {"table": exp.TABLE_CUBOID}}
    world0 = _to_curobo_world({"mesh": {"target": {"pose": se32cart(current_T),
                              "file_path": str(mesh_path)}},
                              "cuboid": {"table": exp.TABLE_CUBOID}})
    _ti = time.time()
    if planner._motion_gen is None:                # _clean_plan/build_one needs it
        planner._init_motion_gen(world0)           # cuda_graph default -> ~7s warmup
    print(f"  init motion_gen {time.time()-_ti:.1f}s", flush=True)
    _ti = time.time()
    if planner._ik_solver is None:
        planner._init_ik_solver(_to_curobo_world(scene_lift), use_cuda_graph=False)
    print(f"  init ik_solver {time.time()-_ti:.1f}s", flush=True)

    tx = float(target_T[0, 3])
    R_tgt = target_T[:3, :3]
    obj_tgt_pos = target_T[:3, 3].astype(np.float32)
    x_grid = np.array([tx], dtype=np.float32)
    yaw_grid = np.array([0.0], dtype=np.float32)

    for tg in transport_cands[:max_tries]:
        tw = wrist_obj[tg]
        pg = np.asarray(pregrasp[tg], np.float32)
        wrist_world = current_T @ tw
        T_obj_in_wrist = np.linalg.inv(wrist_world) @ current_T

        t = time.time()
        ok, approach = exp._clean_plan(planner, tw, pg, current_T, mesh_path)
        print(f"  g{tg} approach {time.time()-t:.1f}s ok={ok}", flush=True)
        if not ok:
            continue
        grasp_qpos = approach[-1].astype(np.float32)

        t = time.time()
        move, info = planner.plan_obj_placement(
            scene_lift, grasp_qpos, T_obj_in_wrist, R_tgt, obj_tgt_pos,
            hold_hand_qpos=pg, x_grid=x_grid, yaw_grid=yaw_grid)
        print(f"  g{tg} move {time.time()-t:.1f}s ok={move is not None}"
              f" reason={info.get('reason','')}", flush=True)
        if move is None:
            continue

        # build phases (object follows the wrist while held)
        obj_app = np.tile(current_T[None], (len(approach), 1, 1))
        obj_move = exp.fk_ee(urdf, move) @ T_obj_in_wrist
        release, retract_t = exp.build_release_retract(
            planner, move[-1].copy(), pg, pg)
        obj_rel = np.tile(target_T[None], (len(release), 1, 1))
        obj_ret = np.tile(target_T[None], (len(retract_t), 1, 1))
        phases = [("rp_approach", approach, obj_app),
                  ("rp_move", move, obj_move),
                  ("rp_release", release, obj_rel),
                  ("rp_retract", retract_t, obj_ret)]
        return phases, tg
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="pepsi")
    ap.add_argument("--hand", default="inspire_left")
    ap.add_argument("--pose", type=int, default=2)
    ap.add_argument("--x0", type=float, default=0.40)
    ap.add_argument("--x1", type=float, default=0.50)
    ap.add_argument("--yaw1", type=float, default=10.0, help="target yaw deg")
    ap.add_argument("--out", default="outputs/seed_sweep/reposition_clean.npz")
    args = ap.parse_args()

    pl = GraspPlanner(hand=args.hand)
    import yourdfpy
    urdf = yourdfpy.URDF.load(str(exp.URDF_BY_HAND[args.hand]))
    mesh = str(exp.MESH_BASE / args.obj / "raw_mesh" / f"{args.obj}.obj")
    data = exp.compute_coverage_data(pl, args.obj, args.hand)
    gmeta = data["grasp_meta"]
    wrist_obj = data["wrist_obj"]; preg = data["pregrasp"]; reach = data["reach"]
    tab = exp.load_tabletop_poses(args.obj)
    print("INIT DONE", flush=True)

    p = args.pose
    gpose = np.array([g[3] for g in gmeta])
    cand = [int(g) for g in np.where(gpose == p)[0]]
    xi = int(np.argmin(np.abs(exp.X_GRID - args.x0)))            # source cell
    xi_t = int(np.argmin(np.abs(exp.X_GRID - args.x1)))          # target cell
    yi_t = int(np.argmin(np.abs(exp.YAW_GRID - np.radians(args.yaw1))))
    current_T = exp._place_T(tab[p], args.x0, 0.0)
    target_T = exp._place_T(tab[p], args.x1, np.radians(args.yaw1))
    # a transport grasp must hold the object at BOTH source and target
    tcands = [g for g in cand if reach[g, xi, 0] and reach[g, xi_t, yi_t]]
    print(f"pose {p}: {len(cand)} grasps, {len(tcands)} reachable at BOTH "
          f"x={args.x0} and (x={args.x1},yaw={args.yaw1})", flush=True)

    obj_verts = exp.load_obj_verts(mesh) if hasattr(exp, "load_obj_verts") else None

    t0 = time.time()
    res = reposition(pl, urdf, args.obj, args.hand, current_T, target_T,
                     tcands, wrist_obj, preg, mesh)
    dt = time.time() - t0
    if res is None:
        print(f"RES reposition FAILED in {dt:.1f}s", flush=True)
        return
    phases, tg = res
    nfr = sum(len(t[1]) for t in phases)
    disc = max((np.abs(np.diff(np.asarray(rt)[:, :6], axis=0)).max()
                for _, rt, _ in phases if len(rt) > 1), default=0.0)

    # object stays above the table through the held-move phase
    _, mv_q, mv_obj = phases[1]
    objbottom = float("nan")
    try:
        import trimesh
        m = trimesh.load(mesh, process=False)
        v = np.asarray(m.vertices, np.float32)
        zr = mv_obj[:, 2, :3]; zt = mv_obj[:, 2, 3]
        vz = v @ zr.T + zt[None, :]
        objbottom = float(vz.min())
    except Exception as e:
        print("  (table-clearance check skipped:", e, ")")

    print(f"RES reposition OK in {dt:.1f}s  grasp g{tg}  {len(phases)} phases "
          f"{nfr} frames  max jump {disc:.3f}rad  obj min-z {objbottom:.3f} "
          f"(table {exp.TABLE_SURFACE_Z})", flush=True)
    print("  phases:", [(nm, len(q)) for nm, q, _ in phases], flush=True)

    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out,
             robot=np.concatenate([q for _, q, _ in phases], 0),
             obj=np.concatenate([o for _, _, o in phases], 0),
             phase_lens=np.array([len(q) for _, q, _ in phases]),
             phase_names=np.array([nm for nm, _, _ in phases]),
             current_T=current_T, target_T=target_T, grasp=tg)
    print("  saved", out, flush=True)


if __name__ == "__main__":
    main()
