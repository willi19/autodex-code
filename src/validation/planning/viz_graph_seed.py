#!/usr/bin/env python3
"""Visualize cuRobo's GRAPH-SEARCH SEED (the coarse keypoints) vs the TRAJOPT
result — i.e. what the planner has BEFORE optimization vs after.

cuRobo plans in two stages (see autodex/planner/CLAUDE.md):
    1. graph search (PRM) -> a coarse collision-free config path = the SEED
       (a few keypoints, topologically correct but not smooth)
    2. trajopt -> optimizes the FULL fixed-length trajectory jointly (smooth,
       dynamically feasible), seeded by the graph path

``MotionGenResult`` exposes both: ``graph_plan`` (the seed keypoints) and
``optimized_plan`` / ``get_interpolated_plan()`` (after trajopt). This script
plans INIT -> a reachable grasp (object on the table as the obstacle to route
around), then shows in viser:
    * the GRAPH SEED keypoints  (robot stepping through graph_plan + EE spheres)
    * the OPTIMIZED trajectory  (smooth interpolated plan + EE path)

so you can see the few keypoints the optimizer starts from.

Usage:
    python src/validation/planning/viz_graph_seed.py --obj attached_container \\
        --hand inspire_left --version selected_100 --port 8080
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import numpy as np
import torch
import trimesh
import yourdfpy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))
from autodex.utils.path import obj_path
from autodex.utils.conversion import se32cart
from autodex.planner import GraspPlanner
from autodex.planner.obstacles import TABLE_CUBOID, add_obstacles
from autodex.planner.planner import _to_curobo_world
from curobo.types.state import JointState
from curobo.wrap.reacher.motion_gen import MotionGenPlanConfig
from paradex.visualization.visualizer.viser import ViserViewer

TABLE_SURFACE_Z = TABLE_CUBOID["pose"][2] + TABLE_CUBOID["dims"][2] / 2
EE_LINK = "base_link"
_URDF_ROOT = Path.home() / "shared_data" / "AutoDex" / "content" / "assets" / "robot"
URDF_BY_HAND = {
    "inspire_left": _URDF_ROOT / "inspire_left_description" / "xarm_inspire_left.urdf",
    "inspire":      _URDF_ROOT / "inspire_description"      / "xarm_inspire.urdf",
    "allegro":      _URDF_ROOT / "allegro_description"      / "xarm_allegro.urdf",
}


def fk_ee(urdf, traj):
    out = np.zeros((len(traj), 3))
    for t, q in enumerate(traj):
        urdf.update_cfg(q)
        out[t] = urdf.get_transform(EE_LINK, urdf.base_link)[:3, 3]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="attached_container")
    ap.add_argument("--hand", default="inspire_left")
    ap.add_argument("--version", default="selected_100")
    ap.add_argument("--start_x", type=float, default=0.45)
    ap.add_argument("--scene", default="table",
                    help="obstacle scene for the plan. table (default, reliable): "
                         "linear seed cuts through the object, optimized curves "
                         "around -> trajopt effect visible. wall/shelf/cluttered: "
                         "harder, more graph routing, but selected_100 grasps may "
                         "not plan against tight walls.")
    ap.add_argument("--wall_gap", type=float, default=0.06)
    ap.add_argument("--wall_angle", type=float, default=0.0)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    # object resting in its first tabletop pose, on the table (acts as obstacle)
    tdir = Path(obj_path) / args.obj / "processed_data" / "info" / "tabletop"
    T_obj = np.load(sorted(tdir.glob("*.npy"))[0])
    if T_obj.shape == (3, 3):
        M = np.eye(4); M[:3, :3] = T_obj; T_obj = M
    T_obj[0, 3], T_obj[1, 3] = args.start_x, 0.0
    T_obj[2, 3] += TABLE_SURFACE_Z
    mesh_path = Path(obj_path) / args.obj / "raw_mesh" / f"{args.obj}.obj"

    scene_cfg = {"mesh": {"target": {"pose": se32cart(T_obj).tolist(),
                                     "file_path": str(mesh_path)}}, "cuboid": {}}
    scene_cfg = add_obstacles(scene_cfg, args.scene,
                              wall_gap=args.wall_gap, wall_angle=args.wall_angle)

    print("[viz] warming up planner...")
    planner = GraspPlanner(hand=args.hand)
    urdf = yourdfpy.URDF.load(str(URDF_BY_HAND[args.hand]))

    # goal config = an IK-reachable grasp for this object
    print("[viz] solving IK for reachable grasps...")
    ik = planner.solve_ik(scene_cfg, args.obj, args.version, hand=args.hand)
    ok = list(np.where(ik["ik_success"])[0])
    if len(ok) == 0:
        sys.exit(f"no IK-reachable grasp for {args.obj}/{args.version}")
    np.random.shuffle(ok)
    planner._ik_solver = None                       # avoid graph-capture goal-type clash

    # motion_gen world = object mesh + table (so the graph must route around obj)
    world = _to_curobo_world(scene_cfg)
    planner._init_motion_gen(world); planner._cached_world = world
    dev = planner._tensor_args.device
    start = JointState.from_position(torch.tensor(planner._init_state, dtype=torch.float32,
                                                  device=dev).unsqueeze(0))
    cfg = MotionGenPlanConfig(enable_graph=True, need_graph_success=True,
                              max_attempts=10, enable_finetune_trajopt=True)
    # planning is stochastic — try grasps until one plans WITH a graph seed.
    res, goal_q = None, None
    for cand in ok[:20]:
        gq = ik["ik_qpos"][cand].astype(np.float32)
        goal = JointState.from_position(torch.tensor(gq, dtype=torch.float32,
                                                     device=dev).unsqueeze(0))
        print(f"[viz] planning grasp #{cand} (graph + trajopt)...")
        r = planner._motion_gen.plan_single_js(start_state=start, goal_state=goal,
                                               plan_config=cfg)
        if r.success.item() and r.graph_plan is not None and r.optimized_plan is not None:
            res, goal_q = r, gq
            break
    if res is None:
        sys.exit("[viz] no grasp planned with a graph seed — try another object/version.")
    print(f"[viz] success  used_graph={res.used_graph}  graph_time={res.graph_time:.2f}s "
          f"trajopt_time={res.trajopt_time:.2f}s")

    T = 64                                              # trajopt_tsteps
    def _interp(traj, n):                               # joint-space resample
        if len(traj) == 1:
            return np.repeat(traj, n, axis=0)
        xp = np.linspace(0, 1, len(traj))
        return np.stack([np.interp(np.linspace(0, 1, n), xp, traj[:, j])
                         for j in range(traj.shape[1])], axis=1)

    # the three things around trajopt:
    linear_q = _interp(np.stack([planner._init_state.astype(np.float32), goal_q]), T)  # naive seed
    gp = res.graph_plan.position.cpu().numpy()
    graph_q = gp[0] if gp.ndim == 3 else gp             # (steps, dof) — squeeze batch
    graph_seed_q = graph_q                              # the actual trajopt seed (graph path)
    opt_q = (res.optimized_plan.position.cpu().numpy()
             if res.optimized_plan is not None else
             res.get_interpolated_plan().position.cpu().numpy())          # trajopt output
    print(f"[viz] linear seed: {len(linear_q)} | graph seed path: {len(graph_q)} | "
          f"optimized: {len(opt_q)} steps")

    # === viser ===
    vis = ViserViewer(port_number=args.port)
    vis.add_floor(height=0.0)
    vis.add_robot("xarm", str(URDF_BY_HAND[args.hand]))
    vis.add_object("obj", trimesh.load(str(mesh_path), process=False), T_obj)
    # draw obstacle cuboids (table/walls) so the routing-around is visible
    from autodex.utils.conversion import cart2se3
    for cname, c in scene_cfg.get("cuboid", {}).items():
        if cname == "table":
            continue
        box = trimesh.creation.box(extents=np.asarray(c["dims"], float))
        vis.add_trimesh(f"obs_{cname}", box, cart2se3(c["pose"]))
    tile = lambda n: np.tile(T_obj[None], (n, 1, 1))
    # 1) naive LINEAR seed (often cuts through the object) — what trajopt would
    #    start from without graph; 2) GRAPH seed (routes around, jagged) = the
    #    real trajopt seed; 3) OPTIMIZED = trajopt output (smooth, collision-free)
    vis.add_traj("1_linear_seed", {"xarm": linear_q}, {"obj": tile(len(linear_q))})
    if graph_seed_q is not None:
        vis.add_traj("2_graph_seed", {"xarm": graph_seed_q}, {"obj": tile(len(graph_seed_q))})
    vis.add_traj("3_optimized", {"xarm": opt_q}, {"obj": tile(len(opt_q))})

    # EE paths: blue = linear seed, red = graph seed path, green = optimized
    for i, p in enumerate(fk_ee(urdf, linear_q)[::4]):
        vis.add_sphere(f"lin_{i}", position=p, radius=0.006, color=(0.2, 0.4, 1.0))
    for i, p in enumerate(fk_ee(urdf, graph_q)[::4]):
        vis.add_sphere(f"gseed_{i}", position=p, radius=0.012, color=(1.0, 0.3, 0.1))
    for i, p in enumerate(fk_ee(urdf, opt_q)[::4]):
        vis.add_sphere(f"opt_{i}", position=p, radius=0.008, color=(0.1, 1.0, 0.1))

    print(f"[viz] http://localhost:{args.port}")
    print("[viz] Playback: 1_linear_seed (naive, may collide) -> 2_graph_seed "
          "(routes around, jagged) -> 3_optimized (trajopt: smooth + collision-free)")
    print("[viz] EE: blue=linear seed, red=graph keypoints, green=optimized")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
