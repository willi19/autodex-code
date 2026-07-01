#!/usr/bin/env python3
"""Side-by-side IK comparison: det (seed=INIT, 1 attempt) vs random (32 attempts).

Two robots OVERLAID at the same base (blue = det, red = random) for one grasp
pose. Where the two IK solutions agree they coincide; where they pick a different
branch (e.g. a wrapped wrist joint) they split — so you SEE the difference. The
joint table shows each joint's det / random value and their difference (deg).

Sliders pick grasp # / radius / yaw; "resample random" re-rolls the random IK.

    conda activate mingi
    python src/visualization/seed_ik_viz.py --obj pepsi --hand inspire_left --pose 2 --port 8080
"""
import argparse, os, sys, threading
from pathlib import Path
import numpy as np, torch, trimesh

sys.path.insert(0, "/home/mingi/AutoDex")
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "paradex"))

import src.visualization.exp as exp
from autodex.planner import GraspPlanner
from autodex.planner.planner import _to_curobo_world, _to_curobo_pose
from autodex.utils.path import project_dir
from src.grasp_generation.order.compute_order import load_grasp_data
from paradex.visualization.visualizer.viser import ViserViewer


def rotz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obj", default="pepsi")
    ap.add_argument("--hand", default="inspire_left")
    ap.add_argument("--pose", type=int, default=2)
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    pl = GraspPlanner(hand=args.hand)
    INIT = pl._init_state
    info, wo, preg = load_grasp_data(
        str(Path(project_dir) / "candidates" / args.hand / exp.GRASP_VERSION), args.obj)
    gm = [(g[1], g[2], g[3], exp.scene_pose_idx(args.obj, g[1], g[2])) for g in info]
    tab = exp.load_tabletop_poses(args.obj)
    gfp = [i for i, m in enumerate(gm) if m[3] == args.pose]
    z0 = float(tab[args.pose][2, 3]) + exp.TABLE_SURFACE_Z
    tabR = tab[args.pose][:3, :3]

    pl._init_ik_solver(_to_curobo_world({"mesh": {}, "cuboid": {"table": exp.TABLE_CUBOID}}),
                       use_cuda_graph=False)
    dev = pl._tensor_args.device
    it = torch.tensor(INIT, dtype=torch.float32, device=dev)

    def placeT(r, yaw):
        T = np.eye(4); T[:3, :3] = rotz(yaw) @ tabR; T[:3, 3] = [r, 0, z0]; return T

    def ik_det(Tw):
        r = pl._ik_solver.solve_single(_to_curobo_pose(Tw[None], dev), retract_config=it.unsqueeze(0),
                                       seed_config=it.view(1, 1, -1), num_seeds=1)
        return r.solution.cpu().numpy().reshape(-1)[:6] if bool(r.success.view(-1)[0]) else None

    def ik_rand(Tw):
        r = pl._ik_solver.solve_batch(_to_curobo_pose(Tw[None], dev), retract_config=it.unsqueeze(0))
        return r.solution.cpu().numpy().reshape(-1)[:6] if bool(r.success.view(-1)[0]) else None

    # ---- viser ----
    URDF = exp.URDF_BY_HAND
    vis = ViserViewer(port_number=args.port)
    vis.add_robot("det", str(URDF[args.hand]))
    vis.add_robot("rand", str(URDF[args.hand]))
    vis.robot_dict["det"].update_cfg(INIT)
    vis.robot_dict["rand"].update_cfg(INIT)
    # robot meshes are created from trimesh vertex_colors (0-255), so robot color
    # must be 0-255 — 0-1 floats render near-black. (objects take 0-1.)
    vis.change_color("det", (50, 110, 240))          # blue
    vis.change_color("rand", (240, 50, 50))          # red
    vis.add_floor(0.0)
    tbl = trimesh.creation.box(extents=exp.TABLE_CUBOID["dims"])
    tp = np.eye(4); tp[:3, 3] = exp.TABLE_CUBOID["pose"][:3]
    vis.add_object("table", tbl, tp)
    vis.change_color("table", (0.9, 0.9, 0.92, 0.4))

    g = vis.server.gui
    sl_g = g.add_slider("grasp #", min=0, max=len(gfp) - 1, step=1, initial_value=0)
    sl_r = g.add_slider("radius (m)", min=0.30, max=0.58, step=0.005, initial_value=0.45)
    sl_y = g.add_slider("yaw (deg)", min=0, max=359, step=1, initial_value=0)
    btn = g.add_button("resample random")
    status = g.add_markdown("```\n...\n```")
    lock = threading.Lock()

    def cfg(q):
        f = INIT.copy(); f[:6] = q; return f

    def update(_=None):
        with lock:
            gi = gfp[int(sl_g.value)]
            Tw = placeT(float(sl_r.value), np.radians(sl_y.value)) @ wo[gi]
            qd, qr = ik_det(Tw), ik_rand(Tw)
            vis.robot_dict["det"].update_cfg(cfg(qd) if qd is not None else INIT)
            vis.robot_dict["rand"].update_cfg(cfg(qr) if qr is not None else INIT)
            rows = "joint |   det |  rand |  diff\n------+-------+-------+------\n"
            for j in range(6):
                d = "    --" if qd is None else f"{np.degrees(qd[j]):6.0f}"
                rr = "    --" if qr is None else f"{np.degrees(qr[j]):6.0f}"
                df = "   --" if (qd is None or qr is None) else f"{np.degrees(qd[j] - qr[j]):5.0f}"
                rows += f"  {j + 1}   |{d} |{rr} |{df}\n"
            mx = ("" if (qd is None or qr is None)
                  else f"max |diff| = {np.degrees(np.abs(qd - qr).max()):.0f} deg")
            status.content = (f"```\nblue=det(seed=INIT,1)   red=random(32)\n"
                              f"det {'OK' if qd is not None else 'FAIL'}   "
                              f"rand {'OK' if qr is not None else 'FAIL'}\n\n{rows}\n{mx}\n```")

    for s in (sl_g, sl_r, sl_y):
        s.on_update(update)
    btn.on_click(update)
    update()
    print(f"[seed_ik_viz] http://localhost:{args.port}")
    vis.start_viewer()


if __name__ == "__main__":
    main()
