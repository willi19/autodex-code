#!/usr/bin/env python3
"""Replay viz — visualize EXACTLY what the sweep computed (NO re-planning).

Loads a `seed_sweep.py --save_results` pkl and steps through the configs, playing
the saved seed (the trajopt input) and, for successes, the saved plan trajectory.
Because it REPLAYS stored values instead of recomputing, a sweep FAIL shows as a
FAIL here too — viz == sweep by construction (no GPU-floating mismatch).

    conda activate mingi
    # 1) sweep saves results:
    python src/visualization/seed_sweep.py --obj pepsi --hand inspire_left --pose 2 \
        --n_yaw 12 --save_results /tmp/sweep_p2.pkl
    # 2) replay:
    python src/visualization/seed_replay_viz.py --results /tmp/sweep_p2.pkl --port 8080
"""
import argparse, os, sys, threading, pickle
from pathlib import Path
import numpy as np, trimesh
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, "/home/mingi/AutoDex")
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "paradex"))

import src.visualization.exp as exp
from autodex.utils.path import obj_path
from paradex.visualization.visualizer.viser import ViserViewer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="pkl from seed_sweep --save_results")
    ap.add_argument("--port", type=int, default=8080)
    args = ap.parse_args()

    import math
    d = pickle.load(open(args.results, "rb"))
    R, obj, hand = d["results"], d["obj"], d["hand"]
    nfail = sum(1 for e in R if not e["ok"])
    print(f"loaded {len(R)} configs ({nfail} fail / {len(R) - nfail} ok) from {args.results}")

    def _hump_deg(e, j):                                   # joint-j overshoot above its endpoints (deg)
        if not e["ok"] or e.get("traj") is None:
            return 0.0
        tr = np.asarray(e["traj"])
        return math.degrees(float(tr[:, j].max() - max(tr[0, j], tr[-1, j])))
    HUMP1 = [_hump_deg(e, 1) for e in R]
    HUMP2 = [_hump_deg(e, 2) for e in R]                   # joint2 (elbow) per-joint proxy
    ZRISE = [float(e.get("zrise", 0.0)) for e in R]        # wrist Cartesian z-rise (m) — the TRUE "솟구침"
    REDUND = [float(e.get("redundancy", 0.0)) for e in R]  # joint redundancy (deg, beyond [start,end])
    ZOLD = [float(e.get("old_zrise", 0.0)) for e in R]     # OLD cache (pre j0-sweep) z-rise, same cell
    BENEFIT = [ZOLD[i] - ZRISE[i] for i in range(len(R))]  # how much flatter NEW is (m)

    URDF = exp.URDF_BY_HAND
    import yourdfpy                                          # arm joint limits for the sliders
    _aj = yourdfpy.URDF.load(str(URDF[hand])).actuated_joints[:6]
    ARM_LIMS = [(float(j.limit.lower), float(j.limit.upper)) for j in _aj]
    mesh_path = str(Path(obj_path) / obj / "raw_mesh" / f"{obj}.obj")
    if not Path(mesh_path).exists():
        mesh_path = str(Path(obj_path) / obj / "processed_data" / "mesh" / "simplified.obj")
    obj_mesh = trimesh.load(mesh_path, force="mesh", process=False)

    vis = ViserViewer(port_number=args.port)
    vis.add_robot("xarm", str(URDF[hand]))
    vis.add_robot("xarm_old", str(URDF[hand]))             # OLD (pre-j0-sweep) ghost overlay
    vis.change_color("xarm_old", (240, 150, 30))           # orange (robots want 0-255)
    vis.add_floor(0.0)
    tbl = trimesh.creation.box(extents=exp.TABLE_CUBOID["dims"])
    tp = np.eye(4); tp[:3, 3] = exp.TABLE_CUBOID["pose"][:3]
    vis.add_object("table", tbl, tp)
    vis.change_color("table", (0.9, 0.9, 0.92, 0.4))
    vis.add_object("obj", obj_mesh, np.eye(4))

    g = vis.server.gui
    dd = g.add_dropdown("filter", ("fails", "success", "all", "worst-zrise", "worst-redund", "j0-benefit", "worst-hump-j2"), initial_value="fails")
    btn_p = g.add_button("◀ prev")
    btn_n = g.add_button("next ▶")
    cb_seed = g.add_checkbox("show SEED (trajopt input) not the plan", initial_value=False)
    cb_old = g.add_checkbox("overlay OLD ghost (orange, pre-j0-sweep)", initial_value=True)
    status = g.add_markdown("```\n...\n```")
    frame_sl = g.add_slider("frame (scrub)", 0, 1, 1, 0)
    joint_sl = [g.add_slider(f"j{k}  [{np.degrees(lo):.0f}..{np.degrees(hi):.0f}]",
                             float(lo), float(hi), 0.001, 0.0) for k, (lo, hi) in enumerate(ARM_LIMS)]
    state = {"idx": 0, "list": [], "new": None, "old": None}
    lock = threading.Lock()
    C_OK, C_FAIL, C_SEED = (0.18, 0.78, 0.28), (0.85, 0.18, 0.16), (0.25, 0.5, 0.95)

    def set_obj_pose(T):
        fr = vis.frame_nodes["obj"]
        fr.position = T[:3, 3].astype(np.float32)
        fr.wxyz = Rot.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]].astype(np.float32)

    def refilter():
        f = dd.value
        if f in ("worst-zrise", "worst-redund", "worst-hump-j2", "j0-benefit"):   # sorted worst/best-first
            idxs = [i for i, e in enumerate(R) if e["ok"] and e.get("traj") is not None]
            key = {"worst-zrise": ZRISE, "worst-redund": REDUND, "worst-hump-j2": HUMP2, "j0-benefit": BENEFIT}[f]
            state["list"] = sorted(idxs, key=lambda i: key[i], reverse=True)
        else:
            state["list"] = [i for i, e in enumerate(R)
                             if f == "all" or (f == "fails" and not e["ok"]) or (f == "success" and e["ok"])]
        state["idx"] = 0

    def show():
        with lock:
            if not state["list"]:
                status.content = f"```\n(no {dd.value})\n```"; return
            e = R[state["list"][state["idx"]]]
            ridx = state["list"][state["idx"]]
            T = np.asarray(e["T"]); set_obj_pose(T)
            vis.clear_traj()
            # NEW (xarm, green) = the j0-sweep plan; for fails fall back to its seed.
            if e["ok"] and e["traj"] is not None:
                new = np.asarray(e["traj"]); vis.change_color("obj", C_OK)
            elif e["adj"] is not None:
                new = np.asarray(e["adj"]); vis.change_color("obj", C_FAIL if not e["ok"] else C_SEED)
            else:
                new = None
            # OLD (xarm_old, orange ghost) = pre-j0-sweep cache trajectory at this cell,
            # re-anchored (joint-space ramp) so its START+END coincide with NEW's — only
            # the PATH (the arc) differs, not the grasp endpoint (removes the 1.8cm off-grid seam).
            old = None
            if cb_old.value and e.get("old_traj") is not None and new is not None:
                old = np.asarray(e["old_traj"], dtype=np.float32).copy()
                a = np.linspace(0.0, 1.0, len(old))[:, None]
                old = old + (1 - a) * (new[0] - old[0]) + a * (new[-1] - old[-1])
            state["new"], state["old"] = new, old
            if new is not None:
                frame_sl.max = len(new) - 1
                frame_sl.value = len(new) // 2             # arcs peak mid-trajectory
                set_frame(len(new) // 2)
            status.content = (f"```\nfilter={dd.value}   {state['idx']+1}/{len(state['list'])}\n"
                              f"gi={e['gi']} r={e['r']} yaw={e['yaw_deg']}  dr={e['dr']} dyaw={e['dyaw']}\n"
                              f"wrist z-rise={100*ZRISE[ridx]:.1f}cm   redundancy={e.get('redundancy',0):.0f}deg\n"
                              f"SWEEP RESULT: {'OK' if e['ok'] else 'FAIL'}    "
                              f"OLD ghost: {'ON' if old is not None else 'off/none'}\n"
                              f"drag `frame` to scrub — watch the orange (OLD) wrist arc up vs green (NEW) flat\n```")

    def set_frame(f):                                      # scrub BOTH robots + joint sliders to frame f
        new, old = state["new"], state["old"]
        if new is None:
            return
        f = int(max(0, min(f, len(new) - 1)))
        qn = np.asarray(new[f])
        vis.robot_dict["xarm"].update_cfg(qn)
        if old is not None and len(old) > 0:               # sync OLD by trajectory fraction
            of = int(round(f / max(len(new) - 1, 1) * (len(old) - 1)))
            vis.robot_dict["xarm_old"].update_cfg(np.asarray(old[of]))
        else:
            vis.robot_dict["xarm_old"].update_cfg(qn)      # no OLD -> overlap NEW (ghost hidden)
        for k in range(6):
            lo, hi = ARM_LIMS[k]
            joint_sl[k].value = float(min(max(qn[k], lo), hi))   # clamp to slider range

    frame_sl.on_update(lambda _: set_frame(int(frame_sl.value)))

    def nav(s):
        if state["list"]:
            state["idx"] = (state["idx"] + s) % len(state["list"]); show()

    btn_p.on_click(lambda _: nav(-1))
    btn_n.on_click(lambda _: nav(1))
    dd.on_update(lambda _: (refilter(), show()))
    cb_seed.on_update(lambda _: show())
    cb_old.on_update(lambda _: show())

    refilter()
    show()
    print(f"[seed_replay_viz] http://localhost:{args.port}")
    vis.start_viewer()


if __name__ == "__main__":
    main()
