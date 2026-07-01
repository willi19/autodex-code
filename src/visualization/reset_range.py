#!/usr/bin/env python3
"""Radial reset-feasibility preview.

Show WHERE a reorient (reset) is possible by placing the actual object mesh at
every pickup configuration that ``sweep_reset.py`` tested, colored by whether the
reset planned successfully there.

The sweep cache (``outputs/reset_cache/{hand}/{obj}/reorient_{h_cm}/{i}_{j}/
sweep_summary.json``) was computed over a grid of pickup x-distance and pickup
yaw ``theta_z``. For a chosen source pose ``i`` (and target ``j``, or "any"), we
lay the object out radially around the robot base — radius = pickup x, azimuth =
pickup theta_z, object yawed by theta_z — and color each copy:

    green  = reset feasible here     red = infeasible     gray = no cache cell

This mirrors ``src/validation/planning/preview_polar_grid.py`` (the polar
reachability preview), but driven by the reset cache instead of IK reachability.

Radio-box (dropdown) controls: hand, object, h_cm, source pose i, target j.

Usage:
    python src/visualization/reset_range.py
    python src/visualization/reset_range.py --obj attached_container --hand inspire_left --port 8080
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
sys.path.insert(0, os.path.join(os.path.expanduser("~"), "paradex"))

from paradex.visualization.visualizer.viser import ViserViewer
from autodex.utils.path import repo_dir, obj_path, project_dir
from autodex.utils.robot_config import INIT_STATE, XARM_INIT, INSPIRE_INIT


CACHE_ROOT = os.path.join(repo_dir, "outputs", "reset_cache")
TABLE_DIMS = [2, 3, 0.2]
TABLE_POSE_XYZ = [1.1, 0, -0.1]
TABLE_TOP_Z = TABLE_POSE_XYZ[2] + TABLE_DIMS[2] / 2  # 0.0
COLOR_TABLE = (0.94, 0.94, 0.96, 0.6)

C_OK = (0.18, 0.78, 0.28)
C_FAIL = (0.85, 0.18, 0.16)
C_NONE = (0.5, 0.5, 0.5)

_ROBOT_ROOT = os.path.join(project_dir, "content", "assets", "robot")
HAND_URDF = {
    "allegro": os.path.join(_ROBOT_ROOT, "allegro_description", "xarm_allegro.urdf"),
    "inspire": os.path.join(_ROBOT_ROOT, "inspire_description", "xarm_inspire.urdf"),
    "inspire_left": os.path.join(_ROBOT_ROOT, "inspire_left_description", "xarm_inspire_left.urdf"),
}


# --------------------------------------------------------------------------- #
# cache discovery
# --------------------------------------------------------------------------- #
def hands_with_cache() -> list[str]:
    if not os.path.isdir(CACHE_ROOT):
        return []
    return sorted(h for h in os.listdir(CACHE_ROOT)
                  if os.path.isdir(os.path.join(CACHE_ROOT, h)))


def objects_for_hand(hand: str) -> list[str]:
    d = os.path.join(CACHE_ROOT, hand)
    return sorted(o for o in os.listdir(d)
                  if os.path.isdir(os.path.join(d, o))) if os.path.isdir(d) else []


def h_cms_for(hand: str, obj: str) -> list[int]:
    d = os.path.join(CACHE_ROOT, hand, obj)
    if not os.path.isdir(d):
        return []
    hs = []
    for n in os.listdir(d):
        if n.startswith("reorient_") and n.split("_", 1)[1].isdigit():
            hs.append(int(n.split("_", 1)[1]))
    return sorted(hs)


def pair_dirs(hand: str, obj: str, h_cm: int) -> dict:
    """(i, j) -> sweep dir for every pair that has a summary."""
    base = os.path.join(CACHE_ROOT, hand, obj, f"reorient_{h_cm}")
    out = {}
    if not os.path.isdir(base):
        return out
    for n in os.listdir(base):
        d = os.path.join(base, n)
        if os.path.isdir(d) and "_" in n:
            a, b = n.split("_", 1)
            if a.isdigit() and b.isdigit() and os.path.exists(os.path.join(d, "sweep_summary.json")):
                out[(int(a), int(b))] = d
    return out


def load_summary(d: str) -> dict | None:
    f = os.path.join(d, "sweep_summary.json")
    return json.load(open(f)) if os.path.exists(f) else None


def rotz(rad: float) -> np.ndarray:
    c, s = np.cos(rad), np.sin(rad)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", default="attached_container")
    parser.add_argument("--hand", default="inspire_left")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    hands = hands_with_cache() or ["inspire_left"]
    hand0 = args.hand if args.hand in hands else hands[0]
    objs0 = objects_for_hand(hand0)
    obj0 = args.obj if args.obj in objs0 else (objs0[0] if objs0 else args.obj)

    vis = ViserViewer(port_number=args.port)
    vis.add_robot("robot", HAND_URDF[hand0])
    vis.add_floor(0.0)

    table_mesh = trimesh.creation.box(extents=TABLE_DIMS)
    table_pose = np.eye(4)
    table_pose[:3, 3] = TABLE_POSE_XYZ
    vis.add_object("table", table_mesh, table_pose)
    vis.change_color("table", COLOR_TABLE)

    # ---- mutable view state ----
    st = {
        "mesh_v": None, "mesh_f": None,   # plain mesh arrays for colored copies
        "tab": {},                        # pose_idx -> tabletop 4x4
        "pool": [],                       # list of (name, xi, ti) currently placed
        "x_values": [], "tz_values": [],
    }

    # ---- GUI ----
    g = vis.server.gui
    dd_hand = g.add_dropdown("hand", tuple(hands), initial_value=hand0)
    dd_obj = g.add_dropdown("object", tuple(objs0) or ("",), initial_value=obj0)
    hcms0 = h_cms_for(hand0, obj0)
    dd_hcm = g.add_dropdown("h_cm", tuple(str(h) for h in hcms0) or ("0",),
                            initial_value=str(hcms0[0]) if hcms0 else "0")
    dd_src = g.add_dropdown("source pose i", ("0",), initial_value="0")
    dd_tgt = g.add_dropdown("target j", ("any",), initial_value="any")
    status = g.add_markdown("```\nloading...\n```")

    def _status(msg):
        status.content = f"```\n{msg}\n```"

    # ---- robot config ----
    def set_robot(hand):
        if hand.startswith("inspire"):
            init = np.concatenate([XARM_INIT, INSPIRE_INIT]).astype(np.float32)
        else:
            init = INIT_STATE
        vis.robot_dict["robot"].update_cfg(init)

    def swap_robot(hand):
        if "robot" in vis.robot_dict:
            vis.robot_dict["robot"].remove()
            del vis.robot_dict["robot"]
        vis.add_robot("robot", HAND_URDF[hand])
        set_robot(hand)

    # ---- mesh ----
    def load_mesh(obj):
        raw = os.path.join(obj_path, obj, "raw_mesh", f"{obj}.obj")
        simp = os.path.join(obj_path, obj, "processed_data", "mesh", "simplified.obj")
        path = raw if os.path.exists(raw) else simp
        m = trimesh.load(path, force="mesh", process=False)
        st["mesh_v"] = np.asarray(m.vertices, dtype=np.float32)
        st["mesh_f"] = np.asarray(m.faces, dtype=np.uint32)
        td = os.path.join(obj_path, obj, "processed_data", "info", "tabletop")
        st["tab"] = {int(f[:-4]): np.load(os.path.join(td, f))
                     for f in os.listdir(td) if f.endswith(".npy")}

    # ---- object pool (one colored copy per (x, tz) cell) ----
    def clear_pool():
        for name, _, _ in st["pool"]:
            try:
                vis.obj_dict[name]["frame"].remove()
                del vis.obj_dict[name]
            except Exception:
                pass
        st["pool"] = []

    def place_cell(name, tab_i, x, tz_rad):
        """Object at radius=x, azimuth=tz, yawed by tz (display layout)."""
        T = tab_i.copy()
        T[:3, :3] = rotz(tz_rad) @ tab_i[:3, :3]
        T[:3, 3] = rotz(tz_rad) @ np.array([x, 0.0, tab_i[2, 3]])
        fr = vis.obj_dict[name]["frame"]
        fr.position = T[:3, 3].astype(np.float32)
        fr.wxyz = Rot.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]].astype(np.float32)

    def set_color(name, color):
        try:
            vis.obj_dict[name]["handle"].remove()
        except Exception:
            pass
        h = vis.server.scene.add_mesh_simple(
            name=f"/objects/{name}_frame/{name}",
            vertices=st["mesh_v"], faces=st["mesh_f"],
            color=tuple(int(round(c * 255)) for c in color), opacity=0.95)
        vis.obj_dict[name]["handle"] = h

    # ---- feasibility lookup ----
    def status_map(hand, obj, h_cm, i, tgt):
        """(x, tz) -> 'ok'/'fail'/None for source i and target j (or 'any')."""
        pairs = pair_dirs(hand, obj, h_cm)
        out_j = sorted(j for (ii, j) in pairs if ii == i)
        if not out_j:
            return {}, [], []
        js = out_j if tgt == "any" else [int(tgt)]
        xs, tzs, agg = None, None, {}
        for j in js:
            s = load_summary(pairs[(i, j)])
            if s is None:
                continue
            if xs is None:
                xs, tzs = list(s["x_values"]), list(s["tz_values"])
            for c in s["cells"]:
                key = (c["x"], c["tz"])
                prev = agg.get(key)
                cur = c["status"]
                # 'any': ok wins; otherwise keep ok over fail
                if prev == "ok":
                    continue
                agg[key] = cur
        return agg, (xs or []), (tzs or [])

    # ---- rebuild everything for the current selection ----
    def rebuild():
        hand, obj = dd_hand.value, dd_obj.value
        if not (hand and obj):
            return
        try:
            h_cm = int(dd_hcm.value)
        except ValueError:
            return
        try:
            i = int(dd_src.value)
        except ValueError:
            return

        amap, xs, tzs = status_map(hand, obj, h_cm, i, dd_tgt.value)
        # (re)build the pool to match the grid size
        need = [(xi, ti) for xi in range(len(xs)) for ti in range(len(tzs))]
        if [(p[1], p[2]) for p in st["pool"]] != need:
            clear_pool()
            for xi, ti in need:
                name = f"cell_{xi}_{ti}"
                vis.add_object(name, trimesh.Trimesh(
                    vertices=st["mesh_v"], faces=st["mesh_f"], process=False), np.eye(4))
                st["pool"].append((name, xi, ti))
        st["x_values"], st["tz_values"] = xs, tzs

        tab_i = st["tab"][i]
        n_ok = n_fail = n_none = 0
        for name, xi, ti in st["pool"]:
            x, tz = xs[xi], tzs[ti]
            place_cell(name, tab_i, float(x), float(np.radians(tz)))
            stt = amap.get((x, tz))
            if stt == "ok":
                set_color(name, C_OK); n_ok += 1
            elif stt == "fail":
                set_color(name, C_FAIL); n_fail += 1
            else:
                set_color(name, C_NONE); n_none += 1

        tgt_txt = "any target" if dd_tgt.value == "any" else f"target {dd_tgt.value}"
        _status(f"{hand} / {obj} / reorient_{h_cm}\n"
                f"source pose i={i}  ->  {tgt_txt}\n"
                f"feasible: {n_ok}   infeasible: {n_fail}   no-cache: {n_none}\n"
                f"(green=ok  red=fail)  radius=pickup x  angle=pickup theta_z")

    # ---- cascade dropdown options ----
    def refresh_src_tgt():
        hand, obj = dd_hand.value, dd_obj.value
        try:
            h_cm = int(dd_hcm.value)
        except ValueError:
            h_cm = 0
        pairs = pair_dirs(hand, obj, h_cm)
        srcs = sorted({ii for (ii, _) in pairs}) or [0]
        dd_src.options = tuple(str(s) for s in srcs)
        if dd_src.value not in dd_src.options:
            dd_src.value = dd_src.options[0]
        i = int(dd_src.value)
        tgts = ["any"] + [str(j) for (ii, j) in sorted(pairs) if ii == i]
        dd_tgt.options = tuple(tgts)
        if dd_tgt.value not in dd_tgt.options:
            dd_tgt.value = "any"

    @dd_hand.on_update
    def _(_):
        hand = dd_hand.value
        swap_robot(hand)
        objs = objects_for_hand(hand) or [""]
        dd_obj.options = tuple(objs)
        if dd_obj.value not in objs:
            dd_obj.value = objs[0]
        # obj handler continues the chain

    @dd_obj.on_update
    def _(_):
        hand, obj = dd_hand.value, dd_obj.value
        if obj:
            load_mesh(obj)
        hcms = h_cms_for(hand, obj)
        dd_hcm.options = tuple(str(h) for h in hcms) or ("0",)
        if dd_hcm.value not in dd_hcm.options:
            dd_hcm.value = dd_hcm.options[0]
        clear_pool()  # mesh changed -> force pool rebuild
        refresh_src_tgt()
        rebuild()

    @dd_hcm.on_update
    def _(_):
        refresh_src_tgt()
        rebuild()

    @dd_src.on_update
    def _(_):
        refresh_src_tgt()
        rebuild()

    @dd_tgt.on_update
    def _(_):
        rebuild()

    # ---- initial draw ----
    set_robot(hand0)
    load_mesh(obj0)
    refresh_src_tgt()
    rebuild()

    print(f"[reset_range] http://localhost:{args.port}")
    vis.start_viewer()


if __name__ == "__main__":
    main()
