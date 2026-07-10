#!/usr/bin/env python3
"""Interactive cuRobo collision-sphere editor.

Loads ``spheres/xarm_<hand>.yml`` + the corresponding URDF, shows the robot
plus all spheres in viser, and exposes per-sphere GUI sliders (radius, cx,
cy, cz IN LINK-LOCAL FRAME). Tweak → click Save → writes a new yaml.

Usage:
    python src/experiment/reset/edit_spheres.py
    python src/experiment/reset/edit_spheres.py --hand inspire_left \\
        --filter thumb --out /tmp/xarm_inspire_left.edited.yml
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import viser
import yaml
import yourdfpy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../.."))

from src.experiment.reset.replay_trial import (  # noqa: E402
    ARM_JOINTS, HAND_JOINTS_INSPIRE_LEFT, URDF_BY_HAND, SPHERES_BY_HAND,
)

# Reasonable seed pose so curled fingertips are visible (arm INIT, hand half-closed).
DEFAULT_ARM = np.array([-1.267, -0.202, -1.136, 2.332, 0.323, 2.365])
DEFAULT_HAND = np.array([0.3, 0.3, 0.5, 0.5, 0.5, 0.5])  # thumb_yaw/pitch + 4 fingers


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _save_yaml(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=False)


def _fk_link_T(urdf: yourdfpy.URDF, arm_q: np.ndarray, hand_q: np.ndarray,
               link: str, arm_names: list[str], hand_names: list[str]):
    cfg = {**{arm_names[i]: float(arm_q[i]) for i in range(6)},
           **{hand_names[i]: float(hand_q[i]) for i in range(6)}}
    urdf.update_cfg(cfg)
    return urdf.get_transform(link, urdf.base_link)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", default="inspire_left",
                        choices=["inspire_left", "inspire", "allegro"])
    parser.add_argument("--filter", default=None,
                        help="Substring filter for link names (e.g. 'thumb' "
                             "→ edit only thumb spheres). Default: all links.")
    parser.add_argument("--out", default=None,
                        help="Output yaml path. Default = original path + "
                             "'.edited.yml'.")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    if args.hand != "inspire_left":
        sys.exit("only inspire_left wired up (extend HAND_JOINTS_* if needed).")

    src_path = SPHERES_BY_HAND[args.hand]
    out_path = (Path(args.out) if args.out
                else src_path.with_suffix(".edited.yml"))
    print(f"[edit] source: {src_path}")
    print(f"[edit] output: {out_path}")

    data = _load_yaml(src_path)
    spheres = data["collision_spheres"]      # dict[link] -> list[{center, radius}]

    urdf = yourdfpy.URDF.load(str(URDF_BY_HAND[args.hand]))

    # Build (link, idx) catalogue.
    catalogue: list[tuple[str, int]] = []
    for link, slist in spheres.items():
        if args.filter and args.filter not in link:
            continue
        for i in range(len(slist)):
            catalogue.append((link, i))
    print(f"[edit] {len(catalogue)} editable spheres "
          f"({len(set(l for l,_ in catalogue))} links)")

    # FK cache per link (recomputed when arm/hand sliders change).
    arm_q = DEFAULT_ARM.copy()
    hand_q = DEFAULT_HAND.copy()
    link_T = {l: _fk_link_T(urdf, arm_q, hand_q, l, ARM_JOINTS,
                            HAND_JOINTS_INSPIRE_LEFT)
              for l, _ in catalogue}

    server = viser.ViserServer(port=args.port)

    # ---- robot visuals (per-link static meshes at current FK) -----------
    robot_url = os.path.dirname(str(URDF_BY_HAND[args.hand]))
    link_handles = {}
    for link_name, link in urdf.link_map.items():
        if not link.visuals:
            continue
        vis = link.visuals[0]
        # URDF visuals can be <mesh> or primitives (<sphere>, <box>, <cylinder>).
        # We only render mesh visuals here — skip primitive-geometry links
        # (e.g. the fingertip dummy markers).
        if vis.geometry.mesh is None or vis.geometry.mesh.filename is None:
            continue
        mf = vis.geometry.mesh.filename
        mp = mf if os.path.isabs(mf) else os.path.join(robot_url, mf)
        if not os.path.exists(mp):
            continue
        try:
            import trimesh as _tm
            m = _tm.load(mp, process=False, force="mesh")
        except Exception:
            continue
        T = (_fk_link_T(urdf, arm_q, hand_q, link_name, ARM_JOINTS,
                        HAND_JOINTS_INSPIRE_LEFT)
             if link_name != urdf.base_link else np.eye(4))
        from scipy.spatial.transform import Rotation as R_
        h = server.scene.add_mesh_trimesh(
            f"/robot/{link_name}", mesh=m,
            position=T[:3, 3],
            wxyz=R_.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]],
        )
        link_handles[link_name] = (h, link_name)

    # ---- spheres --------------------------------------------------------
    # current_state[(link, idx)] = (center_xyz_local, radius)  in METERS
    current_state: dict[tuple[str, int], tuple[np.ndarray, float]] = {}
    sphere_handles: dict[tuple[str, int], object] = {}

    def _world_pos(link: str, c_local: np.ndarray) -> np.ndarray:
        T = link_T[link]
        return T[:3, :3] @ c_local + T[:3, 3]

    def _color(selected: bool) -> tuple[float, float, float]:
        return (1.0, 0.85, 0.0) if selected else (1.0, 0.3, 0.3)

    for (link, idx) in catalogue:
        s = spheres[link][idx]
        c = np.array(s["center"], dtype=np.float64)
        r = float(s["radius"])
        current_state[(link, idx)] = (c.copy(), r)
        h = server.scene.add_icosphere(
            f"/spheres/{link}_{idx:02d}",
            radius=r,
            position=tuple(_world_pos(link, c)),
            color=_color(False),
            opacity=0.45,
        )
        sphere_handles[(link, idx)] = h

    # ---- GUI ------------------------------------------------------------
    options = tuple(f"{link} [{idx}]" for link, idx in catalogue)
    sel = server.gui.add_dropdown("Sphere", options=options)
    r_sl  = server.gui.add_slider("radius (mm)",   1.0, 60.0, step=0.5, initial_value=10.0)
    cx_sl = server.gui.add_slider("cx (mm)",   -100.0, 100.0, step=0.5, initial_value=0.0)
    cy_sl = server.gui.add_slider("cy (mm)",   -100.0, 100.0, step=0.5, initial_value=0.0)
    cz_sl = server.gui.add_slider("cz (mm)",   -100.0, 100.0, step=0.5, initial_value=0.0)
    info_md = server.gui.add_markdown("(select a sphere)")
    save_btn = server.gui.add_button("Save yaml")
    saved_label = server.gui.add_markdown("")

    # Hand-pose sliders (so user can curl fingers etc.).
    server.gui.add_markdown("---\n**Hand pose** (radians)")
    hand_sliders = []
    for i, n in enumerate(["thumb_yaw", "thumb_pitch", "index", "middle", "ring", "little"]):
        s = server.gui.add_slider(n, 0.0, 1.6, step=0.02,
                                    initial_value=float(DEFAULT_HAND[i]))
        hand_sliders.append(s)

    _refresh_lock = {"v": False}   # guard against re-entry during programmatic updates

    def _selected_key():
        s = sel.value
        link = s.rsplit(" [", 1)[0]
        idx = int(s.rsplit(" [", 1)[1].rstrip("]"))
        return (link, idx)

    def _highlight(selected_key):
        for k, h in sphere_handles.items():
            try:
                h.color = _color(k == selected_key)
            except Exception:
                pass

    def _replace_sphere(key, c_local, r):
        link, idx = key
        old = sphere_handles[key]
        try:
            old.remove()
        except Exception:
            pass
        h = server.scene.add_icosphere(
            f"/spheres/{link}_{idx:02d}",
            radius=r,
            position=tuple(_world_pos(link, c_local)),
            color=_color(True),
            opacity=0.45,
        )
        sphere_handles[key] = h

    def _on_dropdown_change(_=None):
        if _refresh_lock["v"]:
            return
        key = _selected_key()
        c, r = current_state[key]
        _refresh_lock["v"] = True
        r_sl.value  = float(r * 1000)
        cx_sl.value = float(c[0] * 1000)
        cy_sl.value = float(c[1] * 1000)
        cz_sl.value = float(c[2] * 1000)
        _refresh_lock["v"] = False
        _highlight(key)
        info_md.content = (f"**{key[0]}** sphere [{key[1]}] — "
                           f"r={r*1000:.1f}mm  c={(c*1000).round(1).tolist()}mm")

    def _on_slider_change(_=None):
        if _refresh_lock["v"]:
            return
        key = _selected_key()
        c = np.array([cx_sl.value, cy_sl.value, cz_sl.value]) / 1000.0
        r = r_sl.value / 1000.0
        current_state[key] = (c, r)
        _replace_sphere(key, c, r)
        info_md.content = (f"**{key[0]}** sphere [{key[1]}] — "
                           f"r={r*1000:.1f}mm  c={(c*1000).round(1).tolist()}mm")

    def _on_hand_change(_=None):
        nonlocal hand_q, link_T
        hand_q = np.array([s.value for s in hand_sliders])
        link_T = {l: _fk_link_T(urdf, arm_q, hand_q, l, ARM_JOINTS,
                                HAND_JOINTS_INSPIRE_LEFT)
                  for l, _ in catalogue}
        # Refresh ALL sphere world positions and robot link visuals.
        for k, (c, r) in current_state.items():
            link, idx = k
            try:
                sphere_handles[k].position = tuple(_world_pos(link, c))
            except Exception:
                _replace_sphere(k, c, r)
        from scipy.spatial.transform import Rotation as R_
        for link_name, (h, _) in link_handles.items():
            if link_name == urdf.base_link:
                continue
            T = _fk_link_T(urdf, arm_q, hand_q, link_name, ARM_JOINTS,
                           HAND_JOINTS_INSPIRE_LEFT)
            try:
                h.position = T[:3, 3]
                h.wxyz = R_.from_matrix(T[:3, :3]).as_quat()[[3, 0, 1, 2]]
            except Exception:
                pass

    def _on_save(_=None):
        # Inject edited values back into the original full yaml structure.
        out = {"collision_spheres": {}}
        for link, slist in spheres.items():
            out["collision_spheres"][link] = []
            for i, s in enumerate(slist):
                key = (link, i)
                if key in current_state:
                    c, r = current_state[key]
                    out["collision_spheres"][link].append(
                        {"center": [float(c[0]), float(c[1]), float(c[2])],
                         "radius": float(r)})
                else:
                    out["collision_spheres"][link].append(
                        {"center": list(s["center"]),
                         "radius": float(s["radius"])})
        _save_yaml(out, out_path)
        n_changed = sum(
            1 for k, (c, r) in current_state.items()
            if (np.linalg.norm(c - np.array(spheres[k[0]][k[1]]["center"])) > 1e-9
                or abs(r - spheres[k[0]][k[1]]["radius"]) > 1e-9)
        )
        msg = f"saved → `{out_path}`  ({n_changed} sphere edits)"
        print(f"[edit] {msg}")
        saved_label.content = msg

    sel.on_update(_on_dropdown_change)
    r_sl.on_update(_on_slider_change)
    cx_sl.on_update(_on_slider_change)
    cy_sl.on_update(_on_slider_change)
    cz_sl.on_update(_on_slider_change)
    for s in hand_sliders:
        s.on_update(_on_hand_change)
    save_btn.on_click(_on_save)

    _on_dropdown_change()
    print(f"[edit] viser  http://localhost:{args.port}")
    try:
        import time
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[edit] bye")


if __name__ == "__main__":
    main()
