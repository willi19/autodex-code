"""
Viser viewer for saved reset (reorient) trajectories.

Loads the output of plan_reset.py and plays back the 7-phase trajectory with
robot + object + physical table. Pillars and BODex-canonical table_j from the
scene file are virtual (they encode the i->j reorient task constraints, not
real obstacles); not drawn.

Object pose during lift/rotate/place is interpolated between phase keyframes
(plan_single_js is joint-space; exact FK would require re-running cuRobo).

Usage:
    python src/grasp_generation/reorient/view_reset.py \
        --plan_dir outputs/reset_plans/inspire_left/attached_container/reorient_0/0_16/r0.30_t090/605 \
        --port 8080
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import trimesh
import yourdfpy
from scipy.spatial.transform import Rotation as Rot

sys.path.insert(0, os.path.join(os.path.expanduser("~"), "paradex"))

from paradex.visualization.visualizer.viser import ViserViewer
from autodex.utils.path import obj_path, project_dir, repo_dir


URDF_BY_HAND = {
    "inspire_left": ("inspire_left_description", "xarm_inspire_left.urdf"),
    "inspire":      ("inspire_description",      "xarm_inspire.urdf"),
    "allegro":      ("allegro_description",      "xarm_allegro.urdf"),
    "fr3_inspire":  ("fr3_inspire_description",  "fr3_inspire.urdf"),
}

# cuRobo ee_link per hand — wrist FK target for computing carried-object pose
EE_LINK_BY_HAND = {
    "inspire_left": "base_link",
    "inspire":      "base_link",
    "allegro":      "base_link",
    "fr3_inspire":  "base_link",
}

# Phases where the object is rigidly carried by the wrist
CARRY_PHASES = {"lift", "rotate", "place"}

TABLE_DIMS = [2.0, 3.0, 0.2]
TABLE_POSE_XYZ = [1.1, 0.0, -0.1]


def cart7_to_se3(p):
    T = np.eye(4)
    T[:3, 3] = p[:3]
    T[:3, :3] = Rot.from_quat([p[4], p[5], p[6], p[3]]).as_matrix()
    return T


def fk_ee_per_frame(urdf: yourdfpy.URDF, joint_traj: np.ndarray, ee_link: str) -> np.ndarray:
    """(T, dof) joint trajectory -> (T, 4, 4) ee_link pose in world frame."""
    out = np.tile(np.eye(4), (len(joint_traj), 1, 1))
    for t, q in enumerate(joint_traj):
        urdf.update_cfg(q)
        out[t] = urdf.get_transform(ee_link, urdf.base_link)
    return out


def build_obj_trajectory(phase_name: str, joint_traj: np.ndarray,
                          T_obj_start: np.ndarray, T_obj_end: np.ndarray,
                          urdf: yourdfpy.URDF, ee_link: str,
                          wrist_se3_obj_inv: np.ndarray) -> np.ndarray:
    """Object pose per frame for one phase.

    During carry phases (lift/rotate/place) the object is rigid w.r.t. the
    wrist: T_obj_world(t) = T_ee_world(t) @ inv(wrist_se3_obj). Outside carry
    phases the object is stationary (on table_i or table_j surface).
    """
    n = len(joint_traj)
    if phase_name in ("approach", "grasp_close"):
        return np.tile(T_obj_start[None], (n, 1, 1))
    if phase_name in ("release", "depart", "retract"):
        return np.tile(T_obj_end[None], (n, 1, 1))
    if phase_name in CARRY_PHASES:
        ee_traj = fk_ee_per_frame(urdf, joint_traj, ee_link)
        return ee_traj @ wrist_se3_obj_inv  # (T,4,4) @ (4,4) broadcasts
    raise ValueError(f"unknown phase {phase_name}")


def add_cuboid(vis: ViserViewer, name: str, dims, pose7, color=(0.7, 0.7, 0.75, 0.6)):
    box = trimesh.creation.box(extents=dims)
    T = cart7_to_se3(pose7)
    vis.add_object(name, box, T)
    vis.change_color(name, color)


def resolve_plan_dir_from_sweep(sweep_dir: Path, x: float, tz: float) -> Path:
    """Look up the cell in sweep_summary.json nearest to (x, tz). Returns
    the seed dir for that successful cell (raises if none)."""
    with open(sweep_dir / "sweep_summary.json") as f:
        summary = json.load(f)
    ok_cells = [c for c in summary["cells"] if c["status"] == "ok"]
    if not ok_cells:
        raise RuntimeError(f"No successful cells in {sweep_dir}")
    def dist(c):
        dx = c["x"] - x
        dtz = (c["tz"] - tz) % 360.0
        dtz = min(dtz, 360.0 - dtz)
        return dx * dx + (dtz / 30.0) ** 2
    best = min(ok_cells, key=dist)
    print(f"[view_reset] sweep nearest: requested (x={x:.2f}, tz={tz:.0f}) "
          f"-> cell (x={best['x']:.2f}, tz={best['tz']:.0f}) seed={best['seed_id']}")
    cell_name = f"x{best['x']:.2f}_tz{int(round(best['tz'])):03d}"
    return sweep_dir / cell_name / best["seed_id"]


def _cell_dir(sweep_dir: Path, x: float, tz: float, seed_id: str) -> Path:
    return sweep_dir / f"x{x:.2f}_tz{int(round(tz)):03d}" / seed_id


def preload_sweep_cells(sweep_dir: Path):
    """Read all successful cells once. Returns (xs, tzs, cells_dict).

    cells_dict[(x, tz)] = {trajs, T_obj_start, T_obj_end, wrist_se3_obj_inv,
                            seed_id} or None for failed cells.
    """
    with open(sweep_dir / "sweep_summary.json") as f:
        summary = json.load(f)
    xs = sorted(set(c["x"] for c in summary["cells"]))
    tzs = sorted(set(c["tz"] for c in summary["cells"]))
    cells = {}
    n_ok = 0
    for c in summary["cells"]:
        key = (c["x"], c["tz"])
        if c["status"] != "ok":
            cells[key] = None
            continue
        d = _cell_dir(sweep_dir, c["x"], c["tz"], c["seed_id"])
        traj_npz = np.load(d / "trajectory.npz")
        meta = json.load(open(d / "meta.json"))
        cells[key] = {
            "trajs": {k: traj_npz[k] for k in traj_npz.files},
            "T_obj_start": np.array(meta["T_obj_start"]),
            "T_obj_end": np.array(meta["T_obj_end"]),
            "wrist_se3_obj_inv": np.linalg.inv(np.array(meta["wrist_se3_obj"])),
            "seed_id": c["seed_id"],
            "phase_names": meta["phase_names"],
            "hand": meta["hand"], "obj_name": meta["obj_name"],
        }
        n_ok += 1
    print(f"[view_reset] preloaded {n_ok}/{len(summary['cells'])} cells from {sweep_dir}")
    return xs, tzs, cells, summary


def discover_pair_dirs(obj: str, h_cm: int, hand: str) -> list:
    """List all pair subdirs containing sweep_summary.json."""
    obj_root = (Path(repo_dir) / "outputs" / "reset_cache" / hand / obj
                / f"reorient_{h_cm}")
    if not obj_root.exists():
        return []
    return sorted(
        [d for d in obj_root.iterdir()
         if d.is_dir() and (d / "sweep_summary.json").exists()],
        key=lambda d: tuple(int(x) for x in d.name.split("_")),
    )


def run_multi_explorer(vis: ViserViewer, urdf_fk: yourdfpy.URDF, ee_link: str,
                        pair_dirs: list):
    """Pair dropdown + (x_idx, tz_idx) sliders. Switching pair reloads cells."""
    # Preload all pairs (cheap — np.load per cell file already done lazily)
    pair_caches = {}  # pair_name -> (xs, tzs, cells, summary)
    for pd in pair_dirs:
        try:
            xs, tzs, cells, summary = preload_sweep_cells(pd)
            pair_caches[pd.name] = (xs, tzs, cells, summary)
        except Exception as e:
            print(f"[view_reset] skip {pd.name}: {e}")

    if not pair_caches:
        raise RuntimeError("no successful pairs to explore")

    pair_names = list(pair_caches.keys())
    pair_dropdown = vis.server.gui.add_dropdown(
        "(i, j) pair", options=tuple(pair_names), initial_value=pair_names[0],
    )
    x_slider = vis.server.gui.add_slider("pickup x idx", min=0, max=1, step=1, initial_value=0)
    tz_slider = vis.server.gui.add_slider("pickup tz idx", min=0, max=1, step=1, initial_value=0)
    cell_info = vis.server.gui.add_text("Cell", initial_value="", disabled=True)
    status_info = vis.server.gui.add_text("Status", initial_value="", disabled=True)

    state = {"xs": None, "tzs": None, "cells": None}

    def load_current_cell():
        if state["xs"] is None:
            return  # state not ready yet (slider callback during init)
        xs, tzs, cells = state["xs"], state["tzs"], state["cells"]
        if not xs or not tzs:
            status_info.value = "empty grid"; vis.clear_traj(); return
        xi = min(x_slider.value, len(xs) - 1)
        tzi = min(tz_slider.value, len(tzs) - 1)
        x, tz = xs[xi], tzs[tzi]
        cell_info.value = f"{pair_dropdown.value}  x={x:.2f}  tz={tz:.0f}°"
        cell = cells.get((x, tz))
        if cell is None:
            status_info.value = "FAIL — no trajectory"
            vis.clear_traj(); return
        status_info.value = f"ok  seed={cell['seed_id']}"
        vis.clear_traj()
        for ph in cell["phase_names"]:
            robot_traj = cell["trajs"][ph]
            obj_traj = build_obj_trajectory(
                ph, robot_traj, cell["T_obj_start"], cell["T_obj_end"],
                urdf_fk, ee_link, cell["wrist_se3_obj_inv"],
            )
            vis.add_traj(ph, {"robot": robot_traj}, {"object": obj_traj})

    def reload_pair():
        """Update state with current pair's xs/tzs/cells, then resize sliders.
        State must be assigned BEFORE mutating slider.max / .value so that any
        on_update callbacks triggered see consistent state."""
        name = pair_dropdown.value
        xs, tzs, cells, _summary = pair_caches[name]
        state["xs"], state["tzs"], state["cells"] = xs, tzs, cells
        x_slider.max = max(len(xs) - 1, 0); x_slider.value = 0
        tz_slider.max = max(len(tzs) - 1, 0); tz_slider.value = 0

    @pair_dropdown.on_update
    def _(_):
        reload_pair()
        load_current_cell()

    @x_slider.on_update
    def _(_): load_current_cell()
    @tz_slider.on_update
    def _(_): load_current_cell()

    reload_pair()
    load_current_cell()


def run_explorer(vis: ViserViewer, urdf_fk: yourdfpy.URDF, ee_link: str,
                  xs, tzs, cells):
    """Add x/tz sliders + load the corresponding cell trajectory on change."""
    x_slider = vis.server.gui.add_slider(
        "pickup x idx", min=0, max=max(len(xs) - 1, 0), step=1, initial_value=0,
    )
    tz_slider = vis.server.gui.add_slider(
        "pickup tz idx", min=0, max=max(len(tzs) - 1, 0), step=1, initial_value=0,
    )
    info = vis.server.gui.add_text("Cell", initial_value="", disabled=True)
    fail_warn = vis.server.gui.add_text("Status", initial_value="", disabled=True)

    def load_current():
        xi = min(x_slider.value, len(xs) - 1)
        tzi = min(tz_slider.value, len(tzs) - 1)
        x, tz = xs[xi], tzs[tzi]
        cell = cells.get((x, tz))
        info.value = f"x={x:.2f}  tz={tz:.0f}°"
        if cell is None:
            fail_warn.value = "FAIL — no trajectory"
            vis.clear_traj()
            return
        fail_warn.value = f"ok  seed={cell['seed_id']}"
        vis.clear_traj()
        for ph in cell["phase_names"]:
            robot_traj = cell["trajs"][ph]
            obj_traj = build_obj_trajectory(
                ph, robot_traj, cell["T_obj_start"], cell["T_obj_end"],
                urdf_fk, ee_link, cell["wrist_se3_obj_inv"],
            )
            vis.add_traj(ph, {"robot": robot_traj}, {"object": obj_traj})

    @x_slider.on_update
    def _(_): load_current()

    @tz_slider.on_update
    def _(_): load_current()

    load_current()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan_dir", default=None, help="Single trajectory dir (trajectory.npz + meta.json)")
    parser.add_argument("--sweep_dir", default=None, help="Sweep pair dir (one (i,j))")
    parser.add_argument("--obj", default=None, help="Object name (multi-pair explorer)")
    parser.add_argument("--h_cm", type=int, default=None, help="lift height (multi-pair)")
    parser.add_argument("--hand", default="inspire_left")
    parser.add_argument("--x", type=float, default=None, help="pickup x (single-cell lookup)")
    parser.add_argument("--tz", type=float, default=None, help="pickup theta_z deg")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()

    # Mode selection:
    #   --plan_dir            : single trajectory replay
    #   --sweep_dir + x + tz  : nearest-cell lookup
    #   --sweep_dir           : explorer over that pair (sliders)
    #   --obj + --h_cm        : multi-pair explorer (dropdown + sliders)
    multi_pair = False
    explorer = False
    plan_dir = None

    if args.plan_dir is not None:
        plan_dir = Path(args.plan_dir)
    elif args.sweep_dir is not None:
        if args.x is not None and args.tz is not None:
            plan_dir = resolve_plan_dir_from_sweep(Path(args.sweep_dir), args.x, args.tz)
        elif args.x is None and args.tz is None:
            explorer = True
        else:
            parser.error("--x and --tz must be given together")
    elif args.obj is not None and args.h_cm is not None:
        multi_pair = True
    else:
        parser.error("specify one of: --plan_dir | --sweep_dir | (--obj + --h_cm)")

    if multi_pair:
        pair_dirs = discover_pair_dirs(args.obj, args.h_cm, args.hand)
        if not pair_dirs:
            raise RuntimeError(
                f"No sweep results under outputs/reset_cache/{args.hand}/{args.obj}"
                f"/reorient_{args.h_cm}. Run sweep_reset.py first."
            )
        # Use a successful cell for sample metadata if any exists; otherwise
        # fall back to canonical Ti for object placement and identity for grasp.
        sample = None
        sample_summary = None
        for pd in pair_dirs:
            try:
                _xs, _tzs, _cells, _summary = preload_sweep_cells(pd)
            except Exception as e:
                print(f"[view_reset] skip {pd.name}: {e}")
                continue
            ok = next((c for c in _cells.values() if c is not None), None)
            if ok is not None and sample is None:
                sample, sample_summary = ok, _summary
                break

        if sample is not None:
            meta = {
                "hand": sample["hand"], "obj_name": sample["obj_name"],
                "i": sample_summary["i"], "j": sample_summary["j"],
                "h_cm": sample_summary["h_cm"],
                "pickup_x": 0.0, "pickup_tz": 0.0,
                "seed_id": "(multi)", "phase_names": sample["phase_names"],
                "T_obj_start": sample["T_obj_start"].tolist(),
                "T_obj_end": sample["T_obj_end"].tolist(),
                "wrist_se3_obj": np.linalg.inv(sample["wrist_se3_obj_inv"]).tolist(),
            }
        else:
            # No successful cell anywhere — still open the viewer to inspect
            # the fail patterns. Default object at canonical Ti from first pair.
            from autodex.utils.path import obj_path as _obj_path
            first_summary = json.load(open(pair_dirs[0] / "sweep_summary.json"))
            i0 = first_summary["i"]
            Ti = np.load(Path(_obj_path) / args.obj / "processed_data"
                         / "info" / "tabletop" / f"{i0:03d}.npy")
            meta = {
                "hand": args.hand, "obj_name": args.obj,
                "i": i0, "j": first_summary["j"], "h_cm": args.h_cm,
                "pickup_x": 0.0, "pickup_tz": 0.0,
                "seed_id": "(no-success)", "phase_names": [],
                "T_obj_start": Ti.tolist(),
                "T_obj_end": Ti.tolist(),
                "wrist_se3_obj": np.eye(4).tolist(),
            }
            print(f"[view_reset] all {len(pair_dirs)} pairs are 0/N — "
                  f"opening viewer in fail-inspection mode")
        traj = None
    elif explorer:
        sweep_dir = Path(args.sweep_dir)
        xs, tzs, cells, summary = preload_sweep_cells(sweep_dir)
        sample = next((c for c in cells.values() if c is not None), None)
        if sample is None:
            raise RuntimeError(f"No successful cells in {sweep_dir} to preview")
        meta = {
            "hand": sample["hand"], "obj_name": sample["obj_name"],
            "i": summary["i"], "j": summary["j"], "h_cm": summary["h_cm"],
            "pickup_x": 0.0, "pickup_tz": 0.0,
            "seed_id": "(explorer)", "phase_names": sample["phase_names"],
            "T_obj_start": sample["T_obj_start"].tolist(),
            "T_obj_end": sample["T_obj_end"].tolist(),
            "wrist_se3_obj": np.linalg.inv(sample["wrist_se3_obj_inv"]).tolist(),
        }
        traj = None
    else:
        with open(plan_dir / "meta.json") as f:
            meta = json.load(f)
        traj = np.load(plan_dir / "trajectory.npz")

    hand = meta["hand"]
    obj_name = meta["obj_name"]
    h_cm, i, j = meta["h_cm"], meta["i"], meta["j"]
    pickup_x = meta.get("pickup_x", meta.get("r", 0))
    pickup_tz = meta.get("pickup_tz", meta.get("theta_deg", 0))
    phase_names = meta["phase_names"]

    print(f"[view_reset] {obj_name} reorient_{h_cm} {i}->{j} "
          f"pickup=(x={pickup_x:.2f}, tz={pickup_tz:.0f}°)")
    print(f"[view_reset] seed={meta['seed_id']}  hand={hand}  phases={phase_names}")

    T_obj_start = np.array(meta["T_obj_start"])
    T_obj_end = np.array(meta["T_obj_end"])
    wrist_se3_obj = np.array(meta["wrist_se3_obj"])
    wrist_se3_obj_inv = np.linalg.inv(wrist_se3_obj)

    vis = ViserViewer(port_number=args.port)

    # Robot
    urdf_dir, urdf_name = URDF_BY_HAND[hand]
    urdf_full = os.path.join(project_dir, "content", "assets", "robot", urdf_dir, urdf_name)
    vis.add_robot("robot", urdf_full)

    # Separate yourdfpy URDF for FK (independent of viser's internal robot state)
    urdf_fk = yourdfpy.URDF.load(urdf_full, build_scene_graph=True)
    ee_link = EE_LINK_BY_HAND[hand]

    # Object mesh (use raw_mesh with texture if available)
    raw = Path(obj_path) / obj_name / "raw_mesh" / f"{obj_name}.obj"
    simp = Path(obj_path) / obj_name / "processed_data" / "mesh" / "simplified.obj"
    mesh_path = raw if raw.exists() else simp
    obj_mesh = trimesh.load(mesh_path, force="mesh", process=False)
    try:
        obj_mesh.visual = obj_mesh.visual.to_color()
    except Exception:
        pass
    vis.add_object("object", obj_mesh, T_obj_start)

    # Physical robot table (same as planning collision world)
    add_cuboid(vis, "cube/table",
               TABLE_DIMS, [*TABLE_POSE_XYZ, 1.0, 0.0, 0.0, 0.0],
               color=(0.85, 0.85, 0.90, 0.7))

    if multi_pair:
        run_multi_explorer(vis, urdf_fk, ee_link, pair_dirs)
    elif explorer:
        run_explorer(vis, urdf_fk, ee_link, xs, tzs, cells)
    else:
        # Build per-phase trajectories and add to viewer
        for ph in phase_names:
            robot_traj = traj[ph]  # (T, dof)
            obj_traj = build_obj_trajectory(
                ph, robot_traj, T_obj_start, T_obj_end,
                urdf_fk, ee_link, wrist_se3_obj_inv,
            )
            vis.add_traj(ph, {"robot": robot_traj}, {"object": obj_traj})

    print(f"[view_reset] serving on port {args.port}")
    vis.start_viewer()


if __name__ == "__main__":
    main()
