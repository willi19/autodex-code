"""What to do next to cover more scenes: where the object is, where to move it.

Answers one question — given the object as it sits right now, what is the next
move and what does it buy? Two kinds of move, cheapest first:

* **reposition** — same resting face, slide/turn on the table.
  ``(x, yaw) -> (x', yaw')``. One place-down, no re-grasp of a new face.
* **reorient** — change which face the object rests on.
  ``tabletop stem -> stem'``. Needs v8 reset candidates in
  ``candidates/{hand}/reset_{h_cm}/{obj}/reorient_{h_cm}/{i}_{j}/``.

This module only COMPUTES and REPORTS. It executes nothing — reorient is a
human-supervised step, so the caller gets the from/to and decides.

Everything is scored on still-UNCOVERED scenes: scenes already covered by an
on-disk success contribute nothing to either option.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from autodex.utils.coverage import (_reorient_cell_solvable, _tabletop_stems,
                                    load_coverage_order, pick_reorient_target,
                                    uncovered_scenes)
from autodex.utils.path import get_obj_root, RESET_RELEASE_HEIGHTS_CM
from autodex.utils.reposition import pick_reposition_target


def current_placement(pose_robot: np.ndarray,
                      tabletop_T: np.ndarray) -> Tuple[float, float, float]:
    """``(x, y, yaw_rad)`` of the object relative to its tabletop pose.

    A tabletop pose fixes the resting FACE; the object is then free in x, y and
    world-z yaw. Recovering yaw is the inverse of the placement convention in
    ``reposition.place_T``: ``R_obj = Rz(yaw) @ R_tab``, so
    ``Rz(yaw) = R_obj @ R_tab.T``.
    """
    R_tab = tabletop_T[:3, :3] if tabletop_T.shape == (4, 4) else tabletop_T
    Rz = pose_robot[:3, :3] @ R_tab.T
    yaw = float(np.arctan2(Rz[1, 0], Rz[0, 0])) % (2 * np.pi)
    return float(pose_robot[0, 3]), float(pose_robot[1, 3]), yaw


def _load_tabletop(obj_name: str, stem: str, obj_root: str) -> np.ndarray:
    import os
    return np.load(os.path.join(obj_root, obj_name, "processed_data",
                                "info", "tabletop", f"{stem}.npy"))


def _z_aligned_geodesic_deg(R_est: np.ndarray, R_tab: np.ndarray) -> float:
    """Tabletop orientation error after factoring out free world-z yaw."""
    M = R_est @ R_tab.T
    theta = np.arctan2(M[1, 0] - M[0, 1], M[0, 0] + M[1, 1])
    c, s = np.cos(theta), np.sin(theta)
    R_z = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    cos = (np.trace(R_est.T @ (R_z @ R_tab)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(float(np.clip(cos, -1.0, 1.0)))))


def classify_stem(obj_name: str, pose_robot: np.ndarray,
                  obj_root: str) -> Optional[Tuple[str, float]]:
    """Closest tabletop stem for ``pose_robot`` plus its residual error (deg)."""
    stems = _tabletop_stems(obj_name, obj_root)
    if not stems:
        return None
    R = pose_robot[:3, :3]
    errs = []
    for s in stems:
        T = _load_tabletop(obj_name, s, obj_root)
        errs.append((_z_aligned_geodesic_deg(
            R, T[:3, :3] if T.shape == (4, 4) else T), s))
    err, stem = min(errs)
    return stem, err


def plan_next_move(
    obj_name: str,
    hand: str,
    version: str,
    pose_robot: np.ndarray,
    *,
    planner=None,
    obj_root: Optional[str] = None,
    h_cm: int = 0,
    top_k: int = 8,
) -> Dict:
    """Compute both moves from the object's current pose. Executes nothing.

    ``pose_robot`` is the object's 4x4 pose in the ROBOT frame (what
    ``pose_world_to_scene_cfg`` derives from perception).

    ``planner`` needs ``ik_pose_batch``; without it the reposition option is
    reported as unavailable rather than guessed at — the reorient option and
    all coverage numbers still work.

    Returns a dict with ``current``, ``uncovered``, ``reposition``,
    ``reorient`` and ``recommend`` (``reposition`` | ``reorient`` | ``done``).
    """
    if version != "v8":
        raise ValueError("plan_next_move supports only the v8 asset contract")
    root = obj_root or get_obj_root(version)
    out: Dict = {"obj": obj_name, "hand": hand, "version": version,
                 "reposition": None, "reorient": None}

    cls = classify_stem(obj_name, pose_robot, root)
    if cls is None:
        out["recommend"] = "done"
        out["error"] = f"no tabletop poses for {obj_name} under {root}"
        return out
    stem, rot_err = cls
    tab_T = _load_tabletop(obj_name, stem, root)
    x, y, yaw = current_placement(pose_robot, tab_T)
    out["current"] = {"stem": stem, "rot_err_deg": round(rot_err, 2),
                      "x": round(x, 4), "y": round(y, 4),
                      "yaw_deg": round(float(np.degrees(yaw)), 1)}

    rem = uncovered_scenes(obj_name, stem, hand=hand, version=version)
    if rem is None:
        out["recommend"] = "done"
        out["error"] = (f"no coverage json for {obj_name}/{version} — run "
                        f"src/dataset/compute_v8_coverage.py first")
        return out
    order = load_coverage_order(obj_name, stem, version=version) or []
    out["uncovered"] = {"n": len(rem), "scenes": sorted(rem),
                        "setcover_len": len(order)}

    # ── option A: reposition (same face) ────────────────────────────────────
    if rem:
        if planner is None:
            out["reposition"] = {"available": False,
                                 "reason": "no planner passed (IK required)"}
        else:
            tgt = pick_reposition_target(
                obj_name, stem, hand, version, planner=planner,
                R_obj_robot=pose_robot[:3, :3], obj_z=float(pose_robot[2, 3]),
                top_k=top_k)
            if tgt is None:
                out["reposition"] = {"available": False,
                                     "reason": "no placement opens uncovered scenes"}
            else:
                d_yaw = (tgt["yaw_deg"] - np.degrees(yaw) + 180.0) % 360.0 - 180.0
                out["reposition"] = {
                    "available": True,
                    "from": {"x": round(x, 4), "yaw_deg": round(float(np.degrees(yaw)), 1)},
                    "to": {"x": tgt["x"], "yaw_deg": round(tgt["yaw_deg"], 1)},
                    "delta": {"x": round(tgt["x"] - x, 4), "yaw_deg": round(float(d_yaw), 1)},
                    "opens": tgt["n_new_scenes"],
                    "scenes": tgt["scenes"],
                    "via_grasps": tgt["grasp_keys"],
                }

    # ── option B: reorient (different face) ─────────────────────────────────
    tgt_j = pick_reorient_target(obj_name, stem, hand=hand, version=version,
                                 h_cm=h_cm, obj_root=root)
    if tgt_j is not None:
        j_int, to_stem, n_rem_target = tgt_j
        n_seeds, n_succ = _reorient_cell_solvable(
            obj_name, hand, int(stem), j_int, version=version)
        out["reorient"] = {
            "available": True,
            "from_stem": stem, "to_stem": to_stem, "to_target_j": j_int,
            "cell": f"{int(stem)}_{j_int}",
            "height_order_cm": list(RESET_RELEASE_HEIGHTS_CM),
            "n_seeds": n_seeds, "n_past_success": n_succ,
            "opens": n_rem_target,
            "command": (f"python src/experiment/reset/reorient.py "
                        f"--obj {obj_name} --hand {hand} --target_j {j_int} "
                        f"--version {version} --auto"),
        }
    else:
        out["reorient"] = {"available": False,
                           "reason": "no reachable tabletop with uncovered scenes"}

    # ── recommendation ──────────────────────────────────────────────────────
    # Reposition first when it opens anything: one place-down beats a reset.
    # Reorient only once this face is exhausted (or nothing here is placeable).
    repos_ok = bool(out["reposition"] and out["reposition"].get("available"))
    reor_ok = bool(out["reorient"] and out["reorient"].get("available"))
    if repos_ok:
        out["recommend"] = "reposition"
    elif reor_ok:
        out["recommend"] = "reorient"
    else:
        out["recommend"] = "done"
    return out


def format_next_move(res: Dict) -> str:
    """Human-readable report of ``plan_next_move``."""
    L = []
    cur = res.get("current")
    if cur is None:
        return f"{res['obj']}: {res.get('error', 'no current pose')}"
    L.append(f"{res['obj']}  [{res['hand']}/{res['version']}]")
    L.append(f"  now      : tabletop {cur['stem']} (fit {cur['rot_err_deg']}deg)  "
             f"x={cur['x']:.3f} y={cur['y']:.3f} yaw={cur['yaw_deg']:.0f}deg")
    if "error" in res:
        L.append(f"  ERROR    : {res['error']}")
        return "\n".join(L)
    unc = res["uncovered"]
    L.append(f"  uncovered: {unc['n']} scenes at this tabletop "
             f"(set cover needs {unc['setcover_len']} grasps)")

    rp = res.get("reposition")
    if rp and rp.get("available"):
        L.append(f"  REPOSITION  x {rp['from']['x']:.3f} -> {rp['to']['x']:.3f} "
                 f"({rp['delta']['x']:+.3f}m),  "
                 f"yaw {rp['from']['yaw_deg']:.0f} -> {rp['to']['yaw_deg']:.0f}deg "
                 f"({rp['delta']['yaw_deg']:+.0f}deg)")
        L.append(f"              opens {rp['opens']} uncovered scenes "
                 f"via {len(rp['via_grasps'])} grasps")
    elif rp:
        L.append(f"  REPOSITION  unavailable — {rp['reason']}")

    ro = res.get("reorient")
    if ro and ro.get("available"):
        L.append(f"  REORIENT    tabletop {ro['from_stem']} -> {ro['to_stem']}  "
                 f"(cell {ro['cell']}; reset heights "
                 f"{ro['height_order_cm']} cm in order, "
                 f"{ro['n_seeds']} seeds, {ro['n_past_success']} past success)")
        L.append(f"              opens {ro['opens']} uncovered scenes at target")
        L.append(f"              {ro['command']}")
    elif ro:
        L.append(f"  REORIENT    unavailable — {ro['reason']}")

    L.append(f"  -> RECOMMEND: {res['recommend'].upper()}")
    return "\n".join(L)
