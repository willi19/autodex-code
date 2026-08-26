"""Where to place the object so the remaining UNCOVERED scenes get covered.

After a pick the runner holds the object and must put it down somewhere. That
placement is free, and a bad one wastes trials: the grasps that would cover the
scenes still missing may be IK-infeasible wherever the object happens to land.

This module picks that placement. It keeps the tabletop CLASS fixed — changing
which face the object rests on is a reorient, handled by
``coverage.pick_reorient_target`` plus a physical reset run — and searches only
what a place-down controls: ``x`` along the robot's +x and ``yaw`` about world z.

Scoring is **strictly uncovered-first**. A placement's score is the size of the
UNION of still-uncovered scenes that the grasps reachable there would cover.
Already-covered scenes contribute nothing, and grasps whose whole cover set is
already covered are dropped before ranking. Union, not sum: two grasps covering
the same three missing scenes are worth three, not six.

That is the difference from ``src/visualization/exp.py:rank_placements``, whose
``restrict`` argument exists but is unused at its call site — it ranks by raw
count of reachable grasps, so a placement serving many redundant grasps beats
one serving the single grasp that unlocks a missing scene. Same grids, same
placement convention, different objective.

Two entry points:

* ``rank_placements_by_coverage`` — array core. Takes exp.py's cached
  reachability map ``reach[g, xi, yi]`` and a coverage matrix; no IK, no I/O.
* ``pick_reposition_target`` — runner-facing. Reads the coverage JSON, resolves
  wrist poses from the candidate tree, and batch-IKs live when no cached
  reachability map is available.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from autodex.utils.coverage import load_coverage_entries
from autodex.utils.path import get_candidate_path

# Placement grids. MUST match src/visualization/exp.py (which in turn matches
# src/visualization/seed_cache) — a cached reach[g, xi, yi] map is indexed by
# these, so a mismatch silently misreads the map.
X_GRID = np.arange(0.30, 0.71, 0.05)
YAW_GRID = np.linspace(0, 2 * np.pi, 36, endpoint=False)

# Table top surface in robot z (TABLE_CUBOID pose_z + dims_z/2), same value the
# executor snaps placements to.
TABLE_SURFACE_Z = 0.039

X_PREFERRED_DEFAULT = 0.50

# Reorienting the held object to a non-zero yaw during a reset is unreliable, so
# reset placements stick to yaw=0 (exp.py's default). A same-pose reposition
# never re-grasps, so it may use the full yaw grid.
YAW_IDXS_RESET = (0,)

# How many still-useful grasps to test when we have to IK live. Cost is
# len(x_grid) * len(yaw_idxs) * top_k queries in one batch.
TOP_K_DEFAULT = 8


def place_T(pose_T: np.ndarray, x: float, yaw: float,
            table_surface_z: float = TABLE_SURFACE_Z) -> np.ndarray:
    """Object world pose: tabletop orientation ``pose_T`` yawed about world z,
    resting at ``(x, 0, table)``. Mirrors ``exp.py:_place_T`` exactly."""
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    T = np.asarray(pose_T, dtype=float).copy()
    T[:3, :3] = Rz @ np.asarray(pose_T, dtype=float)[:3, :3]
    T[0, 3], T[1, 3] = float(x), 0.0
    T[2, 3] = float(pose_T[2, 3]) + table_surface_z
    return T


def rank_placements_by_coverage(
    reach: np.ndarray,
    cand: Sequence[int],
    covers: Dict[int, set],
    covered: Sequence[bool],
    *,
    yaw_idxs: Optional[Sequence[int]] = None,
    x_grid: Optional[Sequence[float]] = None,
    yaw_grid: Optional[Sequence[float]] = None,
    x_preferred: float = X_PREFERRED_DEFAULT,
) -> List[dict]:
    """Rank placements by NEW scene coverage, ignoring already-covered scenes.

    reach   : (G, len(x_grid), len(yaw_grid)) bool — grasp g IK-feasible there.
    cand    : grasp indices eligible at the current tabletop pose.
    covers  : grasp index -> set of scene indices it covers.
    covered : bool per scene; True = already covered, contributes nothing.

    Grasps with no uncovered scene left are dropped BEFORE ranking, so a
    placement can never score on redundant reachability.

    Returns entries sorted by (new-scene count desc, |x - x_preferred|, yaw),
    dropping placements that open nothing. Empty list means: no placement at
    this tabletop helps — reorient instead.
    """
    xs = np.asarray(X_GRID if x_grid is None else x_grid, dtype=float)
    yaws = np.asarray(YAW_GRID if yaw_grid is None else yaw_grid, dtype=float)
    yidx = list(YAW_IDXS_RESET if yaw_idxs is None else yaw_idxs)
    covered = np.asarray(covered, dtype=bool)

    # Uncovered-only remaining set per candidate; drop the exhausted ones.
    remaining: Dict[int, set] = {}
    for g in cand:
        rem = {s for s in covers.get(int(g), ()) if not covered[s]}
        if rem:
            remaining[int(g)] = rem
    if not remaining:
        return []

    useful = np.array(sorted(remaining), dtype=int)
    out: List[dict] = []
    for xi in range(len(xs)):
        for yi in yidx:
            ok = useful[reach[useful, xi, yi]]
            if len(ok) == 0:
                continue
            scenes: set = set()
            for g in ok:
                scenes |= remaining[int(g)]
            if not scenes:
                continue
            out.append({
                "x": float(xs[xi]), "yaw_rad": float(yaws[yi]),
                "yaw_deg": float(np.degrees(yaws[yi])),
                "xi": int(xi), "yi": int(yi),
                "grasps": [int(g) for g in ok],
                "n_new_scenes": len(scenes),
                "scenes": sorted(scenes),
            })
    out.sort(key=lambda e: (-e["n_new_scenes"],
                            abs(e["x"] - x_preferred), e["yaw_rad"]))
    return out


def _wrist_path(hand: str, version: str, obj_name: str,
                key: Tuple[str, str, str]) -> str:
    return os.path.join(get_candidate_path(hand), version, obj_name,
                        key[0], key[1], key[2], "wrist_se3.npy")


def load_useful_grasps(
    obj_name: str,
    tabletop_pose_stem: str,
    hand: str,
    version: str,
    top_k: int = TOP_K_DEFAULT,
) -> List[dict]:
    """Grasps at this tabletop that still cover something, richest first.

    Each entry: ``{"key", "remaining": set[int], "wrist_obj": (4,4)}``.
    ``load_coverage_entries`` has already subtracted every scene covered by an
    on-disk success and dropped the grasps left with nothing, so anything
    returned here is uncovered-relevant by construction.
    """
    entries = load_coverage_entries(
        obj_name, tabletop_pose_stem=tabletop_pose_stem,
        hand=hand, version=version)
    if not entries:
        return []
    out: List[dict] = []
    for e in entries:
        p = _wrist_path(hand, version, obj_name, e["key"])
        if not os.path.exists(p):
            continue
        out.append({**e, "wrist_obj": np.load(p)})
        if len(out) >= top_k:
            break
    return out


def pick_reposition_target(
    obj_name: str,
    tabletop_pose_stem: str,
    hand: str,
    version: str,
    *,
    planner,
    R_obj_robot: np.ndarray,
    obj_z: float,
    x_grid: Optional[Sequence[float]] = None,
    yaw_grid: Optional[Sequence[float]] = None,
    yaw_idxs: Optional[Sequence[int]] = None,
    x_preferred: float = X_PREFERRED_DEFAULT,
    top_k: int = TOP_K_DEFAULT,
    grasps: Optional[List[dict]] = None,
) -> Optional[dict]:
    """Pick ``(x, yaw)`` to place the object at, maximizing NEW coverage.

    ``R_obj_robot`` is the object's CURRENT tabletop rotation in the robot
    frame — pass the perception-time rotation, not one read back after a lift
    (the lift drifts wrist orientation a few degrees, which would tilt the
    target). ``obj_z`` holds the object's height so the place-down neither dips
    into the table nor floats.

    ``planner`` needs ``ik_pose_batch(T: (N,4,4)) -> (N,) bool``. IK runs live
    here (one batch); the array core above is the path for a cached reach map.

    ``yaw_idxs`` defaults to the FULL yaw grid: this is a same-pose reposition,
    not a reset, so the object is re-grasped and any yaw is placeable. Pass
    ``YAW_IDXS_RESET`` when the placement has to survive a reorient.

    Returns ``None`` when nothing is left to cover at this tabletop, or when no
    placement makes a still-useful grasp reachable — the runner's signal to
    fall back to ``pick_reorient_target``. Otherwise::

        {"x", "yaw_rad", "yaw_deg", "n_new_scenes", "scenes", "grasp_keys",
         "n_feasible_grasps", "n_combos", "n_grasps_tested"}
    """
    xs = np.asarray(X_GRID if x_grid is None else x_grid, dtype=float)
    yaws = np.asarray(YAW_GRID if yaw_grid is None else yaw_grid, dtype=float)
    yidx = list(range(len(yaws)) if yaw_idxs is None else yaw_idxs)

    if grasps is None:
        grasps = load_useful_grasps(obj_name, tabletop_pose_stem, hand,
                                    version, top_k=top_k)
    if not grasps:
        return None

    combos = [(xi, yi) for xi in range(len(xs)) for yi in yidx]
    n_c, n_g = len(combos), len(grasps)

    # Wrist target per (placement, grasp), combo-major so the reshape is exact.
    targets = np.zeros((n_c * n_g, 4, 4), dtype=float)
    for ci, (xi, yi) in enumerate(combos):
        yaw = float(yaws[yi])
        c, s = np.cos(yaw), np.sin(yaw)
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        T_obj = np.eye(4)
        T_obj[:3, :3] = Rz @ np.asarray(R_obj_robot, dtype=float)
        T_obj[:3, 3] = [float(xs[xi]), 0.0, float(obj_z)]
        for gi, g in enumerate(grasps):
            targets[ci * n_g + gi] = T_obj @ g["wrist_obj"]

    feasible = np.asarray(planner.ik_pose_batch(targets), dtype=bool)
    feasible = feasible.reshape(n_c, n_g)

    # Reuse the array core's objective so both paths rank identically: build a
    # reach map over just these grasps and treat every listed scene as still
    # uncovered (load_useful_grasps already removed the covered ones).
    reach = np.zeros((n_g, len(xs), len(yaws)), dtype=bool)
    for ci, (xi, yi) in enumerate(combos):
        reach[:, xi, yi] = feasible[ci]
    scene_ids = sorted({s for g in grasps for s in g["remaining"]})
    sid_of = {s: i for i, s in enumerate(scene_ids)}
    covers = {gi: {sid_of[s] for s in g["remaining"]}
              for gi, g in enumerate(grasps)}

    ranked = rank_placements_by_coverage(
        reach, range(n_g), covers, np.zeros(len(scene_ids), dtype=bool),
        yaw_idxs=yidx, x_grid=xs, yaw_grid=yaws, x_preferred=x_preferred)
    if not ranked:
        return None

    best = ranked[0]
    return {
        "x": best["x"],
        "yaw_rad": best["yaw_rad"],
        "yaw_deg": best["yaw_deg"],
        "n_new_scenes": best["n_new_scenes"],
        "scenes": [scene_ids[i] for i in best["scenes"]],
        "grasp_keys": [grasps[gi]["key"] for gi in best["grasps"]],
        "n_feasible_grasps": len(best["grasps"]),
        "n_combos": n_c,
        "n_grasps_tested": n_g,
    }


def describe(target: Optional[Dict]) -> str:
    """One-line log form of a ``pick_reposition_target`` result."""
    if target is None:
        return "no placement opens uncovered scenes — reorient instead"
    return (f"x={target['x']:.2f} yaw={target['yaw_deg']:.0f}deg "
            f"→ {target['n_new_scenes']} uncovered scenes via "
            f"{target['n_feasible_grasps']}/{target['n_grasps_tested']} grasps")
