"""v7 coverage-based grasp ordering.

Reads precomputed coverage JSON (one entry per candidate grasp listing the
scenes it covers) and runs greedy set cover to produce an ordering of grasps
that covers all reachable scenes with as few candidates as possible.

Coverage JSONs live at::

    {project_dir}/experiment/v7/coverage/cov_v7_cand_{obj}.json

Each entry of ``d["grasps"]`` looks like::

    {"type": "wall", "sid": 3, "gid": 17, "pose_idx": "000",
     "covers": [0, 4, 9, ...]}
"""
import json
import os
from typing import Dict, List, Optional, Set, Tuple

from autodex.utils.path import get_candidate_path, obj_path, project_dir


def _coverage_path(obj_name: str, version: str = "v7") -> str:
    return os.path.join(
        project_dir, "experiment", version, "coverage",
        f"cov_{version}_cand_{obj_name}.json"
    )


def load_v7_coverage_order(
    obj_name: str,
    tabletop_pose_stem: Optional[str] = None,
) -> Optional[List[Tuple[str, str, str]]]:
    """Greedy set cover over the candidate grasps in the v7 coverage JSON.

    If ``tabletop_pose_stem`` is given, only candidates whose ``pose_idx``
    matches are considered.

    Returns a list of ``(scene_type, scene_id_str, grasp_id_str)`` tuples in
    selection order. Returns ``None`` if the coverage file is missing.
    """
    path = _coverage_path(obj_name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    grasps = data.get("grasps") or []

    if tabletop_pose_stem is not None:
        grasps = [g for g in grasps
                  if str(g.get("pose_idx", "")) == str(tabletop_pose_stem)]

    if not grasps:
        return []

    covered: set = set()
    order: List[Tuple[str, str, str]] = []
    remaining = list(range(len(grasps)))
    while remaining:
        best = max(remaining,
                   key=lambda i: len(set(grasps[i]["covers"]) - covered))
        gain = len(set(grasps[best]["covers"]) - covered)
        if gain == 0:
            break
        covered |= set(grasps[best]["covers"])
        g = grasps[best]
        order.append((str(g["type"]), str(g["sid"]), str(g["gid"])))
        remaining.remove(best)
    return order


def load_v7_coverage_map(
    obj_name: str,
    tabletop_pose_stem: Optional[str] = None,
    hand: str = "inspire_left",
    version: str = "v7",
) -> Optional[dict]:
    """Return ``dict[(type, sid_str, gid_str) -> n_remaining_uncovered]``
    for every grasp in the v7 coverage json (optionally filtered to a
    single tabletop pose stem). Used as a priority map for sort-by-
    coverage after IK.

    The count is **remaining-uncovered scenes**, not total covers — scenes
    already covered by an on-disk successful grasp are subtracted. Without
    this, the same high-cover candidate ranks first every trial regardless
    of progress, and we keep retrying the same scene.

    Returns ``None`` if coverage file missing.
    """
    path = _coverage_path(obj_name)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    grasps = data.get("grasps") or []
    if tabletop_pose_stem is not None:
        grasps = [g for g in grasps
                  if str(g.get("pose_idx", "")) == str(tabletop_pose_stem)]
    success_keys = _disk_success_keys(obj_name, hand, version)
    # Build set of scenes already covered by any successful grasp.
    covered_scenes: Set[int] = set()
    # NOTE: use the unfiltered grasp list so cross-tabletop successes also
    # count — once a scene is covered, it's covered.
    for g in (data.get("grasps") or []):
        key = (str(g["type"]), str(g["sid"]), str(g["gid"]))
        if key in success_keys:
            covered_scenes.update(g.get("covers", []))
    return {
        (str(g["type"]), str(g["sid"]), str(g["gid"])):
            len(set(g.get("covers", [])) - covered_scenes)
        for g in grasps
    }


def _disk_success_keys(obj_name: str, hand: str,
                       version: str = "v7") -> Set[Tuple[str, str, str]]:
    """Walk the candidate dir tree once and collect
    ``(scene_type, scene_id_dir, grasp_id_dir)`` keys whose ``result.json``
    has ``success=True``. Used to compute already-covered scenes."""
    base = os.path.join(get_candidate_path(hand), version, obj_name)
    if not os.path.isdir(base):
        return set()
    out: Set[Tuple[str, str, str]] = set()
    for dirpath, dirnames, filenames in os.walk(base):
        if "result.json" not in filenames:
            continue
        try:
            with open(os.path.join(dirpath, "result.json")) as f:
                if not json.load(f).get("success", False):
                    continue
        except Exception:
            continue
        rel = os.path.relpath(dirpath, base).split(os.sep)
        if len(rel) == 3:
            out.add((rel[0], rel[1], rel[2]))
        elif len(rel) == 2:
            out.add(("", rel[0], rel[1]))
    return out


def _grasps_at_tabletop(grasps: List[dict], stem: str) -> List[dict]:
    return [g for g in grasps if str(g.get("pose_idx", "")) == str(stem)]


def uncovered_scenes(obj_name: str, tabletop_pose_stem: str,
                     hand: str = "inspire_left",
                     version: str = "v7") -> Optional[Set[int]]:
    """Scene indices at ``tabletop_pose_stem`` not yet covered by any
    on-disk successful grasp.

    Returns ``None`` if the coverage file is missing.
    """
    path = _coverage_path(obj_name, version)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    grasps = data.get("grasps") or []
    tt_grasps = _grasps_at_tabletop(grasps, tabletop_pose_stem)
    all_scenes: Set[int] = set()
    for g in tt_grasps:
        all_scenes.update(g.get("covers", []))
    success_keys = _disk_success_keys(obj_name, hand, version)
    covered: Set[int] = set()
    for g in tt_grasps:
        key = (str(g["type"]), str(g["sid"]), str(g["gid"]))
        if key in success_keys:
            covered.update(g.get("covers", []))
    return all_scenes - covered


def _tabletop_stems(obj_name: str) -> List[str]:
    """Sorted tabletop filename stems for ``obj_name`` (e.g. ``['000','009','016']``)."""
    tt_dir = os.path.join(obj_path, obj_name, "processed_data", "info", "tabletop")
    if not os.path.isdir(tt_dir):
        return []
    return sorted(f[:-4] for f in os.listdir(tt_dir) if f.endswith(".npy"))


def next_grasp_after_success(
    obj_name: str,
    current_grasp_key: Tuple[str, str, str],
    tabletop_pose_stem: Optional[str] = None,
    hand: str = "inspire_left",
    version: str = "v7",
) -> Optional[Tuple[str, str, str]]:
    """Return the ``(type, sid, gid)`` of the next grasp the greedy set-cover
    will pick *after* ``current_grasp_key`` succeeds.

    Computed by: starting from the union of (on-disk successes ∪
    current_grasp_key.covers) as the "already covered" set, scan remaining
    grasps at the same tabletop, return the one with max new-cover gain.
    Returns ``None`` if no remaining grasp adds new coverage.
    """
    path = _coverage_path(obj_name, version)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    grasps = data.get("grasps") or []
    if tabletop_pose_stem is not None:
        grasps = _grasps_at_tabletop(grasps, tabletop_pose_stem)

    cur_key = (str(current_grasp_key[0]), str(current_grasp_key[1]),
               str(current_grasp_key[2]))
    cur_covers: Set[int] = set()
    for g in grasps:
        key = (str(g["type"]), str(g["sid"]), str(g["gid"]))
        if key == cur_key:
            cur_covers = set(g.get("covers", []))
            break

    disk_success = _disk_success_keys(obj_name, hand, version)
    covered: Set[int] = set(cur_covers)
    for g in grasps:
        key = (str(g["type"]), str(g["sid"]), str(g["gid"]))
        if key in disk_success:
            covered.update(g.get("covers", []))

    best_key: Optional[Tuple[str, str, str]] = None
    best_gain = 0
    for g in grasps:
        key = (str(g["type"]), str(g["sid"]), str(g["gid"]))
        if key == cur_key or key in disk_success:
            continue
        gain = len(set(g.get("covers", [])) - covered)
        if gain > best_gain:
            best_gain = gain
            best_key = key
    return best_key


def _grasp_dir(obj_name: str, scene_type: str, scene_id: str, grasp_id: str,
                hand: str = "inspire_left", version: str = "table_only") -> str:
    base = os.path.join(get_candidate_path(hand), version, obj_name)
    if scene_type:
        return os.path.join(base, scene_type, scene_id, grasp_id)
    return os.path.join(base, scene_id, grasp_id)


def read_grasp_stats(grasp_dir: str) -> Tuple[int, int]:
    """Return ``(attempts, successes)`` from ``stats.json`` in the grasp dir.
    Missing file → ``(0, 0)``."""
    p = os.path.join(grasp_dir, "stats.json")
    if not os.path.exists(p):
        return 0, 0
    try:
        with open(p) as f:
            d = json.load(f)
        return int(d.get("attempts", 0)), int(d.get("successes", 0))
    except Exception:
        return 0, 0


def update_grasp_stats(grasp_dir: str, success: bool) -> Tuple[int, int]:
    """Read-modify-write ``stats.json`` with one new attempt result.
    Returns new ``(attempts, successes)``."""
    attempts, successes = read_grasp_stats(grasp_dir)
    attempts += 1
    if success:
        successes += 1
    os.makedirs(grasp_dir, exist_ok=True)
    with open(os.path.join(grasp_dir, "stats.json"), "w") as f:
        json.dump({"attempts": attempts, "successes": successes}, f)
    return attempts, successes


def grasp_priority_score(attempts: int, successes: int) -> float:
    """Laplace-smoothed success rate. Untried = 0.5, 1/1 success = 2/3,
    0/1 fail = 1/3. Higher = pick earlier."""
    return (successes + 1.0) / (attempts + 2.0)


def table_only_grasp_order_by_stats(
    obj_name: str, hand: str = "inspire_left",
    version: str = "table_only"
) -> List[Tuple[str, str, str]]:
    """Walk ``candidates/{hand}/{version}/{obj}/`` for grasp dirs and return
    ``(scene_type, scene_id, grasp_id)`` tuples sorted by stats priority
    descending. Stable order within tied priority.
    """
    base = os.path.join(get_candidate_path(hand), version, obj_name)
    if not os.path.isdir(base):
        return []
    keys: List[Tuple[str, str, str]] = []
    for dirpath, dirnames, filenames in os.walk(base):
        if "wrist_se3.npy" not in filenames:
            continue
        dirnames[:] = []
        rel = os.path.relpath(dirpath, base).split(os.sep)
        if len(rel) == 3:
            keys.append((rel[0], rel[1], rel[2]))
        elif len(rel) == 2:
            keys.append(("", rel[0], rel[1]))
    scored = []
    for k in keys:
        gd = _grasp_dir(obj_name, k[0], k[1], k[2], hand, version)
        a, s = read_grasp_stats(gd)
        scored.append((-grasp_priority_score(a, s), k))
    scored.sort(key=lambda x: x[0])
    return [k for _, k in scored]


def _reorient_cell_solvable(obj_name: str, hand: str,
                             current_int: int, target_int: int,
                             h_cm: int = 0) -> Tuple[int, int]:
    """Inspect ``candidates/{hand}/reset/{obj}/reorient_{h_cm}/{current}_{target}/``
    and return ``(n_total_with_files, n_past_success)``:
      - n_total_with_files: # of grasp dirs that have ``wrist_se3.npy`` (i.e.
        usable candidates, not just preview/aux dirs).
      - n_past_success: # of those whose ``stats.json`` shows successes > 0.
    Returns ``(0, 0)`` if the cell directory is missing.
    """
    import os as _os
    cell = _os.path.join(get_candidate_path(hand), "reset", obj_name,
                          f"reorient_{h_cm}", f"{current_int}_{target_int}")
    if not _os.path.isdir(cell):
        return 0, 0
    n_total = 0
    n_succ = 0
    for entry in _os.listdir(cell):
        d = _os.path.join(cell, entry)
        if not _os.path.isdir(d):
            continue
        if not _os.path.exists(_os.path.join(d, "wrist_se3.npy")):
            continue
        n_total += 1
        _a, _s = read_grasp_stats(d)
        if _s > 0:
            n_succ += 1
    return n_total, n_succ


def pick_reorient_target(obj_name: str, current_stem: str,
                         hand: str = "inspire_left",
                         version: str = "v7",
                         min_candidates: int = 1,
                         h_cm: int = 0,
                         ) -> Optional[Tuple[int, str, int]]:
    """Pick a target tabletop pose to reorient to.

    Filters by:
      1. tabletop != current
      2. tabletop has uncovered scenes (>0)
      3. reorient cell ``{current}_{target}`` is **solvable** — at least
         ``min_candidates`` candidate grasps with ``wrist_se3.npy``. Cells
         with a past success are prioritized over untested ones.

    Among feasible targets, ranks: (n_past_success desc, n_uncovered desc).

    Returns ``(target_j_int, stem_str, n_uncovered)`` or ``None`` if no
    feasible target exists.
    """
    stems = _tabletop_stems(obj_name)
    cur_int = int(current_stem)
    candidates: List[Tuple[int, int, int, str]] = []   # (n_succ, n_rem, j_int, stem)
    for stem in stems:
        if stem == str(current_stem):
            continue
        rem = uncovered_scenes(obj_name, stem, hand, version)
        if rem is None:
            continue
        n_rem = len(rem)
        if n_rem == 0:
            continue
        j_int = int(stem)
        n_total, n_succ = _reorient_cell_solvable(
            obj_name, hand, cur_int, j_int, h_cm=h_cm)
        if n_total < min_candidates:
            continue
        candidates.append((n_succ, n_rem, j_int, stem))
    if not candidates:
        return None
    # Sort: past-success cells first, then more uncovered scenes.
    candidates.sort(key=lambda t: (-t[0], -t[1]))
    n_succ, n_rem, j_int, stem = candidates[0]
    return j_int, stem, n_rem
