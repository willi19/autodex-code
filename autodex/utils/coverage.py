"""Coverage-based grasp ordering (v7 / v8 candidate pools).

Reads precomputed coverage JSON (one entry per candidate grasp listing the
scenes it covers) and runs greedy set cover to produce an ordering of grasps
that covers all reachable scenes with as few candidates as possible.

Coverage JSONs live at::

    {project_dir}/experiment/{version}/coverage/cov_{version}_cand_{obj}.json

``_cand`` is the full candidate pool; the bare ``cov_{version}_{obj}.json`` in
the same dir is the executed-grasp log and is NOT what these readers want.
The ``v7`` defaults are historical — always pass ``version`` explicitly.
Meshes/tabletops for a v8 pool live under object_processing, so pass the
matching ``obj_root`` (see ``autodex.utils.path.get_obj_root``) too.

Coverage is scoped per ARM. Candidate dirs are keyed by hand alone, so xarm and
FR3 campaigns on the same hand share result files; pass ``arm=`` to count only
that arm's successes (results with no ``arm`` field predate the FR3 and read as
``"xarm"``). ``arm=None`` keeps the old, arm-blind behaviour.

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
    version: str = "v7",
) -> Optional[List[Tuple[str, str, str]]]:
    """Greedy set cover over the candidate grasps in the v7 coverage JSON.

    If ``tabletop_pose_stem`` is given, only candidates whose ``pose_idx``
    matches are considered.

    Returns a list of ``(scene_type, scene_id_str, grasp_id_str)`` tuples in
    selection order. Returns ``None`` if the coverage file is missing.
    """
    path = _coverage_path(obj_name, version)
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
    arm: Optional[str] = None,
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
    path = _coverage_path(obj_name, version)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    grasps = data.get("grasps") or []
    if tabletop_pose_stem is not None:
        grasps = [g for g in grasps
                  if str(g.get("pose_idx", "")) == str(tabletop_pose_stem)]
    success_keys = _disk_success_keys(obj_name, hand, version, arm=arm)
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


def load_coverage_entries(
    obj_name: str,
    tabletop_pose_stem: Optional[str] = None,
    hand: str = "inspire_left",
    version: str = "v7",
    arm: Optional[str] = None,
) -> Optional[List[dict]]:
    """Same source as ``load_v7_coverage_map`` but keeps the scene SETS.

    Returns a list of ``{"key": (type, sid, gid), "remaining": set[int]}``,
    sorted by ``len(remaining)`` desc and filtered to entries that still add
    something. The map variant only returns counts, which is enough to rank
    one grasp at a time but not to score a placement that makes several
    grasps reachable at once — union of scene sets is not the sum of counts.

    Returns ``None`` if the coverage file is missing.
    """
    path = _coverage_path(obj_name, version)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    all_grasps = data.get("grasps") or []
    grasps = all_grasps
    if tabletop_pose_stem is not None:
        grasps = _grasps_at_tabletop(all_grasps, tabletop_pose_stem)

    success_keys = _disk_success_keys(obj_name, hand, version, arm=arm)
    covered_scenes: Set[int] = set()
    for g in all_grasps:      # unfiltered: a covered scene stays covered
        key = (str(g["type"]), str(g["sid"]), str(g["gid"]))
        if key in success_keys:
            covered_scenes.update(g.get("covers", []))

    out = []
    for g in grasps:
        key = (str(g["type"]), str(g["sid"]), str(g["gid"]))
        if key in success_keys:
            continue
        remaining = set(g.get("covers", [])) - covered_scenes
        if remaining:
            out.append({"key": key, "remaining": remaining})
    out.sort(key=lambda e: -len(e["remaining"]))
    return out


def _disk_success_keys(obj_name: str, hand: str,
                       version: str = "v7",
                       arm: Optional[str] = None) -> Set[Tuple[str, str, str]]:
    """Walk the candidate dir tree once and collect
    ``(scene_type, scene_id_dir, grasp_id_dir)`` keys whose ``result.json``
    has ``success=True``. Used to compute already-covered scenes.

    ``arm`` scopes the successes to one arm. Candidate dirs are keyed by HAND
    only (``candidates/{hand}/{version}/{obj}/...``), so an xarm campaign and an
    FR3 campaign on the same hand write into the same files — without this
    filter the FR3 inherits the xarm's coverage and reads as "nothing left to
    do". ``None`` keeps the legacy behaviour (count every success).

    Results written before the ``arm`` field existed are all xarm runs, so a
    missing field reads as ``"xarm"``.
    """
    base = os.path.join(get_candidate_path(hand), version, obj_name)
    if not os.path.isdir(base):
        return set()
    out: Set[Tuple[str, str, str]] = set()
    for dirpath, dirnames, filenames in os.walk(base):
        if "result.json" not in filenames:
            continue
        try:
            with open(os.path.join(dirpath, "result.json")) as f:
                rec = json.load(f)
            if not rec.get("success", False):
                continue
            if arm is not None and str(rec.get("arm", "xarm")) != str(arm):
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
                     version: str = "v7",
                     arm: Optional[str] = None) -> Optional[Set[int]]:
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
    success_keys = _disk_success_keys(obj_name, hand, version, arm=arm)
    covered: Set[int] = set()
    for g in tt_grasps:
        key = (str(g["type"]), str(g["sid"]), str(g["gid"]))
        if key in success_keys:
            covered.update(g.get("covers", []))
    return all_scenes - covered


def _tabletop_stems(obj_name: str, obj_root: Optional[str] = None) -> List[str]:
    """Sorted tabletop filename stems for ``obj_name`` (e.g. ``['000','009','016']``).

    ``obj_root`` defaults to the legacy ``obj_path``; pass
    ``get_obj_root(version)`` so a v8 pool enumerates object_processing stems.
    """
    tt_dir = os.path.join(obj_root or obj_path, obj_name,
                          "processed_data", "info", "tabletop")
    if not os.path.isdir(tt_dir):
        return []
    return sorted(f[:-4] for f in os.listdir(tt_dir) if f.endswith(".npy"))


def next_grasp_after_success(
    obj_name: str,
    current_grasp_key: Tuple[str, str, str],
    tabletop_pose_stem: Optional[str] = None,
    hand: str = "inspire_left",
    version: str = "v7",
    arm: Optional[str] = None,
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

    disk_success = _disk_success_keys(obj_name, hand, version, arm=arm)
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


_REORIENT_MAP_WARNED: Dict[str, bool] = {}


def _reorient_cell_solvable(obj_name: str, hand: str,
                             current_int: int, target_int: int,
                             h_cm: int = 0,
                             obj_root: Optional[str] = None) -> Tuple[int, int]:
    """Inspect ``candidates/{hand}/reset/{obj}/reorient_{h_cm}/{current}_{target}/``
    and return ``(n_total_with_files, n_past_success)``:
      - n_total_with_files: # of grasp dirs that have ``wrist_se3.npy`` (i.e.
        usable candidates, not just preview/aux dirs).
      - n_past_success: # of those whose ``stats.json`` shows successes > 0.
    Returns ``(0, 0)`` if the cell directory is missing.

    ``current_int``/``target_int`` are indices in ``obj_root``'s numbering; the
    cell dirs are numbered against the legacy paradex tree, so they are mapped
    (see ``autodex.utils.tabletop_map``). Without that, a v8 stem indexes the
    wrong cell or none at all, and the target drops out silently.
    """
    import os as _os
    from autodex.utils.tabletop_map import to_reset_index

    try:
        cur_cell = to_reset_index(obj_name, current_int, obj_root)
        tgt_cell = to_reset_index(obj_name, target_int, obj_root)
    except (ValueError, KeyError) as exc:
        # The two asset trees disagree about this object's stable poses (apple:
        # paradex has 3, object_processing has 4, and the closest pair is 104
        # degrees apart), so the reset cells were generated for tabletops this
        # pool does not use. That means "no reorient path", not a crash: return
        # an empty cell so pick_reorient_target drops the target and the runner
        # reports there is nowhere left to go.
        if not _REORIENT_MAP_WARNED.get(obj_name):
            _REORIENT_MAP_WARNED[obj_name] = True
            print(f"    [reorient] {obj_name}: no tabletop mapping to the reset "
                  f"cells ({exc}) — reorient unavailable for this pool")
        return 0, 0
    cell = _os.path.join(get_candidate_path(hand), "reset", obj_name,
                          f"reorient_{h_cm}", f"{cur_cell}_{tgt_cell}")
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
                         obj_root: Optional[str] = None,
                         arm: Optional[str] = None,
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

    ``obj_root`` selects the tabletop set to enumerate (see ``_tabletop_stems``).
    """
    stems = _tabletop_stems(obj_name, obj_root)
    cur_int = int(current_stem)
    candidates: List[Tuple[int, int, int, str]] = []   # (n_succ, n_rem, j_int, stem)
    for stem in stems:
        if stem == str(current_stem):
            continue
        rem = uncovered_scenes(obj_name, stem, hand, version, arm=arm)
        if rem is None:
            continue
        n_rem = len(rem)
        if n_rem == 0:
            continue
        j_int = int(stem)
        n_total, n_succ = _reorient_cell_solvable(
            obj_name, hand, cur_int, j_int, h_cm=h_cm, obj_root=obj_root)
        if n_total < min_candidates:
            continue
        candidates.append((n_succ, n_rem, j_int, stem))
    if not candidates:
        return None
    # Sort: past-success cells first, then more uncovered scenes.
    candidates.sort(key=lambda t: (-t[0], -t[1]))
    n_succ, n_rem, j_int, stem = candidates[0]
    return j_int, stem, n_rem
