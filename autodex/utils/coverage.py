"""Coverage-based grasp ordering for the v8 candidate pool.

Reads precomputed coverage JSON (one entry per candidate grasp listing the
scenes it covers) and runs greedy set cover to produce an ordering of grasps
that covers all reachable scenes with as few candidates as possible.

Coverage JSONs live at::

    {project_dir}/experiment/{version}/coverage/cov_{version}_cand_{obj}.json

``_cand`` is the full candidate pool; the bare ``cov_{version}_{obj}.json`` in
the same dir is the executed-grasp log and is NOT what these readers want.
The runtime default is ``v8``.  Tabletop enumeration receives the matching
object-processing ``obj_root`` explicitly, rather than falling back to a
second asset namespace.

Candidate dirs are keyed by hand alone.  ``run_auto.py`` therefore treats a
successful candidate as shared by every arm and passes no ``arm`` filter when
choosing the next grasp.  The optional ``arm=`` filter remains available for
offline, arm-specific analysis; results with no ``arm`` field predate the FR3
and read as ``"xarm"`` in that diagnostic mode.

Each entry of ``d["grasps"]`` looks like::

    {"type": "wall", "sid": 3, "gid": 17, "pose_idx": "000",
     "covers": [0, 4, 9, ...]}
"""
import json
import os
from typing import Dict, List, Optional, Set, Tuple

from autodex.utils.path import get_candidate_path, project_dir


def _coverage_path(obj_name: str, version: str = "v8") -> str:
    return os.path.join(
        project_dir, "experiment", version, "coverage",
        f"cov_{version}_cand_{obj_name}.json"
    )


# Both readers below sit on the network filesystem: the coverage JSON is
# hundreds of KB and the candidate tree is thousands of small dirs, so a bare
# implementation re-reads them once per tabletop pose. run_auto snapshots every
# pose before and after each trial, which turned into a silent multi-second
# stall at the top of every trial. Cache both, keyed so a stale read cannot
# survive a write.
_JSON_CACHE: Dict[str, Tuple[Tuple[int, int], dict]] = {}
_SUCCESS_CACHE: Dict[Tuple[str, str, str, Optional[str]], Set[Tuple[str, str, str]]] = {}


def _load_coverage_json(path: str) -> dict:
    """Read a coverage JSON, reusing the parse while the file is unchanged."""
    try:
        st = os.stat(path)
        stamp = (st.st_mtime_ns, st.st_size)
    except OSError:
        stamp = None
    hit = _JSON_CACHE.get(path)
    if stamp is not None and hit is not None and hit[0] == stamp:
        return hit[1]
    with open(path) as f:
        data = json.load(f)
    if stamp is not None:
        _JSON_CACHE[path] = (stamp, data)
    return data


def invalidate_success_cache() -> None:
    """Drop the cached candidate-success scan.

    Call after writing a candidate ``result.json``; ``write_candidate_result``
    does it for you.
    """
    _SUCCESS_CACHE.clear()


def write_candidate_result(path: str, payload: dict) -> None:
    """Write a candidate ``result.json`` and invalidate the success cache."""
    with open(path, "w") as f:
        json.dump(payload, f)
    invalidate_success_cache()


def load_coverage_order(
    obj_name: str,
    tabletop_pose_stem: Optional[str] = None,
    version: str = "v8",
) -> Optional[List[Tuple[str, str, str]]]:
    """Greedy set cover over the candidate grasps in the v8 coverage JSON.

    If ``tabletop_pose_stem`` is given, only candidates whose ``pose_idx``
    matches are considered.

    Returns a list of ``(scene_type, scene_id_str, grasp_id_str)`` tuples in
    selection order. Returns ``None`` if the coverage file is missing.
    """
    path = _coverage_path(obj_name, version)
    if not os.path.exists(path):
        return None
    data = _load_coverage_json(path)
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


def load_coverage_map(
    obj_name: str,
    tabletop_pose_stem: Optional[str] = None,
    hand: str = "inspire_left",
    version: str = "v8",
    arm: Optional[str] = None,
) -> Optional[dict]:
    """Return ``dict[(type, sid_str, gid_str) -> n_remaining_uncovered]``
    for every grasp in the v8 coverage json (optionally filtered to a
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
    data = _load_coverage_json(path)
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
    version: str = "v8",
    arm: Optional[str] = None,
) -> Optional[List[dict]]:
    """Same source as ``load_coverage_map`` but keeps the scene SETS.

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
    data = _load_coverage_json(path)
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
                       version: str = "v8",
                       arm: Optional[str] = None) -> Set[Tuple[str, str, str]]:
    """Walk the candidate dir tree once and collect
    ``(scene_type, scene_id_dir, grasp_id_dir)`` keys whose ``result.json``
    has ``success=True``. Used to compute already-covered scenes.

    ``arm`` is an optional offline-analysis filter.  Candidate dirs are keyed
    by HAND only (``candidates/{hand}/{version}/{obj}/...``), so runtime
    collection deliberately calls this with ``None`` and shares successes
    across arms.  A future arm still validates the selected grasp with its own
    IK/collision planner before execution.

    Results written before the ``arm`` field existed are all xarm runs, so a
    missing field reads as ``"xarm"``.
    """
    ck = (obj_name, hand, version, arm)
    hit = _SUCCESS_CACHE.get(ck)
    if hit is not None:
        return hit
    base = os.path.join(get_candidate_path(hand), version, obj_name)
    if not os.path.isdir(base):
        _SUCCESS_CACHE[ck] = set()
        return _SUCCESS_CACHE[ck]
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
    _SUCCESS_CACHE[ck] = out
    return out


def _grasps_at_tabletop(grasps: List[dict], stem: str) -> List[dict]:
    return [g for g in grasps if str(g.get("pose_idx", "")) == str(stem)]


def uncovered_scenes(obj_name: str, tabletop_pose_stem: str,
                     hand: str = "inspire_left",
                     version: str = "v8",
                     arm: Optional[str] = None) -> Optional[Set[int]]:
    """Scene indices at ``tabletop_pose_stem`` not yet covered by any
    on-disk successful grasp.

    Returns ``None`` if the coverage file is missing.
    """
    path = _coverage_path(obj_name, version)
    if not os.path.exists(path):
        return None
    data = _load_coverage_json(path)
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


def uncovered_tabletop_counts(obj_name: str, hand: str, version: str,
                              obj_root: str, arm: Optional[str] = None
                              ) -> Optional[Dict[str, int]]:
    """Remaining coverage count for every tabletop in one asset namespace.

    This is deliberately driven by ``obj_root`` rather than a legacy pose
    mapping.  Callers use it to distinguish a genuinely completed v8 pool
    from a pool that still has uncovered tabletops but lacks a v8 reset cell.
    """
    if not os.path.exists(_coverage_path(obj_name, version)):
        return None
    out: Dict[str, int] = {}
    for stem in _tabletop_stems(obj_name, obj_root):
        remaining = uncovered_scenes(obj_name, stem, hand, version, arm=arm)
        out[stem] = len(remaining or set())
    return out


def _tabletop_stems(obj_name: str, obj_root: str) -> List[str]:
    """Sorted tabletop filename stems for ``obj_name`` (e.g. ``['000','009','016']``).

    ``obj_root`` is required: v8 coverage must enumerate the same
    object-processing tabletop namespace that supplied the candidate poses.
    """
    tt_dir = os.path.join(obj_root, obj_name,
                          "processed_data", "info", "tabletop")
    if not os.path.isdir(tt_dir):
        return []
    return sorted(f[:-4] for f in os.listdir(tt_dir) if f.endswith(".npy"))


def next_grasp_after_success(
    obj_name: str,
    current_grasp_key: Tuple[str, str, str],
    tabletop_pose_stem: Optional[str] = None,
    hand: str = "inspire_left",
    version: str = "v8",
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
    data = _load_coverage_json(path)
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


_REORIENT_MAP_WARNED: Dict[Tuple[str, str], bool] = {}


def _reorient_cell_solvable(obj_name: str, hand: str,
                             current_int: int, target_int: int,
                             version: str = "v8",
                             obj_root: Optional[str] = None) -> Tuple[int, int]:
    """Inspect legacy reset cells through a validated v8→legacy pose map.
    and return ``(n_total_with_files, n_past_success)``:
      - n_total_with_files: # of grasp dirs that have ``wrist_se3.npy`` (i.e.
        usable candidates, not just preview/aux dirs).
      - n_past_success: # of those whose ``stats.json`` shows successes > 0.
    Returns ``(0, 0)`` if the cell directory is missing.

    The existing ``reset_<height>`` tree was generated against the legacy
    paradex tabletop namespace.  ``current_int`` and ``target_int`` arrive in
    the v8/object_processing namespace, so they must each pass the strict,
    symmetry-aware pose map before a candidate directory is used.  An absent
    or ambiguous mapping is treated as no path; it must never select a
    numerically coincident but physically different legacy cell.
    """
    from pathlib import Path as _Path
    from autodex.utils.path import iter_reset_candidate_roots
    from autodex.utils.tabletop_map import to_reset_index

    try:
        legacy_current = to_reset_index(
            obj_name, current_int, obj_root, allow_partial=True)
        legacy_target = to_reset_index(
            obj_name, target_int, obj_root, allow_partial=True)
    except (ValueError, KeyError) as exc:
        warn_key = (obj_name, str(obj_root))
        if not _REORIENT_MAP_WARNED.get(warn_key):
            _REORIENT_MAP_WARNED[warn_key] = True
            print(f"    [reorient] {obj_name}: validated v8→legacy tabletop "
                  f"mapping unavailable ({exc}); legacy reset candidates "
                  "will not be used")
        return 0, 0

    n_total = 0
    n_succ = 0
    # Try the real reset-height roots in ascending order: reset_0, reset_4,
    # reset_8, reset_12.  Their names are release heights, not asset versions.
    for h_cm, root in iter_reset_candidate_roots(hand, version=version):
        cell = (_Path(root) / obj_name / f"reorient_{h_cm}" /
                f"{legacy_current}_{legacy_target}")
        if not cell.is_dir():
            continue
        for d in cell.iterdir():
            if not d.is_dir() or not (d / "wrist_se3.npy").exists():
                continue
            n_total += 1
            _a, _s = read_grasp_stats(str(d))
            if _s > 0:
                n_succ += 1
    return n_total, n_succ


def pick_reorient_target(obj_name: str, current_stem: str,
                         hand: str = "inspire_left",
                         version: str = "v8",
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

    ``obj_root`` selects the v8 tabletop set to enumerate and is also used to
    verify the legacy reset-cell mapping.
    """
    from autodex.utils.path import get_obj_root

    root = obj_root or get_obj_root(version)
    stems = _tabletop_stems(obj_name, root)
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
            obj_name, hand, cur_int, j_int, version=version, obj_root=root)
        if n_total < min_candidates:
            continue
        candidates.append((n_succ, n_rem, j_int, stem))
    if not candidates:
        return None
    # Sort: past-success cells first, then more uncovered scenes.
    candidates.sort(key=lambda t: (-t[0], -t[1]))
    n_succ, n_rem, j_int, stem = candidates[0]
    return j_int, stem, n_rem
