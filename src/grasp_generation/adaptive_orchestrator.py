"""Adaptive BODex+sim_filter orchestrator.

For each (obj, scene_type, scene_id), escalates through a schedule of
(gap/height_offset, seed_num) values until at least SUCCESS_THRESHOLD
sim-filter-passing grasps are found, or the schedule is exhausted.

Stable scene IDs across rounds: scene/{type}/{id}.json is the same logical
scene config across iterations; only the gap/height_offset value changes.

Output:
  {AutoDex}/scene/{hand}/{obj}/{type}/  — per-hand adaptive scene set (1st run backs up to {type}_prev/)
  bodex_outputs/{hand}/{version_adaptive}/{obj}/{type}/{id}/{seed}/
  candidates/{hand}/{version_adaptive}/{obj}/{type}/{id}/{seed}/   — sim-filter-passing
  {output_dir}/{obj}/adaptive_summary.json
"""

import os
import sys
import json
import shutil
import argparse
import subprocess
from itertools import product

import numpy as np
import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "scene_generation"))

from generate_scene import get_shelf_scene, get_wall_scene, get_box_scene, get_tabletop_scene  # noqa: E402

from autodex.utils.path import obj_path as default_obj_path, get_scene_dir  # noqa: E402


# --- Schedules ---
GAP_SWEEP_WALL_SHELF = [0.02, 0.04, 0.06, 0.08]
HEIGHT_SWEEP_BOX = [0.05, 0.08]
N_SWEEP = [200, 1000]
SUCCESS_THRESHOLD = 5

# BODex GPU memory budget — W_ks tensor scales as -w * N.
# CLAUDE.md says -w=10 is safe for N=200 (default).  We keep -w*N <= 2000.
PARALLEL_PER_N = {200: 20, 1000: 4, 5000: 1}

Z_ROTS_DEFAULT = [0, 72, 144, 216, 288]
VERTICAL_THRESH = 0.95
AXIS_VEC = {"x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0])}

PYTHON_BODEX = "/home/mingi/miniconda3/envs/bodex/bin/python"
PYTHON_MINGI = "/home/mingi/miniconda3/envs/mingi/bin/python"

HAND_BODEX_CFG_PREFIX = {
    "allegro": "sim_allegro",
    "inspire": "sim_inspire",
    "inspire_left": "sim_inspire_left",
}


# ---------------------------------------------------------------------------
# Symmetry
# ---------------------------------------------------------------------------

def load_symmetry_registry():
    p = os.path.join(REPO_ROOT, "src", "scene_generation", "symmetry.json")
    if not os.path.isfile(p):
        return {}
    with open(p) as f:
        reg = json.load(f)
    reg.pop("_comment", None)
    return reg


_SYM_CACHE = {}


def _obj_sym_axes(obj_name, obj_root):
    """Rotational-symmetry axes from the object_processing detector's output
    ({obj}/processed_data/info/symmetry.json). Returns a list of
    (axis_unit_vec (3,), fold) where fold is an int or 'inf' (continuous)."""
    key = (obj_name, obj_root)
    if key not in _SYM_CACHE:
        p = os.path.join(obj_root, obj_name, "processed_data", "info", "symmetry.json")
        axes = []
        if os.path.isfile(p):
            for a in json.load(open(p)).get("axes", []):
                v = np.asarray(a["axis"], float)
                axes.append((v / np.linalg.norm(v), a["fold"]))
        _SYM_CACHE[key] = axes
    return _SYM_CACHE[key]


def z_rots_for_pose(obj_name, tabletop_pose, obj_root):
    """Fold redundant scene z_rotations using the object's detected symmetry:
      - isotropic (>=3 independent continuous axes, i.e. a sphere) -> [0]
      - a continuous (revolute) axis that is VERTICAL in this pose -> [0]
      - otherwise the default 5-way z_rot sweep."""
    axes = _obj_sym_axes(obj_name, obj_root)
    inf = [v for v, fold in axes if fold == "inf"]
    if len(inf) >= 3 and np.linalg.matrix_rank(np.array(inf), tol=0.1) >= 3:
        return [0]                          # sphere: every z_rotation identical
    for v in inf:
        if abs((tabletop_pose[:3, :3] @ v)[2]) >= VERTICAL_THRESH:
            return [0]                      # revolute axis stands vertical
    return Z_ROTS_DEFAULT


# ---------------------------------------------------------------------------
# Scene config enumeration (stable IDs)
# ---------------------------------------------------------------------------

def enumerate_scene_configs(obj_name, scene_type, obj_root, symmetry_reg, pose_filter=None):
    """Return list of dicts describing each scene's (stable) config.

    Each dict has 'id' (str, stable), 'pose_idx', 'pose' (4x4), and scene-type
    specific params. The 'id' is just a sequential integer string.

    pose_filter: optional iterable of pose_idx strings; if set, only those poses included.
    """
    obj_dir = os.path.join(obj_root, obj_name)
    tabletop_dir = os.path.join(obj_dir, "processed_data", "info", "tabletop")
    pose_files = sorted(os.listdir(tabletop_dir))

    out = []
    sid = 0  # incremented regardless of filter so IDs stay stable across runs
    for fname in pose_files:
        pose_idx = fname.split(".")[0]
        pose = np.load(os.path.join(tabletop_dir, fname))
        include = pose_filter is None or pose_idx in pose_filter
        if scene_type == "wall":
            for z_rot in z_rots_for_pose(obj_name, pose, obj_root):
                if include:
                    out.append({"id": str(sid), "pose_idx": pose_idx, "pose": pose,
                                "z_rot": z_rot})
                sid += 1
        elif scene_type == "shelf":
            face_combos = [(up, side, True) for up, side in product([True, False], repeat=2)
                           if (up or side)]
            for z_rot in z_rots_for_pose(obj_name, pose, obj_root):
                for up, side, back in face_combos:
                    if include:
                        out.append({"id": str(sid), "pose_idx": pose_idx, "pose": pose,
                                    "z_rot": z_rot, "up": up, "side": side, "back": back})
                    sid += 1
        elif scene_type == "box":
            if include:
                out.append({"id": str(sid), "pose_idx": pose_idx, "pose": pose})
            sid += 1
        elif scene_type == "table":
            if include:
                out.append({"id": str(sid), "pose_idx": pose_idx, "pose": pose})
            sid += 1
        else:
            raise ValueError(scene_type)
    return out


def build_scene_dict(obj_name, scene_type, cfg, obb_info, gap_or_h):
    if scene_type == "wall":
        return get_wall_scene(obj_name, cfg["pose"], obb_info, cfg["z_rot"], gap_or_h)
    if scene_type == "shelf":
        return get_shelf_scene(obj_name, cfg["pose"], obb_info, cfg["z_rot"], gap_or_h,
                               up=cfg["up"], side=cfg["side"], back=cfg["back"])
    if scene_type == "box":
        return get_box_scene(obj_name, cfg["pose"], gap_or_h)
    if scene_type == "table":
        return get_tabletop_scene(obj_name, cfg["pose"])
    raise ValueError(scene_type)


def make_meta(scene_type, cfg, gap_or_h):
    meta = {"pose_idx": cfg["pose_idx"], "param": {}}
    if scene_type == "wall":
        meta["param"] = {"z_rotation_deg": cfg["z_rot"], "gap": gap_or_h}
    elif scene_type == "shelf":
        meta["param"] = {"z_rotation_deg": cfg["z_rot"], "gap": gap_or_h,
                         "up": cfg["up"], "side": cfg["side"], "back": cfg["back"]}
    elif scene_type == "box":
        meta["param"] = {"height_offset": gap_or_h}
    elif scene_type == "table":
        meta["param"] = {}
    return meta


def write_scene_file(hand, obj_name, obj_root, scene_type, cfg, obb_info, gap_or_h):
    """Write {hand}/{obj}/{type}/{id}.json. Returns True if scene is non-None."""
    scene = build_scene_dict(obj_name, scene_type, cfg, obb_info, gap_or_h)
    if scene is None:
        return False
    out_dir = get_scene_dir(hand, obj_name, scene_type)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{cfg['id']}.json"), "w") as f:
        json.dump({"scene": scene, "meta": make_meta(scene_type, cfg, gap_or_h)}, f, indent=2)
    return True


def initial_scene_generation(hand, obj_name, scene_type, obj_root, symmetry_reg, initial_gap, pose_filter=None):
    """Backup existing {hand}/{obj}/{type}/ to _prev (once), then write fresh
    scenes with stable IDs at initial_gap. Returns dict {id: cfg}.
    """
    obj_dir = os.path.join(obj_root, obj_name)
    out_dir = get_scene_dir(hand, obj_name, scene_type)

    if os.path.isdir(out_dir):
        prev = out_dir + "_prev"
        if not os.path.isdir(prev):
            os.rename(out_dir, prev)
        else:
            shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    obb_info = json.load(open(os.path.join(obj_dir, "processed_data", "info", "simplified.json")))
    cfgs = enumerate_scene_configs(obj_name, scene_type, obj_root, symmetry_reg, pose_filter=pose_filter)

    scenes = {}
    for cfg in cfgs:
        ok = write_scene_file(hand, obj_name, obj_root, scene_type, cfg, obb_info, initial_gap)
        if not ok:
            continue  # scene impossible at this gap (e.g. box too small)
        scenes[cfg["id"]] = cfg
    return scenes, obb_info


# ---------------------------------------------------------------------------
# Subprocess runners
# ---------------------------------------------------------------------------

def run_bodex(hand, exp_name, scene_type, obj_name, seed_num, scene_ids,
              obj_list_file, parallel=None, obj_root_dir=None,
              task_f=(0.0, 0.0, -1.0), task_gamma=30.0, seed=123):
    """task_f / task_gamma default to gravity (world -z) with 30° cone."""
    if parallel is None:
        parallel = PARALLEL_PER_N.get(seed_num, 1)
    """Invoke BODex generate.py with filtering. Returns stdout/stderr capture for log."""
    bodex_dir = os.path.join(REPO_ROOT, "src", "grasp_generation", "BODex")
    cfg_prefix = HAND_BODEX_CFG_PREFIX[hand]
    cfg_file = f"{cfg_prefix}/paradex_{scene_type}.yml"

    # scene_filter file
    filter_file = os.path.join("/tmp", f"bodex_scene_filter_{os.getpid()}.json")
    with open(filter_file, "w") as f:
        json.dump({scene_type: list(scene_ids)}, f)

    cmd = [
        PYTHON_BODEX, "generate.py",
        "-c", cfg_file,
        "-w", str(parallel),
        "--obj_list_file", obj_list_file,
        "--seed_num", str(seed_num),
        "--exp_name", exp_name,
        "--scene_filter_file", filter_file,
        "--task_f", str(task_f[0]), str(task_f[1]), str(task_f[2]),
        "--task_gamma", str(task_gamma),
        "--seed", str(seed),
    ]
    if obj_root_dir:
        cmd += ["--obj_root_dir", obj_root_dir]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    import time
    last_err = None
    for attempt in range(4):  # up to 4 attempts (initial + 3 retries)
        proc = subprocess.run(cmd, cwd=bodex_dir, env=env, capture_output=True, text=True)
        if proc.returncode == 0:
            try:
                os.remove(filter_file)
            except OSError:
                pass
            return proc.stdout
        last_err = proc.stderr or proc.stdout
        oom = "OutOfMemoryError" in last_err or "CUDA out of memory" in last_err
        if oom and attempt < 3:
            wait = 60 * (attempt + 1)  # 60s, 120s, 180s
            print(f"[BODex OOM, attempt {attempt+1}/4] sleeping {wait}s and retrying...")
            time.sleep(wait)
            continue
        break
    try:
        os.remove(filter_file)
    except OSError:
        pass
    print(f"[BODex FAIL]\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    raise RuntimeError(f"BODex failed (rc={proc.returncode})")


def run_sim_filter(hand, exp_name, obj_name, obj_root_dir=None):
    """Invoke sim_filter for one object. exp_name == version."""
    script = os.path.join(REPO_ROOT, "src", "grasp_generation", "sim_filter", "run_sim_filter.py")
    cmd = [
        PYTHON_MINGI, script,
        "--hand", hand,
        "--version", exp_name,
        "--obj", obj_name,
    ]
    if obj_root_dir:
        cmd += ["--obj_root_dir", obj_root_dir]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[sim_filter FAIL]\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        raise RuntimeError(f"sim_filter failed (rc={proc.returncode})")
    return proc.stdout


# ---------------------------------------------------------------------------
# Per-scene state & cleanup
# ---------------------------------------------------------------------------

def candidate_seed_prefix(N, seed=None):
    """Prefix for candidate seed dirs.

    Includes N (round) and (optionally) seed to avoid collisions when
    multiple BODex seeds are used per round.
    """
    if seed is None:
        return f"n{N}_"
    return f"n{N}_s{seed}_"


def count_passing(hand, exp_name, obj_name, scene_type, scene_id):
    """Count valid candidates for this scene at its current gap.

    Candidates accumulate within a single gap across N escalations. When gap
    advances, the directory is wiped (caller's responsibility).
    """
    cand_dir = os.path.join(REPO_ROOT, "candidates", hand, exp_name,
                            obj_name, scene_type, scene_id)
    if not os.path.isdir(cand_dir):
        return 0
    return sum(1 for d in os.listdir(cand_dir)
               if os.path.isdir(os.path.join(cand_dir, d)))


def tag_fresh_candidates(hand, exp_name, obj_name, scene_type, scene_id, N, seed=None):
    """Rename unprefixed candidate seed dirs (just written by sim_filter)
    so they don't collide with future rounds."""
    cand_dir = os.path.join(REPO_ROOT, "candidates", hand, exp_name,
                            obj_name, scene_type, scene_id)
    if not os.path.isdir(cand_dir):
        return
    prefix = candidate_seed_prefix(N, seed=seed)
    for d in os.listdir(cand_dir):
        if d.startswith("n"):
            continue  # already prefixed from a prior round
        src = os.path.join(cand_dir, d)
        if os.path.isdir(src):
            os.rename(src, os.path.join(cand_dir, prefix + d))


def clear_scene_bodex(hand, exp_name, obj_name, scene_type, scene_id):
    """Wipe bodex_outputs for this scene (candidates kept — same gap)."""
    bodex_dir = os.path.join(REPO_ROOT, "bodex_outputs", hand, exp_name,
                             obj_name, scene_type, scene_id)
    if os.path.isdir(bodex_dir):
        shutil.rmtree(bodex_dir)


def clear_scene_candidates(hand, exp_name, obj_name, scene_type, scene_id):
    """Wipe candidates for this scene (when gap advances — fresh pool)."""
    cand_dir = os.path.join(REPO_ROOT, "candidates", hand, exp_name,
                            obj_name, scene_type, scene_id)
    if os.path.isdir(cand_dir):
        shutil.rmtree(cand_dir)


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def process_obj_scene_type(obj_name, scene_type, hand, exp_name, obj_root,
                           symmetry_reg, obj_list_file, parallel=20, pose_filter=None, seeds=(123,),
                           resume=False, summary_path=None):
    """Run adaptive loop for one (obj, scene_type). Returns summary dict."""
    if scene_type == "box":
        sched = HEIGHT_SWEEP_BOX
    elif scene_type == "table":
        # No gap/height dimension — single scene per tabletop pose, only N escalates.
        sched = [None]
    else:
        sched = GAP_SWEEP_WALL_SHELF
    obb_info_path = os.path.join(obj_root, obj_name, "processed_data", "info", "simplified.json")
    obb_info = json.load(open(obb_info_path))

    # Load prior state if resuming
    prior_state = {}
    if resume and summary_path and os.path.isfile(summary_path):
        try:
            prior_summary = json.load(open(summary_path))
            prior_state = prior_summary.get(scene_type, {})
        except Exception:
            prior_state = {}

    # Initial generation with first gap (always re-generate scene files, but optionally skip wipe of candidates)
    scenes, _ = initial_scene_generation(hand, obj_name, scene_type, obj_root, symmetry_reg, sched[0],
                                         pose_filter=pose_filter)
    print(f"  [{scene_type}] {len(scenes)} initial scenes at gap/h={sched[0]}")

    # Track per-scene state. (gap_idx, N_idx) — gap outer, N inner.
    state = {sid: {"gap_idx": 0, "N_idx": 0, "done": False, "best_valid": 0,
                   "final": None, "history": []}
             for sid in scenes}

    # If resuming: seed state from prior summary. Successful scenes marked done.
    # Exhausted/failed scenes: parse history → set (gap_idx, N_idx) to first untried round.
    if resume:
        n_resumed = 0
        n_skip_ahead = 0
        n_all_tried = 0
        for sid in state:
            prior = prior_state.get(sid)
            if not prior:
                continue
            if prior.get("final", {}).get("status") == "success":
                state[sid]["done"] = True
                state[sid]["final"] = prior["final"]
                state[sid]["best_valid"] = prior.get("best_valid", prior["final"].get("valid", 0))
                state[sid]["history"] = prior.get("history", [])
                prior_gap = prior["final"].get("gap", sched[0])
                write_scene_file(hand, obj_name, obj_root, scene_type, scenes[sid], obb_info, prior_gap)
                n_resumed += 1
            else:
                # Exhausted scene — keep as-is (don't retry). User explicitly requested downscale only.
                state[sid]["done"] = True
                state[sid]["final"] = prior["final"]
                state[sid]["best_valid"] = prior.get("best_valid", 0)
                state[sid]["history"] = prior.get("history", [])
                n_all_tried += 1
        print(f"  [{scene_type}] resume: {n_resumed} success-restored, {n_skip_ahead} skip-ahead, {n_all_tried} all-tried (kept exhausted)")

    # Wipe prior outputs for scenes NOT being resumed (i.e. not done)
    base_bodex = os.path.join(REPO_ROOT, "bodex_outputs", hand, exp_name, obj_name, scene_type)
    base_cand = os.path.join(REPO_ROOT, "candidates", hand, exp_name, obj_name, scene_type)
    if not resume:
        for d in (base_bodex, base_cand):
            if os.path.isdir(d):
                shutil.rmtree(d)
    else:
        # Selective wipe: only non-done scenes
        for sid, s in state.items():
            if not s["done"]:
                clear_scene_bodex(hand, exp_name, obj_name, scene_type, sid)
                clear_scene_candidates(hand, exp_name, obj_name, scene_type, sid)

    # Schedule order: gap outer, N inner.
    # Each gap independently tries to reach SUCCESS_THRESHOLD via N escalation;
    # if exhausted, advance to next gap (candidates from prior gap wiped).
    schedule = [(gap_idx, N_idx)
                for gap_idx in range(len(sched))
                for N_idx in range(len(N_SWEEP))]

    for step_gi, step_ni in schedule:
        active_ids = [sid for sid, s in state.items()
                      if not s["done"] and s["gap_idx"] == step_gi and s["N_idx"] == step_ni]
        if not active_ids:
            continue

        gap = sched[step_gi]
        N = N_SWEEP[step_ni]
        print(f"  [{scene_type}] gap={gap} N={N}: {len(active_ids)} active")

        # When entering a new gap (N_idx==0 and not the very first), regen scene
        # files and WIPE this scene's prior candidates (different gap = fresh pool).
        if step_ni == 0 and step_gi > 0:
            valid_active = []
            for sid in active_ids:
                ok = write_scene_file(hand, obj_name, obj_root, scene_type,
                                      scenes[sid], obb_info, gap)
                if not ok:
                    # Scene not buildable at this gap (e.g. box too small at larger h)
                    if state[sid]["best_valid"] > 0:
                        state[sid]["final"] = {"status": "blocked_partial",
                                               "best_valid": state[sid]["best_valid"],
                                               "blocked_at_gap": gap}
                    else:
                        state[sid]["final"] = {"status": "scene_none",
                                               "blocked_at_gap": gap}
                    state[sid]["done"] = True
                    continue
                clear_scene_candidates(hand, exp_name, obj_name, scene_type, sid)
                valid_active.append(sid)
            active_ids = valid_active
            if not active_ids:
                continue

        # Single BODex call per round; cycle through seeds across rounds (option A).
        round_idx = step_gi * len(N_SWEEP) + step_ni
        bd_seed = seeds[round_idx % len(seeds)]
        for sid in active_ids:
            clear_scene_bodex(hand, exp_name, obj_name, scene_type, sid)
        run_bodex(hand, exp_name, scene_type, obj_name, N, active_ids,
                  obj_list_file, parallel=None,
                  obj_root_dir=obj_root if obj_root != default_obj_path else None,
                  seed=bd_seed)
        run_sim_filter(hand, exp_name, obj_name,
                       obj_root_dir=obj_root if obj_root != default_obj_path else None)
        for sid in active_ids:
            tag_fresh_candidates(hand, exp_name, obj_name, scene_type, sid, N, seed=bd_seed)

        # Cumulative count within current gap
        per_scene_valid = []
        for sid in active_ids:
            cnt = count_passing(hand, exp_name, obj_name, scene_type, sid)
            per_scene_valid.append((sid, cnt))
            s = state[sid]
            s["best_valid"] = max(s["best_valid"], cnt)
            s["history"].append({"gap": gap, "N": N, "valid_cum": cnt})
            if cnt >= SUCCESS_THRESHOLD:
                s["done"] = True
                s["final"] = {"status": "success", "gap": gap, "N": N, "valid": cnt}
            elif s["N_idx"] + 1 < len(N_SWEEP):
                s["N_idx"] += 1  # next N at same gap
            elif s["gap_idx"] + 1 < len(sched):
                s["gap_idx"] += 1
                s["N_idx"] = 0   # next gap, restart N
            else:
                s["done"] = True
                s["final"] = {"status": "exhausted", "best_valid": s["best_valid"]}

        n_succ = sum(1 for _, c in per_scene_valid if c >= SUCCESS_THRESHOLD)
        n_zero = sum(1 for _, c in per_scene_valid if c == 0)
        cnts = sorted([c for _, c in per_scene_valid], reverse=True)[:5]
        print(f"    -> valid (top5): {cnts}  succ={n_succ}/{len(active_ids)}  zero={n_zero}")

    # ── Downscale phase (resume-only, BATCHED): gradual descent across scenes.
    # All active scenes at the same descent step are grouped by (g, N) and BODex-batched.
    if resume and scene_type != "box":
        ds_state = {}
        for sid, s in state.items():
            if not (s["done"] and s.get("final", {}).get("status") == "success"):
                continue
            final_gap = s["final"].get("gap")
            if final_gap is None or final_gap <= sched[0] + 1e-9:
                continue
            tried = set()
            for h in s.get("history", []):
                g, n = h.get("gap"), h.get("N")
                if g is not None and n is not None:
                    tried.add((round(float(g), 4), int(n)))
            # combos: descending gap, N ascending. (gap_idx desc, then N_idx asc)
            combos = []
            for gi in range(len(sched) - 1, -1, -1):
                g_v = sched[gi]
                if g_v is None or g_v >= final_gap - 1e-9:
                    continue
                for n_v in N_SWEEP:
                    if (round(float(g_v), 4), n_v) not in tried:
                        combos.append((g_v, n_v))
            if not combos:
                continue
            cand_dir = os.path.join(REPO_ROOT, "candidates", hand, exp_name,
                                    obj_name, scene_type, sid)
            best_backup = cand_dir + "_ds_backup"
            if os.path.isdir(best_backup):
                shutil.rmtree(best_backup)
            if os.path.isdir(cand_dir):
                shutil.copytree(cand_dir, best_backup)
            ds_state[sid] = {
                "queue": combos,
                "step": 0,
                "best_final": dict(state[sid]["final"]),
                "best_backup": best_backup,
                "active": True,
            }
        if ds_state:
            print(f"  [{scene_type}] downscale phase (batched): {len(ds_state)} scenes")
            while any(s["active"] for s in ds_state.values()):
                # Group active scenes by (g, n) at their current step
                groups = {}
                for sid, s in ds_state.items():
                    if not s["active"]:
                        continue
                    g, n = s["queue"][s["step"]]
                    groups.setdefault((g, n), []).append(sid)
                if not groups:
                    break
                for (g_v, n_v), sids in groups.items():
                    # Per scene: write scene file at g, wipe candidates, clear bodex
                    for sid in sids:
                        ok = write_scene_file(hand, obj_name, obj_root, scene_type,
                                              scenes[sid], obb_info, g_v)
                        if not ok:
                            ds_state[sid]["active"] = False
                            continue
                        clear_scene_candidates(hand, exp_name, obj_name, scene_type, sid)
                        clear_scene_bodex(hand, exp_name, obj_name, scene_type, sid)
                    valid_sids = [s for s in sids if ds_state[s]["active"]]
                    if not valid_sids:
                        continue
                    bd_seed = seeds[0]
                    run_bodex(hand, exp_name, scene_type, obj_name, n_v, valid_sids,
                              obj_list_file, parallel=None,
                              obj_root_dir=obj_root if obj_root != default_obj_path else None,
                              seed=bd_seed)
                    run_sim_filter(hand, exp_name, obj_name,
                                   obj_root_dir=obj_root if obj_root != default_obj_path else None)
                    for sid in valid_sids:
                        tag_fresh_candidates(hand, exp_name, obj_name, scene_type, sid, n_v, seed=bd_seed)
                        cnt = count_passing(hand, exp_name, obj_name, scene_type, sid)
                        state[sid]["history"].append({"gap": g_v, "N": n_v, "valid_cum": cnt, "phase": "downscale"})
                        s = ds_state[sid]
                        if cnt >= SUCCESS_THRESHOLD:
                            s["best_final"] = {"status": "success", "gap": g_v, "N": n_v,
                                                "valid": cnt, "via": "downscale"}
                            cand_dir = os.path.join(REPO_ROOT, "candidates", hand, exp_name,
                                                    obj_name, scene_type, sid)
                            if os.path.isdir(s["best_backup"]):
                                shutil.rmtree(s["best_backup"])
                            if os.path.isdir(cand_dir):
                                shutil.copytree(cand_dir, s["best_backup"])
                            s["step"] += 1
                            if s["step"] >= len(s["queue"]):
                                s["active"] = False
                            print(f"    downscale {sid}: gap={g_v} N={n_v} ({cnt}) ACCEPTED")
                        else:
                            cand_dir = os.path.join(REPO_ROOT, "candidates", hand, exp_name,
                                                    obj_name, scene_type, sid)
                            if os.path.isdir(cand_dir):
                                shutil.rmtree(cand_dir)
                            if os.path.isdir(s["best_backup"]):
                                shutil.move(s["best_backup"], cand_dir)
                            s["best_backup"] = cand_dir + "_ds_backup"
                            write_scene_file(hand, obj_name, obj_root, scene_type, scenes[sid], obb_info,
                                             s["best_final"].get("gap", sched[0]))
                            clear_scene_bodex(hand, exp_name, obj_name, scene_type, sid)
                            s["active"] = False
                            print(f"    downscale {sid}: gap={g_v} N={n_v} fail ({cnt}) → keep gap={s['best_final'].get('gap')}")
            # Finalize states + cleanup backups
            for sid, s in ds_state.items():
                state[sid]["final"] = s["best_final"]
                if os.path.isdir(s["best_backup"]):
                    shutil.rmtree(s["best_backup"])

    # ── Bonus phase: on success at sched[0], try gap 0.0 (v7 behavior).
    if scene_type != "box":
        bonus_gap = 0.0
        bonus_candidates = [sid for sid, s in state.items()
                            if s["final"] and s["final"].get("status") == "success"
                            and abs(s["final"].get("gap", -1) - sched[0]) < 1e-9]
        if bonus_candidates:
            print(f"  [{scene_type}] bonus phase: gap={bonus_gap} on {len(bonus_candidates)} success-at-{sched[0]} scenes")
            # Backup primary candidates, regen scenes at bonus gap, clear bodex
            for sid in bonus_candidates:
                cand_dir = os.path.join(REPO_ROOT, "candidates", hand, exp_name,
                                        obj_name, scene_type, sid)
                backup = cand_dir + "_pre_bonus"
                if os.path.isdir(backup):
                    shutil.rmtree(backup)
                if os.path.isdir(cand_dir):
                    shutil.move(cand_dir, backup)
                ok = write_scene_file(hand, obj_name, obj_root, scene_type,
                                      scenes[sid], obb_info, bonus_gap)
                if not ok:
                    # Scene not buildable at gap=0.0. Restore primary.
                    if os.path.isdir(backup):
                        shutil.move(backup, cand_dir)
                    write_scene_file(hand, obj_name, obj_root, scene_type,
                                     scenes[sid], obb_info, sched[0])
                    bonus_candidates = [s for s in bonus_candidates if s != sid]
                    continue
                clear_scene_bodex(hand, exp_name, obj_name, scene_type, sid)

            # Run BODex+sim_filter at N=200 with first seed (cheap bonus)
            if bonus_candidates:
                bd_seed = seeds[0]
                bonus_N = N_SWEEP[0]
                run_bodex(hand, exp_name, scene_type, obj_name, bonus_N, bonus_candidates,
                          obj_list_file, parallel=None,
                          obj_root_dir=obj_root if obj_root != default_obj_path else None,
                          seed=bd_seed)
                run_sim_filter(hand, exp_name, obj_name,
                               obj_root_dir=obj_root if obj_root != default_obj_path else None)
                for sid in bonus_candidates:
                    tag_fresh_candidates(hand, exp_name, obj_name, scene_type, sid, bonus_N, seed=bd_seed)
                    cnt = count_passing(hand, exp_name, obj_name, scene_type, sid)
                    backup = os.path.join(REPO_ROOT, "candidates", hand, exp_name,
                                          obj_name, scene_type, sid) + "_pre_bonus"
                    if cnt >= SUCCESS_THRESHOLD:
                        # Bonus succeeded. Discard backup. Update final to bonus.
                        if os.path.isdir(backup):
                            shutil.rmtree(backup)
                        s = state[sid]
                        s["final"] = {"status": "success", "gap": bonus_gap,
                                      "N": bonus_N, "valid": cnt, "via": "bonus"}
                        s["history"].append({"gap": bonus_gap, "N": bonus_N,
                                             "valid_cum": cnt, "phase": "bonus"})
                        print(f"    bonus {sid}: {cnt} valid at gap={bonus_gap} -> ACCEPTED")
                    else:
                        # Bonus failed. Restore primary candidates and scene file.
                        cand_dir = os.path.join(REPO_ROOT, "candidates", hand, exp_name,
                                                obj_name, scene_type, sid)
                        if os.path.isdir(cand_dir):
                            shutil.rmtree(cand_dir)
                        if os.path.isdir(backup):
                            shutil.move(backup, cand_dir)
                        write_scene_file(hand, obj_name, obj_root, scene_type,
                                         scenes[sid], obb_info, sched[0])
                        clear_scene_bodex(hand, exp_name, obj_name, scene_type, sid)
                        state[sid]["history"].append({"gap": bonus_gap, "N": bonus_N,
                                                      "valid_cum": cnt, "phase": "bonus_rejected"})
                        print(f"    bonus {sid}: {cnt} valid at gap={bonus_gap} -> REJECTED (primary kept)")

    return {sid: state[sid] for sid in state}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", required=True, choices=list(HAND_BODEX_CFG_PREFIX))
    parser.add_argument("--version", required=True,
                        help="exp_name for BODex / version for sim_filter (e.g. v3_adaptive)")
    parser.add_argument("--obj", default=None, help="Single object name (dry-run mode)")
    parser.add_argument("--obj_list_file", default=None)
    parser.add_argument("--scenes", nargs="+", default=["wall", "shelf", "box"],
                        choices=["wall", "shelf", "box", "table"])
    parser.add_argument("--obj_root", default=default_obj_path)
    parser.add_argument("--parallel", type=int, default=20)
    parser.add_argument("--output_dir", default=None,
                        help="Where to write adaptive_summary.json (default: REPO/logging/adaptive/)")
    parser.add_argument("--pose_filter", nargs="+", default=None,
                        help="Only include scenes whose tabletop pose_idx is in this list (e.g. --pose_filter 002).")
    parser.add_argument("--resume", action="store_true",
                        help="Skip scenes already marked success in prior summary.json; only run failed/missing.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[123],
                        help="One or more random seeds. Each (gap, N) round runs BODex+sim_filter once per seed, "
                             "accumulating candidates (tagged with seed).")
    args = parser.parse_args()

    # Build obj list
    if args.obj:
        obj_list = [args.obj]
        obj_list_file = os.path.join("/tmp", f"adaptive_obj_list_{os.getpid()}.txt")
        with open(obj_list_file, "w") as f:
            f.write(args.obj + "\n")
    else:
        obj_list_file = args.obj_list_file or os.path.join(
            REPO_ROOT, "src", "grasp_generation", "obj_list.txt"
        )
        with open(obj_list_file) as f:
            obj_list = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    output_root = args.output_dir or os.path.join(REPO_ROOT, "logging", "adaptive", args.hand, args.version)
    os.makedirs(output_root, exist_ok=True)

    symmetry_reg = load_symmetry_registry()

    for obj_name in obj_list:
        print(f"\n=== {obj_name} ===")
        obj_summary = {}
        for scene_type in args.scenes:
            print(f" {scene_type} ...")
            summary_path = os.path.join(output_root, f"{obj_name}.json")
            state = process_obj_scene_type(
                obj_name, scene_type, args.hand, args.version,
                args.obj_root, symmetry_reg, obj_list_file, parallel=args.parallel,
                pose_filter=set(args.pose_filter) if args.pose_filter else None,
                seeds=tuple(args.seeds),
                resume=args.resume, summary_path=summary_path,
            )
            obj_summary[scene_type] = {
                sid: {"final": s["final"], "best_valid": s["best_valid"],
                      "history": s["history"]}
                for sid, s in state.items()
            }
            # Quick stats
            ok = sum(1 for s in state.values() if s["final"] and s["final"].get("status") == "success")
            none = sum(1 for s in state.values() if s["final"] and s["final"].get("status") == "scene_none")
            exh = sum(1 for s in state.values() if s["final"] and s["final"].get("status") == "exhausted")
            print(f"  {scene_type}: success={ok} scene_none={none} exhausted={exh}")

        out_path = os.path.join(output_root, f"{obj_name}.json")
        with open(out_path, "w") as f:
            json.dump(obj_summary, f, indent=2)
        print(f"  Summary -> {out_path}")


if __name__ == "__main__":
    main()
