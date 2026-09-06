"""Adaptive BODex+sim_filter orchestrator for reorient (reset) scenes.

Mirrors `adaptive_orchestrator.py` (box/shelf/wall), but escalates over a
different axis: instead of growing a `gap`, it walks a *config schedule* of
(h_cm, yml_variant, seed_num) — varying release height and contact / inflation
strategy until at least SUCCESS_THRESHOLD sim-filter-passing grasps exist for
each (i, j) tabletop-pose pair.

Per (obj, i, j) state is tracked across rounds; a round runs BODex + sim_filter
on all scenes that are still unmet at the current schedule step. exp_name is
forced unique per variant so candidate directories don't collide.

Output:
  obj/scene/reorient_{h}/{i}_{j}.json       (from gen_reorient_scene)
  bodex_outputs/{hand}/{exp_name}/{obj}/reorient_{h}/{i}_{j}/{seed}/
  candidates/{hand}/{exp_name}/{obj}/reorient_{h}/{i}_{j}/{seed}/
  {output_dir}/{obj}/reorient_summary.json
"""

import os
import sys
import re
import json
import shutil
import argparse
import subprocess
from pathlib import Path

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "src", "grasp_generation", "reorient"))

from gen_scene import gen_reorient_scene  # noqa: E402

from autodex.utils.path import get_obj_root  # noqa: E402


# --- Schedules ---
H_SWEEP_CM = [0, 4, 8, 12]
N_SWEEP = [1000, 5000]
SUCCESS_THRESHOLD = 5
DEFAULT_SEEDS = [123, 456]  # one per N step; advances per-round to avoid duplicate sampling

PARALLEL_PER_N = {200: 10, 1000: 2, 5000: 1}

PYTHON_BODEX = "/home/mingi/miniconda3/envs/bodex/bin/python"
PYTHON_MINGI = "/home/mingi/miniconda3/envs/mingi/bin/python"

HAND_BODEX_CFG_PREFIX = {
    "allegro": "sim_allegro",
    "inspire": "sim_inspire",
    "inspire_left": "sim_inspire_left",
    "inspire_f1": "sim_inspire_f1",
}

BODEX_CFG_ROOT = os.path.join(REPO_ROOT, "src", "grasp_generation", "BODex",
                              "src", "curobo", "content", "configs", "manip")


# ---------------------------------------------------------------------------
# Scene preparation
# ---------------------------------------------------------------------------

def discover_pose_pairs(obj_name: str, obj_root: str):
    tt_dir = Path(obj_root) / obj_name / "processed_data" / "info" / "tabletop"
    ids = sorted(int(p.stem) for p in tt_dir.glob("*.npy"))
    return [(i, j) for i in ids for j in ids if i != j]


def ensure_scenes(obj_name: str, h_cm: int, obj_root: str, pairs):
    h_m = h_cm / 100.0
    out_dir = Path(obj_root) / obj_name / "scene" / f"reorient_{h_cm}"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for i, j in pairs:
        p = out_dir / f"{i}_{j}.json"
        if p.exists():
            written.append(f"{i}_{j}")
            continue
        scene = gen_reorient_scene(obj_name, i, j, h_m, obj_root=obj_root)
        scene["meta"]["scene_type"] = f"reorient_{h_cm}"
        with open(p, "w") as f:
            json.dump(scene, f, indent=2)
        written.append(f"{i}_{j}")
    return written


# ---------------------------------------------------------------------------
# YAML variant discovery
# ---------------------------------------------------------------------------

def discover_yml_variants(hand: str, h_cm: int):
    """Return ordered list of (variant_tag, yml_relpath) for this (hand, h_cm).

    Base reorient first, then inflate variants. The new pressure_constraints
    treats all fingers equally, so pinch variants are skipped.
    `variant_tag` is appended to exp_name so output dirs don't collide.
    """
    prefix = HAND_BODEX_CFG_PREFIX[hand]
    cfg_dir = Path(BODEX_CFG_ROOT) / prefix
    out = []
    base = cfg_dir / f"paradex_reorient_{h_cm}.yml"
    if base.is_file():
        out.append(("base", f"{prefix}/{base.name}"))

    # inflate variants: paradex_reorient_{N}_inflate*.yml
    inflate_re = re.compile(rf"^paradex_reorient_{h_cm}_(\w+)\.yml$")
    for p in sorted(cfg_dir.glob(f"paradex_reorient_{h_cm}_*.yml")):
        m = inflate_re.match(p.name)
        if m:
            out.append((m.group(1), f"{prefix}/{p.name}"))
    return out


# ---------------------------------------------------------------------------
# Subprocess runners
# ---------------------------------------------------------------------------

def run_bodex(yml_relpath, exp_name, obj_list_file, scene_type, scene_ids,
              N, seed, obj_root_dir=None):
    parallel = PARALLEL_PER_N.get(N, 1)
    bodex_dir = os.path.join(REPO_ROOT, "src", "grasp_generation", "BODex")
    filter_file = os.path.join("/tmp", f"reorient_scene_filter_{os.getpid()}.json")
    with open(filter_file, "w") as f:
        json.dump({scene_type: list(scene_ids)}, f)
    cmd = [
        PYTHON_BODEX, "generate.py",
        "-c", yml_relpath,
        "-w", str(parallel),
        "--obj_list_file", obj_list_file,
        "--seed_num", str(N),
        "--exp_name", exp_name,
        "--scene_filter_file", filter_file,
        "--seed", str(seed),
    ]
    if obj_root_dir:
        cmd += ["--obj_root_dir", obj_root_dir]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    proc = subprocess.run(cmd, cwd=bodex_dir, env=env, capture_output=True, text=True)
    try:
        os.remove(filter_file)
    except OSError:
        pass
    if proc.returncode != 0:
        print(f"[BODex FAIL]\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        raise RuntimeError(f"BODex failed (rc={proc.returncode})")
    return proc.stdout


def run_sim_filter(hand, exp_name, obj_name, obj_root_dir=None):
    script = os.path.join(REPO_ROOT, "src", "grasp_generation", "sim_filter", "run_sim_filter.py")
    cmd = [PYTHON_MINGI, script, "--hand", hand, "--version", exp_name, "--obj", obj_name]
    if obj_root_dir:
        cmd += ["--obj_root_dir", obj_root_dir]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[sim_filter FAIL]\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        raise RuntimeError(f"sim_filter failed (rc={proc.returncode})")
    return proc.stdout


# ---------------------------------------------------------------------------
# Bookkeeping
# ---------------------------------------------------------------------------

def count_passing(hand, exp_name, obj_name, scene_type, scene_id):
    cand_dir = os.path.join(REPO_ROOT, "candidates", hand, exp_name,
                            obj_name, scene_type, scene_id)
    if not os.path.isdir(cand_dir):
        return 0
    return sum(1 for d in os.listdir(cand_dir)
               if os.path.isdir(os.path.join(cand_dir, d)))


def mirror_pairs_at_h0(hand, obj_name, pairs):
    """For each (i, j) with i < j, mutually copy candidates between i_j and j_i
    across all reset_0* variants. Only valid at h=0: with h=0 the reorient is
    in-place rotation (no pillars), and the i->j and j->i tasks are inverses of
    each other — a grasp at the start of one is usually valid at the start of
    the other (different scene constraints but typically similar enough).

    Copied entries are prefixed `mirror_{src_sid}_` to avoid name collisions
    and to allow re-running this function idempotently."""
    cand_root = os.path.join(REPO_ROOT, "candidates", hand)
    if not os.path.isdir(cand_root):
        return
    variants = [e for e in os.listdir(cand_root)
                if e == "reset_0" or e.startswith("reset_0_")]
    seen = set()
    n_copied = 0
    for i, j in pairs:
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        if key in seen:
            continue
        seen.add(key)
        sid_ij = f"{i}_{j}"
        sid_ji = f"{j}_{i}"
        for exp in variants:
            dir_ij = os.path.join(cand_root, exp, obj_name, "reorient_0", sid_ij)
            dir_ji = os.path.join(cand_root, exp, obj_name, "reorient_0", sid_ji)
            orig_ij = [d for d in os.listdir(dir_ij)
                       if not d.startswith("mirror_")] if os.path.isdir(dir_ij) else []
            orig_ji = [d for d in os.listdir(dir_ji)
                       if not d.startswith("mirror_")] if os.path.isdir(dir_ji) else []
            if orig_ji:
                os.makedirs(dir_ij, exist_ok=True)
                for d in orig_ji:
                    src = os.path.join(dir_ji, d)
                    dst = os.path.join(dir_ij, f"mirror_{sid_ji}_{d}")
                    if os.path.isdir(src) and not os.path.exists(dst):
                        shutil.copytree(src, dst)
                        n_copied += 1
            if orig_ij:
                os.makedirs(dir_ji, exist_ok=True)
                for d in orig_ij:
                    src = os.path.join(dir_ij, d)
                    dst = os.path.join(dir_ji, f"mirror_{sid_ij}_{d}")
                    if os.path.isdir(src) and not os.path.exists(dst):
                        shutil.copytree(src, dst)
                        n_copied += 1
    return n_copied


def count_passing_combined(hand, h_cm, obj_name, scene_type, scene_id):
    """Sum candidates across all variants (exp_names) for the same h.

    A scene is considered "done" when the union of candidates from base,
    inflate10, etc. reaches the threshold — grasps from different variants
    are all valid for the same reorient task at this h.
    """
    cand_root = os.path.join(REPO_ROOT, "candidates", hand)
    if not os.path.isdir(cand_root):
        return 0
    total = 0
    for exp_name in os.listdir(cand_root):
        if not (exp_name == f"reset_{h_cm}" or exp_name.startswith(f"reset_{h_cm}_")):
            continue
        cand_dir = os.path.join(cand_root, exp_name, obj_name, scene_type, scene_id)
        if os.path.isdir(cand_dir):
            total += sum(1 for d in os.listdir(cand_dir)
                         if os.path.isdir(os.path.join(cand_dir, d)))
    return total


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def process_obj(obj_name, hand, obj_root, obj_list_file,
                h_sweep=H_SWEEP_CM, n_sweep=N_SWEEP, threshold=SUCCESS_THRESHOLD,
                seeds=DEFAULT_SEEDS):
    """Walk schedule of (h, variant, N). Per (i, j) scene, mark done when count >= threshold."""
    pairs = discover_pose_pairs(obj_name, obj_root)
    if not pairs:
        return {"status": "no_tabletop_poses"}

    # Per-scene state. h is an escalation axis: a scene marked done at h=0
    # is skipped at h=4 (h=4 is fallback, not separate task).
    state = {f"{i}_{j}": {"valid": 0, "done": False, "satisfied_at_h": None,
                          "history": []} for i, j in pairs}

    # Build schedule: outer h, middle variant, inner N
    for h_cm in h_sweep:
        active_initial = [sid for sid, s in state.items() if not s["done"]]
        if not active_initial:
            print(f"  [h={h_cm}] all scenes satisfied at earlier h — skip", flush=True)
            continue
        ensure_scenes(obj_name, h_cm, obj_root, pairs)
        variants = discover_yml_variants(hand, h_cm)
        if not variants:
            print(f"  [h={h_cm}] no yml variants — skip", flush=True)
            continue
        scene_type = f"reorient_{h_cm}"

        for variant_tag, yml_relpath in variants:
            exp_name = (f"reset_{h_cm}" if variant_tag == "base"
                        else f"reset_{h_cm}_{variant_tag}")
            for n_idx, N in enumerate(n_sweep):
                active_ids = [sid for sid, s in state.items() if not s["done"]]
                if not active_ids:
                    break
                seed = seeds[n_idx % len(seeds)]
                print(f"  [h={h_cm}] variant={variant_tag} N={N} seed={seed}: "
                      f"{len(active_ids)} active (exp={exp_name})", flush=True)
                try:
                    run_bodex(yml_relpath, exp_name, obj_list_file, scene_type,
                              active_ids, N, seed,
                              obj_root_dir=obj_root)
                    run_sim_filter(hand, exp_name, obj_name,
                                   obj_root_dir=obj_root)
                except RuntimeError as e:
                    print(f"    [FAIL] {e} — skipping this round, continuing schedule",
                          flush=True)
                    for sid in active_ids:
                        state[sid]["history"].append({
                            "h": h_cm, "variant": variant_tag, "exp": exp_name,
                            "N": N, "seed": seed, "valid": None, "error": str(e)})
                    continue
                for sid in active_ids:
                    cnt_variant = count_passing(hand, exp_name, obj_name, scene_type, sid)
                    cnt_total_h = count_passing_combined(hand, h_cm, obj_name, scene_type, sid)
                    s = state[sid]
                    s["valid"] = max(s["valid"], cnt_total_h)
                    s["history"].append({
                        "h": h_cm, "variant": variant_tag, "exp": exp_name,
                        "N": N, "seed": seed,
                        "valid_variant": cnt_variant, "valid_total_h": cnt_total_h})
                    if cnt_total_h >= threshold:
                        s["done"] = True
                        s["satisfied_at_h"] = h_cm
                cnts = sorted([state[sid]["valid"] for sid in active_ids],
                              reverse=True)[:5]
                print(f"    -> top5 valid (h={h_cm} total): {cnts}", flush=True)

        # h=0 only: mutually mirror i_j <-> j_i candidates. With no pillars at
        # h=0 the tasks are inverses and a grasp at the start of one is usually
        # usable at the start of the other.
        if h_cm == 0:
            n_copied = mirror_pairs_at_h0(hand, obj_name, pairs)
            if n_copied:
                print(f"  [h=0] mirrored {n_copied} candidate dirs between "
                      f"i_j <-> j_i pairs", flush=True)

        # End of this h's escalation. Accept-partial: scenes with >0 candidates
        # at this h are marked done (won't escalate to next h). h+1 is only for
        # scenes with literally 0 candidates here (h+1 is fallback for hopeless
        # ones, not a way to mix grasps from physically different trajectories).
        for sid, s in state.items():
            if s["done"]:
                continue
            cnt = count_passing_combined(hand, h_cm, obj_name, scene_type, sid)
            if cnt > 0:
                s["valid"] = max(s["valid"], cnt)
                s["done"] = True
                s["satisfied_at_h"] = h_cm
                s["status"] = f"partial_at_h{h_cm}"

    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand", required=True, choices=list(HAND_BODEX_CFG_PREFIX))
    parser.add_argument("--obj", required=True)
    parser.add_argument("--version", default="v8",
                        help="v8 tabletop/reset asset contract (only supported value)")
    parser.add_argument("--obj_root", default=None,
                        help="optional v8 object-root override; default is object_processing")
    parser.add_argument("--h_sweep", type=int, nargs="+", default=H_SWEEP_CM,
                        help="Release heights in cm to try, in order (default: 0 4 8 12)")
    parser.add_argument("--n_sweep", type=int, nargs="+", default=N_SWEEP)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="One seed per N-step (cycled). Different seeds across N rounds "
                             "give genuinely new grasps instead of overlapping samples.")
    parser.add_argument("--threshold", type=int, default=SUCCESS_THRESHOLD)
    parser.add_argument("--output_dir", default=None,
                        help="Where to write reorient_summary.json (default: REPO/logging/reorient/)")
    args = parser.parse_args()
    if args.version != "v8":
        parser.error("reorient_orchestrator supports only --version v8; legacy assets are not used")
    obj_root = args.obj_root or get_obj_root(args.version)

    obj_list_file = os.path.join("/tmp", f"reorient_obj_list_{os.getpid()}.txt")
    with open(obj_list_file, "w") as f:
        f.write(args.obj + "\n")

    out_dir = args.output_dir or os.path.join(REPO_ROOT, "logging", "reorient", args.hand)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n=== {args.obj} ({args.hand}) ===")
    summary = process_obj(args.obj, args.hand, obj_root, obj_list_file,
                          h_sweep=args.h_sweep, n_sweep=args.n_sweep,
                          threshold=args.threshold, seeds=args.seeds)

    summary_path = os.path.join(out_dir, f"{args.obj}_reorient_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nsummary -> {summary_path}")

    try:
        os.remove(obj_list_file)
    except OSError:
        pass


if __name__ == "__main__":
    main()
