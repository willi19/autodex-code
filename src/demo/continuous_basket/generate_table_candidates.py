#!/usr/bin/env python3
"""Generate NAS-visible Inspire tabletop candidates for Franka collection.

The candidates model the Inspire hand only; every candidate still goes through
the real FR3 planner and must earn an ``arm=franka`` success record during the
collection stage.  This command has no robot or camera dependency.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT))

from src.demo.continuous_basket.catalog import parse_catalog
from src.demo.continuous_basket.prepare_franka_catalog import (
    DEFAULT_BODEX_PYTHON,
    DEFAULT_OBJECT_ROOT,
    DEFAULT_PLANNER_PYTHON,
    DEFAULT_SESSION_ROOT,
    processing_readiness,
    write_table_scenes,
)


def _run(command: list[str], *, execute: bool, cwd: Path | None = None) -> None:
    print("$ " + " ".join(command))
    if execute:
        subprocess.run(command, check=True, cwd=str(cwd) if cwd else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objects", nargs="+", required=True)
    parser.add_argument("--object-root", default=str(DEFAULT_OBJECT_ROOT))
    parser.add_argument("--version", default="v8")
    parser.add_argument("--parallel", type=int, default=4)
    parser.add_argument("--seed-num", type=int, default=200)
    parser.add_argument("--seed-offset", type=int, default=0,
                        help="append BODex seeds instead of colliding with a previous pass")
    parser.add_argument("--bodex-python", default=str(DEFAULT_BODEX_PYTHON))
    parser.add_argument("--sim-python", default=str(DEFAULT_PLANNER_PYTHON))
    parser.add_argument("--session-dir", default=None)
    parser.add_argument("--overwrite-scenes", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.parallel < 1 or args.seed_num < 1 or args.seed_offset < 0:
        parser.error("parallel and seed counts must be positive")

    items = parse_catalog(args.objects)
    object_root = Path(args.object_root).expanduser().resolve()
    missing = [row.name for row in (processing_readiness(item, object_root) for item in items)
               if not row.ready]
    if missing:
        raise SystemExit("incomplete object-processing assets: " + ", ".join(missing))
    session_dir = (Path(args.session_dir).expanduser() if args.session_dir else
                   DEFAULT_SESSION_ROOT / dt.datetime.now().strftime("%Y%m%d_%H%M%S") / "candidates")
    scene_root = Path.home() / "shared_data/AutoDex/scene"
    if not args.execute:
        print("[dry-run] scene files and GPU jobs are not started; re-run with --execute after reviewing commands.")
        ids = {item.name: [str(i) for i, _ in enumerate(sorted(
            (object_root / item.name / "processed_data/info/tabletop").glob("*.npy")
        ))] for item in items}
    else:
        session_dir.mkdir(parents=True, exist_ok=True)
        ids = write_table_scenes(items, object_root=object_root, scene_root=scene_root,
                                 overwrite=args.overwrite_scenes)
        (session_dir / "scene_ids.json").write_text(json.dumps(ids, indent=2) + "\n")

    object_list = session_dir / "objects.txt"
    scene_filter = session_dir / "scene_filter.json"
    if args.execute:
        object_list.write_text("\n".join(item.name for item in items) + "\n")
        scene_filter.write_text(json.dumps({"table": sorted({sid for values in ids.values() for sid in values})}) + "\n")

    bodex_dir = REPO_ROOT / "src/grasp_generation/BODex"
    bodex = [
        args.bodex_python, "generate.py", "-c", "sim_inspire/paradex_table.yml",
        "-w", str(args.parallel), "--obj_list_file", str(object_list),
        "--obj_root_dir", str(object_root), "--exp_name", args.version,
        "--seed_num", str(args.seed_num), "--seed_offset", str(args.seed_offset),
        "--scene_filter_file", str(scene_filter),
    ]
    sim = [
        args.sim_python, str(REPO_ROOT / "src/grasp_generation/sim_filter/run_sim_filter.py"),
        "--hand", "inspire", "--version", args.version,
        "--obj_root_dir", str(object_root), "--obj_list_file", str(object_list),
    ]
    _run(bodex, execute=args.execute, cwd=bodex_dir)
    _run(sim, execute=args.execute)
    if args.execute:
        candidate_root = Path.home() / "shared_data/AutoDex/candidates/inspire" / args.version
        counts = {
            item.name: len(list((candidate_root / item.name).glob("**/wrist_se3.npy")))
            for item in items
        }
        print("[candidates] " + json.dumps(counts, sort_keys=True))


if __name__ == "__main__":
    main()
