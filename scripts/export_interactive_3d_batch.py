#!/usr/bin/env python3
"""Batch export animated interactive 3D assets for tracked AutoDex episodes."""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional

from autodex.interactive_3d import ExportConfig, export_episode_assets
from autodex.interactive_3d.episode_exporter import resolve_output_dir


DEFAULT_EXPERIMENT_ROOT = Path.home() / "shared_data" / "AutoDex" / "experiment" / "selected_100"
DEFAULT_OUTPUT_ROOT = Path.home() / "shared_data" / "AutoDex" / "interactive_3d"
DEFAULT_LOG_DIR = Path.home() / "shared_data" / "AutoDex" / "interactive_3d" / "_batch_logs"


@dataclass(frozen=True)
class EpisodeTask:
    episode_root: str
    output_dir: str
    relative_path: str


def discover_tasks(
    experiment_root: Path,
    output_root: Path,
    *,
    limit: Optional[int],
    only_missing: bool,
) -> list[EpisodeTask]:
    records = sorted(experiment_root.glob("**/object_tracking/gotrack_output/world_pose_records.json"))
    tasks: list[EpisodeTask] = []
    for record_path in records:
        episode_root = record_path.parents[2]
        output_dir = resolve_output_dir(
            episode_root=episode_root.resolve(),
            output_root=output_root.expanduser(),
            experiment_root=experiment_root.expanduser(),
        )
        animated_glb = output_dir / "animated.glb"
        if only_missing and animated_glb.is_file():
            continue
        try:
            relative_path = str(episode_root.relative_to(experiment_root))
        except ValueError:
            relative_path = str(episode_root)
        tasks.append(
            EpisodeTask(
                episode_root=str(episode_root),
                output_dir=str(output_dir),
                relative_path=relative_path,
            )
        )
        if limit is not None and len(tasks) >= limit:
            break
    return tasks


def run_one(task: EpisodeTask, args: argparse.Namespace) -> dict:
    started = time.time()
    output_dir = Path(task.output_dir)
    try:
        result = export_episode_assets(
            ExportConfig(
                episode_root=Path(task.episode_root),
                output_root=Path(args.output_root),
                experiment_root=Path(args.experiment_root),
                robot_asset_root=Path(args.robot_asset_root),
                stride=args.stride,
                max_frames=args.max_frames,
                preview_frame=args.preview_frame,
                overwrite=args.overwrite or output_dir.exists(),
            )
        )
        return {
            "status": "ok",
            "relative_path": task.relative_path,
            "episode_root": task.episode_root,
            "output_dir": str(result.output_dir),
            "animated_glb": str(result.animated_glb_path),
            "frames": result.frame_count,
            "robot_geometries": result.robot_geometry_count,
            "elapsed_sec": round(time.time() - started, 3),
        }
    except Exception as exc:
        return {
            "status": "failed",
            "relative_path": task.relative_path,
            "episode_root": task.episode_root,
            "output_dir": task.output_dir,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "elapsed_sec": round(time.time() - started, 3),
        }


def append_jsonl(path: Path, records: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n")
            f.flush()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--robot-asset-root",
        type=Path,
        default=Path.home() / "shared_data" / "AutoDex" / "content" / "assets" / "robot",
    )
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--preview-frame", default="middle")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--include-existing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = time.strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) / f"interactive_3d_export_{run_id}"
    events_path = log_dir / "events.jsonl"
    summary_path = log_dir / "summary.json"

    tasks = discover_tasks(
        Path(args.experiment_root).expanduser().resolve(),
        Path(args.output_root).expanduser().resolve(),
        limit=args.limit,
        only_missing=not args.include_existing and not args.overwrite,
    )
    summary = {
        "run_id": run_id,
        "experiment_root": str(Path(args.experiment_root).expanduser().resolve()),
        "output_root": str(Path(args.output_root).expanduser().resolve()),
        "tasks_total": len(tasks),
        "workers": args.workers,
        "stride": args.stride,
        "max_frames": args.max_frames,
        "started_at": time.time(),
        "completed": 0,
        "failed": 0,
    }
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_jsonl(events_path, [{"event": "start", **summary}])
    print(f"[batch] tasks={len(tasks)} workers={args.workers}")
    print(f"[batch] log_dir={log_dir}")

    if not tasks:
        summary["finished_at"] = time.time()
        summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return

    completed = 0
    failed = 0
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = [executor.submit(run_one, task, args) for task in tasks]
        for future in as_completed(futures):
            record = future.result()
            completed += 1
            if record["status"] == "failed":
                failed += 1
            append_jsonl(events_path, [record])
            summary.update(
                {
                    "completed": completed,
                    "failed": failed,
                    "finished": completed == len(tasks),
                    "updated_at": time.time(),
                }
            )
            summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(
                f"[batch] {completed}/{len(tasks)} {record['status']} "
                f"{record['relative_path']} ({record['elapsed_sec']:.1f}s)",
                flush=True,
            )

    summary["finished_at"] = time.time()
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    append_jsonl(events_path, [{"event": "finish", **summary}])
    print(f"[batch] done completed={completed} failed={failed}")


if __name__ == "__main__":
    main()
