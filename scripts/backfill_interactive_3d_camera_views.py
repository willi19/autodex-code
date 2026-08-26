#!/usr/bin/env python3
"""Backfill camera_views.json for existing AutoDex interactive 3D exports."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from autodex.interactive_3d.episode_exporter import (
    DEFAULT_EXPERIMENT_ROOT,
    DEFAULT_OUTPUT_ROOT,
    build_camera_views,
    infer_episode_metadata,
    load_gotrack_world_to_robot_base,
    write_json,
)


def is_export_episode_dir(path: Path) -> bool:
    return (path / "animated.glb").is_file() and (path / "manifest.json").is_file()


def resolve_episode_root(export_dir: Path, output_root: Path, experiment_root: Path) -> Path:
    relative = export_dir.relative_to(output_root)
    direct = experiment_root / relative
    if direct.is_dir():
        return direct.resolve()
    info = infer_episode_metadata(export_dir, output_root)
    fallback = experiment_root / Path(str(info["relative_path"]))
    return fallback.resolve()


def update_manifest(manifest_path: Path) -> None:
    try:
        manifest: dict[str, Any] = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        manifest = {}
    outputs = manifest.setdefault("outputs", {})
    outputs["camera_views_json"] = "camera_views.json"
    write_json(manifest_path, manifest, pretty=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--experiment-root", type=Path, default=DEFAULT_EXPERIMENT_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    experiment_root = Path(args.experiment_root).expanduser().resolve()
    export_dirs = [
        path.parent
        for path in sorted(output_root.glob("**/animated.glb"))
        if "_batch_logs" not in path.parts
    ]
    if args.limit is not None:
        export_dirs = export_dirs[: max(0, int(args.limit))]

    ok = 0
    skipped = 0
    failed = 0
    for export_dir in export_dirs:
        if not is_export_episode_dir(export_dir):
            skipped += 1
            continue
        try:
            episode_root = resolve_episode_root(export_dir, output_root, experiment_root)
            if not episode_root.is_dir():
                raise FileNotFoundError(f"episode root not found: {episode_root}")
            transform = load_gotrack_world_to_robot_base(episode_root)
            payload = build_camera_views(
                episode_root=episode_root,
                gotrack_world_to_robot_base=transform,
            )
            write_json(export_dir / "camera_views.json", payload, pretty=True)
            update_manifest(export_dir / "manifest.json")
            ok += 1
        except Exception as exc:
            failed += 1
            print(f"[camera-views] failed {export_dir}: {exc}")

    print(f"[camera-views] ok={ok} skipped={skipped} failed={failed} output_root={output_root}")


if __name__ == "__main__":
    main()
