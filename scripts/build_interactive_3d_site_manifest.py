#!/usr/bin/env python3
"""Build gallery-facing manifests for exported AutoDex interactive 3D assets."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional


DEFAULT_OUTPUT_ROOT = Path.home() / "shared_data" / "AutoDex" / "interactive_3d"
DEFAULT_GALLERY_EXPERIMENTS = Path.home() / "autodex-gallery" / "docs" / "experiments.json"


def infer_relative_parts(relative_path: Path) -> Optional[dict[str, str]]:
    parts = relative_path.parts
    hand_idx = next((idx for idx, part in enumerate(parts) if part in {"allegro", "inspire"}), None)
    if hand_idx is None or hand_idx + 2 >= len(parts):
        return None
    return {
        "hand": parts[hand_idx],
        "object": parts[hand_idx + 1],
        "episode": parts[-1],
        "relative_path": str(relative_path),
    }


def build_rank_lookup(experiments_json: Optional[Path]) -> dict[tuple[str, str, str], dict[str, str]]:
    if not experiments_json or not experiments_json.is_file():
        return {}
    data = json.loads(experiments_json.read_text(encoding="utf-8"))
    lookup: dict[tuple[str, str, str], dict[str, str]] = {}
    for hand, objects in data.items():
        for object_name, ranks in (objects or {}).items():
            for rank, payload in (ranks or {}).items():
                dir_idx = str((payload or {}).get("dir_idx") or "")
                if not dir_idx:
                    continue
                rank_str = str(rank).zfill(3)
                lookup[(hand, object_name, dir_idx)] = {
                    "rank": str(rank),
                    "rank_str": rank_str,
                    "gallery_key": "|".join([hand, object_name, str(rank)]),
                }
    return lookup


def exported_entries(output_root: Path, rank_lookup: dict[tuple[str, str, str], dict[str, str]]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for animated_glb in sorted(output_root.glob("**/animated.glb")):
        if "_batch_logs" in animated_glb.parts:
            continue
        relative_glb = animated_glb.relative_to(output_root)
        relative_episode = relative_glb.parent
        info = infer_relative_parts(relative_episode)
        if not info:
            continue
        rank_info = rank_lookup.get((info["hand"], info["object"], info["episode"]), {})
        entry = {
            **info,
            **rank_info,
            "animated_glb": str(relative_glb),
            "scene_glb": str(relative_episode / "scene.glb"),
            "preview_glb": str(relative_episode / "preview.glb"),
            "trajectory_json": str(relative_episode / "trajectory.json"),
            "camera_views_json": str(relative_episode / "camera_views.json"),
            "manifest_json": str(relative_episode / "manifest.json"),
            "site_glb": "interactive_3d/" + str(relative_glb),
            "site_scene_glb": "interactive_3d/" + str(relative_episode / "scene.glb"),
            "site_preview_glb": "interactive_3d/" + str(relative_episode / "preview.glb"),
            "site_trajectory_json": "interactive_3d/" + str(relative_episode / "trajectory.json"),
            "site_camera_views_json": "interactive_3d/" + str(relative_episode / "camera_views.json"),
            "site_manifest_json": "interactive_3d/" + str(relative_episode / "manifest.json"),
        }
        entries.append(entry)
    return entries


def build_assets3d(entries: list[dict[str, Any]]) -> dict[str, Any]:
    assets: dict[str, Any] = {"base_url": "", "objects": {}}
    for entry in entries:
        obj = assets["objects"].setdefault(entry["object"], {"episodes": {}})
        episode_payload = {
            "glb": entry["site_glb"],
            "animated_glb": entry["site_glb"],
            "preview_glb": entry["site_preview_glb"],
            "scene_glb": entry["site_scene_glb"],
            "trajectory_json": entry["site_trajectory_json"],
            "camera_views_json": entry["site_camera_views_json"],
            "manifest_json": entry["site_manifest_json"],
            "source_episode": entry["relative_path"],
            "type": "animated_glb",
        }
        keys = [
            entry["relative_path"],
            "/".join([entry["hand"], entry["object"], entry["episode"]]),
            entry["episode"],
        ]
        if entry.get("gallery_key"):
            keys.append(entry["gallery_key"])
        if entry.get("rank_str"):
            keys.append(entry["rank_str"])
        if entry.get("rank"):
            keys.append(entry["rank"])
        for key in dict.fromkeys(keys):
            obj["episodes"][key] = episode_payload
    return assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gallery-experiments-json", type=Path, default=DEFAULT_GALLERY_EXPERIMENTS)
    parser.add_argument("--index-path", type=Path, default=None)
    parser.add_argument("--assets3d-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    site_dir = output_root / "_site"
    site_dir.mkdir(parents=True, exist_ok=True)
    index_path = args.index_path or site_dir / "index.json"
    assets3d_path = args.assets3d_path or site_dir / "assets3d.json"
    rank_lookup = build_rank_lookup(Path(args.gallery_experiments_json).expanduser())
    entries = exported_entries(output_root, rank_lookup)
    index_payload = {
        "version": 1,
        "output_root": str(output_root),
        "entry_count": len(entries),
        "entries": entries,
    }
    index_path.write_text(json.dumps(index_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assets3d_path.write_text(json.dumps(build_assets3d(entries), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[site-manifest] entries={len(entries)}")
    print(f"[site-manifest] index={index_path}")
    print(f"[site-manifest] assets3d={assets3d_path}")


if __name__ == "__main__":
    main()
