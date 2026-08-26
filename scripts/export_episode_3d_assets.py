#!/usr/bin/env python3
"""Export GLB and trajectory assets for one tracked AutoDex episode."""
from __future__ import annotations

import argparse
from pathlib import Path

from autodex.interactive_3d import ExportConfig, export_episode_assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export scene.glb, animated.glb, preview.glb, trajectory.json, and manifest.json "
        "for one AutoDex episode with GoTrack object poses.",
    )
    parser.add_argument("--episode-root", required=True, type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path.home() / "shared_data" / "AutoDex" / "interactive_3d",
    )
    parser.add_argument(
        "--experiment-root",
        type=Path,
        default=Path.home() / "shared_data" / "AutoDex" / "experiment" / "selected_100",
        help="Used to preserve the episode's relative path under --output-root.",
    )
    parser.add_argument(
        "--robot-asset-root",
        type=Path,
        default=Path.home() / "shared_data" / "AutoDex" / "content" / "assets" / "robot",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument(
        "--preview-frame",
        default="middle",
        help="first, middle, last, or a numeric frame index.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = export_episode_assets(
        ExportConfig(
            episode_root=args.episode_root,
            output_root=args.output_root,
            experiment_root=args.experiment_root,
            robot_asset_root=args.robot_asset_root,
            stride=args.stride,
            max_frames=args.max_frames,
            preview_frame=args.preview_frame,
            overwrite=args.overwrite,
        )
    )
    print(f"[interactive_3d] output_dir={result.output_dir}")
    print(f"[interactive_3d] manifest={result.manifest_path}")
    print(f"[interactive_3d] trajectory={result.trajectory_path}")
    print(f"[interactive_3d] scene_glb={result.scene_glb_path}")
    print(f"[interactive_3d] animated_glb={result.animated_glb_path}")
    print(f"[interactive_3d] preview_glb={result.preview_glb_path}")
    print(
        "[interactive_3d] "
        f"frames={result.frame_count} "
        f"robot_geometries={result.robot_geometry_count} "
        f"object_mesh={result.object_mesh_path}"
    )


if __name__ == "__main__":
    main()
