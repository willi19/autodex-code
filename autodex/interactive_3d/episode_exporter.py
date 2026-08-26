"""Export interactive 3D assets from a tracked AutoDex episode."""
from __future__ import annotations

import json
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


GOTRACK_REL = Path("object_tracking") / "gotrack_output"
DEFAULT_EXPERIMENT_ROOT = (
    Path.home() / "shared_data" / "AutoDex" / "experiment" / "selected_100"
)
DEFAULT_OUTPUT_ROOT = Path.home() / "shared_data" / "AutoDex" / "interactive_3d"
DEFAULT_ROBOT_ASSET_ROOT = (
    Path.home() / "shared_data" / "AutoDex" / "content" / "assets" / "robot"
)
DEFAULT_OBJECT_ROOTS = (
    Path.home() / "shared_data" / "AutoDex" / "object" / "paradex",
    Path.home() / "shared_data" / "AutoDex" / "object" / "robothome",
)


@dataclass(frozen=True)
class ExportConfig:
    episode_root: Path
    output_root: Path = DEFAULT_OUTPUT_ROOT
    experiment_root: Path = DEFAULT_EXPERIMENT_ROOT
    robot_asset_root: Path = DEFAULT_ROBOT_ASSET_ROOT
    object_roots: Tuple[Path, ...] = DEFAULT_OBJECT_ROOTS
    stride: int = 1
    max_frames: Optional[int] = None
    preview_frame: str = "middle"
    overwrite: bool = False


@dataclass(frozen=True)
class ExportResult:
    output_dir: Path
    manifest_path: Path
    trajectory_path: Path
    camera_views_path: Path
    scene_glb_path: Path
    animated_glb_path: Path
    preview_glb_path: Path
    frame_count: int
    robot_geometry_count: int
    object_mesh_path: Path


def export_episode_assets(config: ExportConfig) -> ExportResult:
    episode_root = Path(config.episode_root).expanduser().resolve()
    if not episode_root.is_dir():
        raise FileNotFoundError(f"Episode directory not found: {episode_root}")

    output_dir = resolve_output_dir(
        episode_root=episode_root,
        output_root=Path(config.output_root).expanduser(),
        experiment_root=Path(config.experiment_root).expanduser(),
    )
    if output_dir.exists() and any(output_dir.iterdir()) and not config.overwrite:
        raise FileExistsError(f"Output already exists; pass --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = infer_episode_metadata(episode_root, Path(config.experiment_root).expanduser())
    summary = load_json(episode_root / GOTRACK_REL / "summary.json", default={})
    records = load_world_pose_records(episode_root / GOTRACK_REL / "world_pose_records.json")
    qpos = load_synced_qpos(episode_root)
    timestamps = load_timestamps(episode_root)
    gotrack_world_to_robot_base = load_gotrack_world_to_robot_base(episode_root)

    frame_indices = select_frame_indices(
        records=records,
        qpos_len=len(qpos),
        timestamp_len=len(timestamps),
        stride=max(1, int(config.stride)),
        max_frames=config.max_frames,
    )
    if not frame_indices:
        raise RuntimeError("No common frames found between tracking records and robot qpos")

    robot_hand = str(metadata.get("hand") or "").lower()
    urdf_path = resolve_urdf_path(robot_hand, Path(config.robot_asset_root).expanduser())
    object_name = str(metadata.get("object") or summary.get("object_name") or "")
    object_mesh_path = resolve_object_mesh_path(
        summary=summary,
        object_name=object_name,
        object_roots=tuple(Path(p).expanduser() for p in config.object_roots),
    )

    trimesh, yourdfpy = import_geometry_modules()
    urdf = yourdfpy.URDF.load(
        str(urdf_path),
        mesh_dir=str(urdf_path.parent),
        build_collision_scene_graph=False,
        load_collision_meshes=False,
    )
    robot_geometry_names = list(urdf.scene.geometry.keys())
    robot_node_names = [robot_node_name(name) for name in robot_geometry_names]
    if not robot_geometry_names:
        raise RuntimeError(f"URDF produced no visual geometry: {urdf_path}")

    records_by_frame = {
        int(r["frame_index"]): r
        for r in records
        if isinstance(r, dict) and r.get("pose_world") is not None and "frame_index" in r
    }
    preview_index = choose_preview_frame(frame_indices, config.preview_frame)

    object_mesh = load_object_mesh(object_mesh_path, trimesh)
    mesh_scale = float(summary.get("resolved_mesh_scale", summary.get("mesh_scale", 1.0)) or 1.0)
    if mesh_scale != 1.0:
        object_mesh = object_mesh.copy()
        object_mesh.apply_scale(mesh_scale)

    trajectory_frames: List[Dict[str, Any]] = []
    for frame_index in frame_indices:
        record = records_by_frame[frame_index]
        q = qpos_to_urdf_cfg(qpos[frame_index], len(urdf.actuated_joint_names))
        object_pose_gotrack = pose_from_record(record)
        object_pose = gotrack_world_to_robot_base @ object_pose_gotrack
        link_poses = robot_geometry_poses(urdf, q, robot_geometry_names)
        trajectory_frames.append(
            {
                "frame_index": int(frame_index),
                "time_sec": timestamp_for_frame(timestamps, frame_index, record),
                "status": record.get("status"),
                "stage": record.get("stage"),
                "object_pose_world": matrix_to_nested_list(object_pose),
                "object_pose_gotrack_world": matrix_to_nested_list(object_pose_gotrack),
                "robot_qpos": [float(v) for v in q.tolist()],
                "robot_geometry_poses_world": link_poses,
                "tracking": compact_tracking_fields(record),
            }
        )

    first_frame = frame_indices[0]
    first_q = qpos_to_urdf_cfg(qpos[first_frame], len(urdf.actuated_joint_names))
    first_object_pose = gotrack_world_to_robot_base @ pose_from_record(records_by_frame[first_frame])
    scene_glb_path = output_dir / "scene.glb"
    export_static_scene_glb(
        scene_glb_path,
        urdf=urdf,
        robot_geometry_names=robot_geometry_names,
        qpos=first_q,
        object_mesh=object_mesh,
        object_pose=first_object_pose,
        trimesh=trimesh,
    )
    animated_glb_path = output_dir / "animated.glb"
    bake_animation_into_glb(
        source_glb_path=scene_glb_path,
        output_glb_path=animated_glb_path,
        trajectory_frames=trajectory_frames,
        animated_node_names=["object::mesh", *robot_node_names],
    )

    preview_q = qpos_to_urdf_cfg(qpos[preview_index], len(urdf.actuated_joint_names))
    preview_object_pose = gotrack_world_to_robot_base @ pose_from_record(records_by_frame[preview_index])
    preview_glb_path = output_dir / "preview.glb"
    export_static_scene_glb(
        preview_glb_path,
        urdf=urdf,
        robot_geometry_names=robot_geometry_names,
        qpos=preview_q,
        object_mesh=object_mesh,
        object_pose=preview_object_pose,
        trimesh=trimesh,
    )

    trajectory = {
        "version": 1,
        "coordinate_frame": "robot_base",
        "episode_root": str(episode_root),
        "relative_episode_path": str(metadata["relative_path"]),
        "transforms": {
            "gotrack_world_to_robot_base": matrix_to_nested_list(gotrack_world_to_robot_base),
        },
        "frame_count": len(trajectory_frames),
        "source_frame_count": {
            "tracking_records": len(records),
            "robot_qpos": len(qpos),
            "timestamps": len(timestamps),
        },
        "robot": {
            "hand": robot_hand,
            "urdf_path": str(urdf_path),
            "actuated_joint_names": list(urdf.actuated_joint_names),
            "geometry_node_names": robot_node_names,
            "urdf_geometry_names": robot_geometry_names,
        },
        "object": {
            "name": object_name,
            "node_name": "object::mesh",
            "mesh_path": str(object_mesh_path),
            "mesh_scale_applied": mesh_scale,
            "summary": {
                k: summary.get(k)
                for k in (
                    "unit_scale_mode",
                    "resolved_mesh_scale",
                    "translation_scale_to_gotrack",
                    "external_unit_scale_to_meter",
                    "mesh_max_extent",
                    "detected_external_unit",
                    "unit_scale_reason",
                )
                if k in summary
            },
        },
        "frames": trajectory_frames,
    }
    trajectory_path = output_dir / "trajectory.json"
    write_json(trajectory_path, trajectory, pretty=False)

    camera_views = build_camera_views(
        episode_root=episode_root,
        gotrack_world_to_robot_base=gotrack_world_to_robot_base,
    )
    camera_views_path = output_dir / "camera_views.json"
    write_json(camera_views_path, camera_views, pretty=True)

    manifest = {
        "version": 1,
        "status": "complete",
        "episode_root": str(episode_root),
        "relative_episode_path": str(metadata["relative_path"]),
        "hand": robot_hand,
        "object": object_name,
        "episode": metadata.get("episode"),
        "outputs": {
            "scene_glb": "scene.glb",
            "animated_glb": "animated.glb",
            "preview_glb": "preview.glb",
            "trajectory_json": "trajectory.json",
            "camera_views_json": "camera_views.json",
        },
        "inputs": {
            "world_pose_records": str(episode_root / GOTRACK_REL / "world_pose_records.json"),
            "gotrack_summary": str(episode_root / GOTRACK_REL / "summary.json"),
            "arm_state": str(episode_root / "arm" / "state.npy"),
            "hand_state": str(episode_root / "hand" / "state.npy"),
            "timestamps": str(episode_root / "raw" / "timestamps" / "timestamp.npy"),
            "C2R": str(episode_root / "C2R.npy"),
            "camera_intrinsics": str(episode_root / "cam_param" / "intrinsics.json"),
            "camera_extrinsics": str(episode_root / "cam_param" / "extrinsics.json"),
            "urdf": str(urdf_path),
            "object_mesh": str(object_mesh_path),
        },
        "coordinate_frame": "robot_base",
        "transforms": {
            "gotrack_world_to_robot_base": matrix_to_nested_list(gotrack_world_to_robot_base),
        },
        "frame_count": len(trajectory_frames),
        "preview_frame_index": int(preview_index),
        "robot_geometry_count": len(robot_geometry_names),
    }
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest, pretty=True)

    return ExportResult(
        output_dir=output_dir,
        manifest_path=manifest_path,
        trajectory_path=trajectory_path,
        camera_views_path=camera_views_path,
        scene_glb_path=scene_glb_path,
        animated_glb_path=animated_glb_path,
        preview_glb_path=preview_glb_path,
        frame_count=len(trajectory_frames),
        robot_geometry_count=len(robot_geometry_names),
        object_mesh_path=object_mesh_path,
    )


def import_geometry_modules():
    try:
        import trimesh  # type: ignore
        import yourdfpy  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "trimesh/yourdfpy are required. Run this with the AutoDex conda env, "
            "for example: /home/robot/anaconda3/envs/autodex/bin/python"
        ) from exc
    return trimesh, yourdfpy


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any, *, pretty: bool) -> None:
    if pretty:
        text = json.dumps(payload, indent=2, sort_keys=True)
    else:
        text = json.dumps(payload, separators=(",", ":"))
    path.write_text(text + "\n", encoding="utf-8")


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def resolve_output_dir(episode_root: Path, output_root: Path, experiment_root: Path) -> Path:
    if is_relative_to(episode_root, experiment_root.resolve()):
        rel = episode_root.relative_to(experiment_root.resolve())
    else:
        rel = infer_episode_metadata(episode_root, experiment_root).get("relative_path")
    return (output_root / Path(str(rel))).resolve()


def infer_episode_metadata(episode_root: Path, experiment_root: Path) -> Dict[str, Any]:
    episode_root = episode_root.resolve()
    experiment_root = experiment_root.expanduser().resolve()
    if is_relative_to(episode_root, experiment_root):
        rel = episode_root.relative_to(experiment_root)
    else:
        rel = Path(*episode_root.parts[-3:])

    parts = rel.parts
    hand_idx = None
    for idx, part in enumerate(parts):
        if part in {"allegro", "inspire"}:
            hand_idx = idx
            break
    if hand_idx is None or hand_idx + 2 >= len(parts):
        hand = parts[-3] if len(parts) >= 3 else ""
        obj = parts[-2] if len(parts) >= 2 else ""
    else:
        hand = parts[hand_idx]
        obj = parts[hand_idx + 1]
    return {
        "relative_path": rel,
        "hand": hand,
        "object": obj,
        "episode": episode_root.name,
    }


def localize_shared_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    parts = p.parts
    if len(parts) >= 5 and parts[0] == "/" and parts[1] == "home" and parts[3] == "shared_data":
        return Path.home() / "shared_data" / Path(*parts[4:])
    return p


def load_world_pose_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"GoTrack world pose records not found: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"Expected list in {path}")
    valid = [
        r for r in records
        if isinstance(r, dict) and r.get("pose_world") is not None and "frame_index" in r
    ]
    if not valid:
        raise ValueError(f"No pose_world records found in {path}")
    return valid


def load_synced_qpos(episode_root: Path) -> np.ndarray:
    arm_path = episode_root / "arm" / "state.npy"
    hand_path = episode_root / "hand" / "state.npy"
    if not arm_path.exists() or not hand_path.exists():
        raise FileNotFoundError(
            "Synced qpos missing. Expected episode/arm/state.npy and "
            "episode/hand/state.npy."
        )
    arm = np.asarray(np.load(arm_path), dtype=np.float64)
    hand = np.asarray(np.load(hand_path), dtype=np.float64)
    if arm.ndim != 2 or hand.ndim != 2:
        raise ValueError(f"Expected 2D arm/hand qpos arrays, got {arm.shape}, {hand.shape}")
    n = min(len(arm), len(hand))
    if n <= 0:
        raise ValueError("Empty arm/hand qpos arrays")
    return np.concatenate([arm[:n], hand[:n]], axis=1)


def load_timestamps(episode_root: Path) -> np.ndarray:
    path = episode_root / "raw" / "timestamps" / "timestamp.npy"
    if not path.exists():
        return np.zeros((0,), dtype=np.float64)
    return np.asarray(np.load(path), dtype=np.float64).reshape(-1)


def load_gotrack_world_to_robot_base(episode_root: Path) -> np.ndarray:
    """Return T_robot_base_gotrack_world.

    AutoDex calibration stores C2R.npy as the robot frame expressed in the
    camera-calibration world. GoTrack poses are in that camera world, while the
    URDF FK is in robot base. Therefore panel-world assets should use inv(C2R).
    """
    path = episode_root / "C2R.npy"
    if not path.exists():
        return np.eye(4, dtype=np.float64)
    c2r = np.asarray(np.load(path), dtype=np.float64)
    if c2r.shape != (4, 4):
        raise ValueError(f"Expected 4x4 C2R.npy, got {c2r.shape}: {path}")
    return np.linalg.inv(c2r)


def build_camera_views(
    *,
    episode_root: Path,
    gotrack_world_to_robot_base: np.ndarray,
) -> Dict[str, Any]:
    """Return browser-facing camera poses in the GLB robot-base frame.

    AutoDex camera extrinsics are OpenCV-style camera-from-calibration-world
    transforms. The exported GLB uses robot_base as its world frame. Three.js
    cameras look along local -Z with local +Y as up, so each OpenCV camera pose
    is converted with a fixed OpenCV-to-Three camera-axis transform.
    """
    intr_path = episode_root / "cam_param" / "intrinsics.json"
    extr_path = episode_root / "cam_param" / "extrinsics.json"
    if not intr_path.exists() or not extr_path.exists():
        return {
            "version": 1,
            "coordinate_frame": "robot_base",
            "source": "autodex_cam_param",
            "views": {},
            "warning": "Missing cam_param/intrinsics.json or cam_param/extrinsics.json",
        }

    intrinsics = load_json(intr_path, default={})
    extrinsics = load_json(extr_path, default={})
    cv_camera_to_three_camera = np.eye(4, dtype=np.float64)
    cv_camera_to_three_camera[:3, :3] = np.diag([1.0, -1.0, -1.0])

    views: Dict[str, Any] = {}
    for serial in sorted(set(intrinsics) & set(extrinsics)):
        intr = intrinsics.get(serial) or {}
        k = (
            intr.get("intrinsics_undistort")
            or intr.get("K_undist")
            or intr.get("intrinsics")
            or intr.get("K")
            or intr.get("original_intrinsics")
            or intr.get("K_orig")
        )
        if k is None:
            continue
        k_arr = np.asarray(k, dtype=np.float64).reshape(3, 3)
        height = int(intr.get("height") or intr.get("dist_height") or 1536)
        width = int(intr.get("width") or 2048)
        fx = float(k_arr[0, 0])
        fy = float(k_arr[1, 1])
        cx = float(k_arr[0, 2])
        cy = float(k_arr[1, 2])
        if fy <= 0 or width <= 0 or height <= 0:
            continue

        camera_from_gotrack_world = coerce_homogeneous_matrix(extrinsics[serial])
        robot_base_from_cv_camera = gotrack_world_to_robot_base @ np.linalg.inv(camera_from_gotrack_world)
        robot_base_from_three_camera = robot_base_from_cv_camera @ cv_camera_to_three_camera
        rot_cv = robot_base_from_cv_camera[:3, :3]
        position = robot_base_from_cv_camera[:3, 3]
        forward = normalize_vector(rot_cv @ np.array([0.0, 0.0, 1.0], dtype=np.float64))
        up = normalize_vector(rot_cv @ np.array([0.0, -1.0, 0.0], dtype=np.float64))
        fov_y_deg = float(np.degrees(2.0 * np.arctan(float(height) / (2.0 * fy))))

        views[str(serial)] = {
            "serial": str(serial),
            "width": width,
            "height": height,
            "intrinsics": {
                "fx": fx,
                "fy": fy,
                "cx": cx,
                "cy": cy,
                "matrix": matrix_to_nested_float_list(k_arr),
                "source": "intrinsics_undistort",
            },
            "opencv_camera_from_gotrack_world": matrix_to_nested_list(camera_from_gotrack_world),
            "robot_base_from_opencv_camera": matrix_to_nested_list(robot_base_from_cv_camera),
            "robot_base_from_three_camera": matrix_to_nested_list(robot_base_from_three_camera),
            "three": {
                "position": [float(v) for v in position.tolist()],
                "forward": [float(v) for v in forward.tolist()],
                "up": [float(v) for v in up.tolist()],
                "fov_y_deg": fov_y_deg,
            },
        }

    return {
        "version": 1,
        "coordinate_frame": "robot_base",
        "source": "autodex_cam_param",
        "convention": {
            "opencv_extrinsic": "camera_from_calibration_world",
            "three_camera": "position/forward/up in robot_base; Three.js camera looks along local -Z",
        },
        "views": views,
    }


def coerce_homogeneous_matrix(value: Any) -> np.ndarray:
    if isinstance(value, dict):
        value = value.get("extrinsics_undistort", value.get("extrinsics", value.get("matrix")))
    arr = np.asarray(value, dtype=np.float64)
    if arr.shape == (3, 4):
        out = np.eye(4, dtype=np.float64)
        out[:3, :] = arr
        return out
    if arr.shape == (4, 4):
        return arr
    if arr.size == 12:
        out = np.eye(4, dtype=np.float64)
        out[:3, :] = arr.reshape(3, 4)
        return out
    if arr.size == 16:
        return arr.reshape(4, 4)
    raise ValueError(f"Cannot coerce extrinsic matrix with shape {arr.shape}")


def normalize_vector(value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return arr
    return arr / norm


def select_frame_indices(
    *,
    records: Iterable[Dict[str, Any]],
    qpos_len: int,
    timestamp_len: int,
    stride: int,
    max_frames: Optional[int],
) -> List[int]:
    max_len = qpos_len
    if timestamp_len > 0:
        max_len = min(max_len, timestamp_len)
    indices = sorted({
        int(r["frame_index"])
        for r in records
        if 0 <= int(r.get("frame_index", -1)) < max_len
    })
    indices = indices[:: max(1, stride)]
    if max_frames is not None:
        indices = indices[: max(0, int(max_frames))]
    return indices


def resolve_urdf_path(hand: str, robot_asset_root: Path) -> Path:
    if hand == "allegro":
        path = robot_asset_root / "allegro_description" / "xarm_allegro.urdf"
    elif hand == "inspire":
        path = robot_asset_root / "inspire_description" / "xarm_inspire.urdf"
    else:
        raise ValueError(f"Unsupported or unknown hand type: {hand!r}")
    if not path.exists():
        raise FileNotFoundError(f"URDF not found: {path}")
    return path


def resolve_object_mesh_path(
    *,
    summary: Dict[str, Any],
    object_name: str,
    object_roots: Tuple[Path, ...],
) -> Path:
    summary_mesh = summary.get("mesh_path")
    if summary_mesh:
        p = localize_shared_path(summary_mesh)
        if p.exists():
            return p

    names = [object_name]
    if summary_mesh:
        names.append(Path(str(summary_mesh)).stem)

    for name in [n for n in names if n]:
        for root in object_roots:
            for subdir in ("raw_mesh", "visual_mesh"):
                p = root / name / subdir / f"{name}.obj"
                if p.exists():
                    return p
    raise FileNotFoundError(
        f"Object mesh not found for object={object_name!r}, summary mesh={summary_mesh!r}"
    )


def load_object_mesh(mesh_path: Path, trimesh_module):
    repo_root = Path(__file__).resolve().parents[2]
    gotrack_root = repo_root / "autodex" / "perception" / "thirdparty" / "MV-GoTrack"
    if str(gotrack_root) not in sys.path:
        sys.path.insert(0, str(gotrack_root))
    try:
        from utils.mesh_io import load_trimesh_with_merged_texture  # type: ignore

        mesh = load_trimesh_with_merged_texture(mesh_path, process=False)
    except Exception:
        loaded = trimesh_module.load(str(mesh_path), process=False)
        if isinstance(loaded, trimesh_module.Scene):
            dumped = list(loaded.dump())
            if not dumped:
                raise ValueError(f"Object mesh scene has no geometry: {mesh_path}")
            mesh = trimesh_module.util.concatenate(dumped)
        else:
            mesh = loaded
    if not hasattr(mesh, "vertices") or not hasattr(mesh, "faces"):
        raise ValueError(f"Object mesh did not load as a renderable mesh: {mesh_path}")
    return mesh


def qpos_to_urdf_cfg(qpos_row: np.ndarray, n_actuated: int) -> np.ndarray:
    qpos_row = np.asarray(qpos_row, dtype=np.float64).reshape(-1)
    cfg = np.zeros(int(n_actuated), dtype=np.float64)
    n = min(len(cfg), len(qpos_row))
    cfg[:n] = qpos_row[:n]
    return cfg


def pose_from_record(record: Dict[str, Any]) -> np.ndarray:
    pose = record.get("pose_world")
    if pose is not None:
        arr = np.asarray(pose, dtype=np.float64)
        if arr.shape == (4, 4):
            return arr
        if arr.size == 16:
            return arr.reshape(4, 4)
    rot = np.asarray(record.get("rotation_world"), dtype=np.float64).reshape(3, 3)
    trans = np.asarray(record.get("translation_world_m"), dtype=np.float64).reshape(3)
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = rot
    out[:3, 3] = trans
    return out


def robot_geometry_poses(urdf, qpos: np.ndarray, geometry_names: Iterable[str]) -> Dict[str, Any]:
    urdf.update_cfg(np.asarray(qpos, dtype=np.float64))
    poses: Dict[str, Any] = {}
    for name in geometry_names:
        poses[robot_node_name(name)] = matrix_to_nested_list(urdf.scene.graph.get(name)[0])
    return poses


def robot_node_name(urdf_geometry_name: str) -> str:
    return f"robot::{urdf_geometry_name}"


def compact_tracking_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    keep = (
        "num_triangulated_anchors",
        "num_inlier_anchors",
        "mean_triangulation_residual_mm",
        "mean_anchor_fit_residual_mm",
        "max_anchor_fit_residual_mm",
        "tracking_init_source",
    )
    return {k: record.get(k) for k in keep if k in record}


def timestamp_for_frame(timestamps: np.ndarray, frame_index: int, record: Dict[str, Any]) -> Optional[float]:
    if 0 <= frame_index < len(timestamps):
        return float(timestamps[frame_index])
    if record.get("time_sec") is not None:
        return float(record["time_sec"])
    return None


def choose_preview_frame(frame_indices: List[int], preview_frame: str) -> int:
    if not frame_indices:
        raise ValueError("No frame indices")
    if preview_frame == "first":
        return frame_indices[0]
    if preview_frame == "last":
        return frame_indices[-1]
    try:
        wanted = int(preview_frame)
    except ValueError:
        return frame_indices[len(frame_indices) // 2]
    return min(frame_indices, key=lambda idx: abs(idx - wanted))


def export_static_scene_glb(
    path: Path,
    *,
    urdf,
    robot_geometry_names: Iterable[str],
    qpos: np.ndarray,
    object_mesh,
    object_pose: np.ndarray,
    trimesh,
) -> None:
    urdf.update_cfg(np.asarray(qpos, dtype=np.float64))
    scene = trimesh.Scene(base_frame="gotrack_world")
    for name in robot_geometry_names:
        geom = urdf.scene.geometry[name].copy()
        transform = np.asarray(urdf.scene.graph.get(name)[0], dtype=np.float64)
        safe_name = robot_node_name(name)
        scene.add_geometry(
            geom,
            geom_name=safe_name,
            node_name=safe_name,
            transform=transform,
        )

    scene.add_geometry(
        object_mesh.copy(),
        geom_name="object::mesh",
        node_name="object::mesh",
        transform=np.asarray(object_pose, dtype=np.float64),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    scene.export(str(path))


def bake_animation_into_glb(
    *,
    source_glb_path: Path,
    output_glb_path: Path,
    trajectory_frames: List[Dict[str, Any]],
    animated_node_names: List[str],
) -> None:
    if not trajectory_frames:
        raise ValueError("Cannot bake GLB animation without trajectory frames")

    gltf, bin_payload = read_glb(source_glb_path)
    nodes = gltf.get("nodes") or []
    node_index_by_name = {
        str(node.get("name")): idx
        for idx, node in enumerate(nodes)
        if node.get("name") is not None
    }
    missing = [name for name in animated_node_names if name not in node_index_by_name]
    if missing:
        raise ValueError(
            "Animated node names are missing from the source GLB: "
            + ", ".join(missing[:8])
            + ("..." if len(missing) > 8 else "")
        )

    times = animation_times_for_frames(trajectory_frames)
    buffer = bytearray(bin_payload)
    gltf.setdefault("buffers", [{"byteLength": len(buffer)}])
    gltf.setdefault("bufferViews", [])
    gltf.setdefault("accessors", [])

    time_accessor = append_accessor(
        gltf,
        buffer,
        np.asarray(times, dtype=np.float32).reshape(-1, 1),
        accessor_type="SCALAR",
    )

    samplers: List[Dict[str, Any]] = []
    channels: List[Dict[str, Any]] = []
    for node_name in animated_node_names:
        node_index = node_index_by_name[node_name]
        matrices = frame_matrices_for_node(trajectory_frames, node_name)
        translations, rotations, scales = decompose_transform_sequence(matrices)

        first_t = translations[0].astype(float).tolist()
        first_r = rotations[0].astype(float).tolist()
        first_s = scales[0].astype(float).tolist()
        nodes[node_index].pop("matrix", None)
        nodes[node_index]["translation"] = first_t
        nodes[node_index]["rotation"] = first_r
        nodes[node_index]["scale"] = first_s

        for path_name, values, accessor_type in (
            ("translation", translations, "VEC3"),
            ("rotation", rotations, "VEC4"),
            ("scale", scales, "VEC3"),
        ):
            output_accessor = append_accessor(
                gltf,
                buffer,
                np.asarray(values, dtype=np.float32),
                accessor_type=accessor_type,
            )
            sampler_index = len(samplers)
            samplers.append(
                {
                    "input": time_accessor,
                    "interpolation": "LINEAR",
                    "output": output_accessor,
                }
            )
            channels.append(
                {
                    "sampler": sampler_index,
                    "target": {
                        "node": node_index,
                        "path": path_name,
                    },
                }
            )

    gltf["animations"] = [
        {
            "name": "AutoDex episode motion",
            "samplers": samplers,
            "channels": channels,
        }
    ]
    gltf["buffers"][0]["byteLength"] = len(buffer)
    write_glb(output_glb_path, gltf, bytes(buffer))


def read_glb(path: Path) -> Tuple[Dict[str, Any], bytes]:
    payload = path.read_bytes()
    if len(payload) < 20 or payload[:4] != b"glTF":
        raise ValueError(f"Not a binary glTF file: {path}")
    version, total_length = struct.unpack_from("<II", payload, 4)
    if version != 2:
        raise ValueError(f"Expected GLB version 2, got {version}: {path}")
    if total_length != len(payload):
        raise ValueError(f"GLB length mismatch in {path}: header={total_length}, actual={len(payload)}")

    offset = 12
    gltf: Optional[Dict[str, Any]] = None
    bin_payload = b""
    while offset + 8 <= len(payload):
        chunk_length, chunk_type = struct.unpack_from("<II", payload, offset)
        offset += 8
        chunk = payload[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == 0x4E4F534A:  # JSON
            gltf = json.loads(chunk.rstrip(b" \t\r\n\0").decode("utf-8"))
        elif chunk_type == 0x004E4942:  # BIN
            bin_payload = chunk
    if gltf is None:
        raise ValueError(f"GLB has no JSON chunk: {path}")
    return gltf, bin_payload


def write_glb(path: Path, gltf: Dict[str, Any], bin_payload: bytes) -> None:
    json_payload = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_payload = pad_bytes(json_payload, b" ")
    bin_payload = pad_bytes(bin_payload, b"\0")
    total_length = 12 + 8 + len(json_payload) + 8 + len(bin_payload)
    header = struct.pack("<4sII", b"glTF", 2, total_length)
    json_header = struct.pack("<II", len(json_payload), 0x4E4F534A)
    bin_header = struct.pack("<II", len(bin_payload), 0x004E4942)
    path.write_bytes(header + json_header + json_payload + bin_header + bin_payload)


def pad_bytes(payload: bytes | bytearray, pad_byte: bytes) -> bytes:
    out = bytes(payload)
    remainder = len(out) % 4
    if remainder:
        out += pad_byte * (4 - remainder)
    return out


def append_accessor(
    gltf: Dict[str, Any],
    buffer: bytearray,
    values: np.ndarray,
    *,
    accessor_type: str,
) -> int:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if accessor_type == "SCALAR":
        expected_width = 1
    elif accessor_type == "VEC3":
        expected_width = 3
    elif accessor_type == "VEC4":
        expected_width = 4
    else:
        raise ValueError(f"Unsupported accessor type: {accessor_type}")
    if values.shape[1] != expected_width:
        raise ValueError(f"{accessor_type} accessor expected width {expected_width}, got {values.shape}")

    while len(buffer) % 4:
        buffer.append(0)
    byte_offset = len(buffer)
    raw = np.ascontiguousarray(values, dtype=np.float32).tobytes()
    buffer.extend(raw)
    buffer_view_index = len(gltf["bufferViews"])
    gltf["bufferViews"].append(
        {
            "buffer": 0,
            "byteOffset": byte_offset,
            "byteLength": len(raw),
        }
    )

    accessor: Dict[str, Any] = {
        "bufferView": buffer_view_index,
        "componentType": 5126,
        "count": int(values.shape[0]),
        "type": accessor_type,
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
    }
    accessor_index = len(gltf["accessors"])
    gltf["accessors"].append(accessor)
    return accessor_index


def animation_times_for_frames(trajectory_frames: List[Dict[str, Any]]) -> np.ndarray:
    raw: List[float] = []
    for idx, frame in enumerate(trajectory_frames):
        value = frame.get("time_sec")
        try:
            t = float(value)
        except (TypeError, ValueError):
            t = idx / 30.0
        if not np.isfinite(t):
            t = idx / 30.0
        raw.append(t)
    t0 = raw[0] if raw else 0.0
    times = np.asarray([max(0.0, t - t0) for t in raw], dtype=np.float64)
    for idx in range(1, len(times)):
        if times[idx] <= times[idx - 1]:
            times[idx] = times[idx - 1] + (1.0 / 30.0)
    return times.astype(np.float32)


def frame_matrices_for_node(
    trajectory_frames: List[Dict[str, Any]],
    node_name: str,
) -> np.ndarray:
    matrices: List[np.ndarray] = []
    for frame in trajectory_frames:
        if node_name == "object::mesh":
            matrix = frame.get("object_pose_world")
        else:
            matrix = (frame.get("robot_geometry_poses_world") or {}).get(node_name)
        if matrix is None:
            raise ValueError(f"Missing trajectory matrix for {node_name} at frame {frame.get('frame_index')}")
        matrices.append(np.asarray(matrix, dtype=np.float64).reshape(4, 4))
    return np.stack(matrices, axis=0)


def decompose_transform_sequence(matrices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    translations: List[np.ndarray] = []
    rotations: List[np.ndarray] = []
    scales: List[np.ndarray] = []
    previous_quaternion: Optional[np.ndarray] = None
    for matrix in matrices:
        translation, quaternion, scale = decompose_matrix_to_trs(matrix)
        if previous_quaternion is not None and float(np.dot(previous_quaternion, quaternion)) < 0.0:
            quaternion = -quaternion
        previous_quaternion = quaternion
        translations.append(translation)
        rotations.append(quaternion)
        scales.append(scale)
    return (
        np.asarray(translations, dtype=np.float32),
        np.asarray(rotations, dtype=np.float32),
        np.asarray(scales, dtype=np.float32),
    )


def decompose_matrix_to_trs(matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    translation = matrix[:3, 3].copy()
    linear = matrix[:3, :3].copy()
    scale = np.linalg.norm(linear, axis=0)
    scale[scale == 0.0] = 1.0
    rotation = linear / scale.reshape(1, 3)
    if np.linalg.det(rotation) < 0:
        scale[0] *= -1.0
        rotation[:, 0] *= -1.0
    quaternion = quaternion_from_rotation_matrix(rotation)
    return translation, quaternion, scale


def quaternion_from_rotation_matrix(rotation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (rotation[2, 1] - rotation[1, 2]) / s
        qy = (rotation[0, 2] - rotation[2, 0]) / s
        qz = (rotation[1, 0] - rotation[0, 1]) / s
    elif rotation[0, 0] > rotation[1, 1] and rotation[0, 0] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
        qw = (rotation[2, 1] - rotation[1, 2]) / s
        qx = 0.25 * s
        qy = (rotation[0, 1] + rotation[1, 0]) / s
        qz = (rotation[0, 2] + rotation[2, 0]) / s
    elif rotation[1, 1] > rotation[2, 2]:
        s = np.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
        qw = (rotation[0, 2] - rotation[2, 0]) / s
        qx = (rotation[0, 1] + rotation[1, 0]) / s
        qy = 0.25 * s
        qz = (rotation[1, 2] + rotation[2, 1]) / s
    else:
        s = np.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
        qw = (rotation[1, 0] - rotation[0, 1]) / s
        qx = (rotation[0, 2] + rotation[2, 0]) / s
        qy = (rotation[1, 2] + rotation[2, 1]) / s
        qz = 0.25 * s
    quaternion = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if norm == 0.0:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return quaternion / norm


def matrix_to_nested_list(matrix: Any) -> List[List[float]]:
    arr = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    return [[float(v) for v in row] for row in arr.tolist()]


def matrix_to_nested_float_list(matrix: Any) -> List[List[float]]:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {arr.shape}")
    return [[float(v) for v in row] for row in arr.tolist()]
