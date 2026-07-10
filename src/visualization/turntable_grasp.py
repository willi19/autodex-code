#!/usr/bin/env python3
"""
Turntable rendering of grasp candidates using Open3D offscreen renderer.

Usage:
    # Single grasp
    python turntable_grasp.py --version revalidate --obj soap_dispenser --scene shelf/1/11

    # Top 100 grasps from setcover order
    python turntable_grasp.py --version revalidate --obj soap_dispenser --top 100

    # All 98 objects, top 100 each (reads setcover from order/)
    python turntable_grasp.py --batch-all --top 100

    # Custom render settings
    python turntable_grasp.py --version revalidate --obj soap_dispenser --scene shelf/1/11 --frames 60 --fps 30 --width 1080 --height 1080
"""

import argparse
import json
from tqdm import tqdm
import os
import sys
import tempfile
import shutil
import subprocess

import numpy as np
import open3d as o3d
import trimesh

from autodex.utils.conversion import cart2se3
from autodex.utils.path import obj_path, urdf_path
from autodex.utils.path import repo_dir, get_scene_dir
from paradex.visualization.robot import RobotModule


COLOR_ROBOT = np.array([153, 128, 224]) / 255.0
COLOR_OBJECT = np.array([180, 180, 180]) / 255.0

HAND_URDF = {
    "allegro": os.path.join(urdf_path, "allegro_hand_description_right.urdf"),
    "inspire": os.path.join(urdf_path, "..", "inspire_description", "inspire_hand_right.urdf"),
    "inspire_f1": os.path.join(
        repo_dir, "autodex", "planner", "src", "curobo", "content", "assets",
        "robot", "inspire_f1_description", "inspire_f1_hand_right.urdf",
    ),
    "inspire_left": os.path.join(
        repo_dir, "autodex", "planner", "src", "curobo", "content", "assets",
        "robot", "inspire_description", "inspire_hand_left.urdf",
    ),
    "inspire_f1_left": os.path.join(
        repo_dir, "autodex", "planner", "src", "curobo", "content", "assets",
        "robot", "inspire_f1_left_description", "inspire_f1_hand_left.urdf",
    ),
}

# These globals are set in main() based on --hand
SELECTED_DIR = None
ORDER_ROOT = None
CURRENT_HAND = "allegro"
OBJ_ROOT = None  # set in main() — overrides default obj_path


def get_all_objects() -> list:
    """Get all object names from selected_100/."""
    if not os.path.isdir(SELECTED_DIR):
        print(f"Error: selected_100 not found: {SELECTED_DIR}")
        sys.exit(1)
    objects = []
    for obj in sorted(os.listdir(SELECTED_DIR)):
        if os.path.isdir(os.path.join(SELECTED_DIR, obj)):
            objects.append(obj)
    return objects


def _find_setcover_order(obj_name: str) -> str | None:
    """Find setcover_order.json for an object."""
    path = os.path.join(ORDER_ROOT, obj_name, "setcover_order.json")
    if os.path.exists(path):
        return path
    return None


def load_selected_grasps(obj_name: str, top_n: int) -> list:
    """Load grasps in setcover order, filtered to what exists in selected_100/.
    Returns list of (scene_type, scene_id, grasp_name) tuples, up to top_n."""
    # Build set of available grasps in selected_100
    root = os.path.join(SELECTED_DIR, obj_name)
    if not os.path.exists(root):
        print(f"Warning: object not found in selected_100: {obj_name}")
        return []

    available = set()
    for scene_type in os.listdir(root):
        st_path = os.path.join(root, scene_type)
        if not os.path.isdir(st_path):
            continue
        for scene_id in os.listdir(st_path):
            si_path = os.path.join(st_path, scene_id)
            if not os.path.isdir(si_path):
                continue
            for grasp_name in os.listdir(si_path):
                gp = os.path.join(si_path, grasp_name)
                if os.path.isdir(gp) and os.path.exists(os.path.join(gp, "wrist_se3.npy")):
                    available.add((scene_type, scene_id, grasp_name))

    # Load setcover order and filter to available
    order_path = _find_setcover_order(obj_name)
    if order_path is not None:
        with open(order_path) as f:
            ordered_list = json.load(f)
        grasps = []
        for item in ordered_list:
            # Format: [obj_name, scene_type, scene_id, grasp_name, idx]
            key = (item[1], item[2], item[3])
            if key in available:
                grasps.append(key)
                if len(grasps) >= top_n:
                    return grasps
        return grasps
    else:
        # Fallback: directory walk (no order info)
        print(f"Warning: no setcover order for {obj_name}, using directory order")
        return list(available)[:top_n]


def list_all_grasps(version: str, obj_name: str, scene_type_filter: str = None) -> list:
    """Walk candidates directory to find all grasps.
    Returns list of (scene_type, scene_id, grasp_name) tuples."""
    candidate_root = os.path.join(repo_dir, "candidates", CURRENT_HAND)
    root = os.path.join(candidate_root, version, obj_name)
    if not os.path.exists(root):
        print(f"Error: candidates path not found: {root}")
        sys.exit(1)

    grasps = []
    scene_types = [scene_type_filter] if scene_type_filter else sorted(os.listdir(root))
    for scene_type in scene_types:
        st_path = os.path.join(root, scene_type)
        if not os.path.isdir(st_path):
            continue
        for scene_id in sorted(os.listdir(st_path)):
            si_path = os.path.join(st_path, scene_id)
            if not os.path.isdir(si_path):
                continue
            for grasp_name in sorted(os.listdir(si_path), key=lambda x: int(x) if x.isdigit() else x):
                gp = os.path.join(si_path, grasp_name)
                if os.path.isdir(gp) and os.path.exists(os.path.join(gp, "wrist_se3.npy")):
                    grasps.append((scene_type, scene_id, grasp_name))
    return grasps


def trimesh_to_o3d(mesh: trimesh.Trimesh, color=None) -> o3d.geometry.TriangleMesh:
    """Convert a trimesh.Trimesh to open3d.geometry.TriangleMesh.
    If color is given, paint uniform. Otherwise bake texture/vertex colors."""
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces)
    o3d_mesh.compute_vertex_normals()
    if color is not None:
        o3d_mesh.paint_uniform_color(color)
    else:
        # Bake texture to vertex colors (handles UV-mapped textures)
        try:
            color_visual = mesh.visual.to_color()
            vc = np.array(color_visual.vertex_colors)[:, :3] / 255.0
            o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(vc)
        except Exception:
            o3d_mesh.paint_uniform_color([0.7, 0.7, 0.7])
    return o3d_mesh


def _pick_standing_pose(obj_name: str, root: str) -> np.ndarray:
    """Pick the scene JSON with the highest z translation (standing pose).

    Prefer the ``table`` scenes. Some hands (e.g. inspire) have no ``table``
    scene type -- the object's resting pose is identical across scene types
    (they differ only in obstacles), so fall back to every available scene type.
    """
    table_dir = get_scene_dir(CURRENT_HAND, obj_name, "table")
    if os.path.isdir(table_dir):
        scene_dirs = [table_dir]
    else:
        obj_root = get_scene_dir(CURRENT_HAND, obj_name)
        if not os.path.isdir(obj_root):
            return np.eye(4)
        scene_dirs = [os.path.join(obj_root, st) for st in sorted(os.listdir(obj_root))
                      if os.path.isdir(os.path.join(obj_root, st))]
    best_z = -np.inf
    best_pose = np.eye(4)
    for sd in scene_dirs:
        for fn in os.listdir(sd):
            if not fn.endswith(".json"):
                continue
            with open(os.path.join(sd, fn)) as f:
                cfg = json.load(f)
            pose = cfg['scene']['mesh']['target']['pose']
            if pose[2] > best_z:
                best_z = pose[2]
                best_pose = cart2se3(pose)
    return best_pose


def load_object_mesh(obj_name: str) -> tuple:
    """Load object mesh and its pose from scene JSON. Returns (trimesh, 4x4 pose)."""
    root = OBJ_ROOT if OBJ_ROOT is not None else obj_path
    obj_pose = _pick_standing_pose(obj_name, root)
    mesh_path = os.path.join(root, obj_name, "raw_mesh", f"{obj_name}.obj")
    mesh = trimesh.load(mesh_path, force="mesh")
    return mesh, obj_pose


def find_grasp_path(obj_name: str, scene_type: str,
                    scene_id: str, grasp_name: str) -> str:
    """Find grasp data path from selected_100/."""
    path = os.path.join(SELECTED_DIR, obj_name, scene_type, scene_id, grasp_name)
    if os.path.exists(path):
        return path
    raise FileNotFoundError(f"Grasp not found: {path}")


def load_robot_at_grasp(obj_name: str, scene_type: str,
                        scene_id: str, grasp_name: str, obj_pose: np.ndarray) -> trimesh.Trimesh:
    """Load robot hand mesh at the grasp pose. Returns combined trimesh in world frame."""
    grasp_path = find_grasp_path(obj_name, scene_type, scene_id, grasp_name)

    wrist_se3 = np.load(os.path.join(grasp_path, "wrist_se3.npy"))
    grasp_pose = np.load(os.path.join(grasp_path, "grasp_pose.npy"))

    robot = RobotModule(HAND_URDF[CURRENT_HAND])

    joint_angles = grasp_pose.flatten()[:robot.num_joints]
    cfg = {name: angle for name, angle in zip(robot.joint_names, joint_angles)}
    robot.update_cfg(cfg)

    robot_mesh = robot.get_robot_mesh(collision_geometry=False)
    world_T = obj_pose @ wrist_se3
    robot_mesh.apply_transform(world_T)
    return robot_mesh


def compute_auto_camera(combined_mesh: trimesh.Trimesh, fov_deg: float,
                        aspect_ratio: float, elevation_deg: float = 25.0,
                        padding: float = 1.3) -> tuple:
    """Compute camera orbit params that guarantee the full scene is visible."""
    bsphere = combined_mesh.bounding_sphere
    center = bsphere.primitive.center
    sphere_radius = bsphere.primitive.radius

    vfov = np.radians(fov_deg)
    hfov = 2.0 * np.arctan(np.tan(vfov / 2.0) * aspect_ratio)
    effective_half_fov = min(vfov, hfov) / 2.0

    cam_dist = (sphere_radius * padding) / np.sin(effective_half_fov)
    return center, cam_dist, elevation_deg


def compute_turntable_camera(center: np.ndarray, cam_dist: float,
                             elevation_deg: float, angle_rad: float) -> tuple:
    """Compute camera eye/lookat/up for a turntable frame."""
    elev_rad = np.radians(elevation_deg)
    horiz_r = cam_dist * np.cos(elev_rad)
    vert_h = cam_dist * np.sin(elev_rad)

    eye = np.array([
        center[0] + horiz_r * np.cos(angle_rad),
        center[1] + horiz_r * np.sin(angle_rad),
        center[2] + vert_h,
    ])
    lookat = center.copy()
    up = np.array([0.0, 0.0, 1.0])
    return eye, lookat, up


_renderer = None
_renderer_size = (None, None)


def get_renderer(width, height):
    """Get or create a persistent OffscreenRenderer (reused across grasps)."""
    global _renderer, _renderer_size
    if _renderer is None or _renderer_size != (width, height):
        _renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        _renderer_size = (width, height)
    return _renderer


def render_turntable(obj_mesh_o3d, robot_mesh_o3d, center, cam_dist,
                     elevation_deg, fov_deg, n_frames, width, height_px,
                     output_dir):
    """Render turntable frames using Open3D offscreen renderer."""
    renderer = get_renderer(width, height_px)

    # Clear previous geometry
    renderer.scene.clear_geometry()

    mat_obj = o3d.visualization.rendering.MaterialRecord()
    mat_obj.shader = "defaultLit"

    mat_robot = o3d.visualization.rendering.MaterialRecord()
    mat_robot.shader = "defaultLit"

    renderer.scene.add_geometry("object", obj_mesh_o3d, mat_obj)
    renderer.scene.add_geometry("robot", robot_mesh_o3d, mat_robot)

    renderer.scene.scene.set_sun_light([0.5, -0.5, -1.0], [1.0, 1.0, 1.0], 60000)
    renderer.scene.scene.enable_sun_light(True)
    renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])

    angles = np.linspace(0, 2 * np.pi, n_frames, endpoint=False)

    for i, angle in enumerate(angles):
        eye, lookat, up = compute_turntable_camera(center, cam_dist, elevation_deg, angle)
        renderer.setup_camera(fov_deg, lookat, eye, up)

        img = renderer.render_to_image()
        frame_path = os.path.join(output_dir, f"frame_{i:04d}.png")
        o3d.io.write_image(frame_path, img)

        pass


def frames_to_video(frame_dir: str, output_path: str, fps: int):
    """Encode PNG frames to video using ffmpeg."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "warning",
        "-framerate", str(fps),
        "-i", os.path.join(frame_dir, "frame_%04d.png"),
        "-c:v", "libx264",
        "-preset", "slow",
        "-crf", "18",
        "-pix_fmt", "yuv420p",
        output_path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"ffmpeg error: {result.stderr}")


def render_single_grasp(obj_name, scene_type, scene_id, grasp_name,
                        output_path, args, obj_mesh=None, obj_pose=None):
    """Render a single grasp turntable video. Returns True on success."""
    try:
        if obj_mesh is None or obj_pose is None:
            obj_mesh, obj_pose = load_object_mesh(obj_name)

        robot_mesh = load_robot_at_grasp(obj_name, scene_type, scene_id, grasp_name, obj_pose)

        obj_mesh_world = obj_mesh.copy()
        obj_mesh_world.apply_transform(obj_pose)

        obj_mesh_o3d = trimesh_to_o3d(obj_mesh_world)
        robot_mesh_o3d = trimesh_to_o3d(robot_mesh, color=COLOR_ROBOT)

        combined = trimesh.util.concatenate([obj_mesh_world, robot_mesh])
        aspect_ratio = args.width / args.height
        center, cam_dist, elevation_deg = compute_auto_camera(
            combined, fov_deg=args.fov, aspect_ratio=aspect_ratio,
            elevation_deg=args.elevation, padding=args.padding,
        )

        temp_dir = tempfile.mkdtemp(prefix="turntable_")
        render_turntable(
            obj_mesh_o3d, robot_mesh_o3d, center, cam_dist, elevation_deg,
            args.fov, args.frames, args.width, args.height, temp_dir,
        )

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        frames_to_video(temp_dir, output_path, args.fps)
        shutil.rmtree(temp_dir)
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def render_object_only(obj_name, output_path, args):
    """Render a turntable of the object alone (no robot). Returns True on success."""
    try:
        obj_mesh, obj_pose = load_object_mesh(obj_name)
        obj_mesh_world = obj_mesh.copy()
        obj_mesh_world.apply_transform(obj_pose)
        obj_mesh_o3d = trimesh_to_o3d(obj_mesh_world)

        aspect_ratio = args.width / args.height
        center, cam_dist, elevation_deg = compute_auto_camera(
            obj_mesh_world, fov_deg=args.fov, aspect_ratio=aspect_ratio,
            elevation_deg=args.elevation, padding=args.padding,
        )

        renderer = get_renderer(args.width, args.height)
        renderer.scene.clear_geometry()
        mat = o3d.visualization.rendering.MaterialRecord()
        mat.shader = "defaultLit"
        renderer.scene.add_geometry("object", obj_mesh_o3d, mat)
        renderer.scene.scene.set_sun_light([0.5, -0.5, -1.0], [1.0, 1.0, 1.0], 60000)
        renderer.scene.scene.enable_sun_light(True)
        renderer.scene.set_background([1.0, 1.0, 1.0, 1.0])

        temp_dir = tempfile.mkdtemp(prefix="turntable_obj_")
        angles = np.linspace(0, 2 * np.pi, args.frames, endpoint=False)
        for i, angle in enumerate(angles):
            eye, lookat, up = compute_turntable_camera(center, cam_dist, elevation_deg, angle)
            renderer.setup_camera(args.fov, lookat, eye, up)
            img = renderer.render_to_image()
            o3d.io.write_image(os.path.join(temp_dir, f"frame_{i:04d}.png"), img)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        frames_to_video(temp_dir, output_path, args.fps)
        shutil.rmtree(temp_dir)
        return True
    except Exception as e:
        print(f"  Error: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Turntable rendering of grasp candidates")
    parser.add_argument("--hand", type=str, default="allegro", choices=["allegro", "inspire", "inspire_f1", "inspire_left", "inspire_f1_left"],
                        help="Hand type (default: allegro)")
    parser.add_argument("--obj-root", type=str, default=None,
                        help="Override object mesh root (default: autodex.utils.path.obj_path)")
    parser.add_argument("--version", type=str, default="v3", help="Candidate version (default: v3)")
    parser.add_argument("--obj", type=str, default=None, help="Object name (e.g., soap_dispenser)")
    parser.add_argument("--scene", type=str, default=None,
                        help="Single grasp: scene_type/scene_id/grasp_name (e.g., shelf/1/11)")
    parser.add_argument("--top", type=int, default=None,
                        help="Render top N grasps from setcover order")
    parser.add_argument("--batch-all", action="store_true",
                        help="Batch render all 98 objects (uses setcover, requires --top)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for batch mode (default: data/{hand})")
    parser.add_argument("--frames", type=int, default=60, help="Number of frames (default: 60)")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS (default: 30)")
    parser.add_argument("--width", type=int, default=960, help="Render width (default: 960)")
    parser.add_argument("--height", type=int, default=540, help="Render height (default: 540)")
    parser.add_argument("--fov", type=float, default=45.0, help="Vertical FOV in degrees (default: 45)")
    parser.add_argument("--elevation", type=float, default=25.0, help="Camera elevation angle in degrees (default: 25)")
    parser.add_argument("--padding", type=float, default=1.3, help="Camera padding multiplier (default: 1.3)")
    parser.add_argument("--output", type=str, default=None, help="Output video path (single grasp mode)")
    parser.add_argument("--parallel", type=int, default=1, help="Number of parallel workers for batch-all (default: 1)")
    parser.add_argument("--object-only", action="store_true",
                        help="Render object-only turntable (no robot). Requires --obj. Output: {output-dir}/{obj}/000/turntable.mp4")
    args = parser.parse_args()

    # Set globals based on --hand
    global SELECTED_DIR, ORDER_ROOT, CURRENT_HAND, OBJ_ROOT
    CURRENT_HAND = args.hand
    OBJ_ROOT = args.obj_root
    if args.output_dir is None:
        args.output_dir = os.path.join("data", args.hand)
    candidate_root = os.path.join(repo_dir, "candidates", args.hand)
    selected_dir = os.path.join(candidate_root, "selected_100")
    # Fall back to candidates/{hand}/{version} when selected_100 doesn't exist
    SELECTED_DIR = selected_dir if os.path.isdir(selected_dir) else os.path.join(candidate_root, args.version)
    ORDER_ROOT = os.path.join(repo_dir, "order", args.hand, args.version)

    # ---- Object-only mode ----
    if args.object_only:
        if args.obj is None:
            print("Error: --object-only requires --obj")
            sys.exit(1)
        output_dir = os.path.abspath(args.output_dir)
        video_path = os.path.join(output_dir, args.obj, "000", "turntable.mp4")
        if os.path.exists(video_path):
            print(f"Already exists: {video_path}")
            return
        ok = render_object_only(args.obj, video_path, args)
        if ok:
            print(f"Saved: {video_path}")
        else:
            sys.exit(1)
        return

    # ---- Batch all objects ----
    if args.batch_all:
        if args.top is None:
            print("Error: --batch-all requires --top N")
            sys.exit(1)

        all_objects = get_all_objects()
        output_dir = os.path.abspath(args.output_dir)
        os.makedirs(output_dir, exist_ok=True)

        if args.parallel > 1:
            # Parallel: spawn one subprocess per object
            from concurrent.futures import ProcessPoolExecutor, as_completed
            print(f"Batch rendering {len(all_objects)} objects with {args.parallel} workers")

            cmds = {}
            for obj_name in all_objects:
                cmd = [
                    sys.executable, __file__,
                    "--hand", args.hand, "--version", args.version,
                    "--obj", obj_name, "--top", str(args.top),
                    "--output-dir", output_dir,
                    "--frames", str(args.frames), "--fps", str(args.fps),
                    "--width", str(args.width), "--height", str(args.height),
                    "--fov", str(args.fov), "--elevation", str(args.elevation),
                    "--padding", str(args.padding),
                ]
                cmds[obj_name] = cmd

            procs = {}
            results = []
            pbar = tqdm(total=len(all_objects), desc="Objects")
            # Use subprocess directly with limited concurrency
            running = {}
            obj_list = list(cmds.items())
            idx = 0
            while idx < len(obj_list) or running:
                # Launch up to parallel workers
                while idx < len(obj_list) and len(running) < args.parallel:
                    obj_name, cmd = obj_list[idx]
                    proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    running[obj_name] = proc
                    idx += 1
                # Poll for completion
                import time
                time.sleep(0.5)
                done = []
                for name, proc in running.items():
                    if proc.poll() is not None:
                        done.append(name)
                        results.append((name, proc.returncode))
                        pbar.update(1)
                        pbar.set_description(f"Done: {name}")
                for name in done:
                    del running[name]
            pbar.close()

            ok = sum(1 for _, rc in results if rc == 0)
            failed_names = [n for n, rc in results if rc != 0]
            print(f"\nDone! {ok}/{len(all_objects)} objects succeeded")
            if failed_names:
                print(f"Failed: {failed_names}")
        else:
            print(f"Batch rendering {len(all_objects)} objects, top {args.top} grasps each")

            total_success = 0
            total_fail = 0

            # Count total videos
            all_grasps = []
            for obj_name in all_objects:
                grasps = load_selected_grasps(obj_name, args.top)
                all_grasps.append((obj_name, grasps))

            total_videos = sum(len(g) for _, g in all_grasps)
            pbar = tqdm(total=total_videos, desc="Rendering")

            for obj_name, grasps in all_grasps:
                if not grasps:
                    continue

                try:
                    obj_mesh, obj_pose = load_object_mesh(obj_name)
                except Exception as e:
                    total_fail += len(grasps)
                    pbar.update(len(grasps))
                    continue

                for gi, (scene_type, scene_id, grasp_name) in enumerate(grasps):
                    pbar.set_description(f"{obj_name} [{gi+1}/{len(grasps)}]")
                    episode_dir = os.path.join(output_dir, obj_name, f"{gi:03d}")
                    video_path = os.path.join(episode_dir, "turntable.mp4")
                    if os.path.exists(video_path):
                        total_success += 1
                        pbar.update(1)
                        continue

                    os.makedirs(episode_dir, exist_ok=True)
                    ok = render_single_grasp(
                        obj_name, scene_type, scene_id, grasp_name,
                        video_path, args, obj_mesh=obj_mesh, obj_pose=obj_pose,
                    )
                    if ok:
                        total_success += 1
                    else:
                        total_fail += 1
                    pbar.update(1)

            pbar.close()
            print(f"\nDone! {total_success} succeeded, {total_fail} failed")
            print(f"Output: {output_dir}")
        return

    # ---- Single object, top N grasps ----
    if args.top is not None:
        if args.obj is None:
            print("Error: --top requires --obj")
            sys.exit(1)

        grasps = load_selected_grasps(args.obj, args.top)
        if not grasps:
            print("No grasps found in setcover order")
            sys.exit(1)

        print(f"Rendering top {len(grasps)} grasps for {args.obj}")

        obj_mesh, obj_pose = load_object_mesh(args.obj)
        output_dir = os.path.abspath(args.output_dir)

        success, fail = 0, 0
        for gi, (scene_type, scene_id, grasp_name) in enumerate(
            tqdm(grasps, desc=args.obj)
        ):
            episode_dir = os.path.join(output_dir, args.obj, f"{gi:03d}")
            video_path = os.path.join(episode_dir, "turntable.mp4")

            if os.path.exists(video_path):
                success += 1
                continue

            os.makedirs(episode_dir, exist_ok=True)
            ok = render_single_grasp(
                args.obj, scene_type, scene_id, grasp_name,
                video_path, args, obj_mesh=obj_mesh, obj_pose=obj_pose,
            )
            if ok:
                success += 1
            else:
                fail += 1

        print(f"\nDone! {success} succeeded, {fail} failed")
        print(f"Output: {os.path.join(output_dir, args.obj)}")
        return

    # ---- Single grasp mode ----
    if args.scene is None or args.version is None or args.obj is None:
        print("Error: single grasp mode requires --version, --obj, and --scene")
        parser.print_help()
        sys.exit(1)

    scene_parts = args.scene.strip("/").split("/")
    if len(scene_parts) != 3:
        print(f"Error: --scene must be scene_type/scene_id/grasp_name, got '{args.scene}'")
        sys.exit(1)
    scene_type, scene_id, grasp_name = scene_parts

    if args.output is None:
        output_path = os.path.abspath(f"turntable_{args.obj}_{scene_type}_{scene_id}_{grasp_name}.mp4")
    else:
        output_path = os.path.abspath(args.output)

    print(f"Loading {args.obj} grasp: {scene_type}/{scene_id}/{grasp_name}")
    ok = render_single_grasp(
        args.obj, scene_type, scene_id, grasp_name,
        output_path, args,
    )
    if ok:
        print(f"Saved: {output_path}")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
