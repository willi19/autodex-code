import argparse
import json
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from autodex.utils.path import get_obj_root


def _obj_dir(obj_name: str, obj_root: str) -> Path:
    return Path(obj_root) / obj_name


def _load_tabletop_pose(obj_name: str, pose_idx: int, obj_root: str) -> np.ndarray:
    p = (_obj_dir(obj_name, obj_root) / "processed_data" / "info" /
         "tabletop" / f"{pose_idx:03d}.npy")
    return np.load(p)


def _mesh_target_entry(obj_name: str, pose_xyzquat, obj_root: str) -> dict:
    od = _obj_dir(obj_name, obj_root)
    return {
        "scale": [1.0, 1.0, 1.0],
        "pose": list(map(float, pose_xyzquat)),
        "file_path": str(od / "processed_data" / "mesh" / "simplified.obj"),
        "urdf_path": str(od / "processed_data" / "urdf" / "coacd.urdf"),
    }


def _se3_to_xyzquat(T: np.ndarray):
    t = T[:3, 3]
    q_xyzw = Rotation.from_matrix(T[:3, :3]).as_quat()
    return [float(t[0]), float(t[1]), float(t[2]),
            float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]


def _simplify_polygon_indices(pts: np.ndarray, angle_tol_deg: float = 15.0):
    """Iteratively drop the vertex with the smallest turn angle until all
    remaining turns exceed angle_tol_deg. Returns indices into original pts
    that survive (in CCW order)."""
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    if n <= 3:
        return list(range(n))
    cos_tol = np.cos(np.deg2rad(angle_tol_deg))
    alive = list(range(n))
    while len(alive) > 3:
        worst_pos = -1
        worst_cos = -1.0
        for pos in range(len(alive)):
            i = alive[pos]
            prev_i = alive[(pos - 1) % len(alive)]
            next_i = alive[(pos + 1) % len(alive)]
            e_in = pts[i] - pts[prev_i]
            e_out = pts[next_i] - pts[i]
            nin = np.linalg.norm(e_in)
            no = np.linalg.norm(e_out)
            if nin < 1e-12 or no < 1e-12:
                worst_pos = pos; worst_cos = 1.0; break
            c = float(e_in @ e_out / (nin * no))
            if c > worst_cos:
                worst_cos = c; worst_pos = pos
        if worst_cos > cos_tol:
            alive.pop(worst_pos)
        else:
            break
    return alive


def _rotmat_to_xyzquat(R: np.ndarray, t: np.ndarray):
    q_xyzw = Rotation.from_matrix(R).as_quat()
    return [float(t[0]), float(t[1]), float(t[2]),
            float(q_xyzw[3]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2])]


def gen_reorient_scene(
    obj_name: str,
    pose_i_idx: int,
    pose_j_idx: int,
    h: float,
    obj_root: str,
    thickness: float = 0.01,
    table_size: float = 2.0,
    table_thickness: float = 0.2,
    raw_mesh: bool = True,
) -> dict:
    """Reorientation grasp scene for transition pose_i -> pose_j.

    Scene contents (object placed at pose_i in world frame):
      - "table_i":  pose i's table (world z=0 plane, cuboid)
      - "pillar_kk": pose j column pillars along d_j direction
                     (d_j = R_i @ R_j^T @ (-e_z) in world; pose j's gravity
                     expressed in current world with object at pose i).
                     Pillars start on the plane through the lower extreme along
                     d_j, extend along d_j by length `h`, sit on convex-hull
                     edges of the projection perpendicular to d_j, offset
                     inward by `thickness/2`.
      - "table_j": pose j's table plane: cuboid perpendicular to d_j, located
                   at distance h beyond the lower extreme along d_j.
    """
    od = _obj_dir(obj_name, obj_root)
    mesh_file = od / ("raw_mesh" if raw_mesh else "processed_data/mesh") / (
        f"{obj_name}.obj" if raw_mesh else "simplified.obj"
    )
    mesh = trimesh.load(mesh_file, process=False, force="mesh")

    Ti = _load_tabletop_pose(obj_name, pose_i_idx, obj_root)
    Tj = _load_tabletop_pose(obj_name, pose_j_idx, obj_root)
    Ri, ti = Ti[:3, :3], Ti[:3, 3]
    Rj = Tj[:3, :3]

    verts_w = mesh.vertices @ Ri.T + ti

    # Pose j gravity direction expressed in current world (object at pose i)
    d_j = Ri @ Rj.T @ np.array([0.0, 0.0, -1.0])
    d_j /= np.linalg.norm(d_j)

    # Orthonormal basis (u, v, d_j) right-handed
    ref = np.array([1.0, 0.0, 0.0]) if abs(d_j[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = ref - np.dot(ref, d_j) * d_j
    u /= np.linalg.norm(u)
    v = np.cross(d_j, u)

    proj_2d = np.column_stack([verts_w @ u, verts_w @ v])
    d_coords = verts_w @ d_j  # signed coord along d_j
    lower_extreme_d = float(d_coords.max())  # furthest in d_j direction = lowest

    hull = ConvexHull(proj_2d)
    hull_idx = hull.vertices  # CCW indices into proj_2d
    hull_pts_full = proj_2d[hull_idx]
    kept_pos = _simplify_polygon_indices(hull_pts_full, angle_tol_deg=15.0)

    # Hull centroid for angular wedge partitioning
    centroid_2d = hull_pts_full.mean(axis=0)
    mesh_ang = np.arctan2(proj_2d[:, 1] - centroid_2d[1],
                          proj_2d[:, 0] - centroid_2d[0])
    hull_ang = np.arctan2(hull_pts_full[:, 1] - centroid_2d[1],
                          hull_pts_full[:, 0] - centroid_2d[0])

    cuboids = {
        "table_i": {
            "dims": [2.0, 2.0, 0.2],
            "pose": [0.0, 0.0, -0.1, 1.0, 0.0, 0.0, 0.0],
        },
    }

    if h > 1e-6:
        K = len(kept_pos)
        for k in range(K):
            a = kept_pos[k]
            b = kept_pos[(k + 1) % K]
            p1_2d = hull_pts_full[a]
            p2_2d = hull_pts_full[b]
            edge_2d = p2_2d - p1_2d
            edge_len = float(np.linalg.norm(edge_2d))
            if edge_len < 1e-9:
                continue
            # All mesh vertices whose angle from centroid falls in this edge's wedge
            ang_a = hull_ang[a]; ang_b = hull_ang[b]
            if ang_a <= ang_b:
                in_wedge = (mesh_ang >= ang_a) & (mesh_ang <= ang_b)
            else:  # wraps around
                in_wedge = (mesh_ang >= ang_a) | (mesh_ang <= ang_b)
            if in_wedge.sum() == 0:
                top_d = float(max(d_coords[hull_idx[a]], d_coords[hull_idx[b]]))
            else:
                # MAX d_j = lowest mesh point in wedge → pillar stays under mesh,
                # not blocking grasping access to mesh sides.
                top_d = float(d_coords[in_wedge].max())
            pillar_length = (lower_extreme_d + h) - top_d
            if pillar_length <= 1e-6:
                continue
            edge_dir_2d = edge_2d / edge_len
            edge_dir_3d = edge_dir_2d[0] * u + edge_dir_2d[1] * v
            inward_3d = np.cross(d_j, edge_dir_3d)

            mid_2d = (p1_2d + p2_2d) / 2
            mid_on_plane = mid_2d[0] * u + mid_2d[1] * v + top_d * d_j
            center = mid_on_plane + (thickness / 2) * inward_3d + (pillar_length / 2) * d_j

            R_pillar = np.column_stack([edge_dir_3d, inward_3d, d_j])
            cuboids[f"pillar_{k:02d}"] = {
                "dims": [edge_len, float(thickness), float(pillar_length)],
                "pose": _rotmat_to_xyzquat(R_pillar, center),
            }

    # Pose j table plane: perpendicular to d_j, at distance h beyond lower extreme
    obj_center_2d = np.array([ti @ u, ti @ v])
    table_j_center = (
        obj_center_2d[0] * u
        + obj_center_2d[1] * v
        + (lower_extreme_d + h + table_thickness / 2) * d_j
    )
    R_table_j = np.column_stack([u, v, d_j])
    cuboids["table_j"] = {
        "dims": [float(table_size), float(table_size), float(table_thickness)],
        "pose": _rotmat_to_xyzquat(R_table_j, table_j_center),
    }

    return {
        "scene": {
            "mesh": {"target": _mesh_target_entry(obj_name, _se3_to_xyzquat(Ti), obj_root)},
            "cuboid": cuboids,
        },
        "meta": {
            "scene_type": "reorient",
            "pose_i": f"{pose_i_idx:03d}",
            "pose_j": f"{pose_j_idx:03d}",
            "h": h,
            "thickness": thickness,
            "version": "v8",
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", required=True)
    parser.add_argument("--i", type=int, required=True, help="pose i (current resting)")
    parser.add_argument("--j", type=int, required=True, help="pose j (target after reorient)")
    parser.add_argument("--h", type=float, required=True,
                        help="distance from lower extreme to pose j table along d_j (meters)")
    parser.add_argument("--version", default="v8",
                        help="v8 tabletop asset contract (only supported value)")
    parser.add_argument("--thickness", type=float, default=0.01)
    parser.add_argument("--out", type=str, default=None,
                        help="output json; default outputs/reorient_scenes/{obj}/{i}_{j}_h{h}.json")
    args = parser.parse_args()
    if args.version != "v8":
        parser.error("gen_scene supports only --version v8; legacy assets are not used")
    obj_root = get_obj_root(args.version)

    scene = gen_reorient_scene(
        args.obj, args.i, args.j, args.h, thickness=args.thickness,
        obj_root=obj_root,
    )
    default_name = f"{args.i:03d}_{args.j:03d}_h{int(round(args.h * 1000))}.json"
    out_path = Path(args.out) if args.out else (
        Path(__file__).resolve().parents[3] / "outputs" / "reorient_scenes" / args.version
        / args.obj / default_name
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(scene, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
