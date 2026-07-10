"""Validate recomputed poses against the legacy pose (step 3).

For every trial that has a recomputed ``pose_world.npy`` (new, current framework)
and a legacy ``optimized_pose_world.txt`` (old), measure how far they disagree.

Rotation diff alone is misleading for round/symmetric objects (a sphere fits many
rotations equally), so the primary metric is **ADD-S**: transform the mesh
vertices by each pose and take the mean nearest-neighbour distance between the
two point clouds — symmetry-invariant, in millimetres. Δtrans / Δrot are also
reported for reference.

    ADD-S(old,new) = mean_i  min_j || (R_old x_i + t_old) - (R_new x_j + t_new) ||

Usage:
    python -m src.dataset.validate_pose               # report + write JSON
    python -m src.dataset.validate_pose --top 40      # worst 40 trials
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import cKDTree

SRC_ROOT = "/home/mingi/shared_data/RSS2026_Mingi/experiment/selected_100"
DATASET_ROOTS = [
    "/home/mingi/shared_data/autodex_dataset/selected_100",
    "/home/mingi/shared_data/autodex_dataset/selected_100_wireout",
]
MESH_BASE = Path.home() / "shared_data/AutoDex/object/paradex"
N_SAMPLE = 800  # mesh vertices subsampled for ADD-S


def _mesh_points(obj, cache):
    if obj in cache:
        return cache[obj]
    mp = MESH_BASE / obj / "raw_mesh" / f"{obj}.obj"
    if not mp.exists():
        cache[obj] = None
        return None
    m = trimesh.load(str(mp), process=False)
    if isinstance(m, trimesh.Scene):
        m = trimesh.util.concatenate([g for g in m.geometry.values()
                                      if isinstance(g, trimesh.Trimesh)])
    v = np.asarray(m.vertices, dtype=np.float64)
    if len(v) > N_SAMPLE:
        idx = np.linspace(0, len(v) - 1, N_SAMPLE).astype(int)
        v = v[idx]
    cache[obj] = v
    return v


def _transform(pts, T):
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


def compare(old, new, pts):
    """(add_s_mm, dtrans_mm, drot_deg)."""
    po, pn = _transform(pts, old), _transform(pts, new)
    d1 = cKDTree(pn).query(po)[0]
    d2 = cKDTree(po).query(pn)[0]
    add_s = 0.5 * (d1.mean() + d2.mean()) * 1000.0
    dtrans = np.linalg.norm(old[:3, 3] - new[:3, 3]) * 1000.0
    Rrel = np.linalg.inv(old[:3, :3]) @ new[:3, :3]
    drot = np.degrees(np.arccos(np.clip((np.trace(Rrel) - 1) / 2, -1, 1)))
    return float(add_s), float(dtrans), float(drot)


def iter_trials(roots):
    for root in roots:
        if not os.path.isdir(root):
            continue
        for obj in sorted(os.listdir(root)):
            obj_dir = os.path.join(root, obj)
            if not os.path.isdir(obj_dir):
                continue
            for ts in sorted(os.listdir(obj_dir)):
                td = os.path.join(obj_dir, ts)
                if os.path.isdir(td):
                    yield obj, ts, td


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", nargs="+", default=DATASET_ROOTS)
    ap.add_argument("--top", type=int, default=30, help="print N worst trials")
    ap.add_argument("--out", default="/home/mingi/shared_data/autodex_dataset/pose_validation.json")
    args = ap.parse_args()

    cache = {}
    rows = []
    missing = {"no_new": 0, "no_old": 0, "no_mesh": 0}
    for obj, ts, td in iter_trials(args.roots):
        new_p = os.path.join(td, "pose_world.npy")
        old_p = os.path.join(SRC_ROOT, obj, ts, "outputs", f"{obj}_pose", "optimized_pose_world.txt")
        if not os.path.exists(new_p):
            missing["no_new"] += 1
            continue
        if not os.path.exists(old_p):
            missing["no_old"] += 1
            continue
        pts = _mesh_points(obj, cache)
        if pts is None:
            missing["no_mesh"] += 1
            continue
        add_s, dtrans, drot = compare(np.loadtxt(old_p) if old_p.endswith(".txt")
                                      else np.load(old_p), np.load(new_p), pts)
        rj = os.path.join(td, "recompute_pose.json")
        sil = json.load(open(rj)).get("sil_loss") if os.path.exists(rj) else None
        rows.append({"obj": obj, "ts": ts, "add_s_mm": round(add_s, 1),
                     "dtrans_mm": round(dtrans, 1), "drot_deg": round(drot, 1),
                     "sil_loss": sil})

    rows.sort(key=lambda r: -r["add_s_mm"])
    add = np.array([r["add_s_mm"] for r in rows]) if rows else np.array([0.0])
    print(f"compared {len(rows)} trials  (missing: {missing})")
    print(f"ADD-S mm  p50={np.percentile(add,50):.1f}  p90={np.percentile(add,90):.1f}  "
          f"p99={np.percentile(add,99):.1f}  max={add.max():.1f}")
    for thr in (10, 20, 50):
        print(f"  ADD-S > {thr}mm: {int((add>thr).sum())} ({(add>thr).mean()*100:.1f}%)")
    print(f"\nworst {args.top} (likely-invalid candidates):")
    print(f"  {'ADD-S':>7} {'Δtrans':>7} {'Δrot':>6} {'sil':>8}  obj/ts")
    for r in rows[:args.top]:
        sl = f"{r['sil_loss']:.5f}" if r['sil_loss'] is not None else "  -  "
        print(f"  {r['add_s_mm']:7.1f} {r['dtrans_mm']:7.1f} {r['drot_deg']:6.1f} {sl:>8}  {r['obj']}/{r['ts']}")

    json.dump({"n": len(rows), "missing": missing, "trials": rows}, open(args.out, "w"), indent=1)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
