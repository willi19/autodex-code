"""Fix op-misfit poses via the paradex->op mesh rigid transform.

The op-unverified poses were tracked in the *paradex* mesh frame. If the paradex
mesh and the object_processing (op) mesh differ only by a rigid transform T
(op_local ~= T @ paradex_local, found by ICP), then the pose expressed in the op
frame is  P' = P @ inv(T).  For each failing trial we apply that correction and
re-measure op+antialias sil:

  sil(P') <= thr  -> transform works: adopt P' as pose_world (orig kept as
                     pose_world_paradex.npy), record in recompute_pose.json.
  sil(P') >  thr  -> transform does NOT fix it (bad track, not a frame issue):
                     leave pose_world, flag for recompute.

Input = op_sil_unverified.json (the high=True trials). Paradex mesh per trial:
gotrack summary.json mesh_path (corl/inspire) or paradex raw_mesh (selected_100).

    EGL_PLATFORM=surfaceless ~/miniconda3/envs/gotrack/bin/python \
        -m src.dataset.transform_fix [--thr 0.003] [--write]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import trimesh
from scipy.spatial import cKDTree

import src.dataset.recompute_pose as R
from src.dataset.recompute_pose import _mesh_path
from src.dataset.check_mesh_fit import MASK_DIRS, DS_BASE
from src.dataset.measure_op_sil import aa_loss

IN = "/home/mingi/shared_data/autodex_dataset/op_sil_unverified.json"
OUT = "/home/mingi/shared_data/autodex_dataset/transform_fix.json"


def paradex_mesh(dataset, obj, ts):
    if dataset in ("corl_selected_100", "selected_100_inspire"):
        sp = os.path.join(DS_BASE, dataset, obj, ts,
                          "object_tracking", "gotrack_output", "summary.json")
        if os.path.exists(sp):
            mp = json.load(open(sp)).get("mesh_path")
            if mp and os.path.exists(mp):
                return mp
    p = f"/home/mingi/shared_data/AutoDex/object/paradex/{obj}/raw_mesh/{obj}.obj"
    return p if os.path.exists(p) else None


def icp(src_pts, dst_pts, init, iters=100):
    tree = cKDTree(dst_pts)
    T = init.copy()
    P = (T[:3, :3] @ src_pts.T + T[:3, 3:]).T
    for _ in range(iters):
        _, idx = tree.query(P)
        Q = dst_pts[idx]
        pc, qc = P.mean(0), Q.mean(0)
        H = (P - pc).T @ (Q - qc)
        U, S, Vt = np.linalg.svd(H)
        Rm = Vt.T @ U.T
        if np.linalg.det(Rm) < 0:
            Vt[-1] *= -1
            Rm = Vt.T @ U.T
        t = qc - Rm @ pc
        P = (Rm @ P.T + t[:, None]).T
        T = np.block([[Rm, t[:, None]], [0, 0, 0, 1]]) @ T
    d, _ = tree.query(P)
    return T, float(d.mean())


def best_icp(src_mesh, dst_mesh):
    np.random.seed(0)
    src = src_mesh.sample(6000)
    dst = dst_mesh.sample(6000)
    inits = [np.eye(4)]
    for a in (np.pi,):
        for ax in ([1, 0, 0], [0, 1, 0], [0, 0, 1]):
            inits.append(trimesh.transformations.rotation_matrix(a, ax))
    for a in (np.pi / 2, -np.pi / 2):
        for ax in ([0, 0, 1], [1, 0, 0], [0, 1, 0]):
            inits.append(trimesh.transformations.rotation_matrix(a, ax))
    return min((icp(src, dst, r) for r in inits), key=lambda x: x[1])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--thr", type=float, default=0.003)
    ap.add_argument("--refine_scale", type=float, default=0.5)
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()

    from autodex.perception.silhouette import SilhouetteOptimizer
    R._USE_OP_MESH = True

    high = [t for t in json.load(open(IN))["trials"] if t["high"]]
    high.sort(key=lambda t: (t["obj"], t["dataset"]))
    print(f"[tfix] {len(high)} failing trials, {len(set(t['obj'] for t in high))} objects")

    sil = None
    icp_cache = {}     # (obj, paradex_mesh_path) -> (T, cost)
    rows = []
    cur = None
    for t in high:
        ds, obj, ts = t["dataset"], t["obj"], t["ts"]
        opm = str(_mesh_path(obj))
        if obj != cur:
            sil = SilhouetteOptimizer(opm) if sil is None else (sil.reset_mesh(opm) or sil)
            cur = obj
        pm = paradex_mesh(ds, obj, ts)
        if pm is None:
            rows.append({**t, "status": "no_paradex_mesh"})
            continue
        key = (obj, pm)
        if key not in icp_cache:
            icp_cache[key] = best_icp(trimesh.load(pm, process=False),
                                      trimesh.load(opm, process=False))
        T, cost = icp_cache[key]
        Ti = np.linalg.inv(T)
        td = os.path.join(DS_BASE, ds, obj, ts)
        pw = np.load(os.path.join(td, "pose_world.npy"))
        pw_fix = pw @ Ti
        # measure corrected sil by temporarily swapping the file? no -- aa_loss reads
        # pose_world.npy. Measure directly here instead.
        s2 = _sil_at(sil, td, MASK_DIRS[ds](obj, ts), pw_fix, args.refine_scale)
        ok = s2 is not None and s2 <= args.thr
        rows.append({**t, "icp_mm": round(cost * 1000, 2), "sil_orig": round(t["sil"], 5),
                     "sil_fixed": None if s2 is None else round(s2, 5),
                     "status": "fixed" if ok else "transform_fail"})
        print(f"  {'OK ' if ok else 'NO '} {ds}/{obj}/{ts}: {t['sil']:.5f} -> "
              f"{'None' if s2 is None else f'{s2:.5f}'}  (icp {cost*1000:.1f}mm)")
        if ok and args.write:
            np.save(os.path.join(td, "pose_world_paradex.npy"), pw)
            np.save(os.path.join(td, "pose_world.npy"), pw_fix)
            rp = os.path.join(td, "recompute_pose.json")
            info = json.load(open(rp)) if os.path.exists(rp) else {}
            info.update({"sil_loss": float(s2), "reject": False, "source": "paradex2op_transform",
                         "icp_mm": round(cost * 1000, 2)})
            json.dump(info, open(rp, "w"), indent=1)

    fixed = [r for r in rows if r.get("status") == "fixed"]
    fail = [r for r in rows if r.get("status") == "transform_fail"]
    json.dump({"thr": args.thr, "n": len(rows), "n_fixed": len(fixed),
               "n_fail": len(fail), "wrote": bool(args.write), "trials": rows},
              open(args.out, "w"), indent=1)
    import collections
    print(f"\n[tfix] fixed {len(fixed)}/{len(rows)}  (transform_fail {len(fail)})  written={args.write}")
    print("  still failing (need recompute) by object:")
    for k, c in collections.Counter(f"{r['dataset']}/{r['obj']}" for r in fail).most_common():
        print(f"    {c:3d}  {k}")
    print(f"-> {args.out}")


def _sil_at(sil, trial_dir, mask_dir, pose, refine_scale):
    import cv2
    from src.dataset.recompute_pose import load_cam_param, _scale_K
    from src.dataset.check_mesh_fit import load_masks
    K_all, ext_all, (H, W) = load_cam_param(trial_dir)
    masks = load_masks(mask_dir, H, W)
    masks = {s: m for s, m in masks.items() if s in K_all and s in ext_all and m.any()}
    if not masks:
        return None
    intr = {s: K_all[s] for s in masks}
    if refine_scale < 1.0:
        Hr, Wr = int(round(H * refine_scale)), int(round(W * refine_scale))
        intr = {s: _scale_K(intr[s], Wr / W, Hr / H) for s in masks}
        masks = {s: cv2.resize(m.astype(np.uint8), (Wr, Hr), interpolation=cv2.INTER_NEAREST) > 0
                 for s, m in masks.items()}
    views = [{"mask": (m.astype(np.uint8) * 255), "K": intr[s], "extrinsic": ext_all[s]}
             for s, m in masks.items()]
    _, loss = sil.optimize(initial_pose_world=pose, views=views, iters=1, antialias=True)
    return float(loss)


if __name__ == "__main__":
    main()
