"""Recompute the object init pose with the CURRENT framework, offline.

The legacy `optimized_pose_world.txt` was produced by the old SAM3+FoundationPose
pipeline. Only the pose init changed (FoundationPose -> FoundPose); SAM3 is
unchanged, so we REUSE the old masks and only rerun FoundPose + select + refine.
Reusing the mask also isolates the pose-model change for the old-vs-new check.

Per trial, per camera:
    rgb  (init_capture/images/{serial}.png, undistorted, full-res)
    mask (source outputs/{obj}_pose/masks/{serial}.png, half-res -> upscaled)
      -> FoundPose per-view pose (FoundPoseInit.estimate_one_view; no depth)
    then cross-view IoU select -> silhouette refine -> pose_world

Writes into the trial dir:
    pose_world.npy          refined 4x4 (only if sil_loss <= threshold)
    recompute_pose.json     sil_loss / best_serial / best_iou / n_masks / reject

No SAM3 -> runs in the `gotrack` env (FoundPose/MV-GoTrack needs
pytorch_lightning; nvdiffrast for the silhouette refine):
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True EGL_PLATFORM=surfaceless \
      ~/miniconda3/envs/gotrack/bin/python -m src.dataset.recompute_pose --trial <dir>
"""

import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np

SRC_ROOT = "/home/mingi/shared_data/RSS2026_Mingi/experiment/selected_100"
DATASET_ROOTS = [
    "/home/mingi/shared_data/autodex_dataset/selected_100",
    "/home/mingi/shared_data/autodex_dataset/selected_100_wireout",
]
MESH_BASE = Path.home() / "shared_data/AutoDex/object/paradex"
ASSETS_BASE = Path.home() / "shared_data/AutoDex/foundpose_assets"
OP_MESH_BASE = Path.home() / "shared_data/object_processing"

# Active mesh source, overridable via CLI (--op_mesh + --assets_base). When
# op_mesh is set, the object pose is estimated in the object_processing mesh
# frame (the one that defines tabletop/scene/canon) instead of the paradex mesh.
_USE_OP_MESH = False
_ASSETS_BASE = ASSETS_BASE
# Render resolution scale for the IoU-select + silhouette-refine steps only
# (FoundPose init still runs at full res). <1.0 cuts refine GPU memory ~1/scale^2,
# needed to fit alongside a concurrent GPU job. Refine at half res is ample.
_REFINE_SCALE = 1.0


def _mesh_path(obj):
    if _USE_OP_MESH:
        return OP_MESH_BASE / obj / "processed_data" / "mesh" / "simplified.obj"
    return MESH_BASE / obj / "raw_mesh" / f"{obj}.obj"


def _scale_K(K, sx, sy):
    K = np.asarray(K, dtype=np.float64).copy()
    K[0, 0] *= sx; K[0, 2] *= sx
    K[1, 1] *= sy; K[1, 2] *= sy
    return K

SIL_ITERS = 100
SIL_LR = 0.002
SIL_LOSS_THRESHOLD = 0.003


def _to_44(ext):
    ext = np.asarray(ext, dtype=np.float64)
    if ext.shape == (4, 4):
        return ext
    M = np.eye(4)
    M[:3, :4] = ext.reshape(3, 4)
    return M


def load_cam_param(trial_dir):
    """Return (K_undist{serial}, ext_cw{serial} 4x4 world->cam, (H, W))."""
    cp = os.path.join(trial_dir, "cam_param")
    intr = json.load(open(os.path.join(cp, "intrinsics.json")))
    extr = json.load(open(os.path.join(cp, "extrinsics.json")))
    K = {s: np.asarray(intr[s]["intrinsics_undistort"], dtype=np.float64).reshape(3, 3)
         for s in intr}
    ext = {s: _to_44(extr[s]) for s in extr if s in intr}
    any_s = next(iter(intr))
    H, W = int(intr[any_s]["height"]), int(intr[any_s]["width"])
    return K, ext, (H, W)


def load_images(trial_dir):
    """Return {serial: rgb} from init_capture/images. Fallbacks: the trial's own
    ``raw/images`` (corl trials store undistorted init frames there), then the
    legacy source ``images/`` dir."""
    d = os.path.join(trial_dir, "init_capture", "images")
    if not os.path.isdir(d):
        d = os.path.join(trial_dir, "raw", "images")
    if not os.path.isdir(d):
        obj = os.path.basename(os.path.dirname(trial_dir))
        ts = os.path.basename(trial_dir)
        d = os.path.join(SRC_ROOT, obj, ts, "images")
    out = {}
    if os.path.isdir(d):
        for f in os.listdir(d):
            if f.endswith(".png"):
                bgr = cv2.imread(os.path.join(d, f))
                if bgr is not None:
                    out[f[:-4]] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return out


def load_masks(trial_dir, H, W):
    """Object masks upscaled to (H, W).

    Prefer the trial's own ``obj_mask/`` (freshly segmented, e.g. new captures
    without legacy outputs -- inspire/corl). Fall back to the legacy SAM3 masks
    (source ``outputs/{obj}_pose/masks/``) for the RSS/allegro trials.
    """
    d = os.path.join(trial_dir, "obj_mask")
    if not os.path.isdir(d):
        obj = os.path.basename(os.path.dirname(trial_dir))
        ts = os.path.basename(trial_dir)
        d = os.path.join(SRC_ROOT, obj, ts, "outputs", f"{obj}_pose", "masks")
    out = {}
    if os.path.isdir(d):
        for f in os.listdir(d):
            if not f.endswith(".png"):
                continue
            m = cv2.imread(os.path.join(d, f), cv2.IMREAD_GRAYSCALE)
            if m is None:
                continue
            if m.shape != (H, W):
                m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
            out[f[:-4]] = m > 127
    return out


def recompute_trial(trial_dir, fp, sil, select_fn, write=False):
    K_all, ext_all, (H, W) = load_cam_param(trial_dir)
    images = load_images(trial_dir)
    masks = load_masks(trial_dir, H, W)

    candidates, masks_bool = {}, {}
    for s, rgb in images.items():
        if s not in K_all or s not in ext_all or s not in masks:
            continue
        mask = masks[s]
        if not mask.any():
            continue
        res = fp.estimate_one_view(image_rgb=rgb, mask_bool=mask,
                                   K=K_all[s], ext_cw=ext_all[s])
        if res is None or "pose_world" not in res:
            continue
        candidates[s] = np.asarray(res["pose_world"], dtype=np.float64)
        masks_bool[s] = mask

    info = {"n_candidates": len(candidates), "n_masks": len(masks_bool)}
    if not candidates or not masks_bool:
        info["reject"] = True
        info["reason"] = "no_candidates_or_masks"
        return None, info

    intr_subset = {s: K_all[s] for s in masks_bool}
    extr_subset = {s: ext_all[s] for s in masks_bool}

    # Optionally downscale the render resolution for select + refine (memory).
    Hr, Wr = H, W
    if _REFINE_SCALE < 1.0:
        Hr, Wr = int(round(H * _REFINE_SCALE)), int(round(W * _REFINE_SCALE))
        intr_subset = {s: _scale_K(K, Wr / W, Hr / H) for s, K in intr_subset.items()}
        masks_bool = {s: cv2.resize(m.astype(np.uint8), (Wr, Hr),
                                    interpolation=cv2.INTER_NEAREST) > 0
                      for s, m in masks_bool.items()}

    best_serial, best_pose, best_iou, _ = select_fn(
        candidates=candidates, masks=masks_bool,
        intrinsics=intr_subset, extrinsics=extr_subset,
        H=Hr, W=Wr, glctx=sil.glctx, mesh_tensors=sil.mesh_tensors,
    )
    if best_pose is None:
        info["reject"] = True
        info["reason"] = "iou_select_failed"
        return None, info

    views = [{"mask": (m.astype(np.uint8) * 255), "K": intr_subset[s], "extrinsic": extr_subset[s]}
             for s, m in masks_bool.items()]
    refined, sil_loss = sil.optimize(
        initial_pose_world=best_pose, views=views,
        iters=SIL_ITERS, lr=SIL_LR, antialias=True,
    )
    info.update({"best_serial": best_serial, "best_iou": float(best_iou),
                 "sil_loss": float(sil_loss),
                 "reject": bool(sil_loss > SIL_LOSS_THRESHOLD)})

    pose = np.asarray(refined, dtype=np.float64)
    if write:
        json.dump(info, open(os.path.join(trial_dir, "recompute_pose.json"), "w"), indent=1)
        if not info["reject"]:
            np.save(os.path.join(trial_dir, "pose_world.npy"), pose)
    return (None if info["reject"] else pose), info


def iter_trials(roots, obj_filter=None):
    for root in roots:
        if not os.path.isdir(root):
            continue
        for obj in sorted(os.listdir(root)):
            if obj_filter and obj != obj_filter:
                continue
            obj_dir = os.path.join(root, obj)
            if not os.path.isdir(obj_dir):
                continue
            for ts in sorted(os.listdir(obj_dir)):
                td = os.path.join(obj_dir, ts)
                if os.path.isdir(td):
                    yield obj, td


def build_models_for_obj(obj, sil, fp_cache):
    """(Re)build FoundPose + point silhouette at this object's mesh."""
    from autodex.perception.foundpose_init import FoundPoseInit
    mesh_path = _mesh_path(obj)
    assets_root = _ASSETS_BASE / obj
    repre = assets_root / "object_repre/v1" / obj / "1/repre.pth"
    if not mesh_path.exists() or not repre.exists():
        return None
    if obj not in fp_cache:
        fp_cache.clear()  # keep only current object's FoundPose in memory
        fp_cache[obj] = FoundPoseInit(mesh_path=str(mesh_path), assets_root=str(assets_root),
                                      obj_name=obj, device="cuda:0")
    sil.reset_mesh(str(mesh_path))
    return fp_cache[obj]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--trial", help="single trial dir")
    ap.add_argument("--obj", help="restrict to one object")
    ap.add_argument("--roots", nargs="+", default=DATASET_ROOTS)
    ap.add_argument("--write", action="store_true", help="save pose_world.npy + recompute_pose.json")
    ap.add_argument("--overwrite", action="store_true", help="redo even if recompute_pose.json exists")
    ap.add_argument("--op_mesh", action="store_true",
                    help="estimate pose in the object_processing mesh frame "
                         "(processed_data/mesh/simplified.obj) instead of paradex raw_mesh")
    ap.add_argument("--assets_base", default=None,
                    help="FoundPose repre root (defaults to the paradex foundpose_assets; "
                         "use a separate dir when onboarding from --op_mesh)")
    ap.add_argument("--refine_scale", type=float, default=1.0,
                    help="render-resolution scale for IoU-select + silhouette-refine "
                         "(<1.0 cuts GPU memory; e.g. 0.5 to fit alongside another GPU job)")
    args = ap.parse_args()

    global _USE_OP_MESH, _ASSETS_BASE, _REFINE_SCALE
    _USE_OP_MESH = args.op_mesh
    if args.assets_base:
        _ASSETS_BASE = Path(args.assets_base)
    _REFINE_SCALE = args.refine_scale

    from autodex.perception.silhouette import SilhouetteOptimizer
    from autodex.perception.pose_select import select_best_pose_by_iou

    # collect (obj, trial) list
    if args.trial:
        obj = os.path.basename(os.path.dirname(args.trial))
        work = [(obj, args.trial)]
    else:
        work = list(iter_trials(args.roots, args.obj))
    # group by object so FoundPose/mesh load once per object
    work.sort(key=lambda x: x[0])

    sil = None
    fp_cache = {}
    stats = {"ok": 0, "reject": 0, "no_model": 0, "skip_exist": 0}
    cur_obj = None
    fp = None
    for obj, td in work:
        if os.path.exists(os.path.join(td, "recompute_pose.json")) and not args.overwrite:
            stats["skip_exist"] += 1
            continue
        if obj != cur_obj:
            if sil is None:
                # init silhouette with first available mesh, then reset per object
                sil = SilhouetteOptimizer(str(_mesh_path(obj)))
            fp = build_models_for_obj(obj, sil, fp_cache)
            cur_obj = obj
        if fp is None:
            stats["no_model"] += 1
            print(f"  no_model: {obj}")
            continue
        pose, info = recompute_trial(td, fp, sil, select_best_pose_by_iou, write=args.write)
        key = "reject" if info.get("reject") else "ok"
        stats[key] += 1
        sl = info.get("sil_loss")
        print(f"  [{key}] {obj}/{os.path.basename(td)} "
              f"loss={sl:.5f} best={info.get('best_serial')} n={info.get('n_masks')}"
              if sl is not None else f"  [{key}] {obj}/{os.path.basename(td)} {info.get('reason','')}")
    print("summary:", stats)


if __name__ == "__main__":
    main()
