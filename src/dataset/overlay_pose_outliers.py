"""Render mask + recomputed-pose overlays for quarantined pose outliers.

The sil-loss outliers (``selected_100_pose_outlier/``) were rejected because the
silhouette refine could not match the mesh to the masks (sil_loss >= 0.003), so
no ``pose_world.npy`` was saved. To eyeball *why* (bad mask? symmetry flip that
only fits some cams?), re-run the same FoundPose -> IoU select -> sil refine, then
render two 24-cam grids per trial:

  mask_grid.png  : old SAM3 mask (cyan) over each camera image
  pose_grid.png  : recomputed (rejected) mesh pose (green) over each camera image

Reuses the render/grid helpers from foundpose_overlay_grid and the recompute
logic from recompute_pose (masks reused from SRC_ROOT, same as the recompute).

Runs in the `foundationpose` env (FoundPose + nvdiffrast):
    EGL_PLATFORM=surfaceless ~/miniconda3/envs/foundationpose/bin/python \
      -m src.dataset.overlay_pose_outliers --obj magazine_file --limit 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

from src.validation.perception.foundpose_overlay_grid import _render_overlay, _make_grid
from src.dataset.recompute_pose import (
    load_cam_param, load_images, load_masks, build_models_for_obj,
    MESH_BASE, SIL_ITERS, SIL_LR,
)

OUTLIER_ROOT = Path("/home/mingi/shared_data/autodex_dataset/selected_100_pose_outlier")


def _mask_overlay(image_rgb, mask_bool, color=(0, 255, 255), alpha=0.5):
    out = image_rgb.copy()
    c = np.array(color, np.float32)
    out[mask_bool] = (out[mask_bool] * (1 - alpha) + c * alpha).astype(np.uint8)
    return out


def _mesh_sil(pose_world, K, T, H, W, glctx, mesh_tensors):
    """Boolean silhouette of the mesh at pose_world in this camera."""
    import torch
    sys.path.insert(0, str(REPO_ROOT / "autodex/perception/thirdparty/FoundationPose"))
    from Utils import nvdiffrast_render
    pose_cam = T @ pose_world
    pt = torch.as_tensor(pose_cam, device="cuda", dtype=torch.float32).reshape(1, 4, 4)
    rc, _, _ = nvdiffrast_render(K=np.asarray(K, np.float32), H=H, W=W, ob_in_cams=pt,
                                 glctx=glctx, mesh_tensors=mesh_tensors, use_light=False)
    return (rc[0].sum(dim=2) > 0).detach().cpu().numpy()


def _combined_overlay(image_rgb, mask_bool, mesh_sil, refined_sil=None,
                      mesh_color=(0, 220, 0), mask_color=(255, 40, 40),
                      refined_color=(255, 230, 0), alpha=0.45):
    """mesh (green fill) + mask (red contour) on one image; where they disagree is visible.

    If refined_sil is given, mesh_sil is the INITIAL pose (green fill) and refined_sil
    is drawn as a yellow contour so init vs refined vs mask are all comparable.
    """
    out = image_rgb.copy().astype(np.float32)
    c = np.array(mesh_color, np.float32)
    out[mesh_sil] = out[mesh_sil] * (1 - alpha) + c * alpha
    out = out.astype(np.uint8)
    if refined_sil is not None:
        rc, _ = cv2.findContours(refined_sil.astype(np.uint8) * 255,
                                 cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, rc, -1, refined_color, 4)
    mc, _ = cv2.findContours(mask_bool.astype(np.uint8) * 255,
                             cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, mc, -1, mask_color, 5)
    return out


def analyze_trial(trial_dir, fp, sil, select_fn, refine=False):
    """Run FoundPose -> IoU select and measure the INITIAL pose's sil loss.

    Optionally (refine=True) also run the gradient sil refine. Refine needs
    antialias=True for a non-zero gradient (nvdiffrast: pose gradient flows only
    through dr.antialias), which spikes GPU memory ~2GB and OOMs alongside a
    concurrent GPU job -- so it is OFF by default. The initial loss is measured
    the same way the pipeline's own outlier filter does (antialias-free forward
    MSE), so it is a faithful, memory-cheap number even without refine.

    FoundPose (~7GB) is offloaded to CPU after per-view estimation so it never
    coexists on-GPU with the render -- lets this run alongside a live BODex gen.
    """
    import torch
    K_all, ext_all, (H, W) = load_cam_param(trial_dir)
    images = load_images(trial_dir)
    masks = load_masks(trial_dir, H, W)

    fp.model.to(fp.device)  # ensure resident for estimation (may be on CPU from prev trial)
    candidates, masks_bool = {}, {}
    for s, rgb in images.items():
        if s not in K_all or s not in ext_all or s not in masks:
            continue
        if not masks[s].any():
            continue
        res = fp.estimate_one_view(image_rgb=rgb, mask_bool=masks[s],
                                   K=K_all[s], ext_cw=ext_all[s])
        if res is None or "pose_world" not in res:
            continue
        candidates[s] = np.asarray(res["pose_world"], dtype=np.float64)
        masks_bool[s] = masks[s]
    fp.model.to("cpu")            # free ~7GB before the sil-optimize render
    torch.cuda.empty_cache()
    if not candidates:
        return None

    intr = {s: K_all[s] for s in masks_bool}
    extr = {s: ext_all[s] for s in masks_bool}
    best_serial, best_pose, best_iou, _ = select_fn(
        candidates=candidates, masks=masks_bool, intrinsics=intr, extrinsics=extr,
        H=H, W=W, glctx=sil.glctx, mesh_tensors=sil.mesh_tensors)
    if best_pose is None:
        return None
    views = [{"mask": (m.astype(np.uint8) * 255), "K": intr[s], "extrinsic": extr[s]}
             for s, m in masks_bool.items()]
    # iters=1 returns the loss evaluated AT the initial pose (pre-step) -> init_loss,
    # measured antialias-free (same as the pipeline's outlier filter) so the value
    # is faithful and memory-cheap.
    _, init_loss = sil.optimize(initial_pose_world=best_pose, views=views,
                                iters=1, lr=SIL_LR, antialias=False)
    refined_pose, refined_loss = None, None
    if refine:
        # antialias=True needed for a real gradient; costs ~2GB, may OOM if the
        # GPU is busy. Faithful to the production recompute.
        refined_pose, refined_loss = sil.optimize(initial_pose_world=best_pose,
                                                   views=views, iters=SIL_ITERS,
                                                   lr=SIL_LR, antialias=True)
        refined_pose = np.asarray(refined_pose, np.float64)
        refined_loss = float(refined_loss)
    return dict(init_pose=np.asarray(best_pose, np.float64),
                refined_pose=refined_pose,
                init_loss=float(init_loss), refined_loss=refined_loss,
                images=images, masks_bool=masks_bool, K_all=K_all, ext_all=ext_all,
                H=H, W=W, best_serial=best_serial)


def iter_outliers(root, obj_filter=None):
    for obj in sorted(os.listdir(root)):
        if obj_filter and obj != obj_filter:
            continue
        od = os.path.join(root, obj)
        if not os.path.isdir(od):
            continue
        for ts in sorted(os.listdir(od)):
            td = os.path.join(od, ts)
            if os.path.isdir(td):
                yield obj, td


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=str(OUTLIER_ROOT))
    ap.add_argument("--obj", help="restrict to one object")
    ap.add_argument("--limit", type=int, default=0, help="max trials (0=all)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--mask_only", action="store_true",
                    help="render only mask_grid.png (no GPU: skip FoundPose + sil refine)")
    ap.add_argument("--refine", action="store_true",
                    help="also run the gradient sil refine (antialias=True, ~2GB, "
                         "may OOM alongside another GPU job); yellow contour = refined")
    args = ap.parse_args()

    work = list(iter_outliers(args.root, args.obj))
    work.sort(key=lambda x: x[0])
    if args.limit:
        work = work[:args.limit]
    print(f"[overlay] {len(work)} outlier trials"
          + ("  (mask_only)" if args.mask_only else ""))

    if args.mask_only:
        done = 0
        for obj, td in work:
            out_dir = os.path.join(td, "overlay")
            if os.path.exists(os.path.join(out_dir, "mask_grid.png")) and not args.overwrite:
                continue
            _, _, (H, W) = load_cam_param(td)
            images = load_images(td)
            masks = load_masks(td, H, W)
            mask_ovs = {s: cv2.cvtColor(_mask_overlay(images[s], masks[s]), cv2.COLOR_RGB2BGR)
                        for s in sorted(images) if s in masks and masks[s].any()}
            mg = _make_grid(mask_ovs, cols=6)
            if mg is not None:
                os.makedirs(out_dir, exist_ok=True)
                cv2.imwrite(os.path.join(out_dir, "mask_grid.png"), mg)
                done += 1
                print(f"  [{obj}/{os.path.basename(td)}] {len(mask_ovs)} masks -> {out_dir}")
        print(f"[done] {done} mask grids")
        return

    from autodex.perception.silhouette import SilhouetteOptimizer
    from autodex.perception.pose_select import select_best_pose_by_iou

    sil, fp_cache, cur_obj, fp = None, {}, None, None
    done = 0
    for obj, td in work:
        out_dir = os.path.join(td, "overlay")
        if (os.path.exists(os.path.join(out_dir, "overlay.png"))
                and not args.overwrite):
            continue
        if obj != cur_obj:
            if sil is None:
                sil = SilhouetteOptimizer(str(MESH_BASE / obj / "raw_mesh" / f"{obj}.obj"))
            fp = build_models_for_obj(obj, sil, fp_cache)
            cur_obj = obj
        if fp is None:
            print(f"  no_model: {obj}")
            continue

        t0 = time.perf_counter()
        r = analyze_trial(td, fp, sil, select_best_pose_by_iou, refine=args.refine)
        if r is None:
            print(f"  [{obj}/{os.path.basename(td)}] no pose")
            continue
        os.makedirs(out_dir, exist_ok=True)

        # Combined per-cam overlay: initial mesh (green fill) + old mask (red
        # contour). With --refine, the refined mesh is added as a yellow contour.
        has_ref = r["refined_pose"] is not None
        ovs = {}
        for s in sorted(r["images"].keys()):
            if s not in r["K_all"] or s not in r["ext_all"] or s not in r["masks_bool"]:
                continue
            init_sil = _mesh_sil(r["init_pose"], r["K_all"][s], r["ext_all"][s],
                                 r["H"], r["W"], sil.glctx, sil.mesh_tensors)
            ref_sil = (_mesh_sil(r["refined_pose"], r["K_all"][s], r["ext_all"][s],
                                 r["H"], r["W"], sil.glctx, sil.mesh_tensors)
                       if has_ref else None)
            comb = _combined_overlay(r["images"][s], r["masks_bool"][s], init_sil,
                                     refined_sil=ref_sil)
            ovs[s] = cv2.cvtColor(comb, cv2.COLOR_RGB2BGR)

        grid = _make_grid(ovs, cols=6, highlight_key=r["best_serial"])
        if grid is not None:
            keep = "KEEP" if r["init_loss"] <= 0.003 else "OUTLIER"
            banner = (f"init_loss={r['init_loss']:.5f}  (thr 0.003 -> {keep})"
                      + (f"   refined_loss={r['refined_loss']:.5f}" if has_ref else "")
                      + "   green=init  red=mask" + ("  yellow=refined" if has_ref else ""))
            # Large, high-contrast banner so the loss is legible on the full grid.
            for col, th in ((0, 0, 0), 12), ((0, 255, 255), 5):
                cv2.putText(grid, banner, (20, 80), cv2.FONT_HERSHEY_SIMPLEX,
                            2.2, col, th, cv2.LINE_AA)
            cv2.imwrite(os.path.join(out_dir, "overlay.png"), grid)
        json.dump({"init_loss": r["init_loss"], "refined_loss": r["refined_loss"],
                   "best_serial": r["best_serial"], "threshold": 0.003,
                   "keep_by_init": bool(r["init_loss"] <= 0.003),
                   "init_pose": r["init_pose"].tolist(),
                   "refined_pose": (r["refined_pose"].tolist() if has_ref else None)},
                  open(os.path.join(out_dir, "losses.json"), "w"), indent=1)
        done += 1
        print(f"  [{obj}/{os.path.basename(td)}] init={r['init_loss']:.5f}"
              + (f" refined={r['refined_loss']:.5f}" if has_ref else "")
              + f" best={r['best_serial']} ({time.perf_counter()-t0:.1f}s)")

    print(f"[done] {done} trials")


if __name__ == "__main__":
    main()
