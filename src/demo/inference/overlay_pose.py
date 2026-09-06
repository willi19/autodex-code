#!/usr/bin/env python3
"""Overlay raw Object6D pose and the detected mask from an inference capture.

This renders ``pose_world.npy`` exactly as perception returned it: the mesh is
green and the SAM3 mask saved by InitOrchestrator is red. In particular, it
deliberately does *not* apply planner-only tabletop snapping or symmetry
adjustments, so a bad pose is visible as it was observed.

Example:
    python src/demo/inference/overlay_pose.py \
        --trial /home/robot/shared_data/AutoDex/experiment/v8/inspire/wood_organizer/20260901_031951
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from paradex.calibration.utils import load_camparam
from paradex.image.image_dict import ImageDict

from autodex.utils.path import get_obj_root
from src.execution.scene_cfg import find_planning_mesh


MASK_BGR = np.asarray((0, 0, 255), dtype=np.float32)


def _saved_masks(images: ImageDict, mask_dir: Path) -> dict[str, np.ndarray]:
    """Read any detected masks retained with a capture."""
    if not mask_dir.is_dir():
        return {}
    masks = {}
    for serial, image in images.images.items():
        mask_path = mask_dir / f"{serial}.png"
        if not mask_path.is_file():
            continue
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            continue
        if mask.shape != image.shape[:2]:
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        masks[serial] = mask
    return masks


def _recompute_masks(images: ImageDict, prompt: str) -> dict[str, np.ndarray]:
    """Run the live init daemon's SAM3 image segmentor on saved frames.

    Older trials did not persist the PUB mask payload. This is an offline,
    non-persistent reconstruction for visual diagnosis only; it does not
    modify the trial or re-run Object6D pose estimation.
    """
    from autodex.perception.mask import Sam3ImageSegmentor

    print(f"[overlay] recomputing SAM3 masks (prompt={prompt!r})...")
    segmentor = Sam3ImageSegmentor(gpu=0)
    masks = {}
    for serial, image in images.images.items():
        mask = segmentor.segment(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), prompt)
        if mask is not None:
            masks[serial] = mask
    # The mesh renderer allocates its own CUDA context immediately afterwards.
    # Release SAM3 first so this debug path fits on the robot GPU too.
    del segmentor
    import gc
    import torch
    gc.collect()
    torch.cuda.empty_cache()
    return masks


def _overlay_detected_masks(images: ImageDict, masks: dict[str, np.ndarray],
                            alpha: float) -> tuple[ImageDict, int]:
    """Paint undistorted detection masks red onto capture frames."""
    overlaid = {}
    n_masks = 0
    for serial, image in images.images.items():
        frame = image.copy()
        mask = masks.get(serial)
        if mask is not None:
            inside = mask > 127
            frame[inside] = (
                frame[inside].astype(np.float32) * (1.0 - alpha)
                + MASK_BGR * alpha
            ).astype(np.uint8)
            n_masks += 1
        overlaid[serial] = frame
    return ImageDict(overlaid, images.intrinsic, images.extrinsic, images.path), n_masks


def _project_mesh_cpu(images: ImageDict, mesh: trimesh.Trimesh,
                      color: tuple[int, int, int], alpha: float) -> ImageDict:
    """Rasterize a transformed mesh silhouette without CUDA.

    This diagnostic fallback is intentionally a silhouette renderer, not a
    photorealistic mesh renderer: every projected triangle contributes to one
    object mask, then that mask is alpha blended onto the undistorted capture
    frame.  That is exactly what an Object6D alignment overlay needs, and lets
    us inspect a saved episode while a live robot process owns the GPU.

    It expects ``mesh`` to already be expressed in the calibration world
    frame, just like :meth:`ImageDict.project_mesh`.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2:
        raise ValueError("mesh has no usable triangle geometry")
    vertices_h = np.concatenate(
        [vertices, np.ones((len(vertices), 1), dtype=np.float64)], axis=1)
    color_arr = np.asarray(color, dtype=np.float32)
    rendered = {}
    for serial, image in images.images.items():
        intr = images.intrinsic.get(serial)
        extr = images.extrinsic.get(serial)
        if intr is None or extr is None:
            rendered[serial] = image.copy()
            continue
        K = np.asarray(intr["intrinsics_undistort"], dtype=np.float64)
        T_cam_world = np.asarray(extr, dtype=np.float64)
        if T_cam_world.shape == (3, 4):
            T_cam_world = np.vstack([T_cam_world, [0.0, 0.0, 0.0, 1.0]])
        if K.shape != (3, 3) or T_cam_world.shape != (4, 4):
            raise ValueError(f"invalid calibration for camera {serial}")
        cam = (T_cam_world @ vertices_h.T).T[:, :3]
        z = cam[:, 2]
        pixels_h = (K @ cam.T).T
        pixels = pixels_h[:, :2] / np.maximum(pixels_h[:, 2:3], 1e-9)
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        # An object pose fed to Object6D is fully in front of these cameras.
        # Skip a rare behind-camera triangle rather than allowing its projective
        # divide to draw across a whole frame.
        for face in faces:
            if np.any(z[face] <= 1e-6) or not np.isfinite(pixels[face]).all():
                continue
            polygon = np.rint(pixels[face]).astype(np.int32)
            cv2.fillConvexPoly(mask, polygon, 255, lineType=cv2.LINE_AA)
        frame = image.copy().astype(np.float32)
        inside = mask > 0
        frame[inside] = frame[inside] * (1.0 - alpha) + color_arr * alpha
        rendered[serial] = frame.astype(np.uint8)
    return ImageDict(rendered, images.intrinsic, images.extrinsic, images.path)


def _load_mesh(obj: str, version: str) -> trimesh.Trimesh:
    """Load the same object-frame mesh used by Object6D when available."""
    obj_root = Path(get_obj_root(version))
    raw_mesh = obj_root / obj / "raw_mesh" / f"{obj}.obj"
    mesh_path = raw_mesh if raw_mesh.exists() else Path(find_planning_mesh(obj, str(obj_root)))
    mesh = trimesh.load(mesh_path, process=False)
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"could not load a mesh from {mesh_path}")
    return mesh


def _object_from_result(trial: Path) -> str | None:
    result_path = trial / "result.json"
    if not result_path.exists():
        return None
    with result_path.open() as f:
        return json.load(f).get("object")


def _resolve_pose(trial: Path, source: str) -> tuple[np.ndarray, str]:
    """Return the pose to draw and a description of where it came from.

    A run whose silhouette loss exceeded the gate never writes
    ``pose_world.npy``: the orchestrator returns None and keeps only the
    pre-refinement candidate in ``result.json``.  That rejected run is exactly
    the one an operator needs to look at, so fall back to ``pre_sil_pose``
    instead of refusing.  It is the FoundPose candidate BEFORE silhouette
    refinement -- the refined pose is not retained anywhere.
    """
    pose_path = trial / "pose_world.npy"
    if source in ("auto", "refined") and pose_path.exists():
        return np.load(pose_path), "pose_world.npy (accepted, refined)"
    if source == "refined":
        raise FileNotFoundError(f"Object6D pose not found: {pose_path}")

    result_path = trial / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"neither {pose_path} nor {result_path} exists")
    with result_path.open() as f:
        result = json.load(f)
    perception = result.get("perception") or {}
    pre_sil = perception.get("pre_sil_pose")
    if pre_sil is None:
        raise FileNotFoundError(
            f"{pose_path} is missing and {result_path} has no pre_sil_pose "
            f"(reason: {perception.get('reason', result.get('reason'))})")
    return np.asarray(pre_sil, dtype=np.float64), (
        "result.json pre_sil_pose — PRE-refinement candidate, run rejected: "
        f"{perception.get('reason', 'unknown')}")


def overlay_trial(trial: Path, obj: str, version: str, out_path: Path,
                  alpha: float, mask_alpha: float, recompute_mask: bool,
                  prompt: str, mask_only: bool,
                  pose_source: str = "auto",
                  renderer: str = "gpu") -> tuple[Path, int]:
    # The FoundPose backend saves its capture under init_capture; the
    # DA3 + FoundationPose backend writes da3_capture instead. Take whichever
    # this trial actually produced.
    capture_dir = next((trial / name for name in ("init_capture", "da3_capture")
                        if (trial / name / "images").is_dir()),
                       trial / "init_capture")
    pose, pose_origin = (None, "not loaded (--mask-only)") if mask_only else \
        _resolve_pose(trial, pose_source)
    print(f"[overlay] pose source: {pose_origin}")
    if not (capture_dir / "images").is_dir():
        raise FileNotFoundError(f"init images not found: {capture_dir / 'images'}")
    if not (trial / "cam_param").is_dir():
        raise FileNotFoundError(f"camera calibration not found: {trial / 'cam_param'}")

    # Capture PCs save already-undistorted RGB frames and SAM3 runs on those
    # same frames. BatchRenderer uses intrinsics_undistort, so do not remap a
    # second time here.
    images = ImageDict.from_path(capture_dir)
    intrinsic, extrinsic = load_camparam(trial)
    # Some nvdiffrast builds (including gotrack_cu128) require float32 clip
    # positions. Keep camera matrices at that dtype before BatchRenderer turns
    # them into CUDA tensors.
    for values in intrinsic.values():
        values["original_intrinsics"] = np.asarray(
            values["original_intrinsics"], dtype=np.float32)
        values["intrinsics_undistort"] = np.asarray(
            values["intrinsics_undistort"], dtype=np.float32)
        values["dist_params"] = np.asarray(values["dist_params"], dtype=np.float32)
    extrinsic = {serial: np.asarray(value, dtype=np.float32)
                 for serial, value in extrinsic.items()}
    images.intrinsic = intrinsic
    images.extrinsic = extrinsic
    masks = _saved_masks(images, capture_dir / "masks")
    if recompute_mask:
        masks = _recompute_masks(images, prompt)
    images, n_masks = _overlay_detected_masks(images, masks, mask_alpha)

    if mask_only:
        overlay = images
    else:
        mesh = _load_mesh(obj, version)
        mesh.apply_transform(pose)
        if renderer == "cpu":
            print("[overlay] renderer=cpu (silhouette fallback)")
            overlay = _project_mesh_cpu(images, mesh, color=(0, 255, 0), alpha=alpha)
        else:
            overlay = images.project_mesh(mesh, color=(0, 255, 0), alpha=alpha)
    grid = overlay.merge()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(out_path), grid):
        raise RuntimeError(f"failed to write {out_path}")
    return out_path, n_masks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", required=True,
                        help="demo result directory containing pose_world.npy")
    parser.add_argument("--obj", default=None,
                        help="object name (defaults to result.json's object field)")
    parser.add_argument("--version", default="v8",
                        help="asset version used for the Object6D mesh (default: v8)")
    parser.add_argument("--out", default=None,
                        help="output grid PNG (default: <trial>/object6d_overlay.png)")
    parser.add_argument("--alpha", type=float, default=0.55,
                        help="green Object6D mesh opacity, in [0, 1]")
    parser.add_argument("--mask-alpha", type=float, default=0.35,
                        help="red detected-mask opacity, in [0, 1]")
    parser.add_argument("--recompute-mask", action="store_true",
                        help="re-run SAM3 on this trial's saved images when its "
                             "original masks were not retained; never writes masks")
    parser.add_argument("--prompt", default="object on the checkerboard",
                        help="SAM3 prompt for --recompute-mask")
    parser.add_argument("--pose-source", choices=["auto", "refined", "pre_sil"],
                        default="auto",
                        help="auto = the accepted refined pose, falling back to "
                             "result.json's pre-refinement candidate when the "
                             "run was rejected by the silhouette gate; refined "
                             "= fail instead of falling back; pre_sil = always "
                             "the candidate")
    parser.add_argument("--mask-only", action="store_true",
                        help="write only the red detected-mask overlay; skip "
                             "Object6D mesh rasterization")
    parser.add_argument("--renderer", choices=["gpu", "cpu"], default="gpu",
                        help="mesh overlay renderer; cpu draws the projected mesh "
                             "silhouette and is useful while a live run owns CUDA")
    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0 or not 0.0 <= args.mask_alpha <= 1.0:
        parser.error("--alpha and --mask-alpha must be in [0, 1]")
    trial = Path(args.trial).expanduser().resolve()
    obj = args.obj or _object_from_result(trial)
    if not obj:
        parser.error("pass --obj because result.json does not provide an object")
    out_path = (Path(args.out).expanduser().resolve() if args.out
                else trial / "object6d_overlay.png")
    output, n_masks = overlay_trial(trial, obj, args.version, out_path,
                                    args.alpha, args.mask_alpha,
                                    args.recompute_mask, args.prompt,
                                    args.mask_only, args.pose_source,
                                    args.renderer)
    print(f"[overlay] object={obj} trial={trial}")
    print(f"[overlay] detected masks={n_masks}; expected under "
          f"{trial / 'init_capture' / 'masks'}")
    print(f"[overlay] wrote {output}")


if __name__ == "__main__":
    main()
