"""Small, deterministic SAM3-mask crops for the P2 semantic router.

This module intentionally knows nothing about FoundPose, object identity, or
the robot.  It only turns one RGB frame and its SAM3 mask into the neutral
background crop consumed by the VLM.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np


P2_CROP_SIZE = 448
P2_MASK_BORDER_PX = 16
P2_BACKGROUND_RGB = (127, 127, 127)


@dataclass(frozen=True)
class SemanticCropInfo:
    """Metadata needed to reproduce one P2 semantic crop."""

    bbox_xywh: tuple[int, int, int, int]
    mask_pixels: int
    image_hw: tuple[int, int]
    crop_side_px: int
    border_margin_px: int

    def jsonable(self) -> dict[str, Any]:
        value = asdict(self)
        value["bbox_xywh"] = list(self.bbox_xywh)
        value["image_hw"] = list(self.image_hw)
        return value


def mask_bbox(mask: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]] | None:
    """Return the full binary mask and its ``(x, y, w, h)`` bbox.

    P2 deliberately performs no connected-component or quality filtering.  A
    mask that has any foreground touching an edge is rejected by the single
    protocol check below.
    """
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2 or not binary.any():
        return None
    ys, xs = np.nonzero(binary)
    x = int(xs.min()); y = int(ys.min())
    w = int(xs.max() - x + 1); h = int(ys.max() - y + 1)
    return binary, (x, y, w, h)


def make_semantic_crop(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    *,
    border_px: int = P2_MASK_BORDER_PX,
    output_size: int = P2_CROP_SIZE,
) -> tuple[np.ndarray, SemanticCropInfo] | None:
    """Build the P2 crop, or return ``None`` when the object touches an edge.

    The sole acceptance check is exactly the P2 protocol rule: the foreground
    mask bbox must be at least ``border_px`` inside all four image borders. No
    area, connected-component, shape, pose, or cross-view screening is done.
    """
    rgb = np.asarray(image_rgb)
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError("image_rgb must have shape HxWx3")
    if output_size <= 0 or border_px < 0:
        raise ValueError("output_size must be positive and border_px non-negative")
    mask_and_bbox = mask_bbox(mask)
    if mask_and_bbox is None:
        return None
    foreground, (x, y, w, h) = mask_and_bbox
    image_h, image_w = rgb.shape[:2]
    right = image_w - (x + w)
    bottom = image_h - (y + h)
    margin = min(x, y, right, bottom)
    if margin < border_px:
        return None

    # A square, 1.5x-longest-side crop places a 25% longest-side border around
    # the mask bbox.  Pad before slicing so the fixed crop rule never changes
    # merely because the optional context extends beyond the source frame.
    side = max(1, int(np.ceil(1.5 * max(w, h))))
    center_x = x + (w - 1) / 2.0
    center_y = y + (h - 1) / 2.0
    x0 = int(np.floor(center_x - side / 2.0))
    y0 = int(np.floor(center_y - side / 2.0))

    neutral = np.empty_like(rgb)
    neutral[...] = np.asarray(P2_BACKGROUND_RGB, dtype=rgb.dtype)
    neutral[foreground] = rgb[foreground]
    pad = side + max(abs(x0), abs(y0),
                     max(0, x0 + side - image_w),
                     max(0, y0 + side - image_h))
    padded = cv2.copyMakeBorder(neutral, pad, pad, pad, pad,
                                cv2.BORDER_CONSTANT, value=P2_BACKGROUND_RGB)
    crop = padded[y0 + pad:y0 + pad + side, x0 + pad:x0 + pad + side]
    if crop.shape[:2] != (side, side):  # defensive; padding above should ensure this.
        raise RuntimeError("semantic crop unexpectedly escaped padded image")
    resized = cv2.resize(crop, (output_size, output_size), interpolation=cv2.INTER_AREA)
    return resized, SemanticCropInfo(
        bbox_xywh=(x, y, w, h), mask_pixels=int(foreground.sum()),
        image_hw=(image_h, image_w), crop_side_px=side,
        border_margin_px=int(margin),
    )
