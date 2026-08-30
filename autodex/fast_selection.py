"""Dependency-light pose selection helpers for latency-sensitive loops."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


def select_best_pose_by_quality(
    candidates: Dict[str, np.ndarray],
    pose_payloads: Dict[str, Any],
) -> Tuple[Optional[str], Optional[np.ndarray], Dict[str, float]]:
    """Choose a FoundPose candidate without mesh rendering.

    All candidates must belong to the same already-selected object.  The score
    only uses metadata emitted by the capture PCs, so it is deterministic and
    has no CUDA/OpenGL/silhouette dependency.
    """
    if not candidates:
        return None, None, {}

    def metrics(serial: str) -> Tuple[float, int, int]:
        payload = pose_payloads.get(serial) or {}
        try:
            quality = float(payload.get("quality", 0.0))
        except (TypeError, ValueError):
            quality = 0.0
        try:
            inliers = int(payload.get("inliers", 0))
        except (TypeError, ValueError):
            inliers = 0
        try:
            pixels = int(payload.get("mask_pixels", 0))
        except (TypeError, ValueError):
            pixels = 0
        return quality, inliers, pixels

    # Serial is the final tie-breaker, avoiding a dependence on ZMQ arrival
    # order when two cameras report otherwise equal scores.
    best_serial = max(candidates, key=lambda s: (*metrics(s), str(s)))
    scores = {s: metrics(s)[0] for s in candidates}
    return best_serial, np.asarray(candidates[best_serial], dtype=np.float64), scores
