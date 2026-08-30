"""Fixed known-object catalogue recognition for the continuous demo.

YOLO-E is used only to decide *which one* of a small, pre-onboarded catalogue
is on the table.  Once selected, the normal distributed FoundPose path owns
6D pose estimation.  This avoids serially running FoundPose for every known
object, which is the main multi-object latency trap.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class CatalogObject:
    """A demo-supported object and its open-vocabulary detector prompt."""

    name: str
    prompt: str


@dataclass(frozen=True)
class CatalogMatch:
    """Aggregate evidence for one catalogue item across camera views."""

    name: str
    prompt: str
    score: float
    supporting_views: int
    best_view_score: float


def parse_catalog(values: Sequence[str]) -> List[CatalogObject]:
    """Parse ``name`` or ``name=detector prompt`` CLI values.

    Names must be unique because they map directly to AutoDex assets and grasp
    pools.  A small fixed catalogue (3--6 objects) is deliberate for the
    first uncut demo: it makes detection latency predictable and allows all
    FoundPose/GoTrack assets to be checked before the camera starts rolling.
    """
    out: List[CatalogObject] = []
    seen = set()
    for raw in values:
        name, sep, prompt = raw.partition("=")
        name = name.strip()
        prompt = prompt.strip() if sep else name
        if not name or not prompt:
            raise ValueError(f"invalid catalogue item {raw!r}; use name or name=prompt")
        if name in seen:
            raise ValueError(f"duplicate catalogue object: {name}")
        seen.add(name)
        out.append(CatalogObject(name=name, prompt=prompt))
    if not out:
        raise ValueError("the catalogue must contain at least one object")
    return out


def rank_catalog_detections(
    detections: Mapping[str, Sequence[Optional[Sequence[tuple[np.ndarray, float]]]]],
    catalogue: Sequence[CatalogObject],
    *,
    min_views: int = 2,
    min_score: float = 0.25,
) -> List[CatalogMatch]:
    """Score per-prompt YOLO-E results without depending on the model class.

    A view contributes its highest instance confidence.  Averaging those
    confidences rewards agreement across cameras, while the hard view count
    prevents a single reflection from selecting a wrong mesh.
    """
    if min_views < 1:
        raise ValueError("min_views must be >= 1")
    scored: List[CatalogMatch] = []
    for item in catalogue:
        view_scores: List[float] = []
        for masks in detections.get(item.name, []):
            if masks:
                view_scores.append(float(max(conf for _mask, conf in masks)))
        support = len(view_scores)
        best = max(view_scores, default=0.0)
        score = float(np.mean(view_scores)) if view_scores else 0.0
        if support >= min_views and score >= min_score:
            scored.append(CatalogMatch(item.name, item.prompt, score, support, best))
    return sorted(scored, key=lambda m: (-m.score, -m.supporting_views, m.name))


def read_capture_images(capture_dir: Path, serials: Optional[Iterable[str]] = None) -> Dict[str, np.ndarray]:
    """Load RGB snapshot images written by ParaDex's image capture sink."""
    import cv2

    image_dir = Path(capture_dir) / "images"
    wanted = set(serials) if serials is not None else None
    images: Dict[str, np.ndarray] = {}
    for path in sorted(image_dir.glob("*.png")):
        if wanted is not None and path.stem not in wanted:
            continue
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is not None:
            images[path.stem] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return images


class CatalogRecognizer:
    """One preloaded YOLO-E model used for a complete catalogue scan."""

    def __init__(self, *, gpu: int = 0, conf_threshold: float = 0.25):
        # Import lazily: pure policy tests and planning-only work must not need
        # ultralytics, CUDA, or the YOLO-E weights.
        from autodex.perception.mask import YoloeSegmentor

        self._segmentor = YoloeSegmentor(gpu=gpu, conf_thr=conf_threshold)

    def identify(
        self,
        images: Mapping[str, np.ndarray],
        catalogue: Sequence[CatalogObject],
        *,
        min_views: int = 2,
        min_score: float = 0.25,
    ) -> tuple[Optional[CatalogMatch], List[CatalogMatch]]:
        """Return the best recognised object and all accepted alternatives."""
        if not images:
            return None, []
        ordered_images = [images[s] for s in sorted(images)]
        raw: Dict[str, Sequence[Optional[Sequence[tuple[np.ndarray, float]]]]] = {}
        for item in catalogue:
            raw[item.name] = self._segmentor.segment_batch(ordered_images, item.prompt)
        ranked = rank_catalog_detections(raw, catalogue,
                                         min_views=min_views, min_score=min_score)
        return (ranked[0] if ranked else None), ranked

    def close(self) -> None:
        """Release detector VRAM before FoundPose/planning starts."""
        self._segmentor = None
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
