"""Non-disruptive camera snapshot helpers for the continuous demo."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Mapping


def capture_catalog_snapshot(
    rcc,
    destination: Path,
    *,
    min_images: int,
    settle_timeout_s: float = 2.0,
) -> int:
    """Request a one-shot snapshot without stopping the live capture stream.

    ParaDex's ``snapshot`` sink writes the next frame from every active camera
    while the acquire/stream session stays alive.  The former stop → image →
    stop → restart sequence added avoidable latency and could make a camera
    daemon lose its stream state between continuous pick cycles.
    """
    if min_images < 1:
        raise ValueError("min_images must be >= 1")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    rel = os.path.relpath(destination, Path.home())
    response = rcc.snapshot(rel, count=1)
    if isinstance(response, Mapping):
        failed = {
            pc: info.get("msg", "unknown error")
            for pc, info in response.items()
            if isinstance(info, Mapping) and info.get("status") not in (None, "ok")
        }
        if failed:
            raise RuntimeError(f"ParaDex snapshot rejected: {failed}")

    image_dir = destination / "images"
    deadline = time.monotonic() + float(settle_timeout_s)
    while time.monotonic() < deadline:
        count = sum(1 for _ in image_dir.glob("*.png"))
        if count >= min_images:
            return count
        time.sleep(0.05)
    count = sum(1 for _ in image_dir.glob("*.png"))
    raise TimeoutError(
        f"ParaDex snapshot wrote {count}/{min_images} images under {image_dir}"
    )
