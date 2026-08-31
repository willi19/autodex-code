"""Non-disruptive camera snapshot helpers for the continuous demo."""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Iterable, Mapping, Optional


def capture_catalog_snapshot(
    rcc,
    destination: Path,
    *,
    min_images: int,
    settle_timeout_s: float = 15.0,
    expected_serials: Optional[Iterable[str]] = None,
    require_decodable: bool = False,
) -> int:
    """Request a one-shot snapshot without stopping the live capture stream.

    ParaDex's ``snapshot`` sink writes the next frame from every active camera
    while the acquire/stream session stays alive.  The former stop → image →
    stop → restart sequence added avoidable latency and could make a camera
    daemon lose its stream state between continuous pick cycles.  The timeout
    is deliberately a failure bound, not a fixed delay: return as soon as the
    requested number of NFS-visible images arrives.  A full 20-camera snapshot
    was observed to land over roughly ten seconds on the lab NAS, so the old
    two-second bound was a false failure mode.

    When ``expected_serials`` is supplied, this process pre-creates zero-byte
    entries for the known camera filenames before issuing the remote command.
    This avoids an NFS negative-directory cache: otherwise a client that polls
    a not-yet-created ``images`` directory can keep seeing zero files after
    the capture PCs have written them.  Only non-empty files count as ready.
    ``require_decodable`` additionally waits until PNG writers have closed
    their files.  This is necessary before ArUco/FoundPose reads them: a
    non-zero NFS file can still be a partial PNG for a short interval.
    """
    if min_images < 1:
        raise ValueError("min_images must be >= 1")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    image_dir = destination / "images"
    expected_paths = None
    if expected_serials is not None:
        serials = tuple(dict.fromkeys(str(serial) for serial in expected_serials))
        if len(serials) < min_images:
            raise ValueError("expected_serials has fewer entries than min_images")
        image_dir.mkdir(parents=True, exist_ok=True)
        expected_paths = []
        for serial in serials:
            filename = f"{serial}.png"
            path = image_dir / filename
            if path.name != filename:
                raise ValueError(f"invalid camera serial for snapshot: {serial!r}")
            path.touch(exist_ok=True)
            expected_paths.append(path)

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

    def _is_decodable(path: Path) -> bool:
        if not require_decodable:
            return True
        try:
            # Pillow's load reads the image data (not just the header), so it
            # rejects a capture-PC PNG that is still being written.  Keep this
            # lazy to retain the lightweight camera-smoke dependency surface.
            from PIL import Image
            with Image.open(path) as image:
                image.load()
            return True
        except (OSError, ValueError):
            return False

    def ready_count() -> int:
        candidates = expected_paths if expected_paths is not None else image_dir.glob("*.png")
        return sum(
            1 for path in candidates
            if path.is_file() and path.stat().st_size > 0 and _is_decodable(path)
        )

    deadline = time.monotonic() + float(settle_timeout_s)
    while time.monotonic() < deadline:
        count = ready_count()
        if count >= min_images:
            return count
        time.sleep(0.05)
    count = ready_count()
    raise TimeoutError(
        f"ParaDex snapshot wrote {count}/{min_images} images under {image_dir}"
    )
