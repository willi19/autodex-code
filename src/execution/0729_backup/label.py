"""Trial labeling: manual prompt + charuco-based auto-label.

Auto-label rule: success iff every corner of the required charuco board is
detected by at least one camera (multi-view union). Captured at the moment
the object is lifted up — if the grasp succeeded the table is clear of the
object and the charuco below is visible.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import chime


def get_label():
    """Returns (success: bool|None, note: str|None).

    y       = success
    n       = fail
    c       = issue / skip (success=None)
    ym / nm = with memo
    """
    while True:
        chime.success()
        label = input("Label [y/n/c=issue / ym/nm=with memo]: ").strip().lower()
        if label == "y":
            return True, None
        if label == "ym":
            note = input("  Note: ").strip()
            return True, note or None
        if label == "n":
            return False, None
        if label == "nm":
            note = input("  Note: ").strip()
            return False, note or None
        if label == "c":
            note = input("  Note: ").strip()
            return None, note or "issue"


def auto_label_charuco(image_dir: str, required_board: str = "1") -> Tuple[Optional[bool], dict]:
    """Multi-view charuco union check.

    Returns (success_or_None, info). success=True iff `required_board` has all
    its corners detected across the union of all camera images in `image_dir`.
    success=None if no images found.
    """
    import cv2
    from paradex.image.aruco import detect_charuco, boardinfo_dict

    paths = sorted(p for p in Path(image_dir).iterdir()
                   if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not paths:
        return None, {"reason": "no_images", "image_dir": str(image_dir)}

    cfg = boardinfo_dict.get(required_board)
    if not cfg:
        return None, {"reason": f"board_{required_board}_not_in_config"}
    expected = (cfg["numX"] - 1) * (cfg["numY"] - 1)

    union: set = set()
    per_cam: dict = {}
    for fp in paths:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        det = detect_charuco(img)
        info = det.get(required_board)
        if info is None:
            per_cam[fp.stem] = 0
            continue
        ids = info["checkerIDs"].tolist()
        per_cam[fp.stem] = len(ids)
        union.update(ids)

    covered = len(union)
    success = covered == expected
    return success, {
        "board": required_board,
        "covered": covered,
        "expected": expected,
        "missing_ids": sorted(set(range(expected)) - union),
        "per_cam": per_cam,
        "n_cameras": len(paths),
    }
