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


# Charuco board the object sits on — the one success labelling checks for.
# The floor board was swapped to the 10x7 (paradex calls it board "11":
# 6X6_250, marker ids 70-104, 54 corners); the old 5x6 "1" is no longer on the
# table, and asking for it makes every trial read as FAIL because no camera
# ever detects it. paradex's own hand-eye solve tracks the same id in
# src/calibration/handeye/calculate.py:FLOOR_BOARD — keep the two in step.
CHARUCO_BOARD = "11"

def auto_label_charuco(image_dir: str,
                       required_board: str = CHARUCO_BOARD) -> Tuple[Optional[bool], dict]:
    """Multi-view charuco union check.

    Returns (success_or_None, info). success=True iff `required_board` has all
    its corners detected across the union of all camera images in `image_dir`.
    success=None if no images found.
    """
    import cv2
    from paradex.image.aruco import detect_charuco, boardinfo_dict

    # A capture that wrote nothing leaves the directory ABSENT, not empty, so
    # iterdir() raises FileNotFoundError and takes the whole run down with it.
    # Missing and empty mean the same thing here: no images to label from.
    d = Path(image_dir)
    if not d.is_dir():
        return None, {"reason": "no_images", "image_dir": str(image_dir),
                      "detail": "directory does not exist"}
    paths = sorted(p for p in d.iterdir()
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
