"""Pick grasps that already SUCCEEDED at the object's current tabletop pose.

The demo does not search the candidate pool: the professor's run only has to
show the pick-and-place working, so we replay grasps that are known good. A
grasp is "known good" when its candidate dir holds ``result.json`` with
``success=true`` for this arm, and the coverage JSON lists it at the same
``pose_idx`` (= tabletop pose stem) the object is lying in right now.

Scene obstacles are IGNORED on purpose — the candidate's original scene (wall/
shelf/box) only decided where it was generated; the demo table is bare, so a
grasp that worked there is still valid geometry against the object.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

from autodex.utils.coverage import _coverage_path, _disk_success_keys

Key = Tuple[str, str, str]


def success_keys_at_pose(obj_name: str, hand: str, version: str,
                         pose_stem: Optional[str],
                         arm: Optional[str] = None) -> Tuple[List[Key], List[Key]]:
    """Return ``(keys_at_pose, keys_any_pose)`` — both success-only.

    ``keys_at_pose`` are the successes whose coverage ``pose_idx`` equals
    ``pose_stem``; ``keys_any_pose`` is every success for this arm, used as the
    fallback when the current tabletop pose has none. Keys are
    ``(scene_type, scene_id, grasp_id)``, i.e. what ``candidate_order`` wants.
    """
    succ = _disk_success_keys(obj_name, hand, version, arm=arm)
    if not succ:
        return [], []

    cov_path = _coverage_path(obj_name, version)
    if not os.path.exists(cov_path):
        # No coverage file -> no pose_idx to filter on; everything is "any pose".
        return [], sorted(succ)
    with open(cov_path) as f:
        grasps = json.load(f).get("grasps") or []

    at_pose, any_pose = [], []
    for g in grasps:
        key = (str(g["type"]), str(g["sid"]), str(g["gid"]))
        if key not in succ:
            continue
        any_pose.append(key)
        if pose_stem is not None and str(g.get("pose_idx", "")) == str(pose_stem):
            at_pose.append(key)
    return at_pose, any_pose
