"""Immutable P2 catalogue shared by collection and the later basket demo.

P2 deliberately changes one evaluation axis: object identity.  Collection
therefore keeps using ``src/execution/run_auto.py``'s regular v8 workflow;
the semantic destination is recorded here and is consumed only by the
pick-and-place runner.  Keeping this mapping out of either runner avoids a
collection result silently changing its intended basket.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Keep the preflight usable from a base Python environment.  The robot runner
# imports the canonical resolver; the fallback has the identical documented
# hierarchy and only avoids pulling trimesh/cuRobo into a filesystem check.
try:
    from autodex.utils.path import get_candidate_path as _get_candidate_path
    from autodex.utils.path import project_dir as _project_dir
except ModuleNotFoundError:
    _project_dir = str(Path.home() / "shared_data/AutoDex")

    def _get_candidate_path(hand: str) -> str:
        return str(Path(_project_dir) / "candidates" / hand)


P2_PROTOCOL_ID = "p2_object_diversity_semantic_routing"
# P2 is currently collecting physical execution data into v8.  Keep this
# explicit so a later demo never mistakes the collection namespace for a
# frozen benchmark result set.
P2_COLLECTION_STATE = "collecting_v8_inspire"
P2_GRASP_VERSION = "v8"
P2_HAND = "inspire"
# This is an execution default only.  It must not become part of candidate
# lookup: a grasp is tied to an object and a hand, then revalidated for the
# execution arm by planning/IK.
P2_COLLECTION_ARM = "franka"
P2_GRASP_LOOKUP_FIELDS = ("object", "hand", "grasp_version")
P2_PROJECT_ROOT = Path(_project_dir)
P2_FOUNDPOSE_ROOT = P2_PROJECT_ROOT / "foundpose_assets"


@dataclass(frozen=True)
class P2Object:
    """One fixed P2 object and its semantic destination."""

    name: str
    semantic_class: str
    basket: str


# The basket labels are intentionally physical labels, not robot-frame
# directions.  The later demo owns their calibrated robot poses.
P2_OBJECTS: tuple[P2Object, ...] = (
    P2Object("apple", "fruit", "left"),
    P2Object("banana", "fruit", "left"),
    P2Object("pringles", "nonfruit", "right"),
    P2Object("spam_can", "nonfruit", "right"),
)
P2_OBJECT_BY_NAME = {item.name: item for item in P2_OBJECTS}

# These are final *object bearings* in the robot frame, not J0 increments:
# +X is forward and positive is counter-clockwise.  The inference helper
# recomputes the held-object bearing by FK after the actual lift, then changes
# only J0 until it reaches the selected bearing.
P2_CLASS_ROUTES = {
    "FRUIT": {"semantic_class": "fruit", "basket": "left", "bearing_deg": 50.0},
    "NON_FRUIT": {"semantic_class": "nonfruit", "basket": "right", "bearing_deg": -30.0},
}
P2_REQUIRED_SEMANTIC_CROPS = 3


def candidate_lookup_path(name: str, hand: str = P2_HAND,
                          grasp_version: str = P2_GRASP_VERSION,
                          candidate_root: Path | str | None = None) -> Path:
    """Candidate pool path, deliberately independent of the arm type.

    With no override this resolves to the same hierarchy as
    ``get_candidate_path(hand)`` in ``run_auto.py``.  ``candidate_root`` is
    only a preflight/testing override for the parent
    ``.../AutoDex/candidates`` directory.  The runtime planner, not lookup,
    checks whether a hand grasp is reachable by a particular arm.
    """
    get_p2_object(name)
    base = Path(candidate_root) / hand if candidate_root is not None else Path(
        _get_candidate_path(hand))
    return base / grasp_version / name


def collection_result_root(name: str, hand: str = P2_HAND,
                           grasp_version: str = P2_GRASP_VERSION,
                           project_root: Path | str | None = None) -> Path:
    """Parent of P2 collection episodes in the normal ``run_auto`` layout.

    For the P2 command's default table scene this is exactly
    ``{project_dir}/experiment/v8/inspire/{object}``.  A timestamp child and
    ``result.json`` are created only by ``run_auto.py``; P2 never creates a
    competing output hierarchy.
    """
    get_p2_object(name)
    root = P2_PROJECT_ROOT if project_root is None else Path(project_root)
    return root / "experiment" / grasp_version / hand / name


def coverage_json_path(name: str, grasp_version: str = P2_GRASP_VERSION,
                       project_root: Path | str | None = None) -> Path:
    """Existing coverage location used by ``autodex.utils.coverage``."""
    get_p2_object(name)
    root = P2_PROJECT_ROOT if project_root is None else Path(project_root)
    return root / "experiment" / grasp_version / "coverage" / (
        f"cov_{grasp_version}_cand_{name}.json")


def foundpose_repre_path(name: str,
                         foundpose_root: Path | str | None = None) -> Path:
    """Existing FoundPose representation location for the object."""
    get_p2_object(name)
    root = P2_FOUNDPOSE_ROOT if foundpose_root is None else Path(foundpose_root)
    return root / name / "object_repre/v1" / name / "1/repre.pth"


def get_p2_object(name: str) -> P2Object:
    """Return a P2 object or explain why it is excluded from this protocol."""
    try:
        return P2_OBJECT_BY_NAME[name]
    except KeyError as exc:
        allowed = ", ".join(P2_OBJECT_BY_NAME)
        raise ValueError(f"{name!r} is not a P2 object (allowed: {allowed})") from exc


def basket_for(name: str) -> str:
    """Return the physical P2 basket label for an approved object."""
    return get_p2_object(name).basket
