"""Pure retry and pose-evidence policy for a continuous basket demo."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence, Tuple

import numpy as np


class Verification(Enum):
    HELD = "held"
    NOT_HELD = "not_held"
    UNCERTAIN = "uncertain"
    IN_BASKET = "in_basket"
    NOT_IN_BASKET = "not_in_basket"


@dataclass(frozen=True)
class RetryDecision:
    retry: bool
    terminal: bool
    candidate_order: Tuple[Tuple[str, str, str], ...]
    reason: str


@dataclass
class LocalRetryPolicy:
    """Retry a different grasp at the observed object pose, never home-reset.

    ``candidate_order`` is the pre-ranked, pose-compatible success-grasp pool.
    After a failed candidate it is removed before replanning, so repeated
    attempts add information rather than replaying the exact same motion.
    """

    candidate_order: Sequence[Tuple[str, str, str]]
    max_attempts: int = 3
    attempted: List[Tuple[str, str, str]] = field(default_factory=list)

    def next_after_failure(self, candidate: Optional[Tuple[str, str, str]],
                           verification: Verification) -> RetryDecision:
        if candidate is not None and candidate not in self.attempted:
            self.attempted.append(candidate)
        remaining = tuple(key for key in self.candidate_order if key not in self.attempted)
        if verification is Verification.UNCERTAIN:
            # A lost camera view is not proof the hand is empty.  Keeping the
            # arm in a safe raised pose and asking for intervention is safer
            # than blindly descending onto a possibly held object.
            return RetryDecision(False, True, remaining, "grasp_state_uncertain")
        if len(self.attempted) >= self.max_attempts:
            return RetryDecision(False, True, remaining, "retry_budget_exhausted")
        if not remaining:
            return RetryDecision(False, True, remaining, "no_untried_candidates")
        return RetryDecision(True, False, remaining, "retry_same_object_in_place")


@dataclass(frozen=True)
class PoseEvidence:
    """A selected object's robot-frame translation after a verification init."""

    xyz_robot: Optional[Tuple[float, float, float]]
    quality: float = 0.0


@dataclass(frozen=True)
class PoseVerifier:
    """Conservative automatic success checks from 6D-pose re-observations."""

    lift_delta_z_m: float = 0.045
    same_spot_xy_m: float = 0.045
    basket_radius_m: float = 0.11

    def after_lift(self, before: PoseEvidence, after: PoseEvidence) -> Verification:
        if before.xyz_robot is None or after.xyz_robot is None:
            return Verification.UNCERTAIN
        start = np.asarray(before.xyz_robot, dtype=float)
        now = np.asarray(after.xyz_robot, dtype=float)
        if now[2] >= start[2] + self.lift_delta_z_m:
            return Verification.HELD
        if np.linalg.norm(now[:2] - start[:2]) <= self.same_spot_xy_m:
            return Verification.NOT_HELD
        return Verification.UNCERTAIN

    def after_drop(self, observed: PoseEvidence,
                   basket_center_xy: Tuple[float, float]) -> Verification:
        if observed.xyz_robot is None:
            return Verification.UNCERTAIN
        xy = np.asarray(observed.xyz_robot[:2], dtype=float)
        center = np.asarray(basket_center_xy, dtype=float)
        if np.linalg.norm(xy - center) <= self.basket_radius_m:
            return Verification.IN_BASKET
        return Verification.NOT_IN_BASKET
