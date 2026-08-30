import unittest

import numpy as np

from autodex.fast_selection import select_best_pose_by_quality
from src.demo.continuous_basket.catalog import (
    parse_catalog,
    rank_catalog_detections,
)
from src.demo.continuous_basket.policy import (
    LocalRetryPolicy,
    PoseEvidence,
    PoseVerifier,
    Verification,
)
from src.demo.continuous_basket.tracking import LiveGoTrackSession


class CatalogPolicyTest(unittest.TestCase):
    def test_catalog_requires_multi_view_agreement(self):
        catalogue = parse_catalog(["banana", "brush=tooth brush"])
        mask = np.ones((2, 2), dtype=np.uint8)
        ranked = rank_catalog_detections(
            {
                "banana": [[(mask, 0.7)], [(mask, 0.6)], None],
                "brush": [[(mask, 0.95)], None, None],
            },
            catalogue, min_views=2, min_score=0.25,
        )
        self.assertEqual([m.name for m in ranked], ["banana"])
        self.assertEqual(ranked[0].supporting_views, 2)

    def test_retry_removes_failed_candidate_without_reset(self):
        keys = [("table", "0", "0"), ("table", "0", "1"), ("table", "0", "2")]
        policy = LocalRetryPolicy(keys, max_attempts=3)
        decision = policy.next_after_failure(keys[0], Verification.NOT_HELD)
        self.assertTrue(decision.retry)
        self.assertEqual(decision.reason, "retry_same_object_in_place")
        self.assertEqual(decision.candidate_order, tuple(keys[1:]))
        self.assertEqual(policy.remaining_candidates(), tuple(keys[1:]))

    def test_uncertain_lift_never_blindly_retries(self):
        policy = LocalRetryPolicy([("table", "0", "0"), ("table", "0", "1")])
        decision = policy.next_after_failure(("table", "0", "0"), Verification.UNCERTAIN)
        self.assertFalse(decision.retry)
        self.assertTrue(decision.terminal)

    def test_pose_verifier_distinguishes_empty_lift_and_held_lift(self):
        verifier = PoseVerifier(lift_delta_z_m=0.04, same_spot_xy_m=0.04)
        before = PoseEvidence((0.5, 0.0, 0.03))
        self.assertEqual(verifier.after_lift(before, PoseEvidence((0.5, 0.0, 0.12))),
                         Verification.HELD)
        self.assertEqual(verifier.after_lift(before, PoseEvidence((0.51, 0.01, 0.03))),
                         Verification.NOT_HELD)

    def test_fast_selector_uses_metadata_not_arrival_order(self):
        candidates = {"late": np.eye(4), "early": np.eye(4) * 2}
        payloads = {
            "late": {"quality": 0.6, "inliers": 4, "mask_pixels": 100},
            "early": {"quality": 0.6, "inliers": 5, "mask_pixels": 10},
        }
        serial, pose, scores = select_best_pose_by_quality(candidates, payloads)
        self.assertEqual(serial, "early")
        self.assertTrue(np.array_equal(pose, candidates["early"]))
        self.assertEqual(scores, {"late": 0.6, "early": 0.6})

    def test_tracking_payload_keeps_undistorted_calibration(self):
        session = LiveGoTrackSession(
            pc_list=["capture1"], capture_ips=["10.0.0.1"],
            intrinsics={"serial": {
                "K_undist": np.eye(3), "K_orig": np.eye(3) * 2,
                "dist_params": np.zeros(5), "width": 640, "height": 480,
            }},
            extrinsics={"serial": np.eye(4)}, anchor_root="/tmp/anchors",
        )
        intrinsics, extrinsics = session._payload_calibration()
        self.assertEqual(intrinsics["serial"]["K"], np.eye(3).tolist())
        self.assertEqual(intrinsics["serial"]["width"], 640)
        self.assertEqual(extrinsics["serial"], np.eye(4).tolist())

if __name__ == "__main__":
    unittest.main()
