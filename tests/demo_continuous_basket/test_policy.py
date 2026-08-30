import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from autodex.fast_selection import select_best_pose_by_quality
from src.demo.continuous_basket.catalog import (
    CatalogRecognizer,
    parse_catalog,
    rank_catalog_detections,
)
from src.demo.continuous_basket.camera import capture_catalog_snapshot
from src.demo.continuous_basket.policy import (
    LocalRetryPolicy,
    PoseEvidence,
    PoseVerifier,
    Verification,
    choose_success_candidates,
)
from src.demo.continuous_basket.preflight import build_report, require_ready
from src.demo.continuous_basket.tracking import LiveGoTrackSession

try:
    from autodex.perception.init_orchestrator import InitOrchestrator
except ModuleNotFoundError:  # lightweight policy env intentionally has no OpenCV
    InitOrchestrator = None

try:
    from autodex.perception.mask import _masks_by_class_from_yoloe
except ModuleNotFoundError:  # lightweight policy env intentionally has no OpenCV
    _masks_by_class_from_yoloe = None


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

    def test_catalog_recognizer_uses_one_multi_prompt_inference(self):
        class FakeSegmentor:
            def __init__(self):
                self.calls = []

            def segment_catalog_batch(self, images, prompts):
                self.calls.append((len(images), tuple(prompts)))
                mask = np.ones((2, 2), dtype=np.uint8)
                return {
                    "banana": [[(mask, 0.7)], [(mask, 0.8)]],
                    "tooth brush": [None, None],
                }

        recognizer = object.__new__(CatalogRecognizer)
        recognizer._segmentor = FakeSegmentor()
        catalogue = parse_catalog(["banana", "brush=tooth brush"])
        selected, _ranked = recognizer.identify(
            {"cam0": np.zeros((2, 2, 3)), "cam1": np.zeros((2, 2, 3))}, catalogue,
        )
        self.assertEqual(selected.name, "banana")
        self.assertEqual(recognizer._segmentor.calls, [(2, ("banana", "tooth brush"))])

    @unittest.skipIf(_masks_by_class_from_yoloe is None,
                     "OpenCV YOLO-E helper environment unavailable")
    def test_multiclass_yoloe_result_keeps_prompt_class_mapping(self):
        class TensorLike:
            def __init__(self, value):
                self.value = np.asarray(value)

            def cpu(self):
                return self

            def numpy(self):
                return self.value

        class FakeBoxes:
            conf = TensorLike([0.4, 0.9])
            cls = TensorLike([1, 0])

            def __len__(self):
                return 2

        class FakeMasks:
            data = [TensorLike(np.ones((2, 2))), TensorLike(np.ones((2, 2)))]

        result = type("Result", (), {"boxes": FakeBoxes(), "masks": FakeMasks()})()
        grouped = _masks_by_class_from_yoloe(result, 2, 2, ["banana", "brush"])
        self.assertEqual([conf for _mask, conf in grouped["banana"]], [0.9])
        self.assertEqual([conf for _mask, conf in grouped["brush"]], [0.4])

    def test_catalog_snapshot_keeps_live_stream_running(self):
        """Latest ParaDex uses its one-shot sink instead of a session restart."""
        class SnapshotOnlyRcc:
            def __init__(self):
                self.calls = []

            def snapshot(self, rel, count=1):
                self.calls.append((rel, count))
                image_dir = (Path.home() / rel / "images").resolve()
                image_dir.mkdir(parents=True, exist_ok=True)
                for idx in range(2):
                    (image_dir / f"camera{idx}.png").touch()
                return {"capture1": {"status": "ok"}}

        with tempfile.TemporaryDirectory() as tmp:
            rcc = SnapshotOnlyRcc()
            count = capture_catalog_snapshot(
                rcc, Path(tmp) / "snapshot", min_images=2, settle_timeout_s=0.2,
            )
        self.assertEqual(count, 2)
        self.assertEqual(len(rcc.calls), 1)
        self.assertEqual(rcc.calls[0][1], 1)

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

    @unittest.skipIf(InitOrchestrator is None, "OpenCV/ZeroMQ init environment unavailable")
    def test_quality_refinement_skips_masks_renderer_and_silhouette(self):
        """The <20s continuous route stays independent of the IoU stack."""
        orch = object.__new__(InitOrchestrator)
        orch.intrinsics_undist = {"best": np.eye(3), "other": np.eye(3)}
        # A sentinel rather than an optimizer proves this route does not touch
        # GPU renderer state or call silhouette optimisation.
        orch._sil = object()
        best_pose = np.eye(4)
        best_pose[0, 3] = 0.42
        pose, timing = orch.refine_from_payloads(
            masks={},
            poses={
                "best": {"ok": True, "pose_world": best_pose,
                         "quality": 0.9, "inliers": 40, "mask_pixels": 900},
                "other": {"ok": True, "pose_world": np.eye(4),
                          "quality": 0.4, "inliers": 80, "mask_pixels": 2},
            },
            selection_mode="quality",
        )
        self.assertTrue(np.array_equal(pose, best_pose))
        self.assertTrue(timing["sil_skipped"])
        self.assertEqual(timing["iou_select_s"], 0.0)
        self.assertEqual(timing["sil_refine_s"], 0.0)

    def test_other_tabletop_success_is_default_varied_pose_fallback(self):
        other = [("table", "1", "2")]
        selected, source = choose_success_candidates([], other)
        self.assertEqual(selected, tuple(other))
        self.assertEqual(source, "success_other_tabletop")
        selected, source = choose_success_candidates([], other, strict_tabletop=True)
        self.assertEqual(selected, ())
        self.assertEqual(source, "no_successful_candidate")

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
        self.assertEqual(session.command_timeout_ms, 3000)
        self.assertEqual(session.command_retries, 1)

    def test_preflight_requires_each_runtime_asset(self):
        """The offline check has no ParaDex/robot dependency."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            obj_root = root / "objects"
            assets = root / "assets"
            candidates = root / "candidates"
            anchors = root / "anchors"
            obj = "banana"
            (obj_root / obj / "raw_mesh").mkdir(parents=True)
            (obj_root / obj / "raw_mesh" / f"{obj}.obj").touch()
            repre = assets / obj / "object_repre" / "v1" / obj / "1"
            repre.mkdir(parents=True)
            (repre / "repre.pth").touch()
            grasp = candidates / obj / "table" / "0" / "0"
            grasp.mkdir(parents=True)
            (grasp / "wrist_se3.npy").touch()
            (grasp / "pregrasp_pose.npy").touch()
            (grasp / "result.json").write_text(
                json.dumps({"success": True, "arm": "franka"})
            )

            rows = build_report(
                parse_catalog([obj]), object_root=obj_root, assets_base=assets,
                candidate_root=candidates, anchor_root=anchors, require_gotrack=True,
                arm="franka",
            )
            self.assertFalse(rows[0].ready)
            self.assertEqual(rows[0].missing, ["gotrack_anchor_bank"])

            anchors.mkdir()
            (anchors / f"{obj}.npz").touch()
            rows = build_report(
                parse_catalog([obj]), object_root=obj_root, assets_base=assets,
                candidate_root=candidates, anchor_root=anchors, require_gotrack=True,
                arm="franka",
            )
            self.assertTrue(rows[0].ready)
            self.assertEqual(rows[0].successful_candidate_count, 1)
            require_ready(rows)

            wrong_arm = build_report(
                parse_catalog([obj]), object_root=obj_root, assets_base=assets,
                candidate_root=candidates, anchor_root=anchors, require_gotrack=True,
                arm="xarm",
            )
            self.assertFalse(wrong_arm[0].ready)
            self.assertEqual(wrong_arm[0].missing, ["successful_grasp"])
            with self.assertRaisesRegex(RuntimeError, "banana: successful_grasp"):
                require_ready(wrong_arm)

    def test_preflight_direct_script_invocation_has_repo_import_path(self):
        """The command documented for operators works without PYTHONPATH."""
        script = (Path(__file__).resolve().parents[2]
                  / "src/demo/continuous_basket/preflight.py")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            proc = subprocess.run(
                [sys.executable, str(script), "--objects", "missing_object",
                 "--object-root", str(root / "objects"),
                 "--assets-base", str(root / "assets"),
                 "--candidate-root", str(root / "candidates"), "--no-gotrack"],
                cwd=Path(__file__).resolve().parents[2], text=True,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
            )
        self.assertEqual(proc.returncode, 2, proc.stdout)
        self.assertIn("MISSING: mesh", proc.stdout)

if __name__ == "__main__":
    unittest.main()
