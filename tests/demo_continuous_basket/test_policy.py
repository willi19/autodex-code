import json
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from autodex.fast_selection import select_best_pose_by_quality
from src.demo.continuous_basket.catalog import (
    CatalogRecognizer,
    parse_catalog,
    rank_catalog_detections,
    require_catalog_runtime,
    single_object_match,
)
from src.demo.continuous_basket.camera import capture_catalog_snapshot
from src.demo.continuous_basket.camera_smoke import advancing_frame_errors
from src.demo.continuous_basket.basket_marker import (
    DEFAULT_BASKET_MARKER_ID,
    release_reference_from_marker,
)
from src.demo.continuous_basket.policy import (
    LocalRetryPolicy,
    PoseEvidence,
    PoseVerifier,
    Verification,
    choose_success_candidates,
)
from src.demo.continuous_basket.preflight import build_report, require_ready
from src.demo.continuous_basket.recording import (
    autodex_session_relative,
    create_session_dir,
    parse_autodex_session_relative,
    resolve_signal_generator_params,
    should_auto_upload,
)
from src.demo.continuous_basket.tabletop import mesh_bottom_z, raise_to_table
from src.demo.continuous_basket.tracking import LiveGoTrackSession
from src.demo.continuous_basket.upload_recording import (
    recording_video_paths,
    verify_nas_recording,
)

try:
    from autodex.perception.init_orchestrator import InitOrchestrator
except ModuleNotFoundError:  # lightweight policy env intentionally has no OpenCV
    InitOrchestrator = None

try:
    from autodex.perception.mask import _masks_by_class_from_yoloe
except ModuleNotFoundError:  # lightweight policy env intentionally has no OpenCV
    _masks_by_class_from_yoloe = None


class CatalogPolicyTest(unittest.TestCase):
    def test_table_snap_only_raises_small_mesh_penetrations(self):
        pose = np.eye(4)
        pose[2, 3] = 0.030
        vertices = np.array([[0.0, 0.0, -0.010], [0.0, 0.0, 0.020]])
        corrected, raise_m, bottom = raise_to_table(
            pose, vertices, surface_z=0.035, max_raise_m=0.010
        )
        self.assertAlmostEqual(bottom, 0.020)
        self.assertAlmostEqual(raise_m, 0.0)  # required 15 mm is refused
        self.assertAlmostEqual(corrected[2, 3], 0.030)

        corrected, raise_m, _ = raise_to_table(
            pose, vertices, surface_z=0.035, max_raise_m=0.020
        )
        self.assertAlmostEqual(raise_m, 0.015)
        self.assertAlmostEqual(mesh_bottom_z(corrected, vertices), 0.035)

    def test_auto_upload_requires_a_real_pick_motion(self):
        self.assertTrue(should_auto_upload(
            camera_recording=True, uploads_deferred=False,
            normal_exit=True, robot_motion_started=True,
        ))
        self.assertFalse(should_auto_upload(
            camera_recording=True, uploads_deferred=False,
            normal_exit=True, robot_motion_started=False,
        ))
        self.assertFalse(should_auto_upload(
            camera_recording=True, uploads_deferred=False,
            normal_exit=False, robot_motion_started=True,
        ))

    def test_catalog_runtime_rejects_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, "checkpoint is missing"):
                require_catalog_runtime(weights_path=Path(tmp) / "yoloe-26x-seg.pt")

    def test_single_catalogue_bypasses_detector_selection(self):
        match = single_object_match(parse_catalog(["banana"]))
        self.assertEqual(match.name, "banana")
        self.assertEqual(match.supporting_views, 0)
        with self.assertRaises(ValueError):
            single_object_match(parse_catalog(["banana", "apple"]))

    def test_gotrack_diagnostics_exposes_tracker_status(self):
        session = LiveGoTrackSession(
            pc_list=[], capture_ips=[], intrinsics={}, extrinsics={},
            anchor_root=Path("/tmp/anchors"),
        )
        session.obj_name = "banana"
        session._worker_error = "no observations"
        session._tracker = type("Tracker", (), {
            "_status_lock": threading.Lock(),
            "status": {"counts": {"received": 0}, "per_pc_last_frame": {}},
        })()
        diag = session.diagnostics()
        self.assertEqual(diag["object"], "banana")
        self.assertEqual(diag["worker_error"], "no observations")
        self.assertEqual(diag["tracker_status"]["counts"]["received"], 0)

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
                    (image_dir / f"camera{idx}.png").write_bytes(b"not-a-real-png")
                return {"capture1": {"status": "ok"}}

        with tempfile.TemporaryDirectory() as tmp:
            rcc = SnapshotOnlyRcc()
            count = capture_catalog_snapshot(
                rcc, Path(tmp) / "snapshot", min_images=2, settle_timeout_s=0.2,
                expected_serials=["camera0", "camera1"],
            )
        self.assertEqual(count, 2)
        self.assertEqual(len(rcc.calls), 1)
        self.assertEqual(rcc.calls[0][1], 1)

    def test_camera_smoke_requires_each_stream_to_advance(self):
        before = {"error": False, "pc": {
            "capture1": {"status": "ok", "states": {"cam": "CAPTURING"},
                         "frame_ids": {"cam": 10}},
        }}
        after = {"error": False, "pc": {
            "capture1": {"status": "ok", "states": {"cam": "CAPTURING"},
                         "frame_ids": {"cam": 11}},
        }}
        self.assertEqual(advancing_frame_errors(before, after, ["capture1"]), [])
        after["pc"]["capture1"]["frame_ids"]["cam"] = 10
        self.assertIn("frame id did not advance", advancing_frame_errors(
            before, after, ["capture1"],
        )[0])

    def test_recording_uses_only_unambiguous_discovered_usbtmc_device(self):
        with tempfile.TemporaryDirectory() as tmp:
            devices = Path(tmp)
            (devices / "usbtmc5").touch()
            params, note = resolve_signal_generator_params(
                {"addr": "/dev/usbtmc0"}, device_root=devices,
            )
            self.assertEqual(params["addr"], str(devices / "usbtmc5"))
            self.assertIn("usbtmc5", note)

            (devices / "usbtmc6").touch()
            params, note = resolve_signal_generator_params(
                {"addr": "/dev/usbtmc0"}, device_root=devices,
            )
            self.assertEqual(params["addr"], "/dev/usbtmc0")
            self.assertIsNone(note)

    def test_timestamped_session_layout_is_catalogued_and_non_overwriting(self):
        import datetime as dt
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "AutoDex"
            now = dt.datetime(2026, 8, 31, 18, 0, 1, 123456)
            first, first_id = create_session_dir(
                root, experiment_name="continuous_basket", arm="franka", hand="inspire",
                object_names=["banana", "apple"], now=now,
            )
            second, second_id = create_session_dir(
                root, experiment_name="continuous_basket", arm="franka", hand="inspire",
                object_names=["apple", "banana"], now=now,
            )
            self.assertEqual(first_id, "20260831_180001_123456")
            self.assertEqual(second_id, "20260831_180001_123456_01")
            self.assertEqual(
                autodex_session_relative(root, first),
                Path("AutoDex/experiment/continuous_basket/franka_inspire/apple__banana") / first_id,
            )
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())

    def test_session_identity_is_timestamp_only(self):
        import datetime as dt
        from src.demo.continuous_basket.recording import timestamp_session_name
        self.assertEqual(
            timestamp_session_name(now=dt.datetime(2026, 8, 31, 18, 0, 1, 123456)),
            "20260831_180001_123456",
        )

    def test_upload_selector_and_nas_verification_are_session_scoped(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session = Path("AutoDex/experiment/continuous_basket/franka_inspire/banana/20260831_180001_123456")
            capture = root / "captures1"
            ours = capture / session / "raw/capture/videos"
            other = capture / "shared_data/AutoDex/_fulltest/raw/videos"
            ours.mkdir(parents=True)
            other.mkdir(parents=True)
            (ours / "cam_a.avi").write_bytes(b"raw")
            (other / "stale.avi").write_bytes(b"raw")
            self.assertEqual(
                recording_video_paths([capture], session_relative=session),
                [ours / "cam_a.avi"],
            )

            session_dir = root / "shared_data" / session
            cam_param = session_dir / "cam_param"
            cam_param.mkdir(parents=True)
            (cam_param / "intrinsics.json").write_text(json.dumps({"cam_a": {}, "cam_b": {}}))
            videos = session_dir / "videos/capture"
            videos.mkdir(parents=True)
            (videos / "cam_a.avi").write_bytes(b"done")
            ok, detail = verify_nas_recording(session_dir)
            self.assertFalse(ok)
            self.assertEqual(detail, "missing=cam_b")
            (videos / "cam_b.avi").write_bytes(b"done")
            self.assertEqual(verify_nas_recording(session_dir), (True, "2 camera videos"))

    def test_upload_verification_uses_the_take_active_camera_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp) / "session"
            (session_dir / "cam_param").mkdir(parents=True)
            (session_dir / "cam_param/intrinsics.json").write_text(
                json.dumps({"active": {}, "inactive": {}})
            )
            (session_dir / "recording.json").write_text(json.dumps({"camera_serials": ["active"]}))
            output = session_dir / "videos/capture"
            output.mkdir(parents=True)
            (output / "active.avi").write_bytes(b"done")
            self.assertEqual(verify_nas_recording(session_dir), (True, "1 camera videos"))

    def test_session_argument_rejects_absolute_and_parent_paths(self):
        self.assertEqual(
            parse_autodex_session_relative("AutoDex/experiment/continuous_basket/a"),
            Path("AutoDex/experiment/continuous_basket/a"),
        )
        for bad in ("/home/robot/shared_data/AutoDex/x", "AutoDex/../other", "other/x"):
            with self.assertRaises(ValueError):
                parse_autodex_session_relative(bad)

    def test_basket_marker_offset_uses_marker_frame(self):
        marker_pose = np.eye(4)
        marker_pose[:3, :3] = np.array([
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        release = release_reference_from_marker(
            np.array([0.5, -0.2, 0.1]), marker_pose, np.array([0.1, 0.0, 0.05]),
        )
        np.testing.assert_allclose(release, [0.5, -0.1, 0.15])

    def test_basket_marker_keeps_legacy_banana_marker_as_default(self):
        self.assertEqual(DEFAULT_BASKET_MARKER_ID, 660)

    def test_basket_marker_rejects_bad_geometry(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            release_reference_from_marker(np.zeros(2), np.eye(4), np.zeros(3))

    def test_basket_marker_rejects_side_mounted_tag(self):
        side_marker = np.eye(4)
        side_marker[:3, :3] = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ])
        with self.assertRaisesRegex(ValueError, "horizontally"):
            release_reference_from_marker(np.zeros(3), side_marker, np.zeros(3))

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

            other_arm = build_report(
                parse_catalog([obj]), object_root=obj_root, assets_base=assets,
                candidate_root=candidates, anchor_root=anchors, require_gotrack=True,
                arm="xarm",
            )
            # A candidate's recorded arm is provenance, not a runtime filter.
            # The live planner remains responsible for validating it on the
            # arm selected for the take.
            self.assertTrue(other_arm[0].ready)
            self.assertEqual(other_arm[0].successful_candidate_count, 1)
            require_ready(other_arm)

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
