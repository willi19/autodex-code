import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.demo.inference.grasp_library import (
    DemoGrasp,
    demo_planner_candidates,
    inspire_action_to_qpos,
    load_demo_grasps,
)


class DemoGraspLibraryTest(unittest.TestCase):
    def test_loads_positive_dataset_and_v8_episodes_in_object_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "selected"
            v8 = root / "v8"
            candidates = root / "candidates"
            obj = "demo_obj"

            trial = dataset / obj / "dataset_ok"
            (trial / "executed_grasp").mkdir(parents=True)
            (trial / "human_success_label.json").write_text(
                json.dumps({"reviewed": True, "human_success": True})
            )
            dataset_wrist = np.eye(4, dtype=np.float32)
            dataset_wrist[0, 3] = 0.03
            np.save(trial / "executed_grasp/wrist_se3.npy", dataset_wrist)
            np.save(trial / "executed_grasp/grasp_pose.npy", np.ones(6, dtype=np.float32))

            # Negative labels must never enter the fixed library.
            rejected = dataset / obj / "dataset_rejected"
            (rejected / "executed_grasp").mkdir(parents=True)
            (rejected / "human_success_label.json").write_text(
                json.dumps({"reviewed": True, "human_success": False})
            )
            np.save(rejected / "executed_grasp/wrist_se3.npy", np.eye(4))
            np.save(rejected / "executed_grasp/grasp_pose.npy", np.ones(6))

            # A finite but singular transform would otherwise crash the live
            # marker-reach preflight when it tries to invert the wrist pose.
            singular = dataset / obj / "dataset_singular"
            (singular / "executed_grasp").mkdir(parents=True)
            (singular / "human_success_label.json").write_text(
                json.dumps({"reviewed": True, "human_success": True})
            )
            np.save(singular / "executed_grasp/wrist_se3.npy", np.zeros((4, 4)))
            np.save(singular / "executed_grasp/grasp_pose.npy", np.ones(6))

            episode = v8 / obj / "v8_ok"
            (episode / "plan").mkdir(parents=True)
            (episode / "result.json").write_text(json.dumps({"success": True}))
            c2r = np.eye(4, dtype=np.float32)
            c2r[0, 3] = 1.0
            pose_world = np.eye(4, dtype=np.float32)
            pose_world[0, 3] = 1.4
            wrist_obj = np.eye(4, dtype=np.float32)
            wrist_obj[2, 3] = -0.12
            wrist_robot = (np.linalg.inv(c2r) @ pose_world) @ wrist_obj
            np.save(episode / "C2R.npy", c2r)
            np.save(episode / "pose_world.npy", pose_world)
            np.save(episode / "plan/wrist_se3.npy", wrist_robot)
            np.save(episode / "plan/traj.npy", np.array([[0] * 6 + [0.1] * 6], dtype=np.float32))
            np.save(episode / "squeeze_hand.npy", np.full(6, 500.0, dtype=np.float32))

            grasps = load_demo_grasps(
                obj, selected_root=dataset, v8_root=v8,
                v8_candidate_root=candidates,
            )
            self.assertEqual([g.source for g in grasps], ["v8_inspire", "selected_100_inspire"])
            np.testing.assert_allclose(grasps[0].wrist_obj, wrist_obj)
            np.testing.assert_allclose(grasps[0].pregrasp, np.full(6, 0.1))
            np.testing.assert_allclose(grasps[1].wrist_obj, dataset_wrist)
            # If the original selected candidate is unavailable, preserve the
            # observed successful hand state; never fabricate all-zero fingers.
            np.testing.assert_allclose(grasps[1].pregrasp, np.ones(6))

    def test_selected_success_restores_original_candidate_hand_pair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            obj = "demo_obj"
            episode = root / "selected" / obj / "success"
            (episode / "executed_grasp").mkdir(parents=True)
            (episode / "human_success_label.json").write_text(json.dumps({
                "reviewed": True, "human_success": True,
            }))
            wrist = np.eye(4, dtype=np.float32)
            wrist[0, 3] = 0.03
            observed = np.full(6, 0.2, dtype=np.float32)
            np.save(episode / "executed_grasp/wrist_se3.npy", wrist)
            np.save(episode / "executed_grasp/grasp_pose.npy", observed)

            source = root / "selected_candidates" / obj / "shelf" / "2" / "g7"
            source.mkdir(parents=True)
            np.save(source / "wrist_se3.npy", wrist)
            np.save(source / "pregrasp_pose.npy", observed)
            np.save(source / "grasp_pose.npy", np.full(6, 0.8, dtype=np.float32))

            grasps = load_demo_grasps(
                obj, selected_root=root / "selected",
                selected_candidate_root=root / "selected_candidates",
                v8_root=root / "v8", v8_candidate_root=root / "v8_candidates",
            )
            self.assertEqual(len(grasps), 1)
            np.testing.assert_allclose(grasps[0].pregrasp, observed)
            np.testing.assert_allclose(grasps[0].grasp, np.full(6, 0.8))

    def test_inspire_action_conversion_reverses_controller_order(self):
        qpos = inspire_action_to_qpos(np.array([0, 0, 0, 0, 0, 0], dtype=np.float32))
        np.testing.assert_allclose(qpos, [1.15, 0.55, 1.6, 1.6, 1.6, 1.6])

    def test_v8_prefers_the_candidate_referenced_by_success_scene_info(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            obj = "demo_obj"
            episode = root / "v8" / obj / "run"
            episode.mkdir(parents=True)
            episode.joinpath("result.json").write_text(json.dumps({
                "success": True, "scene_info": ["table", "3", "g7"],
            }))
            candidate = root / "candidates" / obj / "table" / "3" / "g7"
            candidate.mkdir(parents=True)
            wrist = np.eye(4, dtype=np.float32)
            wrist[1, 3] = 0.07
            np.save(candidate / "wrist_se3.npy", wrist)
            np.save(candidate / "pregrasp_pose.npy", np.full(6, 0.2, dtype=np.float32))
            np.save(candidate / "grasp_pose.npy", np.full(6, 0.8, dtype=np.float32))

            grasps = load_demo_grasps(
                obj, selected_root=root / "selected", v8_root=root / "v8",
                v8_candidate_root=root / "candidates",
            )
            self.assertEqual(len(grasps), 1)
            np.testing.assert_allclose(grasps[0].wrist_obj, wrist)
            np.testing.assert_allclose(grasps[0].pregrasp, np.full(6, 0.2))
            np.testing.assert_allclose(grasps[0].grasp, np.full(6, 0.8))

    def test_explicit_planner_candidates_apply_object_frame_symmetry(self):
        object_pose = np.eye(4)
        object_pose[:3, 3] = [0.5, -0.2, 0.1]
        wrist_obj = np.eye(4, dtype=np.float32)
        wrist_obj[0, 3] = 0.04
        grasp = DemoGrasp(
            source="v8_inspire", episode=Path("/tmp/episode"),
            wrist_obj=wrist_obj, pregrasp=np.full(6, 0.1, dtype=np.float32),
            grasp=np.full(6, 0.8, dtype=np.float32),
        )
        Rz_180 = np.diag([-1.0, -1.0, 1.0])
        wrists, pregrasp, grasp_pose, info = demo_planner_candidates(
            [grasp], object_pose, np.stack([np.eye(3), Rz_180]))

        self.assertEqual(wrists.shape, (2, 4, 4))
        np.testing.assert_allclose(wrists[0], object_pose @ wrist_obj)
        T_sym = np.eye(4); T_sym[:3, :3] = Rz_180
        np.testing.assert_allclose(wrists[1], object_pose @ T_sym @ wrist_obj)
        np.testing.assert_allclose(pregrasp, [[0.1] * 6, [0.1] * 6])
        np.testing.assert_allclose(grasp_pose, [[0.8] * 6, [0.8] * 6])
        self.assertEqual(info[1][-1], "1")


if __name__ == "__main__":
    unittest.main()
