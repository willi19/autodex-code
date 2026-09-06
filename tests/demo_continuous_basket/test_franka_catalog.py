from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.demo.continuous_basket.catalog import CatalogObject
from src.demo.continuous_basket.prepare_franka_catalog import (
    _demo_command,
    _successful_tabletops,
    processing_readiness,
    table_scene_payload,
    write_table_scenes,
)


class FrankaCatalogTest(unittest.TestCase):
    def test_demo_command_uses_marker_source_when_provided(self):
        args = type("Args", (), {
            "planner_python": "python", "version": "v8", "max_successes": 1,
            "basket_center": None, "basket_marker_id": 42,
            "basket_marker_dict": "6X6_1000", "basket_marker_offset": [0.0, 0.0, 0.08],
        })()
        command = _demo_command([CatalogObject("banana", "banana")], args)
        self.assertIn("--basket-marker-id", command)
        self.assertIn("42", command)
        self.assertNotIn("--basket-center", command)

    def _make_object(self, root: Path, name: str) -> None:
        base = root / name
        for relative in (
            f"raw_mesh/{name}.obj",
            "processed_data/mesh/simplified.obj",
            "processed_data/urdf/coacd.urdf",
            "processed_data/info/simplified.json",
        ):
            path = base / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("mesh" if path.suffix != ".json" else "{}")
        tabletop = base / "processed_data/info/tabletop"
        tabletop.mkdir(parents=True, exist_ok=True)
        np.save(tabletop / "000.npy", np.eye(4))
        pose = np.eye(4)
        pose[:3, 3] = [0.1, -0.2, 0.03]
        np.save(tabletop / "001.npy", pose)

    def test_processing_audit_and_table_scene_preserve_object_root(self):
        item = CatalogObject("new_object", "new object")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "objects"
            self._make_object(root, item.name)
            row = processing_readiness(item, root)
            self.assertTrue(row.ready)
            self.assertEqual(row.tabletop_pose_count, 2)

            payload = table_scene_payload(item, np.eye(4), root)
            target = payload["scene"]["mesh"]["target"]
            self.assertEqual(target["pose"], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
            self.assertTrue(target["file_path"].startswith(str(root)))
            self.assertTrue(target["urdf_path"].startswith(str(root)))

    def test_scene_writer_refuses_mismatched_existing_scene(self):
        item = CatalogObject("new_object", "new object")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "objects"
            scene_root = Path(tmp) / "scenes"
            self._make_object(root, item.name)
            ids = write_table_scenes([item], object_root=root, scene_root=scene_root,
                                     overwrite=False)
            self.assertEqual(ids[item.name], ["0", "1"])
            first = scene_root / "inspire/new_object/table/0.json"
            data = json.loads(first.read_text())
            data["meta"]["pose_idx"] = "wrong"
            first.write_text(json.dumps(data))
            with self.assertRaises(FileExistsError):
                write_table_scenes([item], object_root=root, scene_root=scene_root,
                                   overwrite=False)

    def test_table_scene_metadata_reports_any_arm_success_pose(self):
        item = CatalogObject("new_object", "new object")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "objects"
            scene_root = Path(tmp) / "scenes"
            candidate_root = Path(tmp) / "candidates"
            self._make_object(root, item.name)
            write_table_scenes([item], object_root=root, scene_root=scene_root,
                               overwrite=False)
            grasp = candidate_root / item.name / "table/0/candidate0"
            grasp.mkdir(parents=True)
            (grasp / "wrist_se3.npy").write_bytes(b"pose")
            # Collection arm is provenance only; either arm's success makes
            # this object pose useful to the continuous catalogue.
            (grasp / "result.json").write_text(json.dumps({"success": True, "arm": "xarm"}))
            self.assertEqual(_successful_tabletops(
                item, candidate_root=candidate_root, scene_root=scene_root,
            ), {"000"})


if __name__ == "__main__":
    unittest.main()
