import os
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from autodex.utils.path import load_candidate


def _candidate_tree(root: Path, obj: str) -> Path:
    grasp = root / obj / "shelf" / "2" / "candidate_a"
    grasp.mkdir(parents=True)
    wrist = np.eye(4, dtype=np.float32)
    wrist[0, 3] = 0.04
    np.save(grasp / "wrist_se3.npy", wrist)
    np.save(grasp / "pregrasp_pose.npy", np.full(6, 0.2, dtype=np.float32))
    np.save(grasp / "grasp_pose.npy", np.full(6, 0.8, dtype=np.float32))
    return root / obj


class CandidateArchiveTest(unittest.TestCase):
    def _load_from_archive(self, archive_suffix: str) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source = tmp_path / "source"
            obj = "donut"
            object_dir = _candidate_tree(source, obj)
            candidates = tmp_path / "candidates" / "v8"
            candidates.mkdir(parents=True)
            archive = candidates / f"{obj}{archive_suffix}"
            if archive_suffix == ".tar.gz":
                with tarfile.open(archive, "w:gz") as out:
                    out.add(object_dir, arcname=obj)
            else:
                with zipfile.ZipFile(archive, "w") as out:
                    for path in object_dir.rglob("*"):
                        out.write(path, path.relative_to(source))

            cache = tmp_path / "cache"
            with patch.dict(os.environ, {"AUTODEX_CANDIDATE_CACHE": str(cache)}):
                wrist, pregrasp, grasp, info = load_candidate(
                    obj, np.eye(4), "v8", hand="inspire", shuffle=False,
                    skip_done=False, candidates_root=str(tmp_path / "candidates"))
                # A second call must re-use the local cache transparently.
                second = load_candidate(
                    obj, np.eye(4), "v8", hand="inspire", shuffle=False,
                    skip_done=False, candidates_root=str(tmp_path / "candidates"))

            self.assertEqual(info, [("shelf", "2", "candidate_a")])
            np.testing.assert_allclose(wrist[0, 0, 3], 0.04)
            np.testing.assert_allclose(pregrasp, [[0.2] * 6])
            np.testing.assert_allclose(grasp, [[0.8] * 6])
            self.assertEqual(second[3], info)
            self.assertTrue(any(cache.iterdir()))

    def test_loads_tar_gz_candidate_archive(self):
        self._load_from_archive(".tar.gz")

    def test_loads_zip_candidate_archive(self):
        self._load_from_archive(".zip")


if __name__ == "__main__":
    unittest.main()
