"""Small CPU checks for the GPU scanner's shared selection and local loading logic."""

import importlib.util
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import faiss
import numpy as np

spec = importlib.util.spec_from_file_location("full_nearest_gpu", Path(__file__).resolve().parents[1] / "benchmarks/full-nearest-gpu.py")
scan = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scan)


class NearestChecks(unittest.TestCase):
    def test_self_exclusion_keeps_distinct_identical_vectors_and_handles_singletons(self):
        values = np.array([[1, 0], [1, 0], [0, 1]], dtype=np.float32)
        ids = np.array([8, 27, 51])
        index = faiss.IndexFlatIP(2)
        index.add(values)
        scores, positions = scan.closest_nonself(index, values, ids, ids)
        np.testing.assert_allclose(scores, [1, 1, 0])
        self.assertTrue(np.all(ids[positions] != ids))
        self.assertEqual(int(ids[positions[0]]), 27)
        index.reset()
        index.add(values[:1])
        scores, _ = scan.closest_nonself(index, values[:1], ids[:1], ids[:1])
        self.assertTrue(np.isneginf(scores[0]))

    def test_local_shard_validates_count_and_reproduces_normalized_fp16_vectors(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "vectors.npy"
            raw = np.zeros((2, 384), dtype="<f2")
            raw[:, :2] = [[3, 4], [0, 2]]
            np.save(path, raw)
            shard = {"path": path, "offset": 7, "count": 2}
            ids, stored, values = scan.read_shard(shard, 384)
            np.testing.assert_array_equal(ids, [7, 8])
            self.assertEqual(stored.dtype, np.dtype("<f2"))
            np.testing.assert_allclose(values[:, :2], [[.6, .8], [0, 1]], atol=3e-4)
            np.testing.assert_allclose(np.linalg.norm(values, axis=1), 1, atol=1e-7)
            shard["count"] = 3
            with self.assertRaises(ValueError):
                scan.read_shard(shard, 384)


if __name__ == "__main__":
    unittest.main()
