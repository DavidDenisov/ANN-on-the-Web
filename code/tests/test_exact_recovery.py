import gzip
import unittest

import faiss
import numpy as np
import torch

from exact_recovery import CLUSTER_RECORD, cluster_batches, scan_top1
from paper_settings import DIMENSIONS


class ExactRecoveryTest(unittest.TestCase):
    def test_scans_all_clusters_and_merges_global_winners(self):
        def cluster(ids, coordinates):
            records = np.zeros(len(ids), dtype=CLUSTER_RECORD)
            records["id"] = ids
            records["vector"][:, :2] = np.asarray(coordinates).reshape(-1, 2)
            return gzip.compress(records.tobytes())

        payloads = [
            cluster([4, 3], [[2, 0], [0, -2]]),
            cluster([], []),
            cluster([2, 0, 1], [[0, 3], [1, 0], [-1, 0]]),
        ]
        queries = torch.zeros((3, DIMENSIONS), dtype=torch.float32)
        queries[:, :2] = torch.tensor([[1, 0], [0.6, 0.8], [0, -1]])
        index = faiss.IndexFlatL2(DIMENSIONS)
        batches = cluster_batches([2, 0, 3], payloads.__getitem__)
        winners, distances = scan_top1(index, queries, 5, batches)
        np.testing.assert_array_equal(winners, [0, 2, 3])
        np.testing.assert_allclose(distances, [0, 0.4, 0], atol=1e-6)
        self.assertEqual(np.count_nonzero(winners == [4, 2, 3]), 2)

        with self.assertRaisesRegex(ValueError, "Wrong record count"):
            batches = cluster_batches([2, 0, 2], payloads.__getitem__)
            scan_top1(index, queries, 5, batches)
