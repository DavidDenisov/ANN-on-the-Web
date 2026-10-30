"""CPU checks for exported vector formats and downloaded shard validation."""

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cluster import centroid_payload, noisy_queries, pack_codes, prepare_vectors
from download_dataset import DIMENSIONS, META_COLUMNS, validate_shard


class PreparationChecks(unittest.TestCase):
    def test_normalization_preserves_input_and_rejects_invalid_vectors(self):
        raw = np.zeros((2, DIMENSIONS), dtype=np.float32)
        raw[:, :2] = [[3, 4], [0, 2]]
        original = raw.copy()
        stored, decoded = prepare_vectors(raw)
        np.testing.assert_array_equal(raw, original)
        self.assertEqual(stored.dtype, np.dtype("<f2"))
        np.testing.assert_allclose(np.linalg.norm(decoded, axis=1), 1, atol=1e-7)
        for invalid in [np.zeros_like(raw), np.full_like(raw, np.nan), raw[:, :2]]:
            with self.assertRaises(ValueError):
                prepare_vectors(invalid)

    def test_noise_is_reproducible_and_normalized(self):
        raw = np.ones((2, DIMENSIONS), dtype=np.float32)
        first = noisy_queries(raw, .4, 17)
        np.testing.assert_array_equal(first, noisy_queries(raw, .4, 17))
        np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1, atol=1e-7)
        self.assertFalse(np.array_equal(first, noisy_queries(raw, .4, 18)))

    def test_scalar_centroid_encodings_handle_constant_dimensions(self):
        values = np.zeros((8192, DIMENSIONS), dtype=np.float32)
        values[:, 0] = np.linspace(-1, 1, len(values))
        for codec, expected_bytes in [('f16', values.size * 2), ('sq6', DIMENSIONS * 8 + values.size * 6 // 8),
                                      ('int8', DIMENSIONS * 4 + values.size)]:
            with self.subTest(codec=codec):
                payload, decoded = centroid_payload(values, codec, seed=17)
                self.assertEqual(len(payload), expected_bytes)
                self.assertTrue(np.isfinite(decoded).all())
                np.testing.assert_allclose(decoded, values, atol=1 / 63)
        self.assertEqual(pack_codes(np.array([0, 1, 63, 32], dtype=np.uint8), 6), bytes([64, 240, 131]))

    def test_shard_row_count_rejects_boolean_even_when_it_matches_shape(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            np.save(root / 'vectors.npy', np.ones((1, DIMENSIONS), dtype='<f2'))
            pq.write_table(pa.table({name: ['value'] for name in META_COLUMNS}), root / 'passages.parquet')
            info = root / 'info.json'
            info.write_text(json.dumps({'source': 'shard.parquet', 'rows': 1}))
            self.assertEqual(validate_shard(root, 'shard.parquet'), 1)
            info.write_text(json.dumps({'source': 'shard.parquet', 'rows': True}))
            with self.assertRaises(ValueError):
                validate_shard(root, 'shard.parquet')


if __name__ == '__main__':
    unittest.main()
