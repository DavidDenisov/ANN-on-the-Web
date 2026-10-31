"""Exercise real dataset/index serialization with tiny data and deterministic training."""

import gzip
import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fixtures import CODE, write_json
import cluster
import download_dataset


def write_source(path, values):
    path.parent.mkdir(parents=True, exist_ok=True)
    table = {key: list(range(len(values))) for key in ("id", "wiki_id", "paragraph_id")}
    table.update(title=[f"Title {i}" for i in range(len(values))],
                 text=[f"Passage {i}" for i in range(len(values))], emb=values.tolist())
    pq.write_table(pa.table(table), path)


class ProducerChecks(unittest.TestCase):
    def test_conversion_preserves_source_and_metadata_alignment_and_rejects_invalid_vectors(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            values = np.zeros((2, 384), dtype=np.float32)
            values[:, :2] = [[3, 4], [0, 2]]
            source = root / "source.parquet"
            write_source(source, values)
            original = source.read_bytes()
            self.assertEqual(download_dataset.convert_shard(source, root / "converted"), 2)
            self.assertEqual(source.read_bytes(), original)
            np.testing.assert_array_equal(np.load(root / "converted/vectors.npy"), values.astype("<f2"))
            self.assertEqual(pq.read_table(root / "converted/passages.parquet")["title"].to_pylist(), ["Title 0", "Title 1"])
            self.assertEqual(download_dataset.validate_shard(root / "converted", "source.parquet"), 2)
            for i, invalid in enumerate((np.zeros_like(values), np.full_like(values, np.nan),
                                         np.full_like(values, 1e9), np.full_like(values, 1e-20), values[:, :2])):
                source = root / f"invalid-{i}.parquet"
                write_source(source, invalid)
                output = root / f"failed-{i}"
                with self.assertRaises(ValueError):
                    download_dataset.convert_shard(source, output)
                self.assertTrue(source.is_file())
                self.assertFalse(output.exists())

    def test_download_resume_cleans_only_successfully_prepared_downloads(self):
        for valid in (True, False):
            with self.subTest(valid=valid), TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = {"dataset": download_dataset.DATASET, "dimensions": 384, "dtype": "<f2",
                            "revision": "pinned-revision", "shards": ["data/a.parquet"]}
                write_json(root / "manifest.json", manifest)
                unrelated = root / "unrelated.txt"
                unrelated.write_text("keep")
                source = root / ".downloads/a/data/a.parquet"
                def download(*args, **kwargs):
                    self.assertEqual(kwargs["revision"], "pinned-revision")
                    values = np.ones((1, 384), dtype=np.float32) if valid else np.zeros((1, 384), dtype=np.float32)
                    write_source(source, values)
                    return str(source)
                with patch.object(download_dataset, "hf_hub_download", side_effect=download) as fetch:
                    if valid:
                        download_dataset.prepare_dataset(root)
                        self.assertFalse((root / ".downloads/a").exists())
                        self.assertTrue((root / "a/vectors.npy").is_file())
                        download_dataset.prepare_dataset(root)
                    else:
                        with self.assertRaises(ValueError):
                            download_dataset.prepare_dataset(root)
                        self.assertTrue(source.exists())
                        self.assertFalse((root / "a").exists())
                    self.assertEqual(fetch.call_count, 1)
                self.assertEqual(unrelated.read_text(), "keep")

    def test_index_records_and_all_six_centroid_formats_match_the_browser(self):
        class Kmeans:
            def __init__(self, dimensions, words, **kwargs):
                self.centroids = np.zeros((words, dimensions), dtype=np.float32)
                if words == 8192:
                    self.centroids[:2, 0] = [1, -1]
                else:
                    self.centroids[1:3, 0] = [1, -1]
                self.index = cluster.faiss.IndexFlatL2(dimensions)
                self.index.add(self.centroids)
            def train(self, values):
                pass
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            for name, scale in (("a", 1), ("b", 2)):
                values = np.zeros((2, 384), dtype=np.float32)
                values[:, 0] = [scale, -scale]
                parquet = root / f"{name}.parquet"
                write_source(parquet, values)
                download_dataset.convert_shard(parquet, source / name)
            write_json(source / "manifest.json", {"dimensions": 384, "dtype": "<f2",
                "dataset": "fixture", "revision": "r", "shards": ["a.parquet", "b.parquet"]})
            output = root / "index"
            with patch.object(cluster, "require_gpu", return_value=1), patch.object(cluster.faiss, "Kmeans", Kmeans), \
                    patch.object(cluster, "TRAINING_SAMPLE_SIZE", 2), patch.object(cluster, "CORPUS_COUNT", 4):
                manifest = cluster.cluster_and_save(source, output, 8192, 17)
            self.assertEqual(manifest["count"], 4)
            self.assertEqual(manifest["record_bytes"], 772)
            self.assertEqual(manifest["cluster_sizes"][:2], [2, 2])
            records = []
            record_type = np.dtype([("id", "<u4"), ("vector", "<f2", (384,))])
            for cluster_id in (0, 1):
                raw = gzip.decompress((output / f"clusters/cluster_{cluster_id}.bin.gz").read_bytes())
                decoded = np.frombuffer(raw, dtype=record_type)
                records.extend(decoded["id"].tolist())
                np.testing.assert_array_equal(decoded["vector"][:, 0], [1, 1] if cluster_id == 0 else [-1, -1])
            self.assertEqual(sorted(records), [0, 1, 2, 3])
            self.assertEqual(gzip.decompress((output / "clusters/cluster_2.bin.gz").read_bytes()), b"")
            script = """
import {readFile} from 'node:fs/promises';
import {pathToFileURL} from 'node:url';
import {loadCentroids} from './client_wiki/assets/retrieval.mjs';
const url = pathToFileURL(process.argv[1] + '/manifest.json');
const index = {...JSON.parse(await readFile(url)), url: url.href};
globalThis.fetch = async url => new Response(await readFile(new URL(url)));
const errors = {};
for (const codec of Object.keys(index.centroids)) {
  const {centroids} = await loadCentroids(index, codec);
  let squared = 0;
  for (let i = 0; i < centroids.length; i++) squared += (centroids[i] - (i === 0 ? 1 : i === 384 ? -1 : 0)) ** 2;
  errors[codec] = Math.sqrt(squared / centroids.length);
}
console.log(JSON.stringify(errors));
"""
            run = subprocess.run(["node", "--input-type=module", "-e", script, str(output)],
                                 cwd=CODE, capture_output=True, text=True, check=True, timeout=30)
            errors = json.loads(run.stdout)
            self.assertEqual(set(errors), set(cluster.CODECS))
            for codec, error in errors.items():
                self.assertAlmostEqual(error, manifest["centroids"][codec]["rmse"], places=7)
                self.assertEqual((output / manifest["centroids"][codec]["file"]).stat().st_size,
                                 manifest["centroids"][codec]["bytes"])


if __name__ == "__main__":
    unittest.main()
