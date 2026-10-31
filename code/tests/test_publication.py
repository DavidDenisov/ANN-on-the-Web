"""Regression checks for measurement and publication failure boundaries."""

import csv
import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fixtures import HttpConnection, write_json
import cloudflare
import figures
import prepare_demo
import run_comparisons
from download_dataset import DIMENSIONS, META_COLUMNS


def setup_record(clusters, codec):
    identity = f"{clusters}-{codec}"
    url = f"https://example.org/{identity}.gz"
    return {"clusters": clusters, "codec": codec, "centroid_setup_id": identity,
            "centroid_setup_repeats": 1000, "centroidTrials": [
                {"setup_id": identity, "sample": sample, "bytes": 10, "downloadMs": 2,
                 "decodeMs": 1, "requested_url": url, "resolved_url": url} for sample in range(1000)]}


def config():
    return {"version": 1, "id": "run", "token": "secret", "centroid_only": True,
            "cluster_counts": [8192, 16384, 32768], "cluster_probe_counts": [1, 4, 8],
            "probe_cluster_counts": [16384], "probe_counts": [1, 2, 3, 4, 6, 8, 10],
            "quantization_clusters": 16384, "quantization_probes": 4,
            "codecs": ["f16", "sq8", "sq6", "sq4", "int8", "pq4"],
            "k": 1, "query_count": 10000, "repeats": 1, "centroid_setup_repeats": 1000,
            "network_mode": "internet", "assets_base_url": "https://example.org/",
            "asset_base_urls": ["https://example.org/"], "fixture_ids": {"small": "fixture"},
            "hosting_bundle_id": "bundle"}


class PublicationChecks(unittest.TestCase):
    def test_asset_roots_preserve_public_host_and_url_validation(self):
        self.assertEqual(figures._asset_base("https://EXAMPLE.org/assets"), "https://example.org/assets/")
        self.assertEqual(figures._asset_base("https://8.8.8.8/assets"), "https://8.8.8.8/assets/")
        for value in ("http://example.org/assets", "https://127.0.0.1/", "https://[::1]/",
                      "https://10.0.0.1/", "https://localhost./", "https://127.1/",
                      "https://example.local/", "https://example.org:invalid/",
                      "https://user@example.org/", "https://example.org/?query", "https://example.org/#fragment"):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "Invalid asset base URL") as raised:
                figures._asset_base(value)
            self.assertIsInstance(raised.exception.__cause__, ValueError)
        for value in (None, 123, []):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "must be a string"):
                figures._asset_base(value)

    def test_collector_returns_json_errors_for_malformed_urls_and_numeric_overflow(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            np.array([1], dtype="<u4").tofile(root / "truth.u32.bin")
            store = run_comparisons.ResultStore(root, config(), {}, root)
            store.expected = {("small", 8192, 1, "f16", False): ["counts"]}
            server = SimpleNamespace(server_address=("127.0.0.1", 8080))
            for field, value in (("resolved_url", 123), ("resolved_url", {"host": "x"}),
                                 ("downloadMs", 10**400)):
                setup = setup_record(8192, "f16")
                setup["centroidTrials"][0][field] = value
                body = json.dumps({"id": "run", "token": "secret", "centroid_setups": [setup], "warmup": []}).encode()
                request = ("POST /api/results HTTP/1.0\r\nHost: localhost:8080\r\n"
                           f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n\r\n").encode() + body
                connection = HttpConnection(request)
                run_comparisons.ComparisonHandler(connection, ("127.0.0.1", 1234), server, store=store)
                response = connection.output.getvalue()
                self.assertIn(b" 400 ", response)
                self.assertIn("error", json.loads(response.split(b"\r\n\r\n", 1)[1]))
                self.assertNotIn("centroid_setups", store.report)

    def test_uploaded_indexes_require_matching_corpora_and_https_assets(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            for clusters in (8192, 16384, 32768):
                path = root / f"c{clusters}/manifest.json"
                descriptor = {"file": "https://example.org/payload.gz", "bytes": 1}
                manifest = {"version": 1, "metric": "normalized-l2", "dimensions": DIMENSIONS,
                    "record_bytes": 772, "count": 35167920, "clusters": clusters,
                    "cluster_sizes": [35167920] + [0] * (clusters - 1), "source": {"revision": "r"},
                    "source_shards": [], "centroids": {"f16": descriptor}, "cluster_files": [descriptor] * clusters}
                write_json(path, manifest)
            with patch.object(prepare_demo, "UPLOADED", root):
                self.assertEqual(len(prepare_demo.load_indexes()), 3)
                path = root / "c8192/manifest.json"
                original = json.loads(path.read_text())
                for manifest in (
                    {**original, "source": {"revision": "other"}},
                    {**original, "cluster_sizes": [35167919] + [0] * 8191},
                    {**original, "centroids": {"f16": {**original["centroids"]["f16"], "file": "http://example.org/payload.gz"}}},
                ):
                    write_json(path, manifest)
                    with self.assertRaises(ValueError):
                        prepare_demo.load_indexes()

    def test_upload_retry_receipts_and_root_publication_order(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            batches = [{"branch": f"batch-{i}", "path": f"batches/batch-{i}",
                        "base_url": f"https://batch-{i}.demo.pages.dev/"} for i in range(2)]
            hosting = {"version": 1, "project_name": "demo", "bundle_id": "a" * 64,
                       "asset_base_url": "https://demo.pages.dev/", "batches": batches,
                       "root": {"branch": "main", "path": "root", "base_url": "https://demo.pages.dev/"}}
            for descriptor in batches + [hosting["root"]]:
                target = root / descriptor["path"]
                target.mkdir(parents=True)
                (target / "payload").write_bytes(b"retained payload")
            for path in (root / "hosting.json", root / "root/routing.json"):
                write_json(path, hosting)
            uploaded = []
            def upload(command, **kwargs):
                self.assertGreater(kwargs["timeout"], 0)
                uploaded.append(command[-1])
                if command[-1] == "batch-1":
                    raise subprocess.CalledProcessError(1, command)
            with patch.object(cloudflare.subprocess, "run", side_effect=upload):
                with self.assertRaises(subprocess.CalledProcessError):
                    cloudflare.deploy_cloudflare_pages(root)
            self.assertEqual(uploaded, ["batch-0", "batch-1"])
            self.assertEqual(set(json.loads((root / "deployment.json").read_text())["completed"]), {"batch-0"})
            self.assertTrue((root / "batches/batch-1/payload").exists())
            with patch.object(cloudflare.subprocess, "run") as retry:
                receipt = cloudflare.deploy_cloudflare_pages(root)
            self.assertEqual([call.args[0][-1] for call in retry.call_args_list], ["batch-1", "main"])
            self.assertEqual(set(receipt["completed"]), {"batch-0", "batch-1", "main"})

    def test_failed_streamed_upload_retains_both_payload_copies(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            data, stage = root / "data", root / "stage"
            for folder in (data / "c8192/clusters", stage / "batches/batch-0"):
                folder.mkdir(parents=True)
                (folder / "payload").write_bytes(b"retained payload")
            publisher = cloudflare.StreamingPagesPublisher(root / "publication", "demo")
            with patch.object(cloudflare, "prepare_cloudflare_pages", return_value=stage), \
                    patch.object(cloudflare, "deploy_cloudflare_pages", side_effect=RuntimeError("upload failed")):
                with self.assertRaisesRegex(RuntimeError, "upload failed"):
                    publisher(data, 8192)
            self.assertTrue((data / "c8192/clusters/payload").exists())
            self.assertTrue((stage / "batches/batch-0/payload").exists())
            self.assertIsNone(publisher.hosting)

    def test_centroid_report_passes_from_collector_to_exporter(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            np.array([1], dtype="<u4").tofile(root / "truth.u32.bin")
            settings = config()
            store = run_comparisons.ResultStore(root, settings, {}, root)
            pairs = {(key[1], key[3]) for key in store.expected}
            setups = [setup_record(*pair) for pair in sorted(pairs)]
            store.accept({"id": "run", "token": "secret", "centroid_setups": setups, "warmup": []})
            store.accept({"id": "run", "token": "secret", "complete": True, "total": 0})
            output = figures.export_figures(store.report, root / "figures")
            self.assertTrue((output / "centroid-counts-table.csv").is_file())
            self.assertTrue((output / "compression-timing-table.tex").is_file())
            self.assertTrue(store.done.is_set())

    def test_measurements_reject_conflicts_duplicates_wrong_targets_and_early_completion(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            np.array([1], dtype="<u4").tofile(root / "truth.u32.bin")
            settings = {**config(), "centroid_only": False, "query_count": 1}
            fixture = {"count": 35167920, "noise": {"level": .05}}
            store = run_comparisons.ResultStore(root, settings, {"small": fixture}, root)
            store.expected = {("small", 8192, 1, "f16", False): ["counts"]}
            setup = setup_record(8192, "f16")
            credentials = {"id": "run", "token": "secret"}
            warmup = [{"clusters": count, "cluster_files": count, "cluster_bytes": 1,
                       "centroid_files": int(count == 8192)} for count in settings["cluster_counts"]]
            store.accept({**credentials, "centroid_setups": [setup], "warmup": warmup})
            trial = {"repeat": 0, "query": 0, "recall": 1, "queryMs": 3, "targetId": 1, "resultIds": [1],
                     "downloads": [{"requested_url": "https://example.org/cluster.gz", "resolved_url": "https://example.org/cluster.gz"}]}
            row = {**setup, "metric": "source-recovery", "probes": 1, "k": 1, "queries": 1, "repeats": 1,
                   "accuracy_only": False, "cluster_fetch_cache": "no-store", "trials": [trial],
                   "network_mode": "internet", "fixture_id": "fixture", "asset_base_url": "https://example.org/",
                   "asset_base_urls": ["https://example.org/"], "hosting_bundle_id": "bundle", "noise": fixture["noise"],
                   "centroidBytes": 10, "bytesPerQuery": 20, "centroidDownloadMs": {"mean": 2}, "centroidDecodeMs": {"mean": 1}}
            packet = {**credentials, "noise_label": "small", "memberships": ["counts"], "result": row}
            with self.assertRaisesRegex(ValueError, "incomplete"):
                store.accept({**credentials, "complete": True, "total": 1})
            for invalid in ({**row, "trials": [{**trial, "resultIds": [2]}]},
                            {**row, "cluster_fetch_cache": "force-cache"},
                            {**row, "repeats": 2},
                            {**row, "centroid_setup_id": "wrong"},
                            {**row, "centroidTrials": [{**sample, "bytes": 20} for sample in row["centroidTrials"]]}):
                with self.assertRaises(ValueError):
                    store.accept({**packet, "result": invalid})
            self.assertFalse((root / "measurements.jsonl").exists())
            store.accept(packet)
            with self.assertRaisesRegex(ValueError, "duplicate"):
                store.accept(packet)
            store.accept({**credentials, "complete": True, "total": 1})
            with (root / "summary.csv").open() as stream:
                saved, = csv.DictReader(stream)
            self.assertEqual(saved["query_mean_ms"], "3")
            self.assertEqual(saved["source_recovery"], "1")
            self.assertEqual(saved["centroid_bytes"], "10")

    def test_metadata_source_rejects_snapshot_order_offset_and_count_changes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = {"dimensions": DIMENSIONS, "dtype": "<f2", "dataset": "fixture", "revision": "r",
                      "shards": ["a.parquet", "b.parquet"]}
            write_json(root / "manifest.json", source)
            for name in ("a", "b"):
                shard = root / name
                shard.mkdir()
                np.save(shard / "vectors.npy", np.ones((1, DIMENSIONS), dtype="<f2"))
                pq.write_table(pa.table({key: ["value"] for key in META_COLUMNS}), shard / "passages.parquet")
                write_json(shard / "info.json", {"source": name + ".parquet", "rows": 1})
            index = {"count": 2, "source": {"dataset": "fixture", "revision": "r"}, "source_shards": [
                {"name": name + ".parquet", "offset": offset, "count": 1} for offset, name in enumerate(("a", "b"))]}
            first, second = index["source_shards"]
            invalid = [
                {**index, "source": {**index["source"], "revision": "other"}},
                {**index, "source_shards": [second, first]},
                {**index, "source_shards": [first, {**second, "offset": 2}]},
                {**index, "source_shards": [first, {**second, "count": 2}]},
            ]
            with patch.object(prepare_demo, "SOURCE", root):
                self.assertEqual([s["offset"] for s in prepare_demo.metadata_source(index)], [0, 1])
                for item in invalid:
                    with self.assertRaises(ValueError):
                        prepare_demo.metadata_source(item)
                    with patch.object(prepare_demo, "load_indexes", return_value=[("manifest.json", item)]):
                        with self.assertRaises(ValueError):
                            prepare_demo.prepare_demo(root / "export")
                    self.assertFalse((root / "export").exists())


if __name__ == "__main__":
    unittest.main()
