"""Deterministic success paths for offline geometry and saved inference reports."""

import csv
import gzip
import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from fixtures import load_benchmark, write_json
from cluster import noisy_queries


class OfflineAnalysisChecks(unittest.TestCase):
    def test_noise_analysis_reproduces_saved_queries_and_aggregates_geometry(self):
        noise = load_benchmark("noise-sensitivity.py")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            prepared = root / "data/benchmark/prepared"
            raw = np.random.default_rng(17).normal(size=(10000, 384))
            raw[:, 0] = np.where(np.arange(10000) % 2, -50, 50)
            raw = raw.astype("<f2")
            folder = root / "data/wikipedia/a"
            folder.mkdir(parents=True)
            np.save(folder / "vectors.npy", raw)
            suite = {"dimensions": 384, "query_count": 10000, "id": "fixture", "seed": 17,
                "source": {"dataset": "fixture", "revision": "r", "shards": [
                    {"name": "a.parquet", "offset": 0, "count": 10000}]}}
            for label, alpha in (("small", .05), ("medium", .15), ("large", .4)):
                write_json(prepared / f"noise-{label}/suite.json", suite)
                noisy_queries(raw, alpha, 17).tofile(prepared / f"noise-{label}/queries.f32.bin")
            np.arange(10000, dtype="<u4").tofile(prepared / "truth.u32.bin")
            centroids = np.zeros((2, 384), dtype="<f2")
            centroids[:, 0] = [1, -1]
            write_json(prepared / "c2/manifest.json", {"clusters": 2,
                "centroids": {"f16": {"file": "centroids.gz"}}})
            (prepared / "c2/centroids.gz").write_bytes(gzip.compress(centroids.tobytes()))
            summary = root / "data/benchmark/retrieval/summary.csv"
            summary.parent.mkdir()
            with summary.open("w", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["clusters", "codec", "checks", "probes", "noise_level", "queries", "source_recovery"])
                writer.writeheader()
                writer.writerows({"clusters": 2, "codec": "f16", "checks": "counts", "probes": probe,
                    "noise_level": level, "queries": 10000, "source_recovery": .9}
                    for level in (.05, .15, .4) for probe in (1, 4, 8))
            with patch.object(noise, "ROOT", root), redirect_stdout(io.StringIO()):
                report, samples = noise.analyze(2)
            self.assertEqual(report["samples"], 1000)
            self.assertEqual(len(report["cases"]), 16)
            self.assertEqual(len(samples), 16000)
            self.assertEqual(len(report["saved_full_run_source_recovery"]), 9)
            clean = next(row for row in report["cases"] if row["kind"] == "gaussian-l2" and row["alpha"] == 0)
            self.assertEqual(clean["source_cluster_coverage_1"], 1)
            self.assertLess(clean["l2_distance_mean"], 1e-12)
            self.assertTrue(all(0 <= row["source_cluster_coverage_1"] <= 1 for row in report["cases"]))

    def test_inference_summary_exports_the_complete_scheduled_fixture(self):
        summarize = load_benchmark("summarize_inference.py")
        with TemporaryDirectory() as directory:
            root = Path(directory)
            lengths = [64, 128, 256]
            passages = [{"id": i, "text": "word " * 256, "words": 256} for i in range(100)]
            schedule = [{"repeat": i // 100, "input": i % 100, "tokens": lengths[i % 3]} for i in range(1000)]
            inputs = {"passages": passages, "eligible_count": 616, "corpus_count": 100000, "lengths": lengths,
                "modelRevision": "751bff37182d3f1213fa05d7196b954e230abad9", "startup_trials": 100,
                "trials": schedule, "seed": 17}
            startups = [{"threads": 4, "crossOriginIsolated": True, "libraryImportMs": 1, "downloadMs": 2,
                "initializationMs": 3, "modelBytes": 40, "runtimeBytes": 20, "assets": [{"bytes": 10}] * 6} for _ in range(100)]
            results = {"threads": 4, "requestedThreads": 4, "model": "Xenova/all-MiniLM-L6-v2", "dtype": "q8",
                "transformers": "4.3.1", "browserVersion": "155.0", "userAgent": "Chromium", "measuredAt": "fixture", "cpu": "fixture",
                "startups": startups, "lengths": [{"id": p["id"], "tokens": tokens} for p in passages for tokens in lengths],
                "trials": [{**trial, "id": passages[trial["input"]]["id"], "modelInputTokens": trial["tokens"],
                    "inferenceMs": trial["tokens"] / 64} for trial in schedule]}
            write_json(root / "inputs.json", inputs)
            write_json(root / "results.json", results)
            with patch.object(sys, "argv", ["summary", str(root)]), redirect_stdout(io.StringIO()):
                summarize.main()
            saved = json.loads((root / "summary.json").read_text())
            self.assertEqual([saved["lengths"][str(tokens)]["samples"] for tokens in lengths], [334, 333, 333])
            self.assertEqual([saved["lengths"][str(tokens)]["mean_ms"] for tokens in lengths], [1, 2, 4])
            self.assertEqual(saved["downloadMs"]["p95_ms"], 2)
            self.assertEqual(saved["modelBytes"] + saved["runtimeBytes"], 60)
            self.assertFalse((root / ".summary.json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
