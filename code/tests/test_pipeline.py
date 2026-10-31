"""Small fixtures for publication boundaries and saved measurement validation."""

import gzip
import hashlib
import json
import struct
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fixtures import CODE, HttpConnection, load_benchmark, write_json
import cloudflare
import measurement_validation
import prepare_demo
import prepare_demo_pages
import run_comparisons


class PipelineChecks(unittest.TestCase):
    def test_network_configuration_returns_fields_without_mutating_suites(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "index.json"
            manifest.write_text("{}")
            suites = [{"id": label, "suite_path": f"noise-{label}/suite.json",
                       "indexes": [{"path": "../index.json"}]} for label in ("small", "medium", "large")]
            for suite in suites:
                (root / suite["suite_path"]).parent.mkdir()
            original = json.dumps(suites, sort_keys=True)
            hosting = root / "hosting.json"
            write_json(hosting, {"source_fixture_ids": {label: label for label in ("small", "medium", "large")},
                "source_files": {"index.json": {"sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}},
                "asset_base_url": "https://example.org/", "asset_base_urls": ["https://example.org/"], "bundle_id": "bundle"})
            network = run_comparisons.configure_network(suites, hosting, root)
            self.assertEqual(network["fixture_ids"], {label: label for label in ("small", "medium", "large")})
            self.assertEqual(network["hosting_manifest_url"], "https://example.org/routing.json")
            self.assertEqual(json.dumps(suites, sort_keys=True), original)

    def test_benchmark_helpers_import_without_running_and_reject_invalid_saved_data(self):
        with patch.object(sys, "argv", ["unrelated-command"]):
            summary = load_benchmark("summarize_inference.py")
        self.assertEqual(summary.stats([1, 3])["mean_ms"], 2)
        for values in ([], [0], [-1], [float("nan")]):
            with self.assertRaises(ValueError):
                summary.stats(values)
        noise = load_benchmark("noise-sensitivity.py")
        with patch.object(noise.Path, "read_text", return_value=json.dumps({"dimensions": 384, "query_count": 10000})), \
                patch.object(noise.np, "fromfile", return_value=np.zeros(10000, dtype="<u4")):
            with self.assertRaisesRegex(ValueError, "sorted distinct"):
                noise.analyze()

    def test_loopback_runner_rejects_foreign_hosts_for_all_methods(self):
        server = SimpleNamespace(server_address=("127.0.0.1", 8080))
        store = SimpleNamespace(config={"token": "secret"})
        for method in ("GET", "HEAD", "POST"):
            request = (f"{method} /experiment/run.json HTTP/1.0\r\nHost: foreign.example:8080\r\n"
                       "Origin: http://foreign.example:8080\r\n\r\n").encode()
            connection = HttpConnection(request)
            run_comparisons.ComparisonHandler(connection, ("127.0.0.1", 1234), server, store=store)
            self.assertIn(b" 403 ", connection.output.getvalue())
            self.assertNotIn(b"secret", connection.output.getvalue())
        for host in ("127.0.0.1:8080", "localhost:8080"):
            connection = HttpConnection(f"GET /experiment/run.json HTTP/1.0\r\nHost: {host}\r\n\r\n".encode())
            run_comparisons.ComparisonHandler(connection, ("127.0.0.1", 1234), server, store=store)
            self.assertIn(b" 200 ", connection.output.getvalue())
            self.assertIn(b"secret", connection.output.getvalue())

    def test_internet_assets_stay_inside_the_deployment(self):
        root = "https://example.org/assets/"
        self.assertEqual(run_comparisons.internet_url(root + "index.json", [root]), root + "index.json")
        self.assertEqual(run_comparisons.internet_url(root + "index.json",
                         ["https://other.example/assets/", root, "https://["]), root + "index.json")
        with self.assertRaisesRegex(ValueError, "outside this deployment"):
            run_comparisons.internet_url(root + "index.json", [])
        for url in ("http://example.org/a", "https://127.0.0.1/a", "https://10.0.0.1/a",
                    "https://[::1]/a", "https://0.0.0.0/a",
                    "https://example.local/a", "https://localhost./a", "https://127.1/a",
                    "https://0177.0.0.1/a", "https://user@example.org/a",
                    "https://example.org/other/a", root + "%2e%2e/private"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                run_comparisons.internet_url(url, [root])

    def test_asset_readiness_timeout_keeps_the_network_error(self):
        with patch.object(run_comparisons.time, "monotonic", side_effect=[0, 0, 2]), \
                patch.object(run_comparisons.socket, "create_connection", side_effect=OSError("DNS lookup failed")):
            with self.assertRaisesRegex(TimeoutError, "example.org:443: DNS lookup failed") as raised:
                run_comparisons.wait_for_asset_hosts_tls(["https://example.org/"], timeout=1)
        self.assertIsInstance(raised.exception.__cause__, OSError)

    def test_wranger_commands_are_bounded_and_project_names_are_validated(self):
        with patch.object(cloudflare.subprocess, "run", return_value=subprocess.CompletedProcess([], 0, stdout="")) as run:
            cloudflare.ensure_pages_project("paper-demo")
            self.assertEqual(run.call_count, 3)
            self.assertTrue(all(call.kwargs["timeout"] > 0 and call.kwargs["check"] for call in run.call_args_list))
            run.reset_mock()
            for name in ("-demo", "demo-", "Demo", "demo;command", "a" * 59):
                with self.assertRaises(ValueError):
                    cloudflare.ensure_pages_project(name)
            run.assert_not_called()

    def test_package_rejects_traversal_and_deployment_rejects_symlinks_and_oversized_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "prepared"
            (root / "secret").write_text("private")
            write_json(source / "noise-small/suite.json", {"queries": {"file": "../../secret"}})
            with self.assertRaisesRegex(ValueError, "leaves"):
                cloudflare.comparison_assets(source)
            deployment = root / "deployment"
            deployment.mkdir()
            (deployment / "asset").write_bytes(b"payload")
            first = cloudflare.deployment_digest(deployment)
            (deployment / "asset").write_bytes(b"changed")
            self.assertNotEqual(first, cloudflare.deployment_digest(deployment))
            (deployment / "link").symlink_to(root / "secret")
            with self.assertRaises(ValueError):
                cloudflare.deployment_digest(deployment)
            (deployment / "link").unlink()
            with (deployment / "asset").open("wb") as stream:
                stream.truncate(cloudflare.MAX_FILE_BYTES + 1)
            with self.assertRaises(ValueError):
                cloudflare.deployment_digest(deployment)

    def test_metadata_batches_preserve_unicode_and_order_across_shards(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "metadata").mkdir()
            rows = [("Café", "one"), ("שלום", "two"), ("third", "三"), ("last", "four")]
            shards = []
            for offset in (0, 2):
                path = root / f"source-{offset}.parquet"
                pq.write_table(pa.table({"title": [r[0] for r in rows[offset:offset + 2]],
                                         "text": [r[1] for r in rows[offset:offset + 2]]}), path)
                shards.append({"offset": offset, "count": 2, "path": path})
            with patch.object(prepare_demo, "BATCH_SIZE", 3):
                self.assertEqual(prepare_demo.write_metadata(shards, root), 4)
            decoded = []
            for batch, expected_first, expected_count in ((0, 0, 3), (1, 3, 1)):
                payload = gzip.decompress((root / "metadata" / f"meta_{batch}.bin.gz").read_bytes())
                self.assertEqual(struct.unpack_from("<4sII", payload), (b"WMD1", expected_first, expected_count))
                offset = 12
                for _ in range(expected_count):
                    title_bytes, text_bytes = struct.unpack_from("<II", payload, offset)
                    offset += 8
                    title = payload[offset:offset + title_bytes].decode("utf-8")
                    offset += title_bytes
                    text = payload[offset:offset + text_bytes].decode("utf-8")
                    offset += text_bytes
                    decoded.append((title, text))
                self.assertEqual(offset, len(payload))
            self.assertEqual(decoded, rows)

    def test_client_refresh_preserves_routes_and_uses_packaged_manifest_urls(self):
        with TemporaryDirectory() as directory:
            public = Path(directory)
            manifest = public / "data/search-full-brotli/manifest.json"
            write_json(manifest, {"count": 500, "metadata": {
                "format": "binary-http-v1", "batch_size": 500, "count": 500}})
            routes = public / "routing.json"
            routes.write_text("retained routes")
            prepare_demo_pages.copy_client(public)
            self.assertEqual(routes.read_text(), "retained routes")
            script = (public / "client_wiki/assets/search.mjs").read_text()
            self.assertIn("../data/search-full-brotli/", script)
            self.assertNotIn("../data/demo/pages/root/", script)
            self.assertIn("Cross-Origin-Embedder-Policy: require-corp", (public / "_headers").read_text())
            with patch.object(prepare_demo_pages.shutil, "copy2", side_effect=OSError("copy failed")):
                with self.assertRaises(OSError):
                    prepare_demo_pages.copy_client(public)
            self.assertEqual((public / "client_wiki/assets/search.mjs").read_text(), script)
            self.assertEqual(list(public.glob(".client-*")), [])
            with patch.object(prepare_demo_pages, "CLIENT_FILES", ("assets/search.mjs",)), \
                    patch.object(prepare_demo_pages.shutil, "copy2", side_effect=lambda source, target: target.write_text("changed layout")):
                with self.assertRaisesRegex(ValueError, "manifest paths changed"):
                    prepare_demo_pages.copy_client(public)
            self.assertEqual((public / "client_wiki/assets/search.mjs").read_text(), script)

    def test_python_and_browser_plans_have_identical_configurations_and_memberships(self):
        script = "import {paperPlan} from './client_wiki/assets/benchmark.mjs'; console.log(JSON.stringify(paperPlan()));"
        result = subprocess.run(["node", "--input-type=module", "-e", script], cwd=CODE,
                                capture_output=True, text=True, check=True, timeout=10)
        browser = json.loads(result.stdout)
        for label in run_comparisons.NOISE_LABELS:
            expected = {(label, item["clusters"], item["probes"], item["codec"], item["accuracy_only"]): item["memberships"]
                        for item in browser}
            actual = {key: value for key, value in run_comparisons.comparison_plan().items() if key[0] == label}
            self.assertEqual(actual, expected)

    def test_result_store_authenticates_and_saves_failure_without_the_token(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            np.array([1], dtype="<u4").tofile(root / "truth.u32.bin")
            config = {"id": "run", "token": "secret"}
            store = run_comparisons.ResultStore(root, config, {}, root)
            self.assertEqual(len(store.expected), 57)
            with self.assertRaises(ValueError):
                store.accept({"id": "run", "token": "wrong", "error": "failure"})
            self.assertFalse(store.done.is_set())
            store.accept({**config, "error": "interrupted download"})
            saved = json.loads((root / "results.json").read_text())
            self.assertNotIn("token", saved["config"])
            self.assertFalse(saved["complete"])
            self.assertEqual(saved["error"], "interrupted download")
            self.assertTrue(store.done.is_set())

    def test_measurement_validation_rejects_nan_and_conflicting_shared_trials(self):
        trial = {"setup_id": "setup", "sample": 0, "bytes": 10, "decodeMs": 0,
                 "downloadMs": 1, "requested_url": "https://example.org/centroid.gz"}
        setup = {"centroid_setup_id": "setup", "centroid_setup_repeats": 1, "centroidTrials": [trial]}
        row = {**setup, "codec": "f16", "clusters": 8192, "probes": 4, "k": 1, "queries": 1,
               "repeats": 1, "accuracy_only": False, "cluster_fetch_cache": "no-store",
               "trials": [{"repeat": 0, "query": 0, "recall": 1, "queryMs": 1}]}
        measurement_validation.validate_result(row)
        with self.assertRaises(ValueError):
            measurement_validation.validate_result({**row, "trials": [{**row["trials"][0], "queryMs": float("nan")}]})
        self.assertEqual(measurement_validation.centroid_downloads([setup, setup]), [trial])
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            measurement_validation.centroid_downloads([setup, {**setup, "centroidTrials": [{**trial, "bytes": 20}]}])

    def test_saved_input_validation_still_runs_with_python_optimization(self):
        script = CODE / "benchmarks/summarize_inference.py"
        with TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "inputs.json", {"passages": []})
            write_json(root / "results.json", {})
            result = subprocess.run([sys.executable, "-O", str(script), str(root)],
                                    capture_output=True, text=True, timeout=10)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("ValueError: Expected 100 distinct sampled passages", result.stderr)
            self.assertFalse((root / "summary.json").exists())


if __name__ == "__main__":
    unittest.main()
