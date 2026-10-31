#!/usr/bin/env python3
"""Prepare and run the paper's comparison matrix in a browser, saving every result."""

import argparse
import csv
import hashlib
import json
import math
import secrets
import socket
import ssl
import statistics
import threading
import time
import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urljoin, urlsplit

import numpy as np

from paper_settings import CLUSTER_COUNTS, CORPUS_COUNT, DEFAULT_INPUT, NOISE_LABELS, NOISE_LEVELS, QUERY_COUNT, comparison_memberships, comparison_settings
from cloudflare import StreamingPagesPublisher, ensure_pages_project
from measurement_validation import centroid_downloads, validate_centroid_samples, finite_number, is_public_host, positive_integer, validate_result


CODE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = CODE_ROOT / "data" / "benchmark" / "run-new"
MAX_RESULT_BYTES = 256 * 1024 * 1024


def internet_url(value: str, bases: list[str] | None = None) -> str:
    """Reject loopback/private assets and requests outside the deployed package."""
    if not isinstance(value, str):
        raise ValueError("Internet assets require a public HTTPS URL")
    parsed = urlsplit(value)
    host = (parsed.hostname or "").lower().rstrip(".")
    if parsed.scheme != "https" or not host or parsed.username or parsed.password:
        raise ValueError("Internet assets require a public HTTPS URL")
    if not is_public_host(host):
        raise ValueError("Internet assets cannot use localhost or private/reserved addresses")
    if bases is not None:
        path = unquote(parsed.path)
        roots = (urlsplit(base) for base in bases)
        if not any(parsed.netloc.lower() == root.netloc.lower()
                   and path.startswith(unquote(root.path).rstrip("/") + "/")
                   and ".." not in path.split("/") for root in roots):
            raise ValueError("Asset URL is outside this deployment")
    return parsed.geturl()


def wait_for_asset_hosts_tls(roots: list[str], timeout: float = 900) -> None:
    """New Pages preview hosts can need time before their TLS certificates are ready."""
    pending = {(parsed.hostname, parsed.port or 443)
               for parsed in (urlsplit(internet_url(root)) for root in roots)}
    failures = {}
    context = ssl.create_default_context()
    deadline = time.monotonic() + timeout
    while pending:
        for host, port in sorted(pending):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                with socket.create_connection((host, port), timeout=min(10, remaining)) as connection:
                    with context.wrap_socket(connection, server_hostname=host):
                        pass
                pending.remove((host, port))
            except OSError as error:
                failures[host, port] = error
        if not pending:
            return
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            details = "; ".join(f"{host}:{port}: {failures.get((host, port), 'not checked before deadline')}"
                                for host, port in sorted(pending))
            raise TimeoutError(f"HTTPS assets are not ready after {timeout}s: {details}") from next(iter(failures.values()), None)
        print(f"Waiting for verified HTTPS on {len(pending)} asset host(s); retrying in {min(15, remaining):g}s…", flush=True)
        time.sleep(min(15, remaining))


def load_prepared_data(data: Path) -> list[dict]:
    suites = []
    for label, noise in zip(NOISE_LABELS, NOISE_LEVELS):
        path = f"noise-{label}/suite.json"
        suite = json.loads((data / path).read_text(encoding="utf-8"))
        if (not suite.get("id") or suite.get("metric") != "source-recovery"
                or suite.get("query_kind") != "noisy-passages"
                or suite.get("count") != CORPUS_COUNT or suite.get("dimensions") != 384
                or suite.get("query_count") != QUERY_COUNT or suite.get("truth_k") != 1
                or suite.get("noise", {}).get("level") != noise
                or sorted(item["clusters"] for item in suite["indexes"]) != list(CLUSTER_COUNTS)
                or suite["source"]["completed_shards"] != suite["source"]["planned_shards"]
                or suite["source"]["available_passages"] != CORPUS_COUNT):
            raise ValueError(f"Prepared {label} suite does not match the paper")
        suite["suite_path"] = path
        suites.append(suite)
    if (type(suites[0].get("seed")) is not int or not 0 <= suites[0]["seed"] < 2**31
            or any(suite.get("seed") != suites[0]["seed"] for suite in suites)):
        raise ValueError("All paper suites must share a recorded seed")
    return suites


def configure_network(suites: list[dict], hosting_manifest: Path, data: Path) -> dict[str, object]:
    """Validate a deployment and return network fields without changing the suites."""
    fixture_ids = dict(zip(NOISE_LABELS, (suite["id"] for suite in suites)))
    hosting = json.loads(Path(hosting_manifest).read_text(encoding="utf-8"))
    if not isinstance(hosting, dict) or hosting.get("source_fixture_ids") != fixture_ids:
        raise ValueError("The deployment was packaged from different prepared data; rebuild and upload it")
    root = Path(data).resolve()
    checked = set()
    for suite in suites:
        for index in suite["indexes"]:
            path = ((root / suite["suite_path"]).parent / index["path"]).resolve(strict=True)
            if not path.is_relative_to(root):
                raise ValueError("An index manifest leaves the prepared directory")
            if path in checked:
                continue
            expected = hosting.get("source_files", {}).get(path.relative_to(root).as_posix(), {}).get("sha256")
            if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError("The hosted clustering does not match the prepared index; rebuild and upload it")
            checked.add(path)
    base = internet_url(hosting["asset_base_url"]).rstrip("/") + "/"
    bases = hosting["asset_base_urls"]
    if not isinstance(bases, list) or not bases or base not in bases:
        raise ValueError("The hosting allowlist must include the asset root")
    for value in bases:
        internet_url(value)
    return {"fixture_ids": fixture_ids, "network_mode": "internet", "assets_base_url": base,
            "asset_base_urls": bases, "hosting_bundle_id": hosting["bundle_id"],
            "hosting_manifest_url": urljoin(base, "routing.json")}


def comparison_plan() -> dict[tuple[str, int, int, str, bool], list[str]]:
    """The paper's 57 unique latency and accuracy configurations."""
    plan = {}
    memberships = comparison_memberships()
    for label in NOISE_LABELS:
        for name, configurations in memberships.items():
            for key in sorted(configurations):
                plan.setdefault((label, *key), []).append(name)
    return plan


class ResultStore:
    def __init__(self, output, config, fixtures, data):
        self.output, self.config = Path(output), config
        self.data = Path(data)
        self.expected = comparison_plan()
        self.seen = set()
        self.lock, self.done = threading.Lock(), threading.Event()
        self.report = {"version": 1, "config": {k: v for k, v in config.items() if k != "token"},
                       "fixtures": fixtures, "user_agent": "", "measurements": [], "complete": False}
        self.targets = np.fromfile(self.data / "truth.u32.bin", dtype="<u4")
        self._save()

    def _save(self):
        temporary = self.output / ".results.json.tmp"
        temporary.write_text(json.dumps(self.report, indent=2) + "\n", encoding="utf-8")
        temporary.replace(self.output / "results.json")

    def _centroid_csv(self, rows):
        setup_fields = ("clusters", "codec", "samples", "centroid_bytes", "download_mean_ms", "download_p95_ms",
                        "decode_mean_ms", "decode_p95_ms", "centroid_setup_id")

        with (self.output / "centroid-summary.csv").open("x", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=setup_fields)
            writer.writeheader()
            for clusters, codec in sorted({(row["clusters"], row["codec"]) for row in rows}):
                shared_rows = [row for row in rows if (row["clusters"], row["codec"]) == (clusters, codec)]
                samples = centroid_downloads(shared_rows)
                downloads = sorted(sample["downloadMs"] for sample in samples)
                decodes = sorted(sample["decodeMs"] for sample in samples)
                writer.writerow({
                    'clusters': clusters,
                    'codec': codec,
                    'samples': len(samples),
                    'centroid_bytes': samples[0]['bytes'],
                    'download_mean_ms': statistics.mean(downloads),
                    'download_p95_ms': downloads[math.ceil(len(downloads) * 0.95) - 1],
                    'decode_mean_ms': statistics.mean(decodes),
                    'decode_p95_ms': decodes[math.ceil(len(decodes) * 0.95) - 1],
                    'centroid_setup_id': shared_rows[0]['centroid_setup_id'],
                })

    def _csv(self):
        fields = ("noise", "noise_level", "checks", "clusters", "probes", "codec", "k", "queries", "repeats",
                  "accuracy_only", "cluster_fetch_cache", "source_recovery", "query_mean_ms", "query_p95_ms", "bytes_per_query", "centroid_bytes", "centroid_setup_id",
                  "centroid_download_mean_ms", "centroid_decode_mean_ms", "network_mode", "asset_base_url", "fixture_id")
        with (self.output / "summary.csv").open("x", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=fields)
            writer.writeheader()
            for measurement in self.report["measurements"]:
                row = measurement["result"]
                trials = row["trials"]
                times = sorted(trial["queryMs"] for trial in trials) if not row["accuracy_only"] else []
                writer.writerow({
                    'noise': measurement['noise_label'],
                    'noise_level': self.report['fixtures'][measurement['noise_label']]['noise']['level'],
                    'checks': ';'.join(measurement['memberships']),
                    'clusters': row['clusters'],
                    'probes': row['probes'],
                    'codec': row['codec'],
                    'k': row['k'],
                    'queries': row['queries'],
                    'repeats': row['repeats'],
                    'accuracy_only': row['accuracy_only'],
                    'cluster_fetch_cache': row['cluster_fetch_cache'],
                    'source_recovery': statistics.mean((trial['recall'] for trial in trials)),
                    'query_mean_ms': statistics.mean(times) if times else '',
                    'query_p95_ms': times[math.ceil(len(times) * 0.95) - 1] if times else '',
                    'bytes_per_query': row['bytesPerQuery'],
                    'centroid_bytes': row['centroidBytes'],
                    'centroid_setup_id': row['centroid_setup_id'],
                    'centroid_download_mean_ms': row['centroidDownloadMs']['mean'],
                    'centroid_decode_mean_ms': row['centroidDecodeMs']['mean'],
                    'network_mode': row['network_mode'],
                    'asset_base_url': row.get('asset_base_url', ''),
                    'fixture_id': row['fixture_id'],
                })

    def _accept_centroid_setups(self, packet):
        setups = packet["centroid_setups"]
        expected = {(key[1], key[3]) for key in self.expected}
        if (self.seen or self.report.get("centroid_setups") or not isinstance(setups, list)
                or len(setups) != len(expected)):
            raise ValueError("Unexpected centroid setup stage")
        seen = set()
        for setup in setups:
            if not isinstance(setup, dict):
                raise ValueError("Each centroid setup must be an object")
            clusters = positive_integer(setup.get("clusters"), "setup clusters")
            codec = setup.get("codec")
            if not isinstance(codec, str) or (clusters, codec) not in expected or (clusters, codec) in seen:
                raise ValueError("Unexpected or duplicate centroid setup")
            seen.add((clusters, codec))
            validate_centroid_samples(setup)
            if setup["centroid_setup_repeats"] != self.config["centroid_setup_repeats"]:
                raise ValueError("Centroid setup sample count does not match this run")
            for sample in setup["centroidTrials"]:
                internet_url(sample["requested_url"], self.config["asset_base_urls"])
                internet_url(sample.get("resolved_url", ""))
        if len({setup["centroid_setup_id"] for setup in setups}) != len(expected):
            raise ValueError("Different centroid files must have distinct setup identities")
        warmup = packet.get("warmup")
        warmed_counts = [] if self.config["centroid_only"] else list(CLUSTER_COUNTS)
        if (not isinstance(warmup, list) or len(warmup) != len(warmed_counts)
                or any(not isinstance(item, dict) or type(item.get("clusters")) is not int for item in warmup)
                or {item.get("clusters") for item in warmup} != set(warmed_counts)):
            raise ValueError("Asset warm-up does not match this run's settings")
        for item in warmup:
            if (item.get("cluster_files") != item["clusters"] or item.get("centroid_files")
                    != len({codec for clusters, codec in expected if clusters == item["clusters"]})):
                raise ValueError("Incomplete asset warm-up")
            positive_integer(item.get("cluster_bytes"), "warm-up cluster bytes")
        self.report["warmup"] = warmup
        self.report["centroid_setups"] = setups
        self._save()
        self._centroid_csv(setups)
        print(f"Saved {len(setups)} centroid setups and centroid-summary.csv", flush=True)

    def _accept_measurement(self, packet):
        row, label = packet.get("result"), packet.get("noise_label")
        if not isinstance(row, dict) or not isinstance(label, str):
            raise ValueError("Missing measurement")
        key = (label, row.get("clusters"), row.get("probes"), row.get("codec"), row.get("accuracy_only"))
        if key not in self.expected or key in self.seen or packet.get("memberships") != self.expected[key]:
            raise ValueError("Unexpected, duplicate or incorrectly grouped measurement")


        validate_result(row)
        if row.get("centroid_setup_repeats") != self.config["centroid_setup_repeats"]:
            raise ValueError("Centroid setup sample count does not match this run")
        setups = self.report.get("centroid_setups", [])
        setup = next((setup for setup in setups if (setup["clusters"], setup["codec"])
                      == (row["clusters"], row["codec"])), None)
        if not setup or row.get("centroid_setup_id") != setup["centroid_setup_id"]:
            raise ValueError("Measure and save centroid setup before query configurations")
        centroid_downloads(setups + [row])
        if (row.get("network_mode") != self.config["network_mode"]
                or row.get("fixture_id") != self.config["fixture_ids"][label]):
            raise ValueError("Measurement network mode or fixture identity does not match this run")
        if self.config["network_mode"] == "internet":
            if (row.get("asset_base_url") != self.config["assets_base_url"]
                    or row.get("asset_base_urls") != self.config["asset_base_urls"]
                    or row.get("hosting_bundle_id") != self.config.get("hosting_bundle_id")):
                raise ValueError("Measurement asset deployment does not match this run")
            transfers = list(row["centroidTrials"])
            for trial in row["trials"]:
                if not trial.get("downloads"):
                    raise ValueError("Internet query trials must record actual asset downloads")
                transfers.extend(trial["downloads"])
            for transfer in transfers:
                internet_url(transfer.get("requested_url", ""), self.config["asset_base_urls"])
                internet_url(transfer.get("resolved_url", ""))
        positive_integer(row.get("centroidBytes"), "centroidBytes")
        finite_number(row.get("bytesPerQuery"), "bytesPerQuery")
        for name in ("centroidDownloadMs", "centroidDecodeMs"):
            if not isinstance(row.get(name), dict):
                raise ValueError(f"Missing {name} measurements")
            finite_number(row[name].get("mean"), name)
        if (row.get("metric") != "source-recovery" or row.get("k") != self.config["k"]
                or row.get("queries") != self.config["query_count"]
                or row.get("noise") != self.report["fixtures"][label]["noise"]):
            raise ValueError("Measurement settings do not match this run")
        for trial in row["trials"]:
            ids = trial.get("resultIds")
            target = int(self.targets[trial["query"]])
            if (not isinstance(ids, list) or len(ids) > row["k"] or len(set(ids)) != len(ids)
                    or any(type(value) is not int or not 0 <= value < self.report["fixtures"][label]["count"] for value in ids)
                    or trial.get("targetId") != target or trial["recall"] != float(target in ids)):
                raise ValueError("Trial does not measure recovery of its source passage")
        measurement = {"noise_label": label, "memberships": packet["memberships"], "result": row}
        with (self.output / "measurements.jsonl").open("a", encoding="utf-8") as journal:
            journal.write(json.dumps(measurement, separators=(",", ":")) + "\n")
        self.report["measurements"].append(measurement)
        self.seen.add(key)
        print(f"Saved {len(self.seen)}/{len(self.expected)}: {label}, C={key[1]}, probes={key[2]}, {key[3]}", flush=True)

    def accept(self, packet: dict) -> None:
        if not isinstance(packet, dict) or packet.get("id") != self.config["id"] or packet.get("token") != self.config["token"]:
            raise ValueError("Invalid run credentials")
        with self.lock:
            if self.done.is_set():
                raise ValueError("This comparison run has already stopped")
            self.report["user_agent"] = str(packet.get("user_agent", ""))[:2000]
            if "error" in packet:
                self.report["error"] = str(packet["error"])[:4000]
                self._save()
                self.done.set()
                return
            if "centroid_setups" in packet:
                self._accept_centroid_setups(packet)
                return
            if packet.get("complete") is True:
                if self.config.get("centroid_only"):
                    if not self.report.get("centroid_setups") or self.seen or packet.get("total") != 0:
                        raise ValueError("The centroid setup stage is incomplete")
                else:
                    if self.seen != set(self.expected) or packet.get("total") != len(self.expected):
                        raise ValueError("The comparison matrix is incomplete")
                    self._csv()
                self.report["complete"] = True
                self._save()
                (self.output / "measurements.jsonl").unlink(missing_ok=True)
                self.done.set()
                return
            if self.config.get("centroid_only"):
                raise ValueError("Centroid-only runs do not accept query measurements")
            self._accept_measurement(packet)


class ComparisonHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, store=None, **kwargs):
        self.store = store
        super().__init__(*args, directory=str(CODE_ROOT), **kwargs)

    def log_message(self, *args):
        pass

    def parse_request(self) -> bool:
        if not super().parse_request():
            return False
        address, port = self.server.server_address[:2]
        allowed = {f"{address}:{port}", f"localhost:{port}"}
        if port == 80:
            allowed.update((address, "localhost"))
        hosts = self.headers.get_all("Host", [])
        # A rebound foreign hostname must not read the local run token.
        if len(hosts) != 1 or hosts[0].lower() not in allowed:
            self.send_error(403, "Use this server's loopback address and port.")
            return False
        return True

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        if urlsplit(self.path).path.startswith("/client_wiki/"):
            self.send_header("Cross-Origin-Opener-Policy", "same-origin")
            self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()

    def _reply(self, status, body=b"", content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json_error(self, status, message):
        self._reply(status, json.dumps({"error": message}).encode("utf-8"))

    def do_GET(self):
        path = urlsplit(self.path).path
        if path.startswith("/experiment/"):
            if self.store is None or path != "/experiment/run.json":
                return self._json_error(404, "Unknown experiment endpoint.")
            return self._reply(200, json.dumps(self.store.config).encode("utf-8"))
        return super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/results" or self.store is None:
            return self._json_error(404, "Unknown endpoint.")
        origin = self.headers.get("Origin")
        if origin:
            try:
                parsed = urlsplit(origin)
                same_origin = parsed.scheme == "http" and parsed.netloc.lower() == self.headers.get("Host", "").lower()
            except ValueError:
                same_origin = False
            if not same_origin:
                return self._json_error(403, "Requests must come from this server's page.")
        if self.headers.get_content_type() != "application/json":
            return self._json_error(415, "Send JSON data.")
        try:
            length = int(self.headers.get("Content-Length", "0"))
            limit = MAX_RESULT_BYTES
            if not 0 < length <= limit:
                raise ValueError(f"Each request must be between 1 byte and {limit // (1024 * 1024)} MiB")
            packet = json.loads(self.rfile.read(length))
        except ValueError as error:
            return self._json_error(400, str(error))
        try:
            self.store.accept(packet)
        except (ValueError, TypeError, KeyError, IndexError) as error:
            return self._json_error(400, str(error))
        except OSError as error:
            with self.store.lock:
                self.store.report.update(complete=False, error=f"Could not save results: {error}")
                self.store.done.set()
            return self._json_error(500, "Could not save results; check the output disk.")
        return self._reply(200)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serve", action="store_true", help="serve the search demo and clustering lab")
    parser.add_argument("--port", type=int, default=8080, help="local server port with --serve")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="new run directory")
    parser.add_argument("--prepare-only", action="store_true", help="prepare the paper indexes without upload or timing")
    parser.add_argument("--prepared-data", type=Path, help="reuse the paper's prepared data")
    parser.add_argument("--hosting-manifest", type=Path, help="hosting.json for that preparation")
    parser.add_argument("--cloudflare-project", help="build and upload the paper indexes, then measure them")
    parser.add_argument("--no-browser", action="store_true", help="print the control URL without opening it")
    parser.add_argument("--centroid-only", action="store_true", help="only measure the eight centroid files, 1,000 trials each")
    args = parser.parse_args()
    if args.serve:
        with ThreadingHTTPServer(("127.0.0.1", args.port), ComparisonHandler) as server:
            server.daemon_threads = True
            print(f"Clustering lab: http://127.0.0.1:{server.server_port}/client_wiki/benchmark.html", flush=True)
            try:
                server.serve_forever()
            except KeyboardInterrupt:
                pass
        return
    if args.prepare_only:
        if args.prepared_data or args.hosting_manifest or args.cloudflare_project or args.centroid_only:
            parser.error("--prepare-only builds fresh data; omit hosting, prepared data and --centroid-only")
    elif args.cloudflare_project:
        if args.prepared_data or args.hosting_manifest or args.centroid_only:
            parser.error("--cloudflare-project builds and uploads fresh data; omit other input modes")
    elif not (args.prepared_data and args.hosting_manifest):
        parser.error("Supply --prepared-data and --hosting-manifest, or prepare fresh paper data")
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; choose a new run directory")
    data = args.prepared_data.resolve() if args.prepared_data else output / "data"
    publisher = None
    if args.prepared_data:
        suites = load_prepared_data(data)
    else:
        # Artifact inspection and serving do not need the GPU/download dependency stack.
        from cluster import build_comparison_data, require_gpu

        require_gpu()
        if args.cloudflare_project:
            ensure_pages_project(args.cloudflare_project)
            publisher = StreamingPagesPublisher(output / "pages", args.cloudflare_project)
        build_comparison_data(args.input, data, publish_index=publisher)
        suites = load_prepared_data(data)
    config = {
        "version": 1, "id": secrets.token_hex(16), "token": secrets.token_hex(24),
        "suites": [{"label": label, "noise": noise, "url": suite["suite_path"]}
                   for label, noise, suite in zip(NOISE_LABELS, NOISE_LEVELS, suites)],
        **comparison_settings(),
        "centroid_setup_repeats": 1000, "centroid_only": args.centroid_only,
        "seed": suites[0]["seed"], "results_url": "/api/results",
    }
    output.mkdir(parents=True, exist_ok=True)
    if args.prepare_only:
        metadata = {"version": 1, "count": CORPUS_COUNT, "dimensions": 384,
                    "fixture_ids": dict(zip(NOISE_LABELS, (suite["id"] for suite in suites))),
                    "settings": {key: value for key, value in config.items() if key not in ("id", "token")}}
        (output / "preparation.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        print(f"Prepared the paper's three indexes and noise suites: {data}", flush=True)
        return
    if publisher is not None:
        args.hosting_manifest = publisher.finish()
    config.update(configure_network(suites, args.hosting_manifest, data))
    for item, suite in zip(config["suites"], suites):
        item["url"] = urljoin(config["assets_base_url"], suite["suite_path"])
    wait_for_asset_hosts_tls(config["asset_base_urls"])
    store = ResultStore(output, config, dict(zip(NOISE_LABELS, suites)), data)
    (output / "run.json").write_text(json.dumps(store.report["config"], indent=2) + "\n", encoding="utf-8")
    with ThreadingHTTPServer(("127.0.0.1", 0), partial(ComparisonHandler, store=store)) as server:
        server.timeout, server.daemon_threads = .5, True
        url = f"http://127.0.0.1:{server.server_port}/client_wiki/comparison-runner.html"
        scope = "eight centroid files, 1,000 trials each" if args.centroid_only else "57 configurations, 10,000 queries each"
        print(f"Internet measurements: {scope}. Results: {output}\nOpen: {url}", flush=True)
        if not args.no_browser:
            webbrowser.open(url)
        try:
            while not store.done.is_set():
                server.handle_request()
        except KeyboardInterrupt:
            with store.lock:
                store.report["error"] = "Interrupted; completed measurements are retained in measurements.jsonl"
                store._save()
                store.done.set()
            return
    if not store.report["complete"]:
        raise RuntimeError(store.report.get("error", "The browser did not complete the comparison matrix"))
    from figures import export_figures

    export_figures(store.report, output / "figures")
    print(f"Saved results.json, centroid-summary.csv and paper figures in {output}", flush=True)


if __name__ == "__main__":
    main()
