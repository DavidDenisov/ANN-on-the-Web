#!/usr/bin/env python3
"""Run the remaining 32k measurements without replacing the saved paper results."""

import argparse
import csv
import hashlib
import json
import math
import shutil
import statistics
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE / "src"))
from run_comparisons import configure_network, load_prepared_data


def read_inputs():
    data = CODE / "data/benchmark"
    prepared = data / "prepared"
    suites = load_prepared_data(prepared)
    config = {"suites": [{"label": label} for label in ("small", "medium", "large")]}
    configure_network(config, suites, data / "pages/hosting.json", prepared)
    fixture = suites[2]
    baseline_path = data / "retrieval/summary.csv"
    with baseline_path.open(newline="") as stream:
        baselines = [r for r in csv.DictReader(stream) if r["clusters"] == "32768"
                     and r["noise"] == "large" and r["codec"] == "f16"
                     and r["accuracy_only"] == "False"]
    assert sorted(int(r["probes"]) for r in baselines) == [1, 4, 8]
    assert all(r["fixture_id"] == fixture["id"] and r["queries"] == "10000"
               and r["k"] == "1" and r["cluster_fetch_cache"] == "no-store" for r in baselines)
    setup_path = data / "centroid-setup/results.json"
    setup_report = json.loads(setup_path.read_text())
    assert setup_report["complete"] and setup_report["config"]["fixture_ids"] == config["fixture_ids"]
    setup, = [s for s in setup_report["centroid_setups"] if s["clusters"] == 32768 and s["codec"] == "f16"]
    assert setup["centroid_setup_repeats"] == len(setup["centroidTrials"]) == 1000
    startup_index = "data/demo/pages/root/data/search-full-brotli/c32768/manifest.json"
    startup = json.loads((CODE / startup_index).read_text())
    hosted_index = json.loads((data / "pages/root/c32768/manifest.json").read_text())
    assert startup["clusters"] == 32768 and startup["count"] == fixture["count"]
    assert startup["centroids"]["sq6"] == hosted_index["centroids"]["sq6"]
    sources = {}
    for path in (baseline_path, setup_path, prepared / "noise-large/suite.json",
                 prepared / "c32768/manifest.json", CODE / startup_index,
                 CODE / "client_wiki/assets/benchmark.mjs", CODE / "client_wiki/assets/retrieval.mjs", Path(__file__),
                 CODE / "benchmarks/remaining-32k.cjs", CODE / "benchmarks/browser.cjs",
                 CODE / "benchmarks/noise-sensitivity.py", CODE / "benchmarks/startup.cjs"):
        sources[path.relative_to(CODE).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    inputs = {"fixture": fixture, "suite_url": fixture["url"], "f16_setup": setup,
              "hosting_manifest_url": config["hosting_manifest_url"],
              "hosting_bundle_id": config["hosting_bundle_id"], "sources": sources,
              "network": {"networkMode": "internet", "assetsBaseUrl": config["assets_base_url"],
                          "assetsBaseUrls": config["asset_base_urls"], "cache": "no-store"}}
    return inputs, baselines, startup_index


def write_summaries(output, inputs, baselines):
    from figures import _centroid_samples, _row

    report = json.loads((output / "retrieval/results.json").read_text())
    assert report["complete"] and len(report["measurements"]) == 9
    rows = []
    for measurement in report["measurements"]:
        result = json.loads((output / "retrieval" / measurement["file"]).read_text())
        _row(result)
        assert result["fixture_id"] == inputs["fixture"]["id"] and result["queries"] == 10000
        assert result["clusters"] == 32768
        times = sorted(t["queryMs"] for t in result["trials"]) if not result["accuracy_only"] else []
        quartiles = statistics.quantiles(times, n=4, method="inclusive") if times else ["", "", ""]
        rows.append({"noise": "large", "noise_level": 0.4, "clusters": 32768,
                     "probes": result["probes"], "codec": result["codec"], "queries": 10000,
                     "accuracy_only": result["accuracy_only"], "source_recovery": result["recall"],
                     "query_mean_ms": statistics.mean(times) if times else "",
                     "query_p95_ms": times[math.ceil(len(times) * .95) - 1] if times else "",
                     "query_median_ms": quartiles[1], "query_p25_ms": quartiles[0], "query_p75_ms": quartiles[2],
                     "bytes_per_query": result["bytesPerQuery"], "origin": "retrieval/" + measurement["file"]})
    fields = list(rows[0])
    for row in baselines:
        rows.append({**{key: row.get(key, "") for key in fields},
                     "origin": "data/benchmark/retrieval/results.json (retained run)"})
    with (output / "summary.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    centroid_rows = []
    for codec in ("f16", "sq8", "sq6", "sq4", "int8", "pq4"):
        setup = json.loads((output / f"retrieval/centroids/{codec}.json").read_text())
        _centroid_samples(setup)
        assert len(setup["centroidTrials"]) == 1000
        row = {"clusters": 32768, "codec": codec, "samples": 1000,
               "centroid_bytes": setup["centroidTrials"][0]["bytes"]}
        for label, key in (("download", "downloadMs"), ("decode", "decodeMs")):
            values = sorted(t[key] for t in setup["centroidTrials"])
            row.update({f"{label}_mean_ms": statistics.mean(values), f"{label}_p95_ms": values[949]})
        row["origin"] = "data/benchmark/centroid-setup (retained run)" if codec == "f16" else f"retrieval/centroids/{codec}.json"
        centroid_rows.append(row)
    with (output / "centroid-summary.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(centroid_rows[0]))
        writer.writeheader()
        writer.writerows(centroid_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new run directory")
    parser.add_argument("--check", action="store_true", help="validate inputs and print the plan without downloads or measurements")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}; choose a new run directory")
    node = shutil.which("node")
    if not node:
        raise RuntimeError("Node.js is required; use the same Node/Playwright environment as the other benchmarks")
    inputs, baselines, startup_index = read_inputs()
    subprocess.run([node, str(CODE / "benchmarks/remaining-32k.cjs"), "--check"],
                   input=json.dumps(inputs), text=True, check=True)
    if args.check:
        print("Also scheduled: 1,000-source 32k routing diagnostic and 100 fresh-context 32k/SQ6 startups.")
        return
    output.mkdir(parents=True)
    retrieval = output / "retrieval"
    retrieval.mkdir()
    (retrieval / "inputs.json").write_text(json.dumps(inputs, indent=2) + "\n")
    with (output / "reused-baselines.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(baselines[0]))
        writer.writeheader()
        writer.writerows(baselines)
    subprocess.run([sys.executable, str(CODE / "benchmarks/noise-sensitivity.py"),
                    str(output / "noise"), "--clusters", "32768"], check=True)
    subprocess.run([node, str(CODE / "benchmarks/remaining-32k.cjs"), str(retrieval)], check=True)
    write_summaries(output, inputs, baselines)
    startup = output / "startup"
    startup.mkdir()
    (startup / "inputs.json").write_text(json.dumps({"startup_trials": 100, "clusters": 32768,
                                                    "index_path": startup_index, "codec": "sq6"}, indent=2) + "\n")
    subprocess.run([node, str(CODE / "benchmarks/startup.cjs"), str(startup)], check=True)
    (output / "complete.json").write_text(json.dumps({"clusters": 32768, "complete": True,
                                                      "new_query_configurations": 9, "reused_query_configurations": 3}, indent=2) + "\n")
    print(f"Completed 32k supplement: {output}. Existing measurements and paper files were not changed.")


if __name__ == "__main__":
    main()
