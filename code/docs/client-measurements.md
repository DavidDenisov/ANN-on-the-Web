# Client measurement protocols

These notes preserve the protocols of the completed paper runs. Inputs and raw trials under `code/data/` are local artifacts and are not included in Git. Compact results are committed in `paper/results/`. Run commands from the repository root with Playwright installed.

## Embedding inference

This completed Transformers.js 4.3.1 run was measured on 2026-10-24. It uses the CPU-only ONNX Runtime 1.31.0-dev.20260914-8d85527a0 build, 100 fresh browser contexts for separately timed startup components and 1,000 timed inference calls in the final context: 334 at 64 tokens, 333 at 128 tokens and 333 at 256 tokens. Actual elapsed client startup, including overlapping loads, is measured separately in the worker-startup run below.

`inputs.json` contains the 100 sampled passage texts, pinned model revision and shuffled schedule. Each length receives three warm-up calls. Every passage/length pair receives three or four timed calls, using regular Chromium 155 and four WebAssembly threads.

`results.json` preserves all startup and inference trials. During collection, `startup.jsonl` also saves each completed startup immediately; it is retained on interruption and removed after the completed results are saved.

Validate the retained measurements and regenerate `summary.json` from the project root:

```sh
/usr/bin/python3 code/benchmarks/summarize_inference.py code/data/benchmark/encoder/inference
```

To repeat measurements, copy `inputs.json` into a new run directory and pass that directory to `code/benchmarks/inference.cjs`, with Playwright available to Node and a writable `TMPDIR` inside the project. The runner fixes 100 startup trials, 1,000 calls split 334/333/333 across 64/128/256 tokens, the pinned model revision and four threads. The runner serves its own isolated control page and refuses existing measurement files. The summary validates the saved schedule, actual tensor lengths, browser and thread identity, and finite timings. This run read approximately 3.8 GB of model/runtime asset bodies across the 100 startup trials, plus library downloads; HTTP compression can reduce transferred bytes.

## Worker startup

The paper uses 100 fresh-context starts of the Transformers.js 4.3.1 client with the full 35,167,920-passage manifest, 32,768 clusters and SQ6 centroids. It uses regular Chromium 155 and verifies four WebAssembly threads and cross-origin isolation. The earlier 16k run is retained locally under `code/data/benchmark/encoder/startup/`.

Mean elapsed startup is 2.46 s; p95 is 2.53 s. Exact values are in `paper/results/startup-summary.json`, copied from `code/data/benchmark/remaining-32k-new/startup/summary.json`.

The clock starts immediately before constructing the actual client worker and stops when its `ready` message arrives: both the model and selected centroids are ready for search. It includes worker imports, manifest loading, model/runtime downloads and initialization, and centroid download/decoding. The client's overlapping loads are preserved. It excludes control-page navigation and query execution.

Each trial uses a fresh browser context with empty browser caches; the Chromium process is reused. Controls, worker modules and the manifest are served locally. Model/runtime assets and centroids use their configured HTTPS hosts. CDN residency and network conditions are uncontrolled.

`inputs.json` records the trial count and index selection. `results.json` includes raw trials, browser/CPU metadata and SHA-256 hashes of the manifest and worker modules. New runs also hash the shared browser setup. During collection, `startups.jsonl` saves each completed trial; it is retained on interruption and removed after `results.json` and `summary.json` are saved. The summary reports the mean and nearest-rank p95.

To repeat, create a new run directory containing a copy of `inputs.json` specifying `"clusters": 32768` and `"codec": "sq6"`, then run from the project root with Playwright available to Node:

```sh
TMPDIR="$PWD/code/data/benchmark/encoder/inference/tmp" \
node code/benchmarks/startup.cjs NEW_RUN_DIRECTORY
```

The runner serves its own isolated control page and refuses to overwrite measurements. The component timings and length-controlled inference calls remain under `code/data/benchmark/encoder/inference/`; they are separate measurements and are not added to estimate elapsed startup.

## Embedding compatibility

This is the re-embedding check described in the paper. `inputs.json` contains 100 reference passage texts and their stored 384-dimensional vectors, with source provenance. No separate sample index is needed.

The runner uses Transformers.js 4.3.1, the pinned q8 all-MiniLM-L6-v2 model, the CPU-only WASM runtime, four threads and the 256-token limit. It saves one cosine similarity per passage and reports the mean and 5th percentile without a similarity cutoff. The paper uses the later [1,000-pair summary](../../paper/results/compatibility-summary.json), whose original inputs and trials are retained under `code/data/benchmark/encoder/compatibility-1k/`.

To rerun from the project root with Playwright available to Node:

```sh
export TMPDIR="$PWD/code/data/benchmark/encoder/inference/tmp"
mkdir code/data/benchmark/compatibility-new
cp code/data/benchmark/encoder/compatibility/inputs.json code/data/benchmark/compatibility-new/
node code/benchmarks/compatibility.cjs code/data/benchmark/compatibility-new
```

Use a fresh output directory. This folder retains the original inputs and the October 26, 2026 100-pair rerun. The 1,000-pair inputs, measurements and reported summary are preserved separately.

## Result text

Run from the project root with Playwright available to Node:

```sh
export TMPDIR="$PWD/code/data/benchmark/encoder/inference/tmp"
mkdir code/data/benchmark/result-text-new
cp code/data/benchmark/result-text/inputs.json code/data/benchmark/result-text-new/
node code/benchmarks/result-text.cjs code/data/benchmark/result-text-new
```

The runner uses regular Chromium 155 and a module worker. It checks the browser version and refuses other versions.

The runner samples 10,000 distinct passage IDs uniformly from the full corpus at startup. It saves the exact sample in `results.json`. Each run uses a new random sample; no embedding model or vector search is run.

Before timing, it derives the set of required text batches and warms each file exactly once, using one sampled passage per file. Warm-up loads use groups of up to ten files. The full warm-up pass finishes before any timed load starts.

The 10,000 sampled IDs are then loaded in order as 1,000 groups of ten. It calls the production client's `loadResultText` function, fetching the required 500-passage Brotli files in parallel, decoding their records, and attaching titles and text. Two results in the same batch share one download. The timer ends after all ten results are ready and excludes manifest/module loading and DOM rendering. Text storage is shared across all cluster counts.

Both warm-up and timed loads bypass the browser HTTP cache and decoded-text cache with `cache: 'no-store'`. The browser context and worker are reused; CDN state and network throughput remain uncontrolled. The run makes at most 20,000 text-file requests before retries.

During collection, `trials.jsonl` saves each completed load immediately, tagged `warmup` or `timed`. It is retained on interruption and removed after `results.json` and `summary.json` are saved. `results.json` separates both phases and records sampled IDs, per-load result IDs, input settings, browser/CPU identity, public URLs, and SHA-256 hashes of the inputs, manifest, runner and client modules. New runs also hash the shared browser setup. The runner checks that each required batch is warmed once, every timed file was warmed, and payload lengths match the warm-up. Missing text, invalid payloads or unexpected files stop the run.

`summary.json` uses only the 1,000 timed groups for mean, nearest-rank p95, minimum and maximum latencies. It reports the number of warmed files, the range of files per timed group, and mean decoded/compressed bytes per group. Its `decode` timing sums JavaScript binary/UTF-8 decoding; browser Brotli decompression is included in download/elapsed timings. Parallel download times must not be added to estimate elapsed time. Encoded sizes are null when the browser does not expose them.

This folder contains the completed varied run used by the paper: 34.9 ms mean and 54.9 ms p95 per ten results. Existing measurement files are never overwritten. Reruns use a fresh directory with a copy of `inputs.json`, as above. The script does not modify the paper or deploy the client.

## Complete ten-result query

This rerun uses the actual client worker with the displayed query, 32,768 clusters, SQ6 centroids, four probes and ten results. It performs 100 complete warm-up searches, then 1,000 timed repetitions. The model and decoded centroids stay loaded. Vector and result-text requests use `cache: no-store`; the decoded-text cache is bypassed. CDN state remains uncontrolled.

The benchmark serves a test copy of `search-index.mjs` that changes only query cache policy and exposes existing download records. Production client files are unchanged. Every timed record contains four vector downloads and one text download per required batch. Computation, decompression and decoding still run for each query; no computed query embedding or result list is reused.

Mean times are 3.0 ms for embedding, 90.7 ms for vector retrieval, 36.3 ms for text loading, and 130.3 ms for the complete worker round trip. The total is measured directly, includes the three stages and message handling, and excludes DOM rendering. Its 95th percentile is 150.3 ms. All values describe this fixed query, not a new estimate for arbitrary questions or the 10,000-query perturbation benchmark.

The fixed ten-result set requires 2,480,789 bytes of gzipped vectors and 767,752 bytes across ten Brotli text files: 3,248,541 compressed payload bytes (3.25 decimal MB), excluding HTTP headers. Vector bytes come from the saved trial records; text bytes are summed from the corresponding stored `.bin.br` files. The text loader's recorded `bytes` field counts the body after the browser's automatic Brotli decompression, so it is not the compressed network payload. These payload totals are preserved in the [published result notes](../../paper/results/README.md); figure generation no longer reads the full trial log or text files.

`inputs.json` records settings. `results.json` preserves warm-up and timed records, download URLs/bytes/times, CPU/browser identity, source hashes and the first result. `summary.json` contains the aggregate values and displayed passage. Earlier benchmark reports are unchanged.

To rerun, create a fresh output directory, copy `inputs.json` into it, then execute from the project root with Playwright available:

    TMPDIR="$PWD/code/data/benchmark/encoder/inference/tmp" node code/benchmarks/live-query.cjs NEW_RUN_DIRECTORY

The runner refuses to overwrite measurements and preserves its trial journal if interrupted.
