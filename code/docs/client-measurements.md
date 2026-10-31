# Historical client measurement protocols

These notes describe the completed paper runs. Raw inputs and trials under `code/data/` are excluded from this checkout. Compact results are preserved in `paper/results/`. The standalone live measurement runners have been removed.

## Embedding inference

The October 24, 2026 run used Transformers.js 4.3.1, the q8 all-MiniLM-L6-v2 model at revision `751bff37182d3f1213fa05d7196b954e230abad9`, and the CPU-only ONNX Runtime 1.31.0-dev.20260914-8d85527a0 build. It used four WebAssembly threads in regular Chromium 155.

There were 100 fresh browser contexts for separately timed startup components. The final context ran 1,000 timed inference calls over 100 saved passage texts: 334 at 64 tokens, 333 at 128 tokens and 333 at 256 tokens. Each length received three warm-up calls. Actual model tensor lengths and normalized 384-dimensional outputs were validated.

The run read approximately 3.8 GB of model/runtime asset bodies across the startup trials, plus library downloads. HTTP compression can reduce transferred bytes. Component timings describe separate stages; elapsed client startup, including overlapping loads, was measured separately.

The offline summarizer remains available for saved inputs and results. From the repository root:

```sh
python3 code/benchmarks/summarize_inference.py SAVED_RUN_DIRECTORY
```

It validates the schedule, token lengths, browser/thread identity, payload sizes and finite timings before writing `summary.json`.

## Worker startup

The paper used 100 fresh-context starts with the full 35,167,920-passage manifest, 32,768 clusters and SQ6 centroids. Regular Chromium 155 verified four WebAssembly threads and cross-origin isolation. The clock started immediately before creating the actual client worker and stopped at its `ready` message, when both model and centroids were ready.

Worker imports, manifest loading, model/runtime downloads and initialization, and centroid download/decoding were included. Overlapping loads were preserved. Control-page navigation and query execution were excluded. Controls, worker modules and the manifest were served locally; model/runtime assets and centroids used their configured HTTPS hosts.

Mean elapsed startup was 2.46 s; nearest-rank p95 was 2.53 s. Exact values are in [the saved startup summary](../../paper/results/startup-summary.json). Fresh contexts had empty browser caches; the Chromium process was reused. CDN residency and network conditions were uncontrolled.

## Embedding compatibility

Saved reference passage texts were re-embedded using the pinned q8 model, Transformers.js 4.3.1, four WASM threads and the 256-token limit. Each output was compared with its stored 384-dimensional vector. Mean and 5th-percentile cosine similarity were reported without a similarity cutoff.

The paper uses [the 1,000-pair summary](../../paper/results/compatibility-summary.json). Earlier 100-pair runs used the same protocol.

## Result text

The run sampled 10,000 distinct passage IDs uniformly from the full corpus, then loaded them as 1,000 groups of ten through the production `loadResultText` function in a module worker. Each required text batch was warmed once before timing. Browser and decoded-text caches were bypassed with `cache: 'no-store'`.

Each timed load fetched the necessary 500-passage Brotli files in parallel, decoded their records and attached titles/text. Results in the same batch shared one download. Timing excluded manifest/module loading and DOM rendering. No embedding or vector search ran. Missing text, invalid payloads or payload-length changes stopped collection.

The saved run reported 34.9 ms mean and 54.9 ms p95 per ten results. JavaScript binary/UTF-8 decoding was measured separately; browser Brotli decompression was included in download/elapsed times. Parallel download times cannot be added to estimate elapsed time. The browser context and worker were reused, and CDN conditions were uncontrolled.

## Complete ten-result query

The illustrated query used the actual client worker with 32,768 clusters, SQ6 centroids, four probes and ten results. It performed 100 complete warm-up searches followed by 1,000 timed repetitions. The model and centroids stayed loaded; vector and text requests bypassed browser and decoded-text caches. Every query recomputed its embedding, retrieval and text decoding.

Mean times were 3.0 ms for embedding, 90.7 ms for vector retrieval, 36.3 ms for text loading, and 130.3 ms for the directly measured worker round trip. Round-trip p95 was 150.3 ms. These values describe one fixed query and exclude DOM rendering.

The ten-result set required 2,480,789 bytes of gzipped vectors and 767,752 bytes across ten Brotli text files: 3,248,541 compressed payload bytes, excluding HTTP headers. Text-loader body lengths were measured after automatic Brotli decompression and therefore differ from compressed network sizes. See [the published result notes](../../paper/results/README.md) for provenance.
