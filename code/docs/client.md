# Browser client

The client has three pages:

- [`index.html`](../client_wiki/index.html): semantic search over prepared Wikipedia passages.
- [`benchmark.html`](../client_wiki/benchmark.html): interactive clustering lab.
- [`comparison-runner.html`](../client_wiki/comparison-runner.html): progress and results for the Python comparison runner.

The three pages share the styles and self-hosted Inter fonts in `assets/`.

Scripts and styles live in `code/client_wiki/assets/`:

- `search.mjs` and `embedding-worker.mjs`: search interface and query embedding in a worker.
- `search-index.mjs`: benchmark-compatible retrieval and passage metadata caching.
- `retrieval.mjs`: shared download, centroid decoding, routing and ranking.
- `benchmark.mjs`: experiment plans, query suites and measurement orchestration.
- `lab.mjs` and `comparisons.mjs`: interactive lab and automatic experiment controllers.
- `styles.css` and `fonts/`: shared styles and local Inter font files for all three pages.

From the `WWW` directory, start the local server:

```sh
python3 code/src/run_comparisons.py --serve
```

Open [the search page](http://127.0.0.1:8080/client_wiki/) or [the clustering lab](http://127.0.0.1:8080/client_wiki/benchmark.html). The server serves the `code` directory, including the shared assets and local data. Opening an HTML file directly cannot run the module worker.

The search page offers the paper's 8,192, 16,384 and 32,768 cluster splits of all 35,167,920 Wikipedia passages. It defaults to 32,768 clusters, SQ6, four probes and ten results. Local search reads the published manifests under `code/data/demo/pages/root/data/search-full-brotli/`; the packaged client reads the same manifests at `data/search-full-brotli/`. Both use the existing HTTPS vector and text deployments, so local search does not need a second metadata dataset.

On opening, index and centroid loading run in parallel with the Transformers.js import and model/runtime prefetch. Prefetch fills the library's existing browser cache, so the pipeline reuses those responses; if storage or prefetch is unavailable, normal pipeline loading handles the downloads. The page accepts query text while these load. Transformers.js is pinned to 4.3.1, with the matching ONNX Runtime 1.31.0-dev.20260914-8d85527a0 CPU-only WASM build. The model revision is pinned to `751bff37182d3f1213fa05d7196b954e230abad9`. The first visit needs Internet access for the library and approximately 38 MB of model/runtime assets. Embedding and ranking run in the browser worker; queries are not sent to a search server.

Embedding inference uses ONNX Runtime's automatic WebAssembly thread count: up to four threads, chosen for the device. The local server and packaged `/client_wiki/*` responses send `Cross-Origin-Opener-Policy: same-origin` and `Cross-Origin-Embedder-Policy: require-corp` so the page and worker are cross-origin isolated. ONNX falls back to one thread on browsers or hosts that cannot support shared WebAssembly memory. External scripts and model/index fetches must allow CORS. The paper's encoder and startup timings use the four-thread rerun; its retrieval measurements are unchanged.

The manifest loads only during initialization, when the page opens or the selected split changes. Search messages contain the query and settings and use the index already held by the worker. A search without a ready index fails instead of loading a manifest implicitly.

All centroid decoders target the current 384-dimensional model. SQ8, SQ6 and INT8 decode groups of four values, SQ4 decodes pairs, and PQ4 expands eight-dimensional blocks. Float16 uses cached exponent scales.

Search uses the benchmark's centroid decoders, nearest-centroid routing, normalized float16 passage vectors and squared L2 distances. The default downloads four selected clusters and returns ten passages. Search options select the cluster split, centroid format, number of clusters to probe and result count. Increasing probes scans more passages and downloads more vectors. The model embeds human-written queries; benchmark timings use prepared query vectors and exclude embedding.

The tokenizer is capped at 256 tokens, including special tokens, through `extractor.tokenizer.config.model_max_length`. Transformers.js 4.3.1's feature-extraction pipeline ignores a per-call `max_length` option, and the tokenizer's `model_max_length` getter is read-only.

Results show embedding inference, centroid setup, vector retrieval, result-text loading and compressed vector payload size separately. Inference includes tokenization, the model forward pass, mean pooling and normalization; model download/loading and time waiting for a canceled inference are excluded. Centroid setup includes download and decoding on a cache miss. Switching centroid formats loads that format on demand and shows its setup time. Vector retrieval uses the benchmark's timer, covering routing, selected cluster loads, decoding/normalization and ranking. Result-text time covers the metadata phase, including memory-cache lookups. The model and decoded centroids are retained. Live centroid, vector and text payloads use `force-cache`; manifests remain `no-store`. Payload URLs identify immutable data: publishing changed payloads requires new URLs. Benchmark centroid loads and timed searches still bypass the browser cache.

Once vectors are ranked, the client fetches only the text batches containing the returned IDs, with `batch = floor(id / 500)`. Up to 16 files download in parallel. The metadata format is `binary-http-v1`, served with `Content-Encoding: br`, so the browser decompresses Brotli automatically. Its 71 `brotli-batch-*` deployments retain the existing vector indexes.

The single local Brotli bundle is `code/data/demo/metadata/`. The upload package under `code/data/demo/pages/` shares its payloads through hard links. Refresh frontend files with `copy_client` on `code/data/demo/pages/root`; it preserves manifests and routes and refreshes the root headers.

Decoded payloads start with ASCII `WMD1`, then little-endian uint32 values for the first passage ID and record count. Each record stores two uint32 UTF-8 byte lengths followed by title and text bytes. IDs are implicit from first ID plus record position. The decoder checks batch identity, string bounds, UTF-8 validity and exact length. Metadata must cover the complete index.

Source article titles can contain HTML entities. The interface decodes them once for display and Wikipedia links, keeping literal markup as plain text. Metadata files retain the source strings.

Decoded batches use an estimated 8 MiB least-recently-used memory cache, cleared on reload. Immutable payloads may remain in the browser HTTP cache; decoded metadata is not stored in IndexedDB or localStorage. Metadata failures retain vector matches and show a text-unavailable message, allowing a later search to retry.

The [result-text benchmark](client-measurements.md#result-text) calls the same `loadResultText` function as interactive search. It samples 10,000 distinct passage IDs, warms each required text batch once, then measures 1,000 groups of ten with browser and decoded-text caches bypassed. It records total text-stage latency and per-file download/JavaScript decoding measurements without running the embedding model or vector search.

The [compatibility check](client-measurements.md#embedding-compatibility) re-embeds 100 saved reference passages and compares them with their stored vectors.

The completed [4.3.1 timing run](client-measurements.md#embedding-inference) supplies the paper's component timings: 100 startup trials and 1,000 inference calls, with four threads in regular Chromium 155. Its inputs contain all 100 sampled passage texts, so reruns do not need a separate sample index. The [elapsed-startup run](client-measurements.md#worker-startup) measures model/centroid readiness in 100 fresh contexts, preserving overlapping downloads. It serves controls and the full-corpus 16k/SQ6 manifest locally and fetches assets from their HTTPS hosts.

Full text regeneration and publishing are documented in the [publishing guide](publish-demo.md). The existing Brotli bundle is complete; regeneration is unnecessary for frontend changes. The exporter preserves uploaded HTTPS descriptors, validates source provenance and streams 500-passage gzip files into a temporary staging bundle for conversion. It refuses existing output directories. The gzip staging copy can be removed after the complete Brotli bundle is verified.

Demo preparation uses the complete Wikipedia source in the uploaded indexes' frozen shard order. It checks dataset/revision, shard offsets and counts before streaming Parquet text into 500-passage gzip batches. All three splits share those batches and retain their existing HTTPS vector URLs. Source dataset IDs can repeat; browser result IDs are corpus ordinals.

For public hosting, [publish the prepared client and 1,000-file text deployments](publish-demo.md). The browser uses the fixed 71-host metadata map and 500-passage Brotli files.

The page has fixed split choices and no manifest URL field. To deploy bundles at different paths, update the fixed manifest URLs in [`search.mjs`](../client_wiki/assets/search.mjs) and serve their referenced payloads over HTTP. The page searches the full 35,167,920-passage source snapshot and exposes 1–100 probes and all six centroid codecs. The prepared public package serves the client and manifests on the main deployment and text files on separate static asset deployments. The current vector/centroid uploads contain no passage text, so keep their existing deployments intact. A public demo deployment also needs the generated metadata bundle; the existing cluster uploads alone cannot display passage text.

The public search package includes the clustering lab at `client_wiki/benchmark.html`. A Low/Medium/High selector chooses one of the three paper noise suites. Each configuration runs 10,000 queries and reports source recovery@1. Cluster counts use 1, 4 and 8 probes; the probe sweep uses 1, 2, 3, 4, 6, 8 and 10 at 16,384 clusters; all six centroid formats use 16,384 clusters and four probes. Shared latency configurations run once. The lab retains the displayed aggregates; the automatic runner saves raw paper measurements.

The lab and automatic comparisons use prepared vector clusters and exclude query embedding from their timings. For paper experiments, the browser downloads centroids and selected clusters directly from remote HTTPS assets. The local server serves controls and saves measurements. See the [comparison guide](benchmarks.md) for the full-data command, Cloudflare deployment and saved figures.
