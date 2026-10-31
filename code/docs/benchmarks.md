# Clustering and browser benchmarks

## Run the paper comparisons over the Internet

The integrated browser comparison runner uses Python and an active browser tab. The recorded paper measurements used regular Chromium 155. Standalone browser/model and GPU measurement runners have been removed.

The Python runner serves the local controls and saves results. The browser fetches centroid and cluster files directly from public HTTPS hosts. It never proxies timed downloads through Python.

Reuse the existing full-corpus preparation and upload:

```sh
python3 code/src/run_comparisons.py \
  --prepared-data code/data/benchmark/prepared \
  --hosting-manifest code/data/benchmark/pages/hosting.json \
  --output code/data/benchmark/run-new
```

Choose a new output directory for every run. Keep the browser tab active; the server stops when the run finishes. `--no-browser` prints the URL without opening it. The hosting manifest supplies the asset roots and bundle identity; the runner checks them against the prepared suites and index hashes before measuring.

The defaults use all 35,167,920 passages, 10,000 query sources per noise level, and source recovery@1:

- Cluster counts: 8,192 at 2, 3, 4 and 5 probes; 16,384 and 32,768 at 1, 4 and 8 probes.
- Probe sweep: 1, 2, 3, 4, 6, 8 and 10 probes at 16,384 clusters.
- Centroid formats: FP16, SQ8, SQ6, SQ4, symmetric INT8 and PQ4 at 16,384 clusters and four probes, measuring accuracy without query timing.

The revised 8k sweep is configured for future runs only; its new measurements have not been collected. Existing summaries retain their recorded settings. New reports record `cluster_probe_counts_8k`; the exporter continues to accept historical reports without that field.

Shared latency settings run once. The fixed matrix contains 42 latency configurations and 18 accuracy checks: 600,000 searches in total. The corpus, query count, probe sweeps, noise levels, formats and source recovery@1 are fixed by the runner.

Queries are normalized source-passage vectors plus Gaussian noise of L2 length 0.05, 0.15 or 0.4, then normalized again. All levels share source passages, random vectors and trained indexes. The original passage remains indexed. Source recovery@1 is the fraction of queries returning their source passage first; this tests robustness to perturbations rather than relevance to natural-language questions.

## Prepare and upload new indexes

Preparation needs a complete [downloaded dataset](dataset.md), an NVIDIA GPU and GPU-enabled Faiss. Follow the [official Faiss installation instructions](https://github.com/facebookresearch/faiss/blob/main/INSTALL.md), then install the remaining dependencies:

```sh
conda install -c pytorch -c nvidia -c conda-forge faiss-gpu
python3 -m pip install -r code/requirements.txt
wrangler login
python3 code/src/run_comparisons.py \
  --cloudflare-project YOUR_PAGES_PROJECT \
  --output code/data/benchmark/run-new
```

This builds one cluster count, uploads it, saves routing metadata and upload receipts, then removes that count's generated cluster payloads before building the next. It retains the suites and centroid/index manifests. Once every count is uploaded, it publishes the combined root routing map and starts measurements. The runner waits for verified HTTPS on new asset domains before opening the experiment.

The full corpus contains approximately 27 GB of vector payload before gzip per cluster count, plus passage IDs and centroids. Sequential upload limits local generated payloads to the active count; source data and temporary files still need disk space. Training uses 2,097,152 vectors, 20 GPU k-means iterations and assignment batches of 65,536. Training and assignment use normalized values decoded from stored float16 vectors; assignment uses the stored float16 centroids. Each preparation generates and records one seed, shared by all indexes and noise levels.

Passage IDs are full-corpus row ordinals. `truth.u32.bin` records the 10,000 sampled query IDs, which are also used to read their source vectors.

To prepare all counts locally without uploading or measuring:

```sh
python3 code/src/run_comparisons.py \
  --input code/data/wikipedia \
  --output code/data/benchmark/preparation-new --prepare-only
```

This retains all generated counts, requiring space for approximately 81 GB before compression. Package and deploy that preparation separately:

```sh
python3 code/src/cloudflare.py prepare \
  --prepared-data code/data/benchmark/preparation-new/data \
  --output code/data/benchmark/pages-new --project-name YOUR_PAGES_PROJECT
python3 code/src/cloudflare.py deploy --deployment code/data/benchmark/pages-new
```

Create the Pages project first if needed with `wrangler pages project create YOUR_PAGES_PROJECT --production-branch main`. Packaging uses up to 1,000 binary files per preview branch and publishes suites/manifests on `main`. Cloudflare Pages limits each deployment to 20,000 files and each file to 25 MiB; oversize assets are rejected. Each cluster remains one gzip file. Payloads use hard links where possible. Credentials and measurements are not published.

`deployment.json` saves a content hash after each successful branch upload. Repeating deployment skips unchanged completed branches and continues interrupted uploads. Use `--force` if another package has replaced the same project's assets. Root publication happens after all binary branches succeed. Never replace live assets while collecting measurements. The streamed workflow removes local cluster payloads after upload; retained manifests alone cannot redeploy them.

Preparation and measurement use the fixed paper settings. To repeat measurements, reuse the prepared data and hosting manifest with a new output directory. Outputs are published atomically and existing destinations are refused; input dataset files stay untouched.

## Timing and caching

Before the full comparison, the runner fetches every cluster and required centroid file once. Bodies are discarded; requests bypass the browser cache. This primes the hosting path but does not guarantee CDN residency. A failed warm-up stops the run.

The runner measures 1,000 download/decode trials for each of eight centroid files: FP16 at all three counts and five additional formats at 16,384. Each query configuration then loads its chosen centroids once outside query timers. Shared setup samples are identified and counted once in aggregates; conflicting copies are rejected. The eight files total about 59.3 MB per cycle, or 59.3 GB across 1,000 trials.

For centroid timings only, reuse the preparation and hosting manifest with `--centroid-only`. Each centroid file is fetched once before its 1,000 timed trials. This skips cluster downloads and retrieval/accuracy queries, saves raw setup trials and exports the two timing tables.

Latency and centroid requests bypass browser caching. Compression accuracy allows cached cluster files, while its centroid preload still bypasses the cache. The live demo has its own caching behavior; see the [client guide](client.md). Configuration order is randomized using the recorded seed. CDN state, network throughput and browser scheduling remain uncontrolled.

Query timing covers centroid routing, cluster download, gzip decompression, passage decoding/normalization and ranking. It excludes query embedding, model loading, centroid setup and result-text loading. Up to 16 clusters download concurrently, each decoded and scored as it arrives. Cancellation or a failed cluster aborts the remaining requests. Transient network/body failures, HTTP 408/429 and server errors receive at most one retry after 0.5 seconds, included in elapsed timing.

The generated headers allow CORS and Resource Timing. Gzip files are served without `Content-Encoding` and decompressed explicitly in the browser. New packages use `Cache-Control: public, max-age=0, s-maxage=604800, no-transform`; existing deployments retain their own headers until redeployed. Complete response bodies are timed. Payload bytes exclude HTTP headers, and recorded retries remain inside elapsed time.

The interactive lab uses the same fixed sweeps and source recovery@1, with a choice of three noise levels. It defaults to 10 queries per configuration, with a 1–10,000 control and cancellation. For a short run, it shuffles the 10,000 query indices once, takes the selected number without repetition, and shares that subset across configurations; full runs preserve the original query order. Result rows show sample counts. Short samples illustrate behavior and do not replace full measurements. A visible warning covers downloads and runtime: even short checks can transfer hundreds of MB; full checks can take hours and transfer hundreds of GB. The complete query suite and index manifests still load at initialization. The automatic comparison runner continues to use all 10,000 queries. The lab shares overlapping configurations and retains completed rows when canceled. Its centroid timing is a single preload per configuration, rather than the automatic runner's shared 1,000 samples. Local asset timings do not predict Internet performance.

## Saved results and paper exports

The original paper runs used these directories under `code/data/benchmark/`; their raw inputs and trials are excluded from this checkout:

- `retrieval/`: completed 10,000-query retrieval/accuracy run, including its original centroid samples.
- `centroid-setup/`: completed 1,000-trial-per-file rerun for Tables 2 and 4.
- `encoder/inference/`: Transformers.js 4.3.1 component timings and 1,000 inference calls; all sampled texts are in `inputs.json`.
- `encoder/startup/`: 100 fresh-context elapsed startups, including overlapping loads.
- `encoder/compatibility/`: reference inputs for the reported re-embedding check.
- `result-text/`: 1,000 varied ten-result loads over 10,000 sampled passages.

The [client measurement notes](client-measurements.md) preserve the historical inference, startup, compatibility, text-loading, and complete-query protocols. These use regular Chromium 155; inference uses four threads in the paper. The text benchmark warms each required text batch once before timing and bypasses browser and decoded-text caches.

Automatic runs save `run.json`, `results.json`, `centroid-summary.csv`, `summary.csv` and `figures/`. Centroid-only runs omit query summaries. `measurements.jsonl` journals each accepted query configuration and is removed after the completed report is saved. Interrupted runs retain their journal; use the same preparation and a new output directory to retry. Results validate settings, source targets, file sizes, asset URLs and identities before accepting measurements.

The exporter produces only the paper's LaTeX/CSV tables, `settings.tex` and `probes-c16384.pdf`. Redraw a completed report without rerunning searches:

```sh
python3 code/src/figures.py code/data/benchmark/retrieval/results.json \
  --output /tmp/www-retrieval-figures
python3 code/src/figures.py code/data/benchmark/centroid-setup/results.json \
  --output /tmp/www-centroid-figures
```

These exports cover the original full comparison matrix. The active paper uses the committed compact summaries in `paper/results/` and `paper/figures/probe-32k-summary.csv`; see [result provenance](../../paper/results/README.md). After collecting new measurements, update the corresponding summaries and run `python3 paper/figures/generate.py`. Refresh inline values in `paper/main.tex` separately, then recompile. The scripts never update the paper automatically.

## Stored formats

Index manifests record source shard offsets, cluster sizes, codec descriptors and training settings. Passage ordinals follow manifest shard order and row order; original dataset IDs need not be unique. Cluster records contain a little-endian uint32 ordinal followed by 384 float16 components. Empty clusters have valid empty gzip files. The record format supports at most 2^32 passages.

Centroids use six gzip-compressed formats with fixed cluster membership:

- FP16 stores the centroid matrix directly.
- SQ8/SQ6/SQ4 store per-dimension float32 minimum and scale arrays, then low-bit-first packed codes; decoding is `minimum + scale * code`.
- Symmetric INT8 follows [I-BERT, Section 3.1, Equations (1)-(2)](https://proceedings.mlr.press/v139/kim21d/kim21d.pdf): per dimension, `alpha = max(max(abs(values)), 1e-8)`, `scale = alpha / 127`, and `q = round(clip(values, -alpha, alpha) / scale)` in `[-127, 127]`, with ties rounded to even. Payloads contain float32 scales followed by signed codes; decoding is `scale * q`. The calibration floor follows the authors' implementation; the symmetric endpoints follow the paper.
- PQ4 uses 16 codewords per eight-component block, with a uint32 block-size header, float32 codebooks and low-nibble-first codes. CPU Faiss trains these small codebooks; corpus clustering and assignment require GPU Faiss.

All formats search the same normalized float16 passage vectors. Centroid parameters and codebooks count toward download size. Inputs are memory-mapped and streamed in batches; use one writer per output directory.
