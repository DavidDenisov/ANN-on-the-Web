# Wikipedia dataset preparation

Downloads the existing MiniLM embeddings from [Hugging Face](https://huggingface.co/datasets/maloyan/wikipedia-22-12-en-embeddings-all-MiniLM-L6-v2). No model inference or image scraping is needed. This stage downloads and converts the Wikipedia data.

From the `WWW` directory, install the dependencies, sign in locally, and start the full download:

```sh
python3 -m pip install -r code/requirements.txt
python3 -c "from huggingface_hub import login; login()"
python3 code/src/download_dataset.py
```

The login command prompts for a Hugging Face [read token](https://huggingface.co/settings/tokens). Skip it if you are already logged in. The downloader automatically uses that saved login, or `HF_TOKEN` if you have set it. Authenticated requests use your account's [rate limits](https://huggingface.co/docs/hub/rate-limits), rather than the anonymous limits; this does not guarantee a particular download speed. No token is stored in this project or printed by the script.

The default destination is `code/data/wikipedia`, regardless of the working directory. To choose another disk:

```sh
python3 code/src/download_dataset.py --output /path/to/wikipedia
```

Progress bars show completed shards, the current download in bytes, and passages converted to float16. The full source download is approximately 85.9 GB across 145 shards. The final vector matrices alone occupy approximately 27.0 GB; passage text and metadata need additional space. One source shard is kept temporarily during conversion.

## Saved data

Each original shard becomes a directory containing:

- `vectors.npy`: a memory-mappable `(passages, 384)` matrix of little-endian float16 values. Conversion rounds the existing values; it does not renormalize or re-encode them.
- `passages.parquet`: Zstandard-compressed `id`, `wiki_id`, `paragraph_id`, `title`, and `text`, in exactly the same row order as the vectors. Unused `views`, `langs`, the derivable `url`, and the original float32 `emb` column are omitted.
- `info.json`: source filename, row count, dimensions, and dtype.

`manifest.json` pins the Hugging Face commit and lists the source shards in order. Source IDs are preserved, without assuming that they are globally unique or consecutive. A passage can always be identified by its shard and row position.

```python
import numpy as np
import pyarrow.parquet as pq

vectors = np.load("code/data/wikipedia/<shard>/vectors.npy", mmap_mode="r")
passages = pq.read_table("code/data/wikipedia/<shard>/passages.parquet")
# vectors[i] corresponds to passages.slice(i, 1).
```

Converted shards are validated and published atomically before their downloaded Parquet and local download metadata are deleted. Only files inside this script's `.downloads` directory are cleaned up; unrelated files and existing Hugging Face caches are untouched. A failed conversion retains its source download for retry. Run the same command again to resume: completed shards are checked and skipped, and unfinished downloads use Hugging Face's resume support. Run only one copy per output directory.

The [paper comparison workflow](benchmarks.md#run-the-paper-comparisons-over-the-internet) uploads every prepared cluster count and centroid format to free Cloudflare Pages, then measures downloads directly from the browser over the Internet. After one-time Cloudflare login:

```sh
wrangler login
python3 code/src/run_comparisons.py --cloudflare-project YOUR_PAGES_PROJECT --output code/data/benchmark/run-new
```

The runner requires every downloader shard and indexes the entire corpus. Real-data clustering and assignment require CUDA-enabled GPU Faiss; follow the [GPU setup instructions](benchmarks.md#run-the-paper-comparisons-over-the-internet) before preparing indexes. The training sample is 2,097,152 vectors; it does not limit the indexed corpus.

On a fresh run, the runner builds and uploads one cluster count at a time, then clears that count's generated local cluster files and upload batches after saving successful upload progress and routing metadata. Batches of 1,000 binary files use count-specific branches such as `c8192-batch-0`, keeping every count available on the same Pages project. It publishes the combined root routing map after all counts upload, then saves JSON/CSV measurements, LaTeX tables and the probe graph PDF. Each cluster stays in one gzip file; packaging rejects any asset above Pages' 25 MiB limit. A content-based bundle identifier is checked before measurements. The local Python server serves the controls and saves results; it does not proxy timed downloads.

Each `c<clusters>/` directory contains the entire float16 corpus partitioned into cluster files. All six centroid formats reuse those files. The CDN stores approximately 27.0 GB of vector payload before gzip per count, or approximately 81 GB across the three paper counts. Sequential upload keeps only the active count's generated cluster payload locally; source data and temporary preparation/upload files still need space. `--prepare-only` retains all counts locally instead. The [benchmark guide](benchmarks.md) gives the separate prepare/package/deploy commands, common project URL and free hosting limits. Runs reject sampled or incomplete preparations.

The comparisons use the full corpus, 10,000 source queries per noise level, and source recovery@1. They vary cluster count, probe count and centroid format at noise levels `0.05`, `0.15` and `0.4`. See the [benchmark guide](benchmarks.md) for the matrix, warm-up, cache policy and output files. Downloaded dataset files remain untouched; text-file packaging is described in the [publishing guide](publish-demo.md).

