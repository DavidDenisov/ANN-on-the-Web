# Publish the demo

The current project is **wiki-ann-demo-20261022**:

- [Search](https://wiki-ann-demo-20261022.pages.dev/client_wiki/)
- [Clustering lab](https://wiki-ann-demo-20261022.pages.dev/client_wiki/benchmark.html)

Use `code/data/demo/pages/`. It contains the full 35,167,920-passage metadata bundle: 70,336 Brotli files with 500 passages each, except the final file of 420. They are split across 71 `brotli-batch-*` deployments with up to 1,000 text files per deployment. Existing vector and centroid URLs are retained.

## Frontend update

Refresh the prepared frontend from the source, then upload only `root/`:

```sh
# From the repository root:
PYTHONPATH=code/src python3 - <<'PYCODE'
from pathlib import Path
from prepare_demo_pages import copy_client
copy_client(Path('code/data/demo/pages/root'))
PYCODE
wrangler pages deploy code/data/demo/pages/root \
  --project-name wiki-ann-demo-20261022 --branch main
```

This copies the current search and lab assets and uses the existing Brotli manifests. It refreshes the root headers, including the isolation headers needed for multithreaded inference, and preserves the data manifests, routing and text deployments. No metadata conversion or vector upload is needed. Refresh the browser after publication. If authentication has expired, run `wrangler login` before uploading.

## Package layout

- `DATA-NOTICE.txt`: Wikipedia text attribution, CC BY-SA 3.0 licence link, source-article URL instructions, and extraction/packaging changes. Retain it with redistributed text.
- `root/`: client pages, shared modules/styles/fonts, three search manifests, `_headers`, `_redirects` and `routing.json`. Both pages link to `client_wiki/DATA-NOTICE.txt`.
- `batches/brotli-batch-0/` through `batches/brotli-batch-70/`: compressed metadata, `DATA-NOTICE.txt` and `_headers` for each deployment. Brotli responses link to the licence and adjacent notice through their HTTP `Link` header.
- `hosting.json`: batch paths and public URLs.
- `deployment.json`: completed upload hashes, used by the resumable uploader.
- `publication.json`: the last recorded publication.

Upload the frontend root, not the whole repository or parent package. Keep `_headers`: text responses require `Content-Encoding: br`, CORS and `Cache-Control: no-transform`. The browser performs Brotli decompression; vector and centroid `.gz` files keep their existing explicit gzip decoding.

The gzip and Brotli exporters also include `DATA-NOTICE.txt` at bundle root.
The notice retains CC BY-SA 3.0 for the archived 2022 text, predating Wikipedia's
2023 move to 4.0. It covers text only; it does not assign a licence to the
precomputed embeddings. Existing article links provide attribution.
Updating the frontend alone adds its visible notice; existing text deployments
need the notice and updated headers too. Repackage into a new output directory
and deploy those batches to update their notices without changing passage data.

## Rebuild only when the data changes

The complete Brotli bundle already exists. The redundant gzip intermediate was removed after conversion. These steps are unnecessary for frontend changes and refuse existing output directories.

To regenerate gzip text from the existing uploaded indexes:

```sh
python3 code/src/prepare_demo.py --output code/data/demo/gzip-staging
```

Convert the complete gzip staging bundle to Brotli:

```sh
UV_THREADPOOL_SIZE=12 node code/src/prepare_brotli.cjs code/data/demo/metadata-new
```

The converter preserves the 500-passage binary layout, uses Brotli quality 11, verifies exact-byte round trips, and resumes verified files from a matching `.partial/` directory. It leaves the staging input untouched. After the complete Brotli output and its upload package have been verified, the gzip staging directory can be removed. Source identities and vector/centroid descriptors must match across all three cluster indexes.

Package a complete bundle into fixed 1,000-file Brotli deployments:

```sh
python3 code/src/prepare_demo_pages.py \
  --bundle code/data/demo/metadata-new \
  --output code/data/demo/pages-new \
  --project-name wiki-ann-demo-20261022
```

Then upload using the resumable helper:

```sh
python3 code/src/cloudflare.py deploy --deployment code/data/demo/pages-new
```

The helper publishes the client only after the text deployments succeed. It preserves completed upload receipts and skips unchanged batches when rerun. Never replace live text or vector data while collecting benchmark measurements.

Keep metadata and its upload package on the same filesystem. Packaging uses hard links and fails if they cannot be created, rather than silently duplicating the full dataset.

## After publication

Check the default search and ten result texts, then the lab. Metadata requests should use `brotli-batch-*` hosts with `.bin.br` filenames and `Content-Encoding: br`. Vector requests should retain their benchmark hosts. Local source edits are not automatically deployed.

For multithreaded inference, check that `crossOriginIsolated` is `true` in both the page and embedding worker. The worker's configured `env.backends.onnx.wasm.numThreads` starts at zero (automatic) and resolves to the runtime's device-dependent count. A capable desktop should create additional ONNX worker threads; non-isolated hosting falls back to one thread.
