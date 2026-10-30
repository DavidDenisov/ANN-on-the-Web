# Wikipedia browser search

Semantic search over 35,167,920 Wikipedia passages using local query embedding and ranking with selective downloads from static hosting.

- [Paper PDF](paper/main.pdf)
- [Review follow-up](code/docs/review-follow-up.md)
- [Search demo](https://wiki-ann-demo-20261022.pages.dev/client_wiki/)
- [Clustering lab](https://wiki-ann-demo-20261022.pages.dev/client_wiki/benchmark.html)

The default search uses 32,768 clusters, SQ6 centroids, four probes, and ten results.

## Repository layout

- `paper/`: the four-page demo paper, bibliography, and bundled ACM class and sources.
- `paper/figures/`: figure generator, required figures/tables, and original screenshots.
- `paper/results/`: compact measurement summaries and LaTeX sample counts.
- `code/client_wiki/`: search, clustering lab, and browser comparison runner.
- `code/src/`: dataset preparation, clustering, publishing, and comparison scripts.
- `code/benchmarks/`: browser measurements and the exact-neighbour scan.
- `code/tests/`: CPU and browser-module regression tests.
- `code/docs/`: [client](code/docs/client.md), [dataset](code/docs/dataset.md), [publishing](code/docs/publish-demo.md), and [benchmark](code/docs/benchmarks.md) guides.

## Build the paper

With TeX Live and `latexmk` installed:

```sh
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error -outdir=../build paper/main.tex
```

The result is `build/main.pdf`. Required figure PDFs are included, so building the paper needs no dataset download or Python environment. The bundled, unmodified ACM class includes its source and installer.

To regenerate the figures from the committed summaries:

```sh
python3 -m pip install matplotlib reportlab
python3 paper/figures/generate.py
```

See [figure notes](paper/figures/README.md) and [result provenance](paper/results/README.md).

## Run locally and reproduce measurements

After preparing the local index manifests described in the [client guide](code/docs/client.md):

```sh
python3 code/src/run_comparisons.py --serve
```

Open [local search](http://127.0.0.1:8080/client_wiki/) or [the lab](http://127.0.0.1:8080/client_wiki/benchmark.html). Use the public demo above when no local preparation is available.

The [benchmark guide](code/docs/benchmarks.md) covers the full comparison, 32k supplement, and exact full-corpus neighbour scan. [Client measurement notes](code/docs/client-measurements.md) describe embedding, startup, compatibility, text-loading, and complete-query protocols. New runs require new output directories and preserve existing measurements.

## Regression checks

The JavaScript checks use Node.js and synthetic local fixtures:

```sh
node --test code/tests/*.test.cjs code/tests/*.test.mjs
```

With the Python dependencies and CPU or GPU Faiss installed:

```sh
python3 -m unittest discover -s code/tests -p 'test_*.py'
```

These checks cover centroid formats, vector ranking, metadata decoding/caching, cancellation, and local shard validation. They do not download the corpus or run the full measurements.

Raw datasets, prepared indexes, model/browser caches, and full trial records belong under `code/data/` and are excluded from Git. Compact paper summaries remain in `paper/results/`. Compiled papers, caches, and local review-tool output are also excluded.

The bundled Inter fonts include their [license](code/client_wiki/assets/fonts/OFL.txt). The ACM template retains its original notices and sources under `paper/`.

## Data attribution and hosting

See [data attribution](DATA-NOTICE.md) for Wikipedia and embedding provenance. The public repository contains source, compact measurements and screenshots; it does not redistribute the full text or vector indexes. Live CDN availability is separate from source availability. The [publishing guide](code/docs/publish-demo.md#self-hosting-and-archival) describes preserving manifests and moving assets to object storage.

The source includes new bandwidth and access-pattern notices; this GitHub publication does not deploy those HTML changes to the existing Cloudflare demo.
