# Published measurement summaries

These compact files are copied from the completed runs used by the paper. They allow the figures to be regenerated without the corpus or full trial records. The original runs remain locally under the ignored `code/data/benchmark/` directory.

- `retrieval-summary.csv`: original retrieval/accuracy run, from `retrieval/summary.csv`.
- `centroid-summary.csv`: centroid setup run, from `centroid-setup/centroid-summary.csv`.
- `32k-summary.csv`: completed supplement, from `remaining-32k-new/summary.csv`.
- `32k-centroid-summary.csv`: supplement centroid measurements, from `remaining-32k-new/centroid-summary.csv`.
- `startup-summary.json`: 100 elapsed 32k/SQ6 worker startups, from `remaining-32k-new/startup/summary.json`.
- `query-example.json`: fixed-question top-10 results and component/total timings, from `live-query-warm/summary.json`.
- `embedding-summary.json`: inference timings by token length, from `encoder/inference/summary.json`.
- `result-text-summary.json`: 1,000 varied ten-result text loads, from `result-text/summary.json`.
- `compatibility-summary.json`: 1,000-passage browser/stored-vector comparison, from `encoder/compatibility-1k/summary.json`.
- `nearest-neighbour-summary.json`: exact full-corpus geometry for 1,000 sources, from `full-nearest-gpu/summary.json`.
- `noise-summary.json`: source-query perturbation geometry, from `remaining-32k-new/noise/summary.json`.
- `settings.tex`: sample-count macros used by the paper.

The probe quartiles are in `../figures/probe-32k-summary.csv`. Figure generation reads these summaries; the paper's inline results remain in `../main.tex`. No measurements were rerun when preparing this repository.

The complete fixed-question run uses 2,480,789 bytes of gzipped vectors and 767,752 bytes of Brotli text: 3,248,541 compressed payload bytes, excluding HTTP headers. The text total comes from the ten corresponding stored `.bin.br` files; the browser loader's raw `bytes` field counts decompressed text bodies.
