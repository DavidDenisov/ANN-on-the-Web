# Published measurement summaries

These compact files are copied from the completed runs used by the paper. They allow the figures to be regenerated without the corpus or full trial records. The corpus and original trial records are not included in this folder. The provenance paths below are relative to the original `code/data/benchmark/` directory, not files available in this manuscript bundle.

- `retrieval-summary.csv`: original retrieval/accuracy run, from `retrieval/summary.csv`.
- `centroid-summary.csv`: centroid setup run, from `centroid-setup/centroid-summary.csv`.
- `32k-summary.csv`: completed supplement, from `remaining-32k-new/summary.csv`.
- `32k-centroid-summary.csv`: supplement centroid measurements, from `remaining-32k-new/centroid-summary.csv`.
- `8k-summary.csv`: completed FP16 supplement at probes 2, 3 and 5, copied from `8k-p235-new/summary.csv`.
- `8k-centroid-summary.csv`: its 1,000-trial FP16 centroid setup, from `8k-p235-new/centroid-summary.csv`.
- `startup-summary.json`: 100 elapsed 32k/SQ6 worker startups, from `remaining-32k-new/startup/summary.json`.
- `query-example.json`: fixed-question top-10 results and component/total timings, from `live-query-warm/summary.json`.
- `embedding-summary.json`: inference timings by token length, from `encoder/inference/summary.json`.
- `result-text-summary.json`: 1,000 varied ten-result text loads, from `result-text/summary.json`.
- `compatibility-summary.json`: 1,000-passage browser/stored-vector comparison, from `encoder/compatibility-1k/summary.json`.
- `nearest-neighbour-summary.json`: exact full-corpus geometry for 1,000 sources, from `full-nearest-gpu/summary.json`.
- `noise-summary.json`: source-query perturbation geometry, from `remaining-32k-new/noise/summary.json`.
- `settings.tex`: sample-count macros used by the paper.
- `top1-agreement.csv`: 31 high-noise retrieval runs rescored with `score_agreement.py`. The exact reference used the offline Wikipedia shard scan over all 35,167,920 passages and the original 10,000 query vectors. The CSV records the fixture ID, browser-result origin, number of matching top-1 IDs and source-recovery counts.

The probe quartiles are in `../figures/probe-32k-summary.csv`. The agreement plots report 16 FP16 configurations, with probes 2, 3, 4 and 5 at 8k and probes 2, 3, 4, 6, 8 and 10 at 16k and 32k. The 8k supplement supplies probes 2, 3 and 5. The four-probe 8k point reuses the original run.

The 8k supplement uses the original seed 357512207, the same large-noise fixture (`ee0b3181155b0b867bdafaec76ced8495364700744e16509687f7cb930acff13`) and the same FP16 index as the retained four-probe measurement. All settings use 10,000 queries at noise magnitude 0.4. The 8k four-probe row and the earlier count table are unchanged. Each point's recorded run is preserved rather than treating the combined curve as one concurrent experiment.

Shading shows pointwise 95% Wilson intervals for top-1 agreement. The paper's main plot uses mean gzipped vector bytes per query, excluding startup, text and HTTP headers. Across the four 8k settings, observed mean retrieval time scaled approximately linearly with payload over 6.88–16.98 MB/query. The supporting time plot uses observed retrieval time, excluding startup, embedding and text loading. CDN throughput was uncontrolled.

The complete fixed-question run uses 2,480,789 bytes of gzipped vectors and 767,752 bytes of Brotli text: 3,248,541 compressed payload bytes, excluding HTTP headers. The text total comes from the ten corresponding stored `.bin.br` files; the browser loader's raw `bytes` field counts decompressed text bodies.

The total-query component estimates sum the 32k/SQ6, four-probe, large-noise vector mean in `32k-summary.csv` (122.392068 ms, k=1), the ten-passage elapsed mean in `result-text-summary.json` (34.891355 ms), and each 64/128/256-token inference mean in `embedding-summary.json` (9.089686/16.865360/33.840000 ms). The resulting 166.4/174.1/191.1 ms budgets combine separate workloads. They exclude startup, DOM rendering, worker messaging and any additional cost of retaining ten vector results; they are not measured end-to-end means or p95 estimates.
