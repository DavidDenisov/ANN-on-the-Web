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

The probe quartiles are in `../figures/probe-32k-summary.csv`. The recovery-versus-time and recovery-versus-bytes figures report 16 FP16 configurations: probes 2, 3, 4 and 5 at 8k, and 2, 3, 4, 6, 8 and 10 at 16k and 32k. The new 8k points use `8k-summary.csv`; its four-probe point reuses the original row in `retrieval-summary.csv`. The supplement's raw results were revalidated against all 30,000 query trials, source IDs, download records and shared centroid setup; regenerating both CSV files reproduced the saved summaries exactly. The five-probe record also contains five distinct cluster downloads per query (50,000 downloads, zero retries); its first four selected clusters reproduce the retained four-probe mean payload exactly.

The 8k supplement uses the original seed 357512207, the same large-noise fixture (`ee0b3181155b0b867bdafaec76ced8495364700744e16509687f7cb930acff13`) and the same FP16 index as the retained four-probe measurement. All settings use 10,000 queries at noise magnitude 0.4. The 8k four-probe row and the earlier count table are unchanged. Each point's recorded run is preserved rather than treating the combined curve as one concurrent experiment.

The recovery axis spans 93.5–99.5%. Shading joins pointwise 95% Wilson intervals for source recovery; it does not show latency or payload variation. The time plot uses observed mean retrieval time, excluding startup, embedding and result-text loading; CDN throughput was uncontrolled. The companion plot uses mean gzipped vector bytes per query, excluding startup and text. Figure generation reads these summaries; the paper's inline results remain in `../main.tex`. The screenshot's short lab run remains an illustration and is separate from these measurements.

The complete fixed-question run uses 2,480,789 bytes of gzipped vectors and 767,752 bytes of Brotli text: 3,248,541 compressed payload bytes, excluding HTTP headers. The text total comes from the ten corresponding stored `.bin.br` files; the browser loader's raw `bytes` field counts decompressed text bodies.
