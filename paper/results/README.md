# Published measurement summaries

These compact files are copied from the completed runs used by the paper. They allow the figures to be regenerated without the corpus or full trial records. The corpus and original trial records are not included in this folder. The provenance paths below are relative to the original `code/data/benchmark/` directory, not files available in this manuscript bundle.

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

The probe quartiles are in `../figures/probe-32k-summary.csv`. The recovery-versus-time and recovery-versus-bytes plots combine all 13 original timed FP16 count/probe settings with the four additional 32k probe settings, deduplicating shared configurations. This retains the 16k sweep at 1, 2, 3, 4, 6, 8 and 10 probes alongside the same seven 32k settings and three 8k settings (17 configurations total). Both plots use the retained 10,000-query results at noise magnitude 0.4. Shading joins pointwise 95% Wilson intervals for source recovery; it does not show latency or payload variation. The time plot uses observed mean retrieval time, excluding startup, embedding and result-text loading; CDN throughput was uncontrolled. The companion plot uses mean gzipped vector bytes per query, excluding startup and text. Figure generation reads these summaries; the paper's inline results remain in `../main.tex`. No new evaluation measurements were collected for this revision; the short lab run in the screenshot is an illustration only.

The complete fixed-question run uses 2,480,789 bytes of gzipped vectors and 767,752 bytes of Brotli text: 3,248,541 compressed payload bytes, excluding HTTP headers. The text total comes from the ten corresponding stored `.bin.br` files; the browser loader's raw `bytes` field counts decompressed text bodies.
