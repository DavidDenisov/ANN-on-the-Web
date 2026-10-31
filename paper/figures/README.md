# Paper figures

From the repository root, run `python3 paper/figures/generate.py` with matplotlib and reportlab installed. The generator reads retained compact summaries and manifest size fields, checks their consistency, and regenerates the architecture, recovery-versus-time, recovery-versus-bytes and probe plots, table rows, recovery intervals and derived LaTeX values. No corpus download is needed.

The paper uses:

- `screenshots/search-desktop.jpg` and `screenshots/lab-desktop.jpg`: full-width desktop page frames captured on October 31, 2026 at 1440×1200 pixels, shown side by side without close-up crops. A temporary local copy of the deployed client used the unchanged public manifests and remote data, with capture-only 20px passage/control/table text, 26px panel/result headings, and desktop layout. Search shows the default ten-result plants/sunlight question with privacy/download notices. The lab shows a completed cluster-count illustration with ten sampled queries per configuration at noise magnitude 0.4. No result content or timings were inserted. The illustrative run and capture timings are separate from the retained 10,000-query paper measurements; capture styles were not applied to the application source or deployment.
- `architecture.pdf`: startup and the five local-search steps, with selective static-file downloads.
- `recovery-bytes.pdf`: the paper's Figure 4, showing source recovery versus mean gzipped vector payload for 16 FP16 settings at noise magnitude 0.4: probes 2, 3, 4 and 5 at 8k, and 2, 3, 4, 6, 8 and 10 at 16k and 32k. The new 8k supplement supplies probes 2, 3 and 5; its four-probe point reuses the original measurement with the same query fixture and index. File payload is fixed for these queries and unaffected by network conditions during multi-day runs, so it is the main comparison metric. Mean retrieval time was approximately linear in payload across the tested 8k range; the paper scopes this observation to those settings. Startup, result-text bytes and HTTP headers are excluded. The single panel focuses on 93.5–99.5% recovery, with 2-point markers and probe labels. Shading joins pointwise 95% Wilson recovery intervals. Each setting uses 10,000 queries. Other historical settings and the four-probe count table remain unchanged.
- `recovery-time.pdf`: supporting plot of the same 16 settings against observed mean retrieval time. Retrieval covers routing, vector download, decompression, decoding and ranking; it excludes startup, embedding and result-text loading. It retains the same recovery range, small markers and shaded recovery intervals. Timings describe the measured desktop/CDN setup with uncontrolled throughput.
- `large-count-table.tex` and rows: the three four-probe count settings. Historical count/probe measurements remain in `../results/retrieval-summary.csv`.
- `large-compression-table.tex` and rows: all six centroid formats at 32k and four probes.

The older `screenshots/search-full.png` and `screenshots/lab-full.png` are historical October 26 captures. They are not used in the current paper; the earlier lab allowed a ten-query illustration. The obsolete stylized search/lab panels have been removed from the active figure set.

The superseded October 30 desktop PNGs are archived locally and are not included in this repository.

`probe-large.pdf` retains the earlier 32k FP16 probe sweep with median latency/interquartile band and source recovery with shaded 95% Wilson intervals. It remains available as a supporting asset; the current Figure 4 uses `recovery-bytes.pdf`.

`probe-32k-summary.csv` preserves timing quartiles. `../results/recovery-intervals.csv` and `review-derived.tex` are computed from retained exact counts. `cluster-distribution.json` records live-manifest provenance; its compressed size-field inputs reproduce the quantiles and conservative four-probe byte bound offline.

The four `*-detail.png` close-ups are retained as earlier assets and are not used in the paper.
