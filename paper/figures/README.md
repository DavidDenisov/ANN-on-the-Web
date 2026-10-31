# Paper figures

From the repository root, run `python3 paper/figures/generate.py` with matplotlib and reportlab installed. The generator reads retained compact summaries and manifest size fields, checks their consistency, and regenerates the architecture, recovery-versus-bytes and probe plots, table rows, recovery intervals and derived LaTeX values. No corpus download is needed.

The paper uses:

- `screenshots/search-desktop.jpg` and `screenshots/lab-desktop.jpg`: full-width desktop page frames captured on October 31, 2026 at 1440×1200 pixels, shown side by side without close-up crops. A temporary local copy of the deployed client used the unchanged public manifests and remote data, with capture-only 20px passage/control/table text, 26px panel/result headings, and desktop layout. Search shows the default ten-result plants/sunlight question with privacy/download notices. The lab shows a completed cluster-count illustration with ten sampled queries per configuration at noise magnitude 0.4. No result content or timings were inserted. The illustrative run and capture timings are separate from the retained 10,000-query paper measurements; capture styles were not applied to the application source or deployment.
- `architecture.pdf`: startup and the five local-search steps, with selective static-file downloads.
- `recovery-bytes.pdf`: source recovery versus mean compressed vector payload for all 13 retained FP16 count/probe settings at noise magnitude 0.4, with probe labels and 95% Wilson intervals. Each setting uses 10,000 queries; startup and result-text bytes are excluded.
- `large-count-table.tex` and rows: the three four-probe count settings. The full nine-setting original count/probe matrix remains in `../results/retrieval-summary.csv`.
- `large-compression-table.tex` and rows: all six centroid formats at 32k and four probes.

The older `screenshots/search-full.png` and `screenshots/lab-full.png` are historical October 26 captures. They are not used in the current paper; the earlier lab allowed a ten-query illustration. The obsolete stylized search/lab panels have been removed from the active figure set.

The superseded October 30 desktop PNGs are archived locally and are not included in this repository.

`probe-large.pdf` retains the earlier 32k FP16 probe sweep with median latency/interquartile band and source recovery with 95% Wilson intervals. It remains available as a supporting asset; the current Figure 4 uses `recovery-bytes.pdf`.

`probe-32k-summary.csv` preserves timing quartiles. `../results/recovery-intervals.csv` and `review-derived.tex` are computed from retained exact counts. `cluster-distribution.json` records live-manifest provenance; its compressed size-field inputs reproduce the quantiles and conservative four-probe byte bound offline.

The four `*-detail.png` close-ups are retained as earlier assets and are not used in the paper.
