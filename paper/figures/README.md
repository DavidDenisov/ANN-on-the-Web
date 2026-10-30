# Paper figures

From the repository root, run `python3 paper/figures/generate.py` with matplotlib and reportlab installed. The generator reads retained compact summaries and manifest size fields, checks their consistency, and regenerates the architecture, probe plot, table rows, recovery intervals and derived LaTeX values. No corpus download is needed.

The paper uses:

- `screenshots/search-desktop.png` and `screenshots/lab-desktop.png`: actual public-page captures on October 30, 2026 at 1440×960 CSS pixels, device scale factor 2 (2880×1920 PNGs). The wider viewport is above the search layout's 1280px stacking breakpoint. Search shows a real ten-result query at the default settings; the lab is ready before a run. No results or timings were inserted. Visible search timings belong to the capture, not the original benchmark averages.
- `architecture.pdf`: startup and the five local-search steps, with selective static-file downloads.
- `probe-large.pdf`: the 32k FP16 probe sweep, median latency/interquartile band and source recovery with 95% Wilson intervals.
- `large-count-table.tex` and rows: the three four-probe count settings. The full nine-setting original count/probe matrix remains in `../results/retrieval-summary.csv`.
- `large-compression-table.tex` and rows: all six centroid formats at 32k and four probes.

The older `screenshots/search-full.png` and `screenshots/lab-full.png` are historical October 26 captures. They are not used in the current paper; the earlier lab allowed a ten-query illustration. The obsolete stylized search/lab panels have been removed from the active figure set.

`probe-32k-summary.csv` preserves timing quartiles. `../results/recovery-intervals.csv` and `review-derived.tex` are computed from retained exact counts. `cluster-distribution.json` records live-manifest provenance; its compressed size-field inputs reproduce the quantiles and conservative four-probe byte bound offline.
