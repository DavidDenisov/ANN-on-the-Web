# Paper figures

From the repository root, run `python3 paper/figures/generate.py` with matplotlib and reportlab installed. The generator reads retained compact summaries and manifest size fields, checks their consistency, and regenerates the architecture, probe plot, table rows, recovery intervals and derived LaTeX values. No corpus download is needed.

The paper uses:

- `screenshots/search-query-detail.png` and `screenshots/search-result-detail.png`: enlarged close-ups of the actual question field and top passage from a real default ten-result search. Browser text is enlarged from 16px to 24px for capture, and the question box is shortened to remove blank space. Each element is captured at 2× pixel density and placed at full column width. No result content or distance is inserted or altered.
- `screenshots/lab-settings-detail.png` and `screenshots/lab-probes-detail.png`: enlarged query-noise control and probe-budget panel, captured from the public lab before a run. The viewport is 740 CSS pixels wide, text is enlarged to 24px, and table headers wrap to fit the responsive layout. Captures are at 2× pixel density. No measurements are inserted.

The detail captures were taken on October 30, 2026. The capture-only text enlargement, shorter textarea and wrapping table headers do not change the deployed interface. The earlier `search-desktop.png` and `lab-desktop.png` overview captures (1440×960 CSS pixels at 2× density) remain as provenance but are no longer reduced into the paper's columns.
- `architecture.pdf`: startup and the five local-search steps, with selective static-file downloads.
- `probe-large.pdf`: the 32k FP16 probe sweep, median latency/interquartile band and source recovery with 95% Wilson intervals.
- `large-count-table.tex` and rows: the three four-probe count settings. The full nine-setting original count/probe matrix remains in `../results/retrieval-summary.csv`.
- `large-compression-table.tex` and rows: all six centroid formats at 32k and four probes.

The older `screenshots/search-full.png` and `screenshots/lab-full.png` are historical October 26 captures. They are not used in the current paper; the earlier lab allowed a ten-query illustration. The obsolete stylized search/lab panels have been removed from the active figure set.

`probe-32k-summary.csv` preserves timing quartiles. `../results/recovery-intervals.csv` and `review-derived.tex` are computed from retained exact counts. `cluster-distribution.json` records live-manifest provenance; its compressed size-field inputs reproduce the quantiles and conservative four-probe byte bound offline.
