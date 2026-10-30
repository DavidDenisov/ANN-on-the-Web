# Paper figures

Run from the repository root:

```sh
python3 -m pip install matplotlib reportlab
python3 paper/figures/generate.py
```

The generator reads only the compact summaries committed in `paper/results/` and `probe-32k-summary.csv`. It does not need the local corpus, full trial logs, or hosted text files.

The active paper uses:

- `search-example.pdf`: the example question and top passage, with startup, embedding, vector retrieval, text loading, and directly measured total query timings.
- `lab-example.pdf`: the ten-query, high-noise lab walkthrough at 16k clusters with one and four probes.
- `architecture.pdf`: startup and the five browser search steps, including selective downloads from static hosting.
- `probe-large.pdf`: the 32k FP16 probe sweep, plotting median retrieval time with a 25th–75th percentile band and source recovery@1.
- `large-count-table.tex` and `large-count-rows.tex`: all nine large-noise cluster/probe combinations.
- `large-compression-table.tex` and `large-compression-rows.tex`: all six centroid formats at 32k clusters and four probes.

`screenshots/search-full.png` and `screenshots/lab-full.png` are original public-demo captures from October 26, 2026. The lab panel's illustrative values are transcribed from the latter. Query timings in the search panel come from the saved 1,000-repeat complete-query run; startup comes from the separate 100-context 32k/SQ6 run.

`probe-32k-summary.csv` preserves quartiles extracted from the saved query timings. The generator retains configuration, sample-count, and layout checks.
