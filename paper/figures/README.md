# Paper figures

From the repository root, run `python3 paper/figures/generate.py` with matplotlib and reportlab installed. The generator combines the retained timing and payload summaries with the scored top-1 results. It regenerates the architecture, agreement plots, table rows and derived LaTeX values.

Production uses the pre-enlargement presentation from commit `0e26a88`. The enlarged HTML/CSS snapshot in `client-preview/` is hosted at the separate [paper preview](https://paper-figures.wiki-ann-demo-20261022.pages.dev/client_wiki/). Use that preview for future paper captures. To refresh it, overlay `client-preview/` onto `client_wiki/` in a separate prepared frontend root and deploy branch `paper-figures`; keep these presentation files out of the production (`main`) deployment. The screenshots below use the enlarged paper presentation.

The paper uses the assets below. LaTeX trims the screenshot mastheads and bottom edges to focus on controls and results. It also trims the right edge of the search capture.

- `screenshots/search-desktop.jpg`: full browser capture (1425×1418 pixels), taken on November 3, 2026. Body text, secondary controls, corpus/result summaries, helper text and passage-distance labels use 24 px text; result titles use 28 px. Pane headings, search status and the primary search button use 32 px. Privacy and download information forms one paragraph; privacy inferences are described as possibilities using “may reveal”. The default ten-result plants/sunlight question uses the actual site styles and unchanged colors. A warm-up search fetched the required files before capture. The captured search ran in a fresh browser context with caching disabled and used no-store requests for vector and result-text downloads. Displayed timings are separate from the retained paper measurements. The stored image is not cropped or resized.
- `screenshots/lab-desktop.jpg`: full browser capture (1425×1165 pixels), taken on November 2, 2026. Body text, form controls, helper text, corpus information and table text use 24 px; section headings use 28 px and the Run all checks button 32 px. Both screenshots use the same width and printed text scale. The full left sidebar and eight complete result rows from a short ten-query-per-configuration cluster-count check remain visible; the last row is below the capture. Colors are unchanged; results are not inserted or edited. The paper's retained measurements are unchanged. The stored image is not cropped or resized.
- `architecture.pdf`: model, WebAssembly runtime and centroid startup downloads, followed by the five local-search steps and selective static-file downloads. Its DejaVu Sans fonts are embedded using the font files bundled with matplotlib.
- `agreement-bytes.pdf` shows top-1 agreement versus mean gzipped vector payload for 16 FP16 settings at noise magnitude 0.4. The curves show probes 2, 3, 4 and 5 at 8k, and 2, 3, 4, 6, 8 and 10 at 16k and 32k. Shading shows pointwise 95% Wilson intervals over 10,000 queries per setting.
- `agreement-time.pdf` shows the same settings against observed mean retrieval time, including routing, vector download, decompression, decoding and ranking.
- `large-count-table.tex` and rows: the three four-probe count settings. Historical count/probe measurements remain in `../results/retrieval-summary.csv`.
- `large-compression-table.tex` and rows: all six centroid formats at 32k and four probes.

`probe-large.pdf` shows the 32k FP16 probe sweep with median latency, its interquartile range, and top-1 agreement with 95% Wilson intervals.

`probe-32k-summary.csv` preserves timing quartiles. `../results/top1-agreement.csv` supplies agreement counts and `review-derived.tex` supplies the generated default agreement and confidence interval.
