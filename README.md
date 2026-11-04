# Wikipedia browser search

Semantic search over 35,167,920 Wikipedia passages using local query embedding and ranking with selective downloads from static hosting.

- [Paper PDF](paper/main.pdf)
- [Search demo](https://wiki-ann-demo-20261022.pages.dev/client_wiki/)
- [Clustering lab](https://wiki-ann-demo-20261022.pages.dev/client_wiki/benchmark.html)

The default search uses 32,768 clusters, SQ6 centroids, four probes, and ten results.

The search page automatically loads roughly 47 MB of model, runtime, and centroid asset bodies at the default settings; HTTP compression and caching can reduce network transfer. Query text and embeddings stay in the browser, while requested cluster/result-text files, IP addresses, and timing remain visible to the hosting provider. Search downloads depend on the question, settings, and cache.

The lab defaults to 10 queries per configuration and accepts 1–10,000. Short runs shuffle the prepared query indices once, take the selected number, and reuse that subset across configurations. Result rows show their sample counts, and runs can be canceled. Small samples illustrate behavior; full paper measurements still use all 10,000 queries. Even short checks can transfer hundreds of MB; full checks can take hours and transfer hundreds of GB.

The data and privacy notices and short-run controls are deployed to the public demo. Update the [Cloudflare frontend](code/docs/publish-demo.md#frontend-update) to publish further changes; source publication alone does not deploy them.

## License

Copyright (c) 2026 David Denisov.

The original application code and accompanying code documentation are open source
under the [GNU Affero General Public License version 3](LICENSE)
(`AGPL-3.0-only`). Commercial and noncommercial use, modification, and
redistribution are permitted under those terms.

Redistributors must preserve copyright and licence notices and provide the
corresponding source as required by AGPLv3. Modified versions that support remote
network interaction must prominently offer their corresponding source, including
modifications, to interacting users at no charge. Credit alone does not satisfy
all licence requirements. The software is provided without warranty; see
[LICENSE](LICENSE) for the full terms.

For permissions beyond AGPLv3, contact the
[repository owner](https://github.com/DavidDenisov) about a separate licence.

The licence covers only the original software and its code documentation.
The paper, figures, datasets, embeddings, model weights, fonts, ACM template,
and third-party components retain their applicable separate terms. Permissions
previously granted under other licences remain subject to those terms.

## Repository layout

- `paper/`: the four-page demo paper, bibliography, and bundled ACM class and sources.
- `paper/figures/`: figure generator, required figures/tables, and original screenshots.
- `paper/results/`: compact measurement summaries and LaTeX sample counts.
- `code/client_wiki/`: search, clustering lab, and browser comparison runner.
- `code/src/`: dataset preparation, clustering, publishing, and comparison scripts.
- `code/benchmarks/`: offline noise analysis and saved inference summaries.
- `code/docs/`: [client](code/docs/client.md), [dataset](code/docs/dataset.md), [publishing](code/docs/publish-demo.md), and [benchmark](code/docs/benchmarks.md) guides.

## Build the paper

With TeX Live and `latexmk` installed:

```sh
latexmk -cd -pdf -interaction=nonstopmode -halt-on-error -outdir=../build paper/main.tex
```

The result is `build/main.pdf`. Required figure PDFs are included, so building the paper needs no dataset download or Python environment. The bundled, unmodified ACM class includes its source and installer.

To regenerate the figures from the committed summaries:

```sh
python3 -m pip install matplotlib reportlab
python3 paper/figures/generate.py
```

See [figure notes](paper/figures/README.md) and [result provenance](paper/results/README.md).

## Run locally and reproduce measurements

After preparing the local index manifests described in the [client guide](code/docs/client.md):

```sh
python3 code/src/run_comparisons.py --serve
```

Open [local search](http://127.0.0.1:8080/client_wiki/) or [the lab](http://127.0.0.1:8080/client_wiki/benchmark.html). Use the public demo above when no local preparation is available.

The paper measures top-1 agreement with exhaustive Faiss search over the full corpus. The [benchmark guide](code/docs/benchmarks.md#agreement-with-exact-search) explains how to build the exact reference and score saved browser runs. [Client measurement notes](code/docs/client-measurements.md) preserve the historical embedding, startup, compatibility, text-loading, and complete-query protocols. Standalone live measurement runners have been removed.

This upload contains source code, documentation, required paper figures, and compact measurement summaries in `paper/results/`. Raw datasets, prepared indexes, model/browser caches, full trial records, build intermediates, and local review-tool output are not included. The current compiled paper is included at `paper/main.pdf`.

The bundled Inter fonts include their [license](code/client_wiki/assets/fonts/OFL.txt). The ACM template retains its original notices and sources under `paper/`.

## Data attribution and hosting

See [data attribution](DATA-NOTICE.md) for Wikipedia and embedding provenance. The repository does not redistribute the full text or vector indexes. Live CDN availability is separate from source availability; the [publishing guide](code/docs/publish-demo.md) describes the existing hosting layout and frontend updates.
