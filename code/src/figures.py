"""Export paper tables and the probe graph from completed Wikipedia measurements."""

import argparse
import csv
import ipaddress
import json
import math
from pathlib import Path
import statistics
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit, urlunsplit

import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator

CODECS = {"f16": "FP16", "sq8": "SQ8", "sq6": "SQ6", "sq4": "SQ4",
          "int8": "INT8", "pq4": "PQ4"}
STYLE = {"font.family": "serif", "font.size": 9, "axes.labelsize": 9,
         "axes.titlesize": 9, "xtick.labelsize": 8,
         "ytick.labelsize": 8, "axes.spines.top": False, "axes.spines.right": False,
         "axes.linewidth": .7, "lines.linewidth": 1.25, "lines.markersize": 4,
         "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
         "pdf.fonttype": 42, "mathtext.fontset": "dejavuserif"}


def _integer(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _number(value, name, minimum=0, maximum=None, positive=False):
    if (type(value) not in (int, float) or not math.isfinite(value)
            or value < minimum or (positive and value <= 0)
            or (maximum is not None and value > maximum)):
        raise ValueError(f"{name} must be finite and {'positive' if positive else 'in range'}")
    return value


def _centroid_samples(row):
    setup_id = row.get("centroid_setup_id")
    if not isinstance(setup_id, str) or not setup_id:
        raise ValueError("Missing shared centroid setup identity")
    repeats = _integer(row.get("centroid_setup_repeats"), "centroid_setup_repeats")
    trials = row.get("centroidTrials")
    if not isinstance(trials, list) or len(trials) != repeats:
        raise ValueError("centroidTrials must match the shared setup sample count")
    seen = set()
    for trial in trials:
        if not isinstance(trial, dict):
            raise ValueError("Each centroid trial must be an object")
        if (trial.get("setup_id") != setup_id or not isinstance(trial.get("requested_url"), str)
                or not trial["requested_url"] or trial["requested_url"] != trials[0].get("requested_url")):
            raise ValueError("Shared centroid setup identity does not match its download")
        sample = trial.get("sample")
        if type(sample) is not int or not 0 <= sample < repeats or sample in seen:
            raise ValueError("Invalid or duplicate centroid setup sample")
        seen.add(sample)
        _number(trial.get("decodeMs"), "centroid decodeMs")
        _number(trial.get("downloadMs"), "centroid downloadMs", positive=True)
        _integer(trial.get("bytes"), "centroid bytes")
    if len({trial["bytes"] for trial in trials}) != 1:
        raise ValueError("Shared centroid samples must use the same payload size")
    return {"centroidTrials": trials, "centroid_setup_id": setup_id, "centroid_setup_repeats": repeats}


def _row(row):
    if not isinstance(row, dict):
        raise ValueError("Each result must be an object")
    codec = row.get("codec")
    if not isinstance(codec, str) or codec not in CODECS:
        raise ValueError(f"Unsupported centroid codec: {codec!r}")
    values = {key: _integer(row.get(key), key)
              for key in ("clusters", "probes", "k", "queries", "repeats")}
    accuracy_only = row.get("accuracy_only")
    if type(accuracy_only) is not bool or values["repeats"] != 1:
        raise ValueError("Measurements require one pass")
    if row.get("cluster_fetch_cache") != ("force-cache" if accuracy_only else "no-store"):
        raise ValueError("Result cache policy does not match its test")
    if accuracy_only and any(key in row for key in ("queryMs", "routingMs", "clusterDownloadMs", "clusterDecodeMs", "scoringMs")):
        raise ValueError("Accuracy-only checks must omit query timing")
    if values["probes"] > values["clusters"]:
        raise ValueError("probes cannot exceed clusters")
    trials = row.get("trials")
    if not isinstance(trials, list) or len(trials) != values["queries"] * values["repeats"]:
        raise ValueError("trials must contain queries × repeats observations")
    seen = set()
    for trial in trials:
        if not isinstance(trial, dict):
            raise ValueError("Each query trial must be an object")
        repeat, query = trial.get("repeat"), trial.get("query")
        if (type(repeat) is not int or not 0 <= repeat < values["repeats"]
                or type(query) is not int or not 0 <= query < values["queries"]
                or (repeat, query) in seen):
            raise ValueError("Invalid or duplicate query/repeat in trials")
        seen.add((repeat, query))
        _number(trial.get("recall"), "trial recall", maximum=1)
        if accuracy_only:
            if any(key in trial for key in ("queryMs", "routingMs", "downloadMs", "decodeMs", "scoringMs")):
                raise ValueError("Accuracy-only trials must omit query timing")
        else:
            _number(trial.get("queryMs"), "queryMs", positive=True)
    return {**values, "codec": codec, "metric": row.get("metric"), "accuracy_only": accuracy_only, "trials": trials,
            **_centroid_samples(row)}


def _centroid_downloads(rows):
    shared = {}
    for row in rows:
        for trial in row["centroidTrials"]:
            key = (row["centroid_setup_id"], trial["sample"])
            if key in shared and shared[key] != trial:
                raise ValueError("Conflicting measurements for a shared centroid setup")
            shared[key] = trial
    return list(shared.values())


def _accuracy_value(trials):
    return f"{statistics.mean(t['recall'] for t in trials) * 100:.1f}".rstrip("0").rstrip(".")


def _latency_cells(rows):
    cells = []
    for row in sorted(rows, key=lambda row: (row["clusters"], row["probes"])):
        trials = row["trials"]
        times = sorted(trial["queryMs"] for trial in trials)
        cells.append([str(row["clusters"]), str(row["probes"]),
                      _accuracy_value(trials),
                      f"{statistics.mean(times):.0f}", f"{times[math.ceil(len(times) * .95) - 1]:.0f}",
                      f"{statistics.mean(t['bytes'] for t in trials) / 1e6:.2f}"])
    return cells


def _latency_comparison_table(report, grouped):
    headers = ["Clusters", "Probes"]
    values = {}
    for suite in report["config"]["suites"]:
        label = suite["label"]
        headers += [f"{label.capitalize()}\nRecovery (%)", f"{label.capitalize()}\nTime (ms)\nmean / p95",
                    f"{label.capitalize()}\nDownload\n(MB/query)"]
        values[label] = {tuple(cell[:2]): cell[2:] for cell in _latency_cells(grouped[label]["counts"])}
    cells = []
    for key in sorted(next(iter(values.values())), key=lambda key: tuple(map(int, key))):
        cell = list(key)
        for group in values.values():
            accuracy, mean, p95, size = group[key]
            cell += [accuracy, f"{mean} / {p95}", size]
        cells.append(cell)
    return headers, cells


def _compression_table(groups):
    headers = ["Format"] + [f"{label.capitalize()}\nSource recovery (%)" for label in groups]
    cells = []
    for codec, label in CODECS.items():
        values = [label]
        for rows in groups.values():
            trials = [trial for row in rows if row["codec"] == codec for trial in row["trials"]]
            values.append(_accuracy_value(trials))
        cells.append(values)
    return headers, cells


def _centroid_timing_table(report, formats=False):
    headers = ["Format" if formats else "Clusters", "Size\n(MB)",
               "Fetch (ms)\nmean / p95", "Decode (ms)\nmean / p95"]
    rows = report.get("centroid_setups") or [item["result"] for item in report["measurements"]]
    cells = []
    pairs = ([(16384, codec) for codec in CODECS] if formats
             else [(clusters, "f16") for clusters in sorted(report["config"]["cluster_counts"])])
    for clusters, codec in pairs:
        samples = _centroid_downloads([row for row in rows if row["clusters"] == clusters and row["codec"] == codec])
        if not samples:
            raise ValueError(f"Missing centroid setup for {clusters} / {codec}")
        downloads = sorted(sample["downloadMs"] for sample in samples)
        decodes = sorted(_number(sample.get("decodeMs"), "centroid decodeMs") for sample in samples)
        p95 = math.ceil(len(samples) * .95) - 1
        cells.append([CODECS[codec] if formats else str(clusters), f"{samples[0]['bytes'] / 1e6:.2f}",
                      f"{statistics.mean(downloads):.0f} / {downloads[p95]:.0f}",
                      f"{statistics.mean(decodes):.0f} / {decodes[p95]:.0f}"])
    return headers, cells


def _paper_config(config):
    fixed = {"cluster_counts": [8192, 16384, 32768], "cluster_probe_counts": [1, 4, 8],
             "probe_cluster_counts": [16384], "probe_counts": [1, 2, 3, 4, 6, 8, 10],
             "quantization_clusters": 16384, "quantization_probes": 4,
             "codecs": list(CODECS), "k": 1, "query_count": 10000, "repeats": 1}
    if not isinstance(config, dict) or any(config.get(key) != value for key, value in fixed.items()):
        raise ValueError("The report does not contain the paper's fixed experiment")


def _asset_base(value):
    if not isinstance(value, str):
        raise ValueError("Asset base URL must be a string")
    try:
        parsed = urlsplit(value)
        if parsed.scheme != "https" or not parsed.netloc or parsed.query or parsed.fragment or parsed.username or parsed.password:
            raise ValueError("Asset base URL must be absolute without query or fragment")
        parsed.port
        host = (parsed.hostname or "").lower().rstrip(".")
        try:
            public = ipaddress.ip_address(host).is_global
        except ValueError:
            public = ("." in host and not host.replace(".", "").isdigit()
                      and not host.endswith((".localhost", ".local", ".internal", ".test", ".invalid")))
        if not public:
            raise ValueError("Asset base URL must use a public HTTPS host")
        return urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), parsed.path.rstrip("/") + "/", "", ""))
    except ValueError as error:
        raise ValueError("Invalid asset base URL") from error


def _validated_comparison(report):
    if not isinstance(report, dict) or report.get("version") != 1 or report.get("complete") is not True:
        raise ValueError("Expected a complete version 1 comparison report")
    config, fixtures, measurements = (report.get(key) for key in ("config", "fixtures", "measurements"))
    if not isinstance(config, dict) or not isinstance(fixtures, dict) or not isinstance(measurements, list):
        raise ValueError("Comparison config, fixtures and measurements are required")
    if config.get("network_mode") != "internet":
        raise ValueError("Comparison reports require Internet measurements")
    suites = config.get("suites")
    if (not isinstance(suites, list) or len(suites) != 3
            or any(not isinstance(suite, dict) for suite in suites)
            or [suite.get("label") for suite in suites] != ["small", "medium", "large"]):
        raise ValueError("Expected small, medium and large noise suites")
    _paper_config(config)
    counts = config["cluster_counts"]
    count_probes = config["cluster_probe_counts"]
    probe_clusters = config["probe_cluster_counts"]
    probes = config["probe_counts"]
    codecs = config["codecs"]
    expected = {
        "counts": {(c, p, "f16", False) for c in counts for p in count_probes},
        "probes": {(c, p, "f16", False) for c in probe_clusters for p in probes},
        "codecs": {(16384, 4, codec, True) for codec in codecs},
    }
    fixed = {key: _integer(config.get(value), value)
             for key, value in (("k", "k"), ("queries", "query_count"))}
    repeats = _integer(config.get("repeats"), "repeats")
    grouped = {suite["label"]: {name: [] for name in expected} for suite in suites}
    levels = []
    first = None
    for suite in suites:
        label, level = suite["label"], _number(suite.get("noise"), "noise")
        levels.append(level)
        fixture = fixtures.get(label)
        if (not isinstance(fixture, dict) or fixture.get("metric") != "source-recovery"
                or not isinstance(fixture.get("noise"), dict)
                or fixture["noise"].get("level") != level
                or not isinstance(suite.get("url"), str)):
            raise ValueError(f"Fixture and configuration must agree for {label} noise")
        if (fixture.get("query_count") != 10000 or fixture.get("count") != 35167920
                or fixture.get("dimensions") != 384):
            raise ValueError("Use the full paper corpus and 10,000 queries")
        shared = [fixture.get(key) for key in ("count", "dimensions", "query_count", "seed", "source")]
        if first is not None and shared != first:
            raise ValueError("Noise suites must use the same corpus and query count")
        first = shared
    if levels != [0.05, 0.15, 0.4]:
        raise ValueError("Use the paper noise levels")
    seen = set()
    for measurement in measurements:
        if (not isinstance(measurement, dict) or not isinstance(measurement.get("noise_label"), str)
                or measurement["noise_label"] not in grouped):
            raise ValueError("Each measurement must identify a configured noise suite")
        label = measurement["noise_label"]
        raw = measurement.get("result")
        if not isinstance(raw, dict) or raw.get("network_mode") != "internet":
            raise ValueError("Comparison measurements must use Internet downloads")
        if _asset_base(raw.get("asset_base_url")) != _asset_base(config.get("assets_base_url")):
            raise ValueError("Measurement asset host and path must match the run")
        row = _row(measurement.get("result"))
        if row.get("metric") != "source-recovery":
            raise ValueError("Comparison measurements must use source recovery")
        if (any(row[key] != value for key, value in fixed.items())
                or row["repeats"] != (1 if row["accuracy_only"] else repeats)):
            raise ValueError("Measurements must match configured k, query count and repeats")
        key = (row["clusters"], row["probes"], row["codec"], row["accuracy_only"])
        if (label, key) in seen:
            raise ValueError("Duplicate measured configuration")
        seen.add((label, key))
        memberships = measurement.get("memberships")
        if (not isinstance(memberships, list) or not memberships
                or any(not isinstance(name, str) or name not in expected for name in memberships)
                or len(set(memberships)) != len(memberships)):
            raise ValueError("Invalid measurement memberships")
        for name in memberships:
            if key not in expected[name]:
                raise ValueError(f"Unexpected {name} measurement")
            grouped[label][name].append(row)
    for label, groups in grouped.items():
        for name, rows in groups.items():
            actual = {(row["clusters"], row["probes"], row["codec"], row["accuracy_only"]) for row in rows}
            if actual != expected[name]:
                raise ValueError(f"Incomplete {label} {name} measurements")
    return grouped


def _probe_figure_16k(report, grouped):
    figure = Figure(figsize=(7, 3.3))
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 3, sharex="col")
    minimum = math.inf
    maximum = 0
    for column, suite in enumerate(report["config"]["suites"]):
        rows = sorted(grouped[suite["label"]]["probes"], key=lambda row: row["probes"])
        probes = [row["probes"] for row in rows]
        quartiles = [statistics.quantiles((t["queryMs"] / 100 for t in row["trials"]),
                                         n=4, method="inclusive") for row in rows]
        medians = [q[1] for q in quartiles]
        accuracy = [100 * statistics.mean(t["recall"] for t in row["trials"]) for row in rows]
        maximum = max(maximum, max(q[2] for q in quartiles))
        minimum = min(minimum, min(q[0] for q in quartiles))
        axes[0, column].fill_between(probes, [q[0] for q in quartiles], [q[2] for q in quartiles],
                                     color="#0072B2", alpha=.2, linewidth=0)
        axes[0, column].plot(probes, medians, marker="o", color="#0072B2")
        axes[1, column].plot(probes, accuracy, marker="o", color="#0072B2")
        padding = max(.4, (100 - min(accuracy)) * (.10 if suite["label"] == "medium" else .15))
        upper = min(100.2, max(accuracy) + .3) if suite["label"] == "small" else 100.2
        axes[1, column].set_ylim(max(0, min(accuracy) - padding), upper)
        axes[1, column].yaxis.set_major_locator(MaxNLocator(6) if suite["label"] == "small"
                                               else MaxNLocator(4, integer=True))
        axes[0, column].set_title(f"{suite['label'].capitalize()} noise")
        for axis in axes[:, column]:
            axis.grid(axis="y", color=".88", linewidth=.45)
        axes[1, column].set_xlabel("Probes")
        axes[1, column].set_xscale("linear")
        axes[1, column].set_xticks(probes, [str(value) for value in probes])
        for axis in axes[:, column]:
            axis.tick_params(axis="x", labelbottom=True, labelrotation=40, labelsize=7)
    axes[0, 0].set_ylabel("Time (ds)")
    padding = (maximum - minimum) * .1
    for axis in axes[0]:
        axis.set_ylim(max(0, minimum - padding), maximum + padding)
        axis.yaxis.set_major_locator(MaxNLocator(6))
    axes[1, 0].set_ylabel("Source recovery (%)")
    figure.tight_layout(rect=(0, 0, 1, .98), h_pad=.7, w_pad=.7)
    return figure


def _write_table(directory, name, result, caption):
    headers, cells = result
    with (directory / f"{name}.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([header.replace("\n", " ") for header in headers])
        writer.writerows(cells)
    wide = name == "latency-table"
    inline = name == "centroid-counts-table"
    environment = "minipage" if inline else ("table*" if wide else "table")
    columns = "|l|" + "c|" * (len(headers) - 1)
    padding = "3.5pt" if wide else "5pt"
    font = r"\normalsize" if wide else r"\normalsize\renewcommand{\arraystretch}{1.2}"
    lines = [r"\begin{" + environment + "}" + (r"[c]{0.43\textwidth}" if inline else ("[tbp]" if wide else "[htbp]")),
             r"\centering" + font + r"\setlength{\tabcolsep}{" + padding + "}",
             r"\arrayrulecolor{black!25}\setlength{\arrayrulewidth}{0.3pt}",
             (r"\captionof{table}{" if inline else r"\caption{") + caption + "}", r"\label{tab:" + name + "}",
             r"\begin{tabular}{" + columns + "}", r"\hline"]
    if wide:
        labels = [headers[index].split("\n")[0] for index in range(2, len(headers), 3)]
        groups = [r"\multicolumn{3}{c|}{" + label + " noise}" for label in labels]
        lines += ["Clusters & Probes & " + " & ".join(groups) + r"\\",
                  r"\cline{3-" + str(len(headers)) + "}",
                  " & & " + " & ".join([r"Recovery (\%)", "Time (ms)", "MB/query"] * len(labels)) + r"\\"]
    else:
        headings = [header.replace("\n", " ").replace("Source recovery (%)", r"(\%)")
                    .replace(" mean / p95", "") for header in headers]
        lines.append(" & ".join(headings) + r"\\")
    lines.append(r"\hline")
    for index, row in enumerate(cells):
        if index % 2 == 1:
            lines.append(r"\rowcolor{black!4}")
        lines.append(" & ".join(row) + r"\\")
        if index < len(cells) - 1:
            lines.append(r"\hline")
    lines += [r"\hline", r"\end{tabular}", r"\end{" + environment + "}"]
    (directory / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_centroid_tables(directory, report):
    samples = report['config']['centroid_setup_repeats']
    timing_note = (f"{samples:,} trials per file. Times are per-trial mean / p95; "
                   "decode includes gzip decompression.")
    _write_table(directory, "centroid-counts-table", _centroid_timing_table(report),
                 "FP16 centroid setup by cluster count. " + timing_note)
    _write_table(directory, "compression-timing-table", _centroid_timing_table(report, formats=True),
                 f"Centroid setup at 16,384 clusters: mean / p95, {samples:,} trials/file. "
                 "Decode includes gzip decompression.")
    (directory / "settings.tex").write_text(
        r"\newcommand{\BenchmarkQueryCount}{" + str(report["config"]["query_count"]) + "}\n"
        + r"\newcommand{\BenchmarkCentroidSamples}{" + str(samples) + "}\n", encoding="utf-8")


def export_figures(report: dict, output: Path):
    """Export the paper tables and probe graph from either completed report type."""
    centroid_only = (isinstance(report, dict) and isinstance(report.get("config"), dict)
                     and report["config"].get("centroid_only"))
    if centroid_only:
        if report.get("version") != 1 or report.get("complete") is not True or not report.get("config", {}).get("centroid_only"):
            raise ValueError("Expected a complete centroid-only report")
        _paper_config(report["config"])
        counts = report["config"]["cluster_counts"]
        if report["config"].get("centroid_setup_repeats") != 1000:
            raise ValueError("Centroid setup requires 1,000 trials per file")
        expected = {(count, "f16") for count in counts} | {(16384, codec) for codec in CODECS}
        setups = report.get("centroid_setups", [])
        if len(setups) != len(expected) or {(row["clusters"], row["codec"]) for row in setups} != expected:
            raise ValueError("Incomplete centroid setups")
        for row in setups:
            _centroid_samples(row)
            if row["centroid_setup_repeats"] != report["config"]["centroid_setup_repeats"]:
                raise ValueError("Centroid sample count does not match the run")
    else:
        grouped = _validated_comparison(report)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".paper-figures-", dir=output.parent) as temporary:
        staged = Path(temporary) / "figures"
        staged.mkdir()
        with matplotlib.rc_context(STYLE):
            _write_centroid_tables(staged, report)
            if not centroid_only:
                _write_table(staged, "latency-table", _latency_comparison_table(report, grouped),
                             "FP16 source recovery@1, vector retrieval latency and downloaded payload across cluster counts and noise levels. "
                             f"Queries per noise level: {report['config']['query_count']:,}. "
                             "Times are mean / p95.")
                groups = {suite["label"]: grouped[suite["label"]]["codecs"] for suite in report["config"]["suites"]}
                _write_table(staged, "compression-table", _compression_table(groups),
                             "Source recovery@1 by centroid format at 16,384 clusters and four probes. "
                             f"{report['config']['query_count']:,} queries per noise level.")
                figure = _probe_figure_16k(report, grouped)
                figure.savefig(staged / "probes-c16384.pdf", format="pdf",
                               metadata={"CreationDate": None, "ModDate": None})
        if output.exists():
            raise FileExistsError(f"Refusing to overwrite {output}")
        staged.rename(output)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, help="Completed paper comparison or centroid setup report")
    parser.add_argument("--output", type=Path, required=True, help="New directory for LaTeX/CSV tables and the probe graph PDF")
    args = parser.parse_args()
    try:
        report = json.loads(args.report.read_text())
        export_figures(report, args.output)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Cannot export figures: {error}\n")
    print(f"Figures saved to {args.output.resolve()}")


if __name__ == "__main__":
    main()
