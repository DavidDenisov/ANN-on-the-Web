"""Build compact figures/tables from the retained paper measurements."""

import csv
import gzip
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent


def read(path):
    with (ROOT / path).open(newline="") as stream:
        return list(csv.DictReader(stream))


def wilson(successes, total):
    assert isinstance(total, int) and total > 0 and 0 <= successes <= total
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return max(0, centre - radius), min(1, centre + radius)


assert math.isclose(wilson(0, 10)[0], 0, abs_tol=1e-15)
assert math.isclose(wilson(10, 10)[1], 1, abs_tol=1e-15)
assert all(math.isclose(a, b, abs_tol=1e-12) for a, b in
           zip(wilson(5, 10), (0.236593090512564, 0.763406909487436)))


rows = read("results/retrieval-summary.csv")
distribution = json.loads((ROOT / 'results/cluster-distribution.json').read_text())
with gzip.open(ROOT / 'results/cluster-distribution-inputs.json.gz', 'rt') as stream:
    distribution_inputs = json.load(stream)
for item, source in zip(distribution['indexes'], distribution_inputs, strict=True):
    sizes = sorted(source['cluster_sizes'])
    payloads = sorted(source['cluster_bytes'])
    assert item['clusters'] == source['clusters'] == len(sizes) == len(payloads)
    assert sum(sizes) == item['count'] == 35167920
    for values, key in [(sizes, 'cluster_passages'), (payloads, 'cluster_gzip_bytes')]:
        expected = dict(min=values[0], max=values[-1], mean=sum(values) / len(values),
                        p50=values[math.ceil(len(values) * .5) - 1],
                        p95=values[math.ceil(len(values) * .95) - 1])
        assert item[key] == expected
    assert item['four_probe_upper_bound_bytes'] == sum(payloads[-4:])
supplement_rows = read("results/32k-summary.csv")
assert len(supplement_rows) == 12
assert all(r['clusters'] == '32768' and r['noise'] == 'large'
           and int(r['queries']) == 10000 for r in supplement_rows)
setups = {(int(r["clusters"]), r["codec"]): r for r in
          read("results/centroid-summary.csv")}
assert len(setups) == 8 and all(int(r["samples"]) == 1000 for r in setups.values())
new_setups = read("results/32k-centroid-summary.csv")
assert len(new_setups) == 6 and all(r['clusters'] == '32768' and int(r['samples']) == 1000
                                   for r in new_setups)
setups.update({(32768, r['codec']): r for r in new_setups})
intervals = []
for row in supplement_rows:
    total = int(row['queries'])
    successes = round(float(row['source_recovery']) * total)
    assert math.isclose(successes / total, float(row['source_recovery']), abs_tol=1e-12)
    low, high = wilson(successes, total)
    intervals.append(dict(clusters=32768, probes=int(row['probes']), codec=row['codec'],
                          queries=total, successes=successes, recovery=successes / total,
                          wilson95_low=low, wilson95_high=high))
with (ROOT / 'results/recovery-intervals.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, fieldnames=list(intervals[0]))
    writer.writeheader()
    writer.writerows(intervals)
default, = [r for r in intervals if r['codec'] == 'sq6' and r['probes'] == 4]
sq4 = setups[32768, 'sq4']
sq4_mbps = int(sq4['centroid_bytes']) * 8 / float(sq4['download_mean_ms']) / 1000
(ROOT / 'results/review-derived.tex').write_text(
    '% Derived from retained summaries; regenerate with paper/figures/generate.py.\n'
    + rf'\newcommand{{\DefaultRecoveryLow}}{{{100 * default["wilson95_low"]:.2f}}}' + '\n'
    + rf'\newcommand{{\DefaultRecoveryHigh}}{{{100 * default["wilson95_high"]:.2f}}}' + '\n'
    + rf'\newcommand{{\SQFourMbps}}{{{sq4_mbps:.0f}}}' + '\n')
probes = [1, 2, 3, 4, 6, 8, 10]
probe_rows = read("figures/probe-32k-summary.csv")
assert [int(r['probes']) for r in probe_rows] == probes
assert all(int(r['queries']) == 10000 and r['clusters'] == '32768'
           and r['codec'] == 'f16' for r in probe_rows)
for row in probe_rows:
    source, = [r for r in supplement_rows if r['codec'] == 'f16' and r['probes'] == row['probes']]
    assert all(float(row[k]) == float(source[k]) for k in
               ('source_recovery', 'query_mean_ms', 'query_p95_ms', 'bytes_per_query'))
    assert 0 <= float(row['query_p25_ms']) <= float(row['query_median_ms']) <= float(row['query_p75_ms'])

plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.labelsize": 9,
                     "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42})
fig, axes = plt.subplots(2, 1, figsize=(3.4, 2.7), sharex=True, layout="constrained")
axes[0].fill_between(probes, [float(r['query_p25_ms']) for r in probe_rows],
                     [float(r['query_p75_ms']) for r in probe_rows], color='#0072B2', alpha=.2, linewidth=0)
axes[0].plot(probes, [float(r['query_median_ms']) for r in probe_rows], color='#0072B2', marker='o', markersize=3)
axes[1].plot(probes, [100 * float(r['source_recovery']) for r in probe_rows],
             color='#0072B2', marker='o', markersize=3)
recovery = [float(r['source_recovery']) for r in probe_rows]
bounds = [wilson(round(p * 10000), 10000) for p in recovery]
axes[1].errorbar(probes, [100 * p for p in recovery],
                 yerr=[[100 * (p - b[0]) for p, b in zip(recovery, bounds)],
                       [100 * (b[1] - p) for p, b in zip(recovery, bounds)]],
                 fmt='none', ecolor='#0072B2', capsize=2)
for axis in axes:
    axis.set_xticks(probes)
    axis.grid(axis="y", color="0.88", linewidth=.5)
axes[0].set_ylabel("Retrieval (ms)")
axes[0].set_ylim(bottom=0)
axes[1].set_ylabel("Recovery@1 (%)")
axes[1].set_ylim(80, 100.5)
axes[1].set_xlabel("Clusters probed")
fig.savefig(OUT / "probe-large.pdf")
plt.close(fig)

count_rows = sorted((r for r in rows if r['noise'] == 'large' and 'counts' in r['checks'].split(';')),
                    key=lambda r: (int(r['clusters']), int(r['probes'])))
assert [(int(r['clusters']), int(r['probes'])) for r in count_rows] == [
    (c, n) for c in (8192, 16384, 32768) for n in (1, 4, 8)]
tradeoff_rows = {(int(r['clusters']), int(r['probes'])): r for r in count_rows}
for row in probe_rows:
    key = (32768, int(row['probes']))
    if key in tradeoff_rows:
        assert all(float(row[k]) == float(tradeoff_rows[key][k])
                   for k in ('source_recovery', 'bytes_per_query'))
    tradeoff_rows[key] = row
assert set(tradeoff_rows) == ({(c, p) for c in (8192, 16384) for p in (1, 4, 8)}
                              | {(32768, p) for p in probes})
assert all(r['codec'] == 'f16' and r['noise'] == 'large' and int(r['queries']) == 10000
           for r in tradeoff_rows.values())
fig, axis = plt.subplots(figsize=(3.4, 2.45), layout='constrained')
for clusters, color, marker in [(8192, '#D55E00', 's'), (16384, '#009E73', '^'), (32768, '#0072B2', 'o')]:
    selected = [tradeoff_rows[key] for key in sorted(tradeoff_rows) if key[0] == clusters]
    payload = [float(r['bytes_per_query']) / 1e6 for r in selected]
    recovery = [float(r['source_recovery']) for r in selected]
    bounds = [wilson(round(p * 10000), 10000) for p in recovery]
    axis.errorbar(payload, [100 * p for p in recovery],
                  yerr=[[100 * (p - b[0]) for p, b in zip(recovery, bounds)],
                        [100 * (b[1] - p) for p, b in zip(recovery, bounds)]],
                  color=color, marker=marker, markersize=3.5, linewidth=.9, capsize=2,
                  label=f'{clusters:,} clusters')
    for x, p, row in zip(payload, recovery, selected):
        offset = -13 if clusters == 16384 or row['probes'] == '1' else 9
        axis.annotate(row['probes'], (x, 100 * p), xytext=(0, offset), textcoords='offset points',
                      ha='center', fontsize=7, color=color)
axis.set_xlabel('Mean vector payload (MB/query)')
axis.set_ylabel('Source recovery@1 (%)')
axis.set_ylim(80, 101.2)
axis.set_xlim(left=0)
axis.grid(axis='y', color='0.88', linewidth=.5)
axis.legend(loc='lower right', fontsize=7, frameon=False)
fig.savefig(OUT / 'recovery-bytes.pdf')
plt.close(fig)
lines = []
for i, row in enumerate(r for r in count_rows if r['probes'] == '4'):
    if i % 2:
        lines.append(r'\rowcolor{black!4}')
    lines.append(f"{int(row['clusters'])} & {int(row['probes'])} & {100 * float(row['source_recovery']):.1f} & "
                 f"{float(row['query_mean_ms']):.0f} / {float(row['query_p95_ms']):.0f} & "
                 f"{float(row['bytes_per_query']) / 1e6:.2f}" + r' \\ \hline')
(OUT / 'large-count-rows.tex').write_text('\n'.join(lines) + '\n')

lines = []
for i, (codec, label) in enumerate([('f16', 'FP16'), ('sq8', 'SQ8'), ('sq6', 'SQ6'), ('sq4', 'SQ4'), ('int8', 'INT8'), ('pq4', 'PQ4')]):
    matches = [r for r in supplement_rows if r['codec'] == codec and r['probes'] == '4']
    assert len(matches) == 1
    setup = setups[32768, codec]
    if i % 2:
        lines.append(r'\rowcolor{black!4}')
    lines.append(f"{label} & {100 * float(matches[0]['source_recovery']):.1f} & "
                 f"{int(setup['centroid_bytes']) / 1e6:.2f} & "
                 f"{float(setup['download_mean_ms']):.0f} / {float(setup['download_p95_ms']):.0f} & "
                 f"{float(setup['decode_mean_ms']):.0f} / {float(setup['decode_p95_ms']):.0f}" + r' \\ \hline')
(OUT / 'large-compression-rows.tex').write_text('\n'.join(lines) + '\n')

c = canvas.Canvas(str(OUT / "architecture.pdf"), pagesize=(246, 288))
c.setTitle("Startup and search: local processing and selected file downloads")
c.setLineWidth(.7)


def box(y, number, title, detail):
    c.setFillColorRGB(.96, .97, .98)
    c.setStrokeColorRGB(.35, .38, .42)
    c.roundRect(4, y, 156, 25, 3, fill=1, stroke=1)
    c.setFillColorRGB(.12, .23, .35)
    c.circle(14, y + 12.5, 6.5, fill=1, stroke=0)
    c.setFillColorRGB(1, 1, 1)
    c.setFont("Helvetica-Bold", 8)
    c.drawCentredString(14, y + 9.7, str(number))
    c.setFillColorRGB(.1, .1, .1)
    c.setFont("Helvetica-Bold", 8.2)
    assert c.stringWidth(title, 'Helvetica-Bold', 8.2) < 129
    c.drawString(25, y + 16, title)
    c.setFont("Helvetica", 7.2)
    assert c.stringWidth(detail, 'Helvetica', 7.2) < 129
    c.drawString(25, y + 6, detail)


def arrow(x1, y1, x2, y2):
    c.line(x1, y1, x2, y2)
    angle = math.atan2(y2 - y1, x2 - x1)
    for delta in (-.45, .45):
        c.line(x2, y2, x2 - 4 * math.cos(angle + delta), y2 - 4 * math.sin(angle + delta))


c.setStrokeColorRGB(.7, .7, .7)
c.roundRect(.5, .5, 245, 287, 3, fill=0, stroke=1)
c.setFillColorRGB(.96, .97, .98)
c.rect(1.5, 1.5, 243, 17, fill=1, stroke=0)
c.setFillColorRGB(.12, .23, .35)
c.setFont("Helvetica-Bold", 8.2)
c.drawString(4, 277, "SEARCH IN THE BROWSER")
c.setFont("Helvetica-Bold", 7.8)
c.drawString(181, 277, "STATIC CDN")
c.setFont("Helvetica", 6.7)
c.drawString(4, 267, "Query and embedding stay local")
c.drawString(181, 267, "Fetched files")
c.setStrokeColorRGB(.7, .7, .7)
c.setDash(2, 2)
c.line(173, 20, 173, 260)
c.setDash()
c.setFillColorRGB(1, 1, 1)
c.roundRect(4, 211, 156, 14, 3, fill=1, stroke=1)
c.setFillColorRGB(.12, .23, .35)
c.setFont("Helvetica", 6.9)
question = "How do plants turn sunlight into energy?"
assert c.stringWidth(question, 'Helvetica', 6.9) < 144
c.drawCentredString(82, 215.5, question)
for y, number, title, detail in [
        (232, 0, "Download model and centroids", "Once, before queries"),
        (176, 1, "Embed query", "Text -> normalized query vector"),
        (141, 2, "Select clusters to probe", "Nearest centroids by L2 distance"),
        (106, 3, "Fetch selected clusters", "Download passage IDs + vectors"),
        (71, 4, "Rank candidates", "Renormalize vectors; rank by L2"),
        (36, 5, "Fetch text & display", "Only batches for top matches")]:
    box(y, number, title, detail)
c.setStrokeColorRGB(.3, .3, .3)
arrow(82, 232, 82, 225)
for y in (211, 176, 141, 106, 71):
    arrow(82, y, 82, y - 10)

for y, title, detail in [(231, "Model + centroids", ""),
                         (109, "Selected clusters", "IDs + FP16 vectors"),
                         (40, "Text batches", "Top matches only")]:
    for offset in (5.4, 3.6, 1.8, 0):
        c.setFillColorRGB(1, 1, 1)
        c.setStrokeColorRGB(.35, .45, .55)
        c.roundRect(190 + offset, y + offset, 23, 26, 1, fill=1, stroke=1)
    c.setStrokeColorRGB(.4, .6, .75)
    for line_y in (y + 7, y + 12, y + 17):
        c.line(194, line_y, 209, line_y)
    c.setFillColorRGB(.12, .23, .35)
    c.setFont("Helvetica-Bold", 6.7)
    c.drawCentredString(209, y - 10, title)
    c.setFont("Helvetica", 6.2)
    c.drawCentredString(209, y - 18, detail)
for y in (244.5, 118.5, 48.5):
    c.setStrokeColorRGB(.05, .35, .62)
    c.setLineWidth(1)
    arrow(184, y, 160, y)
    c.setFillColorRGB(1, 1, 1)
    c.rect(161, y + 3, 22, 8, fill=1, stroke=0)
    c.setFillColorRGB(.05, .35, .62)
    c.setFont("Helvetica", 6)
    c.drawCentredString(172, y + 5, "fetch")
c.setFillColorRGB(.3, .3, .3)
c.setFont("Helvetica", 6.8)
c.drawString(7, 7, "Reuse the model and centroids across queries.")
c.showPage()
c.save()
print("Generated compact tables, probe and recovery--bytes plots, and architecture PDF.")
