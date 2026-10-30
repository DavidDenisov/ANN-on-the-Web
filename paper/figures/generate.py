"""Build compact figures/tables from the retained paper measurements."""

import csv
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph
from reportlab.lib.styles import ParagraphStyle

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent


def read(path):
    with (ROOT / path).open(newline="") as stream:
        return list(csv.DictReader(stream))


rows = read("results/retrieval-summary.csv")
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
lines = []
for i, row in enumerate(count_rows):
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
print("Generated compact tables, a two-panel probe plot, and architecture PDF.")


c = canvas.Canvas(str(OUT / 'lab-example.pdf'), pagesize=(246, 210))
c.setTitle('Simplified clustering lab: settings and one-versus-four-probe comparison')
c.setFillColorRGB(.96, .97, .98)
c.rect(1.5, 1.5, 243, 137.5, fill=1, stroke=0)
c.setStrokeColorRGB(.7, .7, .7)
c.roundRect(.5, .5, 245, 209, 3, stroke=1, fill=0)
c.setFillColorRGB(.12, .23, .35)
c.setFont('Helvetica-Bold', 7)
c.drawString(9, 197, 'CLUSTERING LAB')
c.setFont('Helvetica', 7)
c.drawRightString(237, 197, '16k clusters / FP16')
c.setFillColorRGB(.25, .25, .25)
c.setFont('Helvetica', 6.7)
c.drawString(9, 181, 'Checks: cluster count | probes | centroid compression')
c.setFont('Helvetica', 7.5)
c.drawString(9, 165, '10 queries | High noise | Top 1')
c.drawString(9, 149, 'Probe sweep: 1, 4')
c.setStrokeColorRGB(.7, .7, .7)
c.roundRect(181, 143, 56, 18, 3, stroke=1, fill=0)
c.setFillColorRGB(.12, .23, .35)
c.setFont('Helvetica-Bold', 7)
c.drawCentredString(209, 149, 'Run check')
c.line(9, 139, 237, 139)
c.drawString(9, 127, 'COMPARISON')
c.setFont('Helvetica', 7)
c.drawRightString(176, 127, '1 probe')
c.drawRightString(237, 127, '4 probes')
# Values come from the retained ten-query lab screenshot, not the paper benchmark.
for y, (label, one, four) in zip((109, 93, 77, 61, 45), [
        ('Candidates checked', '2,370', '9,226'),
        ('Source recovery@1', '70%', '100%'),
        ('Mean retrieval time', '65.1 ms', '192.6 ms'),
        ('95th percentile retrieval time', '96.4 ms', '328.1 ms'),
        ('Vector download / query', '1.68 MB', '6.56 MB')]):
    c.setFillColorRGB(.15, .15, .15)
    c.setFont('Helvetica', 7)
    assert c.stringWidth(label, 'Helvetica', 7) < 128
    c.drawString(9, y, label)
    c.setFont('Helvetica-Bold', 8)
    c.drawRightString(176, y, one)
    c.drawRightString(237, y, four)
c.setFillColorRGB(.3, .3, .3)
c.setFont('Helvetica', 6.1)
c.drawString(9, 12, 'Ten-query illustration; embedding excluded.')
c.showPage()
c.save()
print('Created the single-column lab panel from the retained ten-query screenshot.')

from html import unescape
from urllib.parse import quote
from xml.sax.saxutils import escape

example = json.loads((ROOT / 'results/query-example.json').read_text())
startup = json.loads((ROOT / 'results/startup-summary.json').read_text())
assert startup['clusters'] == 32768 and startup['codec'] == 'sq6' and startup['samples'] == 100
passage = example['firstResult']
wiki_url = 'https://en.wikipedia.org/wiki/' + quote(unescape(passage['title']).replace(' ', '_'), safe="!'()*-._~")
c = canvas.Canvas(str(OUT / 'search-example.pdf'), pagesize=(246, 210))
c.setTitle('Example query and passage with startup, query stages, and directly measured total')
c.setFillColorRGB(.96, .97, .98)
c.rect(1.5, 1.5, 243, 95.5, fill=1, stroke=0)
c.setStrokeColorRGB(.7, .7, .7)
c.roundRect(.5, .5, 245, 209, 3, stroke=1, fill=0)
c.saveState()
c.translate(0, 70)
c.setFillColorRGB(.12, .23, .35)
c.setFont('Helvetica-Bold', 7)
c.drawString(9, 127, 'EXAMPLE QUERY')
c.setFont('Helvetica', 7)
c.drawRightString(237, 127, '32k clusters / SQ6')
query_style = ParagraphStyle('query', fontName='Helvetica', fontSize=9, leading=11)
query = Paragraph(escape(example['inputs']['query']), query_style)
_, height = query.wrap(228, 26)
assert height <= 26
query.drawOn(c, 9, 117 - height)
c.setFillColorRGB(.25, .25, .25)
c.setFont('Helvetica', 6.3)
c.drawString(9, 99, f"{example['inputs']['k']} results | {example['candidates']:,} checked | "
             f"{example['inputs']['probes']} clusters | {example['vectorBytes']/1e6:.2f} MB vectors")
c.setFillColorRGB(.12, .23, .35)
c.setFont('Helvetica-Bold', 9)
title = '1  ' + unescape(passage['title'])
c.drawString(9, 85, title)
c.setFillColorRGB(.05, .28, .55)
c.setFont('Helvetica', 6.8)
wiki_label = wiki_url.removeprefix('https://')
wiki_width = c.stringWidth(wiki_label, 'Helvetica', 6.8)
assert wiki_width + c.stringWidth(title, 'Helvetica-Bold', 9) + 18 < 228
c.drawRightString(237, 85, wiki_label)
c.linkURL(wiki_url, (237 - wiki_width, 83, 237, 93), relative=1, thickness=0)
snippet = ' '.join(passage['text'].split()[:24]).rstrip(' ,.;:') + '...'
body_style = ParagraphStyle('snippet', fontName='Helvetica', fontSize=8, leading=9.5)
body = Paragraph(escape(snippet), body_style)
_, height = body.wrap(228, 36)
assert height <= 36
body.drawOn(c, 9, 74 - height)
c.setFillColorRGB(.25, .25, .25)
c.setFont('Helvetica', 6.7)
c.drawString(9, 35.5, f"Passage {passage['id']} | Squared L2 distance: {passage['distance']:.4f}")
c.restoreState()
c.setStrokeColorRGB(.7, .7, .7)
c.line(9, 97, 237, 97)
c.setFillColorRGB(.12, .23, .35)
c.setFont('Helvetica-Bold', 7)
c.drawString(9, 86, 'MEASURED CLIENT COSTS')
c.setFont('Helvetica', 6.7)
c.drawRightString(176, 86, 'Mean')
c.drawRightString(237, 86, '95th percentile')
for y, (label, detail, stats, scale, precision, unit) in zip((70, 56, 42, 28, 14), [
        ('Startup (once; phases overlap)', 'Load model, runtime + 32k/SQ6 centroids', startup, 1000, 2, 's'),
        ('Encode query as vector', 'Displayed question', example['embedding'], 1, 1, 'ms'),
        ('Fetch + rank candidate vectors', '32k/SQ6, four probes, top 10', example['retrieval'], 1, 1, 'ms'),
        ('Load text for top 10 results', 'Download + decompress + decode', example['text'], 1, 1, 'ms'),
        ('Complete query (measured)', 'Embedding + vectors + text; excludes DOM', example['total'], 1, 1, 'ms')]):
    c.setFillColorRGB(.15, .15, .15)
    c.setFont('Helvetica', 7)
    assert c.stringWidth(label, 'Helvetica', 7) < 128
    c.drawString(9, y, label)
    c.setFillColorRGB(.3, .3, .3)
    c.setFont('Helvetica', 6)
    assert c.stringWidth(detail, 'Helvetica', 6) < 128
    c.drawString(9, y - 7.5, detail)
    c.setFillColorRGB(.15, .15, .15)
    c.setFont('Helvetica-Bold', 8)
    c.drawRightString(176, y, f"{stats['mean_ms'] / scale:.{precision}f} {unit}")
    c.drawRightString(237, y, f"{stats['p95_ms'] / scale:.{precision}f} {unit}")
c.showPage()
c.save()
print('Created the single-column query/result panel with reference benchmark timings.')
