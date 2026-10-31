import { loadSuite, benchmark, paperPlan } from './benchmark.mjs';
import { loadIndex, CODEC_LABELS as codecNames } from './retrieval.mjs';

const $ = id => document.getElementById(id);
const results = { counts: [], probes: [], codecs: [] };
let suite;
let indexes = [];
let busy = false;

const percent = value => `${(100 * value).toFixed(1)}%`;
const ms = value => `${value.toFixed(2)} ms`;
function bytes(value) {
  if (value < 1000) return `${Math.round(value)} B`;
  if (value < 1e6) return `${(value / 1000).toFixed(1)} kB`;
  return `${(value / 1e6).toFixed(2)} MB`;
}

function setBusy(value) {
  busy = value;
  document.querySelectorAll('input, select, button').forEach(element => {
    element.disabled = value || (!suite && element.id !== 'suiteUrl');
  });
  $('experiments').setAttribute('aria-busy', String(value));
  $('labProgress').hidden = !value;
}

function status(message, error = false) {
  $('labStatus').textContent = message;
  $('labStatus').classList.toggle('error', error);
}

function tableRow(values) {
  const row = document.createElement('tr');
  values.forEach(value => {
    const cell = document.createElement('td');
    cell.textContent = value;
    row.append(cell);
  });
  return row;
}

function render(check, emptyMessage = 'Ready to measure.') {
  const rows = results[check];
  if (!rows.length) {
    const row = tableRow([emptyMessage]);
    row.className = 'empty';
    row.firstChild.colSpan = check === 'probes' ? 5 : 6;
    $(check + 'Rows').replaceChildren(row);
    return;
  }
  rows.sort((a, b) => {
    if (check === 'counts') return a.clusters - b.clusters || a.probes - b.probes;
    if (check === 'probes') return a.probes - b.probes;
    return Object.keys(codecNames).indexOf(a.codec) - Object.keys(codecNames).indexOf(b.codec) || a.probes - b.probes;
  });
  const values = rows.map(row => {
    if (check === 'codecs') {
      return [codecNames[row.codec], row.probes, bytes(row.centroidBytes), ms(row.centroidDownloadMs.mean),
        ms(row.centroidDecodeMs.mean), percent(row.recall)];
    }
    const measured = [percent(row.recall), `${row.queryMs.mean.toFixed(2)} / ${row.queryMs.p95.toFixed(2)} ms`,
      bytes(row.bytesPerQuery)];
    if (check === 'counts') {
      return [row.clusters, row.probes, Math.round(suite.count / row.clusters).toLocaleString(), ...measured];
    }
    return [row.probes, Math.round(row.candidatesMean).toLocaleString(), ...measured];
  });
  $(check + 'Rows').replaceChildren(...values.map(tableRow));

}

async function loadFixture() {
  if (busy) return;
  suite = undefined;
  setBusy(true);
  status('Loading test queries…');
  for (const check of Object.keys(results)) {
    results[check] = [];
    render(check, 'Loading test queries…');
  }
  try {
    const next = await loadSuite(new URL($('suiteUrl').value, location.href).href);
    const loaded = await Promise.all(next.indexes.map(item => loadIndex(item.url)));
    suite = next;
    indexes = loaded;
    const passageCount = new Intl.NumberFormat('en', { notation: 'compact', maximumFractionDigits: 1 }).format(suite.count);
    $('fixtureInfo').textContent = `${passageCount} passages · ${suite.query_count.toLocaleString('en-US')} test queries`;
    $('measurementNote').textContent = `Queries are indexed passage embeddings with Gaussian noise (L2 magnitude ${suite.noise.level}). Source recovery@1 counts queries whose original passage is the first result. Timings include downloads and browser processing on this device.`;
    Object.keys(results).forEach(check => render(check));
    status('Ready. Run a check to collect live measurements.');
  } catch (error) {
    status(`${error.message} Try choosing another noise level or reloading the page.`, true);
    $('fixtureInfo').textContent = 'Test queries could not be loaded.';
    Object.keys(results).forEach(check => render(check, 'Choose a query noise level to try again.'));
  } finally {
    setBusy(false);
  }
}

async function runChecks(checks) {
  if (busy || !suite) return;
  try {
    const plan = paperPlan().map(item => ({ ...item,
      memberships: item.memberships.filter(check => checks.includes(check)),
    })).filter(item => item.memberships.length);
    for (const check of checks) {
      results[check] = [];
      render(check, 'Measuring…');
    }
    // Randomize configuration order to reduce systematic first/last-run bias.
    for (let i = plan.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [plan[i], plan[j]] = [plan[j], plan[i]];
    }
    setBusy(true);
    $('labProgress').value = 0;
    $('labProgress').max = plan.length;
    for (let position = 0; position < plan.length; position++) {
      const { memberships, clusters, codec, probes, accuracy_only } = plan[position];
      const index = indexes.find(index => index.clusters === clusters);
      const label = `${index.clusters} clusters · ${probes} probed · ${codecNames[codec]}`;
      status(`${position + 1}/${plan.length}: ${label}`);
      const result = await benchmark(index, suite, { codec, probes, accuracyOnly: accuracy_only,
        onProgress: ({ completed, total }) => {
          $('labProgress').value = position + completed / total;
          status(`${position + 1}/${plan.length}: ${label} · ${completed}/${total} queries`);
        },
      });
      const { trials, centroidTrials, ...summary } = result;
      for (const check of memberships) {
        results[check].push(summary);
        render(check);
      }
      $('labProgress').value = position + 1;
    }
    status('Measurements complete.');
  } catch (error) {
    status(error.message, true);
  } finally {
    setBusy(false);
  }
}

$('suiteUrl').addEventListener('change', loadFixture);
$('runAll').addEventListener('click', () => runChecks(['counts', 'probes', 'codecs']));
document.querySelectorAll('.run-check').forEach(button => button.addEventListener('click', () => runChecks([button.dataset.check])));

loadFixture();
