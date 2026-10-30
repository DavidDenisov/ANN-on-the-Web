// Rerun the illustrated query with a warm worker and uncached vector/text loads.
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const openBrowser = require('./browser.cjs');

assert.equal(process.argv.length, 3, 'Usage: node code/benchmarks/live-query.cjs RUN_DIRECTORY');
const run = path.resolve(process.argv[2]), code = path.resolve(__dirname, '..');
const inputBytes = fs.readFileSync(path.join(run, 'inputs.json'));
const inputs = JSON.parse(inputBytes);
assert.equal(inputs.cache, 'no-store');
assert.equal(inputs.warmup_queries, 100);
assert.equal(inputs.timed_queries, 1000);
for (const name of ['results.json', 'summary.json', 'trials.jsonl']) {
  assert.ok(!fs.existsSync(path.join(run, name)), 'Refusing to overwrite measurements');
}
const files = new Map([['/index.json', fs.readFileSync(path.resolve(code, inputs.index_path))]]);
for (const name of ['embedding-worker.mjs', 'search-index.mjs', 'retrieval.mjs']) {
  files.set('/assets/' + name, fs.readFileSync(path.join(code, 'client_wiki/assets', name)));
}
const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const sources = Object.fromEntries([...files].map(([name, bytes]) => [name, sha(bytes)]));
sources['inputs.json'] = sha(inputBytes);
sources['live-query.cjs'] = sha(fs.readFileSync(__filename));
sources['browser.cjs'] = sha(fs.readFileSync(path.join(__dirname, 'browser.cjs')));

function stats(trials, field) {
  const values = trials.map(trial => trial[field]).sort((a, b) => a - b);
  assert.ok(values.every(value => Number.isFinite(value) && value > 0));
  return { samples: values.length, mean_ms: values.reduce((a, b) => a + b) / values.length,
    p95_ms: values[Math.ceil(values.length * .95) - 1] };
}

(async () => {
  const session = await openBrowser(files);
  const { browser, origin } = session;
  const ledger = fs.openSync(path.join(run, 'trials.jsonl'), 'wx');
  let ledgerOpen = true;
  try {
    const report = { measuredAt: new Date().toISOString(), cpu: os.cpus()[0].model,
      browserVersion: await browser.version(), protocol: 'repeated-live-query-warmed-no-store',
      inputs, sources, warmups: [], trials: [] };
    const page = await browser.newPage();
    await page.exposeFunction('saveTrial', trial => {
      const list = trial.phase === 'warmup' ? report.warmups : report.trials;
      assert.equal(trial.repeat, list.length);
      assert.equal(trial.vectorDownloads.length, inputs.probes);
      assert.equal(trial.textDownloads.length, new Set(trial.resultIds.map(id => Math.floor(id / 500))).size,
        'Every required text batch must be downloaded, not reused from memory');
      assert.equal(trial.centroidMs, 0);
      assert.ok(!trial.metadataUnavailable);
      list.push(trial);
      fs.writeSync(ledger, JSON.stringify(trial) + '\n');
      if (list.length % 100 === 0) console.log(trial.phase + ': ' + list.length);
    });
    await page.goto(origin);
    report.ready = await page.evaluate(async ({ origin, inputs }) => {
      const worker = new Worker(origin + '/assets/embedding-worker.mjs', { type: 'module' });
      let requestId = 0;
      async function request(payload, responseType) {
        const id = ++requestId, started = performance.now();
        return new Promise((resolve, reject) => {
          const timer = setTimeout(() => reject(new Error('Query timed out')), 120000);
          worker.onmessage = ({ data }) => {
            if (data.requestId !== id) return;
            if (data.type === 'error') { clearTimeout(timer); reject(new Error(data.message)); }
            if (data.type === responseType) { clearTimeout(timer); resolve({ ...data, queryElapsedMs: performance.now() - started }); }
          };
          worker.onerror = event => { clearTimeout(timer); reject(new Error(event.message)); };
          worker.postMessage({ ...payload, requestId: id });
        });
      }
      try {
        const ready = await request({ type: 'init', indexUrl: origin + '/index.json', codec: inputs.codec }, 'ready');
        if (!ready.crossOriginIsolated || ready.threads !== 4 || ready.clusters !== 32768) throw new Error('Unexpected runtime configuration');
        for (const [phase, count] of [['warmup', inputs.warmup_queries], ['timed', inputs.timed_queries]]) {
          for (let repeat = 0; repeat < count; repeat++) {
            const result = await request({ type: 'search', text: inputs.query,
              codec: inputs.codec, probes: inputs.probes, k: inputs.k, cache: inputs.cache }, 'result');
            const { results, ...timing } = result;
            await globalThis.saveTrial({ phase, repeat, ...timing, resultIds: results.map(item => item.id),
              firstResult: repeat === 0 && phase === 'timed' ? results[0] : undefined });
          }
        }
        return ready;
      } finally { worker.terminate(); }
    }, { origin, inputs });
    assert.equal(report.warmups.length, 100);
    assert.equal(report.trials.length, 1000);
    const summary = { measuredAt: report.measuredAt, protocol: report.protocol, inputs,
      embedding: stats(report.trials, 'inferenceMs'), retrieval: stats(report.trials, 'vectorMs'),
      text: stats(report.trials, 'metadataMs'), total: stats(report.trials, 'queryElapsedMs'),
      candidates: report.trials[0].candidates, vectorBytes: report.trials[0].bytes,
      firstResult: report.trials[0].firstResult };
    fs.writeFileSync(path.join(run, 'results.json'), JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
    fs.writeFileSync(path.join(run, 'summary.json'), JSON.stringify(summary, null, 2) + '\n', { flag: 'wx' });
    console.log(JSON.stringify({ embedding: summary.embedding, retrieval: summary.retrieval,
      text: summary.text, total: summary.total }));
    fs.closeSync(ledger); ledgerOpen = false;
    fs.unlinkSync(path.join(run, 'trials.jsonl'));
  } finally {
    if (ledgerOpen) fs.closeSync(ledger);
    await session.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
