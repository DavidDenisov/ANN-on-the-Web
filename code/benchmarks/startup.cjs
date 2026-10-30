// Measure the actual client worker's cold startup, including overlapping loads.
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const openBrowser = require('./browser.cjs');

assert.equal(process.argv.length, 3, 'Usage: node code/benchmarks/startup.cjs RUN_DIRECTORY');
const run = path.resolve(process.argv[2]);
const code = path.resolve(__dirname, '..');
const inputs = JSON.parse(fs.readFileSync(path.join(run, 'inputs.json'), 'utf8'));
assert.equal(inputs.startup_trials, 100);
assert.equal(inputs.codec, 'sq6');
const files = new Map([['/index.json', fs.readFileSync(path.resolve(code, inputs.index_path))]]);
for (const name of ['embedding-worker.mjs', 'search-index.mjs', 'retrieval.mjs']) {
  files.set('/assets/' + name, fs.readFileSync(path.join(code, 'client_wiki/assets', name)));
}
const index = JSON.parse(files.get('/index.json'));
assert.ok([16384, 32768].includes(inputs.clusters ?? 16384));
assert.equal(index.clusters, inputs.clusters ?? 16384);
assert.equal(index.dimensions, 384);
assert.equal(index.count, 35167920);
assert.equal(new URL(index.centroids[inputs.codec].file).protocol, 'https:');
const sources = Object.fromEntries([...files].map(([name, buffer]) =>
  [name, crypto.createHash('sha256').update(buffer).digest('hex')]));
sources['browser.cjs'] = crypto.createHash('sha256').update(fs.readFileSync(path.join(__dirname, 'browser.cjs'))).digest('hex');

(async () => {
  for (const name of ['startups.jsonl', 'results.json', 'summary.json']) {
    assert.ok(!fs.existsSync(path.join(run, name)), 'Refusing to overwrite saved measurements');
  }
  const session = await openBrowser(files);
  const { browser, origin } = session;
  let ledger;
  try {
    ledger = fs.openSync(path.join(run, 'startups.jsonl'), 'wx');
    const results = { measuredAt: new Date().toISOString(), cpu: os.cpus()[0].model,
      browserVersion: await browser.version(), protocol: 'worker-creation-to-ready',
      index: { clusters: index.clusters, count: index.count, codec: inputs.codec }, sources, trials: [] };
    for (let trial = 0; trial < inputs.startup_trials; trial++) {
      const context = await browser.newContext();
      try {
        const page = await context.newPage();
        await page.goto(origin);
        const result = await page.evaluate(async ({ origin, codec }) => {
          if (!crossOriginIsolated) throw new Error('Startup control page is not isolated');
          const started = performance.now();
          const worker = new Worker(origin + '/assets/embedding-worker.mjs', { type: 'module' });
          let timer;
          try {
            return await new Promise((resolve, reject) => {
              timer = setTimeout(() => reject(new Error('Client startup timed out')), 120000);
              worker.onmessage = ({ data }) => {
                if (data.type === 'ready') resolve({ elapsedMs: performance.now() - started,
                  threads: data.threads, crossOriginIsolated: data.crossOriginIsolated,
                  count: data.count, clusters: data.clusters, userAgent: navigator.userAgent });
                if (data.type === 'error') reject(new Error(data.message));
              };
              worker.onerror = event => reject(new Error(event.message));
              worker.postMessage({ type: 'init', requestId: 1, indexUrl: origin + '/index.json', codec });
            });
          } finally { clearTimeout(timer); worker.terminate(); }
        }, { origin, codec: inputs.codec });
        assert.ok(Number.isFinite(result.elapsedMs) && result.elapsedMs > 0);
        assert.equal(result.crossOriginIsolated, true);
        assert.equal(result.threads, 4);
        assert.equal(result.count, index.count);
        assert.equal(result.clusters, index.clusters);
        assert.ok(!result.userAgent.includes('HeadlessChrome'));
        results.trials.push({ trial, ...result });
        fs.writeSync(ledger, JSON.stringify(results.trials.at(-1)) + '\n');
        if ((trial + 1) % 10 === 0 || trial + 1 === inputs.startup_trials) {
          console.log(`Completed ${trial + 1}/${inputs.startup_trials} cold startups`);
        }
      } finally { await context.close(); }
    }
    const ordered = results.trials.map(trial => trial.elapsedMs).sort((a, b) => a - b);
    const summary = { measuredAt: results.measuredAt, browserVersion: results.browserVersion,
      protocol: results.protocol, clusters: index.clusters, codec: inputs.codec, threads: 4,
      samples: ordered.length, mean_ms: ordered.reduce((a, b) => a + b, 0) / ordered.length,
      p95_ms: ordered[Math.ceil(ordered.length * 0.95) - 1], min_ms: ordered[0], max_ms: ordered.at(-1) };
    fs.writeFileSync(path.join(run, 'results.json'), JSON.stringify(results, null, 2) + '\n', { flag: 'wx' });
    fs.writeFileSync(path.join(run, 'summary.json'), JSON.stringify(summary, null, 2) + '\n', { flag: 'wx' });
    fs.closeSync(ledger); ledger = undefined;
    fs.unlinkSync(path.join(run, 'startups.jsonl'));
    console.log(JSON.stringify(summary, null, 2));
  } finally {
    try { if (ledger !== undefined) fs.closeSync(ledger); }
    finally { await session.close(); }
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
