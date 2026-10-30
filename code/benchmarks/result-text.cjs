// Warm distinct text files, then load 10,000 sampled passages in groups of ten.
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const openBrowser = require('./browser.cjs');

assert.equal(process.argv.length, 3, 'Usage: node code/benchmarks/result-text.cjs RUN_DIRECTORY');
const run = path.resolve(process.argv[2]);
const code = path.resolve(__dirname, '..');
const inputBytes = fs.readFileSync(path.join(run, 'inputs.json'));
const inputs = JSON.parse(inputBytes);
assert.equal(inputs.repeats, 1000);
assert.equal(inputs.result_ids, undefined, 'Use the varied benchmark inputs, without fixed result_ids');
const files = new Map([['/index.json', fs.readFileSync(path.resolve(code, inputs.index_path))]]);
const index = JSON.parse(files.get('/index.json'));
assert.equal(index.count, 35167920);
assert.equal(index.metadata.format, 'binary-http-v1');
assert.ok(inputs.repeats * 10 <= index.count);
const sampled = new Set();
while (sampled.size < inputs.repeats * 10) sampled.add(crypto.randomInt(index.count));
const sampledResultIds = [...sampled];
const groupsOfTen = ids => Array.from({ length: Math.ceil(ids.length / 10) }, (_, i) => ids.slice(i * 10, i * 10 + 10));
const workloads = groupsOfTen(sampledResultIds);
const representatives = new Map();
for (const id of sampledResultIds) {
  const batch = Math.floor(id / index.metadata.batch_size);
  if (!representatives.has(batch)) representatives.set(batch, id);
}
const warmups = groupsOfTen([...representatives.values()]);
const urlsFor = ids => [...new Set(ids.map(id => {
  const batch = Math.floor(id / index.metadata.batch_size);
  return new URL(index.metadata.pattern.replace('{id}', batch),
    index.metadata.base_urls[Math.floor(batch / index.metadata.files_per_deployment)]).href;
}))];
const urls = urlsFor([...representatives.values()]);
assert.ok(urls.every(url => new URL(url).protocol === 'https:'));
for (const name of ['search-index.mjs', 'retrieval.mjs']) {
  files.set('/assets/' + name, fs.readFileSync(path.join(code, 'client_wiki/assets', name)));
}
const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
const sources = Object.fromEntries([...files].map(([name, bytes]) => [name, sha(bytes)]));
sources['inputs.json'] = sha(inputBytes);
sources['result-text.cjs'] = sha(fs.readFileSync(__filename));
sources['browser.cjs'] = sha(fs.readFileSync(path.join(__dirname, 'browser.cjs')));

async function measure({ origin, workloads, warmups }) {
  const { loadSearchIndex, loadResultText } = await import(origin + '/assets/search-index.mjs');
  const { index } = await loadSearchIndex(origin + '/index.json');
  performance.setResourceTimingBufferSize(100);
  async function load(resultIds) {
    const matches = resultIds.map(id => ({ id }));
    const expectedFiles = new Set(resultIds.map(id => Math.floor(id / index.metadata.batch_size))).size;
    performance.clearResourceTimings();
    const result = await loadResultText(index, matches, {
      cache: 'no-store', signal: AbortSignal.timeout(120000),
    });
    if (result.metadataUnavailable || result.results.length !== resultIds.length || result.downloads.length !== expectedFiles
      || result.results.some((item, i) => item.id !== resultIds[i]
        || typeof item.title !== 'string' || !item.title.length || typeof item.text !== 'string' || !item.text.length)) {
      throw new Error('Incomplete result-text load; refusing to record a failed trial. '
        + JSON.stringify(result.failures));
    }
    const resources = performance.getEntriesByType('resource');
    const downloads = result.downloads.map(download => {
      const entry = resources.filter(item => item.name === download.requestedUrl).at(-1);
      return { ...download, encodedBytes: entry?.encodedBodySize > 0 ? entry.encodedBodySize : null };
    });
    return { metadataMs: result.metadataMs,
      decodeMs: downloads.reduce((sum, item) => sum + item.decodeMs, 0),
      decodedBytes: downloads.reduce((sum, item) => sum + item.bytes, 0),
      encodedBytes: downloads.every(item => item.encodedBytes !== null)
        ? downloads.reduce((sum, item) => sum + item.encodedBytes, 0) : null,
      resultCharacters: result.results.reduce((sum, item) => sum + item.title.length + item.text.length, 0),
      downloads };
  }
  for (const [phase, groups] of [['warmup', warmups], ['timed', workloads]]) {
    for (let repeat = 0; repeat < groups.length; repeat++) {
      const resultIds = groups[repeat];
      const trial = await load(resultIds);
      self.postMessage({ trial: { phase, repeat, resultIds, ...trial } });
    }
  }
  return { userAgent: navigator.userAgent, crossOriginIsolated };
}

function stats(values) {
  assert.ok(values.length && values.every(value => Number.isFinite(value) && value >= 0));
  const ordered = [...values].sort((a, b) => a - b);
  return { samples: values.length, mean_ms: values.reduce((a, b) => a + b, 0) / values.length,
    p95_ms: ordered[Math.ceil(ordered.length * .95) - 1], min_ms: ordered[0], max_ms: ordered.at(-1) };
}

(async () => {
  for (const name of ['trials.jsonl', 'results.json', 'summary.json']) {
    assert.ok(!fs.existsSync(path.join(run, name)), 'Refusing to overwrite saved measurements');
  }
  const session = await openBrowser(files);
  const { browser, origin } = session;
  let ledger;
  try {
    const results = { measuredAt: new Date().toISOString(), cpu: os.cpus()[0].model,
      browserVersion: await browser.version(), protocol: 'varied-ten-result-text-warmed-no-store',
      inputs, sources, sampledResultIds, urls, warmupLoads: warmups.length, warmups: [], trials: [] };
    const warmed = new Map();
    ledger = fs.openSync(path.join(run, 'trials.jsonl'), 'wx');
    const page = await browser.newPage();
    await page.exposeFunction('saveTextTrial', trial => {
      assert.ok(trial.phase === 'warmup' || trial.phase === 'timed');
      const records = trial.phase === 'warmup' ? results.warmups : results.trials;
      const groups = trial.phase === 'warmup' ? warmups : workloads;
      assert.equal(trial.repeat, records.length);
      assert.deepEqual(trial.resultIds, groups[trial.repeat]);
      const expectedUrls = urlsFor(trial.resultIds);
      assert.equal(trial.downloads.length, expectedUrls.length);
      assert.deepEqual(trial.downloads.map(item => item.requestedUrl).sort(), expectedUrls.sort());
      assert.ok(Number.isFinite(trial.metadataMs) && trial.metadataMs > 0);
      assert.ok(trial.downloads.every(item => item.bytes > 0 && item.decodeMs >= 0
        && Number.isFinite(item.downloadMs) && item.downloadMs > 0));
      assert.equal(trial.decodedBytes, trial.downloads.reduce((sum, item) => sum + item.bytes, 0));
      if (trial.phase === 'warmup') {
        assert.equal(results.trials.length, 0);
        for (const download of trial.downloads) {
          assert.ok(!warmed.has(download.requestedUrl), 'A text batch was warmed more than once');
          warmed.set(download.requestedUrl, download);
        }
      } else {
        assert.equal(results.warmups.length, warmups.length, 'Finish warming all batches before timing');
        for (const download of trial.downloads) {
          const warmup = warmed.get(download.requestedUrl);
          assert.ok(warmup, 'A timed text batch was not warmed');
          assert.equal(download.bytes, warmup.bytes, 'Text payload changed after warm-up');
          if (download.encodedBytes !== null && warmup.encodedBytes !== null) {
            assert.equal(download.encodedBytes, warmup.encodedBytes, 'Compressed payload changed after warm-up');
          }
        }
      }
      fs.writeSync(ledger, JSON.stringify(trial) + '\n');
      records.push(trial);
      if (records.length % 50 === 0 || records.length === groups.length) {
        console.log(`Completed ${trial.phase} ${records.length}/${groups.length} text loads`);
      }
    });
    await page.goto(origin);
    const details = await page.evaluate(async ({ origin, workloads, warmups, source }) => {
      const url = URL.createObjectURL(new Blob([
        'self.onmessage=async({data})=>{try{self.postMessage({result:await('
        + source + ')(data)});}catch(error){self.postMessage({error:error.stack||String(error)});}};',
      ], { type: 'text/javascript' }));
      const worker = new Worker(url, { type: 'module' });
      try {
        return await new Promise((resolve, reject) => {
          let saved = Promise.resolve();
          worker.onmessage = ({ data }) => {
            if (data.trial) saved = saved.then(() => globalThis.saveTextTrial(data.trial)).catch(reject);
            if (data.result) saved.then(() => resolve(data.result), reject);
            if (data.error) reject(new Error(data.error));
          };
          worker.onerror = event => reject(new Error(event.message));
          worker.postMessage({ origin, workloads, warmups });
        });
      } finally { worker.terminate(); URL.revokeObjectURL(url); }
    }, { origin, workloads, warmups, source: measure.toString() });
    assert.equal(details.crossOriginIsolated, true);
    assert.ok(!details.userAgent.includes('HeadlessChrome'));
    assert.equal(results.warmups.length, warmups.length);
    assert.equal(warmed.size, urls.length);
    assert.equal(results.trials.length, inputs.repeats);
    Object.assign(results, details);
    const summary = { measuredAt: results.measuredAt, browserVersion: results.browserVersion,
      protocol: results.protocol, results_per_trial: 10, sampled_passages: sampledResultIds.length,
      warmed_files: warmed.size, warmup_loads: results.warmups.length,
      files_per_trial: { min: Math.min(...results.trials.map(item => item.downloads.length)),
        max: Math.max(...results.trials.map(item => item.downloads.length)) },
      elapsed: stats(results.trials.map(item => item.metadataMs)),
      decode: stats(results.trials.map(item => item.decodeMs)),
      mean_decoded_bytes_per_trial: results.trials.reduce((sum, item) => sum + item.decodedBytes, 0) / results.trials.length,
      mean_encoded_bytes_per_trial: results.trials.every(item => item.encodedBytes !== null)
        ? results.trials.reduce((sum, item) => sum + item.encodedBytes, 0) / results.trials.length : null };
    fs.writeFileSync(path.join(run, 'results.json'), JSON.stringify(results, null, 2) + '\n', { flag: 'wx' });
    fs.writeFileSync(path.join(run, 'summary.json'), JSON.stringify(summary, null, 2) + '\n', { flag: 'wx' });
    fs.closeSync(ledger); ledger = undefined;
    fs.unlinkSync(path.join(run, 'trials.jsonl'));
    console.log(JSON.stringify(summary, null, 2));
  } finally {
    try { if (ledger !== undefined) fs.closeSync(ledger); }
    finally { await session.close(); }
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
