// Collect only the missing 32k browser measurements; orchestrated by remaining-32k.py.
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const openBrowser = require('./browser.cjs');

async function main() {
  assert.equal(process.argv.length, 3, 'Usage: node remaining-32k.cjs RUN_DIRECTORY | --check');
  const check = process.argv[2] === '--check';
  const run = check ? null : path.resolve(process.argv[2]);
  const inputs = JSON.parse(fs.readFileSync(check ? 0 : path.join(run, 'inputs.json'), 'utf8'));
  const code = path.resolve(__dirname, '..');
  const modulePath = path.join(code, 'client_wiki/assets/benchmark.mjs');
  const { remaining32kPlan } = await import(pathToFileURL(modulePath));
  const plan = remaining32kPlan();
  assert.equal(plan.length, 9);
  assert.equal(plan.filter(item => !item.accuracy_only).length, 5);
  assert.equal(inputs.f16_setup.clusters, 32768);
  assert.equal(inputs.f16_setup.codec, 'f16');
  assert.equal(inputs.f16_setup.centroidTrials.length, 1000);
  assert.equal(inputs.fixture.noise.level, 0.4);
  assert.equal(inputs.fixture.query_count, 10000);
  if (check) {
    console.log('32k plan: FP16 probes 2/3/6/10; SQ6 at 4 probes; SQ8/SQ4/INT8/PQ4 recovery.');
    console.log('Reuse FP16 probes 1/4/8 and FP16 centroid setup; collect 5 × 10,000 timings + 4 × 10,000 accuracy checks.');
    return;
  }
  // Randomize timed configurations without mixing in cache-allowed accuracy checks.
  for (let i = 4; i > 0; i--) {
    const j = crypto.randomInt(i + 1);
    [plan[i], plan[j]] = [plan[j], plan[i]];
  }
  fs.mkdirSync(path.join(run, 'queries'));
  fs.mkdirSync(path.join(run, 'centroids'));
  const write = (name, value) => fs.writeFileSync(path.join(run, name), JSON.stringify(value) + '\n', { flag: 'wx' });
  const report = { version: 1, measuredAt: new Date().toISOString(), cpu: os.cpus()[0].model,
    fixture_id: inputs.fixture.id, sources: inputs.sources, plan, measurements: [], complete: false };
  const saveReport = () => {
    fs.writeFileSync(path.join(run, '.results.tmp'), JSON.stringify(report, null, 2) + '\n');
    fs.renameSync(path.join(run, '.results.tmp'), path.join(run, 'results.json'));
  };
  assert.ok(!fs.existsSync(path.join(run, 'results.json')), 'Refusing to overwrite results');
  saveReport();
  let session;
  try {
    session = await openBrowser(new Map(['benchmark.mjs', 'retrieval.mjs'].map(name =>
      ['/assets/' + name, fs.readFileSync(path.join(path.dirname(modulePath), name))])));
    report.browserVersion = await session.browser.version();
    const context = await session.browser.newContext();
    const page = await context.newPage();
    await page.exposeFunction('record', (name, data) => {
      write(name, data);
      if (name.startsWith('queries/')) report.measurements.push({ file: name });
      saveReport();
      console.log(`Saved ${name}`);
    });
    await page.exposeFunction('progress', message => console.log(message));
    await page.goto(session.origin);
    await page.evaluate(async ({ origin, inputs, plan }) => {
      const { loadSuite, verifyHostingManifest, warmIndexAssets, benchmark } = await import(origin + '/assets/benchmark.mjs');
      const { loadIndex, loadCentroids, searchVectors } = await import(origin + '/assets/retrieval.mjs');
      const network = inputs.network;
      await verifyHostingManifest(inputs.hosting_manifest_url, inputs.hosting_bundle_id, network);
      const suite = await loadSuite(inputs.suite_url, network);
      if (suite.id !== inputs.fixture.id || suite.noise.level !== 0.4) throw new Error('Hosted query fixture changed');
      const descriptor = suite.indexes.find(item => item.clusters === 32768);
      if (!descriptor) throw new Error('Missing 32k index');
      const index = await loadIndex(descriptor.url, network);
      if (index.clusters !== 32768) throw new Error('Wrong cluster count');
      const codecs = Object.keys(index.centroids);
      await window.progress('Warming all 32,768 cluster files and six centroid files; no measurements yet.');
      const warmup = await warmIndexAssets(index, codecs, { ...network, onProgress: p => {
        if (p.completed % 1000 === 0 || p.completed === p.total) {
          window.progress(`Asset warm-up ${p.completed}/${p.total}`);
        }
      } });
      await window.record('warmup.json', warmup);
      const setups = new Map([['f16', inputs.f16_setup]]);
      await window.record('centroids/f16.json', inputs.f16_setup);
      for (const codec of codecs.filter(codec => codec !== 'f16')) {
        await window.progress(`32k ${codec}: 1,000 centroid fetch/decode trials (browser cache bypassed).`);
        const loaded = await loadCentroids(index, codec, { ...network, setupRepeats: 1000 });
        const setup = { clusters: index.clusters, codec, centroid_setup_id: loaded.setupId,
          centroid_setup_repeats: loaded.setupTrials.length, centroidTrials: loaded.setupTrials };
        setups.set(codec, setup);
        await window.record(`centroids/${codec}.json`, setup);
      }
      // Keep timing configurations ahead of cache-allowed accuracy checks.
      for (const item of plan) {
        const name = `${item.codec}-p${item.probes}`;
        await window.progress(`${name}: three untimed query warm-ups.`);
        const { centroids } = await loadCentroids(index, item.codec, network);
        const warmups = [];
        for (let i = 0; i < 3; i++) {
          const query = suite.queriesData.subarray(i * suite.dimensions, (i + 1) * suite.dimensions);
          const found = await searchVectors(index, query, { ...network, cache: 'no-store',
            centroids, probes: item.probes, k: 1, accuracyOnly: true });
          warmups.push({ query: i, resultIds: found.results.map(result => result.id), bytes: found.bytes });
        }
        const result = await benchmark(index, suite, { ...network, codec: item.codec, probes: item.probes,
          accuracyOnly: item.accuracy_only, centroidSetup: setups.get(item.codec), onProgress: p => {
            if (p.completed % 1000 === 0) window.progress(`${name}: ${p.completed}/${p.total} queries`);
          } });
        await window.record(`queries/${name}.json`, { ...result, warmups, memberships: item.memberships,
          hosting_bundle_id: inputs.hosting_bundle_id });
      }
    }, { origin: session.origin, inputs, plan });
    assert.equal(report.measurements.length, plan.length);
    report.complete = true;
    saveReport();
  } catch (error) {
    report.error = error.message;
    saveReport();
    throw error;
  } finally { if (session) await session.close(); }
}

main().catch(error => { console.error(error); process.exitCode = 1; });
