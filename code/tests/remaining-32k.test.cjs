const assert = require('node:assert/strict');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const test = require('node:test');
const moduleUrl = pathToFileURL(path.resolve(__dirname, '../client_wiki/assets/benchmark.mjs'));

test('32k supplement omits completed settings and shares the SQ6 timing/recovery run', async () => {
  const { paperPlan, remaining32kPlan } = await import(moduleUrl);
  assert.equal(paperPlan().length, 19);
  const plan = remaining32kPlan();
  assert.equal(new Set(plan.map(p => `${p.codec}|${p.probes}|${p.accuracy_only}`)).size, 9);
  assert.ok(plan.every(p => p.clusters === 32768));
  assert.deepEqual(plan.filter(p => p.codec === 'f16').map(p => p.probes), [2, 3, 6, 10]);
  assert.deepEqual(plan.filter(p => p.accuracy_only).map(p => p.codec), ['sq8', 'sq4', 'int8', 'pq4']);
  assert.equal(plan.filter(p => p.codec === 'sq6').length, 1);
  assert.equal(plan.find(p => p.codec === 'sq6').accuracy_only, false);
});

test('32k whitelist accepts the missing probes and keeps rejecting unplanned settings', async () => {
  const { benchmark } = await import(moduleUrl);
  const suite = { dimensions: 384, count: 35167920, metric: 'source-recovery', truth_k: 1, query_count: 10000 };
  const index = { clusters: 32768, dimensions: 384, count: suite.count, url: 'https://example.org/index.json',
    centroids: { f16: { file: 'f16.gz', bytes: 1 }, sq6: { file: 'sq6.gz', bytes: 1 } } };
  const originalFetch = global.fetch;
  const requests = [];
  global.fetch = async (url, options) => { requests.push(options.cache); return new Response('', { status: 404 }); };
  try {
    await assert.rejects(benchmark(index, suite, { probes: 5 }), /reported paper configuration/);
    assert.equal(requests.length, 0);
    for (const options of [{ probes: 2 }, { probes: 4, codec: 'sq6' }]) {
      await assert.rejects(benchmark(index, suite, options), /404/);
    }
    assert.deepEqual(requests, ['no-store', 'no-store']);
  } finally { global.fetch = originalFetch; }
});
