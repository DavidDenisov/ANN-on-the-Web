const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { spawnSync } = require('node:child_process');
const test = require('node:test');
const vm = require('node:vm');

if (!vm.SourceTextModule) {
  test('benchmark aggregation and cancellation', () => {
    const run = spawnSync(process.execPath, ['--experimental-vm-modules', __filename], { encoding: 'utf8', timeout: 10000 });
    assert.equal(run.status, 0, run.stdout + run.stderr);
  });
} else {
  async function fixture({ signal, abortAfter } = {}) {
    const retrieval = await import('../client_wiki/assets/retrieval.mjs');
    const context = vm.createContext({ performance, AbortController, URL, setTimeout: done => done() });
    let calls = 0;
    const policies = [];
    const trial = { sample: 0, setup_id: 'setup', bytes: 12, downloadMs: 2, decodeMs: 1,
      requested_url: 'https://example.org/centroid.gz' };
    const exports = { ...retrieval,
      loadCentroids: async (index, codec, options) => {
        options.signal?.throwIfAborted();
        policies.push(options);
        return { centroids: new Float32Array(), requestedUrl: trial.requested_url, bytes: 12,
          setupId: 'setup', setupTrials: [trial] };
      },
      searchVectors: async (index, query, options) => {
        options.signal?.throwIfAborted();
        policies.push(options);
        calls++;
        if (calls === abortAfter) signal.abort(new Error('stopped by caller'));
        return { results: [{ id: calls % 2 ? 1 : 2 }], selectedClusters: [0], candidates: 5,
          queryMs: calls % 2 ? 2 : 4, bytes: 20,
          downloads: [{ cluster_id: 0, requested_url: 'https://example.org/cluster.gz',
            resolved_url: 'https://example.org/cluster.gz', bytes: 20, downloadMs: 1, retries: 0 }] };
      },
    };
    const imports = new vm.SyntheticModule(Object.keys(exports), function () {
      for (const [name, value] of Object.entries(exports)) this.setExport(name, value);
    }, { context });
    const source = fs.readFileSync(path.join(__dirname, '../client_wiki/assets/benchmark.mjs'), 'utf8');
    const module = new vm.SourceTextModule(source, { context });
    await module.link(() => imports);
    await module.evaluate();
    return { benchmark: module.namespace.benchmark, policies, get calls() { return calls; },
      index: { dimensions: 384, count: 35167920, clusters: 16384 },
      suite: { dimensions: 384, count: 35167920, metric: 'source-recovery', truth_k: 1, query_count: 10000,
        queriesData: new Float32Array(384 * 10000), truthData: new Uint32Array(10000).fill(1),
        id: 'fixture', noise: { level: .4 } },
      setup: { clusters: 16384, codec: 'f16', centroid_setup_id: 'setup', centroidTrials: [trial] } };
  }

  for (const accuracyOnly of [false, true]) {
    test(`successful ${accuracyOnly ? 'accuracy-only' : 'timed'} benchmark aggregates all queries`, async () => {
      const f = await fixture();
      const result = await f.benchmark(f.index, f.suite, { accuracyOnly, centroidSetup: f.setup });
      assert.equal(f.calls, 10000);
      assert.equal(result.trials.length, 10000);
      assert.equal(result.recall, .5);
      assert.equal(result.candidatesMean, 5);
      assert.equal(result.bytesPerQuery, 20);
      assert.equal(result.centroid_setup_id, 'setup');
      assert.equal(result.cluster_fetch_cache, accuracyOnly ? 'force-cache' : 'no-store');
      assert.ok(f.policies.slice(1).every(policy => policy.cache === result.cluster_fetch_cache));
      assert.equal(Object.hasOwn(result, 'queryMs'), !accuracyOnly);
      assert.equal(Object.hasOwn(result.trials[0], 'queryMs'), !accuracyOnly);
      assert.equal(Object.hasOwn(result.trials[0].downloads[0], 'downloadMs'), !accuracyOnly);
      if (!accuracyOnly) {
        assert.equal(result.queryMs.mean, 3);
        assert.equal(result.queryMs.p95, 4);
      }
    });
  }

  test('mismatched centroid samples fail before any query runs', async () => {
    const f = await fixture();
    await assert.rejects(f.benchmark(f.index, f.suite, {
      centroidSetup: { ...f.setup, centroidTrials: [{ ...f.setup.centroidTrials[0], bytes: 13 }] },
    }), /do not match/);
    assert.equal(f.calls, 0);
  });

  test('caller cancellation reaches centroid loading and the query loop', async () => {
    const signal = new AbortController();
    const f = await fixture({ signal, abortAfter: 5 });
    await assert.rejects(f.benchmark(f.index, f.suite, { signal: signal.signal }), error => error === signal.signal.reason);
    assert.equal(f.calls, 5);
    assert.ok(f.policies.every(policy => policy.signal === signal.signal));
    const aborted = await fixture();
    await assert.rejects(aborted.benchmark(aborted.index, aborted.suite, { signal: AbortSignal.abort() }), { name: 'AbortError' });
    assert.equal(aborted.calls, 0);
  });
}
