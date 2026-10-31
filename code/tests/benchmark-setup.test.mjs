import assert from 'node:assert/strict';
import test from 'node:test';
import { gzipSync } from 'node:zlib';
import { loadSuite, verifyHostingManifest, warmIndexAssets } from '../client_wiki/assets/benchmark.mjs';

test('suite loading validates binary payloads, normalizes queries and resolves index paths', async t => {
  const base = 'https://example.org/data/';
  const suite = { version: 1, dimensions: 384, count: 35167920, query_count: 10000, truth_k: 1,
    id: 'fixture', metric: 'source-recovery', query_kind: 'noisy-passages', noise: { level: .4 },
    queries: { file: 'queries.bin', count: 10000 }, truth: { file: 'truth.bin', k: 1 },
    indexes: [8192, 16384, 32768].map(clusters => ({ clusters, path: `c${clusters}/manifest.json` })) };
  const queries = new Float32Array(384 * 10000);
  for (let i = 0; i < 10000; i++) queries[i * 384] = 2;
  const truth = Uint32Array.from({ length: 10000 }, (_, i) => i);
  let activeQueries = queries.buffer, activeTruth = truth.buffer;
  t.mock.method(globalThis, 'fetch', async url => ({ ok: true, json: async () => globalThis.structuredClone(suite),
    arrayBuffer: async () => url.endsWith('queries.bin') ? activeQueries : activeTruth }));
  const loaded = await loadSuite(base + 'suite.json');
  assert.equal(loaded.queriesData[0], 1);
  assert.equal(loaded.queriesData[9999 * 384], 1);
  assert.equal(loaded.truthData[9999], 9999);
  assert.equal(loaded.indexes[0].url, base + 'c8192/manifest.json');
  activeQueries = queries.buffer.slice(0, -4);
  await assert.rejects(loadSuite(base + 'suite.json'), /query byte length/);
  activeQueries = queries.buffer;
  activeTruth = truth.buffer.slice(0, -4);
  await assert.rejects(loadSuite(base + 'suite.json'), /ground-truth byte length/);
  activeTruth = truth.buffer;
  for (const value of [0, NaN]) {
    queries[0] = value;
    await assert.rejects(loadSuite(base + 'suite.json'), /query value|query vector/);
  }
  queries[0] = 2;
  truth[0] = suite.count;
  await assert.rejects(loadSuite(base + 'suite.json'), /target passage ID/);
});

test('deployment verification checks the bundle identity and complete host allowlist', async t => {
  const base = 'https://example.org/assets/';
  const options = { networkMode: 'internet', assetsBaseUrl: base, assetsBaseUrls: [base] };
  let manifest = { version: 1, bundle_id: 'bundle', asset_base_url: base, asset_base_urls: [base] };
  t.mock.method(globalThis, 'fetch', async url => ({ ok: true, type: 'cors', url,
    json: async () => manifest }));
  await verifyHostingManifest(base + 'routing.json', 'bundle', options);
  manifest = { ...manifest, bundle_id: 'another' };
  await assert.rejects(verifyHostingManifest(base + 'routing.json', 'bundle', options), /does not match/);
  manifest = { ...manifest, bundle_id: 'bundle', asset_base_urls: [base, 'https://other.example/'] };
  await assert.rejects(verifyHostingManifest(base + 'routing.json', 'bundle', options), /allowlist/);
});

test('warm-up completes real downloads and aborts/drains workers after a failure', async t => {
  const clusters = 8192;
  const centroids = gzipSync(Buffer.alloc(clusters * 384 * 2));
  const index = { url: 'https://example.org/index.json', clusters, dimensions: 384,
    centroids: { f16: { file: 'centroids.gz', bytes: centroids.length } },
    cluster_files: Array.from({ length: clusters }, (_, i) => ({ file: `cluster-${i}.gz`, bytes: 1 })) };
  let fail = false;
  const signals = [];
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    signals.push(options.signal);
    return new Response(url.endsWith('centroids.gz') ? centroids
      : new Uint8Array(fail && url.endsWith('cluster-0.gz') ? [] : [1]));
  });
  const result = await warmIndexAssets(index, ['f16']);
  assert.equal(result.cluster_files, clusters);
  assert.equal(result.cluster_bytes, clusters);
  assert.equal(result.centroid_files, 1);
  signals.length = 0;
  fail = true;
  const controller = new AbortController();
  await assert.rejects(warmIndexAssets(index, [], { signal: controller.signal }), /byte length/);
  assert.equal(controller.signal.aborted, false);
  assert.ok(signals.length <= 16);
  assert.ok(signals.every(signal => signal.aborted));
  signals.length = 0;
  await assert.rejects(warmIndexAssets(index, [], { signal: AbortSignal.abort() }), { name: 'AbortError' });
  assert.equal(signals.length, 0);
});
