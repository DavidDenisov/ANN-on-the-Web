import assert from 'node:assert/strict';
import test from 'node:test';
import { gzipSync } from 'node:zlib';
import { loadCentroids, searchVectors, loadIndex, fetchBytes } from '../client_wiki/assets/retrieval.mjs';

const dimensions = 384, clusters = 8192, length = dimensions * clusters;

function centroidBytes(codec) {
  if (codec === 'f16') {
    const values = Buffer.alloc(length * 2);
    for (let i = 0; i < length; i++) values.writeUInt16LE(0x3800, i * 2);
    return values;
  }
  const bits = { sq8: 8, sq6: 6, sq4: 4 }[codec];
  if (bits || codec === 'int8') {
    const header = Buffer.alloc(dimensions * (bits ? 8 : 4));
    for (let d = 0; d < dimensions; d++) header.writeFloatLE(.5, (bits ? dimensions + d : d) * 4);
    const codes = Buffer.alloc(length * (bits ?? 8) / 8, bits === 4 ? 0x11 : 1);
    if (bits === 6) for (let i = 0; i < codes.length; i += 3) codes.set([0x41, 0x10, 0x04], i);
    return Buffer.concat([header, codes]);
  }
  const header = Buffer.alloc(4 + dimensions * 64);
  header.writeUInt32LE(8);
  for (let offset = 4; offset < header.length; offset += 4) header.writeFloatLE(.5, offset);
  return Buffer.concat([header, Buffer.alloc(clusters * dimensions / 16)]);
}

test('all six centroid codecs decode the stored values and reject corrupt payloads', async t => {
  const files = new Map();
  const index = { url: 'https://example.org/index.json', dimensions, clusters, centroids: {} };
  for (const codec of ['f16', 'sq8', 'sq6', 'sq4', 'int8', 'pq4']) {
    const payload = gzipSync(centroidBytes(codec));
    files.set(`https://example.org/${codec}.gz`, payload);
    index.centroids[codec] = { file: `${codec}.gz`, bytes: payload.length };
  }
  t.mock.method(globalThis, 'fetch', async url => new Response(files.get(String(url))));
  for (const codec of Object.keys(index.centroids)) {
    const loaded = await loadCentroids(index, codec);
    assert.equal(loaded.centroids.length, length);
    assert.ok(loaded.centroids.every(value => value === .5), codec);
    assert.equal(loaded.setupTrials.length, 1);
  }
  files.set('https://example.org/f16.gz', gzipSync(Buffer.alloc(2)));
  index.centroids.f16.bytes = files.get('https://example.org/f16.gz').length;
  await assert.rejects(loadCentroids(index, 'f16'), /byte length/);
  files.set('https://example.org/f16.gz', Buffer.from('invalid gzip'));
  index.centroids.f16.bytes = files.get('https://example.org/f16.gz').length;
  await assert.rejects(loadCentroids(index, 'f16'), error => {
    assert.match(error.message, /https:\/\/example.org\/f16.gz/);
    assert.ok(error.cause);
    return true;
  });
});

function records(items) {
  const bytes = Buffer.alloc(items.length * 772);
  for (const [i, { id, dimension = 0, half = 0x3c00 }] of items.entries()) {
    bytes.writeUInt32LE(id, i * 772);
    bytes.writeUInt16LE(half, i * 772 + 4 + dimension * 2);
  }
  return gzipSync(bytes);
}

test('parallel cluster search ranks known vectors, breaks ties by ID, and rejects duplicate IDs', async t => {
  const payloads = [records([{ id: 8 }, { id: 2, dimension: 1 }]), records([{ id: 3 }])];
  const index = { url: 'https://example.org/index.json', dimensions, clusters: 2, count: 100,
    record_bytes: 772, cluster_sizes: [2, 1],
    cluster_files: payloads.map((bytes, i) => ({ file: `${i}.gz`, bytes: bytes.length })) };
  t.mock.method(globalThis, 'fetch', async url => new Response(payloads[Number(new URL(url).pathname[1])]));
  const query = new Float32Array(dimensions);
  query[0] = 1;
  const options = { centroids: new Float32Array(dimensions * 2), probes: 2, k: 2 };
  const result = await searchVectors(index, query, options);
  assert.deepEqual(result.results, [{ id: 3, distance: 0 }, { id: 8, distance: 0 }]);
  assert.equal(result.candidates, 3);
  assert.equal(result.downloads.length, 2);
  payloads[1] = records([{ id: 8 }]);
  index.cluster_files[1].bytes = payloads[1].length;
  await assert.rejects(searchVectors(index, query, options), /repeated passage ID/);
  payloads[1] = records([{ id: 3, half: 0x7c00 }]);
  index.cluster_files[1].bytes = payloads[1].length;
  await assert.rejects(searchVectors(index, query, options), /Non-finite/);
});

test('fetch options are validated before requests and JSON errors identify their asset', async t => {
  let requests = 0;
  t.mock.method(globalThis, 'fetch', async () => { requests++; return new Response('{invalid'); });
  await assert.rejects(fetchBytes('https://example.org/data', { cache: 'reload' }), /cache/i);
  await assert.rejects(fetchBytes('https://example.org/data', { networkMode: 'unexpected' }), /mode/i);
  assert.equal(requests, 0);
  await assert.rejects(loadIndex('https://example.org/broken-index.json'), error => {
    assert.match(error.message, /broken-index.json/);
    assert.ok(error.cause instanceof SyntaxError);
    return true;
  });
});
