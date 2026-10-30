import assert from 'node:assert/strict';
import test from 'node:test';
import { gzipSync } from 'node:zlib';
import { loadSearchIndex, searchIndex, decodeMetadataBatch, loadResultText } from '../client_wiki/assets/search-index.mjs';

const dimensions = 384, count = 35167920, clusters = 8192;

function metadataBatch() {
  const header = Buffer.alloc(12);
  header.write('WMD1');
  header.writeUInt32LE(500, 8);
  const records = Array.from({ length: 500 }, () => {
    const sizes = Buffer.alloc(8);
    sizes.writeUInt32LE(5);
    sizes.writeUInt32LE(7, 4);
    return Buffer.concat([sizes, Buffer.from('TitlePassage')]);
  });
  return Buffer.concat([header, ...records]);
}

function fixture() {
  const centroids = gzipSync(Buffer.alloc(clusters * dimensions * 2));
  const record = Buffer.alloc(4 + dimensions * 2);
  record.writeUInt32LE(5);
  record.writeUInt16LE(0x3c00, 4);
  const cluster = gzipSync(record);
  const index = {
    version: 1, dimensions, count, clusters, record_bytes: 772, metric: 'normalized-l2',
    embedding_model: 'Xenova/all-MiniLM-L6-v2',
    cluster_sizes: Array.from({ length: clusters }, (_, i) => i === 0 ? 1 : i === clusters - 1 ? count - 1 : 0),
    cluster_files: Array.from({ length: clusters }, (_, i) => ({ file: `cluster-${i}.gz`, bytes: cluster.length })),
    centroids: Object.fromEntries(['f16', 'sq8', 'sq6', 'sq4', 'int8', 'pq4']
      .map(codec => [codec, { file: 'centroids.gz', bytes: centroids.length, rmse: 0 }])),
    metadata: { format: 'binary-http-v1', count, batch_size: 500, pattern: 'meta_{id}.bin.br',
      files_per_deployment: 1000, base_urls: Array.from({ length: 71 }, () => 'https://example.org/') },
  };
  const files = new Map([
    ['https://example.org/index.json', JSON.stringify(index)],
    ['https://example.org/centroids.gz', centroids],
    ['https://example.org/cluster-0.gz', cluster],
    ['https://example.org/meta_0.bin.br', metadataBatch()],
  ]);
  const requests = [];
  return { files, requests, fetch: async (url, options) => {
    requests.push({ url: String(url), cache: options.cache });
    return new Response(files.get(String(url)), { status: files.has(String(url)) ? 200 : 404 });
  } };
}

test('search uses explicit cache policy and returns the actual vector/text downloads', async t => {
  const { requests, fetch } = fixture();
  t.mock.method(globalThis, 'fetch', fetch);
  const { index } = await loadSearchIndex('https://example.org/index.json');
  const query = new Float32Array(dimensions);
  query[0] = 1;
  const options = { codec: 'f16', probes: 1, k: 1 };
  const first = await searchIndex(index, query, options);
  assert.deepEqual(first.results, [{ id: 5, title: 'Title', text: 'Passage', distance: 0 }]);
  assert.equal(first.vectorDownloads.length, 1);
  assert.equal(first.textDownloads.length, 1);
  const cached = await searchIndex(index, query, options);
  assert.equal(cached.textDownloads.length, 0);
  const uncached = await searchIndex(index, query, { ...options, cache: 'no-store' });
  assert.equal(uncached.vectorDownloads.length, 1);
  assert.equal(uncached.textDownloads.length, 1);
  assert.deepEqual(requests.slice(-2).map(request => request.cache), ['no-store', 'no-store']);
  assert.equal(requests.filter(request => request.url.endsWith('centroids.gz')).length, 1);
  const before = requests.length;
  await assert.rejects(searchIndex(index, query, { ...options, cache: 'reload' }), /cache/i);
  assert.equal(requests.length, before);
});

test('metadata decoder rejects truncated, mismatched, trailing and invalid UTF-8 data', () => {
  const raw = metadataBatch();
  const arrayBuffer = bytes => Uint8Array.from(bytes).buffer;
  assert.equal(decodeMetadataBatch(arrayBuffer(raw), 0, 500)[5].text, 'Passage');
  for (const buffer of [raw.subarray(0, 3), raw.subarray(0, -1), Buffer.concat([raw, Buffer.from([0])])]) {
    assert.throws(() => decodeMetadataBatch(arrayBuffer(buffer), 0, 500));
  }
  assert.throws(() => decodeMetadataBatch(arrayBuffer(raw), 500, 500), /does not match/);
  const invalid = Buffer.from(raw);
  invalid[20] = 0xff;
  assert.throws(() => decodeMetadataBatch(arrayBuffer(invalid), 0, 500));
});

test('metadata failure preserves vector matches and cancellation stays a rejection', async t => {
  const { files, fetch } = fixture();
  files.delete('https://example.org/meta_0.bin.br');
  t.mock.method(globalThis, 'fetch', fetch);
  const { index } = await loadSearchIndex('https://example.org/index.json');
  const result = await loadResultText(index, [{ id: 5, distance: 0 }]);
  assert.equal(result.metadataUnavailable, true);
  assert.equal(result.results[0].id, 5);
  assert.match(result.failures[0].message, /404/);
  await assert.rejects(loadResultText(index, [{ id: count }]), /Invalid/);
  await assert.rejects(loadResultText(index, [], { signal: AbortSignal.abort() }), { name: 'AbortError' });
});
