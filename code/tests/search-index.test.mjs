import assert from 'node:assert/strict';
import test from 'node:test';
import fixtures from './search-fixture.cjs';
const { dimensions, count, metadataBatch, fixture } = fixtures;
import { loadSearchIndex, searchIndex, decodeMetadataBatch, loadResultText } from '../client_wiki/assets/search-index.mjs';

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
  assert.equal(first.textDownloads[0].requested_url, 'https://example.org/meta_0.bin.br');
  assert.equal(first.textDownloads[0].resolved_url, first.textDownloads[0].requested_url);
  assert.equal('requestedUrl' in first.textDownloads[0], false);
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
  const query = Float32Array.from({ length: dimensions }, (_, i) => i === 0 ? 1 : 0);
  const searched = await searchIndex(index, query, { codec: 'f16', probes: 1, k: 1 });
  assert.equal(searched.metadataUnavailable, true);
  assert.equal(searched.failures[0].batch, 0);
  assert.match(searched.failures[0].message, /404/);
  files.set('https://example.org/meta_0.bin.br', metadataBatch());
  const retried = await loadResultText(index, result.results.map(result => ({ ...result, extra: 'preserved' })));
  assert.equal(retried.metadataUnavailable, false);
  assert.deepEqual(retried.failures, []);
  assert.deepEqual(retried.results, [{ id: 5, distance: 0, title: 'Title', text: 'Passage', extra: 'preserved' }]);
  await assert.rejects(loadResultText(index, [{ id: count }]), /Invalid/);
  await assert.rejects(loadResultText(index, [], { signal: AbortSignal.abort() }), { name: 'AbortError' });
});
