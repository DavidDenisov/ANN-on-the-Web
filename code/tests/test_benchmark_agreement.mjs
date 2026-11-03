import assert from 'node:assert/strict';
import test from 'node:test';
import { exactTop1ForSuite, top1Agreement } from '../client_wiki/assets/benchmark.mjs';

test('scores sampled queries against exact passage IDs in fixture order', () => {
  const suite = { id: 'fixture', noise: { level: 0.4 }, count: 10, query_count: 4 };
  const reference = {
    fixture_id: suite.id, noise_level: 0.4, passages: 10, queries: 4, exact_top1_ids: [2, 4, 6, 8],
  };
  const ids = exactTop1ForSuite(suite, { fixture: reference });
  assert.equal(top1Agreement([
    { query: 3, resultIds: [8] }, { query: 1, resultIds: [] }, { query: 2, resultIds: [4] },
  ], ids), 1 / 3);
  assert.equal(exactTop1ForSuite(suite, {}), undefined);
  assert.throws(() => exactTop1ForSuite(suite, { fixture: { ...reference, noise_level: 0.15 } }),
    /does not match/);
});
