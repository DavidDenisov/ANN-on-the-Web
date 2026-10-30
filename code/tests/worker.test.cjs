const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const { pathToFileURL } = require('node:url');
const { spawnSync } = require('node:child_process');
const test = require('node:test');
const vm = require('node:vm');

if (!vm.SourceTextModule) {
  test('worker cancellation and request isolation', () => {
    const run = spawnSync(process.execPath, ['--experimental-vm-modules', __filename], { encoding: 'utf8', timeout: 10000 });
    assert.equal(run.status, 0, run.stdout + run.stderr);
  });
} else {
  const deferred = () => {
    let resolve;
    const promise = new Promise(done => { resolve = done; });
    return { promise, resolve };
  };

  function documentFixture() {
    const nodes = new Map();
    const element = () => ({
      value: '', innerHTML: '', textContent: '', listeners: {}, children: [],
      addEventListener(name, callback) { this.listeners[name] = callback; },
      setAttribute() {}, focus() {}, querySelectorAll: () => [], classList: { toggle() {} },
      append(...children) { this.children.push(...children); this.firstChild = this.children[0]; },
      replaceChildren(...children) { this.children = children; this.firstChild = children[0]; },
    });
    const node = id => {
      if (!nodes.has(id)) nodes.set(id, element());
      return nodes.get(id);
    };
    return { node, document: { querySelector: node, getElementById: node,
      querySelectorAll: () => [], createElement: element } };
  }

  test('search controls recover after cancellation and ignore stale worker messages', async () => {
    const { node, document } = documentFixture();
    node('#clusterCount').value = '8192';
    node('#codec').value = 'sq6';
    node('#probes').value = '4';
    node('#resultK').value = '10';
    node('#searchForm').elements = ['#query', '#codec', '#probes', '#resultK', '#clusterCount'].map(node);
    const workers = [];
    const context = vm.createContext({ document, URL, location: { href: 'https://example.org/client_wiki/' },
      Option: function (text, value) { return { text, value }; },
      Worker: class {
        constructor() { this.requests = []; workers.push(this); }
        postMessage(data) { this.requests.push(data); }
      } });
    const labels = new vm.SyntheticModule(['CODEC_LABELS'], function () {
      this.setExport('CODEC_LABELS', { sq6: 'SQ6' });
    }, { context });
    const sourcePath = path.join(__dirname, '../client_wiki/assets/search.mjs');
    const module = new vm.SourceTextModule(fs.readFileSync(sourcePath, 'utf8'), {
      context, initializeImportMeta: meta => { meta.url = pathToFileURL(sourcePath).href; },
    });
    await module.link(() => labels);
    await module.evaluate();
    const worker = workers[0];
    worker.onmessage({ data: { type: 'ready', requestId: 1, codecs: ['sq6'], clusters: 8192, count: 100 } });
    node('#query').value = 'question';
    node('#query').listeners.input();
    const submit = () => node('#searchForm').listeners.submit({ preventDefault() {} });
    submit();
    assert.equal(node('#query').disabled, true);
    node('#cancelBtn').listeners.click();
    assert.equal(node('#query').disabled, false);
    worker.onmessage({ data: { type: 'error', requestId: 2, stage: 'index', message: 'stale failure' } });
    assert.ok(!node('#status').textContent.includes('stale failure'));
    submit();
    const request = worker.requests.at(-1);
    assert.equal(request.type, 'search');
    worker.onmessage({ data: { type: 'result', requestId: request.requestId, results: [],
      candidates: 1, clusters: 1, bytes: 10, inferenceMs: 1, centroidMs: 0, vectorMs: 2, metadataMs: 3 } });
    assert.equal(node('#searchBtn').disabled, false);
    assert.equal(node('#status').textContent, 'Search complete.');
  });

  test('lab renders all three layouts in probe order with their original columns', async () => {
    const { node, document } = documentFixture();
    node('suiteUrl').value = 'suite.json';
    const context = vm.createContext({ document, URL, location: { href: 'https://example.org/' }, Intl });
    const benchmark = new vm.SyntheticModule(['loadSuite', 'benchmark', 'paperPlan'], function () {
      this.setExport('loadSuite', async () => ({ count: 100, query_count: 10, noise: { level: .4 },
        indexes: [{ clusters: 8192, url: 'index.json' }] }));
      this.setExport('paperPlan', () => [4, 1].map(probes => ({ clusters: 8192, codec: 'sq6', probes,
        accuracy_only: false, memberships: ['counts', 'probes', 'codecs'] })));
      this.setExport('benchmark', async (index, suite, options) => ({ clusters: 8192, codec: 'sq6',
        probes: options.probes, recall: .5, centroidBytes: 1000000, centroidDownloadMs: { mean: 2 },
        centroidDecodeMs: { mean: 1 }, queryMs: { mean: 3, p95: 4 }, candidatesMean: 20, bytesPerQuery: 2000000 }));
    }, { context });
    const retrieval = new vm.SyntheticModule(['loadIndex', 'CODEC_LABELS'], function () {
      this.setExport('loadIndex', async () => ({ clusters: 8192 }));
      this.setExport('CODEC_LABELS', { sq6: 'SQ6' });
    }, { context });
    const source = fs.readFileSync(path.join(__dirname, '../client_wiki/assets/lab.mjs'), 'utf8');
    const module = new vm.SourceTextModule(source, { context });
    await module.link(name => name === './benchmark.mjs' ? benchmark : retrieval);
    await module.evaluate();
    await new Promise(setImmediate);
    await node('runAll').listeners.click();
    const cells = id => node(id).children.map(row => row.children.map(cell => cell.textContent));
    assert.deepEqual(cells('countsRows').map(row => row.slice(0, 2)), [[8192, 1], [8192, 4]]);
    assert.deepEqual(cells('probesRows')[0], [1, '20', '50.0%', '3.00 / 4.00 ms', '2.00 MB']);
    assert.deepEqual(cells('codecsRows')[0], ['SQ6', 1, '1.00 MB', '2.00 ms', '1.00 ms', '50.0%']);
  });

  async function worker() {
    const messages = [], searches = [], inference = [], indexLoads = [];
    const modelReady = deferred();
    const model = async () => {
      const call = deferred();
      inference.push(call);
      await call.promise;
      return { data: Float32Array.from({ length: 384 }, (_, i) => i === 0 ? 1 : 0) };
    };
    model.tokenizer = { config: {} };
    const context = vm.createContext({
      self: { postMessage: message => messages.push(message) }, AbortController, performance, crossOriginIsolated: true,
    });
    const library = new vm.SyntheticModule(['pipeline', 'env'], function () {
      this.setExport('pipeline', async () => { await modelReady.promise; return model; });
      this.setExport('env', { backends: { onnx: { wasm: { numThreads: 4 } } } });
    }, { context });
    await library.link(() => {});
    await library.evaluate();
    const search = new vm.SyntheticModule(['loadSearchIndex', 'loadSearchCentroids', 'searchIndex'], function () {
      this.setExport('loadSearchIndex', async url => {
        const call = deferred();
        indexLoads.push(call);
        await call.promise;
        return { index: { dimensions: 384, count: 100, clusters: 8192, centroids: { sq6: {} }, url } };
      });
      this.setExport('loadSearchCentroids', async () => ({}));
      this.setExport('searchIndex', async (index, embedding, options) => {
        searches.push({ index, embedding, options });
        return { results: [] };
      });
    }, { context });
    const source = fs.readFileSync(path.join(__dirname, '../client_wiki/assets/embedding-worker.mjs'), 'utf8');
    const module = new vm.SourceTextModule(source, { context, importModuleDynamically: () => library });
    await module.link(() => search);
    await module.evaluate();
    return { messages, searches, inference, indexLoads, modelReady,
      send: data => context.self.onmessage({ data }) };
  }

  test('canceled initialization cannot publish stale readiness', async () => {
    const w = await worker();
    const first = w.send({ type: 'init', requestId: 1, indexUrl: 'first' });
    await w.send({ type: 'cancel', requestId: 1 });
    const second = w.send({ type: 'init', requestId: 2, indexUrl: 'second' });
    w.indexLoads.forEach(load => load.resolve());
    w.modelReady.resolve();
    await Promise.all([first, second]);
    assert.deepEqual(w.messages.filter(message => message.type === 'ready').map(message => message.requestId), [2]);
    assert.ok(!w.messages.some(message => message.type === 'error'));
  });

  test('canceled inference is drained before the next query and cannot emit its result', async () => {
    const w = await worker();
    const init = w.send({ type: 'init', requestId: 1, indexUrl: 'index' });
    w.indexLoads[0].resolve();
    w.modelReady.resolve();
    await init;
    const first = w.send({ type: 'search', requestId: 2, text: 'first', cache: 'no-store' });
    await new Promise(setImmediate);
    assert.equal(w.inference.length, 1);
    await w.send({ type: 'cancel', requestId: 2 });
    const second = w.send({ type: 'search', requestId: 3, text: 'second', cache: 'no-store' });
    await new Promise(setImmediate);
    assert.equal(w.inference.length, 1);
    w.inference[0].resolve();
    await new Promise(setImmediate);
    assert.equal(w.inference.length, 2);
    w.inference[1].resolve();
    await Promise.all([first, second]);
    assert.deepEqual(w.messages.filter(message => message.type === 'result').map(message => message.requestId), [3]);
    assert.equal(w.searches.length, 1);
    assert.equal(w.searches[0].options.cache, 'no-store');
    assert.equal(w.searches[0].index.url, 'index');
  });
}
