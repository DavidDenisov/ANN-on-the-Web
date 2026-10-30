// Compare the pinned q8 encoder with saved reference passage vectors.
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');
const openBrowser = require('./browser.cjs');

assert.equal(process.argv.length, 3, 'Usage: node code/benchmarks/compatibility.cjs RUN_DIRECTORY');
const run = path.resolve(process.argv[2]);
const bytes = fs.readFileSync(path.join(run, 'inputs.json'));
const inputs = JSON.parse(bytes);
const output = path.join(run, 'results.json');
assert.ok(!fs.existsSync(output), 'Refusing to overwrite saved measurements');
assert.ok([100, 1000].includes(inputs.passages.length), 'Use 100 or 1,000 reference pairs');
assert.equal(new Set(inputs.passages.map(item => item.id)).size, inputs.passages.length);
assert.ok(inputs.passages.every(item => Number.isSafeInteger(item.id) && item.id >= 0
  && typeof item.text === 'string' && item.text.length > 0
  && item.stored.length === 384 && item.stored.every(Number.isFinite) && Math.hypot(...item.stored) > 0));

async function measure(passages) {
  if (!crossOriginIsolated) throw new Error('The compatibility check requires cross-origin isolation');
  const revision = '751bff37182d3f1213fa05d7196b954e230abad9';
  const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1');
  env.allowLocalModels = false;
  env.backends.onnx.wasm.numThreads = 4;
  const base = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.0-dev.20260914-8d85527a0/dist/';
  env.backends.onnx.wasm.wasmPaths = {
    mjs: base + 'ort-wasm-simd-threaded.mjs', wasm: base + 'ort-wasm-simd-threaded.wasm',
  };
  const model = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
    dtype: 'q8', device: 'wasm', revision,
  });
  model.tokenizer.config.model_max_length = 256;
  try {
    const comparisons = [];
    for (const passage of passages) {
      const tokens = await model.tokenizer(passage.text, { truncation: true });
      const vector = (await model(passage.text, { pooling: 'mean', normalize: true })).data;
      const norm = Math.hypot(...vector);
      if (tokens.input_ids.dims.at(-1) > 256 || vector.length !== 384
        || !vector.every(Number.isFinite) || Math.abs(norm - 1) >= 0.001) throw new Error('Invalid embedding');
      const cosine = vector.reduce((sum, value, i) => sum + value * passage.stored[i], 0)
        / (norm * Math.hypot(...passage.stored));
      comparisons.push({ id: passage.id, tokens: tokens.input_ids.dims.at(-1), cosine });
    }
    return { comparisons, threads: env.backends.onnx.wasm.numThreads, userAgent: navigator.userAgent,
      modelRevision: revision, runtimeBase: base, tokenLimit: 256, crossOriginIsolated };
  } finally { await model.dispose(); }
}

(async () => {
  const session = await openBrowser();
  const { browser, origin } = session;
  try {
    const page = await browser.newPage();
    await page.goto(origin);
    const result = await page.evaluate(async ({ source, passages }) => {
      const url = URL.createObjectURL(new Blob(['self.onmessage=async({data})=>{try{self.postMessage({result:await('
        + source + ')(data)});}catch(error){self.postMessage({error:error.stack||String(error)});}};'],
      { type: 'text/javascript' }));
      const worker = new Worker(url, { type: 'module' });
      let timer;
      try {
        return await new Promise((resolve, reject) => {
          timer = setTimeout(() => reject(new Error('Compatibility check timed out')), 120000);
          worker.onmessage = ({ data }) => {
            if (data.result) resolve(data.result);
            if (data.error) reject(new Error(data.error));
          };
          worker.onerror = event => reject(new Error(event.message));
          worker.postMessage(passages);
        });
      } finally { clearTimeout(timer); worker.terminate(); URL.revokeObjectURL(url); }
    }, { source: measure.toString(), passages: inputs.passages });
    assert.equal(result.threads, 4);
    assert.ok(!result.userAgent.includes('HeadlessChrome'));
    assert.deepEqual(result.comparisons.map(item => item.id), inputs.passages.map(item => item.id));
    assert.ok(result.comparisons.every(item => Number.isFinite(item.cosine) && Math.abs(item.cosine) <= 1.000001));
    const cosines = result.comparisons.map(item => item.cosine);
    const summary = { samples: cosines.length,
      cosineMean: cosines.reduce((sum, value) => sum + value, 0) / cosines.length,
      cosineP5: [...cosines].sort((a, b) => a - b)[Math.ceil(cosines.length * 0.05) - 1] };
    const report = { measuredAt: new Date().toISOString(), transformers: '4.3.1',
      browserVersion: await browser.version(), model: 'Xenova/all-MiniLM-L6-v2', dtype: 'q8',
      inputsSha256: crypto.createHash('sha256').update(bytes).digest('hex'),
      runnerSha256: crypto.createHash('sha256').update(fs.readFileSync(__filename)).digest('hex'),
      source: inputs.source, ...result, ...summary };
    fs.writeFileSync(output, JSON.stringify(report, null, 2) + '\n', { flag: 'wx' });
    fs.writeFileSync(path.join(run, 'summary.json'), JSON.stringify(summary, null, 2) + '\n', { flag: 'wx' });
    console.log(JSON.stringify(summary));
  } finally {
    await session.close();
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
