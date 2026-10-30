// Run with Playwright available. The isolated localhost control page is served automatically.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const openBrowser = require('./browser.cjs');

async function measure({ inputs, inference }) {
  if (!crossOriginIsolated) throw new Error("Benchmark worker is not cross-origin isolated");
  const libraryStart = performance.now();
  const { pipeline, env } = await import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1');
  const libraryImportMs = performance.now() - libraryStart;
  const modelId = 'Xenova/all-MiniLM-L6-v2';
  const base = `https://huggingface.co/${modelId}/resolve/${inputs.modelRevision}/`;
  const modelFiles = ['config.json', 'tokenizer.json', 'tokenizer_config.json', 'onnx/model_quantized.onnx'];
  const runtimeBase = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.0-dev.20260914-8d85527a0/dist/';
  const urls = [...modelFiles.map(file => base + file),
    runtimeBase + 'ort-wasm-simd-threaded.mjs', runtimeBase + 'ort-wasm-simd-threaded.wasm'];
  const downloadStart = performance.now();
  const assets = await Promise.all(urls.map(async url => {
    const started = performance.now();
    const response = await fetch(url, { cache: 'no-store' });
    if (!response.ok) throw new Error(`Asset download failed: ${response.status} ${url}`);
    const buffer = await response.arrayBuffer();
    return { url, buffer, bytes: buffer.byteLength, downloadMs: performance.now() - started };
  }));
  const downloadMs = performance.now() - downloadStart;
  const initializationStart = performance.now();
  const cached = new Map(assets.slice(0, modelFiles.length).map(asset => [asset.url, asset.buffer]));
  env.allowLocalModels = false;
  env.useBrowserCache = false;
  env.useWasmCache = false;
  env.useCustomCache = true;
  env.customCache = {
    match: async url => cached.has(url) ? new Response(cached.get(url)) : undefined,
    put: async (url, response) => { cached.set(url, await response.arrayBuffer()); },
  };
  const mjs = URL.createObjectURL(new Blob([assets.at(-2).buffer], { type: 'text/javascript' }));
  const wasm = URL.createObjectURL(new Blob([assets.at(-1).buffer], { type: 'application/wasm' }));
  env.backends.onnx.wasm.numThreads = 4;
  env.backends.onnx.wasm.wasmPaths = { mjs, wasm };
  env.backends.onnx.wasm.wasmBinary = new Uint8Array(assets.at(-1).buffer);
  const originalFetch = self.fetch;
  self.fetch = (url, options) => {
    if (!String(url).startsWith('blob:')) throw new Error(`Unexpected network access after preloading: ${url}`);
    return originalFetch(url, options);
  };
  const model = await pipeline('feature-extraction', modelId, {
    dtype: 'q8', device: 'wasm', revision: inputs.modelRevision,
  });
  const initializationMs = performance.now() - initializationStart;
  const threads = env.backends.onnx.wasm.numThreads;
  if (threads !== 4) throw new Error("The paper benchmark requires four inference threads");
  const startup = { libraryImportMs, downloadMs, initializationMs,
    threads, crossOriginIsolated, hardwareConcurrency: navigator.hardwareConcurrency,
    assets: assets.map(({ buffer, ...asset }) => asset),
    modelBytes: assets.slice(0, modelFiles.length).reduce((sum, asset) => sum + asset.bytes, 0),
    runtimeBytes: assets.slice(modelFiles.length).reduce((sum, asset) => sum + asset.bytes, 0) };
  self.postMessage({ startup });
  const trials = [], lengths = [];
  const options = { pooling: 'mean', normalize: true };
  if (inference) {
    let modelInputTokens;
    model.model = new Proxy(model.model, {
      apply(target, receiver, args) {
        modelInputTokens = args[0].input_ids.dims.at(-1);
        return Reflect.apply(target, receiver, args);
      },
    });
    for (const input of inputs.passages) {
      for (const tokens of inputs.lengths) {
        model.tokenizer.config.model_max_length = tokens;
        const encoded = await model.tokenizer(input.text, { truncation: true });
        if (encoded.input_ids.dims.at(-1) !== tokens) throw new Error('Input is shorter than its target token length');
        lengths.push({ id: input.id, tokens });
      }
    }
    for (const tokens of inputs.lengths) {
      model.tokenizer.config.model_max_length = tokens;
      for (const input of inputs.passages.slice(0, 3)) await model(input.text, options);
    }
    for (const trial of inputs.trials) {
      const text = inputs.passages[trial.input].text;
      model.tokenizer.config.model_max_length = trial.tokens;
      const started = performance.now();
      const output = await model(text, options);
      const inferenceMs = performance.now() - started;
      if (modelInputTokens !== trial.tokens) throw new Error('Actual model input does not match target length');
      if (output.data.length !== 384 || !output.data.every(Number.isFinite)
        || Math.abs(Math.hypot(...output.data) - 1) >= 0.001) throw new Error('Invalid model output');
      trials.push({ ...trial, id: inputs.passages[trial.input].id, modelInputTokens, inferenceMs });
      if (trials.length % 100 === 0) self.postMessage({ progress: `Timed ${trials.length}/${inputs.trials.length} long-input calls` });
    }
  }
  await model.dispose();
  URL.revokeObjectURL(mjs); URL.revokeObjectURL(wasm);
  return { startup, trials, lengths, userAgent: navigator.userAgent };
}

(async () => {
  assert.equal(process.argv.length, 3, 'Usage: node code/benchmarks/inference.cjs RUN_DIRECTORY');
  const run = path.resolve(process.argv[2]);
  const output = path.join(run, 'results.json');
  const startupPath = path.join(run, 'startup.jsonl');
  assert.ok(!fs.existsSync(output) && !fs.existsSync(startupPath), 'Refusing to overwrite saved measurements');
  const inputs = JSON.parse(fs.readFileSync(path.join(run, 'inputs.json'), 'utf8'));
  assert.equal(inputs.modelRevision, '751bff37182d3f1213fa05d7196b954e230abad9');
  assert.equal(inputs.passages.length, 100);
  assert.deepEqual(inputs.lengths, [64, 128, 256]);
  assert.equal(inputs.startup_trials, 100);
  assert.equal(inputs.threads, undefined, 'The paper fixes inference to four threads');
  assert.equal(inputs.trials.length, 1000);
  assert.deepEqual(inputs.lengths.map(tokens => inputs.trials.filter(trial => trial.tokens === tokens).length), [334, 333, 333]);
  assert.ok(inputs.trials.every(trial => Number.isInteger(trial.repeat) && trial.repeat >= 0
    && Number.isInteger(trial.input) && trial.input >= 0 && trial.input < inputs.passages.length
    && inputs.lengths.includes(trial.tokens)));
  const session = await openBrowser();
  const { browser, origin: controlUrl } = session;
  let startupFile;
  try {
    startupFile = fs.openSync(startupPath, "wx");
    const results = { measuredAt: new Date().toISOString(), cpu: os.cpus()[0].model,
      browserVersion: await browser.version(), model: 'Xenova/all-MiniLM-L6-v2',
      transformers: '4.3.1', dtype: 'q8', device: 'wasm', requestedThreads: 4, startups: [] };
    for (let trial = 0; trial < inputs.startup_trials; trial++) {
      const context = await browser.newContext();
      try {
        const page = await context.newPage();
        page.on('console', message => console.log(message.text()));
        await page.goto(controlUrl);
        assert.equal(await page.evaluate(() => crossOriginIsolated), true);
        const result = await page.evaluate(async ({ inputs, code, inference }) => {
          const source = `self.onmessage = async ({ data }) => { try { self.postMessage({ result: await (${code})(data) }); } catch (error) { self.postMessage({ error: error.stack || String(error) }); } };`;
          const url = URL.createObjectURL(new Blob([source], { type: 'text/javascript' }));
          const worker = new Worker(url, { type: 'module' });
          try {
            return await new Promise((resolve, reject) => {
              worker.onmessage = ({ data }) => {
                if (data.progress) console.log(data.progress);
                if (data.startup) console.log(`Assets downloaded in ${data.startup.downloadMs.toFixed(0)} ms; initialized in ${data.startup.initializationMs.toFixed(0)} ms`);
                if (data.result) resolve(data.result);
                if (data.error) reject(new Error(data.error));
              };
              worker.onerror = error => reject(new Error(error.message));
              worker.postMessage({ inputs, inference });
            });
          } finally { worker.terminate(); URL.revokeObjectURL(url); }
        }, { inputs, code: measure.toString(), inference: trial === inputs.startup_trials - 1 });
        results.threads ??= result.startup.threads;
        assert.equal(result.startup.threads, results.threads);
        assert.equal(result.startup.crossOriginIsolated, true);
        results.startups.push({ trial, ...result.startup });
        fs.writeSync(startupFile, JSON.stringify(results.startups.at(-1)) + '\n');
        if (result.trials.length) Object.assign(results, { trials: result.trials, lengths: result.lengths, userAgent: result.userAgent });
        console.log(`Completed startup trial ${trial + 1}/${inputs.startup_trials}`);
      } finally { await context.close(); }
    }
    assert.equal(results.startups.length, inputs.startup_trials);
    assert.equal(results.trials.length, inputs.trials.length);
    assert.equal(results.lengths.length, 300);
    assert.ok(results.startups.every(trial => trial.downloadMs > 0 && trial.initializationMs > 0 && trial.modelBytes > 20000000));
    assert.ok(results.trials.every(trial => Number.isFinite(trial.inferenceMs) && trial.inferenceMs > 0));
    for (const tokens of inputs.lengths) assert.equal(results.trials.filter(trial => trial.tokens === tokens).length,
      inputs.trials.filter(trial => trial.tokens === tokens).length);
    assert.ok(!results.userAgent.includes('HeadlessChrome'));
    fs.writeFileSync(output, JSON.stringify(results, null, 2) + '\n', { flag: 'wx' });
    fs.closeSync(startupFile); startupFile = undefined;
    fs.unlinkSync(startupPath);
    console.log(`Saved ${results.startups.length} startup trials and ${results.trials.length} longer-input inference timings.`);
  } finally {
    try { if (startupFile !== undefined) fs.closeSync(startupFile); }
    finally { await session.close(); }
  }
})().catch(error => { console.error(error); process.exitCode = 1; });
