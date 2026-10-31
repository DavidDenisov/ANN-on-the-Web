import { loadSearchIndex, loadSearchCentroids, searchIndex } from './search-index.mjs';

const MODEL_ID = 'Xenova/all-MiniLM-L6-v2';
const MODEL_REVISION = '751bff37182d3f1213fa05d7196b954e230abad9';
const RUNTIME_BASE = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.0-dev.20260914-8d85527a0/dist/';

let index;
let extractor;
let modelLoading;
let inferenceThreads;
let embeddingWork = Promise.resolve();
let active;

function send(job, type, details = {}) {
  if (active === job && !job.controller.signal.aborted) self.postMessage({ requestId: job.id, type, ...details });
}

function checkCanceled(job) {
  job.controller.signal.throwIfAborted();
}

async function initializeModel(job) {
  if (extractor) return extractor;
  send(job, 'status', { message: 'Loading the embedding model and centroids…' });
  modelLoading ??= (async () => {
    const library = import('https://cdn.jsdelivr.net/npm/@huggingface/transformers@4.3.1');
    // Fill the library's existing cache while its module downloads.
    const assets = (async () => {
      const cache = await globalThis.caches?.open('transformers-cache');
      if (!cache) return;
      const modelBase = 'https://huggingface.co/' + MODEL_ID + '/resolve/' + MODEL_REVISION + '/';
      const urls = [
        ...['config.json', 'tokenizer.json', 'tokenizer_config.json', 'onnx/model_quantized.onnx']
          .map(file => modelBase + file),
        RUNTIME_BASE + 'ort-wasm-simd-threaded.mjs',
        RUNTIME_BASE + 'ort-wasm-simd-threaded.wasm',
      ];
      await Promise.allSettled(urls.map(async url => {
        if (!await cache.match(url)) await cache.add(url);
      }));
    })().catch(() => {
      // The pipeline handles downloads normally if storage or preloading fails.
    });
    const { pipeline, env } = await library;
    env.allowLocalModels = false;
    // Let ONNX choose a thread count and fall back to one when isolation is unavailable.
    env.backends.onnx.wasm.numThreads = 0;
    env.backends.onnx.wasm.wasmPaths = {
      mjs: RUNTIME_BASE + 'ort-wasm-simd-threaded.mjs',
      wasm: RUNTIME_BASE + 'ort-wasm-simd-threaded.wasm',
    };
    await assets;
    extractor = await pipeline('feature-extraction', MODEL_ID, {
      dtype: 'q8',
      device: 'wasm',
      revision: MODEL_REVISION,
      progress_callback: event => {
        if (active && event.status === 'progress' && event.file?.endsWith('.onnx')) {
          send(active, 'model-progress', { progress: event.progress });
        }
      },
    });
    // FeatureExtractionPipeline ignores per-call max_length; the tokenizer getter is read-only.
    extractor.tokenizer.config.model_max_length = 256;
    inferenceThreads = env.backends.onnx.wasm.numThreads;
    return extractor;
  })().catch(error => {
    modelLoading = undefined;
    throw error;
  });
  return modelLoading;
}

self.onmessage = async ({ data }) => {
  if (data.type === 'cancel') {
    if (active?.id === data.requestId) {
      active.controller.abort();
      active = undefined;
    }
    return;
  }
  if (!['init', 'search'].includes(data.type) || !Number.isSafeInteger(data.requestId)) return;
  active?.controller.abort();
  const job = { id: data.requestId, controller: new AbortController() };
  active = job;
  let stage = 'index';
  try {
    if (data.type === 'init') {
      index = undefined;
      if (typeof data.indexUrl !== 'string' || !data.indexUrl) throw new Error('Enter the URL of a prepared search manifest.');
      send(job, 'status', { message: 'Loading passage index…' });
      const [loadedIndex] = await Promise.all([
        (async () => {
          const { index: nextIndex } = await loadSearchIndex(data.indexUrl, { signal: job.controller.signal });
          checkCanceled(job);
          if (nextIndex.dimensions !== 384) throw new Error('This client needs a 384-dimensional all-MiniLM-L6-v2 index.');
          await loadSearchCentroids(nextIndex, data.codec ?? 'sq6', { signal: job.controller.signal });
          return nextIndex;
        })(),
        initializeModel(job),
      ]);
      checkCanceled(job);
      index = loadedIndex;
      send(job, 'ready', { count: index.count, clusters: index.clusters, codecs: Object.keys(index.centroids),
        threads: inferenceThreads, crossOriginIsolated });
      return;
    }
    if (!index) throw new Error('Choose a centroid count and wait for its index to load before searching.');
    const loaded = index;
    stage = 'search';
    if (typeof data.text !== 'string' || !data.text.trim() || data.text.length > 2000) {
      throw new Error('Enter a question or description of up to 2,000 characters.');
    }
    stage = 'model';
    const model = await initializeModel(job);
    checkCanceled(job);
    send(job, 'status', { message: 'Embedding your query locally…' });
    // Inference cannot be aborted; serialize it and discard canceled queries before searching.
    embeddingWork = embeddingWork.catch(() => {}).then(async () => {
      checkCanceled(job);
      const started = performance.now();
      const output = await model(data.text.trim(), { pooling: 'mean', normalize: true });
      return { output, inferenceMs: performance.now() - started };
    });
    const { output, inferenceMs } = await embeddingWork;
    checkCanceled(job);
    const embedding = Array.from(output.data);
    if (embedding.length !== 384 || !embedding.every(Number.isFinite) || !embedding.some(value => value !== 0)) {
      throw new Error('The model returned an invalid query embedding. Please try again.');
    }
    stage = 'search';
    send(job, 'status', { message: 'Finding and searching the nearest clusters…' });
    const found = await searchIndex(loaded, embedding, {
      codec: data.codec, probes: data.probes, k: data.k, cache: data.cache, signal: job.controller.signal,
      onProgress: progress => send(job, 'search-progress', progress),
    });
    checkCanceled(job);
    send(job, 'result', { ...found, inferenceMs });
  } catch (error) {
    if (!job.controller.signal.aborted) {
      send(job, 'error', { stage, message: error.message || 'Search failed. Please try again.' });
    }
  } finally {
    if (active === job) active = undefined;
  }
};
