import { loadIndex, loadCentroids, searchVectors, fetchBytes, internetAssetUrl } from './retrieval.mjs';

/**
 * @typedef {import('./retrieval.mjs').Index & {embedding_model: string,
 *   metadata: import('./retrieval.mjs').PassageMetadata}} SearchIndex
 * @typedef {import('./retrieval.mjs').SearchMatch} SearchMatch
 * @typedef {{id: number, title: string|null, text: string|null, metadataUnavailable?: boolean}} Passage
 * @typedef {Omit<import('./retrieval.mjs').AssetDownload, 'buffer'|'requestedUrl'|'resolvedUrl'> &
 *   {batch: number, decodeMs: number, requested_url: string, resolved_url: string}} TextDownload
 * @typedef {{codec?: import('./retrieval.mjs').Codec, probes?: number, k?: number,
 *   signal?: AbortSignal, cache?: 'force-cache'|'no-store',
 *   onProgress?: (progress: import('./retrieval.mjs').SearchProgress) => void}} SearchOptions
 * @typedef {{results: (Passage & SearchMatch)[], candidates: number, clusters: number, bytes: number,
 *   centroidMs: number, vectorMs: number, metadataMs: number, metadataUnavailable: boolean,
 *   vectorDownloads: import('./retrieval.mjs').VectorDownload[], textDownloads: TextDownload[],
 *   failures: {batch: number, message: string}[], elapsedMs: number}} SearchResult
 */

const states = new WeakMap();
const METADATA_CACHE_BYTES = 8 * 1024 * 1024;
const METADATA_FETCH_CONCURRENCY = 16;
const validInteger = (value, minimum = 0) => Number.isSafeInteger(value) && value >= minimum;

/** @param {string|URL} url @param {{signal?: AbortSignal}} [options] @returns {Promise<{index: SearchIndex}>} */
export async function loadSearchIndex(url, { signal } = {}) {
  const index = await loadIndex(url, { signal });
  signal?.throwIfAborted();
  if (index.embedding_model !== 'Xenova/all-MiniLM-L6-v2') {
    throw new Error('This search page needs a 384-dimensional all-MiniLM-L6-v2 index.');
  }
  const meta = index.metadata;
  if (!meta || meta.format !== 'binary-http-v1'
    || meta.count !== index.count
    || meta.batch_size !== 500 || meta.pattern !== 'meta_{id}.bin.br') {
    throw new Error('This index needs complete 500-passage Brotli metadata; see the client guide.');
  }
  if (meta.files_per_deployment !== 1000 || !Array.isArray(meta.base_urls) || meta.base_urls.length !== 71) {
    throw new Error('Invalid passage metadata hosting map.');
  }
  for (const base of meta.base_urls) internetAssetUrl(base, base);
  const searchIndex = /** @type {SearchIndex} */ (index);
  states.set(searchIndex, { centroids: new Map(), metadata: new Map(), metadataBytes: 0 });
  return { index: searchIndex };
}

/** @param {ArrayBuffer} buffer @param {number} first @param {number} count @returns {Passage[]} */
export function decodeMetadataBatch(buffer, first, count) {
  const view = new DataView(buffer);
  if (buffer.byteLength < 12 || view.getUint32(0, true) !== 0x31444d57
    || view.getUint32(4, true) !== first || view.getUint32(8, true) !== count) {
    throw new Error('Metadata batch does not match the search index.');
  }
  const decoder = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true });
  const records = [];
  let offset = 12;
  for (let position = 0; position < count; position++) {
    if (offset + 8 > buffer.byteLength) throw new Error('Truncated metadata record.');
    const titleBytes = view.getUint32(offset, true), textBytes = view.getUint32(offset + 4, true);
    offset += 8;
    if (titleBytes + textBytes > buffer.byteLength - offset) throw new Error('Truncated metadata strings.');
    const title = decoder.decode(new Uint8Array(buffer, offset, titleBytes));
    offset += titleBytes;
    const text = decoder.decode(new Uint8Array(buffer, offset, textBytes));
    offset += textBytes;
    records.push({ id: first + position, title, text });
  }
  if (offset !== buffer.byteLength) throw new Error('Unexpected trailing metadata bytes.');
  return records;
}

async function loadMetadata(index, batch, state, signal, cache) {
  const base = index.metadata.base_urls[Math.floor(batch / 1000)];
  const url = new URL(index.metadata.pattern.replace('{id}', batch), base).href;
  const cached = cache === 'force-cache' && state.metadata.get(url);
  if (cached) {
    state.metadata.delete(url);
    state.metadata.set(url, cached);
    return { records: cached.records, download: null };
  }
  const { buffer, requestedUrl, resolvedUrl, ...downloaded } = await fetchBytes(url, { signal, cache });
  signal?.throwIfAborted();
  const first = batch * index.metadata.batch_size;
  const count = Math.min(index.metadata.batch_size, index.metadata.count - first);
  const decodeStarted = performance.now();
  const records = decodeMetadataBatch(buffer, first, count);
  const decodeMs = performance.now() - decodeStarted;
  // Match the reference client's conservative estimate for strings and objects.
  const bytes = buffer.byteLength * 2 + records.length * 256;
  if (cache === 'force-cache' && bytes <= METADATA_CACHE_BYTES) {
    const previous = state.metadata.get(url);
    if (previous) {
      state.metadataBytes -= previous.bytes;
      state.metadata.delete(url);
    }
    while (state.metadataBytes + bytes > METADATA_CACHE_BYTES) {
      const oldest = state.metadata.keys().next().value;
      state.metadataBytes -= state.metadata.get(oldest).bytes;
      state.metadata.delete(oldest);
    }
    state.metadata.set(url, { records, bytes });
    state.metadataBytes += bytes;
  }
  return { records, download: { batch, ...downloaded, requested_url: requestedUrl, resolved_url: resolvedUrl, decodeMs } };
}

/**
 * Failed batches retain ranked IDs with null title/text and metadataUnavailable=true.
 * @template {{id: number}} T
 * @param {SearchIndex} index
 * @param {T[]} results
 * @param {{signal?: AbortSignal, cache?: 'force-cache'|'no-store'}} [options]
 * @returns {Promise<{results: (Passage & Omit<T, 'title'|'text'|'metadataUnavailable'>)[], metadataMs: number, metadataUnavailable: boolean,
 *   downloads: TextDownload[], failures: {batch: number, message: string}[]}>}
 */
export async function loadResultText(index, results, { signal, cache = 'force-cache' } = {}) {
  const started = performance.now();
  signal?.throwIfAborted();
  const state = states.get(index);
  if (!state) throw new Error('Load a search index before loading result text.');
  if (!['force-cache', 'no-store'].includes(cache)
    || !Array.isArray(results) || results.some(result => !result || !validInteger(result.id) || result.id >= index.count)) {
    throw new Error('Invalid result-text loading options.');
  }
  const metadata = new Map(), downloads = [], failures = [];
  const batches = [...new Set(results.map(result => Math.floor(result.id / index.metadata.batch_size)))];
  let next = 0;
  await Promise.allSettled(Array.from({ length: Math.min(METADATA_FETCH_CONCURRENCY, batches.length) }, async () => {
    while (next < batches.length) {
      signal?.throwIfAborted();
      const batch = batches[next++];
      try {
        const loaded = await loadMetadata(index, batch, state, signal, cache);
        if (loaded.download) downloads.push(loaded.download);
        for (const record of loaded.records) metadata.set(record.id, record);
      } catch (error) {
        signal?.throwIfAborted();
        failures.push({ batch, message: error.message });
      }
    }
  }));
  signal?.throwIfAborted();
  return { results: results.map(result => {
    const passage = metadata.get(result.id);
    const enriched = { ...result, ...(passage ?? { title: null, text: null, metadataUnavailable: true }) };
    if (passage) delete enriched.metadataUnavailable;
    return enriched;
  }), metadataMs: performance.now() - started, metadataUnavailable: failures.length > 0, downloads, failures };
}

/**
 * @param {SearchIndex} index
 * @param {import('./retrieval.mjs').Codec} codec
 * @param {{signal?: AbortSignal}} [options]
 * @returns {Promise<{centroids: Float32Array, centroidMs: number}>}
 */
export async function loadSearchCentroids(index, codec, { signal } = {}) {
  signal?.throwIfAborted();
  const state = states.get(index);
  if (!state) throw new Error('Load a search index before searching.');
  let centroids = state.centroids.get(codec);
  let centroidMs = 0;
  if (!centroids) {
    const centroidStarted = performance.now();
    centroids = (await loadCentroids(index, codec, { signal, cache: 'force-cache' })).centroids;
    signal?.throwIfAborted();
    centroidMs = performance.now() - centroidStarted;
    state.centroids.set(codec, centroids);
  }
  return { centroids, centroidMs };
}

/** @param {SearchIndex} index @param {ArrayLike<number>} embedding @param {SearchOptions} [options] @returns {Promise<SearchResult>} */
export async function searchIndex(index, embedding, {
  codec = 'sq6', probes = 4, k = 10, signal, onProgress, cache = 'force-cache',
} = {}) {
  const started = performance.now();
  signal?.throwIfAborted();
  if (!states.has(index)) throw new Error('Load a search index before searching.');
  if (!['force-cache', 'no-store'].includes(cache)) throw new Error('Invalid search cache policy.');
  if (!Number.isSafeInteger(probes) || probes < 1 || probes > 100) {
    throw new Error('Choose between 1 and 100 probes.');
  }
  if (embedding?.length !== index.dimensions) throw new Error('The embedding model returned an invalid query vector.');
  const query = Float32Array.from(embedding);
  const norm = Math.hypot(...query);
  if (query.length !== index.dimensions || !Number.isFinite(norm) || norm === 0) {
    throw new Error('The embedding model returned an invalid query vector.');
  }
  for (let d = 0; d < query.length; d++) query[d] /= norm;
  const { centroids, centroidMs } = await loadSearchCentroids(index, codec, { signal });
  const found = await searchVectors(index, query, { centroids, probes, k, signal, onProgress, cache });
  const { results, metadataMs, metadataUnavailable, downloads: textDownloads, failures } = await loadResultText(index, found.results, { signal, cache });
  return { results, candidates: found.candidates, clusters: found.selectedClusters.length, bytes: found.bytes,
    centroidMs, vectorMs: found.queryMs, metadataMs,
    metadataUnavailable, vectorDownloads: found.downloads, textDownloads, failures, elapsedMs: performance.now() - started };
}
