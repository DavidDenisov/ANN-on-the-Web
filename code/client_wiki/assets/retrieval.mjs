/**
 * @typedef {'f16'|'sq8'|'sq6'|'sq4'|'int8'|'pq4'} Codec
 * @typedef {{networkMode?: 'internet', assetsBaseUrl?: string, assetsBaseUrls?: string[],
 *   signal?: AbortSignal, cache?: 'force-cache'|'no-store'}} NetworkOptions
 * @typedef {{file: string, bytes: number, rmse: number}} CentroidDescriptor
 * @typedef {{format: string, count: number, batch_size: number, pattern: string,
 *   files_per_deployment: number, base_urls: string[]}} PassageMetadata
 * @typedef {{url: string, dimensions: number, count: number, clusters: number, record_bytes: number,
 *   cluster_sizes: number[], cluster_files: {file: string, bytes: number}[],
 *   centroids: Record<Codec, CentroidDescriptor>, embedding_model?: string, metadata?: PassageMetadata}} Index
 * @typedef {{buffer: ArrayBuffer, bytes: number, downloadMs: number, retries: number,
 *   requestedUrl: string, resolvedUrl: string}} AssetDownload
 * @typedef {{id: number, distance: number}} SearchMatch
 * @typedef {{cluster_id: number, requested_url: string, resolved_url: string,
 *   bytes: number, downloadMs: number, retries: number}} VectorDownload
 * @typedef {{completed: number, total: number, candidates: number}} SearchProgress
 * @typedef {NetworkOptions & {centroids: Float32Array, probes?: number, k?: number,
 *   accuracyOnly?: boolean, onProgress?: (progress: SearchProgress) => void}} VectorSearchOptions
 * @typedef {{results: SearchMatch[], candidates: number, selectedClusters: number[],
 *   queryMs: number, bytes: number, downloads: VectorDownload[]}} VectorSearchResult
 * @typedef {{setup_id: string, sample: number, bytes: number, downloadMs: number, decodeMs: number,
 *   retries: number, requested_url: string, resolved_url: string}} CentroidTrial
 * @typedef {{clusters: number, codec: Codec, centroid_setup_id: string,
 *   centroid_setup_repeats: number, centroidTrials: CentroidTrial[]}} CentroidSetup
 */

export const CODEC_LABELS = Object.freeze({
  f16: 'Float16', sq8: 'SQ8', sq6: 'SQ6', sq4: 'SQ4',
  int8: 'INT8 (sym.)', pq4: 'PQ4',
});

export const CLUSTER_COUNTS = Object.freeze([8192, 16384, 32768]);

const now = () => performance.now();
const integer = (value, minimum = 1) => Number.isSafeInteger(value) && value >= minimum;
const requireValue = (condition, message) => { if (!condition) throw new Error(message); };
const resolveUrl = (path, base = globalThis.location?.href) => new URL(path, base).href;
const scalarBits = { sq8: 8, sq6: 6, sq4: 4 };
export const CLUSTER_FETCH_CONCURRENCY = 16;


/** @param {string|URL} value @param {string|string[]} [base] @returns {string} */
export function internetAssetUrl(value, base) {
  const url = new URL(value);
  const host = url.hostname.replace(/^\[|\]$/g, '').toLowerCase().replace(/\.$/, '');
  requireValue(url.protocol === 'https:' && !url.username && !url.password,
    'Internet measurements require public HTTPS asset URLs.');
  let publicHost = host.includes('.') && !/(^|\.)(localhost|local|internal|test|invalid)$/.test(host);
  if (/^\d+\.\d+\.\d+\.\d+$/.test(host)) {
    const [a, b, c] = host.split('.').map(Number);
    publicHost = !(a === 0 || a === 10 || a === 127 || a >= 224 || (a === 100 && b >= 64 && b <= 127)
      || (a === 169 && b === 254) || (a === 172 && b >= 16 && b <= 31)
      || (a === 192 && (b === 168 || (b === 0 && (c === 0 || c === 2))))
      || (a === 198 && (b === 18 || b === 19 || (b === 51 && c === 100)))
      || (a === 203 && b === 0 && c === 113));
  } else if (host.includes(':')) {
    publicHost = /^[23][0-9a-f]{3}:/.test(host) && !host.startsWith('2001:db8:');
  }
  requireValue(publicHost, 'Internet measurements cannot use localhost, private or reserved network addresses.');
  if (base) {
    const roots = (Array.isArray(base) ? base : [base]).map(value => {
      const root = new URL(internetAssetUrl(value));
      requireValue(root.pathname.endsWith('/') && !root.search && !root.hash,
        'The Internet asset base URL must end with / and have no query or fragment.');
      return root;
    });
    const pathname = decodeURIComponent(url.pathname);
    requireValue(!pathname.split('/').includes('..')
      && roots.some(root => url.origin === root.origin && pathname.startsWith(decodeURIComponent(root.pathname))),
      'An asset URL escapes the configured Internet asset directory.');
  }
  return url.href;
}

/** @param {NetworkOptions} [options] @returns {NetworkOptions} */
export function networkPolicy({ networkMode, assetsBaseUrl, assetsBaseUrls, signal, cache = 'no-store' } = {}) {
  requireValue(networkMode === undefined || networkMode === 'internet', 'Invalid network measurement mode.');
  requireValue(cache === 'no-store' || cache === 'force-cache', 'Invalid fetch cache policy.');
  if (networkMode === 'internet') {
    requireValue(typeof assetsBaseUrl === 'string', 'Internet measurements require an asset base URL.');
    internetAssetUrl(assetsBaseUrl, assetsBaseUrl);
    assetsBaseUrls ??= [assetsBaseUrl];
    requireValue(Array.isArray(assetsBaseUrls) && assetsBaseUrls.length > 0
      && assetsBaseUrls.every(value => typeof value === 'string') && assetsBaseUrls.includes(assetsBaseUrl),
    'The Internet asset allowlist must include the canonical asset base URL.');
    internetAssetUrl(assetsBaseUrl, assetsBaseUrls);
    requireValue(!globalThis.navigator?.serviceWorker?.controller,
      'Internet measurements require a page without an active service worker.');
  }
  return { networkMode, assetsBaseUrl, assetsBaseUrls, signal, cache };
}

async function fetchAsset(url, policy) {
  if (policy.networkMode === 'internet') internetAssetUrl(url, policy.assetsBaseUrls);
  let response;
  try {
    response = await fetch(url, { cache: policy.cache ?? 'no-store', signal: policy.signal,
      ...(policy.networkMode === 'internet' ? { mode: 'cors', credentials: 'omit' } : {}) });
  } catch (error) {
    policy.signal?.throwIfAborted();
    throw new Error(policy.networkMode === 'internet'
      ? `Internet asset download failed: ${url}. Check connectivity, TLS, browser blocking, and CORS. ${error.message}`
      : error.message, { cause: error });
  }
  if (!response.ok) {
    await response.body?.cancel();
    throw Object.assign(new Error(`Download failed (${response.status}): ${url}`), { status: response.status });
  }
  if (policy.networkMode === 'internet') {
    requireValue(!['opaque', 'opaqueredirect'].includes(response.type) && response.url,
      'Internet downloads require a visible final response URL; opaque responses cannot be measured.');
    internetAssetUrl(response.url);
  }
  return response;
}

/** @param {string} url @param {NetworkOptions} [options] @returns {Promise<AssetDownload>} */
export async function fetchBytes(url, options) {
  const policy = networkPolicy(options);
  const started = now();
  for (let retries = 0; ; retries++) {
    policy.signal?.throwIfAborted();
    let response;
    try {
      response = await fetchAsset(url, policy);
      let buffer;
      try {
        buffer = await response.arrayBuffer();
      } catch (error) {
        throw new Error(`Asset body download failed: ${url}. ${error.message}`, { cause: error });
      }
      const downloadMs = now() - started;
      return { buffer, bytes: buffer.byteLength, downloadMs, retries,
        requestedUrl: url, resolvedUrl: response.url || url };
    } catch (error) {
      policy.signal?.throwIfAborted();
      const transient = error.cause instanceof TypeError || (response && error instanceof TypeError)
        || error.status === 408 || error.status === 429 || (error.status >= 500 && error.status <= 599);
      if (!transient || retries === 1) throw error;
      await new Promise(resolve => setTimeout(resolve, 500));
    }
  }
}

/** @param {string} url @param {NetworkOptions} [options] */
export async function fetchJson(url, options) {
  const response = await fetchAsset(url, networkPolicy(options));
  try {
    return await response.json();
  } catch (error) {
    options?.signal?.throwIfAborted();
    throw new Error(`Invalid JSON asset: ${url}. ${error.message}`, { cause: error });
  }
}

async function unzip(buffer, url) {
  requireValue(typeof DecompressionStream === 'function', 'This browser needs gzip DecompressionStream support.');
  try {
    return await new Response(new Blob([buffer]).stream().pipeThrough(new DecompressionStream('gzip'))).arrayBuffer();
  } catch (error) {
    throw new Error(`Invalid or truncated gzip data: ${url}.`, { cause: error });
  }
}

const halfScales = Float32Array.from({ length: 31 }, (_, exponent) => 2 ** (exponent - 25));

function halfToFloat(bits) {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 31;
  const fraction = bits & 1023;
  if (exponent === 31) return fraction ? NaN : sign * Infinity;
  return exponent ? sign * (1024 + fraction) * halfScales[exponent] : sign * fraction * 2 ** -24;
}

async function decodeCentroids(compressed, codec, dimensions, clusters, url) {
  requireValue(dimensions === 384 && CLUSTER_COUNTS.includes(clusters),
    'Use the paper centroid dimensions and cluster counts.');
  requireValue(Object.hasOwn(CODEC_LABELS, codec), 'Unsupported centroid codec.');
  const buffer = await unzip(compressed, url);
  const length = dimensions * clusters;
  const bits = scalarBits[codec];
  const blocks = dimensions / 8;
  let expected;
  if (codec === 'f16') {
    expected = length * 2;
  } else if (bits) {
    expected = dimensions * 8 + length * bits / 8;
  } else if (codec === 'int8') {
    expected = dimensions * 4 + length;
  } else {
    expected = 4 + dimensions * 64 + clusters * blocks / 2;
  }
  requireValue(buffer.byteLength === expected, `Invalid ${codec} centroid byte length.`);
  const view = new DataView(buffer);
  const minimum = bits ? new Float32Array(dimensions) : null;
  const scale = bits || codec === 'int8' ? new Float32Array(dimensions) : null;
  const codebook = codec === 'pq4' ? new Float32Array(dimensions * 16) : null;
  if (codec === 'pq4') {
    requireValue(view.getUint32(0, true) === 8, 'Invalid product quantization block size.');
    for (let i = 0; i < dimensions * 16; i++) {
      codebook[i] = view.getFloat32(4 + i * 4, true);
      requireValue(Number.isFinite(codebook[i]), 'Non-finite centroid codebook value.');
    }
  } else if (bits || codec === 'int8') {
    for (let d = 0; d < dimensions; d++) {
      scale[d] = view.getFloat32((bits ? dimensions + d : d) * 4, true);
      if (bits) minimum[d] = view.getFloat32(d * 4, true);
      requireValue(Number.isFinite(scale[d]) && scale[d] >= 0
        && (!bits || Number.isFinite(minimum[d])), 'Invalid centroid quantization parameters.');
    }
  }
  const values = new Float32Array(length);
  if (codec === 'f16') {
    for (let i = 0; i < length; i++) {
      values[i] = halfToFloat(view.getUint16(i * 2, true));
      requireValue(Number.isFinite(values[i]), 'Non-finite centroid value.');
    }
  } else if (codec === 'sq8') {
    const codes = new Uint8Array(buffer, dimensions * 8);
    let i = 0;
    for (let row = 0; row < clusters; row++) for (let d = 0; d < dimensions; d += 4, i += 4) {
      values[i] = minimum[d] + Math.fround(scale[d] * codes[i]);
      values[i + 1] = minimum[d + 1] + Math.fround(scale[d + 1] * codes[i + 1]);
      values[i + 2] = minimum[d + 2] + Math.fround(scale[d + 2] * codes[i + 2]);
      values[i + 3] = minimum[d + 3] + Math.fround(scale[d + 3] * codes[i + 3]);
      requireValue(Number.isFinite(values[i]) && Number.isFinite(values[i + 1])
        && Number.isFinite(values[i + 2]) && Number.isFinite(values[i + 3]), 'Non-finite centroid value.');
    }
  } else if (codec === 'sq6') {
    const codes = new Uint8Array(buffer, dimensions * 8);
    let i = 0, offset = 0;
    for (let row = 0; row < clusters; row++) for (let d = 0; d < dimensions; d += 4, i += 4, offset += 3) {
      const a = codes[offset], b = codes[offset + 1], c = codes[offset + 2];
      values[i] = minimum[d] + Math.fround(scale[d] * (a & 63));
      values[i + 1] = minimum[d + 1] + Math.fround(scale[d + 1] * ((a >>> 6) | ((b & 15) << 2)));
      values[i + 2] = minimum[d + 2] + Math.fround(scale[d + 2] * ((b >>> 4) | ((c & 3) << 4)));
      values[i + 3] = minimum[d + 3] + Math.fround(scale[d + 3] * (c >>> 2));
      requireValue(Number.isFinite(values[i]) && Number.isFinite(values[i + 1])
        && Number.isFinite(values[i + 2]) && Number.isFinite(values[i + 3]), 'Non-finite centroid value.');
    }
  } else if (codec === 'sq4') {
    const codes = new Uint8Array(buffer, dimensions * 8);
    let i = 0, offset = 0;
    for (let row = 0; row < clusters; row++) for (let d = 0; d < dimensions; d += 2, i += 2, offset++) {
      const code = codes[offset];
      values[i] = minimum[d] + Math.fround(scale[d] * (code & 15));
      values[i + 1] = minimum[d + 1] + Math.fround(scale[d + 1] * (code >>> 4));
      requireValue(Number.isFinite(values[i]) && Number.isFinite(values[i + 1]), 'Non-finite centroid value.');
    }
  } else if (codec === 'int8') {
    const codes = new Int8Array(buffer, dimensions * 4);
    let i = 0;
    for (let row = 0; row < clusters; row++) for (let d = 0; d < dimensions; d += 4, i += 4) {
      values[i] = Math.fround(scale[d] * codes[i]);
      values[i + 1] = Math.fround(scale[d + 1] * codes[i + 1]);
      values[i + 2] = Math.fround(scale[d + 2] * codes[i + 2]);
      values[i + 3] = Math.fround(scale[d + 3] * codes[i + 3]);
      requireValue(Number.isFinite(values[i]) && Number.isFinite(values[i + 1])
        && Number.isFinite(values[i + 2]) && Number.isFinite(values[i + 3]), 'Non-finite centroid value.');
    }
  } else {
    const codes = new Uint8Array(buffer, 4 + dimensions * 64);
    let codeIndex = 0, i = 0;
    for (let row = 0; row < clusters; row++) for (let block = 0; block < blocks; block++, codeIndex++) {
      const code = (codes[codeIndex >>> 1] >>> ((codeIndex & 1) * 4)) & 15;
      const word = block * 8 * 16 + code * 8;
      for (let d = 0; d < 8; d++) values[i++] = codebook[word + d];
    }
  }
  return values;
}

/** @param {string|URL} path @param {NetworkOptions} [options] @returns {Promise<Index>} */
export async function loadIndex(path, options) {
  const url = resolveUrl(path);
  const index = await fetchJson(url, networkPolicy(options));
  requireValue(index.version === 1 && index.metric === 'normalized-l2' && index.dimensions === 384
    && index.count === 35167920 && CLUSTER_COUNTS.includes(index.clusters)
    && index.record_bytes === 772,
  'Invalid cluster index.');
  requireValue(Array.isArray(index.cluster_sizes) && index.cluster_sizes.length === index.clusters
    && index.cluster_sizes.every(size => integer(size, 0))
    && index.cluster_sizes.reduce((sum, size) => sum + size, 0) === index.count,
  'Invalid cluster sizes.');
  requireValue(Array.isArray(index.cluster_files) && index.cluster_files.length === index.clusters
    && index.cluster_files.every(file => file && typeof file.file === 'string'
      && file.file.length > 0 && integer(file.bytes)),
  'Expected one published file descriptor per cluster.');
  requireValue(index.centroids && JSON.stringify(Object.keys(index.centroids).sort())
    === JSON.stringify([...Object.keys(CODEC_LABELS)].sort()), 'Use the six paper centroid formats.');
  for (const [codec, value] of Object.entries(index.centroids)) {
    requireValue(Object.hasOwn(CODEC_LABELS, codec) && typeof value.file === 'string'
      && integer(value.bytes) && Number.isFinite(value.rmse) && value.rmse >= 0, 'Invalid centroid descriptor.');
  }
  return { ...index, url };
}

function selectClusters(query, centroids, dimensions, probes) {
  const distances = [];
  for (let offset = 0; offset < centroids.length; offset += dimensions) {
    let distance = 0;
    for (let d = 0; d < dimensions; d++) distance += (query[d] - centroids[offset + d]) ** 2;
    distances.push({ id: offset / dimensions, distance });
  }
  distances.sort((a, b) => a.distance - b.distance || a.id - b.id);
  return distances.slice(0, probes).map(item => item.id);
}

function scoreCluster(index, cluster, buffer, query, k, state) {
  const { best, seen } = state;
  const vector = new Float64Array(index.dimensions);
  requireValue(buffer.byteLength === index.cluster_sizes[cluster] * index.record_bytes,
    'Invalid cluster record byte length.');
  const view = new DataView(buffer);
  for (let offset = 0; offset < buffer.byteLength; offset += index.record_bytes) {
    const id = view.getUint32(offset, true);
    const byte = id >>> 3;
    const bit = 1 << (id & 7);
    requireValue(id < index.count && !(seen[byte] & bit), 'Invalid or repeated passage ID.');
    seen[byte] |= bit;
    state.candidates++;
    let norm = 0;
    for (let d = 0; d < index.dimensions; d++) {
      const value = halfToFloat(view.getUint16(offset + 4 + d * 2, true));
      requireValue(Number.isFinite(value), 'Non-finite passage vector.');
      vector[d] = value;
      norm += value * value;
    }
    requireValue(norm > 0, 'Zero passage vector.');
    norm = Math.sqrt(norm);
    let distance = 0;
    for (let d = 0; d < index.dimensions; d++) distance += (query[d] - vector[d] / norm) ** 2;
    let rank = best.length;
    while (rank > 0 && (distance < best[rank - 1].distance
      || (distance === best[rank - 1].distance && id < best[rank - 1].id))) rank--;
    if (rank < k) {
      best.splice(rank, 0, { id, distance });
      if (best.length > k) best.pop();
    }
  }
}

async function fetchCentroids(index, codec, policy) {
  const descriptor = index.centroids[codec];
  const downloaded = await fetchBytes(resolveUrl(descriptor.file, index.url), policy);
  requireValue(downloaded.bytes === descriptor.bytes, 'Centroid download byte length does not match manifest.');
  const started = now();
  const centroids = await decodeCentroids(downloaded.buffer, codec, index.dimensions, index.clusters, downloaded.requestedUrl);
  return { centroids, ...downloaded, decodeMs: now() - started };
}

/**
 * @param {Index} index
 * @param {Codec} [codec]
 * @param {NetworkOptions & {setupRepeats?: 1|1000}} [options]
 * @returns {Promise<Omit<AssetDownload, 'buffer'> & {centroids: Float32Array, decodeMs: number,
 *   setupId: string, setupTrials: CentroidTrial[]}>}
 */
export async function loadCentroids(index, codec = 'f16', options) {
  requireValue(Object.hasOwn(index.centroids, codec), 'Unsupported centroid codec.');
  const url = resolveUrl(index.centroids[codec].file, index.url);
  const setupId = `${url}#${crypto.randomUUID()}`;
  const setupRepeats = options?.setupRepeats ?? 1;
  requireValue(setupRepeats === 1 || setupRepeats === 1000, 'Use one centroid preload or 1,000 setup trials.');
  const policy = networkPolicy(options);
  const setupTrials = [];
  let loaded;
  for (let sample = 0; sample < setupRepeats; sample++) {
    if (setupRepeats > 1) performance.clearResourceTimings?.();
    const { buffer, ...measured } = await fetchCentroids(index, codec, policy);
    loaded = measured;
    setupTrials.push({ setup_id: setupId, sample, bytes: loaded.bytes,
      downloadMs: loaded.downloadMs, decodeMs: loaded.decodeMs, retries: loaded.retries,
      requested_url: loaded.requestedUrl, resolved_url: loaded.resolvedUrl });
  }
  loaded.setupId = setupId;
  loaded.setupTrials = setupTrials;
  return loaded;
}

/**
 * Shared by interactive search and the measured query path.
 * @param {Index} index
 * @param {ArrayLike<number>} query
 * @param {VectorSearchOptions} options
 * @returns {Promise<VectorSearchResult>}
 */
export async function searchVectors(index, query, {
  centroids, probes = 4, k = 10, onProgress, accuracyOnly = false, ...options
} = {}) {
  const policy = networkPolicy(options);
  policy.signal?.throwIfAborted();
  requireValue(query?.length === index.dimensions && Array.from(query).every(Number.isFinite)
    && Array.from(query).some(value => value !== 0), 'Invalid query vector.');
  requireValue(centroids?.length === index.clusters * index.dimensions
    && integer(probes) && probes <= index.clusters && integer(k) && k <= index.count,
  'Invalid search options.');
  const clock = accuracyOnly ? () => 0 : now;
  const started = clock();
  const selected = selectClusters(query, centroids, index.dimensions, probes);
  const state = { best: [], seen: new Uint8Array(Math.ceil(index.count / 8)), candidates: 0 };
  const downloads = [];
  let completed = 0;
  let bytes = 0;
  const signal = policy.signal;
  const controller = new AbortController();
  const abort = () => controller.abort(signal.reason);
  signal?.addEventListener('abort', abort, { once: true });
  if (signal?.aborted) abort();
  policy.signal = controller.signal;
  let next = 0;
  // Each slot retains one cluster through scoring, bounding payload memory.
  try {
    // Drain every worker before propagating the first abort or failure.
    await Promise.all(Array.from({ length: Math.min(CLUSTER_FETCH_CONCURRENCY, selected.length) }, async () => {
      try {
        while (next < selected.length) {
          policy.signal.throwIfAborted();
          const cluster = selected[next++];
          const file = index.cluster_files[cluster];
          const downloaded = await fetchBytes(resolveUrl(file.file, index.url), policy);
          requireValue(downloaded.bytes === file.bytes, 'Cluster download byte length does not match manifest.');
          policy.signal.throwIfAborted();
          bytes += downloaded.bytes;
          downloads.push({ cluster_id: cluster,
            requested_url: downloaded.requestedUrl, resolved_url: downloaded.resolvedUrl,
            bytes: downloaded.bytes, downloadMs: downloaded.downloadMs, retries: downloaded.retries });
          const buffer = await unzip(downloaded.buffer, downloaded.requestedUrl);
          policy.signal.throwIfAborted();
          scoreCluster(index, cluster, buffer, query, k, state);
          onProgress?.({ completed: ++completed, total: selected.length, candidates: state.candidates });
        }
      } catch (error) {
        controller.abort(error);
      }
    }));
    policy.signal.throwIfAborted();
  } finally {
    signal?.removeEventListener('abort', abort);
  }
  const queryMs = clock() - started;
  policy.signal?.throwIfAborted();
  return { results: state.best, candidates: state.candidates, selectedClusters: selected,
    queryMs, bytes, downloads };
}

