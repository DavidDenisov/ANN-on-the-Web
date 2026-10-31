import { CODEC_LABELS, CLUSTER_COUNTS, CLUSTER_FETCH_CONCURRENCY, networkPolicy, fetchJson, fetchBytes, loadCentroids, searchVectors } from './retrieval.mjs';

/**
 * @typedef {import('./retrieval.mjs').Codec} Codec
 * @typedef {{clusters: number, probes: number, codec: Codec,
 *   accuracy_only: boolean, memberships: string[]}} PlanEntry
 * @typedef {{mean: number, p95: number}} TimingSummary
 * @typedef {{url: string, dimensions: number, count: number, metric: string, truth_k: number,
 *   query_count: number, id: string, noise: {level: number}, queriesData: Float32Array,
 *   truthData: Uint32Array, indexes: {clusters: number, path: string, url: string}[]}} LoadedSuite
 * @typedef {{repeat: number, query: number, recall: number, targetId: number, resultIds: number[],
 *   candidates: number, bytes: number, selectedClusters: number[], queryMs?: number,
 *   downloads: (Omit<import('./retrieval.mjs').VectorDownload, 'downloadMs'|'retries'> & {downloadMs?: number, retries?: number})[]}} QueryTrial
 * @typedef {import('./retrieval.mjs').CentroidSetup & {probes: number, k: number,
 *   queries: number, repeats: number, recall: number, trials: QueryTrial[], centroidBytes: number,
 *   centroidDownloadMs: TimingSummary, centroidDecodeMs: TimingSummary, bytesPerQuery: number,
 *   candidatesMean: number, metric: 'source-recovery', noise: LoadedSuite['noise'], fixture_id?: string,
 *   network_mode?: import('./retrieval.mjs').NetworkOptions['networkMode'],
 *   asset_base_url?: import('./retrieval.mjs').NetworkOptions['assetsBaseUrl'],
 *   asset_base_urls?: import('./retrieval.mjs').NetworkOptions['assetsBaseUrls'], hosting_bundle_id?: string,
 *   cluster_fetch_cache: 'force-cache'|'no-store', cluster_fetch_concurrency: number}} BenchmarkSummary
 * @typedef {(BenchmarkSummary & {accuracy_only: true, queryMs?: never}) |
 *   (BenchmarkSummary & {accuracy_only: false, queryMs: TimingSummary})} BenchmarkResult
 */

export const PAPER_SETTINGS = Object.freeze({
  cluster_counts: CLUSTER_COUNTS, cluster_probe_counts: [1, 4, 8],
  probe_cluster_counts: [16384], probe_counts: [1, 2, 3, 4, 6, 8, 10],
  quantization_clusters: 16384, quantization_probes: 4,
  codecs: /** @type {Codec[]} */ (Object.keys(CODEC_LABELS)), k: 1, query_count: 10000,
  repeats: 1, centroid_setup_repeats: 1000,
});

/** @returns {PlanEntry[]} */
export function paperPlan() {
  /** @type {Map<string, PlanEntry>} */
  const plan = new Map();
  /** @param {number} clusters @param {number} probes @param {Codec} codec @param {string} membership */
  function add(clusters, probes, codec, membership) {
    const accuracy_only = membership === 'codecs';
    const key = `${clusters}|${probes}|${codec}|${accuracy_only}`;
    let entry = plan.get(key);
    if (!entry) {
      entry = { clusters, probes, codec, accuracy_only, memberships: [] };
      plan.set(key, entry);
    }
    entry.memberships.push(membership);
  }
  for (const clusters of PAPER_SETTINGS.cluster_counts) {
    for (const probes of PAPER_SETTINGS.cluster_probe_counts) add(clusters, probes, 'f16', 'counts');
  }
  for (const probes of PAPER_SETTINGS.probe_counts) add(16384, probes, 'f16', 'probes');
  for (const codec of PAPER_SETTINGS.codecs) add(16384, 4, codec, 'codecs');
  return [...plan.values()];
}

/** @returns {PlanEntry[]} */
export function remaining32kPlan() {
  return [
    ...[2, 3, 6, 10].map(/** @returns {PlanEntry} */ probes => ({ clusters: 32768, probes, codec: 'f16',
      accuracy_only: false, memberships: ['probes'] })),
    { clusters: 32768, probes: 4, codec: 'sq6', accuracy_only: false, memberships: ['codecs', 'default'] },
    ...(/** @type {Codec[]} */ (['sq8', 'sq4', 'int8', 'pq4'])).map(codec => ({ clusters: 32768, probes: 4, codec,
      accuracy_only: true, memberships: ['codecs'] })),
  ];
}

const integer = (value, minimum = 1) => Number.isSafeInteger(value) && value >= minimum;
const requireValue = (condition, message) => { if (!condition) throw new Error(message); };
const resolveUrl = (path, base = globalThis.location?.href) => new URL(path, base).href;
const mean = values => values.reduce((sum, value) => sum + value, 0) / values.length;
const summary = values => ({ mean: mean(values), p95: [...values].sort((a, b) => a - b)[Math.ceil(values.length * 0.95) - 1] });

/**
 * @param {string|URL} path
 * @param {string} expectedId
 * @param {import('./retrieval.mjs').NetworkOptions} options
 * @returns {Promise<{version: number, bundle_id: string, asset_base_url: string, asset_base_urls: string[]}>}
 */
export async function verifyHostingManifest(path, expectedId, options) {
  const policy = networkPolicy(options);
  requireValue(policy.networkMode === 'internet' && typeof expectedId === 'string' && expectedId.length > 0,
    'A hosting manifest needs Internet mode and an expected bundle identity.');
  const manifest = await fetchJson(resolveUrl(path), policy);
  requireValue(manifest.version === 1 && manifest.bundle_id === expectedId,
    'The hosted deployment does not match the prepared bundle. Deploy this run before measuring.');
  requireValue(manifest.asset_base_url === policy.assetsBaseUrl && Array.isArray(manifest.asset_base_urls)
    && manifest.asset_base_urls.every(value => typeof value === 'string')
    && JSON.stringify([...manifest.asset_base_urls].sort()) === JSON.stringify([...policy.assetsBaseUrls].sort()),
  'The hosted deployment asset allowlist does not match this run.');
  return manifest;
}

/** @param {string|URL} path @param {import('./retrieval.mjs').NetworkOptions} [options] @returns {Promise<LoadedSuite>} */
export async function loadSuite(path, options) {
  const url = resolveUrl(path);
  const policy = networkPolicy(options);
  const suite = await fetchJson(url, policy);
  requireValue(suite.version === 1 && suite.dimensions === 384 && suite.count === 35167920
    && suite.query_count === 10000 && suite.truth_k === 1
    && typeof suite.id === 'string' && suite.id.length > 0, 'Invalid benchmark suite.');
  requireValue(suite.metric === 'source-recovery' && suite.query_kind === 'noisy-passages'
    && [0.05, 0.15, 0.4].includes(suite.noise?.level),
    'Source recovery requires one target passage per query.');
  requireValue(suite.queries?.count === suite.query_count && suite.truth?.k === suite.truth_k
    && typeof suite.queries.file === 'string' && typeof suite.truth.file === 'string'
    && Array.isArray(suite.indexes) && suite.indexes.length > 0, 'Incomplete benchmark suite.');
  for (const index of suite.indexes) {
    requireValue(integer(index.clusters) && typeof index.path === 'string', 'Invalid suite index.');
  }
  requireValue(JSON.stringify(suite.indexes.map(index => index.clusters).sort((a, b) => a - b))
    === JSON.stringify(PAPER_SETTINGS.cluster_counts), 'Use all three paper indexes.');
  const [queries, truth] = await Promise.all([
    fetchBytes(resolveUrl(suite.queries.file, url), policy), fetchBytes(resolveUrl(suite.truth.file, url), policy),
  ]);
  requireValue(queries.bytes === suite.query_count * suite.dimensions * 4, 'Invalid query byte length.');
  requireValue(truth.bytes === suite.query_count * suite.truth_k * 4, 'Invalid ground-truth byte length.');
  const queryView = new DataView(queries.buffer);
  const queriesData = new Float32Array(suite.query_count * suite.dimensions);
  const truthView = new DataView(truth.buffer);
  const truthData = new Uint32Array(suite.query_count * suite.truth_k);
  for (let query = 0; query < suite.query_count; query++) {
    let norm = 0;
    for (let d = 0; d < suite.dimensions; d++) {
      const offset = query * suite.dimensions + d;
      const value = queryView.getFloat32(offset * 4, true);
      requireValue(Number.isFinite(value), 'Non-finite query value.');
      queriesData[offset] = value;
      norm += value * value;
    }
    requireValue(norm > 0, 'Zero query vector.');
    norm = Math.sqrt(norm);
    for (let d = 0; d < suite.dimensions; d++) queriesData[query * suite.dimensions + d] /= norm;
    const id = truthView.getUint32(query * 4, true);
    requireValue(id < suite.count, 'Invalid target passage ID.');
    truthData[query] = id;
  }
  return { ...suite, url, queriesData, truthData,
    indexes: suite.indexes.map(index => ({ ...index, url: resolveUrl(index.path, url) })) };
}

/**
 * @param {import('./retrieval.mjs').Index} index
 * @param {import('./retrieval.mjs').Codec[]} codecs
 * @param {import('./retrieval.mjs').NetworkOptions & {
 *   onProgress?: (progress: {completed: number, total: number, bytes: number}) => void}} [options]
 * @returns {Promise<{clusters: number, cluster_files: number, cluster_bytes: number, centroid_files: number}>}
 */
export async function warmIndexAssets(index, codecs, { onProgress, ...options } = {}) {
  for (const codec of codecs) await loadCentroids(index, codec, options);
  const controller = new AbortController();
  const policy = networkPolicy({ ...options, cache: 'no-store', signal: controller.signal });
  const abort = () => controller.abort(options.signal.reason);
  options.signal?.addEventListener('abort', abort, { once: true });
  if (options.signal?.aborted) abort();
  let next = 0, completed = 0, bytes = 0;
  try {
    await Promise.all(Array.from({ length: Math.min(CLUSTER_FETCH_CONCURRENCY, index.clusters) }, async () => {
      try {
        while (next < index.clusters) {
          policy.signal.throwIfAborted();
          const cluster = next++;
          const file = index.cluster_files[cluster];
          const downloaded = await fetchBytes(resolveUrl(file.file, index.url), policy);
          requireValue(downloaded.bytes === file.bytes,
            'Cluster download byte length does not match manifest.');
          bytes += downloaded.bytes;
          completed++;
          onProgress?.({ completed, total: index.clusters, bytes });
        }
      } catch (error) {
        controller.abort(error);
      }
    }));
    policy.signal.throwIfAborted();
  } finally {
    options.signal?.removeEventListener('abort', abort);
  }
  return { clusters: index.clusters, cluster_files: completed, cluster_bytes: bytes, centroid_files: codecs.length };
}

/**
 * Query timers exclude centroid loading and shared query/truth setup.
 * Accuracy-only results omit queryMs, including per-trial query/download timings.
 * @param {import('./retrieval.mjs').Index} index
 * @param {LoadedSuite} suite
 * @param {Omit<import('./retrieval.mjs').NetworkOptions, 'cache'> & {codec?: import('./retrieval.mjs').Codec,
 *   probes?: number, accuracyOnly?: boolean,
 *   onProgress?: (progress: {completed: number, total: number, repeat: number, query: number}) => void,
 *   centroidSetup?: import('./retrieval.mjs').CentroidSetup}} [options]
 * @returns {Promise<BenchmarkResult>}
 */
export async function benchmark(index, suite, {
  codec = 'f16', probes = 4,
  accuracyOnly = false, onProgress,
  networkMode, assetsBaseUrl, assetsBaseUrls, signal, centroidSetup,
} = {}) {
  const k = 1, queryCount = 10000;
  const policy = networkPolicy({ networkMode, assetsBaseUrl, assetsBaseUrls, signal, cache: accuracyOnly ? 'force-cache' : 'no-store' });
  requireValue(typeof accuracyOnly === 'boolean', 'Invalid accuracy-only setting.');
  requireValue(index.dimensions === suite.dimensions && index.count === suite.count, 'Suite and index do not match.');
  requireValue(suite.metric === 'source-recovery' && suite.truth_k === 1, 'Invalid benchmark metric.');
  requireValue([...paperPlan(), ...remaining32kPlan()].some(item => item.clusters === index.clusters && item.probes === probes
    && item.codec === codec && item.accuracy_only === accuracyOnly)
    && suite.query_count === queryCount, 'Use a reported paper configuration.');
  // Load once per configuration, outside query timers.
  const loaded = await loadCentroids(index, codec, { ...policy, cache: 'no-store' });
  requireValue(!centroidSetup || (centroidSetup.centroidTrials?.[0]?.requested_url === loaded.requestedUrl
    && centroidSetup.centroidTrials[0].bytes === loaded.bytes
    && centroidSetup.clusters === index.clusters && centroidSetup.codec === codec),
  'Centroid setup measurements do not match this test.');
  const trials = [];
  const centroidTrials = centroidSetup?.centroidTrials ?? loaded.setupTrials;
  performance.setResourceTimingBufferSize?.(probes * 3 + 16);
  for (let queryIndex = 0; queryIndex < queryCount; queryIndex++) {
    const query = suite.queriesData.subarray(queryIndex * suite.dimensions, (queryIndex + 1) * suite.dimensions);
    const targetId = suite.truthData[queryIndex];
    performance.clearResourceTimings?.();
    const { results, selectedClusters: selected, candidates, queryMs, bytes, downloads } = await searchVectors(index, query,
      { centroids: loaded.centroids, probes, k, accuracyOnly, ...policy }).catch(error => {
        policy.signal?.throwIfAborted();
        throw new Error(`Query ${queryIndex + 1}/${queryCount}: ${error.message}`, { cause: error });
      });
    const resultIds = results.map(item => item.id);
    const recall = resultIds[0] === targetId ? 1 : 0;
    trials.push({ repeat: 0, query: queryIndex, recall,
      targetId,
      ...(!accuracyOnly ? { queryMs } : {}),
      candidates, bytes,
      selectedClusters: selected, resultIds, downloads: accuracyOnly
        ? downloads.map(({ cluster_id, requested_url, resolved_url, bytes }) =>
          ({ cluster_id, requested_url, resolved_url, bytes })) : downloads });
    onProgress?.({ completed: trials.length, total: queryCount, repeat: 0, query: queryIndex });
    await new Promise(resolve => setTimeout(resolve, 0));
  }
  const recall = mean(trials.map(trial => trial.recall));
  return { codec, metric: 'source-recovery', network_mode: networkMode, ...(suite.id ? { fixture_id: suite.id } : {}),
    ...(networkMode === 'internet' ? { asset_base_url: assetsBaseUrl, asset_base_urls: policy.assetsBaseUrls } : {}),
    noise: suite.noise,
    clusters: index.clusters, probes, k, queries: queryCount, repeats: 1, recall,
    accuracy_only: accuracyOnly, cluster_fetch_cache: policy.cache,
    cluster_fetch_concurrency: CLUSTER_FETCH_CONCURRENCY,
    ...(!accuracyOnly ? { queryMs: summary(trials.map(trial => trial.queryMs)) } : {}),
    centroidDownloadMs: summary(centroidTrials.map(trial => trial.downloadMs)),
    centroidDecodeMs: summary(centroidTrials.map(trial => trial.decodeMs)),
    centroidBytes: mean(centroidTrials.map(trial => trial.bytes)),
    candidatesMean: mean(trials.map(trial => trial.candidates)), bytesPerQuery: mean(trials.map(trial => trial.bytes)),
    trials, centroidTrials,
    centroid_setup_id: centroidSetup?.centroid_setup_id ?? loaded.setupId, centroid_setup_repeats: centroidTrials.length };
}
