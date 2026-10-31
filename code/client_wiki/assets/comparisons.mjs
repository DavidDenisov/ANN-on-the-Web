import { loadSuite, warmIndexAssets, benchmark, verifyHostingManifest, PAPER_SETTINGS, paperPlan } from './benchmark.mjs';
import { loadIndex, loadCentroids, internetAssetUrl, CODEC_LABELS } from './retrieval.mjs';

const requireValue = (condition, message) => { if (!condition) throw new Error(message); };

function validateConfig(config) {
  requireValue(config?.version === 1 && typeof config.id === 'string' && config.id.length > 0
    && typeof config.token === 'string' && config.token.length > 0, 'Invalid comparison run configuration.');
  const noise = [['small', 0.05], ['medium', 0.15], ['large', 0.4]];
  requireValue(Array.isArray(config.suites) && config.suites.length === 3
    && config.suites.every((item, i) => item?.label === noise[i][0] && item.noise === noise[i][1]
      && typeof item.url === 'string' && item.url.length > 0), 'Use the paper noise suites.');
  for (const [field, value] of Object.entries(PAPER_SETTINGS)) {
    requireValue(JSON.stringify(config[field]) === JSON.stringify(value), `Use the paper's ${field}.`);
  }
  requireValue(Number.isSafeInteger(config.seed) && config.seed >= 0 && config.seed <= 0x7fffffff,
    'Invalid recorded seed.');
  requireValue(config.results_url === '/api/results', 'Invalid results destination.');
  requireValue(typeof config.centroid_only === 'boolean', 'Invalid centroid-only setting.');
  requireValue(config.network_mode === 'internet', 'Comparison runs require Internet measurements.');
  requireValue(config.fixture_ids && config.suites.every(item =>
    typeof config.fixture_ids[item.label] === 'string' && config.fixture_ids[item.label].length > 0),
  'Missing expected fixture identities.');
  requireValue(typeof config.assets_base_url === 'string', 'Internet measurements require an asset base URL.');
  internetAssetUrl(config.assets_base_url, config.assets_base_url);
  const bases = config.asset_base_urls;
  requireValue(Array.isArray(bases) && bases.length > 0 && bases.includes(config.assets_base_url),
    'The asset allowlist must include the canonical asset base URL.');
  for (const item of config.suites) internetAssetUrl(item.url, bases);
  requireValue(typeof config.hosting_manifest_url === 'string' && typeof config.hosting_bundle_id === 'string'
    && config.hosting_bundle_id.length > 0, 'Missing hosting deployment identity.');
  internetAssetUrl(config.hosting_manifest_url, bases);
  return config;
}

function buildPlan(config) {
  const values = config.suites.flatMap(noise => paperPlan().map(item => ({ ...item, noise_label: noise.label })));
  let state = config.seed >>> 0;
  // Randomize configuration order so connection drift is not tied to one method or noise level.
  for (let i = values.length - 1; i > 0; i--) {
    state += 0x6d2b79f5;
    let value = Math.imul(state ^ state >>> 15, state | 1);
    value ^= value + Math.imul(value ^ value >>> 7, value | 61);
    const j = Math.floor(((value ^ value >>> 14) >>> 0) / 2 ** 32 * (i + 1));
    [values[i], values[j]] = [values[j], values[i]];
  }
  return values;
}

const $ = id => document.getElementById(id);

async function run() {
  let config;
  let resultsUrl;
  let stage = 'Starting comparison';
  const status = (message, error = false) => {
    if (!error) stage = message;
    $('runStatus').textContent = message;
    $('runStatus').className = `status ${error ? 'error' : 'loading'}`;
  };
  async function save(packet) {
    let response;
    try {
      response = await fetch(resultsUrl, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id: config.id, token: config.token, ...packet }),
      });
    } catch (error) {
      throw new Error(`Saving results to the local runner failed: ${resultsUrl}. ${error.message}`, { cause: error });
    }
    if (!response.ok) {
      const detail = await response.text();
      throw new Error(`Saving results failed (${response.status}): ${detail.slice(0, 160)}`);
    }
  }
  try {
    const configUrl = new URL('/experiment/run.json', location.href);
    const response = await fetch(configUrl, { cache: 'no-store' });
    requireValue(response.ok, `Loading run configuration failed (${response.status}).`);
    config = validateConfig(await response.json());
    if (config.centroid_only) $('runWarmup').textContent = 'Each centroid file is loaded once before its 1,000 timed trials.';
    resultsUrl = new URL(config.results_url, configUrl);
    const network = { networkMode: config.network_mode, assetsBaseUrl: config.assets_base_url,
      assetsBaseUrls: config.asset_base_urls };
    $('runNetwork').textContent = `Internet measurements · ${config.assets_base_url}`;
    status('Checking the hosted deployment identity…');
    await verifyHostingManifest(config.hosting_manifest_url, config.hosting_bundle_id, network);
    const suites = new Map();
    let reference;
    for (const item of config.suites) {
      status(`Loading ${item.label} noise queries…`);
      const suite = await loadSuite(new URL(item.url, configUrl), network);
      requireValue(suite.id === config.fixture_ids[item.label],
        `The ${item.label} hosted fixture does not match the prepared local data. Upload this run's complete data directory.`);
      requireValue(suite.noise.level === item.noise, 'The prepared suite has the wrong noise level.');
      if (reference) {
        requireValue(suite.truthData.every((id, i) => id === reference.truthData[i]),
          'All noise levels must share the same query source IDs.');
      }
      reference ??= suite;
      suites.set(item.label, suite);
    }
    const plan = buildPlan(config);
    const indexes = new Map();
    for (const suite of suites.values()) {
      for (const descriptor of suite.indexes) {
        if (!indexes.has(descriptor.url)) indexes.set(descriptor.url, await loadIndex(descriptor.url, network));
        const index = indexes.get(descriptor.url);
        requireValue(index.clusters === descriptor.clusters, 'A prepared index does not match its suite.');
      }
    }
    const warmup = [];
    for (const index of config.centroid_only ? [] : indexes.values()) {
      const codecs = [...new Set(plan.filter(item => item.clusters === index.clusters).map(item => item.codec))];
      status(`Warming all assets: ${index.clusters} clusters…`);
      warmup.push(await warmIndexAssets(index, codecs, { ...network,
        onProgress: progress => { $('runDetail').textContent = `Warm-up: ${progress.completed} / ${progress.total} cluster files. Measurements have not started.`; },
      }));
    }
    $('runDetail').textContent = 'Measuring centroid loads separately from query latency.';
    const centroidSetups = [];
    for (const index of indexes.values()) {
      const codecs = new Set(plan.filter(item => item.clusters === index.clusters).map(item => item.codec));
      for (const codec of codecs) {
        status(`Measuring centroid loads: ${index.clusters} clusters · ${CODEC_LABELS[codec]}…`);
        if (config.centroid_only) await loadCentroids(index, codec, network);
        const loaded = await loadCentroids(index, codec, { ...network, setupRepeats: 1000 });
        centroidSetups.push({ clusters: index.clusters, codec, centroid_setup_id: loaded.setupId,
          centroid_setup_repeats: loaded.setupTrials.length, centroidTrials: loaded.setupTrials });
      }
    }
    await save({ warmup, centroid_setups: centroidSetups, user_agent: navigator.userAgent });
    if (config.centroid_only) {
      await save({ complete: true, total: 0, user_agent: navigator.userAgent });
      status(`Finished. ${centroidSetups.length} centroid files × ${config.centroid_setup_repeats} trials saved.`);
      $('runStatus').className = 'status success';
      $('runDetail').textContent = 'Fetch and decode results are saved. This tab can be closed.';
      return;
    }
    $('runProgress').max = plan.length;
    $('runMatrix').textContent = `${config.suites.map(item => `${item.label}: ${item.noise}`).join(' · ')} noise magnitude. `
      + `8,192 clusters at ${config.cluster_probe_counts_8k.join(', ')} probes; `
      + `other cluster counts at ${config.cluster_probe_counts.join(', ')} probes; `
      + `probe sweeps at ${config.probe_cluster_counts.join(', ')} clusters; ${Object.keys(CODEC_LABELS).length} centroid formats. `
      + `${plan.filter(item => !item.accuracy_only).length} latency configurations × 1 pass; `
      + `${plan.filter(item => item.accuracy_only).length} cached accuracy checks × 1 pass. ${config.query_count} queries each. `
      + `Accuracy: source recovery@${config.k}.`;
    const rows = [];
    for (const [position, item] of plan.entries()) {
      const suite = suites.get(item.noise_label);
      const descriptor = suite.indexes.find(index => index.clusters === item.clusters);
      status(`${position + 1} / ${plan.length} · ${item.noise_label} noise · ${CODEC_LABELS[item.codec]} · `
        + `${item.clusters} clusters · ${item.probes} probed`);
      const result = await benchmark(indexes.get(descriptor.url), suite, {
        codec: item.codec, probes: item.probes, accuracyOnly: item.accuracy_only,
        ...network, centroidSetup: centroidSetups.find(setup => setup.clusters === item.clusters && setup.codec === item.codec),
        onProgress: progress => { $('runDetail').textContent = `Query trial ${progress.completed} / ${progress.total}. `
          + `${position} configurations saved.`; },
      });
      result.hosting_bundle_id = config.hosting_bundle_id;
      await save({ noise_label: item.noise_label, memberships: item.memberships, result, user_agent: navigator.userAgent });
      const { codec, clusters, probes, recall, queryMs, accuracy_only } = result;
      rows.unshift({ noise: item.noise_label, result: { codec, clusters, probes, recall, queryMs, accuracy_only } });
      if (rows.length > 20) rows.pop();
      $('runRows').replaceChildren(...rows.map(row => {
        const tr = document.createElement('tr');
        for (const value of [row.noise, CODEC_LABELS[row.result.codec], row.result.clusters, row.result.probes,
          `${(100 * row.result.recall).toFixed(1)}%`,
          row.result.accuracy_only ? 'Accuracy only (cache allowed)'
            : `${row.result.queryMs.mean.toFixed(2)} / ${row.result.queryMs.p95.toFixed(2)} ms`]) {
          const td = document.createElement('td');
          td.textContent = value;
          tr.append(td);
        }
        return tr;
      }));
      $('runProgress').value = position + 1;
    }
    status('All measurements are saved. Finishing the comparison files…');
    await save({ complete: true, total: plan.length, user_agent: navigator.userAgent });
    status(`Finished. ${plan.length} configurations measured and saved.`);
    $('runStatus').className = 'status success';
    $('runDetail').textContent = 'Results are saved in the output folder printed by the test script. This tab can be closed.';
  } catch (error) {
    const message = `${stage}: ${error.message}`;
    status(message, true);
    $('runDetail').textContent = 'The run stopped. Completed measurements remain saved; see the test script terminal for details.';
    if (config && resultsUrl) {
      try { await save({ error: message, user_agent: navigator.userAgent }); } catch { /* Keep the original failure visible. */ }
    }
  }
}

run();
