import { CODEC_LABELS } from './retrieval.mjs';

const form = document.querySelector('#searchForm');
const query = document.querySelector('#query');
const searchButton = document.querySelector('#searchBtn');
const cancelButton = document.querySelector('#cancelBtn');
const clearButton = document.querySelector('#clearBtn');
const clusterCount = document.querySelector('#clusterCount');
const indexInfo = document.querySelector('#indexInfo');
const codec = document.querySelector('#codec');
const probes = document.querySelector('#probes');
const resultK = document.querySelector('#resultK');
const status = document.querySelector('#status');
const results = document.querySelector('#results');
const resultsInfo = document.querySelector('#resultsInfo');
const resultsPanel = document.querySelector('.results-panel');
const progress = document.querySelector('#progress');
const emptyResults = results.innerHTML;
let worker;
let busy = false;
let ready = false;
let requestId = 0;
const indexPaths = {
  8192: '../data/demo/pages/root/data/search-full-brotli/c8192/manifest.json',
  16384: '../data/demo/pages/root/data/search-full-brotli/manifest.json',
  32768: '../data/demo/pages/root/data/search-full-brotli/c32768/manifest.json',
};

function setBusy(value) {
  busy = value;
  for (const control of form.elements) control.disabled = value;
  if (!ready) {
    query.disabled = false;
    form.querySelectorAll('[data-query]').forEach(button => { button.disabled = false; });
  }
  searchButton.disabled = value || !ready || !query.value.trim();
  codec.disabled = value || !ready;
  probes.disabled = value || !ready;
  resultK.disabled = value || !ready;
  clearButton.disabled = false;
  clearButton.hidden = !query.value;
  cancelButton.hidden = !value;
  searchButton.textContent = value ? 'Working…' : '🔍 Search passages';
  resultsPanel.setAttribute('aria-busy', String(value));
}

function setStatus(message, kind = '') {
  status.className = `status ${kind}`;
  status.textContent = message;
}

function showError(message, stage) {
  progress.hidden = true;
  resultsInfo.textContent = stage === 'index' ? 'Choose a centroid count to load its index.' : 'Search could not be completed. Try again.';
  if (stage === 'index') {
    ready = false;
    indexInfo.textContent = 'This split could not be loaded. Choose another centroid count or reload the page to retry.';
  }
  setBusy(false);
  setStatus(message, 'error');
}

function renderResults(found) {
  results.replaceChildren();
  if (!found.length) {
    const message = document.createElement('p');
    message.className = 'placeholder-subtitle';
    message.textContent = 'No passages were found in the selected clusters. Try searching more clusters.';
    results.append(message);
    return;
  }
  for (const result of found) {
    const article = document.createElement('article');
    article.className = 'passage';
    const title = document.createElement('h3');
    if (typeof result.title === 'string' && result.title) {
      // Decode source-title entities while keeping literal markup as text.
      title.innerHTML = result.title.replaceAll('<', '&lt;');
      const pageTitle = title.textContent;
      const link = document.createElement('a');
      link.href = `https://en.wikipedia.org/wiki/${encodeURIComponent(pageTitle.replaceAll(' ', '_'))}`;
      link.target = '_blank';
      link.rel = 'noopener noreferrer';
      link.textContent = `${pageTitle} ↗`;
      title.replaceChildren(link);
    } else title.textContent = `Passage ${result.id}`;
    const text = document.createElement('p');
    if (result.metadataUnavailable) {
      text.textContent = 'Passage text is unavailable. Try searching again to retry the metadata download.';
    } else if (typeof result.text === 'string') {
      text.textContent = result.text;
    } else {
      text.textContent = 'None';
    }
    const distance = document.createElement('p');
    distance.className = 'passage-distance';
    distance.textContent = `Squared L2 distance: ${result.distance.toFixed(4)} · Passage ${result.id}`;
    article.append(title, text, distance);
    results.append(article);
  }
}

function getWorker() {
  if (worker) return worker;
  worker = new Worker(new URL('./embedding-worker.mjs', import.meta.url), { type: 'module' });
  worker.onmessage = ({ data }) => {
    if (data.requestId !== requestId) return;
    if (data.type === 'status') {
      progress.hidden = true;
      setStatus(data.message, 'loading');
    } else if (data.type === 'model-progress') {
      progress.hidden = false;
      progress.value = Math.max(0, Math.min(100, data.progress || 0));
      progress.setAttribute('aria-label', 'Embedding model download progress');
      setStatus(`Loading the embedding model… ${Math.round(progress.value)}%`, 'loading');
    } else if (data.type === 'search-progress') {
      progress.hidden = false;
      progress.value = data.total ? data.completed / data.total * 100 : 0;
      progress.setAttribute('aria-label', 'Cluster search progress');
      setStatus(`Searched ${data.completed} of ${data.total} clusters · ${data.candidates.toLocaleString()} passages checked`, 'loading');
      resultsInfo.textContent = `${data.candidates.toLocaleString()} passages checked so far`;
    } else if (data.type === 'ready') {
      ready = true;
      const previousCodec = codec.value;
      codec.replaceChildren(...data.codecs.map(value => new Option(CODEC_LABELS[value] || value, value)));
      codec.value = data.codecs.includes(previousCodec) ? previousCodec : data.codecs[0];
      probes.max = Math.min(100, data.clusters);
      probes.value = Math.min(Number(probes.value), Number(probes.max));
      resultK.max = data.count;
      resultK.value = Math.min(Number(resultK.value), data.count);
      indexInfo.textContent = `${data.count.toLocaleString()} passages · ${data.clusters.toLocaleString()} clusters`;
      resultsInfo.textContent = 'Enter a query to search the loaded passages.';
      progress.hidden = true;
      setBusy(false);
      setStatus('Ready to search. Your query stays in this browser.', 'success');
    } else if (data.type === 'error') {
      showError(data.message, data.stage);
    } else if (data.type === 'result') {
      renderResults(data.results);
      resultsInfo.textContent = `${data.results.length} results · ${data.candidates.toLocaleString()} passages checked · ${data.clusters} clusters · ${(data.bytes / 1e6).toFixed(2)} MB of vectors · embedding ${data.inferenceMs.toFixed(1)} ms${data.centroidMs ? ` · centroids ${data.centroidMs.toFixed(0)} ms` : ''} · vector retrieval ${data.vectorMs.toFixed(0)} ms · result text ${data.metadataMs.toFixed(0)} ms`;
      progress.hidden = true;
      setBusy(false);
      setStatus(data.metadataUnavailable
        ? 'Search complete. Some passage text could not be downloaded; the matching passage IDs are shown.'
        : 'Search complete.', data.metadataUnavailable ? '' : 'success');
    }
  };
  worker.onerror = worker.onmessageerror = () => {
    worker.terminate();
    worker = undefined;
    showError('Search stopped unexpectedly. Choose another centroid count or reload the page to try again.', 'index');
  };
  return worker;
}

function loadIndex() {
  ready = false;
  results.innerHTML = emptyResults;
  resultsInfo.textContent = 'Loading passage index…';
  setBusy(true);
  setStatus('Loading passage index…', 'loading');
  try {
    const indexUrl = new URL(indexPaths[clusterCount.value], location.href).href;
    getWorker().postMessage({ type: 'init', requestId: ++requestId, indexUrl, codec: codec.value });
  } catch (error) { showError(error.message, 'index'); }
}

function cancel() {
  if (!busy) return;
  worker?.postMessage({ type: 'cancel', requestId });
  requestId++;
  progress.hidden = true;
  setBusy(false);
  if (!ready) indexInfo.textContent = 'Choose a centroid count to load its index.';
  resultsInfo.textContent = ready ? 'Search canceled. Enter a query to try again.' : 'Choose a centroid count to load its index.';
  setStatus(ready ? 'Search canceled. Ready for another query.' : 'Index loading canceled.');
}

query.addEventListener('input', () => setBusy(busy));
for (const example of document.querySelectorAll('[data-query]')) {
  example.addEventListener('click', () => {
    query.value = example.dataset.query;
    setBusy(busy);
    query.focus();
  });
}
query.addEventListener('keydown', event => {
  if (event.key === 'Enter' && (event.ctrlKey || event.metaKey)) {
    event.preventDefault();
    form.requestSubmit();
  }
});
form.addEventListener('submit', event => {
  event.preventDefault();
  if (busy || !ready || !query.value.trim()) return;
  setBusy(true);
  results.innerHTML = emptyResults;
  resultsInfo.textContent = 'Searching the loaded passage index…';
  setStatus('Preparing your query…', 'loading');
  try {
    getWorker().postMessage({ type: 'search', requestId: ++requestId,
      text: query.value, codec: codec.value, probes: Number(probes.value), k: Number(resultK.value) });
  } catch (error) { showError(error.message); }
});
clusterCount.addEventListener('change', loadIndex);
cancelButton.addEventListener('click', cancel);
clearButton.addEventListener('click', () => {
  if (busy && !ready) {
    query.value = '';
    setBusy(true);
    query.focus();
    return;
  }
  cancel();
  query.value = '';
  results.innerHTML = emptyResults;
  resultsInfo.textContent = ready ? 'Enter a query to search the loaded passages.' : 'Choose a centroid count to load its index.';
  progress.hidden = true;
  setStatus(ready ? 'Ready for another question.' : 'Choose a centroid count to load its index.');
  setBusy(false);
  query.focus();
});

loadIndex();
