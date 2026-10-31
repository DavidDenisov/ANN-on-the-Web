const { gzipSync } = require('node:zlib');

const dimensions = 384, count = 35167920, clusters = 8192;

function metadataBatch() {
  const header = Buffer.alloc(12);
  header.write('WMD1');
  header.writeUInt32LE(500, 8);
  const records = Array.from({ length: 500 }, () => {
    const sizes = Buffer.alloc(8);
    sizes.writeUInt32LE(5);
    sizes.writeUInt32LE(7, 4);
    return Buffer.concat([sizes, Buffer.from('TitlePassage')]);
  });
  return Buffer.concat([header, ...records]);
}

function fixture() {
  const centroids = gzipSync(Buffer.alloc(clusters * dimensions * 2));
  const record = Buffer.alloc(4 + dimensions * 2);
  record.writeUInt32LE(5);
  record.writeUInt16LE(0x3c00, 4);
  const cluster = gzipSync(record);
  const index = {
    version: 1, dimensions, count, clusters, record_bytes: 772, metric: 'normalized-l2',
    embedding_model: 'Xenova/all-MiniLM-L6-v2',
    cluster_sizes: Array.from({ length: clusters }, (_, i) => i === 0 ? 1 : i === clusters - 1 ? count - 1 : 0),
    cluster_files: Array.from({ length: clusters }, (_, i) => ({ file: `cluster-${i}.gz`, bytes: cluster.length })),
    centroids: Object.fromEntries(['f16', 'sq8', 'sq6', 'sq4', 'int8', 'pq4']
      .map(codec => [codec, { file: 'centroids.gz', bytes: centroids.length, rmse: 0 }])),
    metadata: { format: 'binary-http-v1', count, batch_size: 500, pattern: 'meta_{id}.bin.br',
      files_per_deployment: 1000, base_urls: Array.from({ length: 71 }, () => 'https://example.org/') },
  };
  const files = new Map([
    ['https://example.org/index.json', JSON.stringify(index)],
    ['https://example.org/centroids.gz', centroids],
    ['https://example.org/cluster-0.gz', cluster],
    ['https://example.org/meta_0.bin.br', metadataBatch()],
  ]);
  const requests = [];
  return { files, requests, fetch: async (url, options) => {
    requests.push({ url: String(url), cache: options.cache });
    return new Response(files.get(String(url)), { status: files.has(String(url)) ? 200 : 404 });
  } };
}

module.exports = { dimensions, count, clusters, metadataBatch, fixture };
