// Convert the complete Wikipedia metadata bundle to 500-passage Brotli files.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const z = require('node:zlib');
const { pathToFileURL } = require('node:url');
const { promisify } = require('node:util');
const compressBrotli = promisify(z.brotliCompress);

async function encodeMetadataBatch(source, first, count, cached) {
  assert.ok(Number.isInteger(first) && first >= 0 && first % 500 === 0);
  assert.ok(Number.isInteger(count) && count > 0 && count <= 500);
  const { decodeMetadataBatch } = await import(pathToFileURL(path.join(__dirname, '../client_wiki/assets/search-index.mjs')));
  const gzip = fs.readFileSync(path.join(source, 'metadata', `meta_${first / 500}.bin.gz`));
  const raw = z.gunzipSync(gzip);
  decodeMetadataBatch(Uint8Array.from(raw).buffer, first, count);
  const encoded = cached ?? await compressBrotli(raw, { params: {
    [z.constants.BROTLI_PARAM_QUALITY]: 11,
    [z.constants.BROTLI_PARAM_MODE]: z.constants.BROTLI_MODE_TEXT,
    [z.constants.BROTLI_PARAM_SIZE_HINT]: raw.length,
  } });
  assert.ok(z.brotliDecompressSync(encoded).equals(raw), `Brotli batch starting at ${first} differs from the source`);
  return { encoded, originalGzipBytes: gzip.length };
}

async function prepareBrotli(source, output) {
  assert.ok(!fs.existsSync(output), `Refusing to overwrite ${output}`);
  const manifests = ['manifest.json', 'c8192/manifest.json', 'c32768/manifest.json'];
  const indexes = manifests.map(name => JSON.parse(fs.readFileSync(path.join(source, name))));
  const reference = indexes[0];
  assert.equal(reference.metadata.format, 'binary-gzip-v1');
  assert.equal(reference.metadata.batch_size, 500);
  assert.equal(reference.metadata.count, reference.count);
  assert.equal(reference.count, 35167920);
  assert.equal(reference.dimensions, 384);
  for (const index of indexes) {
    for (const key of ['version', 'count', 'dimensions', 'source', 'source_shards', 'embedding_model']) {
      assert.deepEqual(index[key], reference[key], `Different source ${key}`);
    }
    for (const key of ['format', 'batch_size', 'count']) assert.equal(index.metadata[key], reference.metadata[key]);
    assert.ok(index.cluster_files.every(file => file.file.startsWith('https://')));
    assert.ok(Object.values(index.centroids).every(file => file.file.startsWith('https://')));
  }
  const count = reference.count;
  const total = Math.ceil(count / 500);
  const staged = output + '.partial';
  const marker = { source: crypto.createHash('sha256').update(JSON.stringify(indexes)).digest('hex'),
    count, batch_size: 500, quality: 11 };
  const checkpoint = path.join(staged, 'build.json');
  if (fs.existsSync(staged)) {
    assert.deepEqual(JSON.parse(fs.readFileSync(checkpoint)), marker, 'Partial build belongs to different source data');
  } else {
    fs.mkdirSync(path.join(staged, 'metadata'), { recursive: true });
    fs.writeFileSync(checkpoint, JSON.stringify(marker) + '\n');
  }
  let next = 0, completed = 0, bytes = 0, originalBytes = 0, reused = 0, failed;
  const started = performance.now();
  const progress = () => console.log(JSON.stringify({ completed, total, reused, bytes,
    elapsed_seconds: Math.round((performance.now() - started) / 1000) }));
  const timer = setInterval(progress, 30000);
  try {
    await Promise.allSettled(Array.from({ length: 12 }, async () => {
      try {
        while (next < total && !failed) {
          const batch = next++;
          const target = path.join(staged, 'metadata', `meta_${batch}.bin.br`);
          const cached = fs.existsSync(target) ? fs.readFileSync(target) : undefined;
          const result = await encodeMetadataBatch(source, batch * 500, Math.min(500, count - batch * 500), cached);
          if (cached) reused++;
          else {
            fs.writeFileSync(target + '.tmp', result.encoded);
            fs.renameSync(target + '.tmp', target);
          }
          bytes += result.encoded.length;
          originalBytes += result.originalGzipBytes;
          completed++;
        }
      } catch (error) { failed ??= error; }
    }));
    if (failed) throw failed;
    assert.equal(completed, total);
    assert.equal(fs.readdirSync(path.join(staged, 'metadata')).filter(name => name.endsWith('.bin.br')).length, total);
    for (const [position, index] of indexes.entries()) {
      index.metadata = { format: 'binary-http-v1', count, batch_size: 500,
        pattern: `${position ? '../' : ''}metadata/meta_{id}.bin.br` };
      const target = path.join(staged, manifests[position]);
      fs.mkdirSync(path.dirname(target), { recursive: true });
      fs.writeFileSync(target, JSON.stringify(index) + '\n');
    }
    const report = { count, batch_size: 500, quality: 11, files: total, reused,
      brotli_bytes: bytes, original_gzip_bytes: originalBytes,
      saving_percent: 100 * (1 - bytes / originalBytes),
      elapsed_seconds: (performance.now() - started) / 1000, created: new Date().toISOString() };
    fs.writeFileSync(path.join(staged, 'compression.json'), JSON.stringify(report, null, 2) + '\n');
    assert.ok(!fs.existsSync(output), `Refusing to overwrite ${output}`);
    fs.renameSync(staged, output);
    fs.unlinkSync(path.join(output, 'build.json'));
    console.log(JSON.stringify({ output, ...report }, null, 2));
    return report;
  } finally { clearInterval(timer); }
}

async function main() {
  assert.equal(process.argv.length, 3, 'Usage: node code/src/prepare_brotli.cjs NEW_OUTPUT_DIRECTORY');
  const code = path.resolve(__dirname, '..');
  await prepareBrotli(path.join(code, 'data/demo/gzip-staging'), path.resolve(process.argv[2]));
}
main().catch(error => { console.error(error); process.exitCode = 1; });
