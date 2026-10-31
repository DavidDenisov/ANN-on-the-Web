const assert = require('node:assert/strict');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const crypto = require('node:crypto');
const z = require('node:zlib');
const { spawnSync } = require('node:child_process');
const test = require('node:test');
const vm = require('node:vm');
const { metadataBatch } = require('./search-fixture.cjs');

if (!vm.SourceTextModule) {
  test('Brotli preparation preserves metadata and existing output', () => {
    const run = spawnSync(process.execPath, ['--experimental-vm-modules', __filename], { encoding: 'utf8', timeout: 10000 });
    assert.equal(run.status, 0, run.stdout + run.stderr);
  });
} else {
  for (const scenario of ['fresh', 'resume', 'wrong-checkpoint', 'corrupt-cache']) {
    test(`Brotli preparation ${scenario} preserves publication and checkpoint guarantees`, async t => {
      const root = fs.mkdtempSync(path.join(os.tmpdir(), 'brotli-check-'));
      t.after(() => fs.rmSync(root, { recursive: true, force: true }));
      const source = path.join(root, 'source'), output = path.join(root, 'output');
      const staged = output + '.partial';
      const manifests = ['manifest.json', 'c8192/manifest.json', 'c32768/manifest.json'];
      const indexes = manifests.map(() => ({ version: 1, count: 35167920, dimensions: 384,
        source: { revision: 'fixture' }, source_shards: [], embedding_model: 'Xenova/all-MiniLM-L6-v2',
        metadata: { format: 'binary-gzip-v1', batch_size: 500, count: 35167920 },
        cluster_files: [{ file: 'https://example.org/cluster.gz' }], centroids: { f16: { file: 'https://example.org/centroid.gz' } } }));
      for (const [i, name] of manifests.entries()) {
        fs.mkdirSync(path.dirname(path.join(source, name)), { recursive: true });
        fs.writeFileSync(path.join(source, name), JSON.stringify(indexes[i]));
      }
      const raw = metadataBatch(), gzip = z.gzipSync(raw);
      fs.mkdirSync(path.join(source, 'metadata'));
      fs.writeFileSync(path.join(source, 'metadata/meta_0.bin.gz'), gzip);
      const marker = { source: crypto.createHash('sha256').update(JSON.stringify(indexes)).digest('hex'),
        count: 35167920, batch_size: 500, quality: 11 };
      const cached = scenario === 'corrupt-cache' ? Buffer.from('corrupt') : z.brotliCompressSync(raw);
      if (scenario !== 'fresh') {
        fs.mkdirSync(path.join(staged, 'metadata'), { recursive: true });
        fs.writeFileSync(path.join(staged, 'build.json'), JSON.stringify(scenario === 'wrong-checkpoint'
          ? { ...marker, source: 'another snapshot' } : marker));
        fs.writeFileSync(path.join(staged, 'metadata/meta_0.bin.br'), cached);
      }
      const filename = path.join(__dirname, '../src/prepare_brotli.cjs');
      const code = fs.readFileSync(filename, 'utf8');
      const boundary = code.lastIndexOf('\nmain().catch');
      assert.ok(boundary > 0);
      const context = vm.createContext({ require, __dirname: path.dirname(filename), performance, Buffer,
        setInterval, clearInterval, console: { log() {} },
        // Keep the production corpus guards while bounding the fixture to one real batch.
        Math: Object.assign(Object.create(Math), { ceil: () => 1 }),
      });
      new vm.Script(code.slice(0, boundary), { filename, importModuleDynamically: specifier => import(specifier) }).runInContext(context);
      if (scenario === 'wrong-checkpoint' || scenario === 'corrupt-cache') {
        await assert.rejects(context.prepareBrotli(source, output));
        assert.equal(fs.existsSync(output), false);
        assert.deepEqual(fs.readFileSync(path.join(staged, 'metadata/meta_0.bin.br')), cached);
        assert.equal(fs.existsSync(path.join(staged, 'build.json')), true);
      } else {
        const result = await context.prepareBrotli(source, output);
        assert.equal(result.files, 1);
        assert.equal(result.reused, scenario === 'resume' ? 1 : 0);
        assert.equal(fs.existsSync(staged), false);
        assert.equal(fs.existsSync(path.join(output, 'build.json')), false);
        assert.deepEqual(z.brotliDecompressSync(fs.readFileSync(path.join(output, 'metadata/meta_0.bin.br'))), raw);
        assert.equal(fs.existsSync(path.join(output, 'DATA-NOTICE.txt')), true);
      }
      assert.deepEqual(fs.readFileSync(path.join(source, 'metadata/meta_0.bin.gz')), gzip);
    });
  }

  test('Brotli preparation refuses existing output before writing', async () => {
    const filename = path.join(__dirname, '../src/prepare_brotli.cjs');
    const errors = [];
    let written = 0;
    const files = {
      existsSync: () => true,
      writeFileSync: () => { written++; throw new Error('unexpected write'); },
    };
    const context = vm.createContext({ __filename: filename, __dirname: path.dirname(filename),
      process: { argv: ['node', filename, '/retained-output'], exitCode: 0 },
      console: { log() {}, error: error => errors.push(error) },
      require: name => name === 'node:fs' ? files : require(name),
    });
    const script = new vm.Script(fs.readFileSync(filename, 'utf8'), {
      filename, importModuleDynamically: specifier => import(specifier),
    });
    try { await script.runInContext(context); } catch (error) { errors.push(error); }
    assert.equal(errors.length, 1);
    assert.match(errors[0].message, /Refusing to overwrite/);
    assert.equal(written, 0);
  });
}
