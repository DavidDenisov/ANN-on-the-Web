const assert = require('node:assert/strict');
const http = require('node:http');
const { chromium } = require('playwright');

module.exports = async function openBrowser(files = new Map()) {
  const server = http.createServer((request, response) => {
    const pathname = new URL(request.url, 'http://localhost').pathname;
    const body = files.get(pathname);
    response.writeHead(body || pathname === '/' ? 200 : 404, {
      'Content-Type': pathname.endsWith('.mjs') ? 'text/javascript'
        : pathname.endsWith('.json') ? 'application/json' : 'text/html',
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cache-Control': 'no-store',
    });
    response.end(body || '<!doctype html><title>Paper benchmark</title>');
  });
  let browser;
  const close = async () => {
    try { if (browser) await browser.close(); }
    finally { await new Promise(resolve => server.close(resolve)); }
  };
  try {
    await new Promise((resolve, reject) => {
      server.once('error', reject);
      server.listen(0, '127.0.0.1', resolve);
    });
    browser = await chromium.launch({ executablePath: '/snap/bin/chromium', headless: false, args: ['--no-sandbox'] });
    assert.match(await browser.version(), /^155\./);
    return { browser, origin: 'http://127.0.0.1:' + server.address().port, close };
  } catch (error) {
    await close();
    throw error;
  }
};
