export default [{
  files: ['**/*.mjs', '**/*.cjs'],
  ignores: ['data/**', '.desloppify/**'],
  languageOptions: {
    globals: Object.fromEntries([
      'AbortController', 'AbortSignal', 'Blob', 'Buffer', 'DecompressionStream', 'Option', 'Response', 'TextDecoder',
      'URL', 'Worker', '__dirname', '__filename', 'clearInterval', 'clearTimeout', 'console',
      'crossOriginIsolated', 'document', 'fetch', 'location', 'navigator', 'performance', 'process',
      'self', 'setImmediate', 'setInterval', 'setTimeout', 'crypto', 'window',
    ].map(name => [name, 'readonly'])),
  },
  rules: {
    'no-undef': 'error',
    'no-unused-vars': ['error', { args: 'none', caughtErrors: 'none', ignoreRestSiblings: true }],
  },
}];
