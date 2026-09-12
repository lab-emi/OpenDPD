/// <reference types="vitest/config" />
import { readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath, URL } from 'node:url'
import react from '@vitejs/plugin-react'
import { defineConfig, loadEnv, type Plugin } from 'vite'

/** Version of the Python package this build belongs to (checked by /readyz). */
function opendpdVersion(): string {
  const init = readFileSync(fileURLToPath(new URL('../opendpd/__init__.py', import.meta.url)), 'utf-8')
  const m = init.match(/__version__\s*=\s*"([^"]+)"/)
  if (!m) throw new Error('cannot read __version__ from opendpd/__init__.py')
  return m[1]!
}

function buildInfo(): Plugin {
  let outDir = ''
  return {
    name: 'opendpd-build-info',
    configResolved(config) {
      outDir = config.build.outDir
    },
    closeBundle() {
      const info = { opendpd_version: opendpdVersion(), built_at: new Date().toISOString() }
      writeFileSync(`${outDir}/build-info.json`, JSON.stringify(info, null, 2) + '\n')
    },
  }
}

// Production build lands in opendpd/studio/static and is served by FastAPI
// (S06); in development the API is proxied to a running `opendpd gui`.
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), 'VITE_')
  const web = env.VITE_STUDIO_MODE === 'web'
  const apiOrigin = env.VITE_API_ORIGIN ?? ''
  if (web && (!apiOrigin || new URL(apiOrigin).origin !== apiOrigin || !apiOrigin.startsWith('https://'))) {
    throw new Error('Web builds require VITE_API_ORIGIN as an exact HTTPS origin')
  }
  const csp: Plugin = {
    name: 'public-studio-csp',
    transformIndexHtml: {
      order: 'pre',
      handler: () => web ? [{ tag: 'meta', injectTo: 'head-prepend', attrs: { 'http-equiv': 'Content-Security-Policy', content:
        `default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self' data:; connect-src 'self' ${apiOrigin}; object-src 'none'; base-uri 'self'; form-action 'none'` } },
      { tag: 'meta', injectTo: 'head', attrs: { name: 'referrer', content: 'no-referrer' } }] : [],
    },
  }
  return {
  base: web ? (env.VITE_BASE_PATH || './') : '/',
  plugins: [react(), buildInfo(), csp],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
      '@mocks': fileURLToPath(new URL('./mocks', import.meta.url)),
    },
  },
  build: {
    outDir: web ? 'dist' : '../opendpd/studio/static',
    emptyOutDir: true,
    sourcemap: false,
    chunkSizeWarningLimit: 1500,
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: { '/api': 'http://127.0.0.1:8765', '/bootstrap': 'http://127.0.0.1:8765' },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['src/test/setup.ts'],
    include: ['src/**/*.test.{ts,tsx}'],
    css: false,
  },
  }
})
