/// <reference types="vitest/config" />
import { readFileSync, writeFileSync } from 'node:fs'
import { fileURLToPath, URL } from 'node:url'
import react from '@vitejs/plugin-react'
import { defineConfig, type Plugin } from 'vite'

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
export default defineConfig({
  plugins: [react(), buildInfo()],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
      '@mocks': fileURLToPath(new URL('./mocks', import.meta.url)),
    },
  },
  build: {
    outDir: '../opendpd/studio/static',
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
})
