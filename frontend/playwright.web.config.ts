import { defineConfig, devices } from '@playwright/test'

export default defineConfig({
  testDir: './web-e2e',
  timeout: 60_000,
  workers: 1,
  use: { baseURL: 'http://127.0.0.1:4174', trace: 'retain-on-failure' },
  webServer: {
    command: 'npx vite preview --outDir /tmp/opendpd-web-e2e --port 4174 --strictPort --host 127.0.0.1',
    url: 'http://127.0.0.1:4174',
  },
  projects: [{ name: 'web-chromium', use: { ...devices['Desktop Chrome'] } }],
})
