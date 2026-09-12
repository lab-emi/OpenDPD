import { defineConfig, devices } from '@playwright/test'

export default defineConfig({
  testDir: './touch-e2e',
  timeout: 60_000,
  workers: 1,
  expect: { timeout: 10_000 },
  use: { baseURL: 'http://127.0.0.1:4173', locale: 'zh-CN', trace: 'retain-on-failure' },
  webServer: {
    command: 'npx vite preview --port 4173 --strictPort --host 127.0.0.1',
    url: 'http://127.0.0.1:4173',
    reuseExistingServer: !process.env.CI,
  },
  projects: [
    { name: 'iphone-webkit', use: { ...devices['iPhone 13'], browserName: 'webkit' } },
    { name: 'iphone-chromium', use: { ...devices['iPhone 13'], browserName: 'chromium' } },
  ],
})
