import { test as base } from '@playwright/test'

// Local agents can reuse the configured isolated Chromium service. CI keeps
// its ordinary browser fixture. Each test still receives a fresh context.
const endpoint = process.env['OPENDPD_TEST_CDP']
export const test = endpoint ? base.extend({
  browser: [async ({ playwright }, provide) => {
    const browser = await playwright.chromium.connectOverCDP(endpoint)
    try { await provide(browser) } finally {
      // For connectOverCDP, close disconnects this client transport; it does
      // not terminate the shared browser process.
      await browser.close()
    }
  }, { scope: 'worker' }],
}) : base
