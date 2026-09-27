/** Real CSV → Virtual PA → dataset journey on desktop and mobile. */
import { chromium } from '../frontend/node_modules/playwright/index.mjs'
import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import path from 'node:path'

const [base, out] = process.argv.slice(2)
if (!base || !out || !process.env.OPENDPD_BOOTSTRAP_TOKEN) throw new Error('Usage: OPENDPD_BOOTSTRAP_TOKEN=… node scripts/verify_signal_import.mjs URL OUTPUT_DIR')
await fs.mkdir(out, { recursive: true })
const browser = await chromium.connectOverCDP(process.env.OPENDPD_CDP_URL ?? 'http://127.0.0.1:9222')
const evidence = []
try {
  for (const [width, height] of [[1366, 768], [390, 844]]) {
    const context = await browser.newContext({ viewport: { width, height }, locale: 'en-US', isMobile: width < 600, hasTouch: width < 600 })
    try {
      const page = await context.newPage(), errors = []
      page.on('pageerror', error => errors.push(error.message))
      await page.goto(base + '/bootstrap?token=' + encodeURIComponent(process.env.OPENDPD_BOOTSTRAP_TOKEN))
      await page.goto(base + '/signal-generator')
      await page.getByRole('heading', { name: 'Signal setup', exact: true }).waitFor()
      await page.getByRole('button', { name: /Import custom signal/ }).click()
      const n = 32768
      const csv = 'I,Q\n' + Array.from({ length: n }, (_, i) => {
        const envelope = .12 + .08 * Math.sin(i / 1000)
        return `${envelope * Math.cos(i / 10)},${envelope * Math.sin(i / 10)}`
      }).join('\n') + '\n'
      await page.getByTestId('signal-import-upload').setInputFiles({ name: `custom-${width}.csv`, mimeType: 'text/csv', buffer: Buffer.from(csv) })
      const panel = page.getByTestId('signal-import')
      await panel.getByLabel('Sample rate (MHz)', { exact: true }).fill('20')
      await panel.getByLabel('Baseband bandwidth (MHz)', { exact: true }).fill('5')
      await panel.getByRole('button', { name: 'Import & use in Virtual PA', exact: true }).waitFor()
      await page.screenshot({ path: path.join(out, `signal-import-${width}.png`), fullPage: true })
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
      const importButton = panel.getByRole('button', { name: 'Import & use in Virtual PA', exact: true })
      await importButton.focus()
      const [response] = await Promise.all([
        page.waitForResponse(r => r.url().endsWith('/signal-generator/import') && r.request().method() === 'POST'),
        width < 600 ? importButton.tap() : importButton.click(),
      ])
      assert.equal(response.status(), 201, await response.text())
      const input = await response.json()
      assert.equal(input.n_samples, n)
      assert.equal(input.origin, 'uploaded')
      await page.getByRole('combobox', { name: 'PA Input Dataset (x)', exact: true }).waitFor()
      await page.reload()
      assert.equal(await page.getByRole('combobox', { name: 'PA Input Dataset (x)', exact: true }).inputValue(), input.dataset_name)
      const pairing = page.waitForResponse(r => r.url().endsWith('/pa-library/datasets') && r.request().method() === 'POST')
      await page.getByRole('button', { name: 'Simulate PA output', exact: true }).click()
      const paired = await pairing
      assert.equal(paired.status(), 201, await paired.text())
      const dataset = (await paired.json()).dataset
      assert.equal(dataset.origin, 'synthetic')
      assert.equal(dataset.simulation.input_origin, 'uploaded')
      assert.equal(dataset.n_samples, n)
      await page.waitForURL('**/datasets/' + dataset.dataset_id)
      await page.getByRole('link', { name: 'Train PA & DPD Models', exact: true }).waitFor()
      await page.screenshot({ path: path.join(out, `custom-pa-dataset-${width}.png`), fullPage: true })
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
      // Imported metadata must never be restored as an editable synthesis recipe.
      await page.goto(base + '/signal-generator')
      await page.getByRole('heading', { name: 'Signal setup', exact: true }).waitFor()
      assert.equal(await page.getByTestId('signal-generator-results').count(), 0)
      assert.deepEqual(errors, [])
      evidence.push({ width, samples: n, input: input.signal_id, dataset: dataset.dataset_id, errors })
    } finally { await context.close() }
  }
  await fs.writeFile(path.join(out, 'signal-import.json'), JSON.stringify(evidence, null, 2) + '\n')
  console.log(JSON.stringify(evidence, null, 2))
} finally { await browser.close() }
