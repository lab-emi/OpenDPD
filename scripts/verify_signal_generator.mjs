/** Real local browser verification of private multi-preset PA datasets. */
import { chromium } from '../frontend/node_modules/playwright/index.mjs'
import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import path from 'node:path'

const [baseURL, out] = process.argv.slice(2)
if (!baseURL || !out || !process.env.OPENDPD_BOOTSTRAP_TOKEN) throw new Error('Usage: OPENDPD_BOOTSTRAP_TOKEN=… node scripts/verify_signal_generator.mjs URL OUTPUT_DIR')
await fs.mkdir(out, { recursive: true })
const browser = await chromium.connectOverCDP(process.env.OPENDPD_CDP_URL ?? 'http://127.0.0.1:9222')
const evidence = []
try {
  for (const [width, height] of [[1366, 768], [1920, 1080], [390, 844]]) {
    const context = await browser.newContext({ viewport: { width, height }, locale: 'en-US', acceptDownloads: true })
    try {
      const page = await context.newPage()
      const errors = []
      page.on('pageerror', e => errors.push(e.message))
      await page.goto(baseURL + '/bootstrap?token=' + encodeURIComponent(process.env.OPENDPD_BOOTSTRAP_TOKEN))
      await page.goto(baseURL + '/signal-generator')
      await page.getByRole('heading', { name: 'Signal setup', exact: true }).waitFor()
      assert.equal(await page.getByRole('button', { name: /Wi-Fi 8/ }).count(), 0)
      assert.equal(await page.getByTestId('workflow-paired').count(), 0)
      assert.equal(await page.getByRole('combobox', { name: 'Preset', exact: true }).count(), 0)
      await page.getByLabel('I/Q samples', { exact: true }).fill('16384')
      await page.getByRole('button', { name: /03 ·.*Wi-Fi 7/ }).click()
      await page.getByTestId('preset-wifi7-80-q1024-c2').click()
      await page.getByLabel('I/Q samples', { exact: true }).fill('24576')
      const generating = page.waitForResponse(r => r.url().endsWith('/signal-generator/batches') && r.request().method() === 'POST')
      await page.getByRole('button', { name: 'Generate & preview', exact: true }).click()
      const generated = await generating
      assert.equal(generated.status(), 201, await generated.text())
      const inputs = await generated.json()
      assert.deepEqual(inputs.map(s => s.n_samples), [16384, 24576])
      await page.getByTestId('signal-generator-results').waitFor()
      await page.waitForFunction(() => document.querySelectorAll('.js-plotly-plot .plot-container').length >= 4)
      await page.evaluate(() => window.scrollTo(0, 0))
      await page.screenshot({ path: path.join(out, `generator-${width}.png`), fullPage: true })
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
      const inputDownload = page.waitForEvent('download')
      await page.getByRole('button', { name: 'Download PA input CSV', exact: true }).click()
      const downloadedInput = await inputDownload
      await downloadedInput.saveAs(path.join(out, `pa-input-${width}.csv`))
      await page.getByRole('link', { name: 'Choose Virtual PA →', exact: true }).click()
      await page.getByText('All 2 selected presets will use this PA.', { exact: false }).waitFor()
      const gain = page.getByRole('spinbutton', { name: 'Small-signal gain', exact: true })
      await gain.fill('2.5'); await gain.focus()
      assert.equal(await page.getByTestId('parameter-gain').getAttribute('data-active'), 'true')
      await page.evaluate(() => window.scrollTo(0, 0))
      await page.screenshot({ path: path.join(out, `virtual-pa-${width}.png`), fullPage: true })
      const saving = page.waitForResponse(r => r.url().endsWith('/pa-library/datasets') && r.request().method() === 'POST')
      await page.getByRole('button', { name: 'Simulate PA output', exact: true }).click()
      const response = await saving
      assert.equal(response.status(), 201, await response.text())
      const { dataset } = await response.json()
      await page.waitForURL('**/datasets/' + dataset.dataset_id)
      assert.equal(dataset.captures.length, 2)
      await page.getByRole('combobox', { name: 'Visualize subdataset', exact: true }).click()
      await page.getByRole('option', { name: /80 MHz.*1024-QAM/ }).click()
      await page.getByText('320 MS/s', { exact: true }).waitFor()
      await page.getByRole('link', { name: 'Configure experiment', exact: true }).waitFor({ state: 'visible' })
      assert((await page.getByRole('link', { name: 'Configure experiment', exact: true }).getAttribute('href')).includes(dataset.captures[1].dataset_id))
      const zipDownload = page.waitForEvent('download')
      await page.getByRole('button', { name: 'Download all · ZIP', exact: true }).click()
      await (await zipDownload).saveAs(path.join(out, `dataset-${width}.zip`))
      const csvDownload = page.waitForEvent('download')
      await page.getByRole('button', { name: 'Download CSV', exact: true }).click()
      await (await csvDownload).saveAs(path.join(out, `dataset-selected-${width}.csv`))
      await page.waitForFunction(() => document.querySelectorAll('.js-plotly-plot .plot-container').length >= 3)
      await page.evaluate(() => window.scrollTo(0, 0))
      await page.screenshot({ path: path.join(out, `dataset-${width}.png`), fullPage: true })
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
      // Stored selection is preserved when returning to Signal Generator.
      await page.goto(baseURL + '/signal-generator')
      await page.getByTestId('signal-generator-results').waitFor()
      assert.equal(await page.getByText('2 / 16 presets selected', { exact: true }).count(), 1)
      assert.equal(await page.getByRole('link', { name: 'Choose Virtual PA →', exact: true }).getAttribute('aria-disabled'), null)
      assert.deepEqual(errors, [])
      evidence.push({ width, height, datasetId: dataset.dataset_id, captures: dataset.captures, automaticNavigation: true, downloadZip: true, downloadCsv: true, errors })
      await fs.writeFile(path.join(out, 'signal-generator.json'), JSON.stringify(evidence, null, 2) + '\n')
    } finally { await context.close() }
  }
} finally { await browser.close() }
console.log(JSON.stringify(evidence, null, 2))
