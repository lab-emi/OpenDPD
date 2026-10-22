/**
 * Real local browser verification of private multi-preset PA datasets (Signal Generator -> Virtual PA -> dataset page).
 *
 * The 5G NR family opens on the standard test models (3GPP TS 38.141-1, clause 4.9.2), not on the numerology matrix, so the run mixes the three
 * ways a signal gets into the selection:
 *   1. the default test model (TM1.1, FDD, 30 kHz, 20 MHz), shortened to 2 subframes with the "Custom" capture length (a test model has no
 *      "I/Q samples" field: the test-model panel fixes the length);
 *   2. a generic NR preset behind the "Generic numerology matrix…" button, with its own I/Q sample count;
 *   3. a Wi-Fi 7 preset with its own I/Q sample count.
 * It then generates the three signals in one batch, hands them to one Virtual PA, simulates the PA output and checks the heterogeneous captures,
 * downloads and the restored selection at desktop and mobile widths. Screenshots, the downloads and `signal-generator.json` go to OUTPUT_DIR.
 *
 *   OPENDPD_BOOTSTRAP_TOKEN=token1,token2,token3 node scripts/verify_signal_generator.mjs URL OUTPUT_DIR
 *
 * A bootstrap token opens exactly one browser session and every viewport is a fresh browser context, so pass one token per viewport (three),
 * separated by commas. Set OPENDPD_CDP_URL to use another Chromium than the shared one at http://127.0.0.1:9222.
 */
import { chromium } from '../frontend/node_modules/playwright/index.mjs'
import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import path from 'node:path'

const VIEWPORTS = [[1366, 768], [1920, 1080], [390, 844]]
const TM_SUBFRAMES = 2
const TM_SAMPLES_PER_SUBFRAME = 61440 // TM1.1, 30 kHz, 20 MHz: 61.44 MS/s (oversampling 2x) x 1 ms
const NR_SAMPLES = 16384
const WIFI_SAMPLES = 24576

const [baseURL, out] = process.argv.slice(2)
const tokens = (process.env.OPENDPD_BOOTSTRAP_TOKEN ?? '').split(',').map(token => token.trim()).filter(Boolean)
if (!baseURL || !out || !tokens.length) throw new Error('Usage: OPENDPD_BOOTSTRAP_TOKEN=token1,token2,token3 node scripts/verify_signal_generator.mjs URL OUTPUT_DIR')
if (tokens.length < VIEWPORTS.length) throw new Error(`A bootstrap token opens one session and every viewport is a new browser context: pass ${VIEWPORTS.length} comma-separated tokens, got ${tokens.length}`)
await fs.mkdir(out, { recursive: true })
const browser = await chromium.connectOverCDP(process.env.OPENDPD_CDP_URL ?? 'http://127.0.0.1:9222')
const evidence = []
try {
  for (const [index, [width, height]] of VIEWPORTS.entries()) {
    const context = await browser.newContext({ viewport: { width, height }, locale: 'en-US', acceptDownloads: true })
    try {
      const page = await context.newPage()
      const errors = []
      page.on('pageerror', e => errors.push(e.message))
      await page.goto(baseURL + '/bootstrap?token=' + encodeURIComponent(tokens[index]))
      await page.goto(baseURL + '/signal-generator')
      await page.getByRole('heading', { name: 'Signal setup', exact: true }).waitFor()
      assert.equal(await page.getByRole('button', { name: /Wi-Fi 8/ }).count(), 0)
      assert.equal(await page.getByTestId('workflow-paired').count(), 0)
      assert.equal(await page.getByRole('combobox', { name: 'Preset', exact: true }).count(), 0)

      // 1. Default NR view: the test-model panel with TM1.1 selected, no matrix, no sample-count field.
      await page.getByTestId('nrtm-panel').waitFor()
      await page.getByTestId('nrtm-derived').waitFor()
      assert.equal(await page.getByRole('button', { name: 'TM1.1, QPSK', exact: true }).getAttribute('aria-pressed'), 'true')
      assert.equal(await page.getByTestId('preset-nr-20').count(), 0, 'the numerology matrix must be hidden by default')
      assert.equal(await page.getByRole('button', { name: 'Generic numerology matrix…' }).getAttribute('aria-pressed'), 'false')
      assert.equal(await page.getByLabel('I/Q samples', { exact: true }).count(), 0, 'a test model has no free I/Q sample count')
      assert.equal(await page.locator('[data-testid^="selected-preset-"]').count(), 1)
      await page.getByRole('button', { name: 'Custom', exact: true }).click()
      await page.getByLabel('Subframes (1 ms each)', { exact: true }).fill(String(TM_SUBFRAMES))
      const tmSamples = TM_SAMPLES_PER_SUBFRAME * TM_SUBFRAMES
      await page.getByTestId('nrtm-derived').getByText(`${tmSamples.toLocaleString('en-US')} I/Q samples`, { exact: false }).waitFor()
      await page.waitForFunction(() => !document.querySelector('[data-testid=nrtm-status] [role=progressbar]'))
      assert.equal(await page.getByRole('button', { name: 'Generate & preview', exact: true }).isEnabled(), true)

      // 2. Matrix path: the button opens the preset matrix, a generic NR preset joins the selection with its own sample count.
      await page.getByRole('button', { name: 'Generic numerology matrix…' }).click()
      assert.equal(await page.getByRole('button', { name: 'Generic numerology matrix…' }).getAttribute('aria-pressed'), 'true')
      await page.getByTestId('preset-nr-20').click()
      await page.getByLabel('I/Q samples', { exact: true }).fill(String(NR_SAMPLES))
      assert.equal(await page.locator('[data-testid^="selected-preset-"]').count(), 2)

      // 3. A Wi-Fi 7 preset.
      await page.getByRole('button', { name: /03 ·.*Wi-Fi 7/ }).click()
      await page.getByTestId('preset-wifi7-80-q1024-c2').click()
      await page.getByLabel('I/Q samples', { exact: true }).fill(String(WIFI_SAMPLES))
      assert.equal(await page.locator('[data-testid^="selected-preset-"]').count(), 3)

      const generating = page.waitForResponse(r => r.url().endsWith('/signal-generator/batches') && r.request().method() === 'POST')
      await page.getByRole('button', { name: 'Generate & preview', exact: true }).click()
      const generated = await generating
      assert.equal(generated.status(), 201, await generated.text())
      const inputs = await generated.json()
      assert.deepEqual(inputs.map(s => s.n_samples), [tmSamples, NR_SAMPLES, WIFI_SAMPLES])
      await page.getByTestId('signal-generator-results').waitFor()
      await page.waitForFunction(() => document.querySelectorAll('.js-plotly-plot .plot-container').length >= 4)
      await page.evaluate(() => window.scrollTo(0, 0))
      await page.screenshot({ path: path.join(out, `generator-${width}.png`), fullPage: true })
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
      const inputDownload = page.waitForEvent('download')
      await page.getByRole('button', { name: 'Download PA input CSV', exact: true }).click()
      const downloadedInput = await inputDownload
      await downloadedInput.saveAs(path.join(out, `pa-input-${width}.csv`))
      await page.getByRole('link', { name: 'Choose Virtual PA', exact: true }).click()
      await page.getByText('All 3 selected presets will use this PA.', { exact: false }).waitFor()
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
      assert.equal(dataset.captures.length, 3)
      await page.getByRole('combobox', { name: 'Visualize subdataset', exact: true }).click()
      await page.getByRole('option', { name: /80 MHz.*1024-QAM/ }).click()
      await page.getByText('320 MS/s', { exact: true }).waitFor()
      await page.getByRole('link', { name: 'Train PA & DPD Models', exact: true }).waitFor({ state: 'visible' })
      assert((await page.getByRole('link', { name: 'Train PA & DPD Models', exact: true }).getAttribute('href')).includes(dataset.captures[2].dataset_id))
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
      // The stored selection is restored when returning to Signal Generator: all three signals, the test model from its recipe, the matrix open
      // because a matrix preset is selected. The results live on the Preview page of the same tab.
      await page.goto(baseURL + '/signal-generator')
      await page.getByText('3 / 16 presets selected', { exact: true }).waitFor()
      assert.equal(await page.locator('[data-testid^="selected-preset-"]').count(), 3)
      await page.getByTestId('nrtm-panel').waitFor()
      assert.equal(await page.getByLabel('Subframes (1 ms each)', { exact: true }).inputValue(), String(TM_SUBFRAMES))
      assert.equal(await page.getByRole('button', { name: 'Generic numerology matrix…' }).getAttribute('aria-pressed'), 'true')
      await page.getByRole('link', { name: 'Open Preview', exact: true }).click()
      await page.getByTestId('signal-generator-results').waitFor()
      assert.equal(await page.getByRole('link', { name: 'Choose Virtual PA', exact: true }).getAttribute('aria-disabled'), null)
      assert.deepEqual(errors, [])
      evidence.push({ width, height, datasetId: dataset.dataset_id, captures: dataset.captures, inputSamples: inputs.map(s => s.n_samples), automaticNavigation: true, downloadZip: true, downloadCsv: true, errors })
      await fs.writeFile(path.join(out, 'signal-generator.json'), JSON.stringify(evidence, null, 2) + '\n')
    } finally { await context.close() }
  }
} finally { await browser.close() }
console.log(JSON.stringify(evidence, null, 2))
