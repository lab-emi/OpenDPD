/** Actual GUI authoring, private reproduction, sourced-cost axes and precision preview. No public writes. */
import { chromium } from '../frontend/node_modules/playwright/index.mjs'
import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import path from 'node:path'
const [baseURL, workspace, out] = process.argv.slice(2)
const demo = JSON.parse(await fs.readFile(path.join(workspace, 'publication-demo.json'), 'utf8'))
await fs.mkdir(out, { recursive: true })
const browser = await chromium.connectOverCDP('http://127.0.0.1:9222')
const evidence = []
try {
  for (const [width, height] of [[1366, 768], [1920, 1080]]) {
    const context = await browser.newContext({ viewport: { width, height }, acceptDownloads: true })
    try {
      const page = await context.newPage()
      const errors = []
      page.on('pageerror', error => errors.push(error.message))
      await page.goto(`${baseURL}/bootstrap?token=studio-next-local-review`)
      const resultURL = `${baseURL}/api/v1/results/${demo.run_id}?profile=general-spectral-v1`
      const originalResult = await (await context.request.get(resultURL)).json()
      await page.goto(`${baseURL}/results/${demo.run_id}?profile=general-spectral-v1`)
      await page.getByText(/SYNTHETIC — software demonstration only/).waitFor()
      await page.getByRole('button', { name: 'Publication figure', exact: true }).click()
      let dialog = page.getByRole('dialog', { name: 'Publication figure', exact: true })
      const figureTitle = `Synthetic multi-panel ${width}-${Date.now()}`
      await dialog.getByLabel('Figure title', { exact: true }).fill(figureTitle)
      for (const kind of ['Residual distribution (not EVM)', 'AM/AM', 'AM/PM']) {
        await dialog.getByLabel('Panel type', { exact: true }).click()
        await page.getByRole('option', { name: kind, exact: true }).click()
        await dialog.getByRole('button', { name: 'Add panel', exact: true }).click()
      }
      assert.equal(await dialog.getByRole('button', { name: 'Add panel', exact: true }).isDisabled(), true)
      const previewResponse = page.waitForResponse(r => r.url().endsWith('/api/v1/figure-preview'))
      await dialog.getByRole('button', { name: 'Preview panels', exact: true }).click()
      const response = await previewResponse
      assert.equal(response.status(), 200, await response.text())
      const preview = await response.json()
      assert.equal(preview.figure.spec.panels.length, 4)
      const selector = '[data-testid="publication-panel-0"] .js-plotly-plot'
      await page.waitForFunction(sel => !!document.querySelector(sel)?._fullLayout, selector)
      const plot = page.locator(selector)
      await plot.scrollIntoViewIfNeeded()
      await plot.focus()
      await plot.press('+')
      await page.waitForFunction(sel => document.querySelector(sel)?._fullLayout?.xaxis?.autorange === false, selector)
      const viewport = await plot.evaluate(el => ({ x: el._fullLayout.xaxis.range, y: el._fullLayout.yaxis.range }))
      const saving = page.waitForResponse(r => r.url().endsWith('/api/v1/figures') && r.request().method() === 'POST')
      await dialog.getByRole('button', { name: 'Save view', exact: true }).click()
      const savedResponse = await saving
      assert.equal(savedResponse.status(), 200, await savedResponse.text())
      const saved = await savedResponse.json()
      assert.deepEqual(saved.spec.panels[0].x_range, viewport.x)
      assert.deepEqual(saved.spec.panels[0].y_range, viewport.y)
      await dialog.screenshot({ path: path.join(out, `publication-${width}.png`) })
      for (const [label, filename] of [['Export figure + data', `figure-${width}.zip`], ['Download full metric + figure reproduction', `reproduction-${width}.zip`]]) {
        const waiting = page.waitForEvent('download')
        await dialog.getByRole('link', { name: label, exact: true }).click()
        await (await waiting).saveAs(path.join(out, filename))
      }
      await dialog.getByRole('button', { name: 'Back', exact: true }).click()
      await page.getByRole('button', { name: 'Publication figure', exact: true }).click()
      dialog = page.getByRole('dialog', { name: 'Publication figure', exact: true })
      await dialog.getByLabel('Saved views', { exact: true }).click()
      await page.getByRole('option', { name: figureTitle, exact: true }).click()
      await page.waitForFunction(({ selector, expected }) => JSON.stringify(document.querySelector(selector)?._fullLayout?.xaxis?.range) === JSON.stringify(expected), { selector, expected: viewport.x })
      assert.equal(await dialog.locator('[data-testid^="publication-panel-"]').count(), 4)
      assert.deepEqual(await (await context.request.get(resultURL)).json(), originalResult)

      await page.goto(`${baseURL}/hardware?runs=${demo.run_id}&profile=general-spectral-v1`)
      await page.locator('[data-testid="hardware-scatter"] .js-plotly-plot').waitFor()
      await page.getByRole('button', { name: 'Add cost report', exact: true }).click()
      dialog = page.getByRole('dialog', { name: 'Add cost report', exact: true })
      await dialog.getByLabel('Upload supporting report (up to 5 MiB)', { exact: true }).setInputFiles({ name: 'synthetic-cost.txt', mimeType: 'text/plain', buffer: Buffer.from(`SYNTHETIC GUI fixture ${width}: 0.5 W and 5 nJ/sample, hypothetical FPGA digital board; no physical measurement.`) })
      await dialog.getByText(/SHA256:/).waitFor()
      await dialog.getByLabel('Report title', { exact: true }).fill(`SYNTHETIC cost fixture ${width}`)
      await dialog.getByLabel('Evidence source', { exact: true }).click()
      await page.getByRole('option', { name: 'FPGA board measurement', exact: true }).click()
      await dialog.getByLabel('Device / implementation / host', { exact: true }).fill('SYNTHETIC hypothetical FPGA board')
      await dialog.getByLabel('Workload / switching activity', { exact: true }).fill('Synthetic dense modulated input, batch 1')
      await dialog.getByLabel('Included and excluded cost boundaries', { exact: true }).fill('Digital board rail only; excludes host and RF chain. GUI fixture only.')
      await dialog.getByLabel('Module precision', { exact: true }).fill('weights: INT16 example\nfeatures: FP32 example\naccumulator: not specified')
      await dialog.getByLabel('Power (W)', { exact: true }).fill('0.5')
      await dialog.getByLabel('Energy per sample (nJ/sample)', { exact: true }).fill('5')
      await dialog.getByRole('checkbox', { name: 'This report contains synthetic/example costs', exact: true }).check()
      const recording = page.waitForResponse(r => r.url().endsWith('/api/v1/hardware/costs') && r.request().method() === 'POST')
      await dialog.getByRole('button', { name: 'Record costs', exact: true }).click()
      const recordedResponse = await recording
      assert.equal(recordedResponse.status(), 201, await recordedResponse.text())
      const recorded = await recordedResponse.json()
      assert.equal(recorded.synthetic, true)
      assert.equal(recorded.values.energy_j_per_sample, 5e-9)
      await page.getByLabel('Cost axis', { exact: true }).click()
      await page.getByRole('option', { name: 'Energy per sample (nJ/sample)', exact: true }).click()
      await page.waitForFunction(() => !document.querySelector('.MuiPopover-root'))
      await page.locator('[data-testid="hardware-scatter"]').scrollIntoViewIfNeeded()
      await page.screenshot({ path: path.join(out, `hardware-${width}.png`) })
      const ledgerDownload = page.waitForEvent('download')
      await page.getByRole('link', { name: 'Download cost ledger', exact: true }).click()
      await (await ledgerDownload).saveAs(path.join(out, `hardware-${width}.json`))

      await page.goto(`${baseURL}/sweeps`)
      await page.getByRole('button', { name: 'Create experiment matrix', exact: true }).click()
      dialog = page.getByRole('dialog')
      await dialog.getByLabel('Dataset', { exact: true }).click()
      await page.locator(`[role="option"][data-value="${demo.dataset_id}"]`).click()
      await dialog.getByLabel('Fixed PA surrogate', { exact: true }).click()
      await page.locator(`[role="option"][data-value="${demo.run_id}"]`).click()
      await dialog.getByLabel('Recipe', { exact: true }).click()
      await page.locator('[role="option"][data-value="dpd-gru-smoke-v1"]').click()
      await dialog.locator('summary').filter({ hasText: 'Model parameter overrides (JSON)' }).click()
      await dialog.getByLabel('Model key', { exact: true }).fill('qgru')
      await dialog.getByLabel('Model parameter overrides (JSON)', { exact: true }).fill('{"hidden_size":7}')
      await dialog.getByRole('checkbox', { name: 'Quantization-aware training (QAT)', exact: true }).check()
      await dialog.getByLabel('Activation bits', { exact: true }).fill('12')
      await dialog.getByLabel('Epochs', { exact: true }).fill('1')
      await dialog.getByLabel('Training seeds', { exact: true }).fill('0')
      const checking = page.waitForResponse(r => r.url().endsWith('/api/v1/sweeps/preview'))
      await dialog.getByRole('button', { name: 'Preview matrix', exact: true }).click()
      const checkedResponse = await checking
      assert.equal(checkedResponse.status(), 200, await checkedResponse.text())
      const checked = await checkedResponse.json()
      assert.deepEqual(checked.errors, [])
      assert.equal(checked.draft.methods[0].config.model.key, 'qgru')
      assert.equal(checked.draft.methods[0].config.quantization.n_bits_a, 12)
      await dialog.screenshot({ path: path.join(out, `precision-${width}.png`) })
      assert.deepEqual(errors, [])
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
      evidence.push({ width, height, run: demo.run_id, savedFigure: saved.figure_id, panels: saved.spec.panels.map(p => p.kind), viewport, metricsUnchanged: true, costEntry: recorded.entry_id, costSynthetic: true, precision: checked.draft.methods[0].config.quantization, publicWrites: 0, errors })
    } finally { await context.close() }
  }
} finally { await browser.close() }
await fs.writeFile(path.join(out, 'publication-hardware.json'), JSON.stringify(evidence, null, 2) + '\n')
console.log(JSON.stringify(evidence, null, 2))
