/** Real-server browser acceptance. Reuses the installed isolated Chromium CDP service.
 * node scripts/verify_research_review.mjs /tmp/studio-next-review-demo http://127.0.0.1:8779 /tmp/studio-next-browser-qa
 */
import { chromium } from '../frontend/node_modules/playwright/index.mjs'
import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import path from 'node:path'
const [workspace, baseURL, out] = process.argv.slice(2)
if (!workspace || !baseURL || !out) throw new Error('workspace, base URL and output directory are required')
const demo = JSON.parse(await fs.readFile(path.join(workspace, 'review-demo/index.json'), 'utf8'))
await fs.mkdir(out, { recursive: true })
const browser = await chromium.connectOverCDP('http://127.0.0.1:9222')
const measurements = []
try {
  for (const [width, height] of [[1366, 768], [1920, 1080]]) {
    const context = await browser.newContext({ viewport: { width, height }, acceptDownloads: true })
    try {
      const page = await context.newPage()
      const errors = []
      page.on('pageerror', error => errors.push(error.message))
      await page.goto(`${baseURL}/bootstrap?token=studio-next-local-review`)
      const start = performance.now()
      await page.goto(baseURL + demo.compare_path)
      const selector = '[data-testid="spectrum-plot"] [role="figure"]'
      await page.locator(`${selector} .main-svg`).first().waitFor()
      const loadMs = performance.now() - start
      const beforeMetrics = await page.request.get(`${baseURL}/api/v1/results/${demo.runs.gru}`)
      const original = await beforeMetrics.json()
      const started = performance.now()
      await page.getByRole('combobox', { name: 'Reference result', exact: true }).click()
      await page.getByRole('option', { name: demo.runs.gru, exact: true }).click()
      await page.waitForFunction(run => new URL(location.href).searchParams.get('reference') === run, demo.runs.gru)
      await page.waitForFunction(sel => document.querySelector(sel)?.data?.filter(t => t.visible !== 'legendonly').length === 3, selector)
      const referenceSwitchMs = performance.now() - started
      await page.getByLabel('Frequency cursor (MHz)', { exact: true }).fill('115.2')
      await page.getByLabel('Show reference integration bands', { exact: true }).uncheck()
      await page.getByLabel('Figure title', { exact: true }).fill(`Review ${width}`)
      const plot = page.locator(selector)
      await plot.scrollIntoViewIfNeeded()
      await plot.focus()
      await plot.press('+')
      await page.waitForFunction(sel => document.querySelector(sel)?._fullLayout?.xaxis?.autorange === false, selector)
      const expectedRanges = await plot.evaluate(el => ({ x: el._fullLayout.xaxis.range, y: el._fullLayout.yaxis.range }))
      const saving = page.waitForResponse(r => r.url().endsWith('/api/v1/figures') && r.request().method() === 'POST')
      await page.getByRole('button', { name: 'Save view', exact: true }).click()
      const savedResponse = await saving
      assert.equal(savedResponse.status(), 200)
      const saved = await savedResponse.json()
      assert.deepEqual(saved.spec.panels[0].x_range, expectedRanges.x)
      assert.deepEqual(saved.spec.panels[0].y_range, expectedRanges.y)
      assert.equal(saved.spec.reference_run_id, demo.runs.gru)
      assert.equal(saved.spec.panels[0].show_bands, false)
      assert.equal(saved.spec.panels[0].cursor_x, 115.2)
      await page.reload()
      await page.getByRole('combobox', { name: 'Saved views', exact: true }).click()
      await page.locator(`[role=option][data-value="${saved.figure_id}"]`).click()
      await page.waitForFunction(({ sel, ranges }) => {
        const layout = document.querySelector(sel)?._fullLayout
        return layout && JSON.stringify(layout.xaxis.range) === JSON.stringify(ranges.x) && JSON.stringify(layout.yaxis.range) === JSON.stringify(ranges.y)
      }, { sel: selector, ranges: expectedRanges })
      assert.equal(await page.getByLabel('Frequency cursor (MHz)', { exact: true }).inputValue(), '115.2')
      assert.equal(await page.getByLabel('Show reference integration bands', { exact: true }).isChecked(), false)
      const exportLink = page.getByRole('link', { name: 'Export figure + data', exact: true })
      await exportLink.waitFor({ timeout: 5000 })
      const downloading = page.waitForEvent('download')
      await exportLink.click()
      const download = await downloading
      assert.equal(download.suggestedFilename(), `${saved.figure_id}.zip`)
      await download.saveAs(path.join(out, `review-${width}.zip`))
      const afterMetrics = await (await page.request.get(`${baseURL}/api/v1/results/${demo.runs.gru}`)).json()
      assert.deepEqual(afterMetrics, original)
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
      await page.evaluate(() => window.scrollTo(0, 0))
      await page.screenshot({ path: path.join(out, `comparison-${width}.png`), fullPage: true })
      await page.goto(`${baseURL}/results/${demo.runs.gru}`)
      await page.getByTestId('rf-facts').waitFor()
      await page.locator(`${selector} .main-svg`).first().waitFor()
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
      await page.screenshot({ path: path.join(out, `result-${width}.png`), fullPage: true })
      assert.deepEqual(errors, [])
      measurements.push({ width, height, runs: 2, samplesPerRun: original.dataset.n_samples, loadMs, referenceSwitchMs,
        restoredReference: saved.spec.reference_run_id, viewport: expectedRanges, unchangedMetrics: true, errors })
    } finally { await context.close() }
  }
} finally { await browser.close() }
await fs.writeFile(path.join(out, 'measurements.json'), JSON.stringify(measurements, null, 2) + '\n')
console.log(JSON.stringify(measurements, null, 2))
