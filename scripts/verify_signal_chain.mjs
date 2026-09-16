/** Real saved results: signal positions, independent interactions and screenshots. */
import { chromium } from '../frontend/node_modules/playwright/index.mjs'
import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import path from 'node:path'

const [base, flowFile, out] = process.argv.slice(2)
const flow = JSON.parse(await fs.readFile(flowFile, 'utf8'))[0]
await fs.mkdir(out, { recursive: true })
const bootstrapToken = process.env.OPENDPD_BOOTSTRAP_TOKEN
if (!bootstrapToken) throw new Error('Set OPENDPD_BOOTSTRAP_TOKEN for the local test server')
const browser = await chromium.connectOverCDP(process.env.OPENDPD_CDP_URL ?? 'http://127.0.0.1:9222')
const evidence = []
try {
  for (const [width, height] of [[1366, 768], [1920, 1080], [390, 844]]) {
    const context = await browser.newContext({ viewport: { width, height }, acceptDownloads: true })
    try {
      const page = await context.newPage(), errors = []
      page.on('pageerror', e => errors.push(e.message))
      await page.goto(base + '/bootstrap?token=' + encodeURIComponent(bootstrapToken))
      await page.goto(`${base}/results/${flow.runs[3]}`)
      const review = page.getByTestId('spectrum-review')
      await review.waitFor()
      const chart = node => review.locator(`[data-signal-node="${node}"] .js-plotly-plot`)
      for (const node of ['dpd_input', 'pa_input', 'pa_output']) await chart(node).locator('svg.main-svg').first().waitFor()
      const read = async node => chart(node).evaluate(el => ({ names: el._fullData.map(tr => tr.name), y: el._fullData.map(tr => Array.from(tr.y)), xRange: el._fullLayout.xaxis.range, yRange: el._fullLayout.yaxis.range, visible: el._fullData.map(tr => tr.visible) }))
      const before = await Promise.all(['dpd_input', 'pa_input', 'pa_output'].map(read))
      assert.deepEqual(before.map(p => p.names.length), [1, 1, 4])
      assert.deepEqual(before[0].yRange, before[2].yRange)
      assert(before[2].names.some(n => n.includes('synthetic')))
      const raw = await page.evaluate(async id => (await fetch(`/api/v1/artifacts/${id}/plot-spectrum`)).json(), flow.runs[3])
      assert.deepEqual(before.flatMap(p => p.y), raw.traces.map(tr => tr.psd_db))
      await chart('dpd_input').focus()
      await page.keyboard.press('+')
      await page.waitForFunction(before => {
        const el = document.querySelector('[data-testid="spectrum-review"] [data-signal-node="dpd_input"] .js-plotly-plot')
        return el._fullLayout.xaxis.range[1] - el._fullLayout.xaxis.range[0] < before
      }, before[0].xRange[1] - before[0].xRange[0])
      assert.deepEqual((await read('pa_output')).xRange, before[2].xRange)
      await chart('dpd_input').focus(); await page.keyboard.press('Home')
      if (width >= 1000) {
        await review.getByLabel('Figure title', { exact: true }).fill('Signal-chain PSD · synthetic PA / DPD')
        const response = page.waitForResponse(r => r.url().endsWith('/api/v1/figures') && r.request().method() === 'POST')
        await review.getByRole('button', { name: 'Save view', exact: true }).click()
        const savedResponse = await response
        assert.equal(savedResponse.status(), 200)
        const saved = await savedResponse.json()
        assert.deepEqual(saved.spec.panels.map(p => p.signal_node), ['dpd_input', 'pa_input', 'pa_output'])
        const archive = await page.request.get(`${base}/api/v1/figures/${saved.figure_id}/export`)
        assert.equal(archive.status(), 200)
        await fs.writeFile(path.join(out, `figure-${width}.zip`), await archive.body())
      }
      await page.reload()
      for (const node of ['dpd_input', 'pa_input', 'pa_output']) await chart(node).locator('svg.main-svg').first().waitFor()
      if (width >= 1000) await review.getByLabel('Figure title', { exact: true }).fill('Signal-chain PSD · synthetic PA / DPD')
      await review.screenshot({ path: path.join(out, `psd-${width}.png`), animations: 'disabled' })
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
      await chart('pa_output').locator('.legendtoggle').first().click()
      await page.waitForFunction(() => document.querySelector('[data-testid="spectrum-review"] [data-signal-node="pa_output"] .js-plotly-plot')._fullData[0].visible === 'legendonly')
      assert.equal((await read('dpd_input')).visible[0], true)
      assert.deepEqual(errors, [])
      evidence.push({ width, height, positions: before.map(p => p.names), exactStoredBins: true, independentZoomAndLegend: true, errors })
    } finally { await context.close() }
  }
} finally { await browser.close() }
await fs.writeFile(path.join(out, 'signal-chain.json'), JSON.stringify(evidence, null, 2) + '\n')
console.log(JSON.stringify(evidence, null, 2))
