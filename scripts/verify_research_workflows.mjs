/** Browser check for the real local sweep UI and explicitly MOCK measurement/session evidence. */
import { chromium } from '../frontend/node_modules/playwright/index.mjs'
import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import path from 'node:path'
const [workspace, baseURL, out] = process.argv.slice(2)
const demo = JSON.parse(await fs.readFile(path.join(workspace, 'review-demo/measurement-index.json'), 'utf8'))
await fs.mkdir(out, { recursive: true })
const bootstrapToken = process.env.OPENDPD_BOOTSTRAP_TOKEN
if (!bootstrapToken) throw new Error('Set OPENDPD_BOOTSTRAP_TOKEN for the local test server')
const browser = await chromium.connectOverCDP(process.env.OPENDPD_CDP_URL ?? 'http://127.0.0.1:9222')
const evidence = []
try {
  for (const [width, height] of [[1366, 768], [1920, 1080]]) {
    const context = await browser.newContext({ viewport: { width, height } })
    try {
      const page = await context.newPage()
      const errors = []
      page.on('pageerror', e => errors.push(e.message))
      await page.goto(`${baseURL}/bootstrap?token=${encodeURIComponent(bootstrapToken)}`)
      await page.goto(baseURL + demo.result_path)
      await page.getByTestId('measurement').waitFor()
      const measurement = page.getByTestId('measurement')
      await measurement.scrollIntoViewIfNeeded()
      assert((await measurement.innerText()).includes('123.37'))
      assert((await measurement.innerText()).includes('measurement-fractional-v2'))
      assert((await measurement.innerText()).includes('Before timing correction'))
      assert((await measurement.innerText()).includes('mock instrument adapter'))
      await page.evaluate(() => window.scrollTo(0, 0))
      await page.screenshot({ path: path.join(out, `alignment-${width}.png`), fullPage: true })
      const sessions = page.getByRole('region', { name: 'Measurement sessions', exact: true })
      await sessions.scrollIntoViewIfNeeded()
      await sessions.getByRole('table').first().waitFor()
      assert((await sessions.innerText()).includes('MOCK acquisition grouping'))
      await sessions.screenshot({ path: path.join(out, `sessions-${width}.png`) })
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
      await page.goto(baseURL + '/sweeps')
      await page.getByRole('button', { name: 'Create experiment matrix', exact: true }).click()
      const dialog = page.getByRole('dialog')
      await dialog.getByLabel('Matrix title', { exact: true }).fill(`Browser CPU smoke ${width} ${Date.now()}`)
      await dialog.getByLabel('Epochs', { exact: true }).fill('1')
      await dialog.getByLabel('Training seeds', { exact: true }).fill('0')
      await dialog.getByLabel('Maximum runs', { exact: true }).fill('1')
      await dialog.getByLabel('Wall-clock limit (s)', { exact: true }).fill('120')
      const previewing = page.waitForResponse(r => r.url().endsWith('/api/v1/sweeps/preview') && r.request().method() === 'POST')
      await dialog.getByRole('button', { name: 'Preview matrix', exact: true }).click()
      const previewResponse = await previewing
      assert.equal(previewResponse.status(), 200, await previewResponse.text())
      const preview = await previewResponse.json()
      assert.deepEqual(preview.errors, [])
      assert.equal(preview.training_runs, 1)
      await page.waitForFunction(() => [...document.querySelectorAll('button')].some(b => b.textContent === 'Register plan' && !b.disabled))
      await dialog.screenshot({ path: path.join(out, `matrix-preview-${width}.png`) })
      const registering = page.waitForResponse(r => r.url().endsWith('/api/v1/sweeps') && r.request().method() === 'POST')
      await dialog.getByRole('button', { name: 'Register plan', exact: true }).click()
      const registered = await (await registering).json()
      assert.equal(registered.status, 'ready')
      await page.getByRole('button', { name: 'Start matrix', exact: true }).click()
      // An ordinary CPU smoke worker must finish; no browser-side result fabrication.
      const deadline = Date.now() + 60000
      let record
      while (Date.now() < deadline) {
        record = await (await page.request.get(`${baseURL}/api/v1/sweeps/${registered.sweep_id}`)).json()
        if (record.status !== 'ready' && record.status !== 'running') break
        await new Promise(resolve => setTimeout(resolve, 500))
      }
      assert.equal(record.status, 'complete', JSON.stringify(record.cells))
      assert.equal(record.cells[0].status, 'succeeded')
      await page.getByRole('heading', { name: 'Training-seed summary', exact: true }).waitFor()
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
      await page.evaluate(() => window.scrollTo(0, 0))
      await page.screenshot({ path: path.join(out, `matrix-result-${width}.png`), fullPage: true })
      assert.deepEqual(errors, [])
      evidence.push({ width, height, sweep: registered.sweep_id, run: record.cells[0].run_id, status: record.status, measurement: demo.runs[1], mockOnly: true, errors })
    } finally { await context.close() }
  }
} finally { await browser.close() }
await fs.writeFile(path.join(out, 'workflows.json'), JSON.stringify(evidence, null, 2)+'\n')
console.log(JSON.stringify(evidence, null, 2))
