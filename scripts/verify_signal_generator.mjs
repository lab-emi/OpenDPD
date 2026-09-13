/** Real local UI and workers; all generated samples are synthetic and private. */
import { chromium } from '../frontend/node_modules/playwright/index.mjs'
import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import path from 'node:path'
import { randomUUID } from 'node:crypto'

const [baseURL, out] = process.argv.slice(2)
await fs.mkdir(out, { recursive: true })
const browser = await chromium.connectOverCDP('http://127.0.0.1:9222')
const evidence = []
const suffix = Date.now().toString(36)
try {
  for (const [width, height] of [[1366, 768], [1920, 1080]]) {
    const context = await browser.newContext({ viewport: { width, height }, acceptDownloads: true })
    try {
      const page = await context.newPage()
      const errors = [], publications = []
      page.on('pageerror', e => errors.push(e.message))
      page.on('request', r => { if (r.url().includes('/dataset-publications/') && r.url().endsWith('/submit')) publications.push(r.url()) })
      const api = (url, method = 'GET', body) => page.evaluate(async ({ url, method, body }) => {
        const auth = await (await fetch('/api/v1/session')).json()
        const response = await fetch('/api/v1' + url, { method, headers: { 'Content-Type': 'application/json', 'X-OpenDPD-CSRF': auth.csrf_token }, body: body === undefined ? undefined : JSON.stringify(body) })
        const data = await response.json()
        if (!response.ok) throw new Error(JSON.stringify(data))
        return data
      }, { url, method, body })
      const generated = async (button) => {
        const waiting = page.waitForResponse(r => r.url().endsWith('/signal-generator/signals') && r.request().method() === 'POST')
        await button.click()
        const response = await waiting
        assert.equal(response.status(), 201, await response.text())
        const signal = await response.json()
        await page.getByTestId('signal-generator-results').waitFor()
        return signal
      }
      const plots = () => page.waitForFunction(() => document.querySelectorAll('.js-plotly-plot .plot-container').length === 4)
      await page.goto(baseURL + '/bootstrap?token=studio-next-local-review')
      await page.getByRole('link', { name: 'Get Started', exact: true }).click()
      const guide = page.getByRole('dialog')
      assert.deepEqual((await guide.getByRole('button').allTextContents()).slice(0, 3), ['Signal Generator', 'Use an existing dataset', 'Upload CSV'])
      assert.equal(await guide.locator('.MuiButton-contained').count(), 1)
      await guide.screenshot({ path: path.join(out, `get-started-${width}.png`) })
      const initial = await generated(guide.getByRole('button', { name: 'Signal Generator', exact: true }))
      await plots()
      await page.getByRole('button', { name: 'Create synthetic PA dataset', exact: true }).waitFor()
      await page.screenshot({ path: path.join(out, `generator-${width}.png`), fullPage: true })
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
      await page.getByRole('button', { name: /Wi-Fi 8/ }).click()
      assert.equal(await page.getByRole('button', { name: 'Export I/Q + configuration', exact: true }).isEnabled(), false)
      const wifi8 = await generated(page.getByRole('button', { name: 'Generate & preview', exact: true }))
      assert.equal(wifi8.coverage, 'experimental')
      assert(wifi8.analysis.notes.some(n => n.includes('UHR')))
      await page.getByRole('button', { name: /Custom/ }).click()
      await page.getByLabel('I/Q samples', { exact: true }).fill('32768')
      await page.getByRole('button', { name: 'Advanced parameters', exact: true }).click()
      await page.getByLabel('Subcarriers including pilots', { exact: true }).fill('106')
      await page.getByRole('button', { name: 'Add OFDMA channel', exact: true }).click()
      await page.getByLabel('Subcarriers including pilots', { exact: true }).nth(1).fill('52')
      await page.getByRole('combobox', { name: 'Pilot allocation', exact: true }).click()
      await page.getByRole('option', { name: 'Explicit signed bin indices', exact: true }).click()
      await page.getByLabel('Pilot carrier indices', { exact: true }).fill('-50, 50')
      await page.getByLabel('RF carrier frequency (GHz)', { exact: true }).fill('3.5')
      await page.getByLabel('Add white Gaussian noise', { exact: true }).check()
      await page.getByLabel('SNR relative to target RMS (dB)', { exact: true }).fill('25')
      const custom = await generated(page.getByRole('button', { name: 'Apply & regenerate', exact: true }))
      assert.equal(custom.analysis.sample_count, 32768)
      assert.equal(custom.analysis.active_carriers, 158)
      assert.equal(custom.analysis.pilot_carriers, 2)
      assert(custom.analysis.evm_percent > 0.5)
      await page.getByRole('button', { name: 'Advanced parameters', exact: true }).click()
      await page.getByRole('tab', { name: 'Resource allocation', exact: true }).click()
      await page.screenshot({ path: path.join(out, `allocation-${width}.png`), fullPage: true })
      await page.getByRole('tab', { name: 'Metrics & provenance', exact: true }).click()
      await page.screenshot({ path: path.join(out, `metrics-${width}.png`), fullPage: true })
      const downloading = page.waitForEvent('download')
      await page.getByRole('button', { name: 'Export I/Q + configuration', exact: true }).click()
      await (await downloading).saveAs(path.join(out, `waveform-${width}.zip`))
      await page.getByRole('button', { name: 'Create synthetic PA dataset', exact: true }).click()
      const dialog = page.getByRole('dialog')
      const datasetId = `qa-generator-${width}-${suffix}`
      await dialog.getByLabel('Dataset ID', { exact: true }).fill(datasetId)
      const creating = page.waitForResponse(r => r.url().endsWith('/dataset') && r.request().method() === 'POST')
      await dialog.getByRole('button', { name: 'Create dataset & open PA Model', exact: true }).click()
      const created = await creating
      assert.equal(created.status(), 201, await created.text())
      const dataset = await created.json()
      assert.equal(dataset.dataset.origin, 'synthetic')
      await page.getByRole('heading', { name: 'PA Model', exact: true }).waitFor()
      assert.equal(await page.getByRole('navigation', { name: 'Choose a task' }).getByRole('link').count(), 2)
      assert.equal(await page.getByRole('tab', { name: 'Training', exact: true }).getAttribute('aria-selected'), 'true')
      await page.getByRole('tab', { name: 'Testing', exact: true }).click()
      const summary = page.getByTestId('testing-samples')
      await summary.getByText(dataset.test_samples.toLocaleString('en-US'), { exact: false }).waitFor()
      assert.equal(await page.getByRole('tab', { name: 'Testing', exact: true }).getAttribute('aria-selected'), 'true')
      await page.screenshot({ path: path.join(out, `pa-testing-${width}.png`), fullPage: true })
      const countInfo = await api(`/datasets/${datasetId}/sample-counts`)
      assert.equal(countInfo.counts.test, dataset.test_samples)
      // Exercise actual CPU PA and DPD workers on the generated dataset, then
      // select those checkpoints through the merged Testing interfaces.
      const recipes = await api('/recipes')
      const models = await api('/models')
      const leastSquares = new Set(models.filter(m => m.training_method === 'least_squares').map(m => m.key))
      const runs = []
      const waitRun = async id => {
        for (let attempt = 0; attempt < 120; attempt++) {
          const run = await api('/runs/' + id)
          if (['failed', 'cancelled', 'interrupted'].includes(run.status)) throw new Error(JSON.stringify(run))
          if (run.status === 'succeeded') return run
          await page.waitForTimeout(250)
        }
        throw new Error('CPU run did not finish within 30 seconds')
      }
      for (const task of ['train_pa', 'train_dpd']) {
        const recipe = recipes.find(r => r.task === task && (task === 'train_pa' ? r.model.key === 'gru' && r.purpose === 'smoke' : leastSquares.has(r.model.key)))
        assert(recipe, `Missing CPU recipe for ${task}`)
        const config = { task, dataset: { id: datasetId }, recipe_id: recipe.recipe_id, model: recipe.model, training: recipe.training, execution: { device: 'cpu', num_threads: 2 }, evaluation: { profile_id: 'general-spectral-v1' }, ...(task === 'train_dpd' ? { pa_reference: { run_id: runs[0] } } : {}) }
        const run = await api('/runs', 'POST', { config, idempotency_key: randomUUID(), name: `Generator GUI QA ${task} ${width}` })
        await waitRun(run.run_id); runs.push(run.run_id)
      }
      for (const [index, task] of ['evaluate_pa', 'run_dpd'].entries()) {
        await page.goto(`${baseURL}/experiments/new?task=${task}&modelRun=${runs[index]}&dataset=${datasetId}`)
        await page.getByTestId('testing-samples').getByText(dataset.test_samples.toLocaleString('en-US'), { exact: false }).waitFor()
        await page.getByRole('button', { name: 'Continue', exact: true }).click()
        await page.getByRole('combobox', { name: 'Device', exact: true }).click()
        await page.getByRole('option', { name: 'cpu', exact: true }).click()
        await page.getByRole('button', { name: 'Continue', exact: true }).click()
        const starting = page.waitForResponse(r => r.url().endsWith('/api/v1/runs') && r.request().method() === 'POST')
        await page.getByRole('button', { name: 'Start run', exact: true }).click()
        const response = await starting
        assert.equal(response.status(), 201, await response.text())
        const run = await response.json()
        await waitRun(run.run_id); runs.push(run.run_id)
      }
      assert.deepEqual(errors, [])
      assert.deepEqual(publications, [])
      evidence.push({ width, height, initialSignal: initial.signal_id, customSignal: custom.signal_id, wifi8Coverage: wifi8.coverage, samples: custom.analysis.sample_count, activeCarriers: 158, pilotCarriers: 2, referenceEvmPercent: custom.analysis.evm_percent, dataset: datasetId, testSamples: dataset.test_samples, runs, errors, publications })
      await fs.writeFile(path.join(out, 'signal-generator.json'), JSON.stringify(evidence, null, 2) + '\n')
    } finally { await context.close() }
  }
} finally { await browser.close() }
console.log(JSON.stringify(evidence, null, 2))
