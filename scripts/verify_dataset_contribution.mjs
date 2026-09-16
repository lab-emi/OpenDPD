/** Real GUI: synthetic generation, CSV upload, explicit public preview; no live GitHub writes. */
import { chromium } from '../frontend/node_modules/playwright/index.mjs'
import assert from 'node:assert/strict'
import fs from 'node:fs/promises'
import path from 'node:path'

const [baseURL, out] = process.argv.slice(2)
await fs.mkdir(out, { recursive: true })
const fixture = await fs.readFile(new URL('../dataset/synthetic/studio-research-v1/synthetic-research-d0-r0/data.csv', import.meta.url))
const bootstrapToken = process.env.OPENDPD_BOOTSTRAP_TOKEN
if (!bootstrapToken) throw new Error('Set OPENDPD_BOOTSTRAP_TOKEN for the local test server')
const browser = await chromium.connectOverCDP(process.env.OPENDPD_CDP_URL ?? 'http://127.0.0.1:9222')
const evidence = []
try {
  for (const [width, height] of [[1366, 768], [1920, 1080]]) {
    const context = await browser.newContext({ viewport: { width, height }, acceptDownloads: true })
    try {
      const page = await context.newPage()
      const errors = [], submissions = []
      page.on('pageerror', e => errors.push(e.message))
      page.on('request', r => { if (r.url().includes('/dataset-publications/') && r.url().endsWith('/submit')) submissions.push(r.url()) })
      await page.goto(`${baseURL}/bootstrap?token=${encodeURIComponent(bootstrapToken)}`)
      await page.goto(baseURL + '/datasets')
      await page.getByRole('button', { name: 'Generate synthetic datasets', exact: true }).click()
      let dialog = page.getByRole('dialog')
      await dialog.getByLabel('Dataset ID prefix', { exact: true }).fill(`qa-synthetic-${width}`)
      const generating = page.waitForResponse(r => r.url().endsWith('/api/v1/datasets/synthetic') && r.request().method() === 'POST')
      await dialog.getByRole('button', { name: 'Generate private datasets', exact: true }).click()
      const generated = await (await generating).json()
      assert.equal(generated.datasets.length, 6)
      assert(generated.datasets.every(d => d.origin === 'synthetic' && d.simulation.physical_measurement === false))
      await dialog.getByText('Created 6 explicitly synthetic datasets.', { exact: true }).waitFor()
      await dialog.screenshot({ path: path.join(out, `synthetic-${width}.png`) })
      await dialog.getByRole('link', { name: 'Use these conditions in Sweep Board', exact: true }).click()
      await page.getByRole('dialog').getByText('Create experiment matrix', { exact: true }).waitFor()
      const previewing = page.waitForResponse(r => r.url().endsWith('/api/v1/sweeps/preview') && r.request().method() === 'POST')
      await page.getByRole('dialog').getByRole('button', { name: 'Preview matrix', exact: true }).click()
      const preview = await (await previewing).json()
      assert.deepEqual(preview.errors, [])
      assert.equal(preview.draft.condition_set.conditions.length, 3)
      assert(preview.warnings.some(w => w.includes('protocol rehearsal')))
      await page.goto(baseURL + '/datasets')
      await page.getByRole('button', { name: 'Create Your Own Dataset', exact: true }).click()
      dialog = page.getByRole('dialog')
      const filename = `browser-synthetic-${width}-${Date.now()}`
      await dialog.getByLabel('Choose CSV file', { exact: true }).setInputFiles({ name: filename + '.csv', mimeType: 'text/csv', buffer: fixture })
      await dialog.getByText(/All 16,384 sample rows passed/).waitFor()
      await dialog.getByRole('button', { name: 'Continue', exact: true }).click()
      // The source is our synthetic fixture; explicitly declare its origin.
      await dialog.getByLabel('Origin', { exact: true }).click()
      await page.getByRole('option', { name: 'synthetic', exact: true }).click()
      await dialog.getByRole('button', { name: 'Review', exact: true }).click()
      await dialog.getByText(/Validated 16,384 paired samples/).waitFor()
      const publicChoice = dialog.getByRole('checkbox', { name: 'After creation, prepare a public contribution for human review', exact: true })
      assert.equal(await publicChoice.isChecked(), false)
      await publicChoice.check()
      await dialog.getByRole('button', { name: 'Create dataset', exact: true }).click()
      dialog = page.getByRole('dialog', { name: 'Prepare a public dataset PR', exact: true })
      await dialog.waitFor()
      await dialog.getByLabel('Public dataset description', { exact: true }).fill('Synthetic browser upload demonstration; no measured RF evidence.')
      await dialog.getByLabel('Public author / attribution', { exact: true }).fill('OpenDPD synthetic fixture generator')
      await dialog.getByLabel('Dataset license', { exact: true }).click()
      await page.getByRole('option', { name: 'CC0-1.0', exact: true }).click()
      const preparing = page.waitForResponse(r => r.url().endsWith('/api/v1/dataset-publications/prepare') && r.request().method() === 'POST')
      await dialog.getByRole('button', { name: 'Prepare and inspect package', exact: true }).click()
      const preparedResponse = await preparing
      assert.equal(preparedResponse.status(), 201, await preparedResponse.text())
      const prepared = await preparedResponse.json()
      assert.equal(prepared.status, 'prepared')
      assert.equal(prepared.catalog.origin, 'synthetic')
      assert.deepEqual(prepared.files.map(f => f.path).sort(), ['README.md', 'data.csv', 'dataset.json'])
      const download = page.waitForEvent('download')
      await dialog.getByText('Download exact preview package', { exact: true }).click()
      const downloaded = await download
      assert.equal(downloaded.suggestedFilename(), prepared.publication_id + '.zip')
      await downloaded.saveAs(path.join(out, `private-preview-${width}.zip`))
      await dialog.getByText('Private preview', { exact: true }).scrollIntoViewIfNeeded()
      await dialog.screenshot({ path: path.join(out, `publication-preview-${width}.png`) })
      assert.equal(await dialog.getByRole('button', { name: 'Publish and request human review', exact: true }).isEnabled(), false)
      assert.equal(await dialog.getByRole('link', { name: 'emi.lab@outlook.com', exact: true }).getAttribute('href'), 'mailto:emi.lab@outlook.com')
      assert.deepEqual(submissions, [], 'browser smoke must not actually publish data')
      assert.equal(await page.evaluate(() => document.documentElement.scrollWidth > innerWidth), false)
      assert.deepEqual(errors, [])
      evidence.push({ width, height, syntheticDatasets: generated.datasets.map(d => d.dataset_id), conditionPreviewCells: preview.cells.length, uploadedDataset: prepared.dataset_id, publication: prepared.publication_id, status: prepared.status, publicWrites: submissions.length, errors })
    } finally { await context.close() }
  }
} finally { await browser.close() }
await fs.writeFile(path.join(out, 'datasets.json'), JSON.stringify(evidence, null, 2) + '\n')
console.log(JSON.stringify(evidence, null, 2))
