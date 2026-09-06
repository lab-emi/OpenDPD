import { expect, test } from '@playwright/test'
import { installFakeApi } from './mock-api'

test.describe('J1 — reproduce the built-in example (mock API)', () => {
  test('register example → new experiment → run detail → result, keyboard friendly', async ({ page }) => {
    const state = await installFakeApi(page)
    await page.goto('/')
    await expect(page.getByRole('heading', { level: 1, name: 'Home' })).toBeVisible()
    await expect(page.getByText('built-in measured data', { exact: true })).toBeVisible()
    await page.getByRole('button', { name: 'Register example dataset' }).click()
    await expect(page.getByText('Example dataset registered')).toBeVisible()

    await page.getByRole('link', { name: 'New experiment' }).click()
    await expect(page.getByRole('heading', { level: 1, name: 'New experiment' })).toBeVisible()
    await expect(page.getByText(/Smoke recipe: a few epochs/)).toBeVisible()
    await expect(page.getByText('Configuration is valid')).toBeVisible()
    await page.getByLabel('Name (optional)').fill('e2e smoke')
    // the metric profile is an explicit, registry-backed choice (S08); the default stays the frozen legacy one
    await page.getByRole('button', { name: 'Advanced settings' }).click()
    await page.getByLabel('Metric profile').click()
    await page.getByRole('option', { name: /general-spectral-v1/ }).click()
    await expect(page.getByText('Configuration is valid')).toBeVisible()
    await page.getByRole('button', { name: 'Start run' }).click()

    await expect(page).toHaveURL(/\/runs\/run-e2e-0001$/)
    expect(state.submitted).toHaveLength(1)
    const submittedConfig = state.submitted[0]?.['config'] as { evaluation?: { profile_id?: string } } | undefined
    expect(submittedConfig?.evaluation?.profile_id).toBe('general-spectral-v1')
    await expect(page.getByRole('heading', { level: 1, name: 'e2e smoke' })).toBeVisible()
    await expect(page.getByText('Succeeded').first()).toBeVisible()
    // refresh: same run, no second submission
    await page.reload()
    await expect(page.getByRole('heading', { level: 1, name: 'e2e smoke' })).toBeVisible()
    expect(state.submitted).toHaveLength(1)

    await page.getByRole('tab', { name: 'Logs' }).click()
    await expect(page.getByText('Training Completed...')).toBeVisible()
    await page.getByRole('tab', { name: 'Artifacts' }).click()
    await expect(page.getByRole('link', { name: 'Download' }).first()).toBeVisible()

    await page.getByRole('link', { name: 'Open result' }).click()
    await expect(page.locator('[data-evidence="pa_modeling"]')).toBeVisible()
    await expect(page.locator('[data-mock="true"]')).toBeVisible()
    await expect(page.locator('[data-metric="NMSE"]')).toContainText('dB')
    await expect(page.locator('[data-metric="NMSE"]')).toContainText('lower is better')
    await expect(page.locator('[data-profile="legacy-opendpd-v1"]')).toBeVisible()
    await page.getByLabel('Metric profile').click()
    await page.getByRole('option', { name: 'general-spectral-v1' }).click()
    await expect(page.locator('[data-profile="general-spectral-v1"]')).toBeVisible()
    await expect(page.locator('[data-metric="IBE"]')).toContainText('-23.10 dB')
    await page.getByRole('button', { name: 'Metric definitions' }).click()
    await expect(page.getByText(/pooled over all valid samples/)).toBeVisible()
  })

  test('main pages fit 1366×768 and 1920×1080 without horizontal scroll', async ({ page }, testInfo) => {
    await installFakeApi(page)
    for (const path of ['/', '/datasets', '/experiments', '/experiments/new', '/results', '/settings']) {
      await page.goto(path)
      await expect(page.getByRole('main')).toBeVisible()
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)
      expect(overflow, `${path} overflows horizontally at ${testInfo.project.name}`).toBeLessThanOrEqual(0)
    }
  })

  test('unauthenticated session shows the bootstrap explanation, not a blank page', async ({ page }) => {
    await page.route('**/api/v1/session', (route) => route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify({ authenticated: false, csrf_token: null, version: 'x' }) }))
    await page.goto('/')
    await expect(page.getByRole('heading', { name: 'Session required' })).toBeVisible()
    await expect(page.getByLabel('Bootstrap token')).toBeFocused()
  })
})

test.describe('J3 — share and reproduce (mock API)', () => {
  test('export a share package from a result, then import a package and open the imported run', async ({ page }) => {
    await installFakeApi(page)
    await page.goto('/results/run-pa-0001')
    const panel = page.getByRole('region', { name: 'Export and report' })
    await expect(panel.getByRole('link', { name: 'Report (HTML)' })).toHaveAttribute('href', '/api/v1/results/run-pa-0001/report?format=html')
    await expect(panel.getByRole('link', { name: 'Report (Markdown)' })).toHaveAttribute('href', '/api/v1/results/run-pa-0001/report?format=md')
    await panel.getByRole('button', { name: 'Export share package' }).click()
    const ready = page.getByTestId('export-ready')
    await expect(ready).toContainText('Package ready: run-pa-0001-share-20260906.zip')
    await expect(ready).toContainText('worker logs are not included')
    await expect(ready.getByRole('link', { name: 'Download package' })).toHaveAttribute('href', '/api/v1/exports/run-pa-0001-share-20260906')

    await page.goto('/experiments')
    await page.getByTestId('import-package').setInputFiles({ name: 'run-pa-0001-share.zip', mimeType: 'application/zip', buffer: Buffer.from('zip') })
    const report = page.getByTestId('import-report')
    await expect(report).toContainText('Imported run-imported-0001 into this workspace. Dataset capture: missing.')
    await expect(report).toContainText('dataset capture (raw sha256 …)')
    await expect(page.getByRole('table', { name: 'Experiments' })).toContainText('imported share package')
    await report.getByRole('link', { name: 'Open the imported run' }).click()
    await expect(page).toHaveURL(/\/runs\/run-imported-0001$/)
    await expect(page.getByRole('heading', { level: 1, name: 'imported share package' })).toBeVisible()
  })
})

test.describe('J2 — my own data (mock API)', () => {
  test('import CSV with odd headers → doctor → accept estimates → new version → experiment uses it', async ({ page }) => {
    const state = await installFakeApi(page)
    await page.goto('/datasets')
    await page.getByRole('button', { name: 'Import my data' }).click()
    const dialog = page.getByRole('dialog')
    await dialog.getByText('capture.csv').click()
    await expect(dialog.getByRole('region', { name: 'Column mapping' })).toBeVisible()
    await expect(dialog.getByLabel('I_out')).toHaveText('rx_i')
    await expect(dialog.getByLabel('Dataset id')).toHaveValue('capture')
    await dialog.getByLabel('Sample rate (Hz)').fill('800e6')
    await dialog.getByLabel('Signal bandwidth (Hz)').fill('200e6')
    await dialog.getByLabel('Sub-channels').fill('10')
    await dialog.getByLabel('PSD segment length (nperseg)').fill('2560')
    await dialog.getByRole('button', { name: 'Import', exact: true }).click()

    await expect(page).toHaveURL(/\/datasets\/capture$/)
    await expect(page.getByText(/No report yet/)).toBeVisible()
    await page.getByRole('button', { name: 'Run Dataset Doctor' }).click()
    await expect(page.getByRole('article', { name: 'Time misalignment' })).toBeVisible()

    await page.getByRole('button', { name: 'Preprocess…' }).click()
    const pre = page.getByRole('dialog')
    await pre.getByRole('button', { name: 'Use doctor estimates' }).click()
    await expect(pre.getByLabel('Delay correction (samples)')).toHaveValue('6')
    await pre.getByRole('button', { name: 'Preview' }).click()
    await expect(pre.getByRole('article', { name: 'Aligned' })).toBeVisible()
    await pre.getByLabel('New version name').fill('aligned-v1')
    await pre.getByRole('button', { name: 'Create version' }).click()
    await expect(page.getByText('Version aligned-v1 created.')).toBeVisible()
    await expect(page.locator('[data-version="aligned-v1"]')).toBeVisible()

    // the version is a first-class choice when configuring an experiment and travels in the config
    await page.goto('/experiments/new')
    await expect(page.getByText('Configuration is valid')).toBeVisible()
    await page.getByLabel('Data version').click()
    await page.getByRole('option', { name: 'aligned-v1' }).click()
    await expect(page.getByText('Configuration is valid')).toBeVisible()
    await page.getByRole('button', { name: 'Start run' }).click()
    await expect(page).toHaveURL(/\/runs\/run-e2e-0001$/)
    const config = state.submitted[0]?.['config'] as { dataset: { id: string; preprocessing_version?: string } }
    expect(config.dataset).toEqual({ id: 'capture', preprocessing_version: 'aligned-v1' })
  })
})

test.describe('component gallery', () => {
  test('renders real-size charts quickly and matches the visual baseline', async ({ page }, testInfo) => {
    await installFakeApi(page)
    await page.goto('/gallery')
    const probe = page.getByTestId('perf-probe')
    await expect(probe).toBeVisible({ timeout: 20_000 })
    const spectrumMs = Number(await probe.getAttribute('data-spectrum-ms'))
    const iqMs = Number(await probe.getAttribute('data-iq-ms'))
    test.info().annotations.push({ type: 'perf', description: `spectrum ${spectrumMs} ms, iq ${iqMs} ms` })
    expect(spectrumMs + iqMs).toBeLessThan(2000)
    await expect(page.getByTestId('spectrum-plot').locator('svg.main-svg').first()).toBeVisible()
    await page.getByRole('button', { name: /Enlarge chart: Power spectral density/ }).click()
    await expect(page.getByRole('dialog')).toBeVisible()
    await page.keyboard.press('Escape')
    await expect(page.getByRole('dialog')).toBeHidden()
    await page.getByRole('region', { name: 'StatusChip' }).scrollIntoViewIfNeeded()
    if (testInfo.project.name.startsWith('chromium')) {
      await expect(page.getByRole('region', { name: 'StatusChip' })).toHaveScreenshot('status-chips.png')
      await expect(page.getByRole('region', { name: 'MetricCard' })).toHaveScreenshot('metric-cards.png')
    }
  })
})
