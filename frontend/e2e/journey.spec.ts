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
    await page.getByRole('button', { name: 'Start run' }).click()

    await expect(page).toHaveURL(/\/runs\/run-e2e-0001$/)
    expect(state.submitted).toHaveLength(1)
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

test.describe('component gallery', () => {
  test('renders real-size charts quickly and matches the visual baseline', async ({ page }) => {
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
    await expect(page.getByRole('region', { name: 'StatusChip' })).toHaveScreenshot('status-chips.png')
    await expect(page.getByRole('region', { name: 'MetricCard' })).toHaveScreenshot('metric-cards.png')
  })
})
