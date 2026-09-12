import { AxeBuilder } from '@axe-core/playwright'
import { expect, test, type Page } from '@playwright/test'
import { installFakeApi } from './mock-api'

/**
 * S13 accessibility and offline checks against the mocked API.
 *  - axe-core (WCAG 2.1 A/AA rules) on every main page: no serious or critical violation;
 *  - a keyboard-only journey from the home page to a started run;
 *  - the main journey needs no host other than the loopback one (no fonts, CDN or telemetry).
 */

const PAGES = ['/', '/datasets', '/experiments', '/experiments/new', '/results', '/results/run-pa-0001', '/settings', '/gallery']

async function scan(page: Page, path: string) {
  // Audit the settled page: wait for actual MUI transitions instead of
  // sampling an intermediate dialog opacity after a wall-clock delay.
  await page.waitForLoadState('networkidle')
  await page.evaluate(async () => {
    const animations = document.getAnimations().filter(animation => animation.effect?.getComputedTiming().iterations !== Infinity)
    await Promise.allSettled(animations.map(animation => animation.finished))
  })
  const results = await new AxeBuilder({ page }).withTags(['wcag2a', 'wcag2aa', 'wcag21a', 'wcag21aa']).analyze()
  const blocking = results.violations.filter((v) => v.impact === 'serious' || v.impact === 'critical')
  const summary = results.violations.map((v) => `${v.impact}: ${v.id} (${v.nodes.length} nodes) — ${v.help}`)
  test.info().annotations.push({ type: `axe ${path}`, description: summary.length ? summary.join('; ') : 'no violations' })
  expect(blocking.map((v) => ({
    id: v.id,
    nodes: v.nodes.map(n => ({ target: n.target, summary: n.failureSummary, checks: n.any.map(check => check.data) })),
  })), `${path} has blocking accessibility violations`).toEqual([])
}

test.describe('accessibility (axe-core)', () => {
  test('main pages have no serious or critical WCAG 2.1 AA violations', async ({ page }) => {
    await installFakeApi(page)
    for (const path of PAGES) {
      await page.goto(path)
      await expect(page.getByRole('main')).toBeVisible()
      if (path === '/gallery') await expect(page.getByTestId('spectrum-plot').locator('svg.main-svg').first()).toBeVisible({ timeout: 20_000 })
      await scan(page, path)
    }
  })

  test('run detail tabs are accessible while a run is live', async ({ page }) => {
    await installFakeApi(page)
    await page.goto('/')
    await page.getByRole('button', { name: 'Register example dataset' }).click()
    await page.goto('/experiments/new')
    await expect(page.getByText('Configuration is valid')).toBeVisible()
    await page.getByRole('button', { name: 'Start run' }).click()
    await expect(page).toHaveURL(/\/runs\/run-e2e-0001$/)
    await scan(page, '/runs/run-e2e-0001')
    for (const tab of ['Logs', 'Artifacts', 'Configuration']) {
      await page.getByRole('tab', { name: tab }).click()
      await scan(page, `/runs/run-e2e-0001#${tab}`)
    }
  })
})

/** Press Tab until the focused element has the given accessible name (bounded so a trap fails the test). */
async function tabTo(page: Page, name: RegExp, maxTabs = 80): Promise<void> {
  for (let i = 0; i < maxTabs; i++) {
    await page.keyboard.press('Tab')
    const label = await page.evaluate(() => {
      const el = document.activeElement as HTMLElement | null
      if (!el) return ''
      return el.getAttribute('aria-label') ?? el.textContent?.trim() ?? ''
    })
    if (name.test(label)) return
  }
  throw new Error(`no focusable element named ${name} within ${maxTabs} tabs`)
}

test.describe('keyboard-only journey', () => {
  test('register the example, open New experiment and start a run without a pointer', async ({ page }) => {
    const state = await installFakeApi(page)
    await page.goto('/')
    // the button exists once the (empty) dataset list has loaded; tabbing before that walks past 80 elements
    await expect(page.getByRole('button', { name: 'Register example dataset' })).toBeVisible()
    await tabTo(page, /^Register example dataset$/)
    await page.keyboard.press('Enter')
    await expect(page.getByText('Example dataset registered')).toBeVisible()
    await tabTo(page, /^New experiment$/)
    await page.keyboard.press('Enter')
    await expect(page.getByRole('heading', { level: 1, name: 'New experiment' })).toBeVisible()
    await expect(page.getByText('Configuration is valid')).toBeVisible()
    await tabTo(page, /^Start run$/)
    await page.keyboard.press('Enter')
    await expect(page).toHaveURL(/\/runs\/run-e2e-0001$/)
    expect(state.submitted).toHaveLength(1)
    // tabs follow the ARIA tabs pattern: Tab reaches the selected tab, arrows move between tabs
    await tabTo(page, /^Overview$/)
    await page.keyboard.press('ArrowRight')
    await page.keyboard.press('Enter')
    await expect(page.getByRole('tab', { name: 'Logs' })).toHaveAttribute('aria-selected', 'true')
    await expect(page.getByRole('tabpanel')).toContainText('Training Completed...')
  })
})

test.describe('offline', () => {
  test('the main journey contacts no host but the loopback one', async ({ page, context }) => {
    const external: string[] = []
    await context.route('**/*', (route) => {
      const url = new URL(route.request().url())
      if (url.hostname === '127.0.0.1' || url.hostname === 'localhost') return route.continue()
      external.push(url.href)
      return route.abort()
    })
    await installFakeApi(page)
    await page.goto('/')
    await page.getByRole('button', { name: 'Register example dataset' }).click()
    await page.goto('/experiments/new')
    await page.getByRole('button', { name: 'Start run' }).click()
    await expect(page).toHaveURL(/\/runs\/run-e2e-0001$/)
    await page.goto('/results/run-pa-0001')
    await expect(page.getByRole('region', { name: 'Export and report' })).toBeVisible()
    await page.goto('/gallery')     // charts: the Plotly bundle and every asset come from this origin
    await expect(page.getByTestId('spectrum-plot').locator('svg.main-svg').first()).toBeVisible({ timeout: 20_000 })
    expect(external).toEqual([])
  })
})
