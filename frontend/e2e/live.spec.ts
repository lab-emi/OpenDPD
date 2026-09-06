import { writeFileSync } from 'node:fs'
import { expect, test, type Page } from '@playwright/test'

/**
 * Real-server journey and timing probe (S13). Skipped unless OPENDPD_LIVE_URL is set to the
 * bootstrap URL printed by `opendpd gui` (it carries the one-time token). The server must
 * already serve the built frontend; nothing is mocked here.
 *
 * Optional env:
 *   OPENDPD_PERF_OUT      path of a JSON file to write the timings into (chromium only)
 *   OPENDPD_LIVE_BIG_LOG  id of a run whose worker log has tens of thousands of lines
 *   OPENDPD_LIVE_SAMPLES  interaction samples per metric (default 100)
 */
const LIVE = process.env['OPENDPD_LIVE_URL']
const PERF_OUT = process.env['OPENDPD_PERF_OUT']
const BIG_LOG = process.env['OPENDPD_LIVE_BIG_LOG']
const SAMPLES = Number(process.env['OPENDPD_LIVE_SAMPLES'] ?? 100)

test.skip(!LIVE, 'OPENDPD_LIVE_URL is not set')
test.describe.configure({ mode: 'serial' })
test.setTimeout(10 * 60_000)

const origin = LIVE ? new URL(LIVE).origin : ''
const stats = (xs: number[]) => {
  const s = [...xs].sort((a, b) => a - b)
  const at = (q: number) => s[Math.min(s.length - 1, Math.floor(q * (s.length - 1)))] ?? NaN
  return { n: s.length, p50: at(0.5), p95: at(0.95), max: s[s.length - 1] ?? NaN }
}

function watchConsole(page: Page): string[] {
  const problems: string[] = []
  page.on('console', (msg) => {
    if (msg.type() === 'error') problems.push(msg.text())
  })
  page.on('pageerror', (err) => problems.push(`pageerror: ${err.message}`))
  return problems
}

let runId = ''
const perf: Record<string, unknown> = {}

test('bootstrap, train the smoke recipe, read the result and export a share package', async ({ page, browserName }) => {
  const problems = watchConsole(page)
  await page.goto(LIVE!)
  await expect(page).toHaveURL(`${origin}/`)
  await expect(page.getByRole('heading', { level: 1, name: 'Home' })).toBeVisible()
  const register = page.getByRole('button', { name: 'Register example dataset' })
  if (await register.isVisible()) {
    await register.click()
    await expect(page.getByText('Example dataset registered')).toBeVisible()
  }
  await page.getByRole('link', { name: 'New experiment' }).click()
  await expect(page.getByText('Configuration is valid')).toBeVisible({ timeout: 20_000 })
  await page.getByLabel('Name (optional)').fill(`live ${browserName}`)
  await page.getByRole('button', { name: 'Start run' }).click()
  await expect(page).toHaveURL(/\/runs\/run-/)
  runId = decodeURIComponent(page.url().split('/runs/')[1] ?? '')
  await expect(page.getByText('Succeeded').first()).toBeVisible({ timeout: 300_000 })
  await page.getByRole('tab', { name: 'Logs' }).click()
  await expect(page.getByRole('tabpanel')).toContainText('Training Completed...', { timeout: 30_000 })
  await page.getByRole('tab', { name: 'Overview' }).click()
  await page.getByRole('link', { name: 'Open result' }).click()
  await expect(page.getByTestId('spectrum-plot').locator('svg.main-svg').first()).toBeVisible({ timeout: 30_000 })
  const panel = page.getByRole('region', { name: 'Export and report' })
  await panel.getByRole('button', { name: 'Export share package' }).click()
  const ready = page.getByTestId('export-ready')
  await expect(ready).toContainText('Package ready', { timeout: 60_000 })
  const href = await ready.getByRole('link', { name: 'Download package' }).getAttribute('href')
  const download = await page.request.get(origin + href!)
  expect(download.status()).toBe(200)
  expect(download.headers()['content-type']).toContain('zip')
  expect(download.headers()['content-security-policy']).toContain("script-src 'self'")
  expect(problems, 'console errors (CSP violations show up here)').toEqual([])
})

test('timings: page load, tab switching, chart re-render, DOM size, heap growth', async ({ page, browserName }) => {
  test.skip(browserName !== 'chromium', 'timings are recorded from Chromium only')
  await page.goto(LIVE!)
  await expect(page.getByRole('heading', { level: 1, name: 'Home' })).toBeVisible()

  const loads: number[] = []
  for (let i = 0; i < 20; i++) {
    const t0 = performance.now()
    await page.goto(`${origin}/experiments`)
    await expect(page.getByRole('table', { name: 'Experiments' })).toBeVisible()
    loads.push(performance.now() - t0)
  }
  perf['page_load_to_table_ms'] = stats(loads)
  perf['dom_nodes_experiments'] = await page.evaluate(() => document.getElementsByTagName('*').length)
  perf['dom_rows_experiments'] = await page.locator('table[aria-label="Experiments"] tbody tr').count()

  await page.goto(`${origin}/runs/${encodeURIComponent(runId)}`)
  await expect(page.getByRole('tab', { name: 'Logs' })).toBeVisible()
  const tabs: number[] = []
  for (let i = 0; i < SAMPLES; i++) {
    const name = i % 2 === 0 ? 'Logs' : 'Overview'
    const t0 = performance.now()
    await page.getByRole('tab', { name }).click()
    await expect(page.getByRole('tab', { name })).toHaveAttribute('aria-selected', 'true')
    await expect(page.getByRole('tabpanel')).toBeVisible()
    tabs.push(performance.now() - t0)
  }
  perf['tab_switch_ms'] = stats(tabs)

  await page.goto(`${origin}/results/${encodeURIComponent(runId)}`)
  const plot = page.getByTestId('spectrum-plot')
  await expect(plot.locator('svg.main-svg').first()).toBeVisible({ timeout: 30_000 })
  const charts: number[] = []
  for (let i = 0; i < SAMPLES; i++) {
    const t0 = performance.now()
    await page.getByRole('button', { name: /Enlarge chart: Power spectral density/ }).click()
    await expect(page.getByRole('dialog').locator('svg.main-svg').first()).toBeVisible()
    charts.push(performance.now() - t0)
    await page.keyboard.press('Escape')
    await expect(page.getByRole('dialog')).toBeHidden()
  }
  perf['chart_enlarge_render_ms'] = stats(charts)

  // heap after repeated page switching: sampled through CDP, warm-up excluded
  const cdp = await page.context().newCDPSession(page)
  await cdp.send('Performance.enable')
  const heap = async () => {
    const { metrics } = await cdp.send('Performance.getMetrics')
    return (metrics.find((m) => m.name === 'JSHeapUsedSize')?.value ?? 0) / 1e6
  }
  const samples: number[] = []
  const routes = [`/experiments`, `/runs/${encodeURIComponent(runId)}`, `/results/${encodeURIComponent(runId)}`, `/datasets`]
  for (let i = 0; i < 120; i++) {
    await page.goto(origin + routes[i % routes.length]!)
    await expect(page.getByRole('main')).toBeVisible()
    if (i % 10 === 9) samples.push(await heap())
  }
  perf['heap_mb_samples'] = samples
  // the heap oscillates with garbage collection: compare the floors of the two halves, not two instants
  const half = Math.floor(samples.length / 2)
  const floor = (xs: number[]) => Math.min(...xs)
  perf['heap_floor_growth_mb'] = floor(samples.slice(half)) - floor(samples.slice(0, half))
  perf['heap_max_mb'] = Math.max(...samples)
})

test('a very long worker log stays windowed and searchable', async ({ page }) => {
  test.skip(!BIG_LOG, 'OPENDPD_LIVE_BIG_LOG is not set')
  await page.goto(LIVE!)
  await page.goto(`${origin}/runs/${encodeURIComponent(BIG_LOG!)}?tab=logs`)
  const viewer = page.getByTestId('log-viewer')
  await expect(viewer).toBeVisible()
  const caption = viewer.getByText(/of \d+ loaded lines/)
  await expect(caption).toBeVisible()
  const t0 = performance.now()
  const loadAll = viewer.getByRole('button', { name: 'Load all remaining lines' })
  if (await loadAll.isVisible()) {
    await loadAll.click()
    await expect(loadAll).toBeHidden({ timeout: 120_000 })
  }
  perf['log_load_all_ms'] = performance.now() - t0
  const total = Number(/of (\d+) loaded lines/.exec((await caption.textContent()) ?? '')?.[1] ?? 0)
  perf['log_lines_loaded'] = total
  expect(total).toBeGreaterThan(1000)
  perf['log_rows_in_dom'] = await viewer.locator('[role="log"] > div > div').count()
  expect(perf['log_rows_in_dom'] as number).toBeLessThan(120)
  const needle = `line ${total - 10}:`
  await viewer.getByLabel('Filter lines').fill(needle)
  await expect(viewer.getByText(needle, { exact: false }).first()).toBeVisible()
  await expect(viewer.getByText(/^1 of \d+ loaded lines/)).toBeVisible()
})

test.afterAll(() => {
  if (PERF_OUT && Object.keys(perf).length > 0) writeFileSync(PERF_OUT, JSON.stringify({ run_id: runId, ...perf }, null, 2))
})
