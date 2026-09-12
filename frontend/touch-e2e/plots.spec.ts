import { expect, test } from '@playwright/test'
import { installFakeApi } from '../e2e/mock-api'
import { readFileSync } from 'node:fs'

const dataset = JSON.parse(readFileSync(new URL('../mocks/dataset_builtin.json', import.meta.url), 'utf8')) as { data: Record<string, unknown> }

test('English and CUDA defaults in a non-English mobile browser', async ({ page }) => {
  const state = await installFakeApi(page)
  state.datasets.push(dataset.data)
  await page.route('**/api/v1/system/capabilities', (route) => route.fulfill({ json: {
    version: 'x', workspace: 'test', devices: [{ device: 'cpu', detected: true }, { device: 'cuda', detected: true, name: 'GPU' }],
  } }))
  await page.goto('/experiments/new')
  await expect(page.getByRole('heading', { name: 'PA Model Training', exact: true })).toBeVisible()
  await expect(page.locator('html')).toHaveAttribute('lang', 'en')
  await expect(page.getByText('Configuration is valid', { exact: true })).toBeVisible()
  await page.getByRole('button', { name: 'Continue', exact: true }).click()
  await expect(page.getByRole('combobox', { name: 'Device', exact: true })).toContainText('cuda')
})

test('two-finger zoom follows its midpoint; touch controls and full-screen fit a phone', async ({ page, context, browserName }) => {
  await installFakeApi(page)
  await page.goto('/gallery')
  const chart = page.getByTestId('spectrum-plot').first()
  const plot = chart.locator('.js-plotly-plot')
  const area = plot.locator('.nsewdrag')
  await expect(area).toBeVisible()
  await area.scrollIntoViewIfNeeded()
  const range = () => plot.evaluate((e) => {
    const p = e as HTMLElement & { _fullLayout: { xaxis: { range: number[] }; yaxis: { range: number[] } } }
    return { x: p['_fullLayout'].xaxis.range, y: p['_fullLayout'].yaxis.range }
  })
  const before = await range(), rect = (await area.boundingBox())!
  const cx = rect.x + rect.width / 2, cy = rect.y + rect.height / 2
  const cdp = browserName === 'chromium' ? await context.newCDPSession(page) : null
  const touch = async (type: 'touchStart' | 'touchMove' | 'touchEnd', points: number[][]) => {
    if (cdp) await cdp.send('Input.dispatchTouchEvent', { type, touchPoints: points.map(([x, y], id) => ({ x, y, id })) })
    else await area.evaluate((el, input) => {
      // WebKit's automation API has no multi-touch injection. Exercise its DOM
      // touch path; Chromium above additionally checks native browser arbitration.
      const touches = input.points.map(([clientX, clientY], identifier) => ({ identifier, target: el, clientX, clientY }))
      const event = new Event(input.type.toLowerCase(), { bubbles: true, cancelable: true })
      Object.assign(event, { touches, targetTouches: touches, changedTouches: touches })
      el.dispatchEvent(event)
    }, { type, points })
  }
  await touch('touchStart', [[cx - 30, cy], [cx + 30, cy]])
  for (let i = 1; i <= 10; i++) await touch('touchMove', [[cx - 30 - 3 * i + i, cy + i], [cx + 30 + 3 * i + i, cy + i]])
  await touch('touchEnd', [])
  await expect.poll(async () => { const r = await range(); return (r.x[1]! - r.x[0]!) / (before.x[1]! - before.x[0]!) }).toBeCloseTo(.5, 2)
  const after = await range()
  const midpointValue = after.x[0]! + (after.x[1]! - after.x[0]!) * (.5 + 10 / rect.width)
  expect(midpointValue).toBeCloseTo((before.x[0]! + before.x[1]!) / 2, 3)
  expect(await page.evaluate(() => visualViewport?.scale)).toBe(1)
  const controls = chart.getByTestId('touch-plot-controls')
  const reset = controls.getByRole('button', { name: 'Reset axes', exact: true })
  await expect(reset).toBeVisible()
  expect((await reset.boundingBox())!.width).toBeGreaterThanOrEqual(44)
  await reset.click()
  await expect.poll(range).toEqual(before)
  await controls.getByRole('button', { name: 'Zoom in', exact: true }).click()
  await expect.poll(async () => { const r = await range(); return (r.x[1]! - r.x[0]!) / (before.x[1]! - before.x[0]!) }).toBeCloseTo(1 / 1.2, 3)
  await chart.getByRole('button', { name: /Enlarge chart/ }).click()
  const dialog = page.getByRole('dialog', { name: 'Power spectral density' })
  await expect(dialog).toBeVisible()
  expect((await dialog.boundingBox())!.width).toBe(page.viewportSize()!.width)
  await expect(dialog.getByTestId('touch-plot-controls').getByRole('button', { name: 'Reset axes', exact: true })).toBeVisible()
  expect((await dialog.boundingBox())!.x).toBeCloseTo(0)
  expect((await dialog.boundingBox())!.y).toBeCloseTo(0)
  const legend = (await dialog.locator('.legend').boundingBox())!
  const modebar = (await dialog.locator('.modebar').boundingBox())!
  expect(legend.y).toBeGreaterThanOrEqual(modebar.y + modebar.height)
  await dialog.getByRole('button', { name: 'Close enlarged chart' }).click()
  if (cdp) {
    await area.scrollIntoViewIfNeeded()
    const r = (await area.boundingBox())!, y = await page.evaluate(() => scrollY)
    await touch('touchStart', [[r.x + r.width / 2, r.y + r.height * .7]])
    for (let i = 1; i <= 8; i++) await touch('touchMove', [[r.x + r.width / 2, r.y + r.height * .7 - i * 10]])
    await touch('touchEnd', [])
    await expect.poll(() => page.evaluate(() => scrollY)).toBeGreaterThan(y + 20)
    await cdp.detach()
  }
})
