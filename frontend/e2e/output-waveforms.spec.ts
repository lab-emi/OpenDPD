import { expect, test } from '@playwright/test'
import { readFileSync } from 'node:fs'
import { installFakeApi } from './mock-api'

const run = JSON.parse(readFileSync(new URL('../mocks/run_running.json', import.meta.url), 'utf8')).data as Record<string, unknown>

const traces = [
  { name: 'target input x', role: 'input' },
  { name: 'u = DPD(x)', role: 'predistorted' },
  { name: 'linear target gain*x', role: 'reference' },
  { name: 'with DPD: PA_surrogate(u)', role: 'primary' },
  { name: 'surrogate without DPD', role: 'baseline' },
  { name: 'measured PA without DPD', role: 'baseline', source: 'synthetic dataset' },
].map((s, index) => ({ ...s, i: Array.from({ length: 512 }, (_, k) => Math.sin(k / 12) + index / 10), q: Array.from({ length: 512 }, (_, k) => Math.cos(k / 12) + index / 10) }))

test('training compares two curves per component, including on a narrow screen', async ({ page }) => {
  const state = await installFakeApi(page)
  state.runs.push({ ...run, task: 'train_dpd', run_id: 'run-waveforms', status: 'succeeded' })
  await page.route('**/api/v1/runs/run-waveforms/live', route => route.fulfill({ json: {
    geometry: null,
    preview: { source: 'final_test', samples: 512, metrics: {}, units: {}, updated_at: '2026-09-18T11:00:00Z',
      plots: { time: { start: 50, traces } } },
  } }))
  await page.goto('/runs/run-waveforms')
  const section = page.getByTestId('output-waveforms')
  for (const width of [1366, 390]) {
    await page.setViewportSize({ width, height: 844 })
    for (const component of ['i', 'q'] as const) {
      const chart = section.getByTestId(`output-waveform-${component}`)
      await chart.scrollIntoViewIfNeeded()
      const plot = chart.locator('.js-plotly-plot')
      await expect(plot.locator('.legendtext')).toHaveText(['Linear target g·x', 'With DPD · PA model'])
      await expect.poll(() => plot.evaluate(el => Math.abs(el.querySelector('svg.main-svg')!.getBoundingClientRect().width - el.getBoundingClientRect().width))).toBeLessThanOrEqual(1)
      const actual = await plot.evaluate(el => {
        const p = el as HTMLElement & { data: Array<{ y: number[] }>; layout: { xaxis: { range: number[] } } }
        const rect = el.getBoundingClientRect()
        const legend = el.querySelector('.legend')!.getBoundingClientRect()
        const axis = el.querySelector('.nsewdrag')!.getBoundingClientRect()
        const toolbar = el.querySelector('.modebar')!.getBoundingClientRect()
        const yTitle = el.querySelector('.ytitle')!.getBoundingClientRect()
        return { y: p.data.map(t => Array.from(t.y)), x: p.layout.xaxis.range,
          axisTitleClear: yTitle.left >= rect.left && yTitle.right < axis.left,
          legendClear: legend.bottom <= axis.top + 1 && toolbar.bottom <= legend.top + 1,
          labelsFit: Array.from(el.querySelectorAll('.legendtext')).every(t => { const r = t.getBoundingClientRect(); return parseFloat(getComputedStyle(t).fontSize) >= 14 && r.left >= rect.left && r.right <= rect.right + 1 }) }
      })
      expect(actual.y).toEqual([traces[2]![component], traces[3]![component]])
      expect(actual.x).toEqual([50, 177])
      expect(actual.legendClear).toBe(true)
      expect(actual.axisTitleClear).toBe(true)
      expect(actual.labelsFit).toBe(true)
    }
    expect(await page.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)).toBeLessThanOrEqual(0)
  }
  await section.getByLabel('Waveform', { exact: true }).selectOption('signal:measured PA without DPD')
  const plot = section.getByTestId('output-waveform-i').locator('.js-plotly-plot')
  await plot.scrollIntoViewIfNeeded()
  await expect(plot.locator('.legendtext')).toHaveText(['Without DPD · synthetic data'])
  expect(await plot.evaluate(el => el.querySelector('.legendtext')!.getBoundingClientRect().right <= el.getBoundingClientRect().right + 1)).toBe(true)
  await section.getByLabel('Initial window').selectOption('full')
  await expect.poll(() => plot.evaluate(el => (el as HTMLElement & { layout: { xaxis: { range: number[] } } }).layout.xaxis.range)).toEqual([50, 561])
  await section.getByLabel('Waveform', { exact: true }).selectOption('comparison')
  await expect(plot.locator('.legendtext')).toHaveCount(2)
})
