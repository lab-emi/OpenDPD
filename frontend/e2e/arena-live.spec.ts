import { writeFileSync } from 'node:fs'
import { expect } from '@playwright/test'
import { test } from './arena-browser'
import type { components } from '../src/api/schema'

type ArenaCatalog = components['schemas']['ArenaCatalog']
type ArenaSubmission = components['schemas']['ArenaSubmission']

// Opt-in real-server test: use an isolated workspace, because this executes
// one server-owned MP benchmark and retains its private submission evidence.
const LIVE = process.env['OPENDPD_ARENA_LIVE_URL']
const OUT = process.env['OPENDPD_ARENA_LIVE_OUT'] ?? '/tmp'
test.skip(!LIVE, 'OPENDPD_ARENA_LIVE_URL is not set')
test.setTimeout(420_000)

test('real protocol, MP submission, evidence and rank remain consistent', async ({ page }) => {
  const origin = new URL(LIVE!).origin
  const entryName = `Arena 2.3 live MP verification ${Date.now()}`
  const problems: string[] = []
  page.on('pageerror', error => problems.push(error.message))
  await page.goto(LIVE!)
  await expect(page.getByRole('heading', { level: 1, name: 'Home' })).toBeVisible()
  const catalogResponse = await page.request.get(`${origin}/api/v1/arena`)
  expect(catalogResponse.status()).toBe(200)
  const catalog = await catalogResponse.json() as ArenaCatalog
  if (process.env['OPENDPD_ARENA_EXPECTED_PROTOCOL']) expect(catalog.protocol.protocol_sha256).toBe(process.env['OPENDPD_ARENA_EXPECTED_PROTOCOL'])
  expect(catalog.protocol.boards).toHaveLength(1)
  expect(catalog.protocol.boards.map(board => board.dataset).sort()).toEqual(['APA_200MHz_b'])
  expect(catalog.protocol.boards.every(board => board.conditions.length === 1 && board.evidence_type === 'measured_data_simulation')).toBe(true)
  expect(catalog.protocol.training.stimuli).toContain('original measured capture inputs only')
  expect(catalog.protocol.training.checkpoint_selection).toContain('in-band error + worse-side ACLR')
  expect(catalog.submissions_available).toBe(true)
  expect(catalog.protocol.training.epochs).toBe(240)
  expect(catalog.protocol.training.batch_size).toBe(64)
  expect(catalog.protocol.training.frames_per_epoch).toBe('all training windows')
  expect(catalog.protocol.budgets).toEqual([250, 500, 1000, 2000])
  expect(catalog.protocol.rankings.map(item => item.ranking_id)).toContain('arithmetic_efficiency')
  const mp = catalog.backbones.find(model => model.key === 'mp_ls')!
  expect(catalog.backbones.some(model => model.key === 'ilc_dpd')).toBe(false)
  expect(mp.deterministic).toBe(true)
  await page.goto(`${origin}/arena/submit?board=apa-200mhz-b`)
  await page.getByRole('combobox', { name: 'Backbone', exact: true }).click()
  await page.getByRole('option', { name: `${mp.display_name} · ${mp.family}`, exact: true }).click()
  await expect(page.getByText('One deterministic fit per condition and parameter budget; no artificial seed repetitions.')).toBeVisible()
  await page.getByRole('textbox', { name: 'Entry name' }).fill(entryName)
  await page.getByRole('checkbox', { name: 'I accept this protocol, its parameter sweep and its fixed evaluation budget.' }).check()
  const responsePromise = page.waitForResponse(response => response.url().endsWith('/api/v1/arena/submissions') && response.request().method() === 'POST')
  await page.getByRole('button', { name: 'Run benchmark', exact: true }).click()
  const response = await responsePromise
  expect(response.status()).toBe(202)
  expect(response.request().postDataJSON()).toEqual({ board_id: 'apa-200mhz-b', backbone: 'mp_ls', display_name: entryName, accepted_protocol_sha256: catalog.protocol.protocol_sha256 })
  const accepted = await response.json() as ArenaSubmission
  let final: ArenaSubmission = accepted
  await expect.poll(async () => {
    const current = await page.request.get(`${origin}/api/v1/arena/submissions/${accepted.submission_id}`)
    expect(current.status()).toBe(200)
    final = await current.json() as ArenaSubmission
    return final.status
  }, { timeout: 390_000, intervals: [2500] }).toBe('succeeded')
  // One deterministic MP fit per budget and condition: four budgets × one measured PA condition.
  expect(final.result?.completed_cases).toBe(4)
  expect(final.result?.expected_cases).toBe(4)
  expect(final.result?.budgets?.map(item => item.parameters)).toEqual([250, 490, 1000, 1500])
  expect(final.result?.budgets?.every(item => item.ops === (item.mul ?? 0) + (item.add ?? 0))).toBe(true)
  expect(Object.keys(final.result?.rankings ?? {})).toEqual(catalog.protocol.rankings.map(item => item.ranking_id))
  expect(final.result?.seeds).toEqual([catalog.protocol.seeds[0]])
  expect(final.result?.protocol_sha256).toBe(catalog.protocol.protocol_sha256)
  expect(final.result?.cases?.every(item => item['evaluation_split'] === 'test')).toBe(true)
  expect(final.result?.provenance?.['data_access']).toEqual({ training_splits: ['train', 'val'], evaluation_split: 'test', all_checkpoints_frozen_before_test: true })
  expect(final.result?.metrics?.aclr_db).toEqual(expect.any(Number))
  expect(final.result?.metrics?.evm_pct).toBeGreaterThan(0)      // demodulated on the symbol grid of the measured-source condition
  expect(final.result?.metrics?.evm_improvement_db).toEqual(expect.any(Number))
  expect(final.result?.quality_conservative_db).toEqual(expect.any(Number))
  writeFileSync(`${OUT}/opendpd-arena-23-live-submission.json`, JSON.stringify(final, null, 2))
  await page.goto(`${origin}/arena?board=apa-200mhz-b`)
  const workspace = page.getByText('Backbone summaries and evidence').last()
  await workspace.click()
  await expect(page.getByText(entryName, { exact: true })).toBeVisible()
  await page.getByRole('button', { name: `Details: ${entryName}` }).click()
  await expect(page.getByRole('table', { name: 'Parameter sweep' })).toBeVisible()
  await page.getByText(/^Per-configuration, condition and seed metrics/).click()
  await expect(page.getByRole('table', { name: 'Per-configuration, condition and seed metrics' })).toBeVisible()
  await page.setViewportSize({ width: 1366, height: 1000 })
  await page.evaluate(() => window.scrollTo(0, 0))
  await page.screenshot({ path: `${OUT}/opendpd-arena-23-live-rank.png`, fullPage: true })
  await page.goto(`${origin}/arena/rules?board=apa-200mhz-b`)
  await expect(page.getByText(/240 full passes over every 200-sample training window/)).toBeVisible()
  await page.screenshot({ path: `${OUT}/opendpd-arena-23-live-rules.png`, fullPage: true })
  if (process.env['OPENDPD_ARENA_FULL_MATRIX'] === '1') {
    for (const board of catalog.protocol.boards) {
      const boardResponse = await page.request.get(`${origin}/api/v1/arena/boards/${board.board_id}`)
      const data = await boardResponse.json() as { coverage: { expected: number; succeeded: number }; rows: { backbone: string; cases: { condition_id: string; evaluation_split: string }[]; origin: string }[] }
      expect(data.coverage.expected).toBe(23)
      expect(data.coverage.succeeded).toBe(23)
      expect(data.rows.filter(row => row.origin === 'official')).toHaveLength(23)
      expect(data.rows.every(row => row.cases.every(item => item.condition_id === board.board_id))).toBe(true)
      expect(data.rows.every(row => row.backbone !== 'ilc_dpd' && row.cases.every(item => item.evaluation_split === 'test'))).toBe(true)
      await page.goto(`${origin}/arena?board=${board.board_id}`)
      await expect(page.getByText(/only original measured inputs/)).toBeVisible()
      for (const metric of ['evm', 'aclr']) for (const cost of ['parameters', 'ops']) {
        const plot = page.getByTestId(`arena-${metric}-${cost}`).first()
        await expect(plot).toBeVisible()
        await expect(plot.locator('.js-plotly-plot')).toHaveCount(1)
        await expect.poll(() => plot.locator('.js-plotly-plot').evaluate(element => {
          const graph = element as HTMLElement & { data?: { name?: string; x?: number[]; y?: number[]; marker?: { symbol?: string } }[] }
          const front = graph.data?.find(trace => trace.name === 'Pareto front')
          return front?.marker?.symbol === 'diamond' && !!front.x?.length &&
            front.x.length === front.y?.length && front.x.every(value => Number.isFinite(value) && value > 0) &&
            front.y.every(Number.isFinite)
        })).toBe(true)
      }
      await page.screenshot({ path: `${OUT}/arena-${board.board_id}.png`, fullPage: true })
    }
  }
  expect(problems).toEqual([])
})
