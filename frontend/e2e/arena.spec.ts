import { expect } from '@playwright/test'
import { test } from './arena-browser'
import { AxeBuilder } from '@axe-core/playwright'
import { installFakeApi } from './mock-api'

// Browser-layout fixtures only. These values never enter shipped baseline data.
test('Arena navigation, evidence scope and constrained submission work on desktop and mobile', async ({ page }) => {
  await installFakeApi(page, { language: 'en' })
  const boards = [
  { board_id: 'apa-200mhz-b', title: 'APA B 200 MHz', description: 'Measured APA capture B.', evidence_type: 'measured_data_simulation', evidence_label: 'Measured-data simulation', dataset: 'APA_200MHz_b', conditions: ['apa-200mhz-b'] },
  ]
  const sha = 'a'.repeat(64)
  const cases = [{ budget: 500, condition_id: 'test-condition', seed: 0, judges: [{ judge_id: 'test-oracle', evm_db: -41, baseline_evm_db: -22, nmse_db: -35, baseline_nmse_db: -20, aer_l_db: -56, aer_r_db: -57, baseline_aer_l_db: -30, baseline_aer_r_db: -31, aclr_l_db: -46, aclr_r_db: -47, baseline_aclr_l_db: -25, baseline_aclr_r_db: -26, reference_aclr_l_db: -49, reference_aclr_r_db: -50, power_error_db: .12 }] }]
  const formula = 'FoM = Q − 5 log10(P/1000) − 5 log10(OPs/2000)'
  const rankings = [{ ranking_id: 'overall', title: 'Overall FoM', description: 'Sweep mean.', unit: 'dB' }, { ranking_id: 'arithmetic_efficiency', title: 'Arithmetic efficiency', description: 'Per operation.', unit: 'dB' }, { ranking_id: 'budget-500', title: '≤ 500 parameters', description: 'One budget.', unit: 'dB' }]
  const protocol = { protocol_id: 'arena-browser-fixture', protocol_sha256: sha, training_sha256: 'e'.repeat(64), description: 'Browser test fixture. These are not benchmark results.', boards, seeds: [0, 1, 2], budgets: [250, 500, 1000, 2000], rankings, scoring: {}, cost_model: { nonlinear_cost: { tanh: { mul: 1, add: 1 }, atan2: { mul: 3, add: 4 } } }, score_formula: formula, rules: [{ title: 'Fixed protocol', description: 'The server controls data, seeds, budget and scoring.' }], training: { epochs: 100, device: 'cpu' } }
  const point = (budget: number, quality: number | null, qualified = true) => quality == null ? { budget, available: false, qualified: false, completed_cases: 0, expected_cases: 0 } : { budget, available: true, qualified, completed_cases: 3, expected_cases: 3, model_parameters: { hidden_size: 10 }, parameters: 442, mul: 446, add: 454, ops: 900, nonlinear: { sigmoid: 20, tanh: 10 }, nonlinear_mul: 30, nonlinear_add: 30, operation_items: [{ component: 'GRU cell', mul: 446, add: 454, nonlinear: { sigmoid: 20, tanh: 10 } }], quality_db: quality + .25, quality_conservative_db: quality, parameter_efficiency_db: qualified ? quality + .5 : null, arithmetic_efficiency_db: qualified ? quality + .4 : null, score: qualified ? quality + .45 : null, metrics: { nmse_db: -35, aclr_db: -46, aer_db: -56, nmse_improvement_db: 15, aer_improvement_db: 26 }, reasons: qualified ? [] : ['Output power falls outside the fixed-target ±0.5 dB envelope'] }
  await page.route('**/api/v1/arena**', async route => {
    const path = new URL(route.request().url()).pathname
    let body: unknown = { protocol, backbones: [{ key: 'gru', display_name: 'GRU', family: 'recurrent', deterministic: false }], submissions_available: true, scope_note: 'Software simulation only.' }
    if (path.endsWith('/submissions')) body = []
    if (path.includes('/boards/')) {
      const board = boards.find(item => path.endsWith(item.board_id))!
      body = { board, coverage: { expected: 1, evaluated: 1, succeeded: 1, failed: 0, missing: [] }, rows: [{ entry_id: 'test-gru', board_id: board.board_id, backbone: 'gru', display_name: 'TEST GRU — browser fixture', origin: 'official', status: 'succeeded', protocol_sha256: sha, rank: 1, score: 12.25, eligible: true, parameters: 442, ops_per_parameter: 2.04, seeds: [0, 1, 2], completed_cases: 9, expected_cases: 9, available_budgets: 3, qualified_budgets: 2, best_budget: 500, evidence_type: board.evidence_type, metrics: { nmse_db: -35, aclr_db: -46, aer_db: -56, baseline_aer_db: -30, evm_db: -41, evm_pct: 0.89, baseline_evm_db: -22, baseline_evm_pct: 7.94 }, quality_conservative_db: 14.75, execution_semantics: 'offline_overlap_200_100', rankings: { overall: { rank: 1, score: 12.25 }, arithmetic_efficiency: { rank: 1, score: 11.5 }, 'budget-500': { rank: 1, score: 15.2 } }, budgets: [point(250, 2.4, false), point(500, 14.75), point(1000, 9.5), point(2000, null)], cases }], protocol_sha256: sha, scope_note: 'Browser test fixture only.' }
    }
    await route.fulfill({ json: body })
  })
  await page.route('**/api/v1/backbones/**', async route => route.fulfill({ json: route.request().url().endsWith('/uploads') ? [] : { entries: [] } }))
  for (const width of [1366, 390]) {
    await page.setViewportSize({ width, height: 900 })
    await page.goto('/arena')
    await expect(page.getByRole('table', { name: 'Configuration rankings' })).toBeVisible()
    await expect(page.getByText('Quality and complexity')).toBeVisible()
    await page.getByText('Backbone summaries and evidence').click()
    await expect(page.getByText('TEST GRU — browser fixture', { exact: true })).toBeVisible()
    await expect(page.getByRole('columnheader', { name: 'OPs / parameter ↓' })).toBeVisible()
    await expect(page.getByText('Valid budgets: 2 / 4')).toBeVisible()
    await expect(page.getByRole('table', { name: 'Bundled baseline', exact: true }).getByRole('cell', { name: /0\.89 %/ })).toBeVisible()
    for (const metric of ['evm', 'aclr']) for (const cost of ['parameters', 'ops']) {
      const plot = page.getByTestId(`arena-${metric}-${cost}`)
      await expect(plot).toBeVisible()
      await expect(plot.locator('.js-plotly-plot')).toHaveCount(1)
    }
    await expect(page.getByRole('cell', { name: '-46.0 dBc' })).toBeVisible()
    await page.getByRole('combobox', { name: 'Ranking' }).click()
    await page.getByRole('option', { name: 'Arithmetic efficiency' }).click()
    await expect(page.getByRole('cell', { name: '11.50' })).toBeVisible()
    await expect(page.getByRole('group', { name: 'Benchmark boards' })).toHaveCount(0)
    await page.getByRole('button', { name: 'Details: TEST GRU — browser fixture' }).click()
    await expect(page.getByRole('table', { name: 'Parameter sweep' })).toContainText('MUL / sample')
    await page.getByText('Operation breakdown · ≤ 500 parameters').click()
    await expect(page.getByRole('region', { name: 'Operation breakdown · ≤ 500 parameters' })).toContainText('GRU cell')
    await page.getByText('Per-configuration, condition and seed metrics (1)').click()
    await expect(page.getByRole('table', { name: 'Per-configuration, condition and seed metrics' })).toContainText('EVM (dB)')
    await expect(page.getByRole('table', { name: 'Output and reference ACLR' })).toContainText('Ideal-reference ACLR')
    const layout = await page.evaluate(() => ({ width: window.innerWidth, scroll: document.documentElement.scrollWidth,
      wide: Array.from(document.querySelectorAll('main, main div, main section, main details')).map(el => ({ tag: el.tagName, cls: el.className, width: el.getBoundingClientRect().width, right: el.getBoundingClientRect().right })).filter(el => el.width > window.innerWidth).slice(0, 12) }))
    if (layout.scroll > layout.width) {
      console.error(JSON.stringify(layout))
      await page.screenshot({ path: `/tmp/arena-overflow-${width}.png`, fullPage: true })
    }
    expect(layout.scroll <= layout.width).toBe(true)
    await page.evaluate(() => window.scrollTo(0, 0))
    await page.screenshot({ path: `/tmp/arena-rank-${width}.png`, fullPage: true })
    const nav = page.getByRole('navigation', { name: 'Main navigation' })
    await nav.getByRole('link', { name: 'Submit', exact: true }).click()
    await expect(page.getByRole('button', { name: 'Run benchmark' })).toBeDisabled()
    await page.getByRole('checkbox', { name: 'I accept this protocol, its parameter sweep and its fixed evaluation budget.' }).check()
    await expect(page.getByRole('button', { name: 'Run benchmark' })).toBeEnabled()
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
    await page.evaluate(() => window.scrollTo(0, 0))
    await page.mouse.move(0, 0)
    await page.evaluate(async () => {
      await Promise.allSettled(document.getAnimations().filter(animation => animation.effect?.getComputedTiming().iterations !== Infinity).map(animation => animation.finished))
    })
    await page.screenshot({ path: `/tmp/arena-submit-${width}.png`, fullPage: true })
    const scan = await new AxeBuilder({ page }).include('#main').analyze()
    expect(scan.violations.filter(item => ['critical', 'serious'].includes(item.impact ?? ''))).toEqual([])
    await nav.getByRole('link', { name: 'Rules', exact: true }).click()
    await expect(page.getByText(formula)).toBeVisible()
    await expect(page.getByRole('table', { name: 'Nonlinear function price list' })).toContainText('atan2')
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true)
  }
})
