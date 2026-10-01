import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import { ArenaPage } from './ArenaPage'
import { mockApi, renderWithProviders } from '@/test/utils'
import type { ArenaBoard, ArenaBudget, ArenaCatalog, ArenaLeaderboard, ArenaRow, ArenaSubmission } from '@/api/arena'

// Exercise controls and evidence here. Pareto mathematics has its own tests;
// the real Plotly renderer and responsive layout are covered by arena.spec.ts.
vi.mock('@/components/PlotlyChart', async importOriginal => ({
  ...await importOriginal<typeof import('@/components/PlotlyChart')>(),
  PlotlyChart: ({ title, ...props }: { title: string; 'data-testid'?: string }) => <figure aria-label={title} data-testid={props['data-testid']} />,
}))

const sha = 'a'.repeat(64)
const ilcLabel = 'ILC → MP (training feedback)'
const ilcRegistryName = 'ILC-DPD (ILA) + Ideal waveform benchmark'
const boards: ArenaBoard[] = [
  { board_id: 'apa-200mhz-b', title: 'APA B 200 MHz', description: 'Measured APA capture B.', evidence_type: 'measured_data_simulation', evidence_label: 'Measured-data simulation', dataset: 'APA_200MHz_b', conditions: ['apa-200mhz-b'] },
]
const rankings = [
  { ranking_id: 'overall', title: 'Overall FoM', description: 'Sweep mean.', unit: 'dB' },
  { ranking_id: 'linearization', title: 'Best linearization', description: 'Peak quality.', unit: 'dB' },
  { ranking_id: 'arithmetic_efficiency', title: 'Arithmetic efficiency', description: 'Per operation.', unit: 'dB' },
  { ranking_id: 'budget-500', title: '≤ 500 parameters', description: 'One budget.', unit: 'dB' },
]
const formula = 'FoM = Q − 5 log10(P/1000) − 5 log10(OPs/2000)'
const catalog: ArenaCatalog = {
  protocol: { protocol_id: 'dpd-arena-v6-apa-b', protocol_sha256: sha, training_sha256: 'e'.repeat(64), title: 'Arena', description: 'Fixed benchmark.', boards, seeds: [0, 1, 2], budgets: [250, 500, 1000, 2000], rankings,
    rules: [{ title: 'Fixed seeds', description: 'All required seeds must finish.' }], score_formula: formula, scoring: { reference_parameters: 1000, reference_operations: 2000 },
    cost_model: { unit: 'per output IQ sample', nonlinear_cost: { tanh: { mul: 1, add: 1 }, atan2: { mul: 3, add: 4 }, relu: { mul: 0, add: 0 } } },
    training: { epochs: 240, frames_per_epoch: 'all training windows', batch_size: 64 } },
  backbones: [{ key: 'gru', display_name: 'GRU', family: 'recurrent', deterministic: false }, { key: 'dgru', display_name: 'DeltaGRU', family: 'recurrent', deterministic: false }, { key: 'mp_ls', display_name: 'Memory polynomial', family: 'polynomial', deterministic: true }, { key: 'user_template', display_name: 'Template GRU (bundled example)', family: 'template', deterministic: false }, { key: 'ilc_dpd', display_name: ilcRegistryName, family: 'classical', deterministic: true }],
  submissions_available: true, scope_note: 'Software simulation only.',
}
const judge = { judge_id: 'pa', evm_db: -41, baseline_evm_db: -22, nmse_db: -35, aclr_l_db: -46, aclr_r_db: -47, aer_l_db: -56, aer_r_db: -57, baseline_nmse_db: -20, baseline_aclr_l_db: -25, baseline_aclr_r_db: -26, baseline_aer_l_db: -30, baseline_aer_r_db: -31, reference_aclr_l_db: -49, reference_aclr_r_db: -50, power_error_db: 0.12 }
const budget = (value: number, extra: Partial<ArenaBudget> = {}): ArenaBudget => ({
  budget: value, available: true, qualified: true, completed_cases: 3, expected_cases: 3, model_parameters: { hidden_size: 10, num_layers: 1 }, parameters: 442, mul: 446, add: 454, ops: 900,
  nonlinear: { sigmoid: 20, tanh: 10 }, nonlinear_mul: 30, nonlinear_add: 30, operation_items: [{ component: 'GRU cell', mul: 440, add: 450, nonlinear: { sigmoid: 20, tanh: 10 } }],
  quality_db: 15, quality_conservative_db: 14.75, quality_std_db: 0.25, parameter_efficiency_db: 15.29, arithmetic_efficiency_db: 15.21, score: 15.25,
  metrics: { nmse_db: -35, aclr_db: -46, aer_db: -56, evm_db: -41, evm_pct: 0.89, baseline_evm_db: -22, baseline_evm_pct: 7.94, evm_improvement_db: 19, baseline_nmse_db: -20, baseline_aclr_db: -25, baseline_aer_db: -30, nmse_improvement_db: 15, aclr_improvement_db: 21, aer_improvement_db: 26 }, reasons: [], ...extra,
})
const base: ArenaRow = {
  entry_id: 'official-gru', board_id: 'apa-200mhz-b', backbone: 'gru', display_name: 'GRU reference', origin: 'official', status: 'succeeded', protocol_sha256: sha,
  rank: 1, score: 12.25, eligible: true, parameters: 442, ops_per_parameter: 2.04, seeds: [0, 1, 2], completed_cases: 9, expected_cases: 9, evidence_type: 'measured_data_simulation',
  available_budgets: 3, qualified_budgets: 2, best_budget: 500,
  metrics: { nmse_db: -35, aclr_db: -46, aer_db: -56, evm_db: -41, evm_pct: 0.89, baseline_evm_db: -22, baseline_evm_pct: 7.94, evm_improvement_db: 19, baseline_nmse_db: -20, baseline_aclr_db: -25, baseline_aer_db: -30, nmse_improvement_db: 15, aclr_improvement_db: 21, aer_improvement_db: 26 },
  execution_semantics: 'offline_overlap_200_100', provenance: { source_sha256: 'b'.repeat(64), environment: { torch: '2.10.0', numpy: '2.3.0', device: 'cuda' } },
  quality_db: 15, quality_conservative_db: 14.75,
  rankings: { overall: { rank: 1, score: 12.25 }, linearization: { rank: 2, score: 14.75 }, arithmetic_efficiency: { rank: 1, score: 11.5 }, 'budget-500': { rank: 1, score: 15.25 } },
  budgets: [
    budget(250, { qualified: false, quality_conservative_db: 2.4, score: null, parameter_efficiency_db: null, arithmetic_efficiency_db: null, reasons: ['Output power falls outside the fixed-target ±0.5 dB envelope'], parameters: 247, ops: 504 }),
    budget(500), budget(1000, { quality_conservative_db: 9.5, score: 9.4, parameters: 994, mul: 1000, add: 1016, ops: 2016 }),
    { budget: 2000, available: false, qualified: false, completed_cases: 0, expected_cases: 0 },
  ],
  cases: [{ budget: 500, condition_id: 'apa-200mhz-b', seed: 0, judges: [judge] }],
}

function setup({ available = true, rows = [base], route = '/arena', initialSubmissions = [] }: { available?: boolean; rows?: ArenaRow[]; route?: string; initialSubmissions?: ArenaSubmission[] } = {}) {
  let submissions: ArenaSubmission[] = initialSubmissions
  const boardData = (board: ArenaBoard): ArenaLeaderboard => ({ board, protocol_sha256: sha, rows: rows.map(row => ({ ...row, board_id: board.board_id, evidence_type: board.evidence_type })), coverage: { expected: 2, evaluated: 2, succeeded: 1, failed: 1, missing: [] }, scope_note: 'Software simulation only.' })
  const api = mockApi({
    'GET /api/v1/arena': () => ({ ...catalog, submissions_available: available, submission_unavailable_reason: available ? null : 'Hosted Arena execution is unavailable.' }),
    ...Object.fromEntries(boards.map(board => [`GET /api/v1/arena/boards/${board.board_id}`, () => boardData(board)])),
    'GET /api/v1/arena/submissions': () => submissions,
    'POST /api/v1/arena/submissions': (_url, init) => {
      const saved = { submission_id: 'arena-' + 'c'.repeat(32), request: JSON.parse(String(init.body)), protocol_sha256: sha, status: 'queued' as const, created_at: '2026-09-19T10:00:00Z' }
      submissions = [saved]
      return { status: 202, body: saved }
    },
    'GET /api/v1/backbones/uploads': () => [],
    'GET /api/v1/backbones/catalog': () => ({ entries: [] }),
  })
  renderWithProviders(<ArenaPage />, { route })
  return api
}
const unranked = (row: Partial<ArenaRow>): ArenaRow => ({ ...base, rank: null, score: null, eligible: false, qualified_budgets: 0, rankings: Object.fromEntries(rankings.map(item => [item.ranking_id, { rank: null, score: null }])), ...row })

test('shows only APA_200MHz_b and keeps unranked and failed baselines visible', async () => {
  setup({ rows: [base, unranked({ entry_id: 'unranked', display_name: 'Incomplete coverage', completed_cases: 3, eligibility_reasons: ['Missing required seed.'] }), unranked({ entry_id: 'failed', display_name: 'Failed baseline', status: 'failed', metrics: null, budgets: [], cases: [], error: 'Training timeout.' })] })
  await screen.findByText('GRU reference')
  await userEvent.click(screen.getByText('Backbone summaries and evidence'))
  expect(screen.getByText('GRU reference')).toBeVisible()
  expect(screen.queryByRole('group', { name: 'Benchmark boards' })).not.toBeInTheDocument()
  expect(screen.getByText(/their DPD scores are not hardware measurements/)).toBeVisible()
  expect(screen.getByText('Parameter budgets: 250 · 500 · 1,000 · 2,000')).toBeVisible()
  const ranked = screen.getByText('GRU reference').closest('tr')!
  expect(within(ranked).getByText('12.25')).toBeVisible()
  expect(within(ranked).getAllByText('15.00')).toHaveLength(3) // best quality, and the ≤ 500 budget that reached it
  expect(within(ranked).getByText('@ ≤500')).toBeVisible()
  expect(within(ranked).getByText('(15.00)')).toBeVisible() // trained and judged, but outside the output-power gate
  expect(within(ranked).getByText('0.89 %')).toBeVisible() // what the quality is made of, at the best budget
  expect(within(ranked).getByText('-41.0 dB')).toBeVisible()
  expect(within(ranked).getByText('-46.0 dBc')).toBeVisible()
  expect(within(ranked).getAllByText('15.00')[0]).toBeVisible()
  expect(within(ranked).getByText('2.04')).toBeVisible()
  expect(within(ranked).getByText('Valid budgets: 2 / 4')).toBeVisible()
  for (const name of ['Score (dB) ↑', 'Quality at best FoM (dB) ↑', 'EVM', 'ACLR', '≤ 250', '≤ 2,000', 'OPs / parameter ↓']) expect(screen.getAllByRole('columnheader', { name })[0]).toBeVisible()
  expect(screen.queryByText(/latency|timing cohort|MAC \/ sample/i)).not.toBeInTheDocument()
  const incomplete = screen.getByText('Incomplete coverage').closest('tr')!
  expect(within(incomplete).getByText('Unranked')).toBeVisible()
  expect(within(incomplete).queryByText('12.25')).not.toBeInTheDocument()
  expect(within(screen.getByText('Failed baseline').closest('tr')!).getByText('Failed')).toBeVisible()
  await userEvent.click(screen.getByRole('button', { name: 'Details: Incomplete coverage' }))
  expect(screen.getByText('Missing required seed.')).toBeVisible()
  const sweep = screen.getByRole('table', { name: 'Parameter sweep' })
  expect(sweep).toHaveTextContent('≤ 500H=1044244645490015.0014.750.89 %19.00-46.0 dBc21.0015.2915.2115.25Valid')
  expect(sweep).toHaveTextContent('Outside the power gate')
  for (const name of ['EVM', 'EVM gain (dB)', 'ACLR', 'ACLR gain (dB)']) expect(within(sweep).getByRole('columnheader', { name })).toBeVisible()
  expect(within(sweep).queryByRole('columnheader', { name: /NMSE/ })).not.toBeInTheDocument() // a diagnostic, not part of any score
  expect(sweep).toHaveTextContent('≤ 2,000———')
  expect(sweep).toHaveTextContent('No configuration in this budget')
  await userEvent.click(screen.getByText('Operation breakdown · ≤ 500 parameters'))
  const breakdown = screen.getByRole('region', { name: 'Operation breakdown · ≤ 500 parameters' })
  expect(breakdown).toHaveTextContent('GRU cell440450sigmoid × 20 · tanh × 10')
  expect(breakdown).toHaveTextContent('Nonlinear functions at reference price3030')
  expect(breakdown).toHaveTextContent('Total446454OPs / sample: 900')
  await userEvent.click(screen.getByText('Per-configuration, condition and seed metrics (1)'))
  const audit = screen.getByRole('table', { name: 'Per-configuration, condition and seed metrics' })
  expect(audit).toHaveTextContent('≤ 500apa-200mhz-b0pa')
  expect(audit).toHaveTextContent('-22.00 → -41.00') // EVM first: it is half of the quality
  expect(audit).toHaveTextContent('-20.00 → -35.00')
  expect(audit).toHaveTextContent('-25.00 → -46.00')
  expect(audit).toHaveTextContent('-26.00 → -47.00')
  expect(audit).toHaveTextContent('0.12')
  expect(within(audit).getAllByRole('columnheader').map(cell => cell.textContent).slice(4)).toEqual(['EVM (dB)', 'ACLR L (dBc)', 'ACLR R (dBc)', 'NMSE (dB)', 'Power error (dB)'])
  const diagnostics = screen.getByRole('table', { name: 'Output and reference ACLR' })
  expect(diagnostics).toHaveTextContent('-25.00 → -46.00')
  expect(diagnostics).toHaveTextContent('-26.00 → -47.00')
  expect(diagnostics).toHaveTextContent('-49.00')
  expect(diagnostics).toHaveTextContent('-50.00')
  await waitFor(() => expect(screen.getByRole('link', { name: 'Submit' })).toHaveAttribute('href', '/arena/submit?board=apa-200mhz-b'))
})

test('each ranking shows its own score and order, on the APA B dataset', async () => {
  const other: ArenaRow = { ...base, entry_id: 'official-tcn', backbone: 'tcn', display_name: 'TCN reference', rankings: { overall: { rank: 2, score: 10.5 }, linearization: { rank: 1, score: 16.1 }, arithmetic_efficiency: { rank: 2, score: 9.25 }, 'budget-500': { rank: null, score: null } } }
  setup({ rows: [base, other] })
  await screen.findByText('GRU reference')
  await userEvent.click(screen.getByText('Backbone summaries and evidence'))
  const names = () => within(screen.getByRole('table', { name: 'Bundled baseline' })).getAllByRole('row').slice(1).map(row => within(row).getAllByRole('cell')[1]!.textContent)
  expect(names()).toEqual(['GRU referencegru', 'TCN referencetcn'])
  expect(screen.getByText(/Per-configuration mean quality/)).toBeVisible()
  await userEvent.click(screen.getByRole('combobox', { name: 'Ranking' }))
  expect(screen.getAllByRole('option').map(option => option.textContent)).toEqual(['Overall FoM', 'Best linearization', 'Arithmetic efficiency', '≤ 500 parameters'])
  await userEvent.click(screen.getByRole('option', { name: 'Best linearization' }))
  expect(names()).toEqual(['TCN referencetcn', 'GRU referencegru'])
  expect(within(screen.getByText('TCN reference').closest('tr')!).getByText('16.10')).toBeVisible()
  expect(screen.getByText(/Mean equal-weight EVM and output ACLR improvement/)).toBeVisible()
  await waitFor(() => expect(screen.getByRole('combobox', { name: 'Ranking' })).toHaveTextContent('Best linearization'))
  await userEvent.click(screen.getByRole('combobox', { name: 'Ranking' }))
  await userEvent.click(screen.getByRole('option', { name: '≤ 500 parameters' }))
  const unplaced = screen.getByText('TCN reference').closest('tr')! // ranked overall, but outside the power gate at this budget
  expect(within(unplaced).getAllByRole('cell')[0]).toHaveTextContent('—')
  expect(within(unplaced).getAllByRole('cell')[2]).toHaveTextContent('—')
  expect(within(unplaced).getByText('Ranked')).toBeVisible()
  await userEvent.click(screen.getByRole('button', { name: 'Details: GRU reference' }))
  expect(screen.getByText('Best linearization: 14.75 · #2')).toBeVisible()
  expect(screen.getByText('≤ 500 parameters: 15.25 · #1')).toBeVisible()
})

test('old-protocol results cannot display a current rank or score', async () => {
  setup({ rows: [{ ...base, protocol_sha256: 'b'.repeat(64), rankings: { overall: { rank: 1, score: 999 } } }] })
  await screen.findByText('GRU reference')
  await userEvent.click(screen.getByText('Backbone summaries and evidence'))
  expect(screen.getByText('Unranked')).toBeVisible()
  expect(screen.queryByText('999.00')).not.toBeInTheDocument()
  await userEvent.click(screen.getByRole('button', { name: 'Details: GRU reference' }))
  expect(screen.getByText('This result belongs to a different protocol and cannot receive a rank on this board.')).toBeVisible()
})

test('separates reference and workspace results and execution modes, never hosts', async () => {
  setup({ rows: [base, { ...base, entry_id: 'stream', display_name: 'GRU streaming', backbone: 'gru_stream', execution_semantics: 'streaming_stateful' }, { ...base, entry_id: 'private', display_name: 'My GRU', origin: 'workspace', provenance: { environment: { device: 'cpu', torch: '2.9.0' } } }] })
  await screen.findByText('GRU reference')
  expect(screen.getAllByRole('table', { name: 'Bundled baseline' })).toHaveLength(2)
  expect(screen.getByRole('table', { name: 'Workspace submission' })).toHaveTextContent('My GRU')
  expect(screen.getAllByText('Offline · 200-sample windows / 100-sample hop · Overall FoM')).toHaveLength(2)
  expect(screen.getByText('Stateful streaming · Overall FoM')).toBeVisible()
  expect(screen.getByRole('region', { name: 'Bundled baseline · Stateful streaming' })).toHaveTextContent('GRU streaming')
  expect(screen.getByText(/Operation counts do not depend on the machine that ran the evaluation/)).toBeVisible()
  expect(screen.queryByText(/PyTorch 2|CPU threads/)).not.toBeInTheDocument()
})

test('keeps the offline board above the streaming cohort even when a streaming entry sorts first', async () => {
  setup({ rows: [{ ...base, entry_id: 'stream', display_name: 'GRU streaming', backbone: 'gru_stream', execution_semantics: 'streaming_stateful' }, base] })
  await screen.findByText('GRU reference')
  const regions = screen.getAllByRole('region').map(region => region.getAttribute('aria-label'))
  expect(regions.indexOf('Bundled baseline · Offline · 200-sample windows / 100-sample hop')).toBeGreaterThanOrEqual(0)
  expect(regions.indexOf('Bundled baseline · Offline · 200-sample windows / 100-sample hop')).toBeLessThan(regions.indexOf('Bundled baseline · Stateful streaming'))
})

test('excludes legacy ILC entries from configuration ranks, plots and backbone summaries', async () => {
  setup({ rows: [base, { ...base, entry_id: 'legacy-ilc', backbone: 'ilc_dpd', display_name: ilcRegistryName }, { ...base, entry_id: 'my-ilc', backbone: 'ilc_dpd', origin: 'workspace', display_name: 'My training-feedback ILC' }] })
  await screen.findByText('GRU reference')
  await userEvent.click(screen.getByText('Backbone summaries and evidence'))
  expect(screen.queryByText(ilcLabel)).not.toBeInTheDocument()
  expect(screen.queryByText(ilcRegistryName)).not.toBeInTheDocument()
  expect(screen.queryByText('My training-feedback ILC')).not.toBeInTheDocument()
  expect(screen.queryByRole('table', { name: 'Workspace submission' })).not.toBeInTheDocument()
  expect(screen.getAllByTestId('arena-evm-parameters')).toHaveLength(1)
})

test('excludes ILC from submission choices even if a stale catalogue contains it', async () => {
  const { calls } = setup({ route: '/arena/submit' })
  await screen.findByRole('button', { name: 'Run benchmark' })
  await userEvent.click(screen.getByRole('combobox', { name: 'Backbone' }))
  expect(screen.queryByRole('option', { name: `${ilcRegistryName} · classical` })).not.toBeInTheDocument()
  expect(screen.queryByRole('option', { name: `${ilcLabel} · classical` })).not.toBeInTheDocument()
  expect(screen.getByRole('option', { name: 'Memory polynomial · polynomial' })).toBeVisible()
  expect(calls.filter(call => call.method === 'POST')).toHaveLength(0)
})

test('submits only the acknowledged APA B protocol-bound execution request', async () => {
  const { calls } = setup({ route: '/arena/submit' })
  const launch = await screen.findByRole('button', { name: 'Run benchmark' })
  expect(launch).toBeDisabled()
  await userEvent.type(screen.getByRole('textbox', { name: 'Entry name' }), 'My evaluation')
  await userEvent.click(screen.getByRole('checkbox', { name: /I accept this protocol/ }))
  expect(launch).toBeEnabled()
  await userEvent.click(launch)
  await waitFor(() => expect(calls.filter(call => call.method === 'POST')).toHaveLength(1))
  expect(await screen.findByText('My evaluation')).toBeVisible()
  expect(calls.find(call => call.method === 'POST')!.body).toMatchObject({ board_id: 'apa-200mhz-b', accepted_protocol_sha256: sha })
})

test('hosted read-only mode explains the restriction and cannot submit', async () => {
  const { calls } = setup({ available: false, route: '/arena/submit' })
  expect(await screen.findByText('Hosted Arena execution is unavailable.')).toBeVisible()
  expect(screen.getByRole('button', { name: 'Run benchmark' })).toBeDisabled()
  expect(screen.getByRole('checkbox', { name: /I accept this protocol/ })).toBeDisabled()
  expect(calls.some(call => call.method === 'POST')).toBe(false)
})

test('historical submissions keep evidence downloads but cannot link to current ranks or claim neural training epochs', async () => {
  setup({ route: '/arena/submit', initialSubmissions: [{ submission_id: 'arena-' + 'd'.repeat(32), request: { board_id: 'apa-200mhz-b', backbone: 'mp_ls', display_name: 'Old MP record', accepted_protocol_sha256: 'b'.repeat(64) }, protocol_sha256: 'b'.repeat(64), status: 'succeeded', progress: { phase: 'complete', message: 'Evaluation complete', epoch: 150, epochs: 150, completed_cases: 2, expected_cases: 2 } }] })
  expect(await screen.findByText('Old MP record')).toBeVisible()
  expect(screen.getByText('This result belongs to a different protocol and cannot receive a rank on this board.')).toBeVisible()
  expect(screen.getByRole('status')).toHaveTextContent('Evaluation complete · Conditions: 2 / 2')
  expect(screen.getByRole('status')).not.toHaveTextContent('150 / 150')
  expect(screen.queryByRole('link', { name: 'Rank' })).not.toBeInTheDocument()
  expect(screen.getByRole('link', { name: 'Download evidence JSON' })).toBeVisible()
})

test('deterministic methods show one seed and bundled templates submit without a private backbone ID', async () => {
  const { calls } = setup({ route: '/arena/submit' })
  await screen.findByRole('button', { name: 'Run benchmark' })
  expect(screen.getByText('240 full training epochs · batch 64 · every training window, at each parameter budget')).toBeVisible()
  expect(screen.getByText(/^Parameter budgets: 250 · 500 · 1,000 · 2,000 · every budget is trained and judged separately/)).toBeVisible()
  await userEvent.click(screen.getByRole('combobox', { name: 'Backbone' }))
  await userEvent.click(screen.getByRole('option', { name: 'Memory polynomial · polynomial' }))
  expect(screen.getByText('Seeds: 0 · Conditions: apa-200mhz-b')).toBeVisible()
  expect(screen.getByText('One deterministic fit per condition and parameter budget; no artificial seed repetitions.')).toBeVisible()
  await userEvent.click(screen.getByRole('combobox', { name: 'Backbone' }))
  await userEvent.click(screen.getByRole('option', { name: 'Template GRU (bundled example) · template' }))
  await userEvent.click(screen.getByRole('checkbox', { name: /I accept this protocol/ }))
  await userEvent.click(screen.getByRole('button', { name: 'Run benchmark' }))
  await waitFor(() => expect(calls.filter(call => call.method === 'POST')).toHaveLength(1))
  expect(calls.find(call => call.method === 'POST')?.body).toMatchObject({ backbone: 'user_template', accepted_protocol_sha256: sha })
  expect(calls.find(call => call.method === 'POST')?.body).not.toHaveProperty('backbone_id')
})

test('rules show the server protocol, score definition, rankings, cost model and fixed budget', async () => {
  setup({ route: '/arena/rules?board=apa-200mhz-b' })
  expect(await screen.findByText(formula)).toBeVisible()
  expect(screen.getByText('All required seeds must finish.')).toBeVisible()
  expect(screen.getByText(/"epochs": 240/)).toBeVisible()
  expect(screen.getByText('Arithmetic efficiency')).toBeVisible()
  expect(screen.getByText(/linearization per multiplication and addition/)).toBeVisible()
  expect(screen.getByText('≤ 500 parameters')).toBeVisible()
  // What quality means: EVM and ACLR in equal parts, output power as the only gate.
  expect(screen.getByText(/equal-weight mean of EVM and output ACLR improvement/)).toBeVisible()
  expect(screen.getByText(/Each configuration must keep output power within ±0.5 dB and have positive mean quality/)).toBeVisible()
  expect(screen.getByText(/complete symbols in the held-out APA_200MHz_b input/)).toBeVisible()
  expect(screen.getByText(/AER is a separate diagnostic and does not determine the score/)).toBeVisible()
  expect(screen.getByRole('heading', { name: 'Hardware-agnostic cost model' })).toBeVisible()
  expect(screen.getByText(/so Arena does not time any host/)).toBeVisible()
  const prices = screen.getByRole('table', { name: 'Nonlinear function price list' })
  expect(prices).toHaveTextContent('tanh11')
  expect(prices).toHaveTextContent('atan234')
  expect(prices).toHaveTextContent('relu00')
  expect(screen.getByRole('link', { name: 'Download evidence JSON' })).toHaveAttribute('href', '/api/v1/arena')
})
