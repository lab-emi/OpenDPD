import { screen, within } from '@testing-library/react'
import { vi } from 'vitest'
import comparisonMock from '@mocks/comparison_report_mock.json'
import paResult from '@mocks/result_pa_modeling_mock.json'
import resolvedMock from '@mocks/resolved_train_pa_smoke.json'
import type { EvaluationResult } from '@/api/types'
import { mockApi, renderWithProviders } from '@/test/utils'
import { bestIndex, ComparePage } from './ComparePage'

vi.mock('plotly.js-basic-dist-min', () => ({ default: { react: vi.fn(() => Promise.resolve()), purge: vi.fn() } }))

const spectrum = (primary: string) => ({
  version: 'plots-v1',
  axis: 'hz',
  sample_rate_hz: 800e6,
  nperseg: 8,
  n_samples: 64,
  frequency: [-400e6, -300e6, -200e6, -100e6, 0, 100e6, 200e6, 300e6],
  traces: [
    { name: 'reference', role: 'reference', psd_db: [-80, -70, -60, -50, -50, -60, -70, -80] },
    { name: primary, role: 'primary', psd_db: [-82, -72, -62, -52, -52, -62, -72, -82] },
  ],
  bands: { main: [-100e6, 100e6], adjacent: [[-300e6, -100e6], [100e6, 300e6]] },
  estimator: 'welch',
})

test('incompatible results are shown side by side with the reasons and nothing is ranked', async () => {
  mockApi({
    'GET /api/v1/results/compare': () => comparisonMock.data,
    'GET /api/v1/artifacts/run-pa-0001/plot-spectrum': () => spectrum('PA model output'),
    'GET /api/v1/artifacts/run-dpd-0001/plot-spectrum': () => spectrum('with DPD'),
    'GET /api/v1/runs/run-pa-0001/config': () => resolvedMock.data,
    'GET /api/v1/runs/run-dpd-0001/config': () => ({ ...resolvedMock.data, task: 'train_dpd' }),
  })
  renderWithProviders(<ComparePage />, { route: '/results/compare?runs=run-pa-0001&runs=run-dpd-0001', path: '/results/compare' })
  const verdict = await screen.findByTestId('compare-verdict')
  expect(verdict).toHaveTextContent('Different protocols')
  expect(verdict).toHaveTextContent('evidence type: pa_modeling vs dpd_surrogate')
  const table = screen.getByRole('table', { name: 'Compare results' })
  expect(within(table).getByRole('link', { name: 'run-pa-0001' })).toBeInTheDocument()
  expect(within(table).getByRole('link', { name: 'run-dpd-0001' })).toBeInTheDocument()
  expect(within(table).queryByText('best')).not.toBeInTheDocument()
  expect(screen.getByRole('link', { name: 'Download CSV' })).toHaveAttribute('href', expect.stringContaining('format=csv'))
  await screen.findByRole('figure', { name: 'Power spectral density' })
  await screen.findByRole('table', { name: 'Configuration differences' })
})

test('comparable results mark the best value per metric', async () => {
  const a = paResult.data as unknown as EvaluationResult
  const b: EvaluationResult = { ...a, result_id: 'res-pa-0002', run_id: 'run-pa-0002', metrics: a.metrics.map((m) => (m.name === 'NMSE' ? { ...m, value: (m.value ?? 0) - 5 } : m)) }
  mockApi({
    'GET /api/v1/results/compare': () => ({ ...comparisonMock.data, results: [a, b], comparable: true, pairs: [{ a: 'run-pa-0001', b: 'run-pa-0002', incompatibilities: [] }], note: 'same protocol' }),
    'GET /api/v1/runs/run-pa-0001/config': () => resolvedMock.data,
    'GET /api/v1/runs/run-pa-0002/config': () => resolvedMock.data,
  })
  renderWithProviders(<ComparePage />, { route: '/results/compare?runs=run-pa-0001&runs=run-pa-0002', path: '/results/compare' })
  expect(await screen.findByTestId('compare-verdict')).toHaveTextContent('Same protocol')
  const rows = screen.getAllByRole('row')
  const nmse = rows.find((r) => within(r).queryByText('NMSE'))!
  const cells = within(nmse).getAllByRole('cell')
  expect(cells[2]).toHaveAttribute('data-best', 'true')
  expect(cells[1]).not.toHaveAttribute('data-best')
  expect(bestIndex([a, b], 'NMSE')).toBe(1)
  expect(bestIndex([a, b], 'nope')).toBe(-1)
  await screen.findByText('No derived plot data for this result (it was evaluated before plots-v1). Re-evaluate the run to produce it.')
})

test('fewer than two runs asks the user to pick results', () => {
  mockApi({})
  renderWithProviders(<ComparePage />, { route: '/results/compare?runs=run-pa-0001', path: '/results/compare' })
  expect(screen.getByText('Pick at least two results on the Results page to compare them.')).toBeInTheDocument()
})
