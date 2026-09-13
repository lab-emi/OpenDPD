import { fireEvent, screen } from '@testing-library/react'
import { vi } from 'vitest'
import general from '@mocks/metric_profile_general.json'
import pending from '@mocks/metric_profile_ofdm_evm.json'
import { mockApi, renderWithProviders } from '@/test/utils'
import type { PlotlyChartProps } from '@/components/PlotlyChart'
import { HardwareCostsPage } from './HardwareCostsPage'

vi.mock('@/components/PlotlyChart', () => ({ seriesSymbol: () => 'circle', PlotlyChart: (props: PlotlyChartProps) => <output data-testid="points">{JSON.stringify(props.traces.map(t => [t.x, t.y]))}</output> }))
const base = { entry_id: 'cost-a', run_id: 'run-a', profile_id: 'general-spectral-v1', title: 'Stored constants', source_kind: 'operation_count', source_type: 'checkpoint_shapes', source_file: { sha256: 'a'.repeat(64) }, result_sha256: 'b'.repeat(64), weights_sha256: 'c'.repeat(64), values: { constant_bytes: 2048, energy_j_per_sample: null }, precision: [], metrics: [{ name: 'ACPR_L', value: -40, status: 'ok' }], metric_basis: 'Stored RF result', rf_evidence_type: 'dpd_surrogate', execution_semantics: 'offline', synthetic: true, limitations: [], target: 'Reference', boundary: 'Digital model only', stale: false }

test('unknown energy is unavailable; stale costs do not produce a scatter point or release pending EVM', async () => {
  mockApi({
    'GET /api/v1/runs': () => [{ run_id: 'run-a', status: 'succeeded', result_id: 'res-a' }],
    'GET /api/v1/metrics/profiles': () => [general.data, pending.data],
    'GET /api/v1/hardware/costs': () => ({ entries: [base, { ...base, entry_id: 'cost-stale', title: 'Changed source', stale: true, values: { constant_bytes: 4000, energy_j_per_sample: 1e-9 } }], missing: {}, notes: [] }),
  })
  renderWithProviders(<HardwareCostsPage />, { route: '/hardware?runs=run-a' })
  expect(await screen.findByTestId('points')).toHaveTextContent('[[[2],[-40]]]')
  fireEvent.mouseDown(screen.getByRole('combobox', { name: 'Cost axis' }))
  expect(screen.queryByRole('option', { name: /Energy per sample/ })).not.toBeInTheDocument()
  fireEvent.keyDown(screen.getByRole('listbox'), { key: 'Escape' })
  fireEvent.mouseDown(screen.getByRole('combobox', { name: 'Metric profile' }))
  expect(screen.queryByRole('option', { name: 'ofdm-lte20-evm-v1' })).not.toBeInTheDocument()
})
