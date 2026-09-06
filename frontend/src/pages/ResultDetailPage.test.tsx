import { screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import generalProfile from '@mocks/metric_profile_general.json'
import legacyProfile from '@mocks/metric_profile_legacy.json'
import resultMock from '@mocks/result_pa_modeling_mock.json'
import dpdMock from '@mocks/result_dpd_surrogate_mock.json'
import { mockApi, renderWithProviders } from '@/test/utils'
import { ResultDetailPage } from './ResultDetailPage'

const legacy = resultMock.data
const general = {
  ...legacy,
  result_id: 'res-pa-0001-general-spectral-v1',
  metric_profile_id: 'general-spectral-v1',
  metrics: [
    { name: 'NMSE', value: -22.5, unit: 'dB', better: 'lower', status: 'ok', reason: null },
    { name: 'IBE', value: -23.1, unit: 'dB', better: 'lower', status: 'ok', reason: null },
    { name: 'ACPR_L', value: null, unit: 'dBc', better: 'lower', status: 'not_applicable', reason: 'adjacent channel band [100.0, 300.0] MHz exceeds the captured range' },
    { name: 'ACPR_R', value: -31.0, unit: 'dBc', better: 'lower', status: 'ok', reason: null },
  ],
}

test('shows the profile behind every score, its definitions, and switches to another stored profile', async () => {
  const { calls } = mockApi({
    'GET /api/v1/results/run-pa-0001': (url) => (url.searchParams.get('profile') === 'general-spectral-v1' ? general : legacy),
    'GET /api/v1/results/run-pa-0001/profiles': () => ['legacy-opendpd-v1', 'general-spectral-v1'],
    'GET /api/v1/metrics/profiles': () => [legacyProfile.data, generalProfile.data],
  })
  renderWithProviders(<ResultDetailPage />, { route: '/results/run-pa-0001', path: '/results/:runId' })
  await screen.findByText('legacy-opendpd-v1 v1 · frozen')
  expect(screen.getByRole('heading', { level: 3, name: 'NMSE (mean of segment dB)' })).toBeInTheDocument()
  await userEvent.click(screen.getByRole('button', { name: 'Definition of EVM' }))
  await screen.findByText(/Not a demodulated constellation EVM/)
  expect(screen.getByRole('button', { name: 'Metric definitions' })).toBeInTheDocument()

  await userEvent.click(screen.getByLabelText('Metric profile'))
  await userEvent.click(await screen.findByRole('option', { name: 'general-spectral-v1' }))
  await screen.findByRole('heading', { level: 3, name: 'NMSE (pooled)' })
  expect(screen.getByText('-22.50 dB')).toBeInTheDocument()
  const acprL = screen.getByRole('region', { name: 'ACPR_L' })
  expect(within(acprL).getByText('not applicable')).toBeInTheDocument()
  expect(within(acprL).getByText(/exceeds the captured range/)).toBeInTheDocument()
  expect(calls.some((c) => c.path === '/api/v1/results/run-pa-0001' && c.method === 'GET')).toBe(true)
  expect(screen.getByRole('heading', { level: 1 })).toHaveTextContent('res-pa-0001-general-spectral-v1')
})

test('a DPD result shows the x/u/y chain, baselines under one reference, coverage and the uncalibrated note', async () => {
  mockApi({
    'GET /api/v1/results/run-dpd-0001': () => dpdMock.data,
    'GET /api/v1/results/run-dpd-0001/profiles': () => ['legacy-opendpd-v1'],
    'GET /api/v1/metrics/profiles': () => [legacyProfile.data],
  })
  renderWithProviders(<ResultDetailPage />, { route: '/results/run-dpd-0001', path: '/results/:runId' })
  const chain = await screen.findByRole('region', { name: 'Signal chain' })
  const rows = within(chain).getAllByRole('row').slice(1)
  expect(rows.map((r) => r.getAttribute('data-stage'))).toEqual(['x', 'u', 'y'])
  expect(within(rows[2]!).getByText('simulated')).toBeInTheDocument()
  expect(within(rows[1]!).getByText(/u = DPD\(x\)/)).toBeInTheDocument()
  const baselines = screen.getByRole('region', { name: 'Baselines under the same reference' })
  expect(within(baselines).getByText('Surrogate without DPD')).toBeInTheDocument()
  expect(within(baselines).getByText('Measured PA without DPD')).toBeInTheDocument()
  expect(within(baselines).getByText('-26.82 dBc')).toBeInTheDocument()
  expect(within(baselines).getByText('-19.39 dBc')).toBeInTheDocument()
  expect(screen.getByTestId('surrogate-coverage')).toHaveTextContent('0.31% of the pre-distorted samples above the fitted peak')
  expect(screen.getByTestId('scaling')).toHaveTextContent('No physical calibration')
})
