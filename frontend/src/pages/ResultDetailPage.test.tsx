import { screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import generalProfile from '@mocks/metric_profile_general.json'
import legacyProfile from '@mocks/metric_profile_legacy.json'
import resultMock from '@mocks/result_pa_modeling_mock.json'
import dpdMock from '@mocks/result_dpd_surrogate_mock.json'
import { mockApi, renderWithProviders } from '@/test/utils'
import { ResultDetailPage } from './ResultDetailPage'

vi.mock('plotly.js-basic-dist-min', () => ({ default: { react: vi.fn(() => Promise.resolve()), purge: vi.fn() } }))

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
    'GET /api/v1/artifacts/run-dpd-0001/plot-spectrum': () => ({ version: 'plots-v1', axis: 'hz', sample_rate_hz: 800e6, nperseg: 4, n_samples: 16, frequency: [-400e6, -200e6, 0, 200e6], traces: [{ name: 'with DPD', role: 'primary', psd_db: [-80, -60, -60, -80] }], bands: { main: [-100e6, 100e6], adjacent: [] }, estimator: 'welch' }),
    'GET /api/v1/artifacts/run-dpd-0001/plot-time': () => ({ version: 'plots-v1', start: 0, n: 4, n_samples: 16, traces: [{ name: 'x', role: 'input', i: [0, 1, 0, -1], q: [1, 0, -1, 0] }] }),
    'GET /api/v1/artifacts/run-dpd-0001/plot-amam': () => ({ version: 'plots-v1', stride: 1, n_points: 4, n_samples: 4, amp_in: [0.1, 0.2, 0.3, 0.4], traces: [{ name: 'with DPD', role: 'primary', amp_out: [0.2, 0.4, 0.6, 0.8], phase_deg: [0, 0, 1, 2] }], note: '' }),
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
  const charts = screen.getByRole('region', { name: 'Charts' })
  await within(charts).findByTestId('spectrum-plot')
  expect(within(charts).getByTestId('iq-preview')).toBeInTheDocument()
  expect(within(charts).getByTestId('am-am-plot')).toBeInTheDocument()
  expect(within(charts).getByTestId('am-pm-plot')).toBeInTheDocument()
})

test('a result without plot artifacts says so instead of drawing anything', async () => {
  mockApi({
    'GET /api/v1/results/run-pa-0001': () => legacy,
    'GET /api/v1/results/run-pa-0001/profiles': () => ['legacy-opendpd-v1'],
    'GET /api/v1/metrics/profiles': () => [legacyProfile.data],
  })
  renderWithProviders(<ResultDetailPage />, { route: '/results/run-pa-0001', path: '/results/:runId' })
  const charts = await screen.findByRole('region', { name: 'Charts' })
  await within(charts).findByText(/No derived plot data/)
  expect(within(charts).queryByTestId('spectrum-plot')).not.toBeInTheDocument()
})

test('exporting a share package shows what was left out and a download link; reports link to the server', async () => {
  const manifest = {
    schema_version: 1,
    package_version: 1,
    kind: 'share',
    created_at: '2026-09-06T08:00:00Z',
    opendpd_version: '2.2.0.dev0',
    software: { opendpd_version: '2.2.0.dev0', python_version: '3.13', platform: 'linux' },
    run_id: 'run-pa-0001',
    task: 'train_pa',
    config_sha256: 'a'.repeat(64),
    seed: 0,
    dataset: { dataset_id: 'capture', raw_sha256: 'b'.repeat(64), preprocessing_version: 'raw-v1', split_version: 'contiguous-v1', source_kind: 'upload', included: false, how_to_obtain: 'ask the author for the capture with raw sha256 bbbb…' },
    references: [],
    files: [],
    reproduction: {},
    redaction: ['worker logs are not included', 'machine paths rewritten to <workspace>'],
    missing: ['dataset capture (raw sha256 bbbb…)'],
    retraining_note: 'Re-training is a new experiment, not a reproduction of the stored result.',
  }
  const { calls } = mockApi({
    'GET /api/v1/results/run-pa-0001': () => ({ ...legacy, is_mock: false }),
    'GET /api/v1/results/run-pa-0001/profiles': () => ['legacy-opendpd-v1'],
    'GET /api/v1/metrics/profiles': () => [legacyProfile.data],
    'POST /api/v1/exports': () => ({ status: 201, body: { export_id: 'run-pa-0001-share-20260906T080000', filename: 'run-pa-0001-share-20260906T080000.zip', size_bytes: 2_621_440, download_url: '/api/v1/exports/run-pa-0001-share-20260906T080000', manifest } }),
  })
  renderWithProviders(<ResultDetailPage />, { route: '/results/run-pa-0001', path: '/results/:runId' })
  const panel = await screen.findByRole('region', { name: 'Export and report' })
  expect(within(panel).getByRole('link', { name: 'Report (HTML)' })).toHaveAttribute('href', '/api/v1/results/run-pa-0001/report?format=html')
  expect(within(panel).getByRole('link', { name: 'Report (Markdown)' })).toHaveAttribute('href', '/api/v1/results/run-pa-0001/report?format=md')
  await userEvent.click(within(panel).getByRole('button', { name: 'Export share package' }))
  const ready = await screen.findByTestId('export-ready')
  expect(ready).toHaveTextContent('Package ready: run-pa-0001-share-20260906T080000.zip (2.5 MB)')
  expect(within(ready).getByRole('link', { name: 'Download package' })).toHaveAttribute('href', '/api/v1/exports/run-pa-0001-share-20260906T080000')
  expect(within(ready).getByText('machine paths rewritten to <workspace>')).toBeInTheDocument()
  expect(within(ready).getByText('dataset capture (raw sha256 bbbb…)')).toBeInTheDocument()
  expect(ready).toHaveTextContent('Re-training is a new experiment')
  const post = calls.find((c) => c.method === 'POST' && c.path === '/api/v1/exports')
  expect(post?.body).toEqual({ run_id: 'run-pa-0001', kind: 'share' })
})

test('the export panel is not offered for a mock result', async () => {
  mockApi({
    'GET /api/v1/results/run-pa-0001': () => legacy,
    'GET /api/v1/results/run-pa-0001/profiles': () => ['legacy-opendpd-v1'],
    'GET /api/v1/metrics/profiles': () => [legacyProfile.data],
  })
  renderWithProviders(<ResultDetailPage />, { route: '/results/run-pa-0001', path: '/results/:runId' })
  await screen.findByText('legacy-opendpd-v1 v1 · frozen')
  expect(screen.queryByRole('region', { name: 'Export and report' })).not.toBeInTheDocument()
})
