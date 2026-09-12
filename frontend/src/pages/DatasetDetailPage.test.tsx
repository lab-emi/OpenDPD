import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import datasetMock from '@mocks/dataset_builtin.json'
import { mockApi, renderWithProviders } from '@/test/utils'
import { DatasetDetailPage } from './DatasetDetailPage'

const plotMock = vi.hoisted(() => vi.fn(() => Promise.resolve()))
vi.mock('plotly.js-basic-dist-min', () => ({ default: { react: plotMock, purge: vi.fn() } }))

const rawVersion = { version: 'raw-v1', base_version: null, created_at: '2026-09-06T08:00:00Z', params: null, code_version: null, fit_range: null, record: {}, n_samples: 20000, split: datasetMock.data.split, files: [], sha256: null }
const dataset = { ...datasetMock.data, dataset_id: 'mine', versions: [rawVersion] }
const report = {
  report_id: 'doc-20260906-abcd1234',
  dataset_id: 'mine',
  doctor_version: 'dataset-doctor-v1',
  generated_at: '2026-09-06T08:00:00Z',
  evaluation_blocked: false,
  schema_version: 1,
  dataset_raw_sha256: null,
  items: [
    { code: 'time_misalignment', severity: 'warning', title: 'Time misalignment', message: 'Output lags input by 6.0 samples', evidence: { delay_samples: 6, confidence: 0.98 }, suggestion: 'Apply delay correction', confidence: 0.98, blocking: false },
    { code: 'linear_gain_phase', severity: 'info', title: 'Linear gain and phase', message: 'gain 2.9 dB, phase 25.1 deg', evidence: { gain_db: 2.9, phase_deg: 25.1 }, blocking: false },
  ],
}

const inspection = {
  version: 'dataset-inspection-v1', dataset_id: 'mine', data_version: 'raw-v1', total_samples: 20000,
  sample_range: [0, 20000], metadata_complete: true, inspection_ready: true,
  diagnostics: { ...report, items: [] }, measurements: [], notes: [], spectrum: null, time: null, iq: null, am: null,
}

test('run doctor → accept estimates → preview → create a traceable version', async () => {
  let latest: unknown = null
  const { calls } = mockApi({
    'GET /api/v1/datasets/mine': () => dataset,
    'GET /api/v1/datasets/mine/analysis': () => ({ ...inspection, diagnostics: latest ?? inspection.diagnostics }),
    'POST /api/v1/datasets/mine/diagnostics': () => {
      latest = report
      return report
    },
    'POST /api/v1/datasets/mine/preprocess/preview': () => ({ n_samples_before: 20000, n_samples_after: 19994, record: {}, report_after: { ...report, items: [{ code: 'alignment_ok', severity: 'info', title: 'Aligned', message: 'residual 0.01 samples', evidence: { delay_samples: 0.01 }, blocking: false }] } }),
    'POST /api/v1/datasets/mine/preprocess': (_url, init) => ({ status: 201, body: { ...rawVersion, version: (JSON.parse(String(init.body)) as { version: string }).version, base_version: 'raw-v1', n_samples: 19994 } }),
  })
  renderWithProviders(<DatasetDetailPage />, { route: '/datasets/mine', path: '/datasets/:datasetId' })
  await userEvent.click(await screen.findByRole('tab', { name: /Dataset Doctor/ }))
  await userEvent.click(await screen.findByRole('button', { name: 'Run again' }))
  await screen.findByRole('article', { name: 'Time misalignment' })
  expect(screen.getByText('No blocking problems.')).toBeInTheDocument()

  await userEvent.click(screen.getByRole('button', { name: 'Preprocess…' }))
  const dialog = await screen.findByRole('dialog')
  await userEvent.click(within(dialog).getByRole('button', { name: 'Use doctor estimates' }))
  expect(within(dialog).getByLabelText('Delay correction (samples)')).toHaveValue('6')
  expect(within(dialog).getByLabelText('Gain correction (dB)')).toHaveValue('2.9')
  await userEvent.click(within(dialog).getByRole('button', { name: 'Preview' }))
  await within(dialog).findByRole('article', { name: 'Aligned' })
  expect(within(dialog).getByText(/19.994 samples; doctor after preprocessing/)).toBeInTheDocument()
  await userEvent.type(within(dialog).getByLabelText('New version name'), 'aligned-v1')
  await userEvent.click(within(dialog).getByRole('button', { name: 'Create version' }))
  await screen.findByText('Version aligned-v1 created.')
  const create = calls.find((c) => c.method === 'POST' && c.path === '/api/v1/datasets/mine/preprocess')
  expect(create?.body).toMatchObject({ version: 'aligned-v1', base_version: 'raw-v1', params: { delay_samples: 6, gain_db: 2.9, phase_deg: 25.1, normalize: 'none', remove_outliers: false } })
})

test('metadata editor sends the whole signal spec, carrying over fields it does not show', async () => {
  const { calls } = mockApi({
    'GET /api/v1/datasets/mine': () => dataset,
    'GET /api/v1/datasets/mine/analysis': () => inspection,
    'POST /api/v1/datasets/mine/manifest': (_url, init) => ({ ...dataset, ...(JSON.parse(String(init.body)) as object) }),
  })
  renderWithProviders(<DatasetDetailPage />, { route: '/datasets/mine', path: '/datasets/:datasetId' })
  await userEvent.click(await screen.findByRole('button', { name: 'Edit metadata' }))
  const dialog = await screen.findByRole('dialog')
  const nperseg = within(dialog).getByLabelText('PSD segment length (nperseg)')
  await userEvent.clear(nperseg)
  await userEvent.type(nperseg, '1280')
  await userEvent.click(within(dialog).getByRole('button', { name: 'Save metadata' }))
  await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())
  const call = calls.find((c) => c.path === '/api/v1/datasets/mine/manifest')
  expect(call?.body).toMatchObject({ display_name: dataset.display_name, origin: 'measured', signal: { nperseg: 1280, modulation: '64QAM', sample_rate_hz: 800000000 } })
})

test('known dataset defaults to demodulated symbols and can switch back to raw samples', async () => {
  const rawTrace = { name: 'Input', role: 'input', i: [.01, .02], q: [.03, .04] }
  const symbols = [
    { name: 'Input', role: 'input', i: [-1, 1], q: [1, -1], n_symbols: 2, stride: 1, equalized: false },
    { name: 'PA output (equalized)', role: 'primary', i: [-.9, .95], q: [1.1, -1.05], n_symbols: 2, stride: 1, equalized: true },
  ]
  mockApi({
    'GET /api/v1/datasets/mine': () => dataset,
    'GET /api/v1/datasets/mine/analysis': () => ({ ...inspection,
      iq: { version: 'plots-v1', kind: 'iq', mode: 'samples', stride: 1, n_samples: 2, traces: [rawTrace] },
      constellation: { status: 'ok', dataset_name: 'DPA_200MHz', modulation: '64QAM', demodulator: 'datasets.DPA_200MHz.demod', sample_range: [0, 38400], source_sample_range: [0, 38400], traces: symbols, note: 'Dataset-specific equalization.' },
    }),
  })
  renderWithProviders(<DatasetDetailPage />, { route: '/datasets/mine', path: '/datasets/:datasetId' })
  await screen.findByRole('heading', { name: '64QAM · Constellation' })
  expect(screen.getByRole('button', { name: 'Symbols' })).toHaveAttribute('aria-pressed', 'true')
  await waitFor(() => expect(plotMock).toHaveBeenLastCalledWith(expect.anything(), expect.arrayContaining([
    expect.objectContaining({ x: symbols[0]!.i }), expect.objectContaining({ name: 'PA output (equalized)' }),
  ]), expect.anything(), expect.anything()))
  await userEvent.click(screen.getByRole('button', { name: 'Raw I/Q' }))
  expect(screen.getByRole('heading', { name: 'Raw I/Q' })).toBeInTheDocument()
  await waitFor(() => expect(plotMock).toHaveBeenLastCalledWith(expect.anything(), [expect.objectContaining({ x: rawTrace.i })], expect.anything(), expect.anything()))
})

test('a QAM label without a bound demodulator stays a raw I/Q plot with a reason', async () => {
  mockApi({
    'GET /api/v1/datasets/mine': () => dataset,
    'GET /api/v1/datasets/mine/analysis': () => ({ ...inspection,
      iq: { version: 'plots-v1', kind: 'iq', mode: 'samples', stride: 1, n_samples: 2, traces: [] },
      constellation: { status: 'unavailable', reason: 'No dataset-specific demodulator is bound.', traces: [], note: '' },
    }),
  })
  renderWithProviders(<DatasetDetailPage />, { route: '/datasets/mine', path: '/datasets/:datasetId' })
  await screen.findByRole('heading', { name: 'Raw I/Q' })
  expect(screen.queryByRole('button', { name: 'Symbols' })).not.toBeInTheDocument()
  expect(screen.getByTitle(/No dataset-specific demodulator is bound/)).toBeInTheDocument()
})
