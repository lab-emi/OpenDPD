import { fireEvent, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import { useLocation } from 'react-router'
import fixture from '@mocks/generator_presets.json'
import { mockApi, renderWithProviders } from '@/test/utils'
import { SignalGeneratorPage } from './SignalGeneratorPage'
import { StudioWorkflowProvider } from '@/workflow/StudioWorkflow'

vi.mock('@/components/PlotlyChart', () => ({ PlotlyChart: ({ title }: { title: string }) => <div>{title}</div> }))

const fixturePresets = fixture.data
const config = fixturePresets[0]!.config
const result = {
  signal_id: `sg-${'a'.repeat(64)}`, iq_sha256: 'b'.repeat(64), config, coverage: 'numerology',
  download_url: `/api/v1/signal-generator/signals/sg-${'a'.repeat(64)}/download`,
  analysis: { sample_count: 131072, duration_ms: 1.0666, sample_rate_hz: 122.88e6, subcarrier_spacing_hz: 30000, useful_symbol_us: 33.3333, cp_lengths_samples: [352, 288], complete_symbols: 29, trailing_samples: 100, active_carriers: 612, data_carriers: 560, pilot_carriers: 52, rms: .2, peak: .8, papr_db: 12.04, mean_power_dbfs: -13.98, occupied_bandwidth_99_hz: 18.36e6, dc_magnitude: .001, evm_percent: .000002, evm_symbols: 16, time_us: [0, 1], time_i: [.1, .2], time_q: [.2, .1], time_envelope: [.22, .22], frequency_mhz: [-10, 10], psd_dbfs_hz: [-70, -70], constellation_i: [-1, 1], constellation_q: [1, -1], reference_i: [-1, 1], reference_q: [1, -1], ccdf_db: [0, 10], ccdf_probability: [.5, .01], allocation: [{ channel: 1, subcarriers: [-1, 1], pilots: [1], modulation_order: 64, power_db: 0 }], notes: ['SYNTHETIC'] },
}

function setup() {
  return mockApi({
    'GET /api/v1/signal-generator/presets': () => fixturePresets,
    'GET /api/v1/system/capabilities': () => ({ custom_dataset_imports: true }),
    'POST /api/v1/signal-generator/signals': (_url, init) => ({ ...result, config: JSON.parse(String(init.body)) }),
    [`POST /api/v1/signal-generator/signals/${result.signal_id}/dataset`]: (_url, init) => ({ dataset: { dataset_id: JSON.parse(String(init.body)).dataset_id }, test_samples: 26112 }),
    'POST /api/v1/signal-generator/validate': (_url, init) => JSON.parse(String(init.body)),
  })
}

function Probe() { const location = useLocation(); return <output data-testid="location">{location.pathname}{location.search}</output> }

test('one initial preview, simple family selection, stale-export guard and duration conversion', async () => {
  const { calls } = setup()
  renderWithProviders(<SignalGeneratorPage />)
  await screen.findByTestId('signal-generator-results')
  expect(calls.filter(c => c.path === '/api/v1/signal-generator/signals')).toHaveLength(1)
  expect(screen.getByText('Time-domain I/Q')).toBeVisible()
  expect(screen.getByText('PA Input · PSD')).toBeVisible()
  await userEvent.click(screen.getByRole('button', { name: /Wi-Fi 8/ }))
  expect(screen.getByText(/Wi-Fi 8 is an experimental/)).toBeVisible()
  expect(screen.getByRole('button', { name: 'Export I/Q + configuration' })).toBeDisabled()
  expect(screen.getByText(/Parameters changed/)).toBeVisible()
  await userEvent.click(screen.getByRole('button', { name: 'Elapsed time' }))
  fireEvent.change(screen.getByLabelText('Equivalent duration (ms)'), { target: { value: '.25' } })
  expect(screen.getByTestId('generator-length')).toHaveTextContent('80,000 I/Q samples')
  await userEvent.click(screen.getByRole('button', { name: 'Generate & preview' }))
  await waitFor(() => expect(screen.getByRole('button', { name: 'Export I/Q + configuration' })).toBeEnabled())
  expect(calls.filter(c => c.path === '/api/v1/signal-generator/signals').at(-1)?.body).toMatchObject({ preset_id: 'wifi8-80', length_mode: 'duration', duration_ms: .25 })
})

test('advanced OFDMA channels and pilots reach the generator request', async () => {
  const { calls } = setup()
  renderWithProviders(<SignalGeneratorPage />)
  await screen.findByTestId('signal-generator-results')
  await userEvent.click(screen.getByRole('button', { name: 'Advanced parameters' }))
  await userEvent.click(screen.getByRole('button', { name: 'Add OFDMA channel' }))
  expect(screen.getAllByLabelText('Subcarriers including pilots')).toHaveLength(2)
  fireEvent.change(screen.getAllByLabelText('Subcarriers including pilots')[1]!, { target: { value: '52' } })
  await userEvent.click(screen.getByRole('combobox', { name: 'Pilot allocation' }))
  await userEvent.click(screen.getByRole('option', { name: 'Explicit signed bin indices' }))
  fireEvent.change(screen.getByLabelText('Pilot carrier indices'), { target: { value: '-39, 39' } })
  await userEvent.click(screen.getByRole('button', { name: 'Apply & regenerate' }))
  await waitFor(() => expect(calls.filter(c => c.path === '/api/v1/signal-generator/signals')).toHaveLength(2))
  expect(calls.filter(c => c.path === '/api/v1/signal-generator/signals').at(-1)?.body).toMatchObject({ channel_subcarriers: [612, 52], channel_modulations: [64, 64], channel_power_db: [0, 0], pilot_mode: 'explicit', pilot_indices: [-39, 39] })
})

test('generated signal is input-only, with separate exports and an explicit Virtual PA step', async () => {
  const { calls } = setup()
  renderWithProviders(<><SignalGeneratorPage /><Probe /></>)
  await screen.findByTestId('signal-generator-results')
  expect(screen.getByRole('heading', { name: 'PA Input Dataset' })).toBeVisible()
  expect(screen.getAllByText(/complete training dataset needs matching PA input x and PA output y/).length).toBeGreaterThan(0)
  expect(screen.getByRole('button', { name: 'Download PA input CSV' })).toBeEnabled()
  expect(screen.getByRole('button', { name: 'Download input metadata JSON' })).toBeEnabled()
  await userEvent.click(screen.getByRole('link', { name: 'Choose Virtual PA →' }))
  await waitFor(() => expect(screen.getByTestId('location')).toHaveTextContent('/pa-library?input=' + result.signal_id))
  expect(calls.some(c => c.path.includes('publication'))).toBe(false)
  expect(calls.some(c => c.path.endsWith('/dataset'))).toBe(false)
})

test('returning to Signal Generator restores the selected input without generating again', async () => {
  const key = 'opendpd-workflow-v1:generator-restore-test'
  localStorage.setItem(key, JSON.stringify({ version: 1, origin: 'generated', inputId: result.signal_id, inputName: 'Saved input', parameters: {} }))
  try {
    const { calls } = mockApi({
      'GET /api/v1/system/capabilities': () => ({ workspace: 'generator-restore-test', custom_dataset_imports: true }),
      'GET /api/v1/signal-generator/presets': () => fixturePresets,
      ['GET /api/v1/signal-generator/signals/' + result.signal_id]: () => result,
    })
    renderWithProviders(<StudioWorkflowProvider><SignalGeneratorPage /></StudioWorkflowProvider>)
    await screen.findByTestId('signal-generator-results')
    expect(screen.getByRole('link', { name: 'Choose Virtual PA →' })).toHaveAttribute('href', '/pa-library?input=' + result.signal_id)
    expect(calls.some(call => call.method === 'POST')).toBe(false)
  } finally { localStorage.removeItem(key) }
})
