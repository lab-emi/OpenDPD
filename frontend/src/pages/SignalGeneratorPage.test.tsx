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
  const signals: Record<string, typeof result> = {}
  const routes: Parameters<typeof mockApi>[0] = {
    'GET /api/v1/signal-generator/presets': () => fixturePresets,
    'GET /api/v1/system/capabilities': () => ({ custom_dataset_imports: true }),
    'POST /api/v1/signal-generator/batches': (_url, init) => {
      const configs = JSON.parse(String(init.body)).configs as typeof config[]
      return configs.map((c, i) => {
        const id = 'sg-' + (i === 0 ? 'a' : 'b').repeat(64)
        signals[id] = { ...result, signal_id: id, config: c }
        return { signal_id: id, name: c.preset_id }
      })
    },
    'POST /api/v1/signal-generator/validate': (_url, init) => JSON.parse(String(init.body)),
  }
  for (const letter of ['a', 'b']) {
    const id = 'sg-' + letter.repeat(64)
    routes['GET /api/v1/signal-generator/signals/' + id] = () => signals[id]
  }
  return mockApi(routes)
}

function Probe() { const location = useLocation(); return <output data-testid="location">{location.pathname}{location.search}</output> }

test('matrix selection keeps different preset lengths and disables stale exports', async () => {
  const { calls } = setup()
  renderWithProviders(<SignalGeneratorPage />)
  expect(calls.filter(c => c.method === 'POST')).toHaveLength(0)
  await screen.findByRole('heading', { name: 'Signal setup' })
  expect(screen.queryByRole('combobox', { name: 'Preset' })).not.toBeInTheDocument()
  expect(screen.queryByRole('button', { name: /Wi-Fi 8/ })).not.toBeInTheDocument()
  expect(screen.getByRole('checkbox', { name: 'Ideal PA-input filter (recommended)' })).toBeChecked()
  await userEvent.click(screen.getByRole('button', { name: 'Generate & preview' }))
  await screen.findByTestId('signal-generator-results')
  expect(screen.getByRole('button', { name: 'Download PA input CSV' })).toBeEnabled()
  await userEvent.click(screen.getByRole('button', { name: /02 ·.*Wi-Fi 6/ }))
  await userEvent.click(screen.getByTestId('preset-wifi6-20'))
  expect(screen.getByRole('button', { name: 'Download PA input CSV' })).toBeDisabled()
  await userEvent.click(screen.getByRole('button', { name: 'Elapsed time' }))
  fireEvent.change(screen.getByLabelText('Equivalent duration (ms)'), { target: { value: '.25' } })
  expect(screen.getByTestId('generator-length')).toHaveTextContent('20,000 I/Q')
  await userEvent.click(screen.getByRole('button', { name: 'Generate & preview' }))
  await waitFor(() => expect(screen.getByRole('button', { name: 'Download PA input CSV' })).toBeEnabled())
  expect(calls.filter(c => c.path === '/api/v1/signal-generator/batches').at(-1)?.body).toMatchObject({ configs: [
    { preset_id: 'nr-20', n_samples: 30720, filter_enabled: true },
    { preset_id: 'wifi6-20', length_mode: 'duration', duration_ms: .25, filter_enabled: true },
  ] })
})

test('advanced OFDMA channels and pilots reach the generator request', async () => {
  const { calls } = setup()
  renderWithProviders(<SignalGeneratorPage />)
  expect(calls.filter(c => c.method === 'POST')).toHaveLength(0)
  await userEvent.click(await screen.findByRole('button', { name: 'Generate & preview' }))
  await screen.findByTestId('signal-generator-results')
  await userEvent.click(screen.getByRole('button', { name: 'Advanced parameters' }))
  await userEvent.click(screen.getByRole('checkbox', { name: 'Use the same settings for all channels' }))
  await userEvent.click(screen.getByRole('button', { name: 'Add OFDMA channel' }))
  expect(screen.getAllByLabelText('Subcarriers including pilots')).toHaveLength(2)
  fireEvent.change(screen.getAllByLabelText('Subcarriers including pilots')[1]!, { target: { value: '52' } })
  await userEvent.click(screen.getByRole('combobox', { name: 'Pilot allocation' }))
  await userEvent.click(screen.getByRole('option', { name: 'Explicit signed bin indices' }))
  fireEvent.change(screen.getByLabelText('Pilot carrier indices'), { target: { value: '-39, 39' } })
  await userEvent.click(screen.getByRole('button', { name: 'Apply & regenerate' }))
  await waitFor(() => expect(calls.filter(c => c.path === '/api/v1/signal-generator/batches')).toHaveLength(2))
  expect(calls.filter(c => c.path === '/api/v1/signal-generator/batches').at(-1)?.body).toMatchObject({ configs: [{ channel_subcarriers: [612, 52], channel_modulations: [64, 64], channel_power_db: [0, 0], pilot_mode: 'explicit', pilot_indices: [-39, 39] }] })
})

test('generated signal is input-only, with separate exports and an explicit Virtual PA step', async () => {
  const { calls } = setup()
  renderWithProviders(<><SignalGeneratorPage /><Probe /></>)
  await userEvent.click(await screen.findByRole('button', { name: 'Generate & preview' }))
  await screen.findByTestId('signal-generator-results')
  expect(screen.getByRole('heading', { name: 'PA Input Dataset' })).toBeVisible()
  expect(screen.getByText(/one input\/output CSV per preset/)).toBeVisible()
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
