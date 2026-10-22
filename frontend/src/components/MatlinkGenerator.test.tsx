import { act, fireEvent, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import { useLocation } from 'react-router'
import presets from '@mocks/generator_presets.json'
import models from '@mocks/virtual_pa_models.json'
import { SignalGeneratorPage } from '@/pages/SignalGeneratorPage'
import { matlinkKey, type MatlinkSnapshot } from '@/api/matlink'
import { fixtureCatalog, fixtureResolve, nrTmRoutes } from '@/test/nrTestModelFixtures'
import type { NrTmRecipe } from '@/api/nrTestModels'
import { normalizeRecipe } from '@/utils/nrTestModel'
import { mockApi, renderWithProviders } from '@/test/utils'
vi.mock('@/components/PlotlyChart', () => ({ PlotlyChart: () => <div /> }))
beforeEach(() => vi.spyOn(window, 'scrollTo').mockImplementation(() => undefined))
const recipe = ({ scs, bandwidth }: { scs: 15 | 30 | 60; bandwidth: number }): NrTmRecipe => {
  const base = normalizeRecipe(fixtureCatalog().default_recipe)
  return { ...base, carriers: [{ ...base.carriers[0]!, scs_khz: scs, bandwidth_mhz: bandwidth }] }
}
function Probe() { const location = useLocation(); return <output data-testid="location">{location.pathname}{location.search}</output> }
const connection = { client_id: 'matlab-test', label: 'MATLAB', release: '2026a', connected: true, last_seen: '', variables: [], pending_count: 0, capabilities: ['dataset_export'] as ('dataset_export')[] }
const initial: MatlinkSnapshot = { available: true, protocol_version: 1, workspace: '/test', sessions: [connection], transfers: [] }

test('generator pairs signals and waits for MATLAB acknowledgment before opening the selected experiment', async () => {
  const snapshot = { ...initial }
  const { calls } = mockApi({
    'GET /api/v1/matlink': () => snapshot,
    'GET /api/v1/signal-generator/presets': () => presets.data,
    'GET /api/v1/pa-library/models': () => models.data,
    ...nrTmRoutes(),
    'POST /api/v1/signal-generator/batches': () => [{ signal_id: 'sg-' + 'a'.repeat(64), dataset_id: 'sds-' + 'b'.repeat(64), dataset_name: 'syn_pa_in_nr_test', name: presets.data[0]!.preset_id }],
    'POST /api/v1/pa-library/datasets': () => ({ dataset: { dataset_id: 'paired-test', simulation: { simulation_id: 'sim-test' } } }),
    'POST /api/v1/matlink/requests': (_, init) => {
      const body = JSON.parse(String(init.body))
      const transfer = { ...body, request_id: 'transfer-signals', status: 'queued', result: {}, error: null, created_at: '', updated_at: '' }
      snapshot.transfers = [transfer]
      return transfer
    },
  })
  const { client } = renderWithProviders(<><SignalGeneratorPage /><Probe /></>, { route: '/signal-generator?matlink=matlab-test' })
  await waitFor(() => expect(screen.getByRole('button', { name: 'Generate & prepare experiment' })).toBeEnabled(), { timeout: 5000 })
  expect(screen.getByRole('combobox', { name: 'Virtual PA model' })).toHaveTextContent(models.data.find(m => m.model_id === 'rapp-am-pm')!.name.en)
  await userEvent.click(screen.getByRole('button', { name: 'Generate & prepare experiment' }))
  await waitFor(() => expect(calls.some(c => c.path === '/api/v1/matlink/requests')).toBe(true), { timeout: 5000 })
  // The default of the NR family is now a standard test model; its resolved configuration is what is generated.
  expect(calls.find(c => c.path === '/api/v1/signal-generator/batches')?.body).toMatchObject({ configs: [{ preset_id: fixtureCatalog().default_config.preset_id, nr_test_model: { model: 'NR-FR1-TM1.1' } }] })
  expect(calls.find(c => c.path === '/api/v1/pa-library/datasets')?.body).toMatchObject({ model_id: 'rapp-am-pm', input_signal_ids: ['sg-' + 'a'.repeat(64)] })
  expect(screen.getByTestId('location')).toHaveTextContent('/signal-generator?matlink=matlab-test&paired=paired-test')
  expect(calls.find(c => c.path === '/api/v1/matlink/requests')?.body).toMatchObject({ client_id: 'matlab-test', action: 'import_dataset', payload: { dataset_id: 'paired-test' } })
  act(() => client.setQueryData(matlinkKey, { ...snapshot, transfers: [{ ...snapshot.transfers[0], status: 'succeeded', result: { variable: 'opendpdSignals' } }] }))
  await waitFor(() => expect(screen.getByTestId('location')).toHaveTextContent('/experiments/new?dataset=paired-test&version=raw-v1&task=train_pa&matlink=matlab-test&matlab_variable=opendpdSignals'), { timeout: 5000 })
}, 15000)

test('an old toolbox cannot start the automatic generation workflow', async () => {
  mockApi({
    'GET /api/v1/matlink': () => ({ ...initial, sessions: [{ ...connection, capabilities: [] }] }),
    'GET /api/v1/signal-generator/presets': () => presets.data,
    'GET /api/v1/pa-library/models': () => models.data,
    ...nrTmRoutes(),
  })
  renderWithProviders(<SignalGeneratorPage />, { route: '/signal-generator?matlink=matlab-test' })
  await screen.findByText(/This connection needs toolbox 2.4.0/, undefined, { timeout: 5000 })
  expect(screen.getByRole('button', { name: 'Generate & prepare experiment' })).toBeDisabled()
}, 15000)

// ---- the 8,192-sample minimum of the MATLAB flow against test models --------------------------------------------------------------------------

async function pickOption(select: HTMLElement, option: string | RegExp) {
  await userEvent.click(select)
  await userEvent.click(await screen.findByRole('option', { name: option }))
}

test('a standard-length test-model capture always satisfies the 8,192-sample minimum; only a custom capture of a few subframes can fall below it', async () => {
  mockApi({
    'GET /api/v1/matlink': () => initial,
    'GET /api/v1/signal-generator/presets': () => presets.data,
    'GET /api/v1/pa-library/models': () => models.data,
    ...nrTmRoutes(),
  })
  // The shortest standard-length capture of the whole catalogue (1,622 recipes of 11 models, FDD and TDD, automatic and 1x oversampling, checked against the backend)
  // is NR-FR1-TM1.1 at 3 MHz / 15 kHz with 38,400 samples; the default test model has 614,400.
  const defaultSamples = fixtureCatalog().default_config.n_samples
  expect(defaultSamples).toBeGreaterThanOrEqual(8192)
  expect(fixtureResolve(recipe({ scs: 15, bandwidth: 5 })).summary.n_samples).toBe(307_200)
  renderWithProviders(<SignalGeneratorPage />, { route: '/signal-generator?matlink=matlab-test' })
  const generate = () => screen.getByRole('button', { name: 'Generate & prepare experiment' })
  await waitFor(() => expect(generate()).toBeEnabled(), { timeout: 5000 })
  expect(screen.getByTestId('generator-length')).toHaveTextContent('614,400 I/Q · 10.000 ms')
  const carrier = () => within(screen.getByRole('group', { name: 'Carrier 1' }))
  // 5 MHz at 15 kHz without oversampling: 7.68 MS/s, so a single subframe is 7,680 samples.
  await pickOption(carrier().getByRole('combobox', { name: 'Subcarrier spacing (kHz)' }), '15')
  await pickOption(carrier().getByRole('combobox', { name: 'Channel bandwidth (MHz)' }), '5')
  await pickOption(screen.getByRole('combobox', { name: 'Oversampling' }), '1×')
  await waitFor(() => expect(screen.getByTestId('generator-length')).toHaveTextContent('76,800 I/Q · 10.000 ms'), { timeout: 5000 })
  expect(generate()).toBeEnabled()
  await userEvent.click(screen.getByRole('button', { name: 'Custom' }))
  fireEvent.change(screen.getByLabelText('Subframes (1 ms each)'), { target: { value: '1' } })
  await waitFor(() => expect(screen.getByTestId('generator-length')).toHaveTextContent('7,680 I/Q · 1.0000 ms'), { timeout: 5000 })
  expect(screen.getByText('Each training capture needs 8,192–1,000,000 samples. Longer captures may be needed for DPD spectral evaluation.')).toBeVisible()
  expect(generate()).toBeDisabled()
  fireEvent.change(screen.getByLabelText('Subframes (1 ms each)'), { target: { value: '2' } })
  await waitFor(() => expect(screen.getByTestId('generator-length')).toHaveTextContent('15,360 I/Q · 2.0000 ms'), { timeout: 5000 })
  expect(screen.queryByText(/Each training capture needs/)).not.toBeInTheDocument()
  await waitFor(() => expect(generate()).toBeEnabled(), { timeout: 5000 })
}, 20000)
