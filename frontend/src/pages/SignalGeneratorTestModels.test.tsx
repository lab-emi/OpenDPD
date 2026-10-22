import { fireEvent, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import { useLocation } from 'react-router'
import fixture from '@mocks/generator_presets.json'
import type { NrTmRecipe } from '@/api/nrTestModels'
import { normalizeRecipe } from '@/utils/nrTestModel'
import { fixtureCatalog, fixtureResolve, nrTmRoutes } from '@/test/nrTestModelFixtures'
import { mockApi, renderWithProviders, type Handler } from '@/test/utils'
import { StudioWorkflowProvider } from '@/workflow/StudioWorkflow'
import { SignalGeneratorPage } from './SignalGeneratorPage'

vi.mock('@/components/PlotlyChart', () => ({ PlotlyChart: ({ title }: { title: string }) => <div>{title}</div> }))
beforeEach(() => vi.spyOn(window, 'scrollTo').mockImplementation(() => undefined))

const presets = fixture.data
const catalog = fixtureCatalog()
const DEFAULT_ID = catalog.default_config.preset_id
// Ids and names below are the ones the real backend derived for these recipes (frontend/src/test/nrTmFixtureResponses.json), never typed by hand.
const defaultRecipe = normalizeRecipe(catalog.default_recipe)
const recipeWith = (change: Partial<NrTmRecipe> = {}, carrier0: Partial<NrTmRecipe['carriers'][number]> = {}): NrTmRecipe => ({ ...defaultRecipe, ...change, carriers: [{ ...defaultRecipe.carriers[0]!, ...carrier0 }] })
const chip = (recipe: NrTmRecipe) => 'selected-preset-' + fixtureResolve(recipe).config.preset_id
const CATALOG = 'GET /api/v1/signal-generator/nr-test-models'
const wait = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))
function Probe() { const location = useLocation(); return <output data-testid="location">{location.pathname}{location.search}</output> }

const result = (id: string, config: unknown) => ({ signal_id: id, iq_sha256: 'b'.repeat(64), config, coverage: 'standard_test_model', download_url: `/api/v1/signal-generator/signals/${id}/download`,
  analysis: { sample_count: 614400, duration_ms: 10, sample_rate_hz: 61.44e6, subcarrier_spacing_hz: 30000, useful_symbol_us: 33.3333, cp_lengths_samples: [144], complete_symbols: 140, trailing_samples: 0, active_carriers: 612, data_carriers: 612, pilot_carriers: 0,
    rms: .2, peak: .8, papr_db: 11, mean_power_dbfs: -14, occupied_bandwidth_99_hz: 18e6, dc_magnitude: 0, evm_percent: 0, evm_symbols: 16, time_us: [0, 1], time_i: [0, 1], time_q: [0, 1], time_envelope: [0, 1], frequency_mhz: [-1, 1], psd_dbfs_hz: [-80, -80],
    constellation_i: [1], constellation_q: [1], reference_i: [1], reference_q: [1], ccdf_db: [0], ccdf_probability: [.5], allocation: [{ channel: 1, subcarriers: [-1, 1], pilots: [], modulation_order: 4, power_db: 0 }], notes: [] } })

type Routes = Record<string, Handler>
function setup(extra: Routes = {}, { catalogRoute = true }: { catalogRoute?: boolean } = {}) {
  const signals: Record<string, ReturnType<typeof result>> = {}
  const routes: Routes = {
    'GET /api/v1/signal-generator/presets': () => presets,
    'GET /api/v1/system/capabilities': () => ({ custom_dataset_imports: true }),
    'POST /api/v1/signal-generator/batches': (_url, init) => {
      const body = JSON.parse(String(init.body)) as { configs: Array<{ preset_id: string }>; dataset_name: string }
      return body.configs.map((config, i) => {
        const id = 'sg-' + 'abcdef'[i]!.repeat(64)
        signals[id] = result(id, config)
        return { signal_id: id, name: config.preset_id, dataset_id: 'sds-' + 'd'.repeat(64), dataset_name: body.dataset_name }
      })
    },
    'POST /api/v1/signal-generator/validate': (_url, init) => JSON.parse(String(init.body)),
    ...(catalogRoute ? nrTmRoutes() : {}),
  }
  for (const letter of 'abcdef') routes[`GET /api/v1/signal-generator/signals/sg-${letter.repeat(64)}`] = () => signals['sg-' + letter.repeat(64)]
  return mockApi({ ...routes, ...extra })
}
const resolveCalls = (calls: Array<{ method: string; path: string; body: unknown }>) => calls.filter(c => c.method === 'POST' && c.path.endsWith('/resolve'))
const carrier = (n = 1) => within(screen.getByRole('group', { name: `Carrier ${n}` }))
async function pick(select: HTMLElement, option: string | RegExp) {
  await userEvent.click(select)
  await userEvent.click(await screen.findByRole('option', { name: option }))
}
async function ready() {
  await screen.findByTestId('nrtm-panel')
  await waitFor(() => expect(screen.getByTestId('nrtm-derived')).toBeInTheDocument())
}
const generateButton = () => screen.getByRole('button', { name: 'Generate & preview' })

test('the NR family opens on the standard test models with TM1.1 FDD 30 kHz 20 MHz selected and the matrix hidden', async () => {
  const { calls } = setup()
  renderWithProviders(<SignalGeneratorPage />, { route: '/signal-generator' })
  await ready()
  expect(screen.getByRole('button', { name: /01 ·.*Standard test models.*5G NR/ })).toHaveAttribute('aria-pressed', 'true')
  expect(within(screen.getByRole('group', { name: 'Test model' })).getAllByRole('button')).toHaveLength(11)
  expect(screen.getByRole('button', { name: 'TM1.1, QPSK' })).toHaveAttribute('aria-pressed', 'true')
  expect(screen.getByTestId('selected-preset-' + DEFAULT_ID)).toHaveTextContent('TM1.1 · 20 MHz · 30 kHz · FDD')
  expect(DEFAULT_ID).toMatch(/^nr-tm1p1-fdd-30k-20m-c1-[0-9a-f]{6}$/)
  // bandwidth_hz of the resolved configuration is the symmetric extent 612 x 30 kHz + 30 kHz = 18.39 MHz.
  expect(screen.getByRole('textbox', { name: 'PA input dataset name' })).toHaveValue('syn_pa_in_nrtm1p1_bw18p39M_q4_c1_n1')
  expect(screen.getByRole('button', { name: 'Generic numerology matrix…' })).toHaveAttribute('aria-pressed', 'false')
  // The toggle controls a region that exists (hidden) before it is opened, and the selected chips form a named group.
  const toggle = screen.getByRole('button', { name: 'Generic numerology matrix…' })
  expect(document.getElementById(toggle.getAttribute('aria-controls')!)).toHaveAttribute('hidden')
  expect(within(screen.getByRole('group', { name: /^Selected presets/ })).getAllByRole('button')).toHaveLength(1)
  expect(screen.queryByTestId('preset-nr-20')).not.toBeInTheDocument()
  expect(generateButton()).toBeEnabled()
  expect(carrier().getByRole('combobox', { name: 'Subcarrier spacing (kHz)' })).toHaveTextContent('30')
  expect(carrier().getByRole('combobox', { name: 'Channel bandwidth (MHz)' })).toHaveTextContent('20')
  expect(carrier().getByTestId('nrtm-nrb-0')).toHaveTextContent('N_RB 51 = 612 subcarriers')
  expect(screen.getByTestId('nrtm-derived')).toHaveTextContent('61.44 MS/s (oversampling 2×) · 614,400 I/Q samples · 10 ms · 10 subframes')
  // The only server call besides the two reads is the resolution of the default recipe, which fills "Defined by the standard".
  expect(calls.filter(c => c.method === 'POST')).toHaveLength(1)
  expect(resolveCalls(calls)[0]!.body).toMatchObject({ model: 'NR-FR1-TM1.1', duplex: 'fdd', carriers: [{ scs_khz: 30, bandwidth_mhz: 20 }] })
  // "Defined by the standard" is a closed section so that the chips, "Add another" and the matrix button stay on the first screen.
  const standard = within(screen.getByTestId('nrtm-standard'))
  expect(standard.getByRole('button', { name: 'Defined by the standard' })).toHaveAttribute('aria-expanded', 'false')
  await userEvent.click(standard.getByRole('button', { name: 'Defined by the standard' }))
  expect(standard.getByText('PN23 bits (ITU-T O.150), scrambled per TS 38.211 7.3.1.1 / 7.3.2.3, no channel coding')).toBeVisible()
  expect(screen.getByText('Baseband implementation of the 3GPP TS 38.141-1 test-model definition', { exact: false })).toBeVisible()
})

test('the matrix opens behind its button and its cells join the same selection and dataset', async () => {
  setup()
  renderWithProviders(<SignalGeneratorPage />, { route: '/signal-generator' })
  await ready()
  await userEvent.click(screen.getByRole('button', { name: 'Generic numerology matrix…' }))
  expect(screen.getByRole('button', { name: 'Generic numerology matrix…' })).toHaveAttribute('aria-pressed', 'true')
  expect(screen.getByText('The matrix offers generic engineering stimuli with NR numerology. They are not 3GPP test models.')).toBeVisible()
  await userEvent.click(screen.getByTestId('preset-nr-20'))
  expect(screen.getByText('2 / 16 presets selected')).toBeVisible()
  expect(screen.getByTestId('selected-preset-nr-20')).toHaveTextContent('FR1 · 20 MHz · 64-QAM · 1 ch')
  expect(screen.getByRole('textbox', { name: 'PA input dataset name' })).toHaveValue('syn_pa_in_nr-nrtm1p1_bw18p39-20M_q4-64_c1_n2')
  // The active entry is a matrix preset now: the panel offers the models to add instead of editing one.
  expect(screen.getByText(/to add it to the selection/)).toBeVisible()
  expect(screen.queryByRole('group', { name: 'Carrier 1' })).not.toBeInTheDocument()
  await userEvent.click(screen.getByTestId('selected-preset-' + DEFAULT_ID))
  expect(carrier().getByRole('combobox', { name: 'Channel bandwidth (MHz)' })).toHaveTextContent('20')
  await userEvent.click(screen.getByRole('button', { name: 'Generic numerology matrix…' }))
  expect(screen.queryByTestId('preset-nr-20')).not.toBeInTheDocument()
  expect(screen.getByText('2 / 16 presets selected')).toBeVisible()
})

test('choosing a model with nothing selected adds a test model to the selection', async () => {
  setup()
  renderWithProviders(<SignalGeneratorPage />, { route: '/signal-generator' })
  await ready()
  await userEvent.click(screen.getByTestId('remove-preset-' + DEFAULT_ID))
  expect(generateButton()).toBeDisabled()
  await userEvent.click(screen.getByRole('button', { name: 'TM3.1a, 256QAM' }))
  await waitFor(() => expect(screen.getByTestId(chip(recipeWith({ model: 'NR-FR1-TM3.1a' })))).toHaveTextContent('TM3.1a · 20 MHz · 30 kHz · FDD'))
  expect(screen.getByRole('textbox', { name: 'PA input dataset name' })).toHaveValue('syn_pa_in_nrtm3p1a_bw18p39M_q256_c1_n1')
  await waitFor(() => expect(generateButton()).toBeEnabled())
})

test('an edit is resolved 400 ms after it; Generate waits and the chip is re-keyed to the derived preset_id', async () => {
  const { calls } = setup()
  renderWithProviders(<SignalGeneratorPage />, { route: '/signal-generator' })
  await ready()
  expect(resolveCalls(calls)).toHaveLength(1)
  await pick(carrier().getByRole('combobox', { name: 'Channel bandwidth (MHz)' }), '40')
  // Nothing is sent per edit, the old configuration cannot be generated meanwhile, and the panel says it is waiting for the server.
  expect(resolveCalls(calls)).toHaveLength(1)
  expect(generateButton()).toBeDisabled()
  expect(within(screen.getByTestId('nrtm-status')).getByRole('status')).toHaveTextContent('Resolving with the server…')
  await waitFor(() => expect(resolveCalls(calls)).toHaveLength(2))
  expect(resolveCalls(calls)[1]!.body).toMatchObject({ carriers: [{ scs_khz: 30, bandwidth_mhz: 40, n_cell_id: null, gap_before_mhz: 0 }] })
  await waitFor(() => expect(screen.getByTestId(chip(recipeWith({}, { bandwidth_mhz: 40 })))).toHaveTextContent('TM1.1 · 40 MHz · 30 kHz · FDD'))
  expect(screen.queryByTestId('selected-preset-' + DEFAULT_ID)).not.toBeInTheDocument()
  expect(generateButton()).toBeEnabled()
  expect(carrier().getByTestId('nrtm-nrb-0')).toHaveTextContent('N_RB 106 = 1,272 subcarriers')
  expect(screen.getByRole('textbox', { name: 'PA input dataset name' })).toHaveValue('syn_pa_in_nrtm1p1_bw38p19M_q4_c1_n1')
})

/** Holds every resolve request for the given bandwidth until released; aborting a held request rejects it like fetch does. */
function holdResolve(bandwidth: number) {
  const original = globalThis.fetch
  const held: Array<{ signal: AbortSignal; release: () => void }> = []
  vi.stubGlobal('fetch', (input: RequestInfo | URL, init: RequestInit = {}) => {
    const hold = String(input).endsWith('/resolve') && (JSON.parse(String(init.body)) as NrTmRecipe).carriers[0]!.bandwidth_mhz === bandwidth
    if (!hold) return original(input, init)
    return new Promise<Response>((resolve, reject) => {
      const signal = init.signal as AbortSignal
      held.push({ signal, release: () => original(input, init).then(resolve, reject) })
      signal.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')))
    })
  })
  return held
}

test('a request that a newer edit made stale is aborted and its answer is never applied', async () => {
  const { calls } = setup()
  const held = holdResolve(30)
  renderWithProviders(<SignalGeneratorPage />, { route: '/signal-generator' })
  await ready()
  await pick(carrier().getByRole('combobox', { name: 'Channel bandwidth (MHz)' }), '30')
  await waitFor(() => expect(held).toHaveLength(1))
  expect(held[0]!.signal.aborted).toBe(false)
  expect(generateButton()).toBeDisabled()
  await pick(carrier().getByRole('combobox', { name: 'Channel bandwidth (MHz)' }), '50')
  await waitFor(() => expect(held[0]!.signal.aborted).toBe(true))
  await waitFor(() => expect(screen.getByTestId(chip(recipeWith({}, { bandwidth_mhz: 50 })))).toBeInTheDocument())
  held[0]!.release()
  await wait(50)
  expect(screen.getAllByTestId(/^selected-preset-/)).toHaveLength(1)
  expect(screen.queryByTestId(chip(recipeWith({}, { bandwidth_mhz: 30 })))).not.toBeInTheDocument()
  expect(resolveCalls(calls).map(c => (c.body as NrTmRecipe).carriers[0]!.bandwidth_mhz).sort()).toEqual([20, 30, 50])
})

test('typing in the cell ID field does not send a request per keystroke', async () => {
  const { calls } = setup()
  renderWithProviders(<SignalGeneratorPage />, { route: '/signal-generator' })
  await ready()
  const field = carrier().getByLabelText('Cell ID')
  for (const text of ['1', '10', '100', '1000']) fireEvent.change(field, { target: { value: text } })
  await wait(100)
  expect(resolveCalls(calls)).toHaveLength(1)
  await waitFor(() => expect(resolveCalls(calls)).toHaveLength(2))
  expect(resolveCalls(calls)[1]!.body).toMatchObject({ carriers: [{ n_cell_id: 1000 }] })
  await waitFor(() => expect(screen.getByTestId(chip(recipeWith({}, { n_cell_id: 1000 })))).toBeInTheDocument())
  expect(screen.queryByTestId('selected-preset-' + DEFAULT_ID)).not.toBeInTheDocument()
})

test('a refused setup shows the server message inline, blocks Generate, and recovers when it is fixed', async () => {
  const { calls } = setup()
  renderWithProviders(<SignalGeneratorPage />, { route: '/signal-generator' })
  await ready()
  await userEvent.click(screen.getByRole('button', { name: 'Add carrier' }))
  expect(carrier().getByTestId('nrtm-nrb-0')).toBeVisible()
  fireEvent.change(carrier(2).getByLabelText('Gap before (MHz)'), { target: { value: '0.5' } })
  const alert = await screen.findByRole('alert')
  expect(alert).toHaveTextContent('This setup cannot be generated')
  expect(alert).toHaveTextContent('Carrier 2: the gap before it must be a multiple of 0.3 MHz (the least common multiple of the 100 kHz channel raster and the subcarrier spacings).')
  expect(alert).not.toHaveTextContent('Value error')
  expect(generateButton()).toBeDisabled()
  // The entry keeps its last good configuration; the dataset name and chip still describe it.
  expect(screen.getByTestId('selected-preset-' + DEFAULT_ID)).toBeInTheDocument()
  fireEvent.change(carrier(2).getByLabelText('Gap before (MHz)'), { target: { value: '0.6' } })
  await waitFor(() => expect(screen.queryByRole('alert')).not.toBeInTheDocument())
  await waitFor(() => expect(generateButton()).toBeEnabled())
  expect(screen.getByTestId(/^selected-preset-nr-tm1p1-fdd-30k-20m-c2-[0-9a-f]{6}$/)).toHaveTextContent('TM1.1 · 2 × 20 MHz · 30 kHz · FDD')
  expect(resolveCalls(calls).at(-1)!.body).toMatchObject({ carriers: [{ gap_before_mhz: 0 }, { gap_before_mhz: 0.6 }] })
})

test('"Add another test model" appends a second entry with its own derived preset_id and both are generated with their recipes', async () => {
  const { calls } = setup()
  renderWithProviders(<><SignalGeneratorPage /><Probe /></>, { route: '/signal-generator' })
  await ready()
  await userEvent.click(screen.getByRole('button', { name: 'Add another test model' }))
  expect(await screen.findByTestId(chip(recipeWith({ model: 'NR-FR1-TM1.2' })))).toHaveTextContent('TM1.2 · 20 MHz · 30 kHz · FDD')
  expect(screen.getByText('2 / 16 presets selected')).toBeVisible()
  await waitFor(() => expect(generateButton()).toBeEnabled())
  // The new entry is the active one; changing it leaves the first untouched.
  await pick(carrier().getByRole('combobox', { name: 'Subcarrier spacing (kHz)' }), '15')
  const second = recipeWith({ model: 'NR-FR1-TM1.2' }, { scs_khz: 15 })
  await waitFor(() => expect(screen.getByTestId(chip(second))).toBeInTheDocument())
  expect(screen.getByTestId('selected-preset-' + DEFAULT_ID)).toBeInTheDocument()
  // Two families, bandwidth_hz 18.39 and 19.095 MHz (the extent of the 15 kHz carrier), one QPSK order, two spacings.
  expect(screen.getByRole('textbox', { name: 'PA input dataset name' })).toHaveValue('syn_pa_in_nrtm1p1-nrtm1p2_bw18p39-19p095M_q4_c1_s15-30k_n2')
  await waitFor(() => expect(generateButton()).toBeEnabled())
  await userEvent.click(generateButton())
  await screen.findByTestId('signal-generator-results')
  const batch = calls.find(c => c.path.endsWith('/batches'))!.body as { configs: Array<{ preset_id: string; nr_test_model: NrTmRecipe; n_samples: number }> }
  expect(batch.configs.map(c => c.preset_id)).toEqual([DEFAULT_ID, fixtureResolve(second).config.preset_id])
  expect(batch.configs[1]!.nr_test_model).toMatchObject({ model: 'NR-FR1-TM1.2', carriers: [{ scs_khz: 15, bandwidth_mhz: 20 }] })
  expect(batch.configs[0]!.nr_test_model).toMatchObject({ model: 'NR-FR1-TM1.1' })
  expect(screen.getByTestId('location')).toHaveTextContent('/signal-generator/preview')
  // Preview and result chips name the entries after their recipes, not the opaque ids.
  await userEvent.click(screen.getByRole('combobox', { name: 'Visualize a Signal in the Generated Dataset' }))
  expect(screen.getAllByRole('option').map(o => o.textContent)).toEqual(['TM1.1 · 20 MHz · 30 kHz · FDD', 'TM1.2 · 20 MHz · 15 kHz · FDD'])
  expect(screen.getByText('3GPP test model')).toBeVisible()
})

test('Advanced keeps level, seed, RF metadata, burst and impairments for a test model and hides the generic geometry and length controls', async () => {
  setup()
  renderWithProviders(<SignalGeneratorPage />, { route: '/signal-generator' })
  await ready()
  expect(screen.queryByLabelText('Equivalent duration (ms)')).not.toBeInTheDocument()
  expect(screen.queryByRole('button', { name: 'Elapsed time' })).not.toBeInTheDocument()
  expect(screen.getByTestId('generator-length')).toHaveTextContent('614,400 I/Q · 10.000 ms')
  expect(screen.getByRole('checkbox', { name: 'Ideal band-edge filter (not part of the test model)' })).not.toBeChecked()
  await userEvent.click(screen.getByRole('button', { name: 'Advanced parameters' }))
  for (const label of ['Signal RMS', 'Random seed', 'RF carrier frequency (GHz)', 'Constant phase offset (°)', 'Frequency offset (kHz)', 'I/Q gain mismatch (dB)', 'I DC offset']) expect(screen.getByLabelText(label)).toBeVisible()
  for (const label of ['FFT size before oversampling', 'Subcarrier spacing', 'Sample rate (MS/s)', 'Baseband bandwidth (MHz)', 'Payload source']) expect(screen.queryByLabelText(label)).not.toBeInTheDocument()
  expect(screen.queryByRole('button', { name: 'Add OFDMA channel' })).not.toBeInTheDocument()
  expect(screen.queryByRole('heading', { name: 'OFDM / OFDMA' })).not.toBeInTheDocument()
  expect(screen.getByText('The waveform geometry follows the test-model settings above.', { exact: false })).toBeVisible()
})

test('level and filter edits survive a re-derivation of the test model and reach the request', async () => {
  const { calls } = setup()
  renderWithProviders(<SignalGeneratorPage />, { route: '/signal-generator' })
  await ready()
  await userEvent.click(screen.getByRole('checkbox', { name: 'Ideal band-edge filter (not part of the test model)' }))
  await userEvent.click(screen.getByRole('button', { name: 'Advanced parameters' }))
  fireEvent.change(screen.getByLabelText('Signal RMS'), { target: { value: '0.35' } })
  await pick(carrier().getByRole('combobox', { name: 'Channel bandwidth (MHz)' }), '10')
  await waitFor(() => expect(screen.getByTestId(chip(recipeWith({}, { bandwidth_mhz: 10 })))).toBeInTheDocument())
  await waitFor(() => expect(generateButton()).toBeEnabled())
  expect(screen.getByRole('checkbox', { name: 'Ideal band-edge filter (not part of the test model)' })).toBeChecked()
  expect(screen.getByLabelText('Signal RMS')).toHaveValue(0.35)
  await userEvent.click(generateButton())
  await screen.findByTestId('signal-generator-results')
  expect(calls.find(c => c.path.endsWith('/batches'))!.body).toMatchObject({ configs: [{ preset_id: fixtureResolve(recipeWith({}, { bandwidth_mhz: 10 })).config.preset_id, rms: .35, filter_enabled: true, nr_test_model: { carriers: [{ bandwidth_mhz: 10 }] } }] })
})

test('a generated test model enables the exports; changing its recipe disables them until it is generated again', async () => {
  const { calls } = setup()
  renderWithProviders(<SignalGeneratorPage />, { route: '/signal-generator' })
  await ready()
  await userEvent.click(generateButton())
  await screen.findByTestId('signal-generator-results')
  expect(screen.getByRole('button', { name: 'Download PA input CSV' })).toBeEnabled()
  expect(within(screen.getByTestId('signal-generator-results')).getByText('3GPP test model')).toBeVisible()
  await userEvent.click(screen.getByRole('link', { name: 'Edit signal setup' }))
  await pick(carrier().getByRole('combobox', { name: 'Channel bandwidth (MHz)' }), '10')
  await waitFor(() => expect(screen.getByTestId(chip(recipeWith({}, { bandwidth_mhz: 10 })))).toBeInTheDocument())
  await waitFor(() => expect(generateButton()).toBeEnabled())
  await userEvent.click(screen.getByRole('link', { name: 'Open Preview' }))
  expect(screen.getByRole('button', { name: 'Download PA input CSV' })).toBeDisabled()
  await userEvent.click(screen.getByRole('link', { name: 'Edit signal setup' }))
  await userEvent.click(generateButton())
  await waitFor(() => expect(screen.getByRole('button', { name: 'Download PA input CSV' })).toBeEnabled())
  expect(calls.filter(c => c.path.endsWith('/batches'))).toHaveLength(2)
  expect(calls.filter(c => c.path.endsWith('/batches')).at(-1)!.body).toMatchObject({ configs: [{ preset_id: fixtureResolve(recipeWith({}, { bandwidth_mhz: 10 })).config.preset_id, nr_test_model: { carriers: [{ bandwidth_mhz: 10 }] } }] })
})

test('a configuration saved by this release restores the panel from its recipe instead of the default', async () => {
  const recipe: NrTmRecipe = { ...normalizeRecipe(catalog.default_recipe), model: 'NR-FR1-TM3.1a', duplex: 'tdd', carriers: [{ scs_khz: 30, bandwidth_mhz: 100, n_cell_id: null, gap_before_mhz: 0 }, { scs_khz: 30, bandwidth_mhz: 100, n_cell_id: 5, gap_before_mhz: 0.6 }] }
  const config = fixtureResolve(recipe).config
  const key = 'opendpd-workflow-v1:generator-restore-tm'
  localStorage.setItem(key, JSON.stringify({ version: 1, origin: 'generated', inputId: 'sg-' + 'a'.repeat(64), inputIds: ['sg-' + 'a'.repeat(64)], inputConfigs: [config], inputName: 'x', parameters: {} }))
  try {
    const { calls } = setup({ 'GET /api/v1/system/capabilities': () => ({ workspace: 'generator-restore-tm', custom_dataset_imports: true }), ['GET /api/v1/signal-generator/signals/sg-' + 'a'.repeat(64)]: () => result('sg-' + 'a'.repeat(64), config) })
    renderWithProviders(<StudioWorkflowProvider><SignalGeneratorPage /></StudioWorkflowProvider>, { route: '/signal-generator' })
    await ready()
    expect(screen.getByRole('button', { name: 'TM3.1a, 256QAM' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('button', { name: 'TDD' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByTestId('nrtm-carrier-count')).toHaveTextContent('2')
    expect(carrier(1).getByRole('combobox', { name: 'Channel bandwidth (MHz)' })).toHaveTextContent('100')
    expect(carrier(2).getByLabelText('Cell ID')).toHaveValue(5)
    expect(carrier(2).getByLabelText('Gap before (MHz)')).toHaveValue(0.6)
    expect(screen.getAllByTestId(/^selected-preset-/).map(node => node.getAttribute('data-testid'))).toEqual(['selected-preset-' + config.preset_id])
    expect(screen.getByTestId('selected-preset-' + config.preset_id)).toHaveTextContent('TM3.1a · 2 × 100 MHz · 30 kHz · TDD')
    expect(screen.queryByTestId('preset-nr-20')).not.toBeInTheDocument()
    await userEvent.click(within(screen.getByTestId('nrtm-standard')).getByRole('button', { name: 'Defined by the standard' }))
    expect(screen.getByRole('list', { name: 'Slots of one TDD period' })).toBeVisible()
    expect(resolveCalls(calls)[0]!.body).toEqual(recipe)
    expect(screen.getByRole('link', { name: 'Open Preview' })).toBeVisible()
  } finally { localStorage.removeItem(key) }
})

test.each([['without the key', undefined], ['with a null recipe', null]])('a 2.4.0 configuration %s keeps working: the matrix shows it, the panel offers models to add', async (_name, value) => {
  const matrix = { ...presets[0]!.config, ...(value === null ? { nr_test_model: null } : {}) }
  const key = 'opendpd-workflow-v1:generator-restore-240'
  localStorage.setItem(key, JSON.stringify({ version: 1, origin: 'generated', inputId: 'sg-' + 'a'.repeat(64), inputIds: ['sg-' + 'a'.repeat(64)], inputConfigs: [matrix], inputName: 'x', parameters: {} }))
  try {
    const { calls } = setup({ 'GET /api/v1/system/capabilities': () => ({ workspace: 'generator-restore-240', custom_dataset_imports: true }), ['GET /api/v1/signal-generator/signals/sg-' + 'a'.repeat(64)]: () => ({ ...result('sg-' + 'a'.repeat(64), matrix), coverage: 'numerology' }) })
    renderWithProviders(<StudioWorkflowProvider><SignalGeneratorPage /></StudioWorkflowProvider>, { route: '/signal-generator' })
    await screen.findByTestId('nrtm-panel')
    expect(screen.getByTestId('selected-preset-nr-20')).toHaveTextContent('FR1 · 20 MHz · 64-QAM · 1 ch')
    expect(screen.getByTestId('preset-nr-20')).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('button', { name: 'Generic numerology matrix…' })).toHaveAttribute('aria-pressed', 'true')
    expect(screen.getByRole('textbox', { name: 'PA input dataset name' })).toHaveValue('syn_pa_in_nr_bw20M_q64_c1_n1')
    expect(screen.getByText(/to add it to the selection/)).toBeVisible()
    expect(generateButton()).toBeEnabled()
    expect(resolveCalls(calls)).toHaveLength(0)
    // The generic advanced controls are all there for a matrix preset.
    await userEvent.click(screen.getByRole('button', { name: 'Advanced parameters' }))
    expect(screen.getByRole('button', { name: 'Add OFDMA channel' })).toBeVisible()
    expect(screen.getByLabelText('FFT size before oversampling')).toBeVisible()
  } finally { localStorage.removeItem(key) }
})

test.each([[403, 'feature_unavailable'], [404, 'not_found'], [500, 'internal_error']])('when the catalogue answers %i the generator keeps the old matrix behaviour', async (status, code) => {
  const { calls } = setup({ [CATALOG]: () => ({ status, body: { error: { code, message: 'This endpoint is not available', details: [], hint: null } } }) })
  renderWithProviders(<SignalGeneratorPage />, { route: '/signal-generator' })
  await screen.findByRole('heading', { name: 'Signal setup' })
  expect(screen.queryByTestId('nrtm-panel')).not.toBeInTheDocument()
  expect(screen.queryByRole('button', { name: 'Generic numerology matrix…' })).not.toBeInTheDocument()
  expect(screen.getByTestId('preset-nr-20')).toHaveAttribute('aria-pressed', 'true')
  expect(screen.getByRole('button', { name: /01 ·.*FR1 \/ FR2 matrix.*5G NR/ })).toBeVisible()
  expect(screen.getByRole('textbox', { name: 'PA input dataset name' })).toHaveValue('syn_pa_in_nr_bw20M_q64_c1_n1')
  expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  expect(generateButton()).toBeEnabled()
  expect(calls.filter(c => c.method === 'POST')).toHaveLength(0)
  expect(calls.filter(c => c.path === '/api/v1/signal-generator/nr-test-models')).toHaveLength(1)
})

test('loading a saved test-model configuration selects the NR family and the entry, and a failed catalogue still lets it be generated', async () => {
  const recipe: NrTmRecipe = { ...normalizeRecipe(catalog.default_recipe), model: 'NR-FR1-TM2', carriers: [{ scs_khz: 15, bandwidth_mhz: 10, n_cell_id: null, gap_before_mhz: 0 }] }
  const loaded = fixtureResolve(recipe).config
  const { calls } = setup({ 'POST /api/v1/signal-generator/validate': () => loaded })
  renderWithProviders(<SignalGeneratorPage />, { route: '/signal-generator' })
  await ready()
  fireEvent.click(screen.getByRole('button', { name: /02 ·.*Wi-Fi 6/ }))
  const file = new File([JSON.stringify(loaded)], 'signal-config.json', { type: 'application/json' })
  await userEvent.upload(screen.getByTestId('generator-config-upload'), file)
  await waitFor(() => expect(screen.getByTestId('selected-preset-' + loaded.preset_id)).toHaveTextContent('TM2 · 10 MHz · 15 kHz · FDD'))
  expect(screen.getByRole('button', { name: /01 ·.*5G NR/ })).toHaveAttribute('aria-pressed', 'true')
  expect(screen.getByRole('button', { name: 'TM2, 64QAM' })).toHaveAttribute('aria-pressed', 'true')
  expect(carrier().getByRole('combobox', { name: 'Channel bandwidth (MHz)' })).toHaveTextContent('10')
  expect(calls.filter(c => c.path.endsWith('/validate'))).toHaveLength(1)
})

test('the web build is unaffected: nothing is sent to the resolve endpoint before a test model is on screen', async () => {
  const { calls } = setup({}, { catalogRoute: false })
  renderWithProviders(<SignalGeneratorPage />, { route: '/signal-generator' })
  await screen.findByRole('heading', { name: 'Signal setup' })
  expect(calls.some(c => c.path.includes('nr-test-models/resolve'))).toBe(false)
})

// ---- the batch limit and equal recipes ---------------------------------------------------------------------------------------------------------

/** Restores a saved selection of test-model configurations from the workflow state, as when the user returns to this tab. */
async function openWith(configs: Array<ReturnType<typeof fixtureResolve>['config']>, scope: string, extra: Routes = {}, route = '/signal-generator') {
  const ids = configs.map((_, i) => 'sg-' + 'abcdef'[i]!.repeat(64))
  const key = 'opendpd-workflow-v1:' + scope
  localStorage.setItem(key, JSON.stringify({ version: 1, origin: 'generated', inputId: ids[0], inputIds: ids, inputConfigs: configs, inputName: 'x', parameters: {} }))
  const routes: Routes = { 'GET /api/v1/system/capabilities': () => ({ workspace: scope, custom_dataset_imports: true }), ...extra }
  configs.forEach((config, i) => { routes['GET /api/v1/signal-generator/signals/' + ids[i]] = () => result(ids[i]!, config) })
  const api = setup(routes)
  renderWithProviders(<StudioWorkflowProvider><SignalGeneratorPage /></StudioWorkflowProvider>, { route })
  return { ...api, cleanup: () => localStorage.removeItem(key) }
}

test('five full-length 100 MHz captures exceed one dataset: the page names the numbers before any request, keeps Generate off and recovers when one is removed', async () => {
  const models = ['NR-FR1-TM3.1a', 'NR-FR1-TM3.1', 'NR-FR1-TM3.1b', 'NR-FR1-TM1.1', 'NR-FR1-TM2'] as const
  const configs = models.map(model => fixtureResolve(recipeWith({ model }, { bandwidth_mhz: 100 })).config)
  // A full-length capture is 0.6-1.0 M samples: at 122.88 MS/s the 10 ms of the standard would need 1.23 M, so the server shortens it to 8 ms = 983,040.
  expect(configs.map(c => c.n_samples)).toEqual([983_040, 983_040, 983_040, 983_040, 983_040])
  const { calls, cleanup } = await openWith(configs, 'generator-batch-limit')
  try {
    await ready()
    expect(screen.getByText('5 / 16 presets selected')).toBeVisible()
    expect(screen.getByText('4,915,200 I/Q samples across selected presets')).toBeVisible()
    expect(screen.getByTestId('generator-batch-limit')).toHaveTextContent('The 5 selected signals hold 4,915,200 I/Q samples together, but one dataset is limited to 4,000,000. Remove a signal or shorten a capture (for a test model: fewer subframes or a lower oversampling), then generate again.')
    expect(generateButton()).toBeDisabled()
    expect(calls.filter(c => c.path.endsWith('/batches'))).toHaveLength(0)
    await userEvent.click(screen.getByTestId('remove-preset-' + configs[4]!.preset_id))
    await waitFor(() => expect(screen.queryByTestId('generator-batch-limit')).not.toBeInTheDocument())
    expect(screen.getByText('3,932,160 I/Q samples across selected presets')).toBeVisible()
    expect(generateButton()).toBeEnabled()
  } finally { cleanup() }
})

test('a signal that the server refuses for size is explained with the numbers of the selection, not with its generic text', async () => {
  const refuse = (status: number, code: string, message: string, details: Array<{ field: string; message: string }> = []) => ({ 'POST /api/v1/signal-generator/batches': () => ({ status, body: { error: { code, message, details, hint: null } } }) })
  // Texts and shapes are the ones captured from the local API (422) and written in opendpd/web/mutation_policy.py (hosted 413).
  setup(refuse(413, 'workspace_limit', 'This operation and its exports would exceed temporary storage. Use fewer samples.'))
  renderWithProviders(<SignalGeneratorPage />, { route: '/signal-generator' })
  await ready()
  await userEvent.click(generateButton())
  const alert = await screen.findByRole('alert')
  expect(alert).toHaveTextContent('This server’s temporary workspace cannot hold 614,400 I/Q samples together with their exports. Remove a signal or shorten a capture (test model: fewer subframes), then generate again.')
  expect(alert).not.toHaveTextContent(/workspace_limit|exceed temporary storage/)
})

test('the 422 of the local server for too many samples reads as the same readable message', async () => {
  setup({ 'POST /api/v1/signal-generator/batches': () => ({ status: 422, body: { error: { code: 'invalid_request', message: 'request validation failed', hint: null,
    details: [{ field: '', message: 'Value error, Select at most 4,000,000 samples across all presets.' }] } } }) })
  renderWithProviders(<SignalGeneratorPage />, { route: '/signal-generator' })
  await ready()
  await userEvent.click(generateButton())
  expect(await screen.findByRole('alert')).toHaveTextContent('The 1 selected signals hold 614,400 I/Q samples together, but one dataset is limited to 4,000,000.')
})

test('the same recipe cannot be in the selection twice: the second entry is held with an explanation until it differs or the first is removed', async () => {
  setup()
  renderWithProviders(<SignalGeneratorPage />, { route: '/signal-generator' })
  await ready()
  await userEvent.click(screen.getByRole('button', { name: 'Add another test model' }))
  const second = chip(recipeWith({ model: 'NR-FR1-TM1.2' }))
  expect(await screen.findByTestId(second)).toBeInTheDocument()
  await waitFor(() => expect(generateButton()).toBeEnabled())
  // Back to TM1.1 with the same bandwidth and spacing: the server would derive the id of the first entry.
  await userEvent.click(screen.getByRole('button', { name: 'TM1.1, QPSK' }))
  const warning = await screen.findByTestId('nrtm-duplicate')
  expect(warning).toHaveTextContent('“TM1.1 · 20 MHz · 30 kHz · FDD” is already in the selection with exactly these settings. A dataset holds each signal once: change a setting here or remove one of the two entries.')
  expect(screen.getByText('2 / 16 presets selected')).toBeVisible()
  expect(screen.getAllByTestId(/^selected-preset-/)).toHaveLength(2)
  expect(screen.getByTestId(second)).toHaveTextContent('TM1.2')
  expect(generateButton()).toBeDisabled()
  expect(within(screen.getByTestId('nrtm-status')).queryByRole('progressbar')).not.toBeInTheDocument()
  // Changing something makes it a different signal again.
  await pick(carrier().getByRole('combobox', { name: 'Channel bandwidth (MHz)' }), '10')
  await waitFor(() => expect(screen.queryByTestId('nrtm-duplicate')).not.toBeInTheDocument())
  await waitFor(() => expect(screen.getByTestId(chip(recipeWith({}, { bandwidth_mhz: 10 })))).toBeInTheDocument())
  await waitFor(() => expect(generateButton()).toBeEnabled())
  // Edit it back into a duplicate, then remove the first entry: the held entry takes the id and Generate is possible again.
  await pick(carrier().getByRole('combobox', { name: 'Channel bandwidth (MHz)' }), '20')
  await screen.findByTestId('nrtm-duplicate')
  await userEvent.click(screen.getByTestId('remove-preset-' + DEFAULT_ID))
  await waitFor(() => expect(screen.queryByTestId('nrtm-duplicate')).not.toBeInTheDocument())
  expect(screen.getAllByTestId(/^selected-preset-/)).toHaveLength(1)
  expect(screen.getByTestId('selected-preset-' + DEFAULT_ID)).toHaveTextContent('TM1.1 · 20 MHz · 30 kHz · FDD')
  await waitFor(() => expect(generateButton()).toBeEnabled())
})
