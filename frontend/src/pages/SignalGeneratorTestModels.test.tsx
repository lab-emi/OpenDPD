import { fireEvent, screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { act } from 'react'
import { focusManager } from '@tanstack/react-query'
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
  // The wording of the server's text is the backend's; the field is named as the form names it ("Carrier 2 · Gap before") or by the server ("Carrier 2: ...").
  expect(alert).toHaveTextContent(/Carrier 2.*multiple of 0\.3 MHz/)
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

test.each([[403, 'feature_unavailable'], [404, 'not_found']])('when the catalogue answers %i the generator keeps the old matrix behaviour without a notice', async (status, code) => {
  const { calls } = setup({ [CATALOG]: () => ({ status, body: { error: { code, message: 'This endpoint is not available', details: [], hint: null } } }) })
  renderWithProviders(<SignalGeneratorPage />, { route: '/signal-generator' })
  await screen.findByRole('heading', { name: 'Signal setup' })
  expect(screen.queryByTestId('nrtm-panel')).not.toBeInTheDocument()
  expect(screen.queryByRole('button', { name: 'Generic numerology matrix…' })).not.toBeInTheDocument()
  expect(screen.getByTestId('preset-nr-20')).toHaveAttribute('aria-pressed', 'true')
  expect(screen.getByRole('button', { name: /01 ·.*FR1 \/ FR2 matrix.*5G NR/ })).toBeVisible()
  expect(screen.getByRole('textbox', { name: 'PA input dataset name' })).toHaveValue('syn_pa_in_nr_bw20M_q64_c1_n1')
  expect(screen.queryByRole('alert')).not.toBeInTheDocument()
  expect(screen.queryByTestId('nrtm-catalog-problem')).not.toBeInTheDocument()
  expect(generateButton()).toBeEnabled()
  expect(calls.filter(c => c.method === 'POST')).toHaveLength(0)
  expect(calls.filter(c => c.path === '/api/v1/signal-generator/nr-test-models')).toHaveLength(1)
})

const REFUSAL_500 = { status: 500, body: { error: { code: 'internal_error', message: 'Something broke', details: [], hint: null } } }

test('a catalogue that fails for another reason keeps the matrix, says so in one line with a Retry, and Retry brings the test models in without losing the selection', async () => {
  let attempts = 0
  const { calls } = setup({ [CATALOG]: () => ++attempts === 1 ? REFUSAL_500 : fixtureCatalog() })
  renderWithProviders(<SignalGeneratorPage />, { route: '/signal-generator' })
  const notice = await screen.findByTestId('nrtm-catalog-problem')
  expect(notice).toHaveTextContent('The standard test models could not be loaded, so the generic numerology matrix is shown instead.')
  expect(within(notice).getByRole('button', { name: 'Retry' })).toBeEnabled()
  // The matrix underneath works as before.
  expect(screen.getByTestId('preset-nr-20')).toHaveAttribute('aria-pressed', 'true')
  await userEvent.click(screen.getByTestId('preset-nr-fr1-30-20-q16-c1'))
  expect(screen.getByText('2 / 16 presets selected')).toBeVisible()
  await userEvent.click(within(notice).getByRole('button', { name: 'Retry' }))
  await screen.findByTestId('nrtm-panel')
  expect(screen.queryByTestId('nrtm-catalog-problem')).not.toBeInTheDocument()
  // The page was not replaced by a loading state: both matrix presets are still selected, and the panel offers the models to add.
  expect(screen.getByText('2 / 16 presets selected')).toBeVisible()
  expect(screen.getByTestId('selected-preset-nr-20')).toBeInTheDocument()
  expect(screen.getByText(/to add it to the selection/)).toBeVisible()
  expect(calls.filter(c => c.path === '/api/v1/signal-generator/nr-test-models')).toHaveLength(2)
})

test.each([
  ['an object without models', { models: [] }],
  ['a page that is not a catalogue', { version: 'nr-test-model-catalog-v1', models: [{ model: 'NR-FR1-TM1.1' }] }],
  ['a bare list', []],
])('%s is not a catalogue: the generator does not crash, keeps the matrix and offers a Retry', async (_name, body) => {
  setup({ [CATALOG]: () => body })
  renderWithProviders(<SignalGeneratorPage />, { route: '/signal-generator' })
  await screen.findByRole('heading', { name: 'Signal setup' })
  expect(screen.getByTestId('nrtm-catalog-problem')).toBeVisible()
  expect(screen.queryByTestId('nrtm-panel')).not.toBeInTheDocument()
  expect(screen.getByTestId('preset-nr-20')).toBeVisible()
  expect(generateButton()).toBeEnabled()
})

test('a 500 for the catalogue keeps the notice with its Retry; the failed retry keeps it too', async () => {
  const { calls } = setup({ [CATALOG]: () => REFUSAL_500 })
  renderWithProviders(<SignalGeneratorPage />, { route: '/signal-generator' })
  await userEvent.click(within(await screen.findByTestId('nrtm-catalog-problem')).getByRole('button', { name: 'Retry' }))
  await waitFor(() => expect(calls.filter(c => c.path === '/api/v1/signal-generator/nr-test-models')).toHaveLength(2))
  await waitFor(() => expect(within(screen.getByTestId('nrtm-catalog-problem')).getByRole('button', { name: 'Retry' })).toBeEnabled())
  expect(screen.getByRole('heading', { name: 'Signal setup' })).toBeVisible()
  expect(screen.getByTestId('preset-nr-20')).toBeVisible()
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

// ---- review round: what the user sees when edits are pending, the limit, the labels, the matrix ---------------------------------------------

const REFUSED_CELL = { error: { code: 'invalid_request', message: 'request validation failed', hint: null, details: [{ field: 'carriers.0.n_cell_id', message: 'Input should be less than or equal to 1007' }] } }
/** The real resolve answers, except that a cell ID above 1007 is refused with the envelope the backend sent for it (captured in the review round). */
function refuseCell5000(): Routes {
  const resolve = nrTmRoutes()['POST /api/v1/signal-generator/nr-test-models/resolve']!
  return { 'POST /api/v1/signal-generator/nr-test-models/resolve': (url, init) => (JSON.parse(String(init.body)) as NrTmRecipe).carriers[0]!.n_cell_id === 5000 ? { status: 422, body: REFUSED_CELL } : resolve(url, init) }
}
const chips = () => screen.getAllByTestId(/^selected-preset-/)

test('an entry with edits the server has not accepted is marked, named next to Generate when it is not on screen, and one click shows it', async () => {
  setup(refuseCell5000())
  renderWithProviders(<SignalGeneratorPage />, { route: '/signal-generator' })
  await ready()
  await userEvent.click(screen.getByRole('button', { name: 'Add another test model' }))
  const second = await screen.findByTestId(chip(recipeWith({ model: 'NR-FR1-TM1.2' })))
  await waitFor(() => expect(generateButton()).toBeEnabled())
  fireEvent.change(carrier().getByLabelText('Cell ID'), { target: { value: '5000' } })
  expect(await screen.findByRole('alert')).toHaveTextContent('Carrier 1 · Cell ID: Input should be less than or equal to 1007')
  expect(generateButton()).toBeDisabled()
  // On screen, the panel says it; no second line next to Generate.
  expect(screen.queryByTestId('generator-blocked-entry')).not.toBeInTheDocument()
  expect(second).toHaveAttribute('data-draft', 'true')
  expect(second).toHaveAttribute('data-cause', 'error')
  expect(within(second).getByTitle('The server refused these settings')).toBeInTheDocument()
  // Another chip: the panel shows the other entry, and Generate says why it is off.
  await userEvent.click(screen.getByTestId('selected-preset-' + DEFAULT_ID))
  const line = screen.getByTestId('generator-blocked-entry')
  expect(line).toHaveTextContent('The server refused the settings of “TM1.2 · 20 MHz · 30 kHz · FDD”, so Generate is off. Correct them or remove the entry.')
  expect(generateButton()).toBeDisabled()
  expect(screen.getByTestId('selected-preset-' + DEFAULT_ID)).not.toHaveAttribute('data-draft')
  // Another family tab: still named.
  await userEvent.click(screen.getByRole('button', { name: /02 ·.*Wi-Fi 6/ }))
  expect(screen.getByTestId('generator-blocked-entry')).toBeVisible()
  expect(screen.queryByTestId('nrtm-panel')).not.toBeInTheDocument()
  // The button brings the NR panel and the entry back, with the refusal.
  await userEvent.click(within(screen.getByTestId('generator-blocked-entry')).getByRole('button', { name: 'Show it' }))
  expect(await screen.findByTestId('nrtm-panel')).toBeVisible()
  expect(screen.getByRole('button', { name: /01 ·.*5G NR/ })).toHaveAttribute('aria-pressed', 'true')
  expect(carrier().getByLabelText('Cell ID')).toHaveValue(5000)
  expect(screen.getByRole('alert')).toHaveTextContent('Carrier 1 · Cell ID')
  expect(screen.queryByTestId('generator-blocked-entry')).not.toBeInTheDocument()
  // The button that was activated is gone; the keyboard lands on the panel's heading instead of the top of the page.
  expect(screen.getByRole('heading', { name: 'Standard 5G NR test model' })).toHaveFocus()
  // Fixing the entry clears the marker.
  fireEvent.change(carrier().getByLabelText('Cell ID'), { target: { value: '' } })
  await waitFor(() => expect(generateButton()).toBeEnabled())
  expect(chips().filter(node => node.getAttribute('data-draft'))).toHaveLength(0)
})

test('Save configuration is off while the highlighted entry has edits the server has not accepted, and writes the accepted recipe afterwards', async () => {
  setup(refuseCell5000())
  const blobs: Blob[] = []
  vi.stubGlobal('URL', Object.assign(URL, { createObjectURL: (blob: Blob) => { blobs.push(blob); return 'blob:export' }, revokeObjectURL: () => undefined }))
  vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => undefined)
  renderWithProviders(<SignalGeneratorPage />, { route: '/signal-generator' })
  await ready()
  const save = () => screen.getByRole('button', { name: 'Save configuration' })
  expect(save()).toBeEnabled()
  // A pending edit (before the server answered) and a refused one both keep the previous recipe from being saved as if it were the one on screen.
  fireEvent.change(carrier().getByLabelText('Cell ID'), { target: { value: '5000' } })
  expect(save()).toBeDisabled()
  expect(save()).toHaveAccessibleDescription(/are not part of the signal yet.*saving now would write the previous ones/)
  await screen.findByRole('alert')
  expect(save()).toBeDisabled()
  fireEvent.change(carrier().getByLabelText('Cell ID'), { target: { value: '1000' } })
  await waitFor(() => expect(save()).toBeEnabled())
  await userEvent.click(save())
  const saved = JSON.parse(await blobs.at(-1)!.text()) as { nr_test_model: NrTmRecipe }
  expect(saved.nr_test_model.carriers[0]!.n_cell_id).toBe(1000)
})

test('with 16 signals selected and a matrix preset highlighted, the model cards cannot add a 17th: they are off and the limit is explained once', async () => {
  const { calls } = setup()
  renderWithProviders(<SignalGeneratorPage />, { route: '/signal-generator' })
  await ready()
  await userEvent.click(screen.getByRole('button', { name: 'Generic numerology matrix…' }))
  for (const id of ['nr-20', 'nr-fr1-30-20-q16-c1', 'nr-fr1-30-20-q64-c2', 'nr-fr1-30-100-q16-c1', 'nr-100', 'nr-fr1-30-100-q64-c2']) await userEvent.click(screen.getByTestId('preset-' + id))
  await userEvent.click(screen.getByRole('button', { name: 'FR2-1 · 120 kHz' }))
  await userEvent.click(screen.getByTestId('preset-nr-fr2'))
  for (const [tab, ids] of [[/02 ·.*Wi-Fi 6/, ['wifi6-20', 'wifi6-80']], [/03 ·.*Wi-Fi 7/, ['wifi7-20', 'wifi7-320']]] as const) {
    await userEvent.click(screen.getByRole('button', { name: tab }))
    for (const id of ids) await userEvent.click(screen.getByTestId('preset-' + id))
  }
  await userEvent.click(screen.getByRole('button', { name: /04 ·.*Custom/ }))
  for (const label of ['Custom OFDM / OFDMA', 'Single-carrier QAM / PSK', 'Single tone', 'Multitone']) await userEvent.click(screen.getByRole('button', { name: label }))
  expect(screen.getByText('16 / 16 presets selected')).toBeVisible()
  await userEvent.click(screen.getByRole('button', { name: /01 ·.*5G NR/ }))
  const cards = within(screen.getByRole('group', { name: 'Test model' })).getAllByRole('button')
  for (const card of cards) expect(card).toBeDisabled()
  expect(screen.getAllByText(/^Select up to 16 presets and 4,000,000 total I\/Q samples\./)).toHaveLength(1)
  expect(screen.getByTestId('nrtm-selection-full')).toBeVisible()
  fireEvent.click(cards[2]!)
  expect(screen.getByText('16 / 16 presets selected')).toBeVisible()
  expect(chips()).toHaveLength(16)
  // Highlighting the test model again turns the cards into editors, and the limit is then stated by the page.
  await userEvent.click(screen.getByTestId('selected-preset-' + DEFAULT_ID))
  expect(within(screen.getByRole('group', { name: 'Test model' })).getAllByRole('button').every(card => !card.hasAttribute('disabled'))).toBe(true)
  expect(screen.queryByTestId('nrtm-selection-full')).not.toBeInTheDocument()
  expect(screen.getAllByText(/^Select up to 16 presets and 4,000,000 total I\/Q samples\./)).toHaveLength(1)
  expect(screen.getByRole('button', { name: 'Add another test model' })).toBeDisabled()
  expect(calls.filter(c => c.method === 'POST' && c.path.endsWith('/batches'))).toHaveLength(0)
}, 30000)

test('entries that would read the same are told apart in the chips, the preview list and the duplicate warning', async () => {
  setup()
  renderWithProviders(<SignalGeneratorPage />, { route: '/signal-generator' })
  await ready()
  await userEvent.click(screen.getByRole('button', { name: 'Add another test model' }))
  await screen.findByTestId(chip(recipeWith({ model: 'NR-FR1-TM1.2' })))
  await userEvent.click(screen.getByRole('button', { name: 'TM1.1, QPSK' }))
  // Held as a duplicate of the first entry, whose name is still the plain one.
  expect(await screen.findByTestId('nrtm-duplicate')).toHaveTextContent('“TM1.1 · 20 MHz · 30 kHz · FDD” is already in the selection')
  // A custom length of 10 subframes is another signal with the same base label.
  await userEvent.click(screen.getByRole('button', { name: 'Custom' }))
  const second = chip(recipeWith({ subframes: 10 }))
  await waitFor(() => expect(screen.getByTestId(second)).toBeInTheDocument())
  expect(chips().map(node => node.querySelector('.MuiChip-label')?.textContent)).toEqual(['TM1.1 · 20 MHz · 30 kHz · FDD', 'TM1.1 · 20 MHz · 30 kHz · FDD · 10 subframes'])
  await waitFor(() => expect(generateButton()).toBeEnabled())
  await userEvent.click(generateButton())
  await screen.findByTestId('signal-generator-results')
  await userEvent.click(screen.getByRole('combobox', { name: 'Visualize a Signal in the Generated Dataset' }))
  expect(screen.getAllByRole('option').map(option => option.textContent)).toEqual(['TM1.1 · 20 MHz · 30 kHz · FDD', 'TM1.1 · 20 MHz · 30 kHz · FDD · 10 subframes'])
})

test('hiding the matrix keeps the numerology group that was chosen in it', async () => {
  setup()
  renderWithProviders(<SignalGeneratorPage />, { route: '/signal-generator' })
  await ready()
  const toggle = screen.getByRole('button', { name: 'Generic numerology matrix…' })
  await userEvent.click(toggle)
  expect(screen.getByRole('button', { name: 'FR1 · 30 kHz' })).toHaveAttribute('aria-pressed', 'true')
  await userEvent.click(screen.getByRole('button', { name: 'FR2-1 · 120 kHz' }))
  expect(screen.getByTestId('preset-nr-fr2')).toBeVisible()
  await userEvent.click(toggle)
  expect(screen.queryByTestId('preset-nr-fr2')).not.toBeInTheDocument()
  await userEvent.click(toggle)
  expect(screen.getByRole('button', { name: 'FR2-1 · 120 kHz' })).toHaveAttribute('aria-pressed', 'true')
  expect(screen.getByTestId('preset-nr-fr2')).toBeVisible()
  expect(screen.queryByTestId('preset-nr-20')).not.toBeInTheDocument()
})

test('a configuration the server refuses is listed with its fields named as the form names them', async () => {
  setup({ 'POST /api/v1/signal-generator/validate': () => ({ status: 422, body: { error: { code: 'invalid_request', message: 'request validation failed', hint: null, details: [
    { field: 'nr_test_model.carriers.1.n_cell_id', message: 'Input should be less than or equal to 1007' }, { field: 'nr_test_model.subframes', message: 'Input should be greater than or equal to 1' }] } } }) })
  renderWithProviders(<SignalGeneratorPage />, { route: '/signal-generator' })
  await ready()
  await userEvent.upload(screen.getByTestId('generator-config-upload'), new File(['{}'], 'signal-config.json', { type: 'application/json' }))
  const alert = await screen.findByRole('alert')
  expect(alert).toHaveTextContent('Carrier 2 · Cell ID: Input should be less than or equal to 1007')
  expect(alert).toHaveTextContent('Capture length: Input should be greater than or equal to 1')
  expect(alert).not.toHaveTextContent('request validation failed')
})

// ---- final review round: the reason an entry is held, focus, the PN23 scope, names in the results, a refocus of the window ------------------------

const GENERATE_HELD = /Generate is off/

test('an entry that repeats another is marked and named as a duplicate of it, not as a refusal or an unchecked edit', async () => {
  setup()
  renderWithProviders(<SignalGeneratorPage />, { route: '/signal-generator' })
  await ready()
  await userEvent.click(screen.getByRole('button', { name: 'Add another test model' }))
  const second = chip(recipeWith({ model: 'NR-FR1-TM1.2' }))
  await screen.findByTestId(second)
  await waitFor(() => expect(generateButton()).toBeEnabled())
  // TM1.2 is edited back into TM1.1: the server accepts it, and it is the signal of the first entry.
  await userEvent.click(screen.getByRole('button', { name: 'TM1.1, QPSK' }))
  await screen.findByTestId('nrtm-duplicate')
  expect(screen.getByTestId(second)).toHaveAttribute('data-cause', 'duplicate')
  expect(within(screen.getByTestId(second)).getByTitle('Same settings as another selected entry')).toBeInTheDocument()
  // The first entry is on screen now; the line next to Generate names the held entry by its chip and the entry it repeats by its name.
  await userEvent.click(screen.getByTestId('selected-preset-' + DEFAULT_ID))
  const line = screen.getByTestId('generator-blocked-entry')
  expect(line).toHaveAttribute('data-cause', 'duplicate')
  expect(line).toHaveTextContent('The new settings of “TM1.2 · 20 MHz · 30 kHz · FDD” are the same as those of “TM1.1 · 20 MHz · 30 kHz · FDD”. A dataset holds each signal once, so Generate is off: change one of them or remove one.')
  expect(line).not.toHaveTextContent(/refused|not accepted|still being checked/)
  expect(generateButton()).toBeDisabled()
  // Save configuration is off for the highlighted entry only; the first entry is clean.
  expect(screen.getByRole('button', { name: 'Save configuration' })).toBeEnabled()
  // Removing the entry it repeats lets the held one take its id: the line and the marker are gone.
  await userEvent.click(screen.getByTestId('remove-preset-' + DEFAULT_ID))
  await waitFor(() => expect(screen.queryByTestId('generator-blocked-entry')).not.toBeInTheDocument())
  expect(screen.queryByTestId('nrtm-duplicate')).not.toBeInTheDocument()
  await waitFor(() => expect(generateButton()).toBeEnabled())
})

test('an entry whose edits the server is still checking is named as such, and the line leaves with the answer', async () => {
  setup()
  renderWithProviders(<SignalGeneratorPage />, { route: '/signal-generator' })
  await ready()
  fireEvent.change(carrier().getByLabelText('Cell ID'), { target: { value: '1000' } })
  // Within the debounce: the edit is still a draft that no request has been sent for.
  await userEvent.click(screen.getByRole('button', { name: /02 ·.*Wi-Fi 6/ }))
  const line = screen.getByTestId('generator-blocked-entry')
  expect(line).toHaveAttribute('data-cause', 'pending')
  expect(line).toHaveTextContent('“TM1.1 · 20 MHz · 30 kHz · FDD” is still being checked by the server, so Generate is off for the moment.')
  expect(line).not.toHaveTextContent(/refused/)
  expect(screen.getByTestId('selected-preset-' + DEFAULT_ID)).toHaveAttribute('data-cause', 'pending')
  expect(within(screen.getByTestId('selected-preset-' + DEFAULT_ID)).getByTitle('The server is still checking these settings')).toBeInTheDocument()
  expect(generateButton()).toBeDisabled()
  // The answer comes without the user returning to the tab: the entry takes the derived id and the line goes.
  await waitFor(() => expect(screen.queryByTestId('generator-blocked-entry')).not.toBeInTheDocument())
  expect(screen.getByTestId(chip(recipeWith({}, { n_cell_id: 1000 })))).not.toHaveAttribute('data-draft')
  await waitFor(() => expect(generateButton()).toBeEnabled())
  expect(GENERATE_HELD.test(document.body.textContent ?? '')).toBe(false)
})

test('"Show it" from the keyboard leaves the focus in the panel instead of dropping it to the document body', async () => {
  setup(refuseCell5000())
  renderWithProviders(<SignalGeneratorPage />, { route: '/signal-generator' })
  await ready()
  fireEvent.change(carrier().getByLabelText('Cell ID'), { target: { value: '5000' } })
  await screen.findByRole('alert')
  await userEvent.click(screen.getByRole('button', { name: /02 ·.*Wi-Fi 6/ }))
  const button = within(screen.getByTestId('generator-blocked-entry')).getByRole('button', { name: 'Show it' })
  act(() => button.focus())
  expect(button).toHaveFocus()
  await userEvent.keyboard('{Enter}')
  await screen.findByTestId('nrtm-panel')
  expect(button).not.toBeInTheDocument()
  expect(document.activeElement).not.toBe(document.body)
  const heading = screen.getByRole('heading', { name: 'Standard 5G NR test model' })
  expect(heading).toHaveFocus()
  expect(heading).toHaveAttribute('tabindex', '-1')
  // The heading is not a tab stop: one Tab moves on to the first control of the panel.
  await userEvent.tab()
  expect(heading).not.toHaveFocus()
  expect(within(screen.getByTestId('nrtm-panel')).getAllByRole('button')[0]).toHaveFocus()
})

test('switching TM3.2 to TDD and selecting all symbols sends the TDD PN23 scope', async () => {
  const { calls } = setup()
  renderWithProviders(<SignalGeneratorPage />, { route: '/signal-generator' })
  await ready()
  const tm32 = { model: 'NR-FR1-TM3.2' } as const
  await userEvent.click(screen.getByRole('button', { name: 'TM3.2, QPSK, 16QAM' }))
  await waitFor(() => expect(screen.getByTestId(chip(recipeWith(tm32)))).toBeInTheDocument())
  await userEvent.click(screen.getByRole('button', { name: 'TDD' }))
  await waitFor(() => expect(screen.getByTestId(chip(recipeWith({ ...tm32, duplex: 'tdd' })))).toBeInTheDocument())
  await pick(screen.getByRole('combobox', { name: 'PN23 scope (TDD)' }), 'All symbols')
  const tddAll = recipeWith({ ...tm32, duplex: 'tdd', pn_scope: 'all_symbols' })
  await waitFor(() => expect(screen.getByTestId(chip(tddAll))).toBeInTheDocument())
  expect(resolveCalls(calls).at(-1)!.body).toMatchObject(tddAll)
})

test('switching to NR-N-TM (FDD only) and back never leaves the TDD-only PN23 scope on an FDD recipe', async () => {
  // Restore the state exercised above so this test measures the two model switches,
  // without spending three more real debounce intervals setting up that state.
  const tm32 = { model: 'NR-FR1-TM3.2' } as const
  const tddAll = recipeWith({ ...tm32, duplex: 'tdd', pn_scope: 'all_symbols' })
  const { calls, cleanup } = await openWith([fixtureResolve(tddAll).config], 'generator-pn23-switch')
  try {
    await ready()
    expect(screen.getByTestId(chip(tddAll))).toBeInTheDocument()
    const before = resolveCalls(calls).length
    // NR-N-TM exists for FDD only: the recipe becomes FDD and the scope returns to its default.
    await userEvent.click(screen.getByRole('button', { name: 'NR-N-TM, QPSK' }))
    const ntm = recipeWith({ model: 'NR-N-TM', nb_iot: { position: 'lower', boost_db: 6 } }, { scs_khz: 15 })
    await waitFor(() => expect(screen.getByTestId(chip(ntm))).toBeInTheDocument())
    expect(screen.queryByRole('combobox', { name: 'PN23 scope (TDD)' })).not.toBeInTheDocument()
    await userEvent.click(screen.getByRole('button', { name: 'TM3.2, QPSK, 16QAM' }))
    const fdd = recipeWith(tm32, { scs_khz: 15 })
    await waitFor(() => expect(screen.getByTestId(chip(fdd))).toBeInTheDocument())
    // Every request after the switch carried the default scope, and the entry has the id of the plain FDD recipe (the same IQ must not get a second id).
    const bodies = resolveCalls(calls).slice(before).map(c => c.body as NrTmRecipe)
    expect(bodies.length).toBeGreaterThanOrEqual(2)
    expect(bodies.map(body => `${body.model}/${body.duplex}/${body.pn_scope}`)).toEqual(expect.arrayContaining(['NR-N-TM/fdd/dl_symbols', 'NR-FR1-TM3.2/fdd/dl_symbols']))
    expect(bodies.some(body => body.duplex === 'fdd' && body.pn_scope === 'all_symbols')).toBe(false)
    expect(screen.getAllByTestId(/^selected-preset-/).map(node => node.getAttribute('data-testid'))).toEqual(['selected-preset-' + fixtureResolve(fdd).config.preset_id])
  } finally { cleanup() }
})

test('the results header names a signal like its chip and its entry in the preview list, so two test models that differ only in the PN convention are not the same line', async () => {
  setup()
  renderWithProviders(<SignalGeneratorPage />, { route: '/signal-generator' })
  await ready()
  await userEvent.click(screen.getByRole('button', { name: 'Add another test model' }))
  await screen.findByTestId(chip(recipeWith({ model: 'NR-FR1-TM1.2' })))
  await userEvent.click(screen.getByRole('button', { name: 'TM1.1, QPSK' }))
  await screen.findByTestId('nrtm-duplicate')
  await pick(screen.getByRole('combobox', { name: 'PN23 data convention' }), 'MATLAB 5G Toolbox')
  await waitFor(() => expect(screen.getByTestId(chip(recipeWith({ data_convention: 'matlab' })))).toBeInTheDocument())
  await waitFor(() => expect(generateButton()).toBeEnabled())
  await userEvent.click(generateButton())
  await screen.findByTestId('signal-generator-results')
  const preview = screen.getByRole('combobox', { name: 'Visualize a Signal in the Generated Dataset' })
  // The plots are replaced while the other signal loads, so the header is looked up again each time.
  const header = () => screen.getByTestId('generator-result-name').textContent
  // The entry that was highlighted when Generate was pressed is shown first: the MATLAB-convention one.
  await waitFor(() => expect(header()).toMatch(/^TM1\.1 · 20 MHz · 30 kHz · FDD · MATLAB 5G Toolbox · 614,400 I\/Q samples/))
  await userEvent.click(preview)
  expect(screen.getAllByRole('option').map(option => option.textContent)).toEqual(['TM1.1 · 20 MHz · 30 kHz · FDD', 'TM1.1 · 20 MHz · 30 kHz · FDD · MATLAB 5G Toolbox'])
  await userEvent.click(screen.getByRole('option', { name: 'TM1.1 · 20 MHz · 30 kHz · FDD' }))
  await waitFor(() => expect(header()).toMatch(/^TM1\.1 · 20 MHz · 30 kHz · FDD · 614,400 I\/Q samples/))
  // The preset id stays reachable as the tooltip of the name.
  expect(within(screen.getByTestId('generator-result-name')).getByText(/^TM1\.1/)).toHaveAttribute('title', DEFAULT_ID)
})

const focusWindow = () => act(() => { focusManager.setFocused(false); focusManager.setFocused(true) })

test.each([[403, 'feature_unavailable'], [404, 'not_found']])('a window refocus does not ask a server that answered %i for the catalogue again, and no failure notice flashes', async (status, code) => {
  const { calls } = setup({ [CATALOG]: () => ({ status, body: { error: { code, message: 'This endpoint is not available', details: [], hint: null } } }) })
  renderWithProviders(<SignalGeneratorPage />, { route: '/signal-generator' })
  await screen.findByRole('heading', { name: 'Signal setup' })
  const notices: number[] = []
  const watch = new MutationObserver(() => notices.push(document.querySelectorAll('[data-testid=nrtm-catalog-problem]').length))
  watch.observe(document.body, { childList: true, subtree: true })
  focusWindow(); focusWindow()
  await wait(80)
  watch.disconnect()
  expect(calls.filter(c => c.path === '/api/v1/signal-generator/nr-test-models')).toHaveLength(1)
  expect(notices.every(count => count === 0)).toBe(true)
  expect(screen.queryByTestId('nrtm-catalog-problem')).not.toBeInTheDocument()
  expect(screen.getByTestId('preset-nr-20')).toHaveAttribute('aria-pressed', 'true')
})

test('a catalogue that failed with a 500 is not asked again on a window refocus: the notice stays and its Retry stays available', async () => {
  const { calls } = setup({ [CATALOG]: () => REFUSAL_500 })
  renderWithProviders(<SignalGeneratorPage />, { route: '/signal-generator' })
  const notice = await screen.findByTestId('nrtm-catalog-problem')
  focusWindow()
  await wait(80)
  expect(calls.filter(c => c.path === '/api/v1/signal-generator/nr-test-models')).toHaveLength(1)
  expect(screen.getByTestId('nrtm-catalog-problem')).toBe(notice)
  expect(within(notice).getByRole('button', { name: 'Retry' })).toBeEnabled()
})
