import { fireEvent, render, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { ThemeProvider } from '@mui/material/styles'
import { useState } from 'react'
import { vi } from 'vitest'
import { ApiConnectionError, ApiError } from '@/api/client'
import type { NrTmRecipe } from '@/api/nrTestModels'
import { fixtureAnswer, fixtureCatalog, fixtureResolve, type Envelope } from '@/test/nrTestModelFixtures'
import { theme } from '@/theme'
import { describeNrTmError, normalizeRecipe } from '@/utils/nrTestModel'
import { NrTestModelPanel, type NrTestModelPanelProps } from './NrTestModelPanel'

const catalog = fixtureCatalog()
const base = normalizeRecipe(catalog.default_recipe)
const carrier = (scs: 15 | 30 | 60, bandwidth_mhz: number) => ({ scs_khz: scs, bandwidth_mhz, n_cell_id: null, gap_before_mhz: 0 })

/** A stand-in for the page: the recipe is state, and the "server" answers instantly with the fixture. */
function Harness({ initial = base as NrTmRecipe | null, onChange = () => undefined, onAdd = () => undefined, ...rest }: { initial?: NrTmRecipe | null; onChange?: (r: NrTmRecipe) => void; onAdd?: (r: NrTmRecipe) => void } & Partial<NrTestModelPanelProps>) {
  const [recipe, setRecipe] = useState(initial)
  // A test that needs a recipe nobody captured from the backend passes `resolution={undefined}` (or one of its own) instead of a fixture lookup.
  const { resolution: given, ...others } = rest
  const resolution = 'resolution' in rest ? given : recipe ? fixtureResolve(recipe) : undefined
  return <ThemeProvider theme={theme}><NrTestModelPanel catalog={catalog} recipe={recipe} status="ready" fresh loading={false} disabled={false}
    selectedModels={recipe ? [recipe.model] : []} canAdd onChange={r => { setRecipe(r); onChange(r) }} onAdd={onAdd} {...others} resolution={resolution} /></ThemeProvider>
}
const carrierGroup = (n = 1) => within(screen.getByRole('group', { name: `Carrier ${n}` }))
const lastCall = (spy: ReturnType<typeof vi.fn>) => spy.mock.calls.at(-1)![0] as NrTmRecipe
async function pick(select: HTMLElement, option: string | RegExp) {
  await userEvent.click(select)
  await userEvent.click(await screen.findByRole('option', { name: option }))
}

/** The standard's fixed parameters sit in a collapsed section; it is closed until the reader asks for it. */
async function openStandard() {
  const toggle = within(screen.getByTestId('nrtm-standard')).getByRole('button', { name: 'Defined by the standard' })
  expect(toggle).toHaveAttribute('aria-expanded', 'false')
  await userEvent.click(toggle)
  expect(toggle).toHaveAttribute('aria-expanded', 'true')
}

test('all eleven models of the catalogue are offered with their modulation; picking one converts the recipe', async () => {
  const onChange = vi.fn()
  render(<Harness onChange={onChange} />)
  const cards = within(screen.getByRole('group', { name: 'Test model' })).getAllByRole('button')
  expect(cards.map(card => card.getAttribute('aria-label'))).toEqual(['TM1.1, QPSK', 'TM1.2, QPSK', 'TM2, 64QAM', 'TM2a, 256QAM', 'TM2b, 1024QAM', 'TM3.1, 64QAM', 'TM3.1a, 256QAM', 'TM3.1b, 1024QAM', 'TM3.2, QPSK, 16QAM', 'TM3.3, QPSK', 'NR-N-TM, QPSK'])
  expect(screen.getByRole('button', { name: 'TM1.1, QPSK' })).toHaveAttribute('aria-pressed', 'true')
  await userEvent.click(screen.getByRole('button', { name: 'TM3.2, QPSK, 16QAM' }))
  expect(lastCall(onChange)).toMatchObject({ model: 'NR-FR1-TM3.2', duplex: 'fdd', carriers: [carrier(30, 20)] })
  expect(screen.getByRole('button', { name: 'TM3.2, QPSK, 16QAM' })).toHaveAttribute('aria-pressed', 'true')
  // Title and clause are the catalogue's own words.
  const tm32 = catalog.models.find(model => model.model === 'NR-FR1-TM3.2')!
  expect(tm32.title).toBe('De-boosted 16QAM among boosted QPSK')
  expect(screen.getByTestId('nrtm-model-details')).toHaveTextContent(tm32.title)
  expect(screen.getByTestId('nrtm-model-details')).toHaveTextContent('TS 38.141-1 clause 4.9.2.2.7')
})

test('with no test model active, a card adds one instead of editing, and the editors stay hidden', async () => {
  const onAdd = vi.fn()
  render(<Harness initial={null} onAdd={onAdd} />)
  expect(screen.getByText(/to add it to the selection/)).toBeVisible()
  expect(screen.queryByRole('group', { name: 'Carrier 1' })).not.toBeInTheDocument()
  expect(screen.queryByRole('button', { name: 'Add another test model' })).not.toBeInTheDocument()
  await userEvent.click(screen.getByRole('button', { name: 'NR-N-TM, QPSK' }))
  expect(onAdd).toHaveBeenCalledWith(expect.objectContaining({ model: 'NR-N-TM', duplex: 'fdd', nb_iot: { position: 'lower', boost_db: 6 }, carriers: [carrier(15, 20)] }))
})

test('bandwidth choices follow the subcarrier spacing, and a spacing change moves an impossible bandwidth to the closest one', async () => {
  const onChange = vi.fn()
  render(<Harness initial={{ ...base, carriers: [carrier(30, 5)] }} onChange={onChange} />)
  const options = async (name: string) => {
    await userEvent.click(carrierGroup().getByRole('combobox', { name }))
    const values = screen.getAllByRole('option').map(o => o.textContent)
    await userEvent.keyboard('{Escape}')
    return values
  }
  expect(await options('Channel bandwidth (MHz)')).toEqual(['5', '10', '15', '20', '25', '30', '35', '40', '45', '50', '60', '70', '80', '90', '100'])
  expect(await options('Subcarrier spacing (kHz)')).toEqual(['15', '30', '60'])
  await pick(carrierGroup().getByRole('combobox', { name: 'Subcarrier spacing (kHz)' }), '60')
  expect(lastCall(onChange).carriers).toEqual([carrier(60, 10)])
  expect(await options('Channel bandwidth (MHz)')).toEqual(['10', '15', '20', '25', '30', '35', '40', '45', '50', '60', '70', '80', '90', '100'])
  await pick(carrierGroup().getByRole('combobox', { name: 'Subcarrier spacing (kHz)' }), '15')
  expect(lastCall(onChange).carriers).toEqual([carrier(15, 10)])
  expect((await options('Channel bandwidth (MHz)'))[0]).toBe('3')
  expect(carrierGroup().getByTestId('nrtm-nrb-0')).toHaveTextContent('N_RB 52 = 624 subcarriers')
})

test('the carrier table adds, clones and removes carriers up to 16 and offers the gap only after the first', async () => {
  const onChange = vi.fn()
  render(<Harness initial={{ ...base, carriers: [carrier(30, 100)] }} onChange={onChange} />)
  expect(screen.queryByLabelText('Gap before (MHz)')).not.toBeInTheDocument()
  expect(screen.getByRole('button', { name: 'Remove carrier 1' })).toBeDisabled()
  await userEvent.click(screen.getByRole('button', { name: 'Add a carrier like the last one' }))
  expect(lastCall(onChange).carriers).toEqual([carrier(30, 100), carrier(30, 100)])
  expect(screen.getByTestId('nrtm-carrier-count')).toHaveTextContent('2')
  expect(carrierGroup(1).queryByLabelText('Gap before (MHz)')).not.toBeInTheDocument()
  expect(carrierGroup(2).getByLabelText('Gap before (MHz)')).toHaveValue(0)
  expect(screen.getByText('0 = contiguous at the nominal channel spacing. Multiples of 0.3 MHz add a gap (non-contiguous sub-blocks).')).toBeVisible()
  await pick(carrierGroup(2).getByRole('combobox', { name: 'Channel bandwidth (MHz)' }), '20')
  fireEvent.change(carrierGroup(2).getByLabelText('Gap before (MHz)'), { target: { value: '0.9' } })
  fireEvent.change(carrierGroup(2).getByLabelText('Cell ID'), { target: { value: '12' } })
  expect(lastCall(onChange).carriers[1]).toEqual({ scs_khz: 30, bandwidth_mhz: 20, n_cell_id: 12, gap_before_mhz: 0.9 })
  // Growing clones the last row but starts with the default cell ID and no gap.
  await userEvent.click(screen.getByRole('button', { name: 'Add carrier' }))
  expect(lastCall(onChange).carriers[2]).toEqual(carrier(30, 20))
  for (let i = 0; i < 20; i++) if (!screen.getByRole('button', { name: 'Add carrier' }).hasAttribute('disabled')) await userEvent.click(screen.getByRole('button', { name: 'Add carrier' }))
  expect(screen.getByTestId('nrtm-carrier-count')).toHaveTextContent('16')
  expect(screen.getByRole('button', { name: 'Add carrier' })).toBeDisabled()
  expect(screen.getByRole('button', { name: 'Add a carrier like the last one' })).toBeDisabled()
  // The cap is Studio's choice, and the panel says so right beside the stepper.
  expect(screen.getByTestId('nrtm-carrier-limit')).toHaveTextContent('16 carriers is the Studio limit (the standard sets none).')
  await userEvent.click(screen.getByRole('button', { name: 'Remove carrier 2' }))
  expect(lastCall(onChange).carriers).toHaveLength(15)
  await userEvent.click(screen.getByRole('button', { name: 'Remove the last carrier' }))
  expect(lastCall(onChange).carriers).toHaveLength(14)
  expect(screen.queryByTestId('nrtm-carrier-limit')).not.toBeInTheDocument()
  // An emptied cell ID means "automatic" again; the placeholder shows what the server derived.
  fireEvent.change(carrierGroup(2).getByLabelText('Cell ID'), { target: { value: '' } })
  expect(lastCall(onChange).carriers[1]!.n_cell_id).toBeNull()
  expect(carrierGroup(2).getByLabelText('Cell ID')).toHaveAttribute('placeholder', 'auto (2)')
})

test('layers and the exported port exist only for TM1.1', async () => {
  const onChange = vi.fn()
  render(<Harness onChange={onChange} />)
  const layers = screen.getByRole('group', { name: 'Layers' })
  expect(within(layers).getByRole('button', { name: '1' })).toHaveAttribute('aria-pressed', 'true')
  expect(screen.queryByRole('combobox', { name: 'Exported antenna port' })).not.toBeInTheDocument()
  await userEvent.click(within(layers).getByRole('button', { name: '2' }))
  expect(lastCall(onChange)).toMatchObject({ layers: 2, output_port: 0 })
  await pick(screen.getByRole('combobox', { name: 'Exported antenna port' }), 'Port 1001')
  expect(lastCall(onChange)).toMatchObject({ layers: 2, output_port: 1 })
  await userEvent.click(within(layers).getByRole('button', { name: '1' }))
  expect(lastCall(onChange)).toMatchObject({ layers: 1, output_port: 0 })
  await userEvent.click(within(layers).getByRole('button', { name: '2' }))
  await userEvent.click(screen.getByRole('button', { name: 'TM3.1, 64QAM' }))
  expect(lastCall(onChange)).toMatchObject({ model: 'NR-FR1-TM3.1', layers: 1, output_port: 0 })
  expect(screen.queryByRole('group', { name: 'Layers' })).not.toBeInTheDocument()
  expect(screen.queryByRole('combobox', { name: 'Exported antenna port' })).not.toBeInTheDocument()
})

test('TDD adds the PN scope choice and shows the slot pattern the server resolved; FDD has neither', async () => {
  render(<Harness initial={{ ...base, duplex: 'tdd', carriers: [carrier(60, 100)] }} />)
  expect(screen.getByRole('combobox', { name: 'PN23 scope (TDD)' })).toHaveTextContent('Downlink symbols only')
  expect(screen.queryByRole('list', { name: 'Slots of one TDD period' })).not.toBeInTheDocument()
  await openStandard()
  const slots = within(screen.getByRole('list', { name: 'Slots of one TDD period' }))
  expect(slots.getAllByRole('listitem')).toHaveLength(20)
  expect(slots.getAllByRole('listitem').map(item => item.textContent).slice(13, 18)).toEqual(['D', 'S 12D', 'S 8U', 'U', 'U'])
  expect(screen.getByText('5 ms period: 14 downlink slots, then 12 downlink and 8 uplink symbols, then 4 uplink slots (TS 38.141-1 Table 4.9.2.2-1).')).toBeVisible()
  await userEvent.click(screen.getByRole('button', { name: 'FDD' }))
  expect(screen.queryByRole('combobox', { name: 'PN23 scope (TDD)' })).not.toBeInTheDocument()
  expect(screen.queryByRole('list', { name: 'Slots of one TDD period' })).not.toBeInTheDocument()
})

test('NR-N-TM is FDD and 15 kHz only and brings the NB-IoT block with the eligible RBs of the catalogue', async () => {
  const onChange = vi.fn()
  render(<Harness initial={{ ...base, duplex: 'tdd', carriers: [carrier(60, 100)] }} onChange={onChange} />)
  expect(screen.queryByRole('region', { name: 'NB-IoT in-band' })).not.toBeInTheDocument()
  await userEvent.click(screen.getByRole('button', { name: 'NR-N-TM, QPSK' }))
  expect(lastCall(onChange)).toMatchObject({ model: 'NR-N-TM', duplex: 'fdd', nb_iot: { position: 'lower', boost_db: 6 }, carriers: [carrier(15, 50)] })
  expect(screen.getByRole('button', { name: 'TDD' })).toBeDisabled()
  expect(screen.getByText('Studio implements NR-N-TM for FDD only; the standard also defines a TDD form (Table 4.9.2.2-1a) that is not implemented.')).toBeVisible()
  await userEvent.click(carrierGroup().getByRole('combobox', { name: 'Subcarrier spacing (kHz)' }))
  expect(screen.getAllByRole('option').map(o => o.textContent)).toEqual(['15'])
  await userEvent.keyboard('{Escape}')
  const block = within(screen.getByRole('region', { name: 'NB-IoT in-band' }))
  expect(block.getByTestId('nrtm-nbiot-rbs')).toHaveTextContent('Carrier 1: eligible NR RBs at 50 MHz: 54 positions, from RB 4 to RB 269.')
  await pick(block.getByRole('combobox', { name: 'Punctured NR RB' }), 'Both band edges')
  expect(lastCall(onChange).nb_iot).toEqual({ position: 'both', boost_db: 6 })
  fireEvent.change(block.getByLabelText('NB-IoT power boost X (dB)'), { target: { value: '9.5' } })
  expect(lastCall(onChange).nb_iot).toEqual({ position: 'both', boost_db: 9.5 })
  await pick(carrierGroup().getByRole('combobox', { name: 'Channel bandwidth (MHz)' }), '20')
  expect(block.getByTestId('nrtm-nbiot-rbs')).toHaveTextContent('Carrier 1: eligible NR RBs at 20 MHz: 21 positions, from RB 2 to RB 102.')
  expect(screen.getByTestId('nrtm-standard')).toHaveTextContent('NB-IoT punctures NR RB 2, 102.')
  await userEvent.click(screen.getByRole('button', { name: 'TM1.1, QPSK' }))
  expect(lastCall(onChange).nb_iot).toBeNull()
  expect(screen.queryByRole('region', { name: 'NB-IoT in-band' })).not.toBeInTheDocument()
})

test('oversampling and capture length expose automatic and custom values', async () => {
  const onChange = vi.fn()
  render(<Harness onChange={onChange} />)
  expect(screen.getByRole('combobox', { name: 'Oversampling' })).toHaveTextContent('Auto')
  await pick(screen.getByRole('combobox', { name: 'Oversampling' }), '4×')
  expect(lastCall(onChange).oversampling).toBe(4)
  await pick(screen.getByRole('combobox', { name: 'Oversampling' }), 'Auto')
  expect(lastCall(onChange).oversampling).toBeNull()
  expect(screen.getByRole('button', { name: 'Standard (10 subframes)' })).toHaveAttribute('aria-pressed', 'true')
  await userEvent.click(screen.getByRole('button', { name: 'Custom' }))
  expect(lastCall(onChange).subframes).toBe(10)
  fireEvent.change(screen.getByLabelText('Subframes (1 ms each)'), { target: { value: '3' } })
  expect(lastCall(onChange).subframes).toBe(3)
  await userEvent.click(screen.getByRole('button', { name: 'TDD' }))
  expect(screen.getByRole('button', { name: 'Standard (20 subframes)' })).toBeVisible()
  await userEvent.click(screen.getByRole('button', { name: 'Standard (20 subframes)' }))
  expect(lastCall(onChange).subframes).toBeNull()
  expect(screen.getByText(/^The standard defines 10 ms \(FDD\) or 20 ms \(TDD\)\. A custom capture may hold 1 to 20 subframes; a longer one repeats the pattern frame by frame, and with the standard PN23 convention the PN sequence restarts at every frame boundary\. A capture holds at most 1,000,000 I\/Q samples/)).toBeVisible()
})

test('the standard section lists the fixed parameters and what the server resolved: RBGs, users, levels, truncation', async () => {
  render(<Harness initial={{ ...base, model: 'NR-FR1-TM3.2', carriers: [carrier(30, 100)] }} />)
  await openStandard()
  const standard = within(screen.getByTestId('nrtm-standard'))
  expect(standard.getByText('60 % of the PRBs 3...N_RB-1 (resource-block groups of n_RNTI 1)')).toBeVisible()
  expect(standard.getByText('-3 dB (n_RNTI 1)')).toBeVisible()
  const carrierBlock = within(screen.getByTestId('nrtm-carrier-summary-0'))
  expect(carrierBlock.getByText('Carrier 1: 30 kHz, 100 MHz')).toBeVisible()
  expect(carrierBlock.getByText('273')).toBeVisible()
  expect(carrierBlock.getByText('3,276')).toBeVisible()
  expect(carrierBlock.getByText('4,096')).toBeVisible()
  // 273 PRBs at 30 kHz: P = 16, nine de-boosted 16QAM groups, the last one being RBG 16 (numbers of the real backend, see nrTmFixtureResponses.json).
  expect(carrierBlock.getByText('RBG P').nextSibling).toHaveTextContent('16')
  expect(carrierBlock.getByText('N_RBG').nextSibling).toHaveTextContent('9')
  expect(carrierBlock.getByText('Boosted / de-boosted RBGs').nextSibling).toHaveTextContent('1, 3, 5, 7, 9, 11, 13, 15, 16')
  const rows = within(carrierBlock.getByRole('table')).getAllByRole('row').slice(1)
  expect(rows).toHaveLength(3)
  expect(within(rows[1]!).getAllByRole('cell').map(c => c.textContent).slice(0, 4)).toEqual(['1', 'de-boosted 16QAM RBGs, EVM measured (type 0 allocation)', '16QAM', '144'])
  expect(within(rows[1]!).getAllByRole('cell')[6]).toHaveTextContent('-3.00 dB')
  expect(within(rows[0]!).getAllByRole('cell')[3]).toHaveTextContent('126')
  expect(within(rows[0]!).getAllByRole('cell')[4]).toHaveTextContent('3–15, 32–47, 64–79, 96–111, 128–143, 160–175, 192–207, 224–239, 272')
  expect(within(rows[0]!).getAllByRole('cell')[6]).toHaveTextContent('+1.96 dB')
  expect(screen.getByTestId('nrtm-derived')).toHaveTextContent('122.88 MS/s (oversampling 1×) · 983,040 I/Q samples · 8 ms · 8 subframes')
  expect(screen.getByText('Shortened: 8 of 10 subframes')).toBeVisible()
  // The server's warnings are not hidden in the closed section: they sit under the derived numbers, outside "Defined by the standard".
  expect(standard.queryByText(/^The standard duration of 10 ms/)).not.toBeInTheDocument()
  expect(within(screen.getByTestId('nrtm-warnings')).getByText(/^The standard duration of 10 ms \(one radio frame\) would need 1,228,800 samples; the capture is shortened to 8 ms \(983,040 samples, limit 1,000,000\)/)).toBeVisible()
})

/** The refusal the real backend sent for a recipe (captured fixture), as the client raises it. */
function refusal(recipe: NrTmRecipe) {
  const { status, body } = fixtureAnswer(recipe)
  const { error } = body as Envelope
  return new ApiError(status, error.code, error.message, error.details, error.hint)
}

test('describeNrTmError reads the pydantic envelope of the real backend: no "Value error, " prefix, no generic summary line', () => {
  const sixty = refusal({ ...base, carriers: [carrier(60, 5)] })
  expect(sixty.message).toBe('request validation failed')
  expect(sixty.details[0]).toMatchObject({ field: '', message: expect.stringMatching(/^Value error, Carrier 1: A 5 MHz channel is not defined for 60 kHz/) })
  expect(describeNrTmError(sixty)).toEqual(['Carrier 1: A 5 MHz channel is not defined for 60 kHz subcarrier spacing in TS 38.104 table 5.3.2-1; valid bandwidths: 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100 MHz.'])
  expect(describeNrTmError(refusal({ ...base, carriers: Array.from({ length: 16 }, () => carrier(30, 100)) }))).toEqual(
    ['The aggregated signal (1596.8 MHz) needs a sample rate of 1966.08 MS/s, i.e. 1,966,080 samples per subframe; at most 1,000,000 samples can be generated. Use fewer or narrower carriers.'])
})

test('describeNrTmError also reads a plain refusal without details, a field detail, a lost connection and a bare exception', () => {
  // routes/signal_generator_routes.py answers ValueErrors that the schema did not catch as { code: 'invalid_signal', message } with no details.
  expect(describeNrTmError(new ApiError(422, 'invalid_signal', 'Carrier 2 overlaps carrier 1.'))).toEqual(['Carrier 2 overlaps carrier 1.'])
  expect(describeNrTmError(new ApiError(422, 'invalid_signal', 'Value error, Carrier 2 overlaps carrier 1.'))).toEqual(['Carrier 2 overlaps carrier 1.'])
  expect(describeNrTmError(new ApiError(422, 'invalid_request', 'request validation failed', [{ field: 'carriers.1.gap_before_mhz', message: 'Value error, must be a multiple of 0.3' }, { message: 'second' }]))).toEqual(
    ['Carrier 2 · Gap before: must be a multiple of 0.3', 'second'])
  expect(describeNrTmError(new ApiError(429, 'rate_limited', 'Too many requests', [], 'Try again in a minute.'))).toEqual(['Too many requests', 'Try again in a minute.'])
  expect(describeNrTmError(new ApiConnectionError('https://api.example.test'))).toEqual(['The browser could not reach https://api.example.test.'])
  expect(describeNrTmError(new Error('boom'))).toEqual(['boom'])
  expect(describeNrTmError('text')).toEqual(['text'])
})

test('while the server works the previous numbers stay but dim; a refusal lists the readable server text', () => {
  const { rerender } = render(<Harness status="waiting" fresh={false} loading />)
  expect(screen.getByTestId('nrtm-panel')).toHaveAttribute('aria-busy', 'true')
  expect(screen.getByTestId('nrtm-standard').querySelector('[aria-busy="true"]')).not.toBeNull()
  expect(within(screen.getByTestId('nrtm-status')).getByRole('status')).toHaveTextContent('Resolving with the server…')
  expect(screen.getByRole('progressbar', { name: 'Resolving with the server…' })).toBeVisible()
  rerender(<Harness status="error" error={refusal({ ...base, carriers: [carrier(15, 100)] })} fresh={false} />)
  const alert = screen.getByRole('alert')
  expect(alert).toHaveTextContent('This setup cannot be generated')
  expect(alert).toHaveTextContent('Carrier 1: A 100 MHz channel is not defined for 15 kHz subcarrier spacing in TS 38.104 table 5.3.2-1; valid bandwidths: 3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50 MHz.')
  expect(alert).not.toHaveTextContent(/Value error|request validation failed/)
})

test('"Add another test model" starts from the next model that is not selected; controls lock while generating', async () => {
  const onAdd = vi.fn()
  const { rerender } = render(<Harness onAdd={onAdd} />)
  await userEvent.click(screen.getByRole('button', { name: 'Add another test model' }))
  expect(onAdd).toHaveBeenCalledWith(expect.objectContaining({ model: 'NR-FR1-TM1.2', carriers: [carrier(30, 20)] }))
  rerender(<Harness onAdd={onAdd} disabled canAdd={false} />)
  expect(screen.getByRole('button', { name: 'Add another test model' })).toBeDisabled()
  for (const control of [screen.getByRole('button', { name: 'TM2, 64QAM' }), screen.getByRole('button', { name: 'TDD' }), screen.getByRole('button', { name: 'Add carrier' })]) expect(control).toBeDisabled()
  expect(carrierGroup().getByRole('combobox', { name: 'Channel bandwidth (MHz)' })).toHaveAttribute('aria-disabled', 'true')
})

test('removing carrier 1 gives the carrier that becomes first no gap: the field is hidden for it, so a leftover gap could never be cleared', async () => {
  const onChange = vi.fn()
  render(<Harness initial={{ ...base, carriers: [carrier(30, 20), { scs_khz: 30, bandwidth_mhz: 10, n_cell_id: 9, gap_before_mhz: 0.6 }, { scs_khz: 30, bandwidth_mhz: 10, n_cell_id: null, gap_before_mhz: 1.2 }] }} resolution={undefined} onChange={onChange} />)
  await userEvent.click(screen.getByRole('button', { name: 'Remove carrier 1' }))
  expect(lastCall(onChange).carriers).toEqual([{ scs_khz: 30, bandwidth_mhz: 10, n_cell_id: 9, gap_before_mhz: 0 }, { scs_khz: 30, bandwidth_mhz: 10, n_cell_id: null, gap_before_mhz: 1.2 }])
  expect(carrierGroup(1).queryByLabelText('Gap before (MHz)')).not.toBeInTheDocument()
  expect(carrierGroup(2).getByLabelText('Gap before (MHz)')).toHaveValue(1.2)
})

test('with the selection full and no test model active, the model cards are off and the limit is explained; nothing is added', async () => {
  const onAdd = vi.fn()
  render(<Harness initial={null} canAdd={false} onAdd={onAdd} />)
  const cards = within(screen.getByRole('group', { name: 'Test model' })).getAllByRole('button')
  expect(cards).toHaveLength(11)
  for (const card of cards) expect(card).toBeDisabled()
  expect(screen.getByTestId('nrtm-selection-full')).toHaveTextContent('Select up to 16 presets and 4,000,000 total I/Q samples.')
  fireEvent.click(cards[0]!)
  expect(onAdd).not.toHaveBeenCalled()
})

test('a custom capture may hold 1 to 20 subframes for FDD too; the help names the standard length and the PN restart', async () => {
  const onChange = vi.fn()
  render(<Harness initial={{ ...base, subframes: 15 }} resolution={undefined} onChange={onChange} />)
  const field = screen.getByLabelText('Subframes (1 ms each)')
  expect(field).toHaveAttribute('max', '20')
  expect(field).toHaveAttribute('min', '1')
  expect(field).not.toHaveAttribute('aria-invalid', 'true')
  fireEvent.change(field, { target: { value: '12' } })
  expect(lastCall(onChange).subframes).toBe(12)
  expect(screen.getByText(/repeats the pattern frame by frame/)).toBeVisible()
})

test('the PN23 scope is offered only where it has an effect: TDD with the standard convention; other combinations keep the default', async () => {
  const onChange = vi.fn()
  render(<Harness initial={{ ...base, duplex: 'tdd', carriers: [carrier(30, 20)] }} resolution={undefined} onChange={onChange} />)
  await pick(screen.getByRole('combobox', { name: 'PN23 scope (TDD)' }), 'All symbols')
  expect(lastCall(onChange).pn_scope).toBe('all_symbols')
  await pick(screen.getByRole('combobox', { name: 'PN23 data convention' }), 'MATLAB 5G Toolbox')
  expect(lastCall(onChange)).toMatchObject({ data_convention: 'matlab', pn_scope: 'dl_symbols' })
  expect(screen.queryByRole('combobox', { name: 'PN23 scope (TDD)' })).not.toBeInTheDocument()
  await pick(screen.getByRole('combobox', { name: 'PN23 data convention' }), 'Standard text')
  await pick(screen.getByRole('combobox', { name: 'PN23 scope (TDD)' }), 'All symbols')
  await userEvent.click(screen.getByRole('button', { name: 'FDD' }))
  expect(lastCall(onChange)).toMatchObject({ duplex: 'fdd', pn_scope: 'dl_symbols' })
})

test('NR-N-TM names the eligible RBs of the host carrier of each NB-IoT carrier: the first for lower, the last for upper, both for both', () => {
  const ntm = { ...base, model: 'NR-N-TM' as const, carriers: [{ ...carrier(15, 10), scs_khz: 15 as const }, { ...carrier(15, 20), scs_khz: 15 as const }] }
  const lines = (nb_iot: NonNullable<NrTmRecipe['nb_iot']>) => {
    const { unmount } = render(<Harness initial={{ ...ntm, nb_iot }} resolution={undefined} />)
    const text = Array.from(screen.getByTestId('nrtm-nbiot-rbs').children).map(line => line.textContent)
    unmount()
    return text
  }
  expect(lines({ position: 'lower', boost_db: 6 })).toEqual(['Carrier 1: eligible NR RBs at 10 MHz: 11 positions, from RB 0 to RB 50.'])
  expect(lines({ position: 'upper', boost_db: 6 })).toEqual(['Carrier 2: eligible NR RBs at 20 MHz: 21 positions, from RB 2 to RB 102.'])
  expect(lines({ position: 'both', boost_db: 6 })).toEqual(['Carrier 1: eligible NR RBs at 10 MHz: 11 positions, from RB 0 to RB 50.', 'Carrier 2: eligible NR RBs at 20 MHz: 21 positions, from RB 2 to RB 102.'])
})

test('TDD recipes with different spacings show the slot pattern of each spacing, labelled with its carriers', async () => {
  render(<Harness initial={{ ...base, duplex: 'tdd', carriers: [carrier(30, 40), carrier(15, 10), carrier(30, 20)] }} resolution={undefined} />)
  await openStandard()
  const thirty = within(screen.getByTestId('nrtm-tdd-30'))
  const fifteen = within(screen.getByTestId('nrtm-tdd-15'))
  expect(thirty.getByText('30 kHz · Carrier 1, Carrier 3')).toBeVisible()
  expect(fifteen.getByText('15 kHz · Carrier 2')).toBeVisible()
  expect(thirty.getAllByRole('listitem').map(item => item.textContent)).toEqual(['D', 'D', 'D', 'D', 'D', 'D', 'D', 'S 6D/4U', 'U', 'U'])
  expect(fifteen.getAllByRole('listitem').map(item => item.textContent)).toEqual(['D', 'D', 'D', 'S 10D/2U', 'U'])
  expect(screen.getByRole('list', { name: 'Slots of one TDD period · 15 kHz' })).toBeVisible()
})

test('the fixed parameters are stacked name / value rows (a definition list), not a table that has to scroll sideways on a phone', async () => {
  render(<Harness />)
  await openStandard()
  const fixed = within(screen.getByRole('region', { name: 'Parameters fixed by the test model' }))
  const info = catalog.models.find(model => model.model === base.model)!
  expect(fixed.getAllByRole('term').map(term => term.textContent)).toEqual(info.fixed_parameters.map(row => row.name))
  expect(fixed.getAllByRole('definition').map(definition => definition.textContent)).toEqual(info.fixed_parameters.map(row => row.value))
  expect(fixed.queryByRole('table')).not.toBeInTheDocument()
  expect(screen.queryByRole('region', { name: 'Parameters fixed by the test model', hidden: true })?.getAttribute('tabindex') ?? null).toBeNull()
})

test('a refused setup hides the numbers of the last valid one instead of printing them under the error', () => {
  const resolution = fixtureResolve(base)
  const { rerender } = render(<Harness status="ready" fresh />)
  expect(screen.getByTestId('nrtm-derived')).toBeVisible()
  rerender(<Harness status="error" fresh={false} resolution={resolution} error={refusal({ ...base, carriers: [carrier(15, 100)] })} />)
  expect(screen.getByRole('alert')).toBeVisible()
  expect(screen.queryByTestId('nrtm-derived')).not.toBeInTheDocument()
  expect(screen.queryByTestId('nrtm-warnings')).not.toBeInTheDocument()
  expect(screen.queryByText(/Shortened:/)).not.toBeInTheDocument()
})

test('the NB-IoT boost field takes its bounds from the catalogue: the largest minimum of the host carriers and the Studio maximum, stated in the help', () => {
  const ntm = { ...base, model: 'NR-N-TM' as const, nb_iot: { position: 'both' as const, boost_db: 6 }, carriers: [{ ...carrier(15, 10), scs_khz: 15 as const }, { ...carrier(15, 40), scs_khz: 15 as const }] }
  const field = () => screen.getByLabelText('NB-IoT power boost X (dB)')
  // The backend's catalogue: 6 dB up to 20 MHz, 3 dB from 25 MHz (TS 38.104 Table 6.3.4.2-1), at most 12 dB. Both hosts count: 10 MHz needs 6.
  expect(catalog.nb_iot.min_boost_db).toMatchObject({ '10': 6, '20': 6, '25': 3, '40': 3 })
  const { unmount } = render(<Harness initial={ntm} resolution={undefined} />)
  expect([field().getAttribute('min'), field().getAttribute('max')]).toEqual(['6', '12'])
  expect(screen.getByText('Declared maximum NB-IoT power dynamic range X. At least 6 dB for this host bandwidth; Studio accepts up to 12 dB (the standard sets no upper bound).')).toBeVisible()
  unmount()
  const upper = { ...ntm, nb_iot: { position: 'upper' as const, boost_db: 6 } }
  const { unmount: second } = render(<Harness initial={upper} resolution={undefined} />)
  expect(field().getAttribute('min')).toBe('3')
  second()
  // A catalogue without the bounds (an older API) keeps the schema's 3 to 12 dB; one with another maximum is followed.
  const older = { ...catalog, nb_iot: { ...catalog.nb_iot, min_boost_db: undefined, max_boost_db: undefined } } as unknown as typeof catalog
  const { unmount: third } = render(<Harness initial={ntm} resolution={undefined} catalog={older} />)
  expect([field().getAttribute('min'), field().getAttribute('max')]).toEqual(['3', '12'])
  third()
  render(<Harness initial={ntm} resolution={undefined} catalog={{ ...catalog, nb_iot: { ...catalog.nb_iot, max_boost_db: 15 } }} />)
  expect(field().getAttribute('max')).toBe('15')
  expect(screen.getByText(/At least 6 dB for this host bandwidth; Studio accepts up to 15 dB/)).toBeVisible()
})

test('the headings of the panel nest: h3 for the panel, h4 for its sections and accordions, h5 inside "Defined by the standard", never a jump', async () => {
  render(<Harness initial={{ ...base, duplex: 'tdd', model: 'NR-FR1-TM3.2', carriers: [carrier(30, 100)] }} resolution={undefined} />)
  const levels = () => within(screen.getByTestId('nrtm-panel')).getAllByRole('heading', { hidden: true }).map(heading => Number(heading.tagName.slice(1)))
  const seen = levels()
  expect(seen[0]).toBe(3)
  expect(seen).not.toContain(6)
  seen.forEach((level, i) => { if (i > 0) expect(level, `heading ${i} follows h${seen[i - 1]}`).toBeLessThanOrEqual(seen[i - 1]! + 1) })
  expect(seen.filter(level => level === 3)).toHaveLength(1)
  expect(within(screen.getByTestId('nrtm-standard')).getAllByRole('heading', { hidden: true }).map(heading => heading.tagName)).toEqual(['H4', 'H5'])
})

test('a model card never leaves the TDD-only PN23 scope on a recipe that cannot use it, so the same IQ cannot get a second preset id', async () => {
  const onChange = vi.fn()
  const tddAll: NrTmRecipe = { ...base, model: 'NR-FR1-TM3.2', duplex: 'tdd', pn_scope: 'all_symbols' }
  render(<Harness initial={tddAll} onChange={onChange} resolution={undefined} />)
  expect(screen.getByRole('combobox', { name: 'PN23 scope (TDD)' })).toHaveTextContent('All symbols')
  // Another model that exists for TDD keeps the choice.
  await userEvent.click(screen.getByRole('button', { name: 'TM3.3, QPSK' }))
  expect(lastCall(onChange)).toMatchObject({ model: 'NR-FR1-TM3.3', duplex: 'tdd', pn_scope: 'all_symbols' })
  // NR-N-TM is FDD only: the recipe becomes FDD with the default scope, and the (now hidden) select cannot be used to reset it.
  await userEvent.click(screen.getByRole('button', { name: 'NR-N-TM, QPSK' }))
  expect(lastCall(onChange)).toMatchObject({ model: 'NR-N-TM', duplex: 'fdd', pn_scope: 'dl_symbols' })
  expect(screen.queryByRole('combobox', { name: 'PN23 scope (TDD)' })).not.toBeInTheDocument()
  await userEvent.click(screen.getByRole('button', { name: 'TM3.2, QPSK, 16QAM' }))
  expect(lastCall(onChange)).toMatchObject({ model: 'NR-FR1-TM3.2', duplex: 'fdd', pn_scope: 'dl_symbols' })
})

test.each([
  ['the next TDD model keeps the scope', ['NR-FR1-TM3.1'], { model: 'NR-FR1-TM3.1a', duplex: 'tdd', pn_scope: 'all_symbols' }],
  ['NR-N-TM (FDD only) drops it', ['NR-FR1-TM3.1', 'NR-FR1-TM3.1a', 'NR-FR1-TM3.1b', 'NR-FR1-TM3.2', 'NR-FR1-TM3.3'], { model: 'NR-N-TM', duplex: 'fdd', pn_scope: 'dl_symbols' }],
])('"Add another test model" converts the recipe like a model card: %s', async (_name, used, expected) => {
  const onAdd = vi.fn()
  render(<Harness initial={{ ...base, model: 'NR-FR1-TM3.1', duplex: 'tdd', pn_scope: 'all_symbols' }} onAdd={onAdd} resolution={undefined} selectedModels={used} />)
  await userEvent.click(screen.getByRole('button', { name: 'Add another test model' }))
  expect(lastCall(onAdd)).toMatchObject(expected)
})
