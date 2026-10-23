import { screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import fixture from '@mocks/generator_presets.json'
import type { GeneratedSignal } from '@/api/signalGenerator'
import { fixtureResolve, fixtureSignal } from '@/test/nrTestModelFixtures'
import { renderWithProviders } from '@/test/utils'
import { SignalGeneratorPlots } from './SignalGeneratorPlots'

// The chart is replaced by its inputs, so the tests can see which axis and which points the page asks for.
vi.mock('@/components/PlotlyChart', () => ({
  PlotlyChart: ({ title, traces, layout }: { title: string; traces: Array<{ name?: string; x?: number[]; y?: number[] }>; layout?: { xaxis?: { title?: { text?: string } }; yaxis?: { title?: { text?: string } } } }) =>
    <div data-testid={'plot-' + title} data-x-title={layout?.xaxis?.title?.text} data-y-title={layout?.yaxis?.title?.text} data-traces={JSON.stringify(traces.map(trace => ({ name: trace.name, x: trace.x, y: trace.y })))} />,
}))

// Both signals are real answers of the backend (scripts/capture_nr_tm_fixtures.py): resolve + generate + read, long plot arrays decimated.
//  - two carriers, 15 kHz 10 MHz and 30 kHz 20 MHz: analysis.subcarrier_spacing_hz = 15 kHz (the lowest spacing), so the 30 kHz carrier lists every second bin;
//  - NR-N-TM at 10 MHz / 15 kHz with the NB-IoT carrier at both band edges.
const twoNumerologies = () => structuredClone(fixtureSignal('tm1p1-15k10m-plus-30k20m').signal)
const nbIotSignal = () => structuredClone(fixtureSignal('ntm-15k10m-both').signal)
function matrixSignal(): GeneratedSignal {
  const base = twoNumerologies()
  return { ...base, config: fixture.data[0]!.config, coverage: 'numerology', analysis: { ...base.analysis, nr_test_model: undefined, notes: ['SYNTHETIC'],
    allocation: [{ channel: 1, subcarriers: [-2, -1, 1, 2], pilots: [], modulation_order: 64, power_db: 0 }], evm_subcarrier_indices: [-2, -1, 1, 2], evm_per_subcarrier_percent: [1, 1, 1, 1] } } as unknown as GeneratedSignal
}
const plotData = (id: string) => JSON.parse(screen.getByTestId(id).getAttribute('data-traces')!) as Array<{ name: string; x: number[]; y: number[] }>
const mhz = (bin: number, hz = 15000) => bin * hz / 1e6

test('the captured fixture is what the backend documents: bins are absolute, in units of the lowest spacing, the 30 kHz carrier takes every second one', () => {
  const { analysis } = twoNumerologies()
  const [first, second] = analysis.allocation as Array<{ subcarriers: number[]; subcarrier_step: number; scs_khz: number; center_offset_hz: number }>
  expect(analysis.subcarrier_spacing_hz).toBe(15000)
  expect([first!.scs_khz, second!.scs_khz, first!.subcarrier_step, second!.subcarrier_step]).toEqual([15, 30, 1, 1])
  expect([first!.subcarriers[1]! - first!.subcarriers[0]!, second!.subcarriers[1]! - second!.subcarriers[0]!]).toEqual([1, 2])
  expect([first!.subcarriers[0], first!.subcarriers.at(-1), second!.subcarriers[0], second!.subcarriers.at(-1)]).toEqual([-951, -328, -271, 951])
  expect([first!.subcarriers.length, second!.subcarriers.length]).toEqual([624, 612])
})

test('a test-model signal places each carrier at bin x subcarrier spacing: carrier 2 starts at -4.065 MHz, carrier 1 at -14.265 MHz, centre offsets are not added', async () => {
  const signal = twoNumerologies()
  renderWithProviders(<SignalGeneratorPlots result={signal} stale={false} />)
  expect(screen.getByText('3GPP test model')).toBeVisible()
  await userEvent.click(screen.getByRole('tab', { name: 'Resource allocation' }))
  const plot = screen.getByTestId('plot-Resource allocation')
  expect(plot).toHaveAttribute('data-x-title', 'Baseband frequency (MHz)')
  expect(plot).toHaveAttribute('data-y-title', 'Carrier')
  const [one, pilots1, two, pilots2] = plotData('plot-Resource allocation')
  expect([one!.name, pilots1!.name, two!.name, pilots2!.name]).toEqual(['Carrier 1', 'Pilots 1', 'Carrier 2', 'Pilots 2'])
  // 15 kHz carrier: bins -951 ... -328; 30 kHz carrier: bins -271, -269, ... 951.
  expect([one!.x[0], one!.x.at(-1)]).toEqual([-14.265, -4.92])
  expect([two!.x[0], two!.x[1], two!.x.at(-1)]).toEqual([-4.065, -4.035, 14.265])
  expect(one!.x).toHaveLength(624)
  expect(two!.x).toHaveLength(612)
  expect(one!.x[0]).toBeCloseTo(-14.27, 2)
  // Both rows share the axis and do not overlap; adding center_offset_hz (-9.589 / +5.111 MHz) to these bins would shift them out of the sampled band.
  expect(Math.max(...one!.x)).toBeLessThan(Math.min(...two!.x))
  expect(Math.min(...one!.x)).toBeGreaterThan(-signal.analysis.sample_rate_hz / 2e6)
  expect(Math.max(...two!.x)).toBeLessThan(signal.analysis.sample_rate_hz / 2e6)
  expect([new Set(one!.y), new Set(two!.y)]).toEqual([new Set([1]), new Set([2])])
  expect([pilots1!.x, pilots2!.x]).toEqual([[], []])
})

test('a wide capture lists one bin per resource block and says so; the markers sit at bin x spacing of the 30 kHz capture', async () => {
  const { signal } = fixtureSignal('tm1p1-2x100m-30k-thinned')
  renderWithProviders(<SignalGeneratorPlots result={structuredClone(signal)} stale={false} />)
  await userEvent.click(screen.getByRole('tab', { name: 'Resource allocation' }))
  const [one, , two] = plotData('plot-Resource allocation')
  expect(signal.analysis.subcarrier_spacing_hz).toBe(30000)
  expect([one!.x.length, two!.x.length]).toEqual([273, 273])
  // First subcarrier of each resource block: bins -3302, -3290, ... of 30 kHz.
  expect([one!.x[0], one!.x[1], one!.x.at(-1), two!.x[0], two!.x.at(-1)]).toEqual([mhz(-3302, 30000), mhz(-3290, 30000), mhz(-38, 30000), mhz(28, 30000), mhz(3292, 30000)])
  expect(one!.x[0]).toBeCloseTo(-99.06, 2)
  expect(screen.getByText(/Wide captures show one marker per resource block \(12 subcarriers\) instead of every subcarrier\./)).toBeVisible()
})

test('the EVM-by-subcarrier plot of a test model uses the same absolute frequency axis', async () => {
  renderWithProviders(<SignalGeneratorPlots result={twoNumerologies()} stale={false} />)
  await userEvent.click(screen.getByRole('tab', { name: 'EVM trends' }))
  const plot = screen.getByTestId('plot-EVM by data subcarrier')
  expect(plot).toHaveAttribute('data-x-title', 'Baseband frequency (MHz)')
  const [trace] = plotData('plot-EVM by data subcarrier')
  expect(trace!.x).toHaveLength(1236)
  expect([trace!.x[0], trace!.x.at(-1)]).toEqual([mhz(-951), mhz(951)])
})

test('the metrics tab lists the model, the capture length, one row per carrier with its EVM, and describes the test-model scope', async () => {
  const signal = twoNumerologies()
  const summary = signal.analysis.nr_test_model!
  renderWithProviders(<SignalGeneratorPlots result={signal} stale={false} />)
  await userEvent.click(screen.getByRole('tab', { name: 'Metrics & provenance' }))
  const metrics = screen.getAllByRole('row').map(row => row.textContent ?? '')
  expect(metrics).toContainEqual(expect.stringMatching(/^Test modelTM1\.1$/))
  expect(metrics).toContainEqual(expect.stringContaining('Captured length10 of 10 subframes'))
  expect(metrics).toContainEqual(expect.stringMatching(/^Native sample rate × oversampling61\.44 MHz × 1$/))
  const table = within(screen.getByTestId('nrtm-result-carriers'))
  expect(table.getAllByRole('row')).toHaveLength(1 + summary.carriers.length)
  const rows = table.getAllByRole('row').slice(1).map(row => within(row).getAllByRole('cell').map(cell => cell.textContent))
  // carrier, scs, bandwidth, N_RB, FFT, cell id, centre offset, occupied, EVM
  expect(rows[0]).toEqual(['1', '15', '10', '52', '4,096', '1', '-9.589 MHz', '9.36 MHz', '1.91'])
  expect(rows[1]).toEqual(['2', '30', '20', '51', '2,048', '2', '5.111 MHz', '18.36 MHz', '1.31'])
  expect(screen.getByText(/Baseband implementation of the 3GPP TS 38\.141-1 test-model definition/)).toBeVisible()
  expect(screen.queryByText(/no complete protocol frames or conformance certification/)).not.toBeInTheDocument()
  expect(screen.queryByTestId('nrtm-result-nbiot')).not.toBeInTheDocument()
})

test('the per-signal warnings stay visible above the tabs, long lines included, and are not repeated in the metrics tab', async () => {
  const signal = twoNumerologies()
  const own = signal.analysis.nr_test_model!.warnings ?? []
  const long = fixtureResolve({ ...fixtureSignal('tm1p1-15k10m-plus-30k20m').recipe, carriers: [{ scs_khz: 30, bandwidth_mhz: 100, n_cell_id: null, gap_before_mhz: 0 }] }).summary.warnings?.[0] ?? ''
  expect(long).toMatch(/^The standard duration of 10 ms \(one radio frame\) would need 1,228,800 samples; the capture is shortened to 8 ms \(983,040 samples, limit 1,000,000\)/)
  signal.analysis.nr_test_model!.warnings = [long, ...own]
  renderWithProviders(<SignalGeneratorPlots result={signal} stale={false} />)
  expect(screen.getAllByTestId('nrtm-result-warning').map(node => node.textContent)).toEqual([long, own[0]])
  expect(screen.getByText(long)).toBeVisible()
  await userEvent.click(screen.getByRole('tab', { name: 'Metrics & provenance' }))
  expect(screen.getAllByText(long)).toHaveLength(1)
})

test('NR-N-TM shows what the generator did to the host: N_ID, levels, resource-element counts and both punctured RBs, and marks them in the allocation plot', async () => {
  const signal = nbIotSignal()
  renderWithProviders(<SignalGeneratorPlots result={signal} stale={false} />)
  // The per-host warning of the real response is a 490-character sentence; it is shown in full on the first tab.
  const warning = signal.analysis.nr_test_model!.warnings![0]!
  expect(warning.length).toBeGreaterThan(400)
  expect(screen.getByText(warning)).toBeVisible()
  await userEvent.click(screen.getByRole('tab', { name: 'Metrics & provenance' }))
  const block = within(screen.getByTestId('nrtm-result-nbiot-0'))
  expect(block.getByText('Carrier 1: N_ID^Ncell 103, host cell ID 1')).toBeVisible()
  const facts = Object.fromEntries(block.getAllByRole('term').map(term => [term.textContent, term.nextElementSibling?.textContent]))
  expect(facts).toEqual({ 'NB-IoT power boost X (dB)': '6', 'NB-IoT RE level': '+6.00 dB', 'NR RE level': '-0.34 dB', 'Resource elements per frame': 'NR 85,190 · NB-IoT 2,170 · total 87,360' })
  const rows = block.getAllByRole('row').slice(1).map(row => within(row).getAllByRole('cell').map(cell => cell.textContent))
  expect(rows).toEqual([
    ['Lower', '0', '0', '1,085', '108, 108, 108, 108, 108, 105, 108, 108, 108, 116'],
    ['Upper', '50', '600', '1,085', '108, 108, 108, 108, 108, 105, 108, 108, 108, 116'],
  ])
  await userEvent.click(screen.getByRole('tab', { name: 'Resource allocation' }))
  const traces = plotData('plot-Resource allocation')
  expect(traces.map(trace => trace.name)).toEqual(['Carrier 1', 'Pilots 1', 'NB-IoT in-band · RB 0', 'NB-IoT in-band · RB 50'])
  expect(traces[2]!.x).toEqual(Array.from({ length: 12 }, (_, i) => mhz(-312 + i)))
  expect(traces[3]!.x).toEqual(Array.from({ length: 12 }, (_, i) => mhz(288 + i)))
  expect(traces[0]!.x[0]).toBe(mhz(-312))
})

test('the result is named like the chip that selected it, not by its preset id; a note the server also lists as warning is shown once', async () => {
  const signal = twoNumerologies()
  const warning = signal.analysis.nr_test_model!.warnings?.[0] ?? ''
  expect(warning).not.toBe('')
  signal.analysis.notes = [warning, 'A note only the notes list has.']
  renderWithProviders(<SignalGeneratorPlots result={signal} stale={false} />)
  const name = screen.getByTestId('generator-result-name')
  expect(name).toHaveTextContent(/^TM1\.1 · 10\+20 MHz · 15\/30 kHz · FDD/)
  expect(name).not.toHaveTextContent(signal.config.preset_id)
  expect(within(name).getByText(/^TM1\.1/)).toHaveAttribute('title', signal.config.preset_id)
  await userEvent.click(screen.getByRole('tab', { name: 'Metrics & provenance' }))
  expect(screen.getAllByText(warning)).toHaveLength(1)
  expect(screen.getByText('A note only the notes list has.')).toBeVisible()
})

test('a name given by the page (entries that would read the same are told apart there) replaces the default name; the preset id stays its tooltip', () => {
  const signal = twoNumerologies()
  renderWithProviders(<SignalGeneratorPlots result={signal} stale={false} name="TM1.1 · 10+20 MHz · 15/30 kHz · FDD · MATLAB 5G Toolbox" />)
  const name = screen.getByTestId('generator-result-name')
  expect(name).toHaveTextContent(/^TM1\.1 · 10\+20 MHz · 15\/30 kHz · FDD · MATLAB 5G Toolbox · [\d,]+ I\/Q samples/)
  expect(within(name).getByText(/^TM1\.1/)).toHaveAttribute('title', signal.config.preset_id)
})

test('a matrix signal keeps the bin-index allocation, the numerology label and the generic scope text without a carrier table', async () => {
  renderWithProviders(<SignalGeneratorPlots result={matrixSignal()} stale={false} />)
  expect(screen.getByText('Numerology preset')).toBeVisible()
  await userEvent.click(screen.getByRole('tab', { name: 'Resource allocation' }))
  const plot = screen.getByTestId('plot-Resource allocation')
  expect(plot).toHaveAttribute('data-x-title', 'Signed subcarrier index')
  expect(plot).toHaveAttribute('data-y-title', 'Channel')
  expect(plotData('plot-Resource allocation')[0]).toMatchObject({ name: 'Channel 1', x: [-2, -1, 1, 2] })
  await userEvent.click(screen.getByRole('tab', { name: 'EVM trends' }))
  expect(screen.getByTestId('plot-EVM by data subcarrier')).toHaveAttribute('data-x-title', 'Signed subcarrier index')
  expect(plotData('plot-EVM by data subcarrier')[0]!.x).toEqual([-2, -1, 1, 2])
  await userEvent.click(screen.getByRole('tab', { name: 'Metrics & provenance' }))
  expect(screen.queryByTestId('nrtm-result-carriers')).not.toBeInTheDocument()
  expect(screen.queryByText('Test model')).not.toBeInTheDocument()
  expect(screen.getByText(/no complete protocol frames or conformance certification/)).toBeVisible()
  expect(screen.getByTestId('generator-result-name')).toHaveTextContent(new RegExp('^' + matrixSignal().config.preset_id + ' ·'))
})

test('a capture longer than the standard reads "15 subframes (standard 10)", never "15 of 10 subframes"; shorter ones keep "of"', async () => {
  const signal = twoNumerologies()
  signal.analysis.nr_test_model = { ...signal.analysis.nr_test_model!, subframes: 15, standard_subframes: 10 }
  const { unmount } = renderWithProviders(<SignalGeneratorPlots result={signal} stale={false} />)
  await userEvent.click(screen.getByRole('tab', { name: 'Metrics & provenance' }))
  const metrics = screen.getAllByRole('row').map(row => row.textContent ?? '')
  expect(metrics).toContainEqual(expect.stringContaining('Captured length15 subframes (standard 10)'))
  expect(metrics.join('\n')).not.toMatch(/15 of 10 subframes/)
  unmount()
  const shorter = twoNumerologies()
  shorter.analysis.nr_test_model = { ...shorter.analysis.nr_test_model!, subframes: 8, standard_subframes: 10 }
  renderWithProviders(<SignalGeneratorPlots result={shorter} stale={false} />)
  await userEvent.click(screen.getByRole('tab', { name: 'Metrics & provenance' }))
  expect(screen.getAllByRole('row').map(row => row.textContent ?? '')).toContainEqual(expect.stringContaining('Captured length8 of 10 subframes'))
})
