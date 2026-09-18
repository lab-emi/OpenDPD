import { fireEvent, screen } from '@testing-library/react'
import { vi } from 'vitest'
import { renderWithProviders } from '@/test/utils'
import type { PlotlyChartProps } from './PlotlyChart'
import { OutputWaveforms, type OutputWaveform } from './OutputWaveforms'

const drawn = new Map<string, PlotlyChartProps>()
vi.mock('./PlotlyChart', () => ({ PlotlyChart: (props: PlotlyChartProps) => {
  drawn.set(props['data-testid']!, props)
  return <div data-testid={props['data-testid']} />
} }))

const series: OutputWaveform[] = [
  { name: 'target input x', role: 'input' },
  { name: 'u = DPD(x)', role: 'predistorted' },
  { name: 'linear target gain*x', role: 'reference' },
  { name: 'with DPD: PA_surrogate(u)', role: 'primary' },
  { name: 'surrogate without DPD', role: 'baseline' },
  { name: 'measured PA without DPD', role: 'baseline', source: 'synthetic dataset' },
].map((s, index) => ({ ...s, i: Float64Array.from({ length: 512 }, (_, k) => k / 512 + index), q: Float64Array.from({ length: 512 }, (_, k) => -k / 512 - index) }))

const plots = () => ['i', 'q'].map(c => drawn.get(`output-waveform-${c}`)!)

test('DPD compares only output and reference with separate components and unchanged samples', () => {
  renderWithProviders(<OutputWaveforms start={300} series={series} />)
  for (const [index, plot] of plots().entries()) {
    const component = index ? 'q' : 'i'
    expect(plot.traces).toHaveLength(2)
    expect(plot.traces[0]!.y).toBe(series[2]![component])
    expect(plot.traces[1]!.y).toBe(series[3]![component])
    expect(Array.from(plot.traces[0]!.x)).toEqual(Array.from({ length: 512 }, (_, k) => 300 + k))
    expect(plot.layout!.xaxis!.range).toEqual([300, 427])
  }
  expect(plots()[0]!.layout!.yaxis!.range).toEqual(plots()[1]!.layout!.yaxis!.range)
  expect(plots()[0]!.traces.map(s => s.name)).toEqual(['Linear target g·x', 'With DPD · PA model'])
  fireEvent.change(screen.getByLabelText('Initial window'), { target: { value: 'full' } })
  expect(plots()[0]!.layout!.xaxis!.range).toEqual([300, 811])
  expect(plots()[0]!.traces[0]!.y).toBe(series[2]!.i)
})

test('every input and baseline remains inspectable alone, with its source label', () => {
  renderWithProviders(<OutputWaveforms start={0} series={series} />)
  for (const s of series) {
    fireEvent.change(screen.getByLabelText('Waveform'), { target: { value: `signal:${s.name}` } })
    expect(plots()[0]!.traces).toHaveLength(1)
    expect(plots()[1]!.traces).toHaveLength(1)
    expect(plots()[0]!.traces[0]!.y).toBe(s.i)
    expect(plots()[1]!.traces[0]!.y).toBe(s.q)
  }
  expect(plots()[0]!.traces[0]!.name).toBe('Without DPD · synthetic data')
  fireEvent.change(screen.getByLabelText('Waveform'), { target: { value: 'comparison' } })
  expect(plots()[0]!.traces).toHaveLength(2)
})

test('PA roles, including synthetic references, determine the pair regardless of trace order', () => {
  const reference = { ...series[0]!, name: 'measured PA output', role: 'reference', source: 'synthetic dataset' }
  const output = { ...series[1]!, name: 'PA model output', role: 'primary' }
  renderWithProviders(<OutputWaveforms start={0} series={[output, series[0]!, reference]} />)
  expect(plots()[0]!.traces.map(s => s.name)).toEqual(['Dataset output · synthetic', 'PA model prediction'])
  expect(plots()[0]!.traces[0]!.y).toBe(reference.i)
})

test('live updates preserve the selected signal; final artifacts recover from renamed signals', () => {
  const live = [{ ...series[0]!, name: 'Input x' }, { ...series[2]!, name: 'Linear target' }, { ...series[3]!, name: 'DPD → PA surrogate' }]
  const { rerender } = renderWithProviders(<OutputWaveforms start={0} series={live} viewKey="run" />)
  fireEvent.change(screen.getByLabelText('Waveform'), { target: { value: 'signal:Input x' } })
  const key = plots()[0]!.viewKey
  const updated = live.map(s => ({ ...s, i: Float64Array.from(s.i, v => v + 1) }))
  rerender(<OutputWaveforms start={0} series={updated} viewKey="run" />)
  expect(plots()[0]!.viewKey).toBe(key)
  expect(plots()[0]!.traces[0]!.y).toBe(updated[0]!.i)
  rerender(<OutputWaveforms start={0} series={series} viewKey="run:final" />)
  expect(screen.getByLabelText('Waveform')).toHaveValue('comparison')
  expect(plots()[0]!.traces).toHaveLength(2)
})

test('known legacy names work without roles, while an unknown capture is never called a comparison', () => {
  const legacy = series.slice(2, 4).map(({ name, i, q }) => ({ name, i, q }))
  const { rerender } = renderWithProviders(<OutputWaveforms start={0} series={legacy} />)
  expect(plots()[0]!.traces).toHaveLength(2)
  rerender(<OutputWaveforms start={0} series={[{ name: 'comparison', i: [0], q: [0] }]} />)
  expect(screen.queryByRole('option', { name: 'Output vs. reference' })).not.toBeInTheDocument()
  expect(plots()[0]!.traces).toHaveLength(1)
  expect(plots()[0]!.traces[0]!.name).toBe('comparison')
  expect(plots()[0]!.layout!.yaxis!.range).toEqual([-1, 1])
  expect(screen.queryByLabelText('Initial window')).not.toBeInTheDocument()
  rerender(<OutputWaveforms start={0} series={[]} />)
  expect(screen.queryByTestId('output-waveforms')).not.toBeInTheDocument()
})
