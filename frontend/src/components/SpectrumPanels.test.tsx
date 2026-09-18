import { fireEvent, screen, within } from '@testing-library/react'
import { vi } from 'vitest'
import { useEffect, useRef, useState } from 'react'
import { renderWithProviders } from '@/test/utils'
import { SpectrumPanels } from './SpectrumPanels'
import type { SpectrumPlotProps } from './SpectrumPlot'
import { spectrumGroups, spectrumLegend } from './spectrumNodes'

vi.mock('./SpectrumPlot', () => ({ SpectrumPlot: function TestPlot(props: SpectrumPlotProps) {
  const { traces, onRendered } = props
  const callback = useRef(onRendered)
  useEffect(() => { callback.current = onRendered }, [onRendered])
  useEffect(() => { callback.current?.(traces.length ? 10 : 0) }, [traces])
  return <section aria-label={props.title}>
  {props.traces.map(tr => <span key={tr.name}>{tr.name}</span>)}
  <button onClick={() => props.onVisibilityChange?.(props.traces.map(() => false))}>Hide this position</button>
  <button onClick={() => props.onViewportChange?.({ x: [-1, 1], y: [-90, -40], autoX: false, autoY: false, dragmode: 'pan' })}>Zoom this position</button>
</section> } }))

const traces = [
  { name: 'target input x', role: 'input', stage: 'x' },
  { name: 'u = DPD(x)', role: 'predistorted', stage: 'u' },
  { name: 'linear target gain*x', role: 'reference' },
  { name: 'with DPD: PA_surrogate(u)', role: 'primary' },
  { name: 'surrogate without DPD', role: 'baseline' },
  { name: 'measured PA without DPD', role: 'baseline', source: 'synthetic dataset' },
].map(tr => ({ ...tr, psdDb: [-90, -40, -90] }))

test('three locations retain all PA output comparisons and independent legend/zoom events', () => {
  const visibility = vi.fn(), viewport = vi.fn()
  renderWithProviders(<SpectrumPanels frequencyHz={[-1e6, 0, 1e6]} traces={traces} onVisibilityChange={visibility} onViewportChange={viewport} />)
  const input = screen.getByRole('region', { name: 'DPD Input · PSD' })
  const drive = screen.getByRole('region', { name: 'DPD Output / PA Input · PSD' })
  const output = screen.getByRole('region', { name: 'PA Output · PSD' })
  expect(within(input).getByText('Input x')).toBeInTheDocument()
  expect(within(drive).getByText('Predistorted u')).toBeInTheDocument()
  expect(within(output).getByText('Without DPD · synthetic data')).toBeInTheDocument()
  expect(within(output).getByText('With DPD · PA model')).toBeInTheDocument()
  expect(within(output).getByText('Without DPD · PA model')).toBeInTheDocument()
  expect(within(drive).queryByText('With DPD · PA model')).not.toBeInTheDocument()
  const explanation = screen.getByTestId('pa-model-explanation')
  expect(explanation.closest('[data-signal-node]')).toHaveAttribute('data-signal-node', 'pa_output')
  expect(explanation).toHaveTextContent('original input → DPD → same PA model')
  expect(explanation).toHaveTextContent('predictions')
  fireEvent.click(within(drive).getByRole('button', { name: 'Hide this position' }))
  expect(visibility).toHaveBeenCalledWith([true, false, true, true, true, true])
  fireEvent.click(within(input).getByRole('button', { name: 'Zoom this position' }))
  expect(viewport).toHaveBeenCalledWith(expect.objectContaining({ x: [-1, 1] }), 'dpd_input')
})

test('PA datasets remain PA-only and unlabelled legacy probes are not guessed', () => {
  expect(spectrumGroups([{ name: 'PA input x', role: 'input', stage: 'x' }, { name: 'measured PA output', role: 'reference' }]).map(g => g.node)).toEqual(['pa_input', 'pa_output'])
  expect(spectrumGroups([{ name: 'unknown probe' }])[0]?.node).toBe('unknown')
  expect(spectrumLegend({ name: 'measured PA output', source: 'synthetic dataset' })).toBe('Dataset output · synthetic')
})

test('hardware measurements do not acquire a PA-model explanation', () => {
  renderWithProviders(<SpectrumPanels frequencyHz={[-1e6, 0, 1e6]} traces={[
    { name: 'measured PA output with DPD', role: 'primary', source: 'measured capture', psdDb: [-80, -40, -80] },
    { name: 'measured PA output without DPD', role: 'baseline', source: 'measured capture', psdDb: [-70, -40, -70] },
  ]} />)
  expect(screen.getByText('With DPD · measured')).toBeInTheDocument()
  expect(screen.getByText('Without DPD · measured')).toBeInTheDocument()
  expect(screen.queryByTestId('pa-model-explanation')).not.toBeInTheDocument()
})


test('reporting completed draws can update the parent without a render feedback loop', async () => {
  function Parent() {
    const [draws, setDraws] = useState(0)
    return <><output data-testid="draws">{draws}</output><SpectrumPanels frequencyHz={[-1e6, 0, 1e6]} traces={traces} onRendered={() => setDraws(n => n + 1)} /></>
  }
  renderWithProviders(<Parent />)
  expect(await screen.findByTestId('draws')).toHaveTextContent('1')
})
