import { fireEvent, screen, waitFor } from '@testing-library/react'
import { vi } from 'vitest'
import paMock from '@mocks/result_pa_modeling_mock.json'
import type { EvaluationResult, FigureSpec, SavedFigure } from '@/api/types'
import { mockApi, renderWithProviders } from '@/test/utils'
import { nearestBin, SpectrumReview } from './SpectrumReview'
import type { SpectrumPlotProps } from './SpectrumPlot'

vi.mock('./SpectrumPlot', () => ({ SpectrumPlot: (props: SpectrumPlotProps) => <div>
  <button onClick={() => props.onViewportChange?.({ x: [-10, 12], y: [-100, -50], autoX: false, autoY: false, dragmode: 'pan' })}>Zoom test spectrum</button>
  <button onClick={() => props.onVisibilityChange?.([false, true])}>Hide input trace</button>
  <output data-testid="restored-ranges">{JSON.stringify([props.xRange, props.yRange])}</output>
</div> }))
const result = paMock.data as unknown as EvaluationResult
const data = { axis: 'hz', frequency: [-20e6, 0, 20e6], traces: [
  { name: 'input', role: 'reference', psd_db: [-90, -80, -90] },
  { name: 'output', role: 'primary', psd_db: [-70, -60, -70] },
], estimator: 'stored Welch display' }

test('a saved review preserves the actual viewport, visible traces and cursor after reopening', async () => {
  let stored: SavedFigure | null = null
  const { calls } = mockApi({
    'GET /api/v1/artifacts/run-pa-0001/plot-spectrum': () => data,
    'GET /api/v1/figures': () => stored ? [stored] : [],
    'POST /api/v1/figures': (_url, init) => {
      stored = { figure_id: 'fig-example', spec: JSON.parse(String(init.body)) as FigureSpec, bindings: [], created_at: '2026-09-13T12:00:00Z' }
      return stored
    },
    'GET /api/v1/figures/fig-example': () => stored,
  })
  const first = renderWithProviders(<SpectrumReview results={[result]} referenceRunId="run-pa-0001" />)
  fireEvent.click(await screen.findByRole('button', { name: 'Zoom test spectrum' }))
  fireEvent.click(screen.getByRole('button', { name: 'Hide input trace' }))
  fireEvent.change(screen.getByLabelText('Frequency cursor (MHz)'), { target: { value: '9' } })
  fireEvent.change(screen.getByLabelText('Figure title'), { target: { value: 'My RF review' } })
  fireEvent.click(screen.getByRole('button', { name: 'Save view' }))
  await screen.findByRole('link', { name: 'Export figure + data' })
  const request = calls.find(c => c.method === 'POST')!.body as FigureSpec
  expect(request.panels[0]?.x_range).toEqual([-10, 12])
  expect(request.panels[0]?.y_range).toEqual([-100, -50])
  expect(request.panels[0]?.traces.map(t => t.visible)).toEqual([false, true])
  expect(request.panels[0]?.cursor_x).toBe(9)
  expect(request.profiles).toEqual({ 'run-pa-0001': result.metric_profile_id })
  first.unmount()
  renderWithProviders(<SpectrumReview results={[result]} referenceRunId="run-pa-0001" />)
  await screen.findByRole('button', { name: 'Zoom test spectrum' })
  fireEvent.mouseDown(screen.getByRole('combobox', { name: 'Saved views' }))
  fireEvent.click(await screen.findByRole('option', { name: /My RF review/ }))
  await waitFor(() => expect(screen.getByTestId('restored-ranges')).toHaveTextContent('[[-10,12],[-100,-50]]'))
  expect(screen.getByLabelText('Frequency cursor (MHz)')).toHaveValue(9)
  expect(screen.getByLabelText('Figure title')).toHaveValue('My RF review')
})

test('cursor picks each trace’s actual nearest bin and refuses out-of-capture readings', () => {
  expect(nearestBin([-30, -5, 20, 60], 19)).toBe(2)
  expect(nearestBin([-30, -5, 20, 60], -17.5)).toBe(0)
  expect(nearestBin([-30, -5, 20, 60], 61)).toBe(-1)
  expect(nearestBin([], 1)).toBe(-1)
})
