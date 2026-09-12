import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { PlotlyChart, type PlotTrace } from './PlotlyChart'
import type { PlotElement, Range } from './plotInteractions'
import { ThemeProvider } from '@mui/material/styles'
import { themeFor } from '@/theme'

const api = vi.hoisted(() => ({ react: vi.fn(), relayout: vi.fn(), purge: vi.fn(), Plots: { resize: vi.fn() } }))
vi.mock('@/vendor/plotly-scatter-strict.cjs', () => ({ default: api }))
vi.mock('plotly.js-basic-dist-min', () => ({ default: api }))

beforeEach(() => {
  vi.clearAllMocks()
  api.react.mockImplementation(async (element: PlotElement, _data: PlotTrace[], layout: { xaxis: { range?: Range; autorange?: boolean }; yaxis: { range?: Range; autorange?: boolean }; dragmode: string }) => {
    element['_fullLayout'] = {
      xaxis: { range: layout.xaxis.range ?? [0, 100], autorange: layout.xaxis.autorange ?? true },
      yaxis: { range: layout.yaxis.range ?? [-10, 10], autorange: layout.yaxis.autorange ?? true },
      dragmode: layout.dragmode,
    }
  })
  api.relayout.mockImplementation(async (element: PlotElement, update: Record<string, unknown>) => {
    element['_fullLayout'] = {
      xaxis: { range: update['xaxis.range'] as Range, autorange: !!update['xaxis.autorange'] },
      yaxis: { range: update['yaxis.range'] as Range, autorange: !!update['yaxis.autorange'] },
      dragmode: update.dragmode as string,
    }
  })
})

const traces: PlotTrace[] = [{ x: [0, 1, 2], y: [2, 3, 4] }]

test('changing appearance repaints an existing chart without clearing its data or manual viewport', async () => {
  const { rerender } = render(<ThemeProvider theme={themeFor('en', 'light')}><PlotlyChart title="signal" traces={traces} viewKey="a" /></ThemeProvider>)
  await waitFor(() => expect(api.react).toHaveBeenCalledTimes(1))
  const element = screen.getByRole('figure', { name: 'signal' })
  const light = api.react.mock.lastCall![2]
  fireEvent.keyDown(element, { key: 'ArrowRight' })
  await waitFor(() => expect(api.relayout).toHaveBeenCalledTimes(1))
  rerender(<ThemeProvider theme={themeFor('en', 'dark')}><PlotlyChart title="signal" traces={traces} viewKey="a" /></ThemeProvider>)
  await waitFor(() => expect(api.react).toHaveBeenCalledTimes(2))
  const dark = api.react.mock.lastCall![2]
  expect(dark.paper_bgcolor).not.toBe(light.paper_bgcolor)
  expect(dark.colorway).not.toEqual(light.colorway)
  expect(dark.font.color).not.toBe(light.font.color)
  expect(dark.xaxis.range).toEqual([5, 105])
  expect(api.react.mock.lastCall![1][0].y).toBe(traces[0]!.y)
  expect(screen.getByRole('figure', { name: 'signal' })).toBe(element)
  expect(api.purge).not.toHaveBeenCalled()
})
const dense: PlotTrace[] = [{ x: Array.from({ length: 1200 }, (_, i) => i), y: Array.from({ length: 1200 }, (_, i) => i % 20), mode: 'markers' }]
const enableWebGL = () => {
  vi.stubGlobal('WebGLRenderingContext', vi.fn())
  vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({ getExtension: () => ({ loseContext: vi.fn() }) } as unknown as WebGLRenderingContext)
}
afterEach(() => vi.unstubAllGlobals())

test('new chart data waits for an in-flight gesture draw before React redraws', async () => {
  let finish!: () => void
  api.relayout.mockImplementationOnce(() => new Promise<void>(resolve => { finish = resolve }))
  const { rerender } = render(<PlotlyChart title="signal" traces={traces} viewKey="a" />)
  await waitFor(() => expect(api.react).toHaveBeenCalledTimes(1))
  fireEvent.keyDown(screen.getByRole('figure', { name: 'signal' }), { key: '+' })
  await waitFor(() => expect(api.relayout).toHaveBeenCalledTimes(1))
  rerender(<PlotlyChart title="signal" traces={[...traces]} viewKey="a" />)
  await new Promise(resolve => setTimeout(resolve, 30))
  expect(api.react).toHaveBeenCalledTimes(1)
  finish()
  await waitFor(() => expect(api.react).toHaveBeenCalledTimes(2))
  const layout = api.react.mock.lastCall![2]
  expect(layout.xaxis.autorange).toBe(false)
  expect(layout.xaxis.range[1] - layout.xaxis.range[0]).toBeCloseTo(100 / 1.2)
})

test('live snapshots wait until every finger lifts without resetting the touch viewport', async () => {
  const { rerender } = render(<PlotlyChart title="signal" traces={traces} />)
  await waitFor(() => expect(api.react).toHaveBeenCalledTimes(1))
  const figure = screen.getByRole('figure', { name: 'signal' })
  fireEvent.touchStart(figure, { touches: [{ identifier: 1 }, { identifier: 2 }] })
  fireEvent.keyDown(figure, { key: '+' })
  await waitFor(() => expect(api.relayout).toHaveBeenCalledTimes(1))
  const latest = [{ x: [0, 1, 2], y: [5, 6, 7] }]
  rerender(<PlotlyChart title="signal" traces={latest} />)
  await act(async () => { await new Promise(resolve => setTimeout(resolve, 30)) })
  expect(api.react).toHaveBeenCalledTimes(1)
  fireEvent.touchEnd(figure, { touches: [{ identifier: 1 }] })
  expect(api.react).toHaveBeenCalledTimes(1)
  fireEvent.touchEnd(figure, { touches: [] })
  await waitFor(() => expect(api.react).toHaveBeenCalledTimes(2))
  expect(api.react.mock.lastCall![1][0].y).toEqual([5, 6, 7])
  expect(api.react.mock.lastCall![2].xaxis.range[1] - api.react.mock.lastCall![2].xaxis.range[0]).toBeCloseTo(100 / 1.2)
})

test('unmount waits for the gesture draw before purging its graph', async () => {
  let finish!: () => void
  api.relayout.mockImplementationOnce(() => new Promise<void>(resolve => { finish = resolve }))
  const { unmount } = render(<PlotlyChart title="signal" traces={traces} />)
  await waitFor(() => expect(api.react).toHaveBeenCalledTimes(1))
  fireEvent.keyDown(screen.getByRole('figure', { name: 'signal' }), { key: '+' })
  await waitFor(() => expect(api.relayout).toHaveBeenCalledTimes(1))
  unmount()
  await new Promise(resolve => setTimeout(resolve, 30))
  expect(api.purge).not.toHaveBeenCalled()
  finish()
  await waitFor(() => expect(api.purge).toHaveBeenCalledTimes(1))
})

test('live updates preserve the camera but a different signal starts at its own bounds', async () => {
  const { rerender } = render(<PlotlyChart title="signal" traces={traces} viewKey="a" />)
  await waitFor(() => expect(api.react).toHaveBeenCalledTimes(1))
  fireEvent.keyDown(screen.getByRole('figure', { name: 'signal' }), { key: 'ArrowRight' })
  await waitFor(() => expect(api.relayout).toHaveBeenCalledTimes(1))
  const update = [...traces, { x: [3], y: [5] }]
  rerender(<PlotlyChart title="signal" traces={update} viewKey="a" />)
  await waitFor(() => expect(api.react).toHaveBeenCalledTimes(2))
  expect(api.react.mock.lastCall![2].xaxis.range).toEqual([5, 105])
  rerender(<PlotlyChart title="other signal" traces={update} viewKey="b" />)
  await waitFor(() => expect(api.react).toHaveBeenCalledTimes(3))
  expect(api.react.mock.lastCall![2].xaxis.range).toBeUndefined()
})

test('uses WebGL for dense scatter without dropping any data points', async () => {
  enableWebGL()
  render(<PlotlyChart title="dense" traces={dense} />)
  // This is the first lazy GL import. Settle Vitest's module transformation
  // before asserting the draw; browser performance has its own real test.
  await act(() => vi.dynamicImportSettled())
  await waitFor(() => expect(api.react).toHaveBeenCalledTimes(1))
  const data = api.react.mock.lastCall![1]
  expect(data[0].type).toBe('scattergl')
  expect(data[0].x).toBe(dense[0]!.x)
  expect(data[0].y).toBe(dense[0]!.y)
})

test('uses SVG when WebGL is unavailable', async () => {
  vi.stubGlobal('WebGLRenderingContext', undefined)
  render(<PlotlyChart title="dense" traces={dense} />)
  await waitFor(() => expect(api.react).toHaveBeenCalledTimes(1))
  expect(api.react.mock.lastCall![1][0].type).toBe('scatter')
  expect(api.react.mock.lastCall![1][0].x).toHaveLength(1200)
})

test('recovers from WebGL context loss with the same points and camera in SVG', async () => {
  enableWebGL()
  render(<PlotlyChart title="dense" traces={dense} />)
  await waitFor(() => expect(api.react).toHaveBeenCalledTimes(1))
  const figure = screen.getByRole('figure', { name: 'dense' })
  fireEvent.keyDown(figure, { key: 'ArrowRight' })
  await waitFor(() => expect(api.relayout).toHaveBeenCalledTimes(1))
  const canvas = document.createElement('canvas'); figure.append(canvas)
  fireEvent(canvas, new Event('webglcontextlost', { cancelable: true }))
  await waitFor(() => expect(api.react).toHaveBeenCalledTimes(2))
  expect(api.react.mock.lastCall![1][0].type).toBe('scatter')
  expect(api.react.mock.lastCall![1][0].x).toHaveLength(1200)
  expect(api.react.mock.lastCall![2].xaxis.range).toEqual([5, 105])
})

test('retries a failed WebGL initialization using SVG', async () => {
  enableWebGL()
  api.react.mockRejectedValueOnce(new Error('context creation failed'))
  render(<PlotlyChart title="dense" traces={dense} />)
  await waitFor(() => expect(api.react).toHaveBeenCalledTimes(2))
  expect(api.react.mock.lastCall![1][0].type).toBe('scatter')
  expect(screen.getByRole('figure', { name: 'dense' })).toBeVisible()
  expect(screen.queryByText('context creation failed')).not.toBeInTheDocument()
})

test('recovery can be disabled per chart and the enlarged view shares that setting', async () => {
  render(<><PlotlyChart title="first" traces={traces} data-testid="first" /><PlotlyChart title="second" traces={traces} data-testid="second" /></>)
  await waitFor(() => expect(api.react).toHaveBeenCalledTimes(2))
  fireEvent.click(within(screen.getByTestId('first')).getByRole('button', { name: 'Plot controls' }))
  const toggle = screen.getByLabelText('Auto-fit empty views')
  expect(toggle).toBeChecked()
  expect(screen.getByRole('combobox', { name: 'Empty area threshold' })).toHaveTextContent('85%')
  fireEvent.click(toggle)
  expect(screen.getByRole('combobox', { name: 'Empty area threshold' })).toHaveAttribute('aria-disabled', 'true')
  fireEvent.click(screen.getByRole('button', { name: 'Close plot help' }))
  await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Plot controls' })).not.toBeInTheDocument())
  fireEvent.click(within(screen.getByTestId('second')).getByRole('button', { name: 'Plot controls' }))
  expect(screen.getByLabelText('Auto-fit empty views')).toBeChecked()
  fireEvent.click(screen.getByRole('button', { name: 'Close plot help' }))
  await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Plot controls' })).not.toBeInTheDocument())
  fireEvent.click(screen.getByRole('button', { name: 'Enlarge chart: first' }))
  fireEvent.click(within(screen.getByRole('dialog', { name: 'first' })).getByRole('button', { name: 'Plot controls' }))
  expect(screen.getByLabelText('Auto-fit empty views')).not.toBeChecked()
})

test('a fit in the enlarged view replaces Plotly’s stale manual autorange state on return', async () => {
  render(<PlotlyChart title="signal" traces={traces} />)
  await waitFor(() => expect(api.react).toHaveBeenCalledTimes(1))
  fireEvent.keyDown(screen.getByRole('figure', { name: 'signal' }), { key: 'ArrowRight' })
  await waitFor(() => expect(api.relayout).toHaveBeenCalledTimes(1))
  fireEvent.click(screen.getByRole('button', { name: 'Enlarge chart: signal' }))
  await waitFor(() => expect(api.react).toHaveBeenCalledTimes(2))
  fireEvent.keyDown(within(screen.getByRole('dialog', { name: 'signal' })).getByRole('figure'), { key: 'Home' })
  await waitFor(() => expect(api.relayout).toHaveBeenCalledTimes(2))
  const normal = api.react.getMockImplementation()!
  api.react.mockImplementationOnce(async (element: PlotElement, data: PlotTrace[], layout: object, config: object) => {
    await normal(element, data, layout, config)
    element['_fullLayout']!.xaxis = { range: [500, 600], autorange: false }
    element['_fullLayout']!.yaxis!.autorange = false
  })
  fireEvent.click(screen.getByRole('button', { name: 'Close enlarged chart' }))
  await waitFor(() => expect(api.relayout).toHaveBeenCalledTimes(3))
  expect(api.relayout.mock.lastCall![1]).toMatchObject({ 'xaxis.autorange': true, 'yaxis.autorange': true })
})
