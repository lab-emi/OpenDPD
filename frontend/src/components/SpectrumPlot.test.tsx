import { render } from '@testing-library/react'
import { vi } from 'vitest'
import { SpectrumPlot } from './SpectrumPlot'

const reactMock = vi.fn(() => Promise.resolve())
vi.mock('plotly.js-basic-dist-min', () => ({ default: { react: reactMock, purge: vi.fn() } }))

test('re-rendering with an inline bands literal does not redraw the plot', async () => {
  const f = Float64Array.from({ length: 64 }, (_, i) => i * 1e6)
  const traces = [{ name: 'a', psdDb: Float64Array.from({ length: 64 }, () => -50) }]
  const onRendered = vi.fn()
  const ui = (cb: () => void) => <SpectrumPlot frequencyHz={f} traces={traces} bands={{ main: [0, 1e6], adjacent: [] }} onRendered={cb} />
  const { rerender } = render(ui(onRendered))
  await vi.waitFor(() => expect(reactMock).toHaveBeenCalledTimes(1))
  rerender(ui(() => onRendered()))
  rerender(ui(() => onRendered()))
  await new Promise((r) => setTimeout(r, 20))
  expect(reactMock).toHaveBeenCalledTimes(1)
})

test('every trace differs in dash pattern, not only in colour', async () => {
  reactMock.mockClear()
  const f = Float64Array.from({ length: 16 }, (_, i) => i * 1e6)
  const traces = ['input', 'PA output', 'DPD+PA'].map((name) => ({ name, psdDb: Float64Array.from({ length: 16 }, () => -40) }))
  render(<SpectrumPlot frequencyHz={f} traces={traces} />)
  await vi.waitFor(() => expect(reactMock).toHaveBeenCalledTimes(1))
  const data = (reactMock.mock.calls[0] as unknown[])[1] as Array<{ line: { dash: string } }>
  expect(new Set(data.map((tr) => tr.line.dash)).size).toBe(3)
})
