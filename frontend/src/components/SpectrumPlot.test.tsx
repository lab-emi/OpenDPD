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
