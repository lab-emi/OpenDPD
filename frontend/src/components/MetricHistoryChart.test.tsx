import { render } from '@testing-library/react'
import { expect, test, vi } from 'vitest'
import { MetricHistoryChart } from './MetricHistoryChart'

const reactMock = vi.fn(() => Promise.resolve())
vi.mock('plotly.js-basic-dist-min', () => ({ default: { react: reactMock, purge: vi.fn() } }))

test('one trace per split with distinct marker symbols and dashes', async () => {
  const points = [
    { epoch: 0, split: 'val', values: { NMSE: -10 } },
    { epoch: 1, split: 'val', values: { NMSE: -12 } },
    { epoch: 1, split: 'test', values: { NMSE: -11 } },
  ]
  render(<MetricHistoryChart points={points} metric="NMSE" />)
  await vi.waitFor(() => expect(reactMock).toHaveBeenCalledTimes(1))
  const data = (reactMock.mock.calls[0] as unknown[])[1] as Array<{ name: string; marker: { symbol: string }; line: { dash: string } }>
  expect(data.map((tr) => tr.name)).toEqual(['val NMSE', 'test NMSE'])
  expect(data[0]?.marker.symbol).not.toBe(data[1]?.marker.symbol)
  expect(data[0]?.line.dash).not.toBe(data[1]?.line.dash)
})
