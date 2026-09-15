import { act, fireEvent, render } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { ApiError, reportWebActivity } from '@/api/client'
import { useWebActivity } from './useWebActivity'

vi.mock('@/api/client', async original => ({ ...await original<typeof import('@/api/client')>(), WEB_MODE: true, reportWebActivity: vi.fn() }))
const info = { authenticated: true, version: '2.2.8', mode: 'web' as const, idle_expires_at: '2026-09-15T14:00:00Z' }
let visibility: DocumentVisibilityState
beforeEach(() => {
  vi.useFakeTimers()
  visibility = 'visible'
  vi.spyOn(document, 'visibilityState', 'get').mockImplementation(() => visibility)
  vi.mocked(reportWebActivity).mockReset().mockResolvedValue(info)
})
afterEach(() => { vi.useRealTimers(); vi.restoreAllMocks() })
function Probe() { useWebActivity(true); return <button>Studio action</button> }
function setup() {
  const client = new QueryClient()
  return { client, ...render(<QueryClientProvider client={client}><Probe /></QueryClientProvider>) }
}
const advance = async (ms: number) => { await act(async () => { await vi.advanceTimersByTimeAsync(ms) }) }

test('an open but untouched tab has no periodic keep-alive', async () => {
  const { client } = setup()
  await advance(1)
  expect(reportWebActivity).toHaveBeenCalledTimes(1)
  expect(client.getQueryData(['session'])).toEqual(info)
  await advance(2 * 3600_000)
  expect(reportWebActivity).toHaveBeenCalledTimes(1)
})

test('frequent pointer and keyboard activity coalesces without renewing in the background', async () => {
  setup(); await advance(1)
  for (let i = 0; i < 100; i++) fireEvent.pointerMove(window)
  fireEvent.keyDown(window, { key: 'ArrowRight' })
  await advance(59_998)
  expect(reportWebActivity).toHaveBeenCalledTimes(1)
  await advance(2)
  expect(reportWebActivity).toHaveBeenCalledTimes(2)
  visibility = 'hidden'; fireEvent(document, new Event('visibilitychange'))
  fireEvent.pointerDown(window)
  await advance(600_000)
  expect(reportWebActivity).toHaveBeenCalledTimes(2)
  visibility = 'visible'; fireEvent(document, new Event('visibilitychange'))
  await advance(1)
  expect(reportWebActivity).toHaveBeenCalledTimes(3)
})

test('failed reports do not turn into automatic retries that extend inactivity', async () => {
  vi.mocked(reportWebActivity).mockRejectedValue(new ApiError(429, 'rate_limited', 'busy'))
  setup(); await advance(600_000)
  expect(reportWebActivity).toHaveBeenCalledTimes(1)
  fireEvent.wheel(window); await advance(1)
  expect(reportWebActivity).toHaveBeenCalledTimes(2)
})

test('a hidden tab never renews until it is brought to the foreground', async () => {
  visibility = 'hidden'
  setup(); await advance(600_000)
  expect(reportWebActivity).not.toHaveBeenCalled()
  visibility = 'visible'; fireEvent(document, new Event('visibilitychange')); await advance(1)
  expect(reportWebActivity).toHaveBeenCalledTimes(1)
})

test('ending a session aborts its report and ignores the late response', async () => {
  let resolve!: (value: typeof info) => void
  vi.mocked(reportWebActivity).mockImplementation(() => new Promise(done => { resolve = done }))
  const { client, unmount } = setup(); await advance(1)
  const signal = vi.mocked(reportWebActivity).mock.calls[0]![0]!
  unmount()
  expect(signal.aborted).toBe(true)
  await act(async () => { resolve(info) })
  expect(client.getQueryData(['session'])).toBeUndefined()
})
