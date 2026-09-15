import { act, render, screen } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { afterEach, beforeEach, expect, test, vi } from 'vitest'
import { clearWebSession, loadSession } from '@/api/client'
import { SessionGate } from './SessionGate'

vi.mock('@/components/useWebActivity', () => ({ useWebActivity: () => undefined }))
vi.mock('@/api/client', async original => ({ ...await original<typeof import('@/api/client')>(), WEB_MODE: true, loadSession: vi.fn(), clearWebSession: vi.fn(), hasQueuedWebSession: () => false }))
const live = { authenticated: true, mode: 'web' as const, version: '2.2.8', server_time: '2026-09-15T10:00:00Z', idle_expires_at: '2026-09-15T12:00:00Z', expires_at: '2026-09-15T23:55:00Z' }
const renewed = { ...live, server_time: '2026-09-15T12:00:00Z', idle_expires_at: '2026-09-15T13:00:00Z' }
const advance = async (ms: number) => { await act(async () => { await vi.advanceTimersByTimeAsync(ms) }) }
beforeEach(() => {
  // The local clock is deliberately wrong; the lease uses server elapsed time.
  vi.useFakeTimers({ now: new Date('2099-01-01T00:00:00Z') })
  vi.mocked(loadSession).mockReset().mockResolvedValueOnce(live)
  vi.mocked(clearWebSession).mockClear()
})
afterEach(() => vi.useRealTimers())
function setup() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(<QueryClientProvider client={client}><SessionGate><p>Workspace ready</p></SessionGate></QueryClientProvider>)
  return client
}

test('another workspace on the same IP can renew the deadline without discarding this session', async () => {
  const client = setup(); await advance(1)
  expect(screen.getByText('Workspace ready')).toBeInTheDocument()
  expect(loadSession).toHaveBeenCalledTimes(1)
  vi.mocked(loadSession).mockResolvedValueOnce(renewed)
  await advance(2 * 3600_000)
  expect(loadSession).toHaveBeenCalledTimes(2)
  expect(client.getQueryData(['session'])).toEqual(renewed)
  expect(clearWebSession).not.toHaveBeenCalled()
  vi.mocked(loadSession).mockResolvedValueOnce({ authenticated: false, version: '2.2.8', mode: 'web' })
  await advance(3600_000)
  expect(clearWebSession).toHaveBeenCalledTimes(1)
  await advance(1)
  expect(screen.queryByText('Workspace ready')).not.toBeInTheDocument()
  expect(screen.getByText(/2 hours without user activity/)).toBeVisible()
})

test('a failed deadline check preserves credentials and rechecks without sending activity', async () => {
  const client = setup(); await advance(1)
  vi.mocked(loadSession).mockRejectedValueOnce(new Error('offline')).mockResolvedValueOnce(renewed)
  await advance(2 * 3600_000)
  expect(clearWebSession).not.toHaveBeenCalled()
  expect(screen.getByText('Workspace ready')).toBeInTheDocument()
  await advance(30_000)
  expect(client.getQueryData(['session'])).toEqual(renewed)
  expect(clearWebSession).not.toHaveBeenCalled()
})
