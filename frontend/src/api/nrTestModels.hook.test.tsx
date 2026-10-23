import { focusManager, QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { act, renderHook, waitFor } from '@testing-library/react'
import type { ReactNode } from 'react'
import { fixtureCatalog } from '@/test/nrTestModelFixtures'
import { mockApi } from '@/test/utils'
import { useNrTestModelCatalog } from './nrTestModels'

const PATH = '/api/v1/signal-generator/nr-test-models'
const refusal = (status: number, code: string) => ({ status, body: { error: { code, message: 'This endpoint is not available', details: [], hint: null } } })
const wait = (ms: number) => new Promise(resolve => setTimeout(resolve, ms))

function mount() {
  // The default QueryClient of the app refetches on a window refocus and keeps unused queries for five minutes; the hook has to hold up under both.
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, staleTime: 10_000, refetchOnWindowFocus: true } } })
  const wrapper = ({ children }: { children: ReactNode }) => <QueryClientProvider client={client}>{children}</QueryClientProvider>
  return { client, wrapper }
}
const asked = (calls: Array<{ path: string }>) => calls.filter(c => c.path === PATH).length

test.each([[403, 'feature_unavailable'], [404, 'not_found']])('a %i is an answer ("not offered"), not an error: the data is null, nothing is retried and a refocus asks nothing', async (status, code) => {
  const { calls } = mockApi({ ['GET ' + PATH]: () => refusal(status, code) })
  const { client, wrapper } = mount()
  const { result, unmount } = renderHook(() => useNrTestModelCatalog(), { wrapper })
  await waitFor(() => expect(result.current.isSuccess).toBe(true))
  expect(result.current.data).toBeNull()
  expect(result.current.isError).toBe(false)
  expect(result.current.errorUpdateCount).toBe(0)
  act(() => { focusManager.setFocused(false); focusManager.setFocused(true) })
  await wait(60)
  expect(asked(calls)).toBe(1)
  unmount(); client.clear()
})

test('visiting the page again asks a server that did not offer the catalogue once more, without passing through a pending or error state', async () => {
  let offered = false
  const { calls } = mockApi({ ['GET ' + PATH]: () => offered ? fixtureCatalog() : refusal(403, 'feature_unavailable') })
  const { client, wrapper } = mount()
  const first = renderHook(() => useNrTestModelCatalog(), { wrapper })
  await waitFor(() => expect(first.result.current.isSuccess).toBe(true))
  first.unmount()
  const seen: string[] = []
  // (The hook is read in render like the page does; react-query only re-renders for the properties that were read.)
  const second = renderHook(() => { const query = useNrTestModelCatalog(); seen.push(`${query.status}/${query.errorUpdateCount}/${query.data === null ? 'null' : query.data ? 'catalogue' : 'undefined'}/${query.isFetching ? 'fetching' : 'idle'}`); return query }, { wrapper })
  await waitFor(() => expect(asked(calls)).toBe(2))
  await waitFor(() => expect(second.result.current.isFetching).toBe(false))
  // The earlier "not offered" stays on screen while it is asked again: no "pending", no error, no undefined data in between.
  expect(new Set(seen)).toEqual(new Set(['success/0/null/fetching', 'success/0/null/idle']))
  // A server that has been upgraded in the meantime is picked up by the next visit.
  second.unmount()
  offered = true
  const third = renderHook(() => useNrTestModelCatalog(), { wrapper })
  await waitFor(() => expect(third.result.current.data?.models.length).toBe(11))
  expect(asked(calls)).toBe(3)
  third.unmount(); client.clear()
})

test('a catalogue that was loaded is kept for the whole session: neither a remount nor a refocus asks again', async () => {
  const { calls } = mockApi({ ['GET ' + PATH]: () => fixtureCatalog() })
  const { client, wrapper } = mount()
  const first = renderHook(() => useNrTestModelCatalog(), { wrapper })
  await waitFor(() => expect(first.result.current.data?.models.length).toBe(11))
  act(() => { focusManager.setFocused(false); focusManager.setFocused(true) })
  first.unmount()
  const second = renderHook(() => useNrTestModelCatalog(), { wrapper })
  await wait(60)
  expect(second.result.current.data?.models.length).toBe(11)
  expect(asked(calls)).toBe(1)
  second.unmount(); client.clear()
})

test('any other failure stays an error with its count, is not retried by itself and is not asked again on a refocus', async () => {
  const { calls } = mockApi({ ['GET ' + PATH]: () => refusal(500, 'internal_error') })
  const { client, wrapper } = mount()
  const { result, unmount } = renderHook(() => useNrTestModelCatalog(), { wrapper })
  await waitFor(() => expect(result.current.isError).toBe(true))
  expect(result.current.data).toBeUndefined()
  expect(result.current.errorUpdateCount).toBe(1)
  act(() => { focusManager.setFocused(false); focusManager.setFocused(true) })
  await wait(60)
  expect(asked(calls)).toBe(1)
  expect(result.current.isError).toBe(true)
  unmount(); client.clear()
})
