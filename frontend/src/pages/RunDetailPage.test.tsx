import { screen, waitFor, act } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import runningMock from '@mocks/run_running.json'
import failedMock from '@mocks/run_failed.json'
import eventsMock from '@mocks/events_running.json'
import type { RunEvent, RunView } from '@/api/types'
import { installFakeEventSource, mockApi, renderWithProviders } from '@/test/utils'
import { RunDetailPage } from './RunDetailPage'

const running = runningMock.data as unknown as RunView
const failed = failedMock.data as unknown as RunView
const events = eventsMock.data as unknown as RunEvent[]

function routes(run: RunView, extra: Record<string, () => unknown> = {}) {
  return {
    [`GET /api/v1/runs/${run.run_id}`]: () => run,
    [`GET /api/v1/runs/${run.run_id}/logs`]: () => ({ lines: [], next_offset: 0, eof: true, size: 0 }),
    [`GET /api/v1/runs/${run.run_id}/artifacts`]: () => ({ run_id: run.run_id, artifacts: [], complete: false }),
    ...extra,
  }
}

test('running run: live stream updates progress; refresh never submits anything', async () => {
  const ES = installFakeEventSource()
  const { calls } = mockApi(routes(running))
  renderWithProviders(<RunDetailPage />, { route: `/runs/${running.run_id}`, path: '/runs/:runId' })
  await screen.findByRole('heading', { level: 1 })
  await waitFor(() => expect(ES.instances.length).toBe(1))
  const source = ES.instances[0]!
  expect(source.url).toContain(`/runs/${running.run_id}/events?after=0`)
  act(() => {
    source.open()
    for (const e of events) source.emit(e.type, e)
  })
  await screen.findByText('live')
  expect(screen.getByText(/Epoch \d+ of \d+/)).toBeInTheDocument()
  expect(calls.every((c) => c.method === 'GET')).toBe(true)
})

test('lost stream shows the disconnected state with a refresh action', async () => {
  const ES = installFakeEventSource()
  mockApi(routes(running))
  renderWithProviders(<RunDetailPage />, { route: `/runs/${running.run_id}`, path: '/runs/:runId' })
  await waitFor(() => expect(ES.instances.length).toBe(1))
  act(() => ES.instances[0]!.fail())
  const banner = await screen.findByText('Live updates disconnected')
  expect(banner).toBeInTheDocument()
  expect(screen.getByRole('button', { name: 'Refresh now' })).toBeInTheDocument()
})

test('failed run shows stage, hint and the retry action; cancel is not offered', async () => {
  installFakeEventSource()
  mockApi(routes(failed))
  renderWithProviders(<RunDetailPage />, { route: `/runs/${failed.run_id}`, path: '/runs/:runId' })
  await screen.findByTestId('run-error')
  expect(screen.getByText(new RegExp(`Failed during ${failed.error!.stage}`))).toBeInTheDocument()
  if (failed.error!.hint) expect(screen.getByText(failed.error!.hint)).toBeInTheDocument()
  expect(screen.getByRole('button', { name: 'Retry as new run' })).toBeInTheDocument()
  expect(screen.queryByRole('button', { name: 'Cancel run' })).not.toBeInTheDocument()
})

test('cancel asks for confirmation and then posts once', async () => {
  installFakeEventSource()
  const { calls } = mockApi(routes(running, { [`POST /api/v1/runs/${running.run_id}/cancel`]: () => ({ ...running, status: 'cancel_requested' }) }))
  renderWithProviders(<RunDetailPage />, { route: `/runs/${running.run_id}`, path: '/runs/:runId' })
  await userEvent.click(await screen.findByRole('button', { name: 'Cancel run' }))
  expect(screen.getByText('Cancel this run? Artifacts produced so far are kept.')).toBeInTheDocument()
  await userEvent.click(screen.getByRole('button', { name: 'Cancel run' }))
  await screen.findAllByText('Stopping…')
  expect(calls.filter((c) => c.method === 'POST').length).toBe(1)
})
