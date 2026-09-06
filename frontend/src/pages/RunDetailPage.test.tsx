import { screen, waitFor, act, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import runningMock from '@mocks/run_running.json'
import failedMock from '@mocks/run_failed.json'
import eventsMock from '@mocks/events_running.json'
import lineageMock from '@mocks/run_lineage_dpd.json'
import historyMock from '@mocks/history_points_mock.json'
import { vi } from 'vitest'
import type { RunEvent, RunView } from '@/api/types'
import { installFakeEventSource, mockApi, renderWithProviders } from '@/test/utils'
import { RunDetailPage } from './RunDetailPage'

vi.mock('plotly.js-basic-dist-min', () => ({ default: { react: vi.fn(() => Promise.resolve()), purge: vi.fn() } }))

const running = runningMock.data as unknown as RunView
const failed = failedMock.data as unknown as RunView
const events = eventsMock.data as unknown as RunEvent[]

function routes(run: RunView, extra: Record<string, () => unknown> = {}) {
  return {
    [`GET /api/v1/runs/${run.run_id}`]: () => run,
    [`GET /api/v1/runs/${run.run_id}/logs`]: () => ({ lines: [], next_offset: 0, eof: true, size: 0 }),
    [`GET /api/v1/runs/${run.run_id}/artifacts`]: () => ({ run_id: run.run_id, artifacts: [], complete: false }),
    [`GET /api/v1/runs/${run.run_id}/lineage`]: () => ({ run_id: run.run_id, parents: [], children: [] }),
    [`GET /api/v1/runs/${run.run_id}/history`]: () => ({ status: 404, body: { error: { code: 'history_not_available', message: 'no history', details: [], hint: null } } }),
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

test('a succeeded DPD run shows its lineage and can be applied through another surrogate', async () => {
  installFakeEventSource()
  const dpd: RunView = { ...running, run_id: 'run-dpd-0001', task: 'train_dpd', model_key: 'gru', status: 'succeeded', result_id: 'res-dpd-0001', finished_at: '2026-09-06T08:10:00Z' }
  const pa = (id: string): RunView => ({ ...running, run_id: id, name: `PA ${id}`, task: 'train_pa', status: 'succeeded', result_id: `res-${id}`, finished_at: '2026-09-06T08:00:00Z' })
  const applied: RunView = { ...running, run_id: 'run-apply-0003', task: 'run_dpd', status: 'queued', started_at: null, worker: null, result_id: null }
  const { calls } = mockApi(
    routes(dpd, {
      [`GET /api/v1/runs/${dpd.run_id}/lineage`]: () => lineageMock.data,
      [`GET /api/v1/runs/${dpd.run_id}/history`]: () => historyMock.data,
      'GET /api/v1/runs': () => [dpd, pa('run-pa-0001'), pa('run-pa-0002')],
      'POST /api/v1/runs': () => ({ status: 201, body: applied }),
      [`GET /api/v1/runs/${applied.run_id}`]: () => applied,
      [`GET /api/v1/runs/${applied.run_id}/lineage`]: () => ({ run_id: applied.run_id, parents: [], children: [] }),
    }),
  )
  renderWithProviders(<RunDetailPage />, { route: `/runs/${dpd.run_id}`, path: '/runs/:runId' })
  const lineage = await screen.findByRole('region', { name: 'Lineage' })
  if (typeof dpd.progress_epoch === 'number' && dpd.progress_epoch > 0) expect(screen.getByText(`Epoch ${dpd.progress_epoch} of ${dpd.progress_total_epochs}`)).toBeInTheDocument()
  expect(within(lineage).getByRole('link', { name: 'run-pa-0001' })).toBeInTheDocument()
  expect(within(lineage).getAllByText(/DPD model:/)).toHaveLength(2)
  expect(within(lineage).getByText(new RegExp(`weights ${lineageMock.data.parents[0]!.checkpoint_sha256!.slice(0, 12)}`))).toBeInTheDocument()
  await screen.findByText("Curves come from the run's history log.")
  expect(screen.getByTestId('history-NMSE')).toBeInTheDocument()

  await userEvent.click(screen.getByRole('button', { name: 'Apply DPD to the test split…' }))
  const dialog = await screen.findByRole('dialog', { name: 'Apply this DPD to the test split' })
  expect(within(dialog).getByText(/not a PA output/)).toBeInTheDocument()
  await userEvent.click(within(dialog).getByLabelText('PA surrogate'))
  await userEvent.click(await screen.findByRole('option', { name: /PA run-pa-0002/ }))
  await userEvent.click(within(dialog).getByRole('button', { name: 'Apply' }))
  await waitFor(() => expect(calls.some((c) => c.method === 'POST')).toBe(true))
  const body = calls.find((c) => c.method === 'POST')!.body as { config: Record<string, unknown> }
  expect(body.config).toMatchObject({ task: 'run_dpd', dpd_reference: { run_id: 'run-dpd-0001' }, pa_reference: { run_id: 'run-pa-0002' }, dataset: { id: dpd.dataset_id } })
  await screen.findByText('run-apply-0003')
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  expect(screen.getByText(/applied a trained DPD to the test split/)).toBeInTheDocument()
})
