import { screen, waitFor, act, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import runningMock from '@mocks/run_running.json'
import failedMock from '@mocks/run_failed.json'
import eventsMock from '@mocks/events_running.json'
import lineageMock from '@mocks/run_lineage_dpd.json'
import historyMock from '@mocks/history_points_mock.json'
import datasetMock from '@mocks/dataset_builtin.json'
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
  expect(screen.getByText(/there is no training history/)).toBeInTheDocument()
})

test('a succeeded run_dpd run offers the measured-capture import: files are uploaded, conditions declared, one evaluate_measured run submitted', async () => {
  installFakeEventSource()
  const applied: RunView = { ...running, run_id: 'run-apply-0001', task: 'run_dpd', status: 'succeeded', finished_at: '2026-09-06T08:10:00Z', result_id: 'res-apply-0001', dataset_id: 'dpa-200mhz', model_key: 'gru', progress_epoch: null, progress_total_epochs: null }
  const created: RunView = { ...running, run_id: 'run-meas-0002', task: 'evaluate_measured', status: 'queued', started_at: null, worker: null, result_id: null, name: 'measured run-apply-0001' }
  const { calls } = mockApi({
    ...routes(applied),
    ...routes(created),
    'GET /api/v1/datasets/dpa-200mhz': () => datasetMock.data,
    'POST /api/v1/datasets/upload': () => ({ status: 201, body: { root_id: 'imports', path: 'uploads/20260906-with.npy', size_bytes: 8 } }),
    'POST /api/v1/runs': () => ({ status: 201, body: created }),
  })
  renderWithProviders(<RunDetailPage />, { route: `/runs/${applied.run_id}`, path: '/runs/:runId' })
  await screen.findByRole('heading', { level: 1 })
  await userEvent.click(screen.getByRole('button', { name: 'Import measured captures…' }))
  const dialog = await screen.findByRole('dialog', { name: 'Score captures of a physical PA driven by this export' })
  expect(within(dialog).getByText(/not independently verified/)).toBeInTheDocument()
  const submitButton = within(dialog).getByRole('button', { name: 'Align and score' })
  expect(submitButton).toBeDisabled()
  await userEvent.upload(within(dialog).getByLabelText('Capture with DPD'), new File(['iq-bytes'], 'with.npy'))
  await within(dialog).findByText(/with\.npy/)
  await userEvent.type(within(dialog).getByLabelText(/Device under test/), 'GaN Doherty unit 2')
  await userEvent.type(within(dialog).getByLabelText(/Capture chain/), 'SMW200A -> PA -> 30 dB pad -> FSW')
  await userEvent.type(within(dialog).getByLabelText(/^Drive/), 'generator -12 dBm')
  await userEvent.type(within(dialog).getByLabelText(/Output power with DPD/), '30')
  // the dataset rate is prefilled; the operator may overwrite it
  expect(within(dialog).getByLabelText(/Capture sample rate/)).toHaveValue(datasetMock.data.signal.sample_rate_hz)
  await waitFor(() => expect(submitButton).toBeEnabled())
  await userEvent.click(submitButton)
  await screen.findByText('run-meas-0002')
  const upload = calls.filter((c) => c.path === '/api/v1/datasets/upload')
  expect(upload).toHaveLength(1)
  const submitted = calls.find((c) => c.path === '/api/v1/runs' && c.method === 'POST')!.body as { config: Record<string, unknown> }
  const config = submitted.config
  expect(config.task).toBe('evaluate_measured')
  expect(config.evaluation).toEqual({ evidence_type: 'dpd_measured' })
  const measurement = config.measurement as { apply_run_id: string; with_dpd: { path: string; declared_output_power_dbm: number | null }; without_dpd: unknown; conditions: Record<string, unknown>; source: string; playback: string }
  expect(measurement.apply_run_id).toBe('run-apply-0001')
  expect(measurement.with_dpd).toEqual({ path: 'uploads/20260906-with.npy', declared_output_power_dbm: 30 })
  expect(measurement.without_dpd).toBeNull()
  expect(measurement.source).toBe('manual')
  expect(measurement.playback).toBe('loop')
  expect(measurement.conditions).toMatchObject({ pa: 'GaN Doherty unit 2', capture_chain: 'SMW200A -> PA -> 30 dB pad -> FSW', drive: 'generator -12 dBm', sample_rate_hz: datasetMock.data.signal.sample_rate_hz, calibration: 'none' })
  expect(typeof measurement.conditions.measured_at).toBe('string')
})

test('a succeeded PA run whose model has a streaming variant can be scored under streaming semantics', async () => {
  installFakeEventSource()
  const pa: RunView = { ...running, run_id: 'run-pa-0009', task: 'train_pa', model_key: 'gru', status: 'succeeded', result_id: 'res-pa-0009', finished_at: '2026-09-06T08:10:00Z' }
  const streamed: RunView = { ...running, run_id: 'run-stream-0001', task: 'evaluate_pa', model_key: 'gru_stream', status: 'queued', started_at: null, worker: null, result_id: null }
  const variant = { key: 'gru_stream', display_name: 'GRU (streaming, stateful)', family: 'recurrent', legacy_backbone: 'gru', training_method: 'gradient', roles: ['pa', 'dpd'], params: [], status: 'experimental', devices_tested: ['cpu'], lookahead_samples: 0, lookahead_note: 'causal', execution_semantics: 'streaming_stateful', weights_from: 'gru', export_formats: [], constraints: null, reference: null, evidence: null }
  const { calls } = mockApi(
    routes(pa, {
      'GET /api/v1/models': () => [{ ...variant, key: 'gru', display_name: 'GRU', execution_semantics: 'offline_segmented', weights_from: null }, variant],
      'GET /api/v1/runs': () => [pa],
      'POST /api/v1/runs': () => ({ status: 201, body: streamed }),
      [`GET /api/v1/runs/${streamed.run_id}`]: () => streamed,
      [`GET /api/v1/runs/${streamed.run_id}/lineage`]: () => ({ run_id: streamed.run_id, parents: [], children: [] }),
    }),
  )
  renderWithProviders(<RunDetailPage />, { route: `/runs/${pa.run_id}`, path: '/runs/:runId' })
  await userEvent.click(await screen.findByRole('button', { name: 'Score under streaming semantics' }))
  await waitFor(() => expect(calls.some((c) => c.method === 'POST')).toBe(true))
  const body = calls.find((c) => c.method === 'POST')!.body as { config: Record<string, unknown> }
  expect(body.config).toMatchObject({ task: 'evaluate_pa', model: { key: 'gru_stream' }, pa_reference: { run_id: 'run-pa-0009' }, dataset: { id: pa.dataset_id } })
  await screen.findByText('run-stream-0001')
})

test('a run whose model has no streaming variant offers no streaming action', async () => {
  installFakeEventSource()
  const lstm: RunView = { ...running, run_id: 'run-pa-0010', task: 'train_pa', model_key: 'lstm', status: 'succeeded', result_id: 'res-pa-0010', finished_at: '2026-09-06T08:10:00Z' }
  mockApi(routes(lstm, { 'GET /api/v1/models': () => [], 'GET /api/v1/runs': () => [lstm] }))
  renderWithProviders(<RunDetailPage />, { route: `/runs/${lstm.run_id}`, path: '/runs/:runId' })
  await screen.findByRole('link', { name: 'Open result' })
  expect(screen.queryByRole('button', { name: 'Score under streaming semantics' })).not.toBeInTheDocument()
})
