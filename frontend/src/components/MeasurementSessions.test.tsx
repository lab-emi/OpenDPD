import { fireEvent, screen, waitFor, within } from '@testing-library/react'
import measuredMock from '@mocks/result_dpd_measured_mock.json'
import type { EvaluationResult, MeasurementSessionSpec } from '@/api/types'
import { mockApi, renderWithProviders } from '@/test/utils'
import { MeasurementSessions } from './MeasurementSessions'

test('session creation requires an explicitly entered acquisition ID and preserves mock evidence', async () => {
  const result = measuredMock.data as unknown as EvaluationResult
  let request: MeasurementSessionSpec | undefined
  mockApi({
    'GET /api/v1/measurement-sessions': () => [],
    'GET /api/v1/runs': () => [],
    'GET /api/v1/results/run-meas-0001': () => result,
    'POST /api/v1/measurement-sessions': (_url, init) => {
      request = JSON.parse(String(init.body)) as MeasurementSessionSpec
      return { session_id: 'ms-example', spec: request, captures: [], repeats: [], result_hashes: {}, power_matching: 'unverified' }
    },
  })
  renderWithProviders(<MeasurementSessions result={result} />)
  fireEvent.click(screen.getByRole('button', { name: 'Create measurement session' }))
  const dialog = screen.getByRole('dialog')
  const create = within(dialog).getByRole('button', { name: 'Create measurement session' })
  fireEvent.change(within(dialog).getByLabelText(/Session title/), { target: { value: 'Repeat capture review' } })
  expect(create).toBeDisabled()
  expect(within(dialog).getByLabelText(/Acquisition ID/)).toHaveValue('')
  fireEvent.change(within(dialog).getByLabelText(/Acquisition ID/), { target: { value: 'bench-acquisition-001' } })
  fireEvent.change(within(dialog).getByLabelText('Power tolerance (dB)'), { target: { value: '0.2' } })
  fireEvent.click(create)
  await waitFor(() => expect(request).toBeDefined())
  expect(request?.source).toBe('mock')
  expect(request?.captures[0]?.acquisition_id).toBe('bench-acquisition-001')
  expect(request?.captures[0]?.run_id).toBe(result.run_id)
  expect(request?.captures[0]?.acquired_at).toBe(result.measurement?.conditions.measured_at)
  expect(request?.power_tolerance_db).toBe(.2)
})
