import { screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import reportMock from '@mocks/adaptation_report_mock.json'
import { mockApi, renderWithProviders } from '@/test/utils'
import { RobustnessPage } from './RobustnessPage'

const report = reportMock.data
const summary = {
  plan_sha256: report.plan_sha256,
  set_id: report.condition_set.set_id,
  device: report.condition_set.device,
  dimension: report.condition_set.dimension,
  n_conditions: report.condition_set.conditions.length,
  n_cells: report.cells.length,
  n_without_number: report.cells.filter((c) => c.status !== 'ok').length,
  evidence_met: report.evidence_bar.met,
  generated_at: report.generated_at,
  report_sha256: report.report_sha256,
}

test('shows the condition matrix with every failure, the cost of adaptation and the evidence bar verdict', async () => {
  mockApi({
    'GET /api/v1/adaptation/reports': () => [summary],
    [`GET /api/v1/adaptation/reports/${report.plan_sha256.slice(0, 12)}`]: () => report,
  })
  renderWithProviders(<RobustnessPage />, { route: `/robustness/${report.plan_sha256.slice(0, 12)}`, path: '/robustness/:planSha' })
  await screen.findByTestId('adaptation-report')
  const list = screen.getByRole('table', { name: 'Adaptation reports' })
  expect(within(list).getByRole('link', { name: 'synthetic-drive-v0' })).toBeInTheDocument()
  expect(within(list).getByText('6 of 7 with a number')).toBeInTheDocument()
  // below the bar: said in the list, on the report and in the limitations
  expect(screen.getAllByTestId('evidence-bar').map((c) => c.textContent)).toEqual(['not met', 'not met'])
  expect(screen.getByRole('alert')).toHaveTextContent('rehearsal of the protocol, not evidence')
  expect(screen.getByTestId('limitations')).toHaveTextContent('nothing here generalises to other devices')

  const matrix = screen.getByRole('table', { name: 'Condition matrix' })
  expect(within(matrix).getByRole('columnheader', { name: 'few-shot · 2000 samples' })).toBeInTheDocument()
  const zero = within(matrix).getByRole('row', { name: /drive-2/ })
  expect(zero).toHaveTextContent('-19.30 (n=1)')
  expect(zero).toHaveTextContent('target reached 0%')
  const failure = within(matrix).getByTestId('cell-failure')
  expect(failure).toHaveTextContent('FAILED 1/1')
  expect(failure).toHaveTextContent('dataset_too_short [train]')
  // the source row never has a zero-update or few-shot cell
  expect(within(matrix).getByRole('row', { name: /drive-0/ })).toHaveTextContent('—')

  // cost: a zero-update cell used no new samples, a few-shot cell exactly the budget
  const cost = screen.getByRole('table', { name: 'Cost of adaptation' })
  const rows = within(cost).getAllByRole('row').map((r) => r.textContent)
  expect(rows.some((r) => r?.startsWith('zero update') && r.includes('drive-1') && r.includes('0') && r.includes('0.0'))).toBe(true)
  expect(rows.some((r) => r?.startsWith('few-shot · 2000 samples') && r.includes('2000'))).toBe(true)

  // switching the metric changes the numbers, nothing is recomputed on the client
  await userEvent.click(screen.getByRole('combobox', { name: 'Metric' }))
  await userEvent.click(await screen.findByRole('option', { name: 'ACLR_AVG' }))
  expect(within(matrix).getByRole('row', { name: /drive-2/ })).toHaveTextContent('-25.30 (n=1)')

  // failures only keeps the rows with a failed cell
  await userEvent.click(screen.getByRole('switch', { name: 'Failures only' }))
  expect(within(matrix).getAllByRole('row')).toHaveLength(2)
  expect(within(matrix).getByRole('row', { name: /drive-2/ })).toBeInTheDocument()
  expect(screen.getByRole('link', { name: 'Download Markdown report' })).toHaveAttribute('href', `/api/v1/adaptation/reports/${report.plan_sha256.slice(0, 12)}?format=md`)
})

test('an empty workspace explains how a report is produced', async () => {
  mockApi({ 'GET /api/v1/adaptation/reports': () => [] })
  renderWithProviders(<RobustnessPage />, { route: '/robustness', path: '/robustness' })
  expect(await screen.findByText(/No adaptation report in this workspace yet/)).toBeInTheDocument()
  expect(screen.queryByTestId('adaptation-report')).not.toBeInTheDocument()
})
