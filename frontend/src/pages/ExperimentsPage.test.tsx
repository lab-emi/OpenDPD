import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import runMock from '@mocks/run_running.json'
import { mockApi, renderWithProviders } from '@/test/utils'
import { ExperimentsPage } from './ExperimentsPage'

const report = {
  package_version: 1,
  kind: 'share',
  run_id: 'run-imported-0001',
  imported_runs: ['run-pa-0001', 'run-imported-0001'],
  dataset_status: 'missing',
  dataset_id: 'capture',
  missing: ['dataset capture (raw sha256 abcdef…)'],
  evaluate_command: 'opendpd evaluate run-imported-0001 --workspace <workspace>',
  note: 'The run was imported without its data; results are the stored ones until the dataset is registered.',
}

test('lists runs with 1-based epoch progress', async () => {
  mockApi({ 'GET /api/v1/runs': () => [{ ...runMock.data, progress_epoch: 2, progress_total_epochs: 3 }] })
  renderWithProviders(<ExperimentsPage />, { route: '/experiments', path: '/experiments' })
  const table = await screen.findByRole('table', { name: 'Experiments' })
  expect(within(table).getByText('Epoch 2 of 3')).toBeInTheDocument()
})

test('importing a package uploads it as multipart, shows the report and links to the imported run', async () => {
  const { calls, fetchMock } = mockApi({
    'GET /api/v1/runs': () => [],
    'POST /api/v1/imports': () => ({ status: 201, body: report }),
  })
  renderWithProviders(<ExperimentsPage />, { route: '/experiments', path: '/experiments' })
  await screen.findByText('No experiments yet.')
  const input = screen.getByTestId('import-package')
  await userEvent.upload(input, new File(['zip'], 'run-pa-0001-share.zip', { type: 'application/zip' }))
  const banner = await screen.findByTestId('import-report')
  expect(banner).toHaveTextContent('Imported run-pa-0001, run-imported-0001 into this workspace. Dataset capture: missing.')
  expect(within(banner).getByText('dataset capture (raw sha256 abcdef…)')).toBeInTheDocument()
  expect(within(banner).getByRole('link', { name: 'Open the imported run' })).toHaveAttribute('href', '/runs/run-imported-0001')
  expect(calls.some((c) => c.method === 'POST' && c.path === '/api/v1/imports')).toBe(true)
  const init = fetchMock.mock.calls.find((c) => String(c[0]).endsWith('/api/v1/imports'))?.[1] as RequestInit
  expect(init.body).toBeInstanceOf(FormData)
  expect((init.body as FormData).get('file')).toBeInstanceOf(File)
  await waitFor(() => expect(calls.filter((c) => c.method === 'GET' && c.path === '/api/v1/runs').length).toBeGreaterThan(1))
})

test('a refused package shows the server reason', async () => {
  mockApi({
    'GET /api/v1/runs': () => [],
    'POST /api/v1/imports': () => ({ status: 422, body: { error: { code: 'hash_mismatch', message: 'save/model.pt does not match its manifest hash', details: [], hint: 'Export the run again.' } } }),
  })
  renderWithProviders(<ExperimentsPage />, { route: '/experiments', path: '/experiments' })
  await screen.findByText('No experiments yet.')
  await userEvent.upload(screen.getByTestId('import-package'), new File(['zip'], 'bad.zip', { type: 'application/zip' }))
  await screen.findByText(/does not match its manifest hash/)
  expect(screen.queryByTestId('import-report')).not.toBeInTheDocument()
})
