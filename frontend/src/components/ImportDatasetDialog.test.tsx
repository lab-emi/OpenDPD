import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import datasetMock from '@mocks/dataset_builtin.json'
import { mockApi, renderWithProviders } from '@/test/utils'
import { ImportDatasetDialog } from './ImportDatasetDialog'

const source = {
  kind: 'csv_import',
  path: 'capture.csv',
  columns: ['tx_i', 'tx_q', 'rx_q', 'rx_i'],
  n_rows: 20000,
  arrays: {},
  legacy_files: [],
  preview: [{ tx_i: 0.1, tx_q: -0.2, rx_q: 0.3, rx_i: 0.4 }],
  problems: [] as string[],
  suggested_mapping: { I_in: 'tx_i', Q_in: 'tx_q', I_out: 'rx_i', Q_out: 'rx_q' },
}

function routes(info = source) {
  return mockApi({
    'GET /api/v1/datasets/import-roots': () => [{ root_id: 'imports', path: '/ws/imports', exists: true }],
    'GET /api/v1/datasets/import-roots/imports/files': () => [{ path: 'capture.csv', kind: 'file', size_bytes: 1234, suffix: '.csv' }],
    'POST /api/v1/datasets/inspect': () => info,
    'POST /api/v1/datasets/import': () => ({ status: 201, body: { ...datasetMock.data, dataset_id: 'capture' } }),
  })
}

test('browse → inspect → correct a column → import sends only (root, path), mapping and metadata', async () => {
  const onImported = vi.fn()
  const { calls } = routes()
  renderWithProviders(<ImportDatasetDialog onClose={() => {}} onImported={onImported} />)
  await userEvent.click(await screen.findByText('capture.csv'))
  const mapping = await screen.findByRole('region', { name: 'Column mapping' })
  expect(within(mapping).getByLabelText('I_out')).toHaveTextContent('rx_i')
  await userEvent.click(within(mapping).getByLabelText('Q_out'))
  await userEvent.click(await screen.findByRole('option', { name: 'rx_i' }))
  expect(screen.getByLabelText('Dataset id')).toHaveValue('capture')
  await userEvent.type(screen.getByLabelText('Sample rate (Hz)'), '800e6')
  await userEvent.click(screen.getByRole('button', { name: 'Import' }))
  await waitFor(() => expect(onImported).toHaveBeenCalledWith('capture'))
  const body = calls.find((c) => c.path === '/api/v1/datasets/import')?.body as Record<string, unknown>
  expect(body).toMatchObject({
    source: { root_id: 'imports', path: 'capture.csv' },
    dataset_id: 'capture',
    mapping: { I_in: 'tx_i', Q_in: 'tx_q', I_out: 'rx_i', Q_out: 'rx_i' },
    signal: { sample_rate_hz: 800e6, amplitude_units: 'unknown' },
    guard_samples: 256,
  })
})

test('problems found by the server block the import and are shown verbatim', async () => {
  routes({ ...source, problems: ['row 12: expected 4 numeric fields, got 3'] })
  renderWithProviders(<ImportDatasetDialog onClose={() => {}} onImported={() => {}} />)
  await userEvent.click(await screen.findByText('capture.csv'))
  await screen.findByText('row 12: expected 4 numeric fields, got 3')
  expect(screen.getByRole('button', { name: 'Import' })).toBeDisabled()
})
