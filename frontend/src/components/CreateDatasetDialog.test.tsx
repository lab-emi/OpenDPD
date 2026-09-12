import { within, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import datasetMock from '@mocks/dataset_builtin.json'
import { mockApi, renderWithProviders } from '@/test/utils'
import { CreateDatasetDialog } from './CreateDatasetDialog'

const defaults = { ratios: { train: .6, val: .2, test: .2 }, guard_samples: 256 }
const options = { format: 'complex_pair', header: 'present', mapping: { input: 0, output: 1 } }
const good = { valid: true, data_valid: true, sha256: 'a'.repeat(64), columns: ['input', 'output'], options,
  preview: [['0.1+0.2j', '0.3-0.4i']], n_samples: 4096, issue_count: 0, issues: [], split: defaults,
  boundaries: { train: [0, 2150], val: [2406, 3122], test: [3378, 4096] }, split_counts: { train: 2150, val: 716, test: 718 } }

function setup(inspect: (body: Record<string, unknown>) => unknown = () => good) {
  return mockApi({
    'GET /api/v1/datasets/import-defaults': () => defaults,
    'POST /api/v1/datasets/upload': () => ({ root_id: 'imports', path: 'uploads/paired.csv', size_bytes: 10000 }),
    'POST /api/v1/datasets/csv/preview': (_url, init) => inspect(JSON.parse(init.body as string)),
    'POST /api/v1/datasets/csv': () => ({ status: 201, body: { ...datasetMock.data, dataset_id: 'paired' } }),
  })
}

test('paired CSV → server defaults → custom split review → create; edits invalidate review', async () => {
  const user = userEvent.setup()
  const { calls } = setup((body) => ({ ...good, split: body.split }))
  const imported = vi.fn()
  renderWithProviders(<CreateDatasetDialog onClose={() => {}} onImported={imported} />)
  await user.upload(await screen.findByLabelText('Choose CSV file'), new File(['input,output\n0.1+0.2j,0.3-0.4i'], 'paired.csv', { type: 'text/csv' }))
  await screen.findByText(/All 4,096 sample rows passed/)
  expect(screen.getByLabelText('input')).toHaveTextContent('input')
  await user.click(screen.getByRole('button', { name: 'Continue' }))
  expect(screen.getByLabelText('train (%)')).toHaveValue('60')
  expect(screen.getByLabelText('val (%)')).toHaveValue('20')
  expect(screen.getByLabelText('test (%)')).toHaveValue('20')
  await user.clear(screen.getByLabelText('train (%)')); await user.type(screen.getByLabelText('train (%)'), '70')
  await user.clear(screen.getByLabelText('test (%)')); await user.type(screen.getByLabelText('test (%)'), '10')
  await user.click(screen.getByRole('button', { name: 'Review' }))
  await screen.findByText(/Validated 4,096 paired samples/)
  expect(screen.getByTestId('step-1-complete')).toBeInTheDocument()
  await user.click(screen.getByRole('button', { name: 'Back' }))
  await user.clear(screen.getByLabelText('train (%)')); await user.type(screen.getByLabelText('train (%)'), '60')
  expect(screen.queryByTestId('step-1-complete')).not.toBeInTheDocument()
  expect(screen.getByRole('tab', { name: /Review/ })).toBeDisabled()
  await user.clear(screen.getByLabelText('train (%)')); await user.type(screen.getByLabelText('train (%)'), '70')
  await user.click(screen.getByRole('button', { name: 'Review' }))
  await user.click(await screen.findByRole('button', { name: 'Create dataset' }))
  await waitFor(() => expect(imported).toHaveBeenCalledWith('paired'))
  expect(calls.find((c) => c.path === '/api/v1/datasets/csv')?.body).toMatchObject({
    source: { root_id: 'imports', path: 'uploads/paired.csv' }, options,
    split: { ratios: { train: .7, val: .2, test: .1 }, guard_samples: 256 }, expected_sha256: 'a'.repeat(64),
  })
})

test('a bad row past the preview is explained and blocks progression', async () => {
  setup(() => ({ ...good, valid: false, data_valid: false, issue_count: 1, issues: [{ code: 'numeric', line: 901, column: 'output', message: 'Invalid sample NaN.', fix: 'Replace it with the correct paired finite sample.' }] }))
  renderWithProviders(<CreateDatasetDialog onClose={() => {}} onImported={() => {}} />)
  await userEvent.upload(await screen.findByLabelText('Choose CSV file'), new File(['input,output'], 'invalid.csv', { type: 'text/csv' }))
  await screen.findByText('Line 901 · output: Invalid sample NaN.')
  expect(screen.getByText(/Replace it with the correct paired finite sample/)).toBeInTheDocument()
  expect(screen.getByRole('button', { name: 'Continue' })).toBeDisabled()
})

test('mapping changes immediately clear file confirmation and require revalidation', async () => {
  setup()
  const user = userEvent.setup()
  renderWithProviders(<CreateDatasetDialog onClose={() => {}} onImported={() => {}} />)
  await user.upload(await screen.findByLabelText('Choose CSV file'), new File(['input,output'], 'paired.csv', { type: 'text/csv' }))
  await screen.findByText(/All 4,096 sample rows passed/)
  await user.click(screen.getByLabelText('output'))
  await user.click(screen.getByRole('option', { name: '1. input' }))
  expect(screen.getByRole('button', { name: 'Continue' })).toBeDisabled()
  expect(screen.queryByTestId('step-0-complete')).not.toBeInTheDocument()
})

test('reset discards the CSV draft only after confirmation and never creates a dataset', async () => {
  const { calls } = setup()
  renderWithProviders(<CreateDatasetDialog onClose={() => {}} onImported={() => {}} />)
  await userEvent.upload(await screen.findByLabelText('Choose CSV file'), new File(['input,output'], 'paired.csv', { type: 'text/csv' }))
  await screen.findByText(/All 4,096 sample rows passed/)
  await userEvent.click(screen.getByRole('button', { name: 'Continue' }))
  await userEvent.type(screen.getByLabelText('train (%)'), '5')
  await userEvent.click(screen.getByRole('button', { name: 'Reset page' }))
  expect(screen.getByRole('dialog', { name: 'Reset this page?' })).toHaveTextContent('Saved datasets, results and running jobs are kept.')
  await userEvent.click(within(screen.getByRole('dialog', { name: 'Reset this page?' })).getByRole('button', { name: 'Continue' }))
  expect(await screen.findByRole('tab', { name: /CSV & columns/ })).toHaveAttribute('aria-selected', 'true')
  expect(screen.getByRole('button', { name: 'Continue' })).toBeDisabled()
  expect(screen.queryByText('paired.csv')).not.toBeInTheDocument()
  expect(calls.some((call) => call.path === '/api/v1/datasets/csv')).toBe(false)
})
