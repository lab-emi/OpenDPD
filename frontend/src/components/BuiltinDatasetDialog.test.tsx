import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import datasetMock from '@mocks/dataset_builtin.json'
import { mockApi, renderWithProviders } from '@/test/utils'
import { BuiltinDatasetDialog } from './BuiltinDatasetDialog'

test('every server catalog entry is selectable, including the synthetic tutorial', async () => {
  const names = ['APA_200MHz', 'APA_200MHz_b', 'DPA_160MHz', 'DPA_200MHz', 'MyCustomPA']
  const { calls } = mockApi({
    'GET /api/v1/datasets': () => [],
    'GET /api/v1/datasets/builtin': () => names.map((name) => ({ name, description: name === 'MyCustomPA' ? 'dummy dataset for tutorial purpose' : '', dataset_format: 'split_csv', n_samples: 102400, signal: datasetMock.data.signal, has_demodulator: true, origin: name === 'MyCustomPA' ? 'synthetic' : 'measured', raw_sha256: 'a'.repeat(64) })),
    'POST /api/v1/datasets/import-builtin': () => ({ ...datasetMock.data, dataset_id: 'mycustompa' }),
  })
  const selected = vi.fn()
  renderWithProviders(<BuiltinDatasetDialog onClose={() => {}} onSelected={selected} />)
  for (const name of names) expect(await screen.findByRole('button', { name: `Add & inspect ${name}` })).toBeEnabled()
  expect(screen.getByText('dummy dataset for tutorial purpose')).toBeInTheDocument()
  await userEvent.click(screen.getByRole('button', { name: 'Add & inspect MyCustomPA' }))
  await waitFor(() => expect(selected).toHaveBeenCalledWith('mycustompa'))
  expect(calls.find((call) => call.path === '/api/v1/datasets/import-builtin')?.body).toEqual({ name: 'MyCustomPA' })
})
