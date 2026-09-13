import { screen, waitFor } from '@testing-library/react'
import { mockApi, renderWithProviders } from '@/test/utils'
import { TestingSampleSummary } from './TestingSampleSummary'

test('test counts follow dataset and preprocessing version, without retaining another selection', async () => {
  mockApi({
    'GET /api/v1/datasets/a/sample-counts': url => ({ dataset_id: 'a', version: url.searchParams.get('version'), sample_rate_hz: 80e6, counts: { test: url.searchParams.get('version') === 'cropped-v1' ? 7900 : 10000 } }),
    'GET /api/v1/datasets/b/sample-counts': () => ({ status: 409, body: { error: { code: 'missing', message: 'missing version' } } }),
  })
  const view = renderWithProviders(<TestingSampleSummary datasetId="a" version="raw-v1" />)
  await screen.findByText('10,000')
  view.rerender(<TestingSampleSummary datasetId="a" version="cropped-v1" />)
  await screen.findByText('7,900')
  expect(screen.getByTestId('testing-samples')).toHaveTextContent('cropped-v1')
  expect(screen.queryByText('10,000')).not.toBeInTheDocument()
  view.rerender(<TestingSampleSummary datasetId="b" version="cropped-v1" />)
  await waitFor(() => expect(screen.getByRole('alert')).toHaveTextContent('Could not read'))
  expect(screen.queryByText('7,900')).not.toBeInTheDocument()
})
