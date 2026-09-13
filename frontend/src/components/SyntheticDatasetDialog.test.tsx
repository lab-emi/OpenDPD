import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { mockApi, renderWithProviders } from '@/test/utils'
import { SyntheticDatasetDialog } from './SyntheticDatasetDialog'

test('synthetic generation is labelled and never initiates publication', async () => {
  const { calls } = mockApi({ 'POST /api/v1/datasets/synthetic': () => ({ datasets: [{ dataset_id: 'synthetic-fixture', display_name: 'Synthetic fixture' }], condition_set: {} }) })
  renderWithProviders(<SyntheticDatasetDialog onClose={() => {}} />)
  expect(screen.getByText(/not physical measurements/)).toBeInTheDocument()
  await userEvent.click(screen.getByRole('button', { name: 'Generate private datasets' }))
  await screen.findByText('Created 1 explicitly synthetic datasets.')
  await waitFor(() => expect(calls).toHaveLength(1))
  expect(calls[0]?.body).toEqual({ prefix: 'synthetic-research', seed: 20260913, samples_per_capture: 16384, repeats: 2 })
  expect(screen.getByRole('link', { name: 'Synthetic fixture' })).toHaveAttribute('href', '/datasets/synthetic-fixture')
})
