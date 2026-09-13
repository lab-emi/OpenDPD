import { fireEvent, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import mockDataset from '@mocks/dataset_builtin.json'
import type { DatasetManifest } from '@/api/types'
import { mockApi, renderWithProviders } from '@/test/utils'
import { DatasetPublicationPanel } from './DatasetPublicationPanel'

const dataset = mockDataset.data as unknown as DatasetManifest
const id = `dspr-${'a'.repeat(64)}`
const record = { publication_id: id, dataset_id: dataset.dataset_id, package_sha256: 'a'.repeat(64), repository: 'lab-emi/OpenDPD', directory: 'dataset/community/synthetic/fixture/hash',
  status: 'prepared', files: [{ path: 'data.csv', size_bytes: 4096 }], catalog: { description: 'Public fixture', attribution: 'Contributor', license: 'CC0-1.0' }, created_at: '2026-09-13T12:00:00Z' }

test('public disclosure needs exact preview and rights consent; editing invalidates the preview', async () => {
  const user = userEvent.setup()
  let records: unknown[] = []
  const { calls } = mockApi({
    'GET /api/v1/dataset-publications/capability': () => ({ available: true }),
    'GET /api/v1/dataset-publications': () => records,
    'POST /api/v1/dataset-publications/prepare': (_url, init) => { records = [{ ...record, catalog: JSON.parse(String(init.body)) }]; return records[0] },
    [`POST /api/v1/dataset-publications/${id}/submit`]: () => { records = [{ ...record, status: 'submitted', pull_request_url: 'https://github.com/lab-emi/OpenDPD/pull/123', pull_request_state: 'OPEN' }]; return records[0] },
  })
  renderWithProviders(<DatasetPublicationPanel dataset={dataset} />)
  expect(calls).toHaveLength(0)
  await user.click(screen.getByRole('button', { name: 'Prepare a public dataset PR' }))
  expect(screen.getByText(/Dataset contributions require human review/)).toBeInTheDocument()
  expect(screen.getByRole('link', { name: 'emi.lab@outlook.com' })).toHaveAttribute('href', 'mailto:emi.lab@outlook.com')
  expect(screen.getByRole('button', { name: 'Publish and request human review' })).toBeDisabled()
  fireEvent.change(screen.getByLabelText('Public dataset description'), { target: { value: 'Public fixture' } })
  fireEvent.change(screen.getByLabelText('Public author / attribution'), { target: { value: 'Contributor' } })
  await user.click(screen.getByLabelText('Dataset license'))
  await user.click(screen.getByRole('option', { name: 'CC0-1.0' }))
  await user.click(screen.getByRole('button', { name: 'Prepare and inspect package' }))
  await screen.findByText('Private preview')
  expect(calls.some(c => c.path.endsWith('/submit'))).toBe(false)
  await user.click(screen.getByRole('checkbox'))
  expect(screen.getByRole('button', { name: 'Publish and request human review' })).toBeEnabled()
  fireEvent.change(screen.getByLabelText('Public dataset description'), { target: { value: 'Edited fixture' } })
  expect(screen.getByRole('button', { name: 'Publish and request human review' })).toBeDisabled()
  await user.click(screen.getByRole('button', { name: 'Prepare and inspect package' }))
  await waitFor(() => expect(screen.getByRole('checkbox')).not.toBeChecked())
  await user.click(screen.getByRole('checkbox'))
  await user.click(screen.getByRole('button', { name: 'Publish and request human review' }))
  await waitFor(() => expect(calls.some(c => c.path.endsWith('/submit'))).toBe(true))
  expect(calls.find(c => c.path.endsWith('/submit'))?.body).toEqual({ package_sha256: 'a'.repeat(64), publish_publicly: true, rights_confirmed: true })
  expect((await screen.findAllByRole('link', { name: 'Open PR and human review' }))[0]).toHaveAttribute('href', 'https://github.com/lab-emi/OpenDPD/pull/123')
})

test('lack of GitHub authentication explains availability without making a publication', async () => {
  const { calls } = mockApi({
    'GET /api/v1/dataset-publications/capability': () => ({ available: false, reason: 'Sign in with gh auth login on the Studio host.' }),
    'GET /api/v1/dataset-publications': () => [],
  })
  renderWithProviders(<DatasetPublicationPanel dataset={dataset} />)
  await userEvent.click(screen.getByRole('button', { name: 'Prepare a public dataset PR' }))
  await screen.findByText('Sign in with gh auth login on the Studio host.')
  expect(screen.getByRole('button', { name: 'Publish and request human review' })).toBeDisabled()
  expect(calls.every(c => c.method === 'GET')).toBe(true)
})
