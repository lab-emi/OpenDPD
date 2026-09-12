import { fireEvent, screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { mockApi, renderWithProviders } from '@/test/utils'
import { DatasetsPage } from './DatasetsPage'
import { ExperimentsPage } from './ExperimentsPage'

test('all dataset entry buttons are grey and inert; built-in examples remain available', async () => {
  const { calls } = mockApi({
    'GET /api/v1/system/capabilities': () => ({ custom_dataset_imports: false }),
    'GET /api/v1/datasets': () => [],
  })
  renderWithProviders(<DatasetsPage />, { route: '/datasets' })
  const create = screen.getByRole('button', { name: 'Create Your Own Dataset · Coming soon' })
  const advanced = screen.getByRole('button', { name: 'Advanced import · Coming soon' })
  expect(create).toBeDisabled()
  expect(advanced).toBeDisabled()
  fireEvent.click(create)
  fireEvent.click(advanced)
  expect(screen.queryByRole('dialog')).not.toBeInTheDocument()
  expect(document.querySelector('input[type="file"]')).toBeNull()
  expect(screen.getByRole('button', { name: 'Built-in datasets' })).toBeEnabled()
  await userEvent.click(screen.getByRole('button', { name: 'Show tutorial' }))
  const guide = screen.getByRole('dialog')
  expect(within(guide).getByRole('button', { name: 'Use my CSV file · Coming soon' })).toBeDisabled()
  expect(within(guide).getByRole('button', { name: 'Try a built-in dataset' })).toBeEnabled()
  expect(calls.every(call => call.method === 'GET')).toBe(true)
  expect(calls.some(call => /upload|import-roots|csv\/preview/.test(call.path))).toBe(false)
})

test('a missing capability defaults to closed and no package file picker is mounted', async () => {
  const { calls } = mockApi({ 'GET /api/v1/runs': () => [] })
  renderWithProviders(<ExperimentsPage />, { route: '/experiments' })
  const button = screen.getByRole('button', { name: /Import.*Coming soon/ })
  expect(button).toBeDisabled()
  fireEvent.click(button)
  expect(document.querySelector('input[type="file"]')).toBeNull()
  expect(calls.every(call => call.method === 'GET')).toBe(true)
})
