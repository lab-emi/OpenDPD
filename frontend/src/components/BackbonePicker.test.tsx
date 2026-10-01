import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import { mockApi, renderWithProviders } from '@/test/utils'
import { BackbonePicker } from './BackbonePicker'

const digest = 'a'.repeat(64)
const record = { backbone_id: `ub-${digest}`, name: 'Example GRU', description: 'A small recurrent network.', author: 'Contributor',
  source_sha256: digest, definition_sha256: digest, publication_id: `bbpr-${digest}`, package_sha256: 'b'.repeat(64),
  model: { key: 'user_template', parameters: { definition: '{}' } }, parameter_count: 2000, node_count: 3,
  origin: 'private', status: 'prepared', files: [], branch: 'codex/backbone-fixture', directory: 'backbones/user_uploaded/fixture',
  created_at: '2026-09-19T00:00:00Z', updated_at: '2026-09-19T00:00:00Z', license: 'Apache-2.0' } as const

function mock(available = true) {
  let saved: unknown[] = []
  return mockApi({
    'GET /api/v1/backbones/uploads': () => saved,
    'GET /api/v1/backbones/catalog': () => ({ entries: [] }),
    'GET /api/v1/backbones/capability': () => ({ uploads_available: true, publication_available: available, reason: available ? null : 'The repository review gate is not enabled.' }),
    'POST /api/v1/backbones/uploads': () => { saved = [record]; return record },
    [`POST /api/v1/backbones/uploads/${record.publication_id}/submit`]: () => ({ ...record, status: 'queued' }),
    'POST /api/v1/backbones/catalog/refresh': () => ({ entries: [{ ...record, origin: 'community' }] }),
  })
}

function file(name = 'network.py') {
  const value = new File(['BACKBONE = {}'], name, { type: 'text/x-python' })
  Object.defineProperty(value, 'arrayBuffer', { value: async () => new TextEncoder().encode('BACKBONE = {}').buffer })
  return value
}

test('upload is private by default and selects its validated graph without opening a PR', async () => {
  const user = userEvent.setup()
  const { calls } = mock()
  const onSelect = vi.fn()
  renderWithProviders(<BackbonePicker selected={null} onSelect={onSelect} />)
  await user.click(screen.getByRole('button', { name: 'Upload backbone' }))
  expect(screen.getByRole('checkbox')).not.toBeChecked()
  await user.upload(screen.getByLabelText('Choose a .py template'), file())
  await waitFor(() => expect(screen.getByRole('button', { name: 'Validate & add to workspace' })).toBeEnabled())
  await user.click(screen.getByRole('button', { name: 'Validate & add to workspace' }))
  await waitFor(() => expect(onSelect).toHaveBeenCalledWith(record))
  expect(calls.some(c => c.path.endsWith('/submit'))).toBe(false)
  expect(calls.find(c => c.method === 'POST')?.body).toEqual({ filename: 'network.py', source: 'BACKBONE = {}' })
})

test('public checkbox submits exact validated hashes; changing the file revokes consent', async () => {
  const user = userEvent.setup()
  const { calls } = mock()
  renderWithProviders(<BackbonePicker selected={null} onSelect={() => undefined} />)
  await user.click(screen.getByRole('button', { name: 'Upload backbone' }))
  await user.upload(screen.getByLabelText('Choose a .py template'), file())
  await waitFor(() => expect(screen.getByRole('checkbox')).toBeEnabled())
  await user.click(screen.getByRole('checkbox'))
  await user.upload(screen.getByLabelText('Choose a .py template'), file('second.py'))
  await waitFor(() => expect(screen.getByRole('checkbox')).not.toBeChecked())
  expect(calls.every(c => c.method === 'GET')).toBe(true)
  await user.click(screen.getByRole('checkbox'))
  await user.click(screen.getByRole('button', { name: 'Validate, add & open PR' }))
  await waitFor(() => expect(calls.some(c => c.path.endsWith('/submit'))).toBe(true))
  expect(calls.find(c => c.path.endsWith('/submit'))?.body).toEqual({ source_sha256: digest, package_sha256: 'b'.repeat(64), publish_publicly: true, rights_confirmed: true })
})

test('missing repository enforcement disables contribution but leaves private upload usable', async () => {
  const user = userEvent.setup()
  mock(false)
  renderWithProviders(<BackbonePicker selected={null} onSelect={() => undefined} />)
  await user.click(screen.getByRole('button', { name: 'Upload backbone' }))
  await screen.findByText('The repository review gate is not enabled.')
  expect(screen.getByRole('checkbox')).toBeDisabled()
  await user.upload(screen.getByLabelText('Choose a .py template'), file())
  await waitFor(() => expect(screen.getByRole('button', { name: 'Validate & add to workspace' })).toBeEnabled())
})

test('merged catalog uses its own dropdown; pending and private uploads cannot enter it', async () => {
  const user = userEvent.setup()
  const onSelect = vi.fn()
  mock()
  renderWithProviders(<BackbonePicker selected={null} onSelect={onSelect} />)
  await screen.findByText('No merged contributions loaded. Refresh to check GitHub main.')
  await user.click(screen.getByRole('button', { name: 'Refresh community backbones' }))
  await screen.findByText('Community contributions merged into OpenDPD main.')
  await user.click(screen.getByRole('combobox', { name: 'User Uploaded Backbones' }))
  await user.click(screen.getByRole('option', { name: 'Example GRU · aaaaaaaa' }))
  expect(onSelect).toHaveBeenCalledWith(expect.objectContaining({ origin: 'community', model: record.model }))
})

test('invalid and oversized files cannot be uploaded', async () => {
  const user = userEvent.setup({ applyAccept: false })
  const { calls } = mock()
  renderWithProviders(<BackbonePicker selected={null} onSelect={() => undefined} />)
  await user.click(screen.getByRole('button', { name: 'Upload backbone' }))
  await user.upload(screen.getByLabelText('Choose a .py template'), file('evil.exe'))
  await screen.findByText(/Choose a non-empty UTF-8/)
  expect(screen.getByRole('button', { name: 'Validate & add to workspace' })).toBeDisabled()
  expect(calls.every(c => c.method === 'GET')).toBe(true)
})
