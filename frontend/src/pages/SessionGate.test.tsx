import { screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { expect, test, vi } from 'vitest'
import { ApiConnectionError, createWebSession, loadSession } from '@/api/client'
import { SessionGate } from '@/pages/SessionGate'
import { renderWithProviders } from '@/test/utils'

vi.mock('@/api/client', async (importOriginal) => ({
  ...await importOriginal<typeof import('@/api/client')>(),
  WEB_MODE: true,
  loadSession: vi.fn(),
  createWebSession: vi.fn(),
}))

test('a failed connection explains recovery and Retry opens the workspace when connectivity returns', async () => {
  vi.mocked(loadSession).mockResolvedValue({ authenticated: false, version: '', mode: 'web' })
  vi.mocked(createWebSession).mockRejectedValueOnce(new ApiConnectionError('https://api.opendpd.com'))
    .mockResolvedValueOnce({ authenticated: true, version: '', mode: 'web' })
  renderWithProviders(<SessionGate><p>Workspace ready</p></SessionGate>)
  const user = userEvent.setup()
  await user.click(await screen.findByRole('button', { name: 'Start a temporary session' }))
  expect(await screen.findByText('Cannot connect to the compute server')).toBeInTheDocument()
  expect(screen.getByText(/old DNS records/)).toBeInTheDocument()
  expect(screen.queryByText('Workspace ready')).not.toBeInTheDocument()
  await user.click(screen.getByRole('button', { name: 'Retry' }))
  expect(await screen.findByText('Workspace ready')).toBeInTheDocument()
  expect(createWebSession).toHaveBeenCalledTimes(2)
})
