import { screen } from '@testing-library/react'
import { vi } from 'vitest'
import { mockApi, renderWithProviders } from '@/test/utils'

vi.mock('@/api/client', async importOriginal => ({ ...await importOriginal<typeof import('@/api/client')>(), WEB_MODE: true }))

import { MatlinkPage } from './MatlinkPage'

test('hosted Studio explains the local workflow without contacting a MATLAB bridge', () => {
  const { calls } = mockApi({})
  renderWithProviders(<MatlinkPage />)
  expect(screen.getByText(/MATLINK connects to desktop MATLAB through a local OpenDPD workspace/)).toBeInTheDocument()
  expect(screen.getByText('opendpd.studio()')).toBeInTheDocument()
  expect(screen.queryByRole('button', { name: 'Send to MATLAB' })).not.toBeInTheDocument()
  expect(calls).toHaveLength(0)
})
