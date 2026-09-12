import { screen } from '@testing-library/react'
import { mockApi, renderWithProviders } from '@/test/utils'
import { AboutPage } from './AboutPage'

test('shows requested leadership, original logos, and linked GitHub activity', async () => {
  mockApi({ 'GET /api/v1/system/about': () => ({ version: '2.2.0.dev0', local_commit: '123456abcdef', status: 'current', updated_at: '2026-09-12T11:00:00Z', contributors: [{ login: 'colleague', contributions: 3, url: 'https://github.com/colleague' }], commits: [{ sha: '12345678', url: 'https://github.com/lab-emi/OpenDPD/commit/12345678', message: 'Recent work', author: 'Contributor', date: '2026-09-12T11:00:00Z' }] }) })
  renderWithProviders(<AboutPage />)
  await screen.findByRole('link', { name: 'Recent work' })
  expect(screen.getByRole('img', { name: 'EMI Lab' })).toBeInTheDocument()
  expect(screen.getByRole('img', { name: 'TU Delft' })).toBeInTheDocument()
  expect(screen.getByText('Project Leader')).toBeInTheDocument()
  expect(screen.getByText('Leading Developer')).toBeInTheDocument()
  expect(screen.getByRole('link', { name: 'colleague' })).toHaveAttribute('href', 'https://github.com/colleague')
})
