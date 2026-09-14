import { screen, within } from '@testing-library/react'
import { mockApi, renderWithProviders } from '@/test/utils'
import { ServerStatusPage } from './ServerStatusPage'
import { WorkspaceExpiry } from '@/components/WorkspaceExpiry'

const snapshot = {
  mode: 'web', sampled_at: '2026-09-14T12:00:00Z', active_sessions: 3, workspaces: 4,
  workspace_capacity: 16, running_jobs: 1, queued_jobs: 2, parallel_capacity: 1,
  api: { stale: false, load: { cpu_percent: 23, memory_percent: 40, sampled_at: '2026-09-14T12:00:00Z' } },
  compute: { stale: false, load: { cpu_percent: 17, memory_percent: 50, sampled_at: '2026-09-14T12:00:00Z', gpu: { utilization_percent: 67, memory_used_bytes: 2**30, memory_total_bytes: 8 * 2**30 } } },
}

test('displays scoped machine metrics and anonymous session estimate', async () => {
  mockApi({ 'GET /api/v1/system/status': () => snapshot })
  renderWithProviders(<ServerStatusPage />)
  await screen.findByText('Active users')
  expect(screen.getByText(/one person may have multiple sessions/)).toBeInTheDocument()
  expect(screen.getByRole('progressbar', { name: 'GPU 0 utilization' })).toHaveAttribute('aria-valuenow', '67')
  expect(screen.getByText('4 / 16 temporary workspaces open')).toBeInTheDocument()
})

test('stale compute data is never displayed as a current utilization value', async () => {
  mockApi({ 'GET /api/v1/system/status': () => ({ ...snapshot, compute: { ...snapshot.compute, stale: true } }) })
  renderWithProviders(<ServerStatusPage />)
  await screen.findByText('No recent sample. Utilization is unavailable until telemetry resumes.')
  expect(screen.queryByText('67%')).not.toBeInTheDocument()
})

test('one unambiguous UTC deletion timestamp replaces the page banner', () => {
  renderWithProviders(<WorkspaceExpiry expiresAt="2026-09-14T23:55:00Z" />)
  const expiry = screen.getByTestId('workspace-expiry')
  expect(within(expiry).getByText('2026-09-14 23:55:00 UTC')).toHaveAttribute('datetime', '2026-09-14T23:55:00.000Z')
  expect(screen.queryByRole('alert')).not.toBeInTheDocument()
})
