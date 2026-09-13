import { screen, waitFor } from '@testing-library/react'
import runningMock from '@mocks/run_running.json'
import type { RunView } from '@/api/types'
import { mockApi, renderWithProviders } from '@/test/utils'
import { ModelDownloadButton } from './ModelDownloadButton'

test('waits for a saved checkpoint, then changes to final model on completion', async () => {
  const run = runningMock.data as unknown as RunView
  let info = { available: false, final: false, download_url: null as string | null }
  mockApi({ [`GET /api/v1/runs/${run.run_id}/checkpoint`]: () => info })
  const { client, rerender } = renderWithProviders(<ModelDownloadButton run={run} />)
  expect(screen.getByRole('button', { name: 'Download checkpoint' })).toBeDisabled()
  await waitFor(() => expect(client.getQueryState(['run', run.run_id, 'checkpoint', run.status])?.status).toBe('success'))
  info = { available: true, final: false, download_url: `/api/v1/runs/${run.run_id}/checkpoint/download` }
  await client.invalidateQueries({ queryKey: ['run', run.run_id, 'checkpoint'] })
  expect(await screen.findByRole('link', { name: 'Download checkpoint' })).toHaveAttribute('href', info.download_url)
  info = { ...info, final: true }
  rerender(<ModelDownloadButton run={{ ...run, status: 'succeeded' }} />)
  await waitFor(() => expect(screen.getByRole('link', { name: 'Download final model' })).toHaveAttribute('href', info.download_url))
})
