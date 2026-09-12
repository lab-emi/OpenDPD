import { act, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { renderWithProviders } from '@/test/utils'
import { ApiError } from './client'
import { useRunStream } from './events'

const get = vi.hoisted(() => vi.fn())
vi.mock('./client', async (original) => ({ ...await original<typeof import('./client')>(), WEB_MODE: true, api: { get } }))

function Viewer() {
  const stream = useRunStream('remote-run', true)
  return <><button onClick={stream.reconnect}>Reconnect</button><output>{stream.connection} / {stream.lastSeq} / {stream.metrics.map((m) => m.values.NMSE).join(',')}</output></>
}

test('a failed poll reconnects from the cursor and preserves plots, including completion while disconnected', async () => {
  const metric = (seq: number) => ({ seq, run_id: 'remote-run', ts: new Date().toISOString(), type: 'metric', payload: { epoch: seq, values: { NMSE: -seq } } })
  get.mockReset().mockResolvedValueOnce({ events: [metric(1)], last_seq: 1, terminal: false })
    .mockRejectedValueOnce(new TypeError('network disconnected'))
    .mockResolvedValueOnce({ events: [metric(2)], last_seq: 2, terminal: true })
  renderWithProviders(<Viewer />)
  await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('live / 1 / -1'))
  await userEvent.click(screen.getByRole('button', { name: 'Reconnect' }))
  await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('disconnected / 1 / -1'))
  await userEvent.click(screen.getByRole('button', { name: 'Reconnect' }))
  await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('ended / 2 / -1,-2'))
  expect(get.mock.calls[1]![0]).toContain('after=1')
  expect(get.mock.calls[2]![0]).toContain('after=1')
})

test('a rejected cursor replays stored events instead of remaining disconnected forever', async () => {
  get.mockReset().mockRejectedValueOnce(new ApiError(409, 'cursor_out_of_range', 'stale cursor'))
    .mockResolvedValueOnce({ events: [], last_seq: 0, terminal: true })
  renderWithProviders(<Viewer />)
  await waitFor(() => expect(get).toHaveBeenCalledOnce())
  await act(async () => window.dispatchEvent(new Event('online')))
  // Retry-After/backoff is not bypassed by online notifications; manual recovery is explicit.
  await userEvent.click(screen.getByRole('button', { name: 'Reconnect' }))
  await waitFor(() => expect(screen.getByRole('status')).toHaveTextContent('ended / 0'))
})
