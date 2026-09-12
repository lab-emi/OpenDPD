import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { mockApi, renderWithProviders } from '@/test/utils'
import { LogViewer } from './LogViewer'

test('pages logs by byte offset and filters loaded lines', async () => {
  const lines = Array.from({ length: 600 }, (_, i) => `line ${i} ${i % 2 ? 'odd' : 'even'}`)
  const { calls } = mockApi({
    'GET /api/v1/runs/run-1/logs': (url) => {
      const offset = Number(url.searchParams.get('offset'))
      const limit = Number(url.searchParams.get('limit'))
      const page = lines.slice(offset, offset + limit)
      return { lines: page, next_offset: offset + page.length, eof: offset + page.length >= lines.length, size: lines.length }
    },
  })
  renderWithProviders(<LogViewer runId="run-1" live={false} height={200} />)
  await screen.findByText('line 0 even')
  expect(screen.getByText('500 of 500 loaded lines')).toBeInTheDocument()
  await userEvent.click(screen.getByRole('button', { name: 'Load more' }))
  await waitFor(() => expect(screen.getByText('600 of 600 loaded lines')).toBeInTheDocument())
  expect(calls.filter((c) => c.path.endsWith('/logs')).length).toBe(2)
  await userEvent.type(screen.getByLabelText('Filter lines'), 'line 599')
  await waitFor(() => expect(screen.getByText('1 of 600 loaded lines')).toBeInTheDocument())
  // windowed: only a handful of rows are in the DOM
  expect(document.querySelectorAll('[role="log"] div div').length).toBeLessThan(40)
})
