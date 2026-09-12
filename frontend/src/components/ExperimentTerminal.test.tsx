import { screen, within, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Link, useLocation } from 'react-router'
import runningMock from '@mocks/run_running.json'
import { mockApi, renderWithProviders } from '@/test/utils'
import { ExperimentTerminal } from './ExperimentTerminal'

function Workspace() {
  const location = useLocation()
  return <><output data-testid="page-path">{location.search}</output><Link to="/experiments/new?task=train_dpd">Go to DPD training</Link><ExperimentTerminal /></>
}

test('collapsed terminal follows steps; selecting a different log does not navigate or submit', async () => {
  const pa = { ...runningMock.data, run_id: 'run-pa', name: 'PA run', task: 'train_pa', status: 'running' }
  const dpd = { ...runningMock.data, run_id: 'run-dpd', name: 'DPD run', task: 'train_dpd', status: 'running' }
  const { calls } = mockApi({
    'GET /api/v1/runs': () => [dpd, pa],
    'GET /api/v1/runs/run-pa/logs': () => ({ lines: ['PA worker: batch 25'], next_offset: 20, eof: true, size: 20 }),
    'GET /api/v1/runs/run-dpd/logs': () => ({ lines: ['DPD worker: batch 50'], next_offset: 20, eof: true, size: 20 }),
  })
  renderWithProviders(<Workspace />, { route: '/experiments/new?task=train_pa' })
  const header = await screen.findByRole('button', { name: /Terminal.*Running/ })
  expect(header).toHaveAttribute('aria-expanded', 'false')
  expect(calls.some((call) => call.path.endsWith('/logs'))).toBe(false)
  await userEvent.click(header)
  expect(await screen.findByRole('log')).toHaveTextContent('PA worker')
  await userEvent.click(screen.getByRole('link', { name: 'Go to DPD training' }))
  await waitFor(() => expect(screen.getByRole('log')).toHaveTextContent('DPD worker'))
  const tabs = screen.getByRole('tablist', { name: 'Terminal steps' })
  await userEvent.click(within(tabs).getByRole('tab', { name: 'PA Model Training' }))
  await waitFor(() => expect(screen.getByRole('log')).toHaveTextContent('PA worker'))
  expect(screen.getByTestId('page-path')).toHaveTextContent('task=train_dpd')
  expect(calls.every((call) => call.method === 'GET')).toBe(true)
})

test('Stop cancels only the selected experiment and provides no command input', async () => {
  const run = { ...runningMock.data, run_id: 'run-selected', task: 'train_pa', status: 'running' }
  const { calls } = mockApi({
    'GET /api/v1/runs': () => [run],
    'GET /api/v1/runs/run-selected/logs': () => ({ lines: ['Training on CUDA'], next_offset: 20, eof: true, size: 20 }),
    'POST /api/v1/runs/run-selected/cancel': () => ({ ...run, status: 'cancel_requested' }),
  })
  renderWithProviders(<ExperimentTerminal />, { route: '/experiments/new?task=train_pa' })
  await userEvent.click(await screen.findByRole('button', { name: /Terminal.*Running/ }))
  const terminal = screen.getByTestId('experiment-terminal')
  expect(within(terminal).getAllByRole('textbox')).toHaveLength(1)
  expect(within(terminal).getByRole('textbox')).toHaveAccessibleName('Filter lines')
  await userEvent.click(within(terminal).getByRole('button', { name: 'Stop experiment' }))
  await waitFor(() => expect(calls.filter((call) => call.method === 'POST').map((call) => call.path)).toEqual(['/api/v1/runs/run-selected/cancel']))
})
