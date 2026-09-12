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
