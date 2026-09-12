import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useState } from 'react'
import { Route, Routes, useLocation } from 'react-router'
import runMock from '@mocks/run_running.json'
import { mockApi, renderWithProviders } from '@/test/utils'
import { AppShell } from './AppShell'

function Page() {
  const location = useLocation()
  const [draft, setDraft] = useState('')
  return <><output data-testid="location">{location.pathname}{location.search}{location.hash}</output><label>Draft<input value={draft} onChange={(event) => setDraft(event.target.value)} /></label></>
}

function setup(route: string, task = 'train_pa') {
  vi.spyOn(window, 'scrollTo').mockImplementation(() => undefined)
  const { calls } = mockApi({
    'GET /api/v1/system/capabilities': () => ({ workspace: '/reset-test', version: 'test' }),
    'GET /api/v1/runs': () => [{ ...runMock.data, task, run_id: 'selected-run' }],
    'GET /api/v1/runs/selected-run': () => ({ ...runMock.data, task, run_id: 'selected-run' }),
    'GET /api/v1/runs/selected-run/logs': () => ({ lines: ['Actual saved log'], next_offset: 16, eof: true, size: 16 }),
  })
  renderWithProviders(<Routes><Route element={<AppShell />}><Route path="*" element={<Page />} /></Route></Routes>, { route })
  return calls
}

async function confirmReset() {
  await userEvent.click(screen.getByRole('button', { name: 'Reset page' }))
  await userEvent.click(within(screen.getByRole('dialog')).getByRole('button', { name: 'Continue' }))
  await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())
}

test.each([
  ['/datasets/picked?tab=doctor&version=processed#chart', '/datasets'],
  ['/results/selected-run?profile=custom', '/results'],
  ['/results/compare?runs=one&runs=two', '/results'],
  ['/experiments?status=failed&q=search&page=3&size=25', '/experiments'],
  ['/experiments/new?task=train_dpd&dataset=picked&from=old#review', '/experiments/new?task=train_dpd'],
  ['/?panel=recent#bottom', '/'],
  ['/settings#devices', '/settings'],
  ['/about#contributors', '/about'],
])('reset returns %s to %s with draft and scroll position cleared', async (route, destination) => {
  const calls = setup(route)
  await userEvent.type(screen.getByLabelText('Draft'), 'Discard this progress')
  await confirmReset()
  expect(screen.getByTestId('location')).toHaveTextContent(destination)
  expect(screen.getByTestId('location').textContent).toBe(destination)
  expect(screen.getByLabelText('Draft')).toHaveValue('')
  expect(window.scrollTo).toHaveBeenCalledWith({ top: 0, left: 0, behavior: 'instant' })
  expect(screen.getByRole('alert')).toHaveTextContent('Reset complete.')
  expect(calls.every((call) => call.method === 'GET')).toBe(true)
})

test.each(['train_pa', 'evaluate_pa', 'train_dpd', 'run_dpd'])('a %s run restarts at setup without reusing the old progress URL', async (task) => {
  const calls = setup('/runs/selected-run?tab=logs', task)
  await waitFor(() => expect(calls.some((call) => call.path === '/api/v1/runs/selected-run')).toBe(true))
  await userEvent.click(await screen.findByRole('button', { name: /Terminal.*Running/ }))
  await screen.findByRole('log')
  await confirmReset()
  expect(screen.getByTestId('location').textContent).toBe(`/experiments/new?task=${task}`)
  expect(screen.getByRole('button', { name: /Terminal.*Running/ })).toHaveAttribute('aria-expanded', 'false')
  expect(calls.every((call) => call.method === 'GET')).toBe(true)
})

test('same-task reset clears the independent terminal tab and collapses it', async () => {
  setup('/experiments/new?task=train_pa')
  await userEvent.click(await screen.findByRole('button', { name: /Terminal.*Running/ }))
  await userEvent.click(within(screen.getByRole('tablist', { name: 'Terminal steps' })).getByRole('tab', { name: 'DPD Model Training' }))
  await confirmReset()
  const terminal = screen.getByRole('button', { name: /Terminal.*Running/ })
  expect(terminal).toHaveAttribute('aria-expanded', 'false')
  await userEvent.click(terminal)
  expect(within(screen.getByRole('tablist', { name: 'Terminal steps' })).getByRole('tab', { name: 'PA Model Training' })).toHaveAttribute('aria-selected', 'true')
})
