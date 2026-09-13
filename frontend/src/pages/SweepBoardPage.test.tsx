import { fireEvent, screen, waitFor, within } from '@testing-library/react'
import { mockApi, renderWithProviders } from '@/test/utils'
import { SweepBoardPage } from './SweepBoardPage'
import dataset from '@mocks/dataset_builtin.json'
import experiment from '@mocks/experiment_train_pa_smoke.json'
import legacyProfile from '@mocks/metric_profile_legacy.json'

const recipe = { recipe_id: 'pa-gru-smoke-v1', title: 'PA GRU smoke', task: 'train_pa', model: experiment.data.model, training: experiment.data.training }

test('requires a fresh matrix preview before registration and sends explicit budgets', async () => {
  const calls = mockApi({
    'GET /api/v1/sweeps': () => [],
    'GET /api/v1/runs': () => [],
    'GET /api/v1/datasets': () => [dataset.data],
    'GET /api/v1/recipes': () => [recipe],
    'GET /api/v1/metrics/profiles': () => [legacyProfile.data],
    'POST /api/v1/sweeps/preview': (_url, init) => ({ draft: JSON.parse(String(init.body)), cells: [], training_runs: 3, evaluation_runs: 0, sample_epochs: 300, errors: [], warnings: [] }),
    'POST /api/v1/sweeps': (_url, init) => ({ sweep_id: 'sweep-test', preview: { draft: JSON.parse(String(init.body)) }, cells: [], status: 'ready' }),
  }).calls
  renderWithProviders(<SweepBoardPage />, { route: '/sweeps', path: '/sweeps/*' })
  fireEvent.click(screen.getByRole('button', { name: 'Create experiment matrix' }))
  const dialog = screen.getByRole('dialog')
  await within(dialog).findByText('PA GRU smoke')
  const register = within(dialog).getByRole('button', { name: 'Register plan' })
  expect(register).toBeDisabled()
  fireEvent.change(within(dialog).getByLabelText('Epochs'), { target: { value: '1' } })
  fireEvent.change(within(dialog).getByLabelText('Maximum runs'), { target: { value: '6' } })
  fireEvent.change(within(dialog).getByLabelText('Wall-clock limit (s)'), { target: { value: '120' } })
  fireEvent.click(within(dialog).getByRole('button', { name: 'Preview matrix' }))
  await waitFor(() => expect(register).toBeEnabled())
  const preview = calls.find(c => c.path.endsWith('/preview'))?.body as { seeds: number[]; max_runs: number; max_wall_clock_seconds: number; methods: Array<{ config: { training: { epochs: number } } }> }
  expect(preview.seeds).toEqual([0, 1, 2])
  expect(preview.max_runs).toBe(6)
  expect(preview.max_wall_clock_seconds).toBe(120)
  expect(preview.methods[0]?.config.training.epochs).toBe(1)
  expect(calls.some(c => c.method === 'POST' && c.path === '/api/v1/runs')).toBe(false)
  fireEvent.change(within(dialog).getByLabelText('Training seeds'), { target: { value: '2, 3, 4' } })
  expect(register).toBeDisabled()
  fireEvent.click(within(dialog).getByRole('button', { name: 'Preview matrix' }))
  await waitFor(() => expect(register).toBeEnabled())
  fireEvent.click(register)
  await waitFor(() => expect(calls.some(c => c.method === 'POST' && c.path === '/api/v1/sweeps')).toBe(true))
  expect(calls.some(c => c.path.endsWith('/start'))).toBe(false)
})
