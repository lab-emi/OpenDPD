import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { Route, Routes } from 'react-router'
import datasetMock from '@mocks/dataset_builtin.json'
import resolvedMock from '@mocks/resolved_train_pa_smoke.json'
import runMock from '@mocks/run_running.json'
import { mockApi, renderWithProviders } from '@/test/utils'
import { AppShell } from '@/layout/AppShell'
import { HomePage } from './HomePage'
import { DatasetsPage } from './DatasetsPage'
import { NewExperimentPage } from './NewExperimentPage'

function setup(route: string, customDatasets = false) {
  vi.spyOn(window, 'scrollTo').mockImplementation(() => undefined)
  const { calls } = mockApi({
    'GET /api/v1/system/capabilities': () => ({ custom_dataset_imports: customDatasets, workspace: '/my-workspace', version: 'test', devices: [{ device: 'cpu', detected: true }] }),
    'GET /api/v1/datasets': () => [datasetMock.data],
    'GET /api/v1/runs': () => [runMock.data],
    'GET /api/v1/recipes': () => [{ ...resolvedMock.data, title: 'Preset', purpose: 'smoke', description: '', limits: '', expected_duration: '' }],
    'GET /api/v1/models': () => [{ key: 'gru', display_name: 'GRU', training_method: 'gradient', params: [], devices_tested: ['cpu'] }],
    'GET /api/v1/datasets/import-defaults': () => ({ ratios: { train: .6, val: .2, test: .2 }, guard_samples: 256 }),
    'POST /api/v1/experiments/validate': () => ({ ok: true, errors: [], warnings: [], resolved: null }),
  })
  renderWithProviders(<Routes><Route element={<AppShell />}>
    <Route path="/" element={<HomePage />} />
    <Route path="/datasets" element={<DatasetsPage />} />
    <Route path="/experiments/new" element={<NewExperimentPage />} />
  </Route></Routes>, { route })
  return calls
}

test('Get Started opens an optional dataset guide without registering anything', async () => {
  const calls = setup('/')
  await userEvent.click(await screen.findByRole('link', { name: 'Get Started' }))
  expect(await screen.findByRole('dialog', { name: /Create your first dataset/ })).toBeVisible()
  await userEvent.click(screen.getByRole('button', { name: 'Skip tutorial' }))
  await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())
  expect(screen.getByRole('heading', { level: 1, name: 'Datasets' })).toBeVisible()
  expect(await screen.findByRole('link', { name: datasetMock.data.display_name })).toBeVisible()
  expect(calls.every((call) => call.method === 'GET')).toBe(true)
})

test('the guide opens the real CSV form and can be skipped without closing that form', async () => {
  setup('/datasets?guide=start', true)
  await userEvent.click(await screen.findByRole('button', { name: 'Use my CSV file' }))
  const dialog = screen.getByRole('dialog', { name: 'Create Your Own Dataset' })
  await within(dialog).findByText(/Step 1 · Upload a CSV/)
  await userEvent.click(within(dialog).getByRole('button', { name: 'Skip tutorial' }))
  expect(dialog).toBeVisible()
  expect(within(dialog).queryByText(/Step 1 · Upload a CSV/)).not.toBeInTheDocument()
  expect(within(dialog).getByRole('button', { name: 'Choose CSV file' })).toBeVisible()
})

test('page reset restores a pending experiment; global reset restarts the guide and preserves server records', async () => {
  const calls = setup('/experiments/new?task=train_pa')
  const user = userEvent.setup()
  await screen.findByText('Configuration is valid')
  await user.click(screen.getByRole('button', { name: 'Continue' }))
  await user.type(screen.getByLabelText('Epochs'), '9')
  await user.type(screen.getByLabelText('Name (optional)'), 'Discard this draft')
  await user.click(screen.getByRole('button', { name: 'Reset page' }))
  await user.click(screen.getByRole('button', { name: 'Cancel' }))
  await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())
  expect(screen.getByLabelText('Name (optional)')).toHaveValue('Discard this draft')
  await user.click(screen.getByRole('button', { name: 'Reset page' }))
  await user.click(within(screen.getByRole('dialog', { name: 'Reset this page?' })).getByRole('button', { name: 'Continue' }))
  await waitFor(() => expect(screen.getByRole('tab', { name: /Data & model/ })).toHaveAttribute('aria-selected', 'true'))
  await waitFor(() => expect(screen.getByRole('button', { name: 'Continue' })).toBeEnabled())
  await user.click(screen.getByRole('button', { name: 'Continue' }))
  expect(screen.getByLabelText('Name (optional)')).toHaveValue('')
  expect(screen.getByLabelText('Epochs')).toHaveValue(null)
  expect(screen.getByLabelText('Epochs')).toHaveAttribute('placeholder', String(resolvedMock.data.training.epochs))
  await user.click(screen.getByRole('button', { name: 'Reset Studio' }))
  expect(screen.getByRole('dialog')).toHaveTextContent('running jobs and your language choice are kept')
  await user.click(screen.getByRole('button', { name: 'Reset' }))
  await screen.findByRole('dialog', { name: /Create your first dataset/ })
  await user.click(screen.getByRole('button', { name: 'Skip tutorial' }))
  expect(await screen.findByRole('link', { name: datasetMock.data.display_name })).toBeVisible()
  expect(calls.filter((call) => call.method !== 'GET').every((call) => call.path === '/api/v1/experiments/validate')).toBe(true)
// This journey crosses several forms and confirmation dialogs on shared CI runners.
}, 15000)
