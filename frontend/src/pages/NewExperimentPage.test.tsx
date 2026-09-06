import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import datasetMock from '@mocks/dataset_builtin.json'
import runQueued from '@mocks/run_queued.json'
import { mockApi, renderWithProviders } from '@/test/utils'
import { NewExperimentPage } from './NewExperimentPage'

const recipe = {
  recipe_id: 'pa-gru-smoke-v1',
  title: 'PA model · GRU · smoke',
  purpose: 'smoke',
  task: 'train_pa',
  model: { key: 'gru', parameters: { hidden_size: 23, num_layers: 1 } },
  training: { epochs: 3, batch_size: 64, batch_size_eval: 256, learning_rate: 0.005, lr_end: 0.00005, lr_schedule: true, decay_factor: 0.5, patience: 5, optimizer: 'adamw', loss: 'l2', grad_clip: 200, frame_length: 50, frame_stride: 16, seed: 0, reproducibility: 'soft', eval_val: true, eval_test: true },
  description: 'Proves the PA pipeline.',
  limits: 'Not a benchmark.',
  expected_duration: '~1 min on CPU',
}
const model = {
  key: 'gru',
  display_name: 'GRU',
  family: 'recurrent',
  legacy_backbone: 'gru',
  training_method: 'gradient',
  roles: ['pa', 'dpd'],
  params: [
    { name: 'hidden_size', type: 'int', default: 23, description: 'Hidden state size of the backbone', minimum: 1, maximum: null, choices: null, legacy_arg: {} },
    { name: 'num_layers', type: 'int', default: 1, description: 'Number of stacked recurrent layers', minimum: 1, maximum: 8, choices: null, legacy_arg: {} },
  ],
  status: 'supported',
  devices_tested: ['cpu', 'cuda'],
  lookahead_samples: 0,
  lookahead_note: '',
  execution_semantics: 'offline_segmented',
  export_formats: [],
}
const caps = { version: 'x', workspace: '/ws', note: '', devices: [{ device: 'cpu', detected: true, count: 1, tested_models: ['gru'] }, { device: 'cuda', detected: false, count: 0, tested_models: ['gru'] }] }

function base(validate: (config: Record<string, unknown>) => unknown) {
  return mockApi({
    'GET /api/v1/recipes': () => [recipe],
    'GET /api/v1/datasets': () => [datasetMock.data],
    'GET /api/v1/models': () => [model],
    'GET /api/v1/system/capabilities': () => caps,
    'GET /api/v1/runs': () => [],
    'POST /api/v1/experiments/validate': (_url, init) => validate((JSON.parse(String(init.body)) as { config: Record<string, unknown> }).config),
    'POST /api/v1/runs': () => ({ status: 201, body: runQueued.data }),
  })
}

test('server validation errors are shown on the field and block submit', async () => {
  base((config) => {
    const epochs = (config['training'] as { epochs: number }).epochs
    return epochs > 0 ? { ok: true, errors: [], warnings: [], resolved: null } : { ok: false, errors: [{ field: 'training.epochs', message: 'must be at least 1', hint: 'use 3 for a smoke run' }], warnings: [], resolved: null }
  })
  renderWithProviders(<NewExperimentPage />, { route: '/experiments/new', path: '/experiments/new' })
  await screen.findByText('Smoke recipe: a few epochs to prove the pipeline. Results are not a benchmark.')
  await screen.findByText('Configuration is valid')
  await userEvent.click(screen.getByRole('button', { name: 'Advanced settings' }))
  const epochs = screen.getByLabelText('Epochs')
  await userEvent.clear(epochs)
  await userEvent.type(epochs, '0')
  await screen.findByText('Fix the following before submitting')
  expect(screen.getByLabelText('Epochs')).toHaveAttribute('aria-invalid', 'true')
  expect(screen.getAllByText(/must be at least 1/).length).toBeGreaterThanOrEqual(1)
  expect(screen.getByRole('button', { name: 'Start run' })).toBeDisabled()
})

test('keyboard-only: tab to the submit button, Enter submits once with an idempotency key', async () => {
  const { calls } = base(() => ({ ok: true, errors: [], warnings: [], resolved: null }))
  renderWithProviders(<NewExperimentPage />, { route: '/experiments/new', path: '/experiments/new' })
  await screen.findByText('Configuration is valid')
  const submit = screen.getByRole('button', { name: 'Start run' })
  await waitFor(() => expect(submit).toBeEnabled())
  const user = userEvent.setup()
  await user.tab()
  let guard = 0
  while (document.activeElement !== submit && guard++ < 40) await user.tab()
  expect(document.activeElement).toBe(submit)
  await user.keyboard('{Enter}')
  await waitFor(() => expect(calls.filter((c) => c.path === '/api/v1/runs' && c.method === 'POST').length).toBe(1))
  const body = calls.find((c) => c.path === '/api/v1/runs' && c.method === 'POST')!.body as { idempotency_key: string; config: { task: string; dataset: { id: string } } }
  expect(body.idempotency_key.length).toBeGreaterThan(8)
  expect(body.config.task).toBe('train_pa')
  expect(body.config.dataset.id).toBe(datasetMock.data.dataset_id)
})


test('model parameters come from the registry: fields are generated per spec and edits reach the configuration', async () => {
  const { calls } = base(() => ({ ok: true, errors: [], warnings: [], resolved: null }))
  renderWithProviders(<NewExperimentPage />, { route: '/experiments/new', path: '/experiments/new' })
  await screen.findByText('Configuration is valid')
  await userEvent.click(screen.getByRole('button', { name: 'Advanced settings' }))
  expect(screen.queryByLabelText('Hidden size')).not.toBeInTheDocument()
  const layers = screen.getByLabelText('num_layers')
  expect(layers).toHaveAttribute('placeholder', '1')
  expect(screen.getByText('Number of stacked recurrent layers')).toBeInTheDocument()
  await userEvent.type(layers, '2')
  await waitFor(() => {
    const last = calls.filter((c) => c.path === '/api/v1/experiments/validate').at(-1)
    const body = last?.body as { config: { model: { parameters: Record<string, unknown> } } } | undefined
    expect(body?.config.model.parameters).toEqual({ hidden_size: 23, num_layers: 2 })
  })
})

test('an exported configuration can be imported and is submitted as is (resolution block dropped)', async () => {
  const { calls } = base(() => ({ ok: true, errors: [], warnings: [], resolved: null }))
  renderWithProviders(<NewExperimentPage />, { route: '/experiments/new', path: '/experiments/new' })
  await screen.findByText('Configuration is valid')
  const exported = { task: 'train_pa', recipe_id: 'pa-gru-smoke-v1', name: null, dataset: { id: 'dpa-200mhz', preprocessing_version: 'aligned-v1' }, model: { key: 'gru', parameters: { hidden_size: 8, num_layers: 1 } }, training: { epochs: 7 }, execution: { device: 'cpu' }, resolution: { config_sha256: 'abc', warnings: [] } }
  const input = screen.getByTestId('import-config')
  await userEvent.upload(input, new File([JSON.stringify(exported)], 'exp.json', { type: 'application/json' }))
  await screen.findByTestId('imported-banner')
  await waitFor(() => {
    const last = calls.filter((c) => c.path === '/api/v1/experiments/validate').at(-1)
    const body = last?.body as { config: Record<string, unknown> } | undefined
    expect(body?.config['resolution']).toBeUndefined()
    expect(body?.config['dataset']).toEqual({ id: 'dpa-200mhz', preprocessing_version: 'aligned-v1' })
    expect(body?.config['training']).toEqual({ epochs: 7 })
  })
  // discarding returns to the recipe form; importing again restores the imported configuration
  await userEvent.click(screen.getByRole('button', { name: 'Discard import' }))
  await waitFor(() => expect(screen.queryByTestId('imported-banner')).not.toBeInTheDocument())
  await userEvent.upload(input, new File([JSON.stringify(exported)], 'exp.json', { type: 'application/json' }))
  await screen.findByTestId('imported-banner')
  await screen.findByText('Configuration is valid')
  const submit = screen.getByRole('button', { name: 'Start run' })
  await waitFor(() => expect(submit).toBeEnabled())
  await userEvent.click(submit)
  await waitFor(() => expect(calls.some((c) => c.method === 'POST' && c.path === '/api/v1/runs')).toBe(true))
  const posted = calls.find((c) => c.method === 'POST' && c.path === '/api/v1/runs')?.body as { config: Record<string, unknown> }
  expect(posted.config['training']).toEqual({ epochs: 7 })
  expect(posted.config['resolution']).toBeUndefined()
})
