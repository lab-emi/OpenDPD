import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import datasetMock from '@mocks/dataset_builtin.json'
import generalProfile from '@mocks/metric_profile_general.json'
import legacyProfile from '@mocks/metric_profile_legacy.json'
import ofdmProfile from '@mocks/metric_profile_ofdm_evm.json'
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

function base(validate: (config: Record<string, unknown>) => unknown, availableDatasets: unknown[] = [datasetMock.data], availableCaps = caps) {
  return mockApi({
    'GET /api/v1/recipes': () => [recipe],
    'GET /api/v1/datasets': () => availableDatasets,
    'GET /api/v1/models': () => [model],
    'GET /api/v1/system/capabilities': () => availableCaps,
    'GET /api/v1/runs': () => [],
    'POST /api/v1/experiments/validate': (_url, init) => validate((JSON.parse(String(init.body)) as { config: Record<string, unknown> }).config),
    'POST /api/v1/runs': () => ({ status: 201, body: runQueued.data }),
  })
}

async function importConfiguration(config: object) {
  await userEvent.click(screen.getByRole('button', { name: 'Advanced JSON' }))
  await userEvent.upload(screen.getByTestId('import-config'), new File([JSON.stringify(config)], 'exp.json', { type: 'application/json' }))
  await userEvent.click(screen.getByRole('button', { name: 'Use configuration' }))
}

async function continueStep() {
  const button = await screen.findByRole('button', { name: 'Continue' })
  await waitFor(() => expect(button).toBeEnabled())
  await userEvent.click(button)
}

test('prefers detected CUDA for a new experiment and keeps an explicit CPU choice', async () => {
  const availableCaps = { ...caps, devices: caps.devices.map((d) => ({ ...d, detected: true, count: 1 })) }
  const { calls } = base(() => ({ ok: true, errors: [], warnings: [], resolved: null }), [datasetMock.data], availableCaps)
  renderWithProviders(<NewExperimentPage />, { route: '/experiments/new', path: '/experiments/new' })
  await screen.findByText('Configuration is valid'); await continueStep()
  expect(screen.getByRole('combobox', { name: 'Device' })).toHaveTextContent('cuda')
  await userEvent.click(screen.getByRole('combobox', { name: 'Device' }))
  await userEvent.click(screen.getByRole('option', { name: 'cpu' }))
  await screen.findByText('Configuration is valid'); await continueStep()
  await userEvent.click(screen.getByRole('button', { name: 'Start run' }))
  await waitFor(() => expect(calls.find((c) => c.path === '/api/v1/runs' && c.method === 'POST')?.body).toMatchObject({ config: { execution: { device: 'cpu' } } }))
})

test('server validation errors are shown on the field and block submit', async () => {
  base((config) => {
    const epochs = (config['training'] as { epochs: number }).epochs
    return epochs > 0 ? { ok: true, errors: [], warnings: [], resolved: null } : { ok: false, errors: [{ field: 'training.epochs', message: 'must be at least 1', hint: 'use 3 for a smoke run' }], warnings: [], resolved: null }
  })
  renderWithProviders(<NewExperimentPage />, { route: '/experiments/new', path: '/experiments/new' })
  await screen.findByText('Quick trial: a few epochs to check the workflow. Results are not a benchmark.')
  await screen.findByText('Configuration is valid')
  await continueStep()
  const epochs = screen.getByLabelText('Epochs')
  await userEvent.clear(epochs)
  await userEvent.type(epochs, '0')
  await screen.findByText('Fix the following before submitting')
  expect(screen.getByLabelText('Epochs')).toHaveAttribute('aria-invalid', 'true')
  expect(screen.getAllByText(/must be at least 1/).length).toBeGreaterThanOrEqual(1)
  expect(screen.getByText('Start run')).toBeDisabled()
})

test('keyboard-only: tab to the submit button, Enter submits once with an idempotency key', async () => {
  const { calls } = base(() => ({ ok: true, errors: [], warnings: [], resolved: null }))
  renderWithProviders(<NewExperimentPage />, { route: '/experiments/new', path: '/experiments/new' })
  await screen.findByText('Configuration is valid')
  await continueStep()
  await continueStep()
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
  await continueStep()
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
  await importConfiguration(exported)
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
  await importConfiguration(exported)
  await screen.findByTestId('imported-banner')
  await screen.findByText('Configuration is valid')
  await continueStep()
  await continueStep()
  const submit = screen.getByRole('button', { name: 'Start run' })
  await waitFor(() => expect(submit).toBeEnabled())
  await userEvent.click(submit)
  await waitFor(() => expect(calls.some((c) => c.method === 'POST' && c.path === '/api/v1/runs')).toBe(true))
  const posted = calls.find((c) => c.method === 'POST' && c.path === '/api/v1/runs')?.body as { config: Record<string, unknown> }
  expect(posted.config['training']).toEqual({ epochs: 7 })
  expect(posted.config['resolution']).toBeUndefined()
})

test('a profile pending cross-validation is computed by the service but never offered by the form', async () => {
  mockApi({
    'GET /api/v1/recipes': () => [recipe],
    'GET /api/v1/datasets': () => [datasetMock.data],
    'GET /api/v1/models': () => [model],
    'GET /api/v1/system/capabilities': () => caps,
    'GET /api/v1/runs': () => [],
    'GET /api/v1/metrics/profiles': () => [legacyProfile.data, generalProfile.data, ofdmProfile.data],
    'POST /api/v1/experiments/validate': () => ({ ok: true, errors: [], warnings: [], resolved: null }),
  })
  renderWithProviders(<NewExperimentPage />, { route: '/experiments/new', path: '/experiments/new' })
  await screen.findByText('Configuration is valid')
  expect(ofdmProfile.data.validation).toBe('pending_cross_validation')
  await continueStep()
  await userEvent.click(screen.getByRole('button', { name: 'Advanced settings' }))
  await userEvent.click(screen.getByLabelText('Metric profile'))
  const options = screen.getAllByRole('option').map((o) => o.textContent)
  expect(options.some((t) => t?.startsWith('legacy-opendpd-v1'))).toBe(true)
  expect(options.some((t) => t?.startsWith('general-spectral-v1'))).toBe(true)
  expect(options.some((t) => t?.startsWith('ofdm-lte20-evm-v1'))).toBe(false)
})

test('fractional integer parameters reach server validation without being rounded', async () => {
  const { calls } = base((config) => {
    const layers = (config['model'] as { parameters: { num_layers: number } }).parameters.num_layers
    return Number.isInteger(layers)
      ? { ok: true, errors: [], warnings: [], resolved: null }
      : { ok: false, errors: [{ field: 'model.parameters.num_layers', message: 'num_layers must be an integer' }], warnings: [], resolved: null }
  })
  renderWithProviders(<NewExperimentPage />, { route: '/experiments/new', path: '/experiments/new' })
  await screen.findByText('Configuration is valid')
  await continueStep()
  await userEvent.type(screen.getByLabelText('num_layers'), '1.5')
  await screen.findByText('Fix the following before submitting')
  expect(screen.getByLabelText('num_layers')).toHaveAttribute('aria-invalid', 'true')
  expect(screen.getByText('Start run')).toBeDisabled()
  const last = calls.filter((c) => c.path === '/api/v1/experiments/validate').at(-1)
  expect(last?.body).toMatchObject({ config: { model: { parameters: { num_layers: 1.5 } } } })
})

test('a validation request failure explains the problem and can be retried without changing the configuration', async () => {
  let unavailable = true
  base(() => unavailable
    ? { status: 503, body: { error: { code: 'unavailable', message: 'Validation service unavailable' } } }
    : { ok: true, errors: [], warnings: [], resolved: null })
  renderWithProviders(<NewExperimentPage />, { route: '/experiments/new', path: '/experiments/new' })
  await screen.findByText(/Validation service unavailable/)
  expect(screen.getByText('Start run')).toBeDisabled()
  unavailable = false
  await userEvent.click(screen.getByRole('button', { name: 'Retry' }))
  await screen.findByText('Configuration is valid')
  await continueStep()
  await continueStep()
  expect(screen.getByRole('button', { name: 'Start run' })).toBeEnabled()
})

test('an unavailable source run never falls back to submitting the default recipe', async () => {
  const { calls } = base(() => ({ ok: true, errors: [], warnings: [], resolved: null }))
  renderWithProviders(<NewExperimentPage />, { route: '/experiments/new?from=missing-run', path: '/experiments/new' })
  await screen.findByText(/no mock for GET \/api\/v1\/runs\/missing-run\/config/)
  expect(screen.queryByRole('button', { name: 'Start run' })).not.toBeInTheDocument()
  expect(calls.filter((c) => c.path === '/api/v1/experiments/validate')).toHaveLength(0)
})

test('import mode exposes only the configuration that will run and the optional name', async () => {
  const { calls } = base(() => ({ ok: true, errors: [], warnings: [], resolved: null }))
  renderWithProviders(<NewExperimentPage />, { route: '/experiments/new', path: '/experiments/new' })
  await screen.findByText('Configuration is valid')
  const exported = { task: 'train_pa', dataset: { id: 'audit-data' }, model: { key: 'gru', parameters: { hidden_size: 8 } }, training: { epochs: 7 }, execution: { device: 'cpu' } }
  await importConfiguration(exported)
  await screen.findByTestId('imported-banner')
  expect(screen.queryByRole('combobox', { name: 'Recipe' })).not.toBeInTheDocument()
  expect(screen.queryByRole('combobox', { name: 'Dataset' })).not.toBeInTheDocument()
  expect(screen.queryByRole('button', { name: 'Advanced settings' })).not.toBeInTheDocument()
  await continueStep()
  await userEvent.type(screen.getByLabelText('Name (optional)'), 'Imported audit')
  await screen.findByText('Configuration is valid')
  await continueStep()
  await userEvent.click(screen.getByRole('button', { name: 'Start run' }))
  await waitFor(() => expect(calls.find((c) => c.path === '/api/v1/runs' && c.method === 'POST')?.body)
    .toMatchObject({ config: { ...exported, name: 'Imported audit' } }))
})


test('steps highlight the active stage and clear completion after a relevant edit', async () => {
  base(() => ({ ok: true, errors: [], warnings: [], resolved: null }))
  renderWithProviders(<NewExperimentPage />, { route: '/experiments/new', path: '/experiments/new' })
  await screen.findByText('Configuration is valid')
  expect(screen.queryByTestId('step-0-complete')).not.toBeInTheDocument()
  expect(screen.getByRole('tab', { name: /Data & model/ })).toHaveAttribute('aria-selected', 'true')
  await continueStep()
  expect(screen.getByTestId('step-0-complete')).toBeInTheDocument()
  expect(screen.getByRole('tab', { name: /Model & training/ })).toHaveAttribute('aria-selected', 'true')
  await continueStep()
  expect(screen.getByTestId('step-1-complete')).toBeInTheDocument()
  await userEvent.click(screen.getByRole('tab', { name: /Model & training/ }))
  await userEvent.type(screen.getByLabelText('Epochs'), '5')
  await waitFor(() => expect(screen.queryByTestId('step-1-complete')).not.toBeInTheDocument())
  await screen.findByText('Configuration is valid')
  expect(screen.getByRole('tab', { name: /Review & run/ })).toBeDisabled()
  expect(screen.getByTestId('step-0-complete')).toBeInTheDocument()
})

test('a linked preprocessing version reaches validation; choosing raw invalidates both completed steps and submits raw', async () => {
  const prepared = { ...datasetMock.data, versions: [{ version: 'aligned-v1', base_version: 'raw-v1', n_samples: 38000, split: datasetMock.data.split, files: [], record: {} }] }
  const { calls } = base(() => ({ ok: true, errors: [], warnings: [], resolved: null }), [prepared])
  renderWithProviders(<NewExperimentPage />, { route: '/experiments/new?dataset=dpa-200mhz&version=aligned-v1', path: '/experiments/new' })
  await screen.findByText('Configuration is valid')
  expect(calls.filter((c) => c.path === '/api/v1/experiments/validate').at(-1)?.body)
    .toMatchObject({ config: { dataset: { id: 'dpa-200mhz', preprocessing_version: 'aligned-v1' } } })
  await continueStep()
  await continueStep()
  await userEvent.click(screen.getByRole('tab', { name: /Data & model/ }))
  await userEvent.click(screen.getByRole('combobox', { name: 'Data version' }))
  await userEvent.click(screen.getByRole('option', { name: 'raw-v1' }))
  await screen.findByText('Configuration is valid')
  expect(screen.queryByTestId('step-0-complete')).not.toBeInTheDocument()
  expect(screen.queryByTestId('step-1-complete')).not.toBeInTheDocument()
  expect(screen.getByRole('tab', { name: /Review & run/ })).toBeDisabled()
  await continueStep()
  await continueStep()
  await userEvent.click(screen.getByRole('button', { name: 'Start run' }))
  await waitFor(() => expect(calls.find((c) => c.method === 'POST' && c.path === '/api/v1/runs')?.body)
    .toMatchObject({ config: { dataset: { id: 'dpa-200mhz', preprocessing_version: 'raw-v1' } } }))
})

test('JSON is hidden by default; invalid edits stay in the editor and a valid edit uses server validation', async () => {
  const { calls } = base(() => ({ ok: true, errors: [], warnings: [], resolved: null }))
  renderWithProviders(<NewExperimentPage />, { route: '/experiments/new' })
  await screen.findByText('Configuration is valid')
  expect(screen.queryByLabelText('Configuration JSON')).not.toBeInTheDocument()
  expect(screen.queryByRole('combobox', { name: 'Recipe' })).not.toBeInTheDocument()
  await userEvent.click(screen.getByRole('button', { name: 'Advanced JSON' }))
  await userEvent.clear(screen.getByLabelText('Configuration JSON'))
  await userEvent.type(screen.getByLabelText('Configuration JSON'), 'invalid')
  await userEvent.click(screen.getByRole('button', { name: 'Use configuration' }))
  expect(screen.getByRole('dialog')).toHaveTextContent('Not a JSON configuration file:')
  expect(calls.filter((call) => call.path === '/api/v1/runs' && call.method === 'POST')).toHaveLength(0)
  await userEvent.click(screen.getByRole('button', { name: 'Cancel' }))
  await waitFor(() => expect(screen.queryByLabelText('Configuration JSON')).not.toBeInTheDocument())
})

test.each(['evaluate_pa', 'run_dpd'] as const)('%s submits a selected checkpoint and test options without editable training parameters', async (task) => {
  const sourceId = task === 'evaluate_pa' ? 'pa-trained' : 'dpd-trained'
  const trainingTask = task === 'evaluate_pa' ? 'train_pa' : 'train_dpd'
  const { calls } = mockApi({
    'GET /api/v1/recipes': () => [recipe],
    'GET /api/v1/models': () => [model, { ...model, key: 'gru_stream', display_name: 'Streaming GRU', weights_from: 'gru', execution_semantics: 'streaming_stateful' }],
    'GET /api/v1/datasets': () => [datasetMock.data],
    'GET /api/v1/system/capabilities': () => caps,
    'GET /api/v1/metrics/profiles': () => [legacyProfile.data, generalProfile.data],
    'GET /api/v1/runs': () => [{ ...runQueued.data, run_id: sourceId, task: trainingTask, status: 'succeeded', name: 'My trained model', dataset_id: datasetMock.data.dataset_id, model_key: 'gru' }],
    [`GET /api/v1/runs/${sourceId}/config`]: () => ({ task: trainingTask, model: recipe.model, dataset: { id: datasetMock.data.dataset_id }, training: recipe.training, execution: { device: 'cpu' } }),
    'POST /api/v1/experiments/validate': () => ({ ok: true, errors: [], warnings: [], resolved: null }),
    'POST /api/v1/runs': () => ({ status: 201, body: { ...runQueued.data, task } }),
  })
  renderWithProviders(<NewExperimentPage />, { route: `/experiments/new?task=${task}` })
  await screen.findByRole('combobox', { name: /^Trained model/ })
  expect(screen.getByRole('button', { name: 'Continue' })).toBeDisabled()
  await userEvent.click(screen.getByRole('combobox', { name: /^Trained model/ }))
  await userEvent.click(screen.getByRole('option', { name: 'My trained model · gru' }))
  await continueStep()
  expect(screen.queryByLabelText('Epochs')).not.toBeInTheDocument()
  expect(screen.queryByLabelText('Learning rate')).not.toBeInTheDocument()
  await userEvent.click(screen.getByLabelText('Execution mode'))
  await userEvent.click(screen.getByRole('option', { name: 'Streaming GRU' }))
  await userEvent.type(screen.getByLabelText('Samples per streaming chunk'), '64')
  await screen.findByText('Configuration is valid')
  expect(screen.getByTestId('step-0-complete')).toBeInTheDocument()
  await userEvent.click(screen.getByRole('button', { name: 'Advanced settings' }))
  await userEvent.type(screen.getByLabelText('CPU threads (optional)'), '2')
  await continueStep()
  await userEvent.click(screen.getByRole('button', { name: 'Start run' }))
  await waitFor(() => expect(calls.filter((call) => call.path === '/api/v1/runs' && call.method === 'POST')).toHaveLength(1))
  const posted = calls.find((call) => call.path === '/api/v1/runs' && call.method === 'POST')?.body as { config: Record<string, unknown> }
  expect(posted.config).toMatchObject({ task, dataset: { id: datasetMock.data.dataset_id }, model: { key: 'gru_stream' }, execution: { device: 'cpu', num_threads: 2 }, evaluation: { chunk_samples: 64 }, [task === 'evaluate_pa' ? 'pa_reference' : 'dpd_reference']: { run_id: sourceId } })
  expect(posted.config.training).toBeUndefined()
})

test('DPD training requires a PA model and only offers DPD starting settings', async () => {
  const dpdRecipe = { ...recipe, task: 'train_dpd', recipe_id: 'dpd-gru-smoke-v1' }
  const { calls } = mockApi({
    'GET /api/v1/recipes': () => [recipe, dpdRecipe],
    'GET /api/v1/models': () => [model],
    'GET /api/v1/datasets': () => [datasetMock.data],
    'GET /api/v1/system/capabilities': () => caps,
    'GET /api/v1/runs': () => [{ ...runQueued.data, run_id: 'pa-ready', name: 'Ready PA', task: 'train_pa', status: 'succeeded', dataset_id: datasetMock.data.dataset_id }],
    'POST /api/v1/experiments/validate': () => ({ ok: true, errors: [], warnings: [], resolved: null }),
  })
  renderWithProviders(<NewExperimentPage />, { route: '/experiments/new?task=train_dpd' })
  await screen.findByText('Configuration is valid')
  expect(screen.getByRole('button', { name: 'Continue' })).toBeDisabled()
  await userEvent.click(screen.getByRole('combobox', { name: /^Trained PA model/ }))
  await userEvent.click(screen.getByRole('option', { name: 'Ready PA' }))
  await continueStep()
  expect(screen.getByLabelText('Epochs')).toBeVisible()
  expect(calls.filter((call) => call.path === '/api/v1/experiments/validate').at(-1)?.body).toMatchObject({ config: { task: 'train_dpd', recipe_id: 'dpd-gru-smoke-v1', pa_reference: { run_id: 'pa-ready' } } })
})
