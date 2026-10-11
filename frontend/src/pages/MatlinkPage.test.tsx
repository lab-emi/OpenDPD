import { act, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import { useLocation } from 'react-router'
import { matlinkKey, type MatlinkSnapshot, type MatlinkTransfer } from '@/api/matlink'
import runMock from '@mocks/run_running.json'
import reportMock from '@mocks/result_pa_modeling_mock.json'
import { mockApi, renderWithProviders } from '@/test/utils'
import { MatlinkPage } from './MatlinkPage'
vi.mock('@/components/PlotlyChart', () => ({ PlotlyChart: ({ title }: { title: string }) => <div>{title}</div> }))

const variables = [
  { name: 'paInput', class_name: 'single', size: [4096, 1], complex: true, eligible: true, n_samples: 4096 },
  { name: 'paOutput', class_name: 'single', size: [4096, 1], complex: true, eligible: true, n_samples: 4096 },
  { name: 'shortSignal', class_name: 'double', size: [100, 1], complex: true, eligible: true, n_samples: 100 },
]
const session = { client_id: 'matlab-session-a', label: 'MATLAB R2026a', release: '2026a', connected: true, last_seen: '2026-09-29T10:00:00Z', variables, pending_count: 0, capabilities: ['dataset_export', 'result_bundle'] as const }
const initial: MatlinkSnapshot = { available: true, protocol_version: 1, workspace: '/local/workspace', sessions: [{ ...session, capabilities: [...session.capabilities] }], transfers: [] }
const finished = { ...runMock.data, run_id: 'completed-id', name: 'Wideband GRU', status: 'succeeded', result_id: 'result-123', dataset_id: 'Bench PA', model_key: 'gru' }
const active = { ...runMock.data, run_id: 'running-id', name: 'Memory sweep', result_id: null, status: 'running' }
const transfer = (patch: Partial<MatlinkTransfer> = {}): MatlinkTransfer => ({ request_id: 'transfer-123', client_id: session.client_id, action: 'import_result', status: 'queued', payload: {}, result: {}, error: null, created_at: '2026-09-29T10:00:00Z', updated_at: '2026-09-29T10:00:00Z', ...patch })
function Probe() { const location = useLocation(); return <output data-testid="location">{location.pathname}{location.search}</output> }
function setup(data = initial, runs: unknown[] = [finished, active], post?: (payload: unknown) => unknown) {
  const result = mockApi({
    'GET /api/v1/matlink': () => data,
    'GET /api/v1/runs': () => runs,
    'GET /api/v1/runs/count': () => ({ count: runs.length }),
    'GET /api/v1/runs/completed-id': () => finished,
    'GET /api/v1/runs/running-id': () => active,
    'GET /api/v1/results/completed-id': () => ({ ...reportMock.data, run_id: finished.run_id, is_mock: false }),
    'POST /api/v1/matlink/requests': (_, init) => post?.(JSON.parse(String(init.body))) ?? transfer(),
  })
  return { ...result, ...renderWithProviders(<><MatlinkPage /><Probe /></>, { route: '/matlink' }) }
}
async function choose(label: string, name: string) {
  await userEvent.click(screen.getByRole('combobox', { name: label }))
  await userEvent.click(screen.getByRole('option', { name: new RegExp(`^${name} ·`) }))
}

test('shows two signal sources and hides stale sessions from the connection control', async () => {
  setup({ ...initial, sessions: [{ ...initial.sessions[0]!, client_id: 'old', label: 'Old test', connected: false }, initial.sessions[0]!] })
  expect(await screen.findByRole('heading', { name: 'MATLAB R2026a' })).toBeInTheDocument()
  expect(screen.queryByRole('combobox', { name: 'MATLAB session' })).not.toBeInTheDocument()
  expect(screen.queryByText('Old test')).not.toBeInTheDocument()
  expect(screen.queryByRole('button', { name: 'Create demo signals' })).not.toBeInTheDocument()
  expect(screen.getByRole('link', { name: 'Open Signal Generator' })).toHaveAttribute('href', '/signal-generator?matlink=matlab-session-a')
  expect(screen.getByRole('button', { name: /MATLAB workspace Select existing/ })).toBeInTheDocument()
})

test('disconnected users cannot save results or start generation', async () => {
  const { calls } = setup({ ...initial, sessions: [] })
  await screen.findByText('Connect MATLAB once')
  expect(screen.getByText('opendpd.studio()')).toBeInTheDocument()
  expect(await screen.findByRole('button', { name: 'Save to MATLAB' })).toBeDisabled()
  expect(screen.queryByLabelText(/Run ID/i)).not.toBeInTheDocument()
  expect(calls.filter(c => c.method === 'POST')).toHaveLength(0)
})

test('previews metrics and saves a bundle with an editable safe destination', async () => {
  const { calls, client } = setup()
  await screen.findByRole('button', { name: 'Save to MATLAB' })
  await screen.findByRole('region', { name: reportMock.data.metrics[0]!.name })
  const field = screen.getByLabelText('MATLAB variable name')
  await userEvent.clear(field); await userEvent.type(field, 'for')
  expect(screen.getByRole('button', { name: 'Save to MATLAB' })).toBeDisabled()
  await userEvent.clear(field); await userEvent.type(field, 'myResult')
  await userEvent.click(screen.getByRole('button', { name: 'Save to MATLAB' }))
  await waitFor(() => expect(calls.filter(c => c.method === 'POST')).toHaveLength(1))
  expect(calls.find(c => c.method === 'POST')!.body).toMatchObject({ action: 'import_result', payload: { run_id: finished.run_id, variable: 'myResult', bundle: true } })
  const done = transfer({ payload: { run_id: finished.run_id, variable: 'myResult', bundle: true }, status: 'succeeded', result: { variable: 'myResult_1' } })
  act(() => client.setQueryData(matlinkKey, { ...initial, transfers: [done] }))
  expect((await screen.findAllByText('Saved as myResult_1'))[0]).toBeInTheDocument()
  await userEvent.click(screen.getAllByRole('button', { name: 'Open in MATLAB' })[0]!)
  await waitFor(() => expect(calls.filter(c => c.method === 'POST')).toHaveLength(2))
  expect(calls.filter(c => c.method === 'POST')[1]!.body).toMatchObject({ action: 'open_variable', payload: { variable: 'myResult_1' } })
})

test('selecting a running experiment allows saving when ready', async () => {
  const { calls, client } = setup()
  await userEvent.click(await screen.findByRole('button', { name: /Memory sweep/ }))
  await userEvent.click(await screen.findByRole('button', { name: 'Send when ready' }))
  await waitFor(() => expect(calls.filter(c => c.method === 'POST')).toHaveLength(1))
  const payload = { run_id: active.run_id, variable: 'opendpd_Memory_sweep', bundle: true }
  expect(calls.find(c => c.method === 'POST')!.body).toMatchObject({ action: 'import_result', payload })
  act(() => client.setQueryData(matlinkKey, { ...initial, transfers: [transfer({ status: 'waiting', payload })] }))
  expect(await screen.findByRole('button', { name: 'Waiting for experiment' })).toBeDisabled()
})

test('workspace import validates the pair, preserves edits and opens the selected experiment', async () => {
  const imported = transfer({ action: 'import_iq', status: 'queued' })
  const { calls, client } = setup(initial, [], () => imported)
  await screen.findByText('Connected')
  await userEvent.click(screen.getByRole('button', { name: /MATLAB workspace Select existing/ }))
  await choose('PA input variable', 'paInput'); await choose('PA output variable', 'shortSignal')
  expect(screen.getByRole('button', { name: 'Use these variables in an experiment' })).toBeDisabled()
  await choose('PA output variable', 'paOutput')
  await userEvent.type(screen.getByLabelText('Dataset name (optional)'), 'My-PA-capture')
  await userEvent.type(screen.getByLabelText('Sample rate (MHz)'), '80')
  await userEvent.type(screen.getByLabelText('Occupied bandwidth (MHz)'), '20')
  // the segment length has no default: the import stays disabled until it is entered
  expect(screen.getByLabelText(/^PSD segment length \(nperseg\)/)).toHaveValue(null)
  expect(screen.getByRole('button', { name: 'Use these variables in an experiment' })).toBeDisabled()
  await userEvent.type(screen.getByLabelText(/^PSD segment length \(nperseg\)/), '2048')
  act(() => client.setQueryData(matlinkKey, { ...initial, sessions: [{ ...initial.sessions[0]!, last_seen: '2026-09-29T10:00:03Z' }] }))
  expect(screen.getByLabelText('Dataset name (optional)')).toHaveValue('My-PA-capture')
  expect(screen.getByLabelText(/^PSD segment length \(nperseg\)/)).toHaveValue(2048)
  await userEvent.click(screen.getByRole('button', { name: 'Use these variables in an experiment' }))
  await waitFor(() => expect(calls.filter(c => c.method === 'POST')).toHaveLength(1))
  expect(calls.find(c => c.method === 'POST')!.body).toMatchObject({ action: 'import_iq', payload: { input: 'paInput', output: 'paOutput', sample_rate_mhz: 80, bandwidth_mhz: 20, segment_samples: 2048 } })
  act(() => client.setQueryData(matlinkKey, { ...initial, transfers: [{ ...imported, status: 'succeeded', result: { dataset_id: 'my-capture' } }] }))
  await waitFor(() => expect(screen.getByTestId('location')).toHaveTextContent('/experiments/new?dataset=my-capture&version=raw-v1&task=train_pa&matlink=matlab-session-a'))
})

test('losing the pinned session never silently writes into another MATLAB session', async () => {
  const second = { ...initial.sessions[0]!, client_id: 'matlab-b', label: 'Other MATLAB' }
  const { client, calls } = setup({ ...initial, sessions: [initial.sessions[0]!, second] })
  await screen.findByText('Connected')
  act(() => client.setQueryData(matlinkKey, { ...initial, sessions: [{ ...initial.sessions[0]!, connected: false }, second] }))
  await screen.findAllByText(/MATLAB is not responding/)
  expect(await screen.findByRole('button', { name: 'Save to MATLAB' })).toBeDisabled()
  expect(calls.filter(c => c.method === 'POST')).toHaveLength(0)
})

test('lost responses retry the same transfer without creating a duplicate', async () => {
  let attempts = 0
  const { calls } = setup(initial, [finished], () => {
    if (++attempts === 1) throw new TypeError('Failed to fetch')
    return transfer()
  })
  await userEvent.click(await screen.findByRole('button', { name: 'Save to MATLAB' }))
  await waitFor(() => expect(calls.filter(c => c.method === 'POST')).toHaveLength(2), { timeout: 2500 })
  const posts = calls.filter(c => c.method === 'POST')
  expect(posts[0]!.body).toEqual(posts[1]!.body)
})
