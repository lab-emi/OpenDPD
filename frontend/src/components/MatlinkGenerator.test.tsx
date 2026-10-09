import { act, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import { useLocation } from 'react-router'
import presets from '@mocks/generator_presets.json'
import models from '@mocks/virtual_pa_models.json'
import { SignalGeneratorPage } from '@/pages/SignalGeneratorPage'
import { matlinkKey, type MatlinkSnapshot } from '@/api/matlink'
import { mockApi, renderWithProviders } from '@/test/utils'
vi.mock('@/components/PlotlyChart', () => ({ PlotlyChart: () => <div /> }))
beforeEach(() => vi.spyOn(window, 'scrollTo').mockImplementation(() => undefined))
function Probe() { const location = useLocation(); return <output data-testid="location">{location.pathname}{location.search}</output> }
const connection = { client_id: 'matlab-test', label: 'MATLAB', release: '2026a', connected: true, last_seen: '', variables: [], pending_count: 0, capabilities: ['dataset_export'] as ('dataset_export')[] }
const initial: MatlinkSnapshot = { available: true, protocol_version: 1, workspace: '/test', sessions: [connection], transfers: [] }

test('generator pairs signals and waits for MATLAB acknowledgment before opening the selected experiment', async () => {
  const snapshot = { ...initial }
  const { calls } = mockApi({
    'GET /api/v1/matlink': () => snapshot,
    'GET /api/v1/signal-generator/presets': () => presets.data,
    'GET /api/v1/pa-library/models': () => models.data,
    'POST /api/v1/signal-generator/batches': () => [{ signal_id: 'sg-' + 'a'.repeat(64), dataset_id: 'sds-' + 'b'.repeat(64), dataset_name: 'syn_pa_in_nr_test', name: presets.data[0]!.preset_id }],
    'POST /api/v1/pa-library/datasets': () => ({ dataset: { dataset_id: 'paired-test', simulation: { simulation_id: 'sim-test' } } }),
    'POST /api/v1/matlink/requests': (_, init) => {
      const body = JSON.parse(String(init.body))
      const transfer = { ...body, request_id: 'transfer-signals', status: 'queued', result: {}, error: null, created_at: '', updated_at: '' }
      snapshot.transfers = [transfer]
      return transfer
    },
  })
  const { client } = renderWithProviders(<><SignalGeneratorPage /><Probe /></>, { route: '/signal-generator?matlink=matlab-test' })
  await waitFor(() => expect(screen.getByRole('button', { name: 'Generate & prepare experiment' })).toBeEnabled(), { timeout: 5000 })
  expect(screen.getByRole('combobox', { name: 'Virtual PA model' })).toHaveTextContent(models.data.find(m => m.model_id === 'rapp-am-pm')!.name.en)
  await userEvent.click(screen.getByRole('button', { name: 'Generate & prepare experiment' }))
  await waitFor(() => expect(calls.some(c => c.path === '/api/v1/matlink/requests')).toBe(true), { timeout: 5000 })
  expect(calls.find(c => c.path === '/api/v1/pa-library/datasets')?.body).toMatchObject({ model_id: 'rapp-am-pm', input_signal_ids: ['sg-' + 'a'.repeat(64)] })
  expect(screen.getByTestId('location')).toHaveTextContent('/signal-generator?matlink=matlab-test&paired=paired-test')
  expect(calls.find(c => c.path === '/api/v1/matlink/requests')?.body).toMatchObject({ client_id: 'matlab-test', action: 'import_dataset', payload: { dataset_id: 'paired-test' } })
  act(() => client.setQueryData(matlinkKey, { ...snapshot, transfers: [{ ...snapshot.transfers[0], status: 'succeeded', result: { variable: 'opendpdSignals' } }] }))
  await waitFor(() => expect(screen.getByTestId('location')).toHaveTextContent('/experiments/new?dataset=paired-test&version=raw-v1&task=train_pa&matlink=matlab-test&matlab_variable=opendpdSignals'), { timeout: 5000 })
}, 15000)

test('an old toolbox cannot start the automatic generation workflow', async () => {
  mockApi({
    'GET /api/v1/matlink': () => ({ ...initial, sessions: [{ ...connection, capabilities: [] }] }),
    'GET /api/v1/signal-generator/presets': () => presets.data,
    'GET /api/v1/pa-library/models': () => models.data,
  })
  renderWithProviders(<SignalGeneratorPage />, { route: '/signal-generator?matlink=matlab-test' })
  await screen.findByText(/This connection needs toolbox 0.4.0/, undefined, { timeout: 5000 })
  expect(screen.getByRole('button', { name: 'Generate & prepare experiment' })).toBeDisabled()
}, 15000)
