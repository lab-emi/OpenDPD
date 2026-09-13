import { fireEvent, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, vi } from 'vitest'
import { useLocation } from 'react-router'
import fixture from '@mocks/virtual_pa_models.json'
import { mockApi, renderWithProviders } from '@/test/utils'
import { WorkflowProgress } from '@/components/WorkflowProgress'
import { StudioWorkflowProvider } from '@/workflow/StudioWorkflow'
import { PALibraryPage } from './PALibraryPage'
import { DatasetsPage } from './DatasetsPage'

vi.mock('@/components/PlotlyChart', () => ({ seriesDash: () => 'solid', PlotlyChart: ({ title }: { title: string }) => <div>{title}</div> }))
const inputId = 'sg-' + 'a'.repeat(64), simulationId = 'vpa-' + 'b'.repeat(64)
const model = fixture.data.find(m => m.model_id === 'rapp-am-pm')!
const paired = { dataset_id: 'bench-pair', display_name: 'Bench measured pair', source: { kind: 'csv' }, n_samples: 32768, origin: 'measured', signal: { sample_rate_hz: 122.88e6 }, versions: [] }
const input = { signal_id: inputId, name: 'NR input', n_samples: 32768, sample_rate_hz: 122.88e6, bandwidth_hz: 20e6, kind: 'pa_input' }
const analysis = { n_samples: 32768, duration_ms: .2667, input_rms: .2, output_rms: .38, rms_gain_db: 5.57, input_papr_db: 10, output_papr_db: 8,
  time_us: [0, 1], input_envelope: [.1, .3], output_envelope: [.2, .5], am_input: [.1, .3], am_output: [.2, .5], am_pm_deg: [0, 5],
  frequency_mhz: [-1, 1], input_psd: [-60, -60], output_psd: [-54, -54], states: {}, notes: [] }
function Probe() { const loc = useLocation(); return <output data-testid="location">{loc.pathname}{loc.search}</output> }
beforeEach(() => localStorage.clear())

function setup() {
  let simulation = {}
  return mockApi({
    'GET /api/v1/system/capabilities': () => ({ workspace: 'virtual-pa-test', custom_dataset_imports: true }),
    'GET /api/v1/pa-library/models': () => fixture.data,
    'GET /api/v1/signal-generator/signals': () => [input],
    'GET /api/v1/datasets': () => [paired],
    'POST /api/v1/pa-library/simulations': (_url, init) => {
      simulation = { simulation_id: simulationId, config: JSON.parse(String(init.body)), model, analysis,
        output_csv_url: '/output.csv', paired_csv_url: '/paired.csv', metadata_url: '/metadata.json' }
      return simulation
    },
    ['GET /api/v1/pa-library/simulations/' + simulationId]: () => simulation,
    ['POST /api/v1/pa-library/simulations/' + simulationId + '/dataset']: (_url, init) => ({
      dataset: { ...paired, dataset_id: JSON.parse(String(init.body)).dataset_id, origin: 'synthetic' }, test_samples: 6452 }),
  })
}

test('formula and slider fields agree; only explicit simulation and pairing advance the workflow', async () => {
  const { calls } = setup()
  renderWithProviders(<StudioWorkflowProvider><WorkflowProgress /><PALibraryPage /><Probe /></StudioWorkflowProvider>, { route: '/pa-library?input=' + inputId })
  await screen.findByRole('heading', { name: model.name.en })
  await waitFor(() => expect(screen.getByTestId('workflow-input')).toHaveAttribute('data-complete', 'true'))
  expect(screen.getByTestId('workflow-output')).toHaveAttribute('data-complete', 'false')
  expect(calls.some(c => c.method === 'POST')).toBe(false)
  const gain = screen.getByRole('spinbutton', { name: 'Small-signal gain' })
  fireEvent.focus(gain)
  expect(screen.getByTestId('parameter-gain')).toHaveAttribute('data-active', 'true')
  for (const token of screen.getAllByTestId('equation-gain')) expect(token).toHaveAttribute('aria-pressed', 'true')
  fireEvent.change(gain, { target: { value: '2.5' } })
  expect(screen.getByRole('slider', { name: 'Small-signal gain' })).toHaveValue('2.5')
  await userEvent.click(screen.getByRole('button', { name: 'Simulate PA output' }))
  await screen.findByTestId('pa-output-preview')
  expect(calls.find(c => c.method === 'POST')?.body).toMatchObject({ input_signal_id: inputId, model_id: model.model_id, parameters: { gain: 2.5 } })
  await waitFor(() => expect(screen.getByTestId('workflow-output')).toHaveAttribute('data-complete', 'true'))
  expect(screen.getByTestId('workflow-paired')).toHaveAttribute('data-complete', 'false')
  expect(screen.getByText('Training 19,353 · validation 6,451 · testing 6,452 I/Q samples')).toBeVisible()
  // A temporarily blank numeric field must invalidate the old output immediately.
  const saturation = screen.getByRole('spinbutton', { name: 'Saturation envelope' })
  fireEvent.change(saturation, { target: { value: '' } })
  expect(screen.queryByTestId('pa-output-preview')).not.toBeInTheDocument()
  expect(screen.getByRole('button', { name: 'Simulate PA output' })).toBeDisabled()
  await waitFor(() => expect(screen.getByTestId('workflow-output')).toHaveAttribute('data-complete', 'false'))
  fireEvent.change(saturation, { target: { value: '.6' } })
  await userEvent.click(screen.getByRole('button', { name: 'Simulate PA output' }))
  await screen.findByTestId('pa-output-preview')
  fireEvent.change(screen.getByLabelText('Dataset ID'), { target: { value: 'explicit-virtual-pair' } })
  await userEvent.click(screen.getByRole('button', { name: 'Create paired dataset & train PA' }))
  await waitFor(() => expect(screen.getByTestId('location')).toHaveTextContent('/experiments/new?task=train_pa&dataset=explicit-virtual-pair'))
  expect(screen.getByTestId('workflow-paired')).toHaveAttribute('data-complete', 'true')
  expect(screen.getByTestId('workflow-pa')).toHaveAttribute('data-complete', 'false')
  expect(calls.filter(c => c.path.endsWith('/dataset'))).toHaveLength(1)
})

test('existing paired data skips dataset making and opens PA training directly', async () => {
  setup()
  renderWithProviders(<StudioWorkflowProvider><WorkflowProgress /><DatasetsPage /><Probe /></StudioWorkflowProvider>, { route: '/datasets?guide=start' })
  await userEvent.click(await screen.findByRole('button', { name: 'Use an existing dataset' }))
  await userEvent.click(await screen.findByRole('button', { name: /Bench measured pair/ }))
  await waitFor(() => expect(screen.getByTestId('location')).toHaveTextContent('/experiments/new?task=train_pa&dataset=bench-pair'))
  for (const key of ['input', 'virtual', 'output', 'paired']) expect(screen.getByTestId('workflow-' + key)).toHaveAttribute('data-complete', 'true')
  expect(screen.getByTestId('workflow-pa')).toHaveAttribute('data-complete', 'false')
  expect(screen.getByTestId('workflow-dpd')).toHaveAttribute('data-complete', 'false')
  await userEvent.click(await screen.findByRole('button', { name: 'Expand' }))
  expect(screen.getByRole('dialog')).toHaveTextContent('signal generation and Virtual PA simulation are bypassed')
})
