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
function Probe() { const loc = useLocation(); return <output data-testid="location">{loc.pathname}{loc.search}</output> }
beforeEach(() => localStorage.clear())

function setup() {
  return mockApi({
    'GET /api/v1/system/capabilities': () => ({ workspace: 'virtual-pa-test', custom_dataset_imports: true }),
    'GET /api/v1/pa-library/models': () => fixture.data,
    'GET /api/v1/signal-generator/signals': () => [input],
    'GET /api/v1/datasets': () => [paired],
    'POST /api/v1/pa-library/datasets': () => ({ dataset: { ...paired, dataset_id: 'auto-created', simulation: { simulation_id: simulationId } }, test_samples: 6452 }),
  })
}

test('formula controls stay synchronized and one simulation opens the completed dataset', async () => {
  const { calls } = setup()
  renderWithProviders(<StudioWorkflowProvider><WorkflowProgress /><PALibraryPage /><Probe /></StudioWorkflowProvider>, { route: '/pa-library?input=' + inputId })
  await screen.findByRole('heading', { name: model.name.en })
  await waitFor(() => expect(screen.getByTestId('workflow-input')).toHaveAttribute('data-complete', 'true'))
  expect(screen.queryByTestId('workflow-paired')).not.toBeInTheDocument()
  expect(screen.getByTestId('workflow-output')).toHaveAttribute('data-complete', 'false')
  expect(calls.some(c => c.method === 'POST')).toBe(false)
  const gain = screen.getByRole('spinbutton', { name: 'Small-signal gain' })
  fireEvent.focus(gain)
  expect(screen.getByTestId('parameter-gain')).toHaveAttribute('data-active', 'true')
  for (const token of screen.getAllByTestId('equation-gain')) expect(token).toHaveAttribute('aria-pressed', 'true')
  fireEvent.change(gain, { target: { value: '2.5' } })
  expect(screen.getByRole('slider', { name: 'Small-signal gain' })).toHaveValue('2.5')
  const saturation = screen.getByRole('spinbutton', { name: 'Saturation envelope' })
  fireEvent.change(saturation, { target: { value: '' } })
  expect(screen.getByRole('button', { name: 'Simulate PA output' })).toBeDisabled()
  fireEvent.change(saturation, { target: { value: '.6' } })
  await userEvent.click(screen.getByRole('button', { name: 'Simulate PA output' }))
  expect(calls.find(c => c.method === 'POST')?.body).toMatchObject({ input_signal_ids: [inputId], model_id: model.model_id, parameters: { gain: 2.5, saturation: .6 } })
  await waitFor(() => expect(screen.getByTestId('location')).toHaveTextContent('/datasets/auto-created'))
  expect(screen.getByTestId('workflow-output')).toHaveAttribute('data-complete', 'true')
  expect(screen.getByTestId('workflow-pa')).toHaveAttribute('data-complete', 'false')
  expect(screen.queryByText('Make a complete training dataset')).not.toBeInTheDocument()
  expect(calls.filter(c => c.method === 'POST')).toHaveLength(1)
})

test('the complete selected batch is simulated with a single request', async () => {
  const ids = [inputId, 'sg-' + 'c'.repeat(64)]
  localStorage.setItem('opendpd-workflow-v1:virtual-pa-test', JSON.stringify({ version: 1, origin: 'generated', inputId, inputIds: ids, parameters: {} }))
  const { calls } = setup()
  renderWithProviders(<StudioWorkflowProvider><PALibraryPage /><Probe /></StudioWorkflowProvider>, { route: '/pa-library?input=' + inputId })
  await userEvent.click(await screen.findByRole('button', { name: 'Simulate PA output' }))
  await waitFor(() => expect(screen.getByTestId('location')).toHaveTextContent('/datasets/auto-created'))
  expect(calls.find(c => c.method === 'POST')?.body).toMatchObject({ input_signal_ids: ids })
})

test('existing paired data skips dataset making and opens PA training directly', async () => {
  setup()
  renderWithProviders(<StudioWorkflowProvider><WorkflowProgress /><DatasetsPage /><Probe /></StudioWorkflowProvider>, { route: '/datasets?guide=start' })
  await userEvent.click(await screen.findByRole('button', { name: 'Use an existing dataset' }))
  await userEvent.click(await screen.findByRole('button', { name: /Bench measured pair/ }))
  await waitFor(() => expect(screen.getByTestId('location')).toHaveTextContent('/experiments/new?task=train_pa&dataset=bench-pair'))
  for (const key of ['input', 'virtual', 'output']) expect(screen.getByTestId('workflow-' + key)).toHaveAttribute('data-complete', 'true')
  expect(screen.getByTestId('workflow-pa')).toHaveAttribute('data-complete', 'false')
  expect(screen.getByTestId('workflow-dpd')).toHaveAttribute('data-complete', 'false')
  await userEvent.click(await screen.findByRole('button', { name: 'Expand' }))
  expect(screen.getByRole('dialog')).toHaveTextContent('signal generation and Virtual PA simulation are bypassed')
})
