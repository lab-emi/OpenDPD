import { useState } from 'react'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import type { RunView } from '@/api/types'
import type { PAInputConfig } from '@/api/signalGenerator'
import { fixtureCatalog, fixtureResolve } from '@/test/nrTestModelFixtures'
import { mockApi, renderWithProviders } from '@/test/utils'
import { normalizeRecipe } from '@/utils/nrTestModel'
import { StudioWorkflowProvider, useRunWorkflow, useStudioWorkflow } from './StudioWorkflow'

test('saved run navigation restores lineage and inspecting its PA keeps the linked DPD', async () => {
  const pa = { run_id: 'saved-pa', dataset_id: 'capture-b', task: 'train_pa', status: 'succeeded' } as RunView
  const dpd = { run_id: 'saved-dpd', dataset_id: 'capture-b', task: 'train_dpd', status: 'succeeded' } as RunView
  mockApi({
    'GET /api/v1/system/capabilities': () => ({ workspace: 'lineage-ui-test' }),
    'GET /api/v1/runs/saved-pa': () => pa,
    'GET /api/v1/runs/saved-dpd': () => dpd,
    'GET /api/v1/runs/saved-pa/config': () => ({ dataset: { id: 'capture-b', preprocessing_version: 'aligned-v1' } }),
    'GET /api/v1/runs/saved-dpd/config': () => ({ dataset: { id: 'capture-b', preprocessing_version: 'aligned-v1' }, pa_reference: { run_id: 'saved-pa' } }),
  })
  function Inspector() {
    const [run, setRun] = useState(dpd)
    useRunWorkflow(run)
    const workflow = useStudioWorkflow()
    return <><output data-testid="lineage">{JSON.stringify(workflow.state)}</output><button onClick={() => setRun(pa)}>Inspect PA</button></>
  }
  try {
    renderWithProviders(<StudioWorkflowProvider><Inspector /></StudioWorkflowProvider>)
    await waitFor(() => expect(JSON.parse(screen.getByTestId('lineage').textContent!)).toMatchObject({ datasetId: 'capture-b', datasetVersion: 'aligned-v1', paRunId: 'saved-pa', dpdRunId: 'saved-dpd' }))
    await userEvent.click(screen.getByRole('button', { name: 'Inspect PA' }))
    expect(JSON.parse(screen.getByTestId('lineage').textContent!)).toMatchObject({ paRunId: 'saved-pa', dpdRunId: 'saved-dpd' })
  } finally { localStorage.removeItem('opendpd-workflow-v1:lineage-ui-test') }
})

test('the input of a generated test model is named by its recipe label, a matrix preset by its id', async () => {
  const tm = fixtureResolve({ ...normalizeRecipe(fixtureCatalog().default_recipe), model: 'NR-FR1-TM3.1a' }).config as PAInputConfig
  mockApi({ 'GET /api/v1/system/capabilities': () => ({ workspace: 'input-name-test' }) })
  function Selector() {
    const workflow = useStudioWorkflow()
    return <>
      <output data-testid="name">{workflow.state.inputName}</output>
      <button onClick={() => workflow.selectInputs(['sg-1', 'sg-2'], [tm, { ...tm, nr_test_model: null, preset_id: 'nr-20' }] as PAInputConfig[])}>Select</button>
    </>
  }
  try {
    renderWithProviders(<StudioWorkflowProvider><Selector /></StudioWorkflowProvider>)
    await userEvent.click(await screen.findByRole('button', { name: 'Select' }))
    expect(screen.getByTestId('name')).toHaveTextContent('TM3.1a · 20 MHz · 30 kHz · FDD, nr-20')
  } finally { localStorage.removeItem('opendpd-workflow-v1:input-name-test') }
})
