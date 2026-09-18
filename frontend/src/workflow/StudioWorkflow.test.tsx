import { useState } from 'react'
import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import type { RunView } from '@/api/types'
import { mockApi, renderWithProviders } from '@/test/utils'
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
