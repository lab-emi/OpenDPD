import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from 'react'
import { useCapabilities, useRun, useRunConfig } from '@/api/hooks'
import { WEB_MODE } from '@/api/client'
import type { RunView } from '@/api/types'
import type { GeneratorConfig } from '@/api/signalGenerator'
import { parameterKey, type PASimulation } from '@/api/virtualPA'
import { LoadingState } from '@/components/StateBlock'

export interface WorkflowState {
  version: 1
  origin: 'generated' | 'existing' | null
  inputIds?: string[]
  inputConfigs?: GeneratorConfig[]
  inputDatasetId?: string
  inputDatasetName?: string
  inputId: string | null
  inputName: string
  modelId: string | null
  parameters: Record<string, number>
  simulationId: string | null
  datasetId: string | null
  datasetVersion: string
  paRunId: string | null
  dpdRunId: string | null
}
const empty = (): WorkflowState => ({ version: 1, origin: null, inputId: null, inputName: '', modelId: null,
  parameters: {}, simulationId: null, datasetId: null, datasetVersion: 'raw-v1', paRunId: null, dpdRunId: null })

interface WorkflowActions {
  selectInput: (id: string, name: string) => void
  selectInputs: (ids: string[], configs: GeneratorConfig[], dataset?: { id: string; name: string }) => void
  completeDataset: (id: string, simulationId: string) => void
  selectCapture: (id: string, version: string) => void
  configurePA: (id: string, parameters: Record<string, number>) => void
  simulated: (result: PASimulation) => void
  paired: (datasetId: string, simulationId: string) => void
  selectDataset: (id: string, version?: string, supplied?: boolean) => void
  selectPAReference: (id: string) => void
  invalidateOutput: () => void
  resetPA: () => void
  trackRun: (run: RunView, version: string, paRunId?: string) => void
  reset: () => void
}
interface WorkflowContextValue extends WorkflowActions {
  state: WorkflowState
  paDone: boolean
  dpdDone: boolean
}
const noop = () => undefined
const fallback: WorkflowContextValue = { state: empty(), paDone: false, dpdDone: false,
  selectInputs: noop, completeDataset: noop, selectCapture: noop, selectInput: noop, configurePA: noop, simulated: noop, paired: noop, selectDataset: noop,
  selectPAReference: noop, invalidateOutput: noop, resetPA: noop, trackRun: noop, reset: noop }
const Context = createContext<WorkflowContextValue>(fallback)
export const useStudioWorkflow = () => useContext(Context)

function read(key: string): WorkflowState {
  try {
    const value = JSON.parse((WEB_MODE ? sessionStorage : localStorage).getItem(key) ?? 'null') as WorkflowState | null
    if (!value || value.version !== 1 || !['generated', 'existing', null].includes(value.origin)) return empty()
    if (value.inputIds && (!Array.isArray(value.inputIds) || value.inputIds.length > 16 || value.inputIds.some(id => !/^sg-[a-f0-9]{64}$/.test(id)))) return empty()
    if (value.inputConfigs && (!Array.isArray(value.inputConfigs) || value.inputConfigs.length !== value.inputIds?.length)) return empty()
    if (value.inputId && !/^sg-[a-f0-9]{64}$/.test(value.inputId)) return empty()
    if (value.inputDatasetId && !/^sds-[a-f0-9]{64}$/.test(value.inputDatasetId)) return empty()
    if (value.inputDatasetName && (typeof value.inputDatasetName !== 'string' || value.inputDatasetName.length > 96)) return empty()
    if (value.simulationId && !/^vpa-[a-f0-9]{64}$/.test(value.simulationId)) return empty()
    if (value.datasetId && !/^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$/.test(value.datasetId)) return empty()
    if (!value.parameters || Object.values(value.parameters).some(v => typeof v !== 'number' || !Number.isFinite(v))) return empty()
    return { ...empty(), ...value }
  } catch { return empty() }
}

function ScopedWorkflow({ scope, children }: { scope: string; children: ReactNode }) {
  const key = 'opendpd-workflow-v1:' + scope
  const [state, set] = useState<WorkflowState>(() => read(key))
  useEffect(() => {
    try { (WEB_MODE ? sessionStorage : localStorage).setItem(key, JSON.stringify(state)) } catch { /* The in-memory workflow remains usable. */ }
  }, [key, state])
  const pa = useRun(state.paRunId ?? '', !!state.paRunId)
  const dpd = useRun(state.dpdRunId ?? '', !!state.dpdRunId)
  const paConfig = useRunConfig(state.paRunId ?? '', !!state.paRunId)
  const dpdConfig = useRunConfig(state.dpdRunId ?? '', !!state.dpdRunId)
  const paDone = pa.data?.status === 'succeeded' && pa.data.task === 'train_pa'
    && paConfig.data?.dataset.id === state.datasetId && (paConfig.data?.dataset.preprocessing_version ?? 'raw-v1') === state.datasetVersion
  const dpdDone = paDone && dpd.data?.status === 'succeeded' && dpd.data.task === 'train_dpd'
    && dpdConfig.data?.dataset.id === state.datasetId && (dpdConfig.data?.dataset.preprocessing_version ?? 'raw-v1') === state.datasetVersion
    && dpdConfig.data?.pa_reference?.run_id === state.paRunId
  const actions = useMemo<WorkflowActions>(() => ({
    selectInputs: (ids, configs, dataset) => set(old => ({ ...empty(), origin: 'generated', inputId: ids[0] ?? null,
      inputIds: ids, inputConfigs: configs, inputDatasetId: dataset?.id, inputDatasetName: dataset?.name,
      inputName: dataset?.name ?? configs.map(c => c.preset_id).join(', '), modelId: old.modelId, parameters: old.parameters })),
    selectCapture: (id, version) => set(old => old.datasetId === id && old.datasetVersion === version ? old : { ...old, datasetId: id, datasetVersion: version, paRunId: null, dpdRunId: null }),
    completeDataset: (id, simulationId) => set(old => ({ ...old, datasetId: id, simulationId, datasetVersion: 'raw-v1', paRunId: null, dpdRunId: null })),
    selectInput: (id, name) => set(old => old.origin === 'generated' && old.inputId === id ? old
      : { ...empty(), origin: 'generated', inputId: id, inputName: name, modelId: old.modelId, parameters: old.parameters }),
    configurePA: (id, parameters) => set(old => old.origin === 'generated' && old.modelId === id && parameterKey(old.parameters) === parameterKey(parameters) ? old
      : { ...old, origin: 'generated', modelId: id, parameters, simulationId: null, datasetId: null, paRunId: null, dpdRunId: null }),
    simulated: result => set(old => old.inputId === result.config.input_signal_id && old.modelId === result.config.model_id
      && parameterKey(old.parameters) === parameterKey(result.config.parameters ?? {})
      ? { ...old, simulationId: result.simulation_id } : old),
    paired: (datasetId, simulationId) => set(old => old.simulationId === simulationId
      ? { ...old, datasetId, datasetVersion: 'raw-v1', paRunId: null, dpdRunId: null } : old),
    selectDataset: (id, version = 'raw-v1', supplied = false) => set(old => old.datasetId === id && !supplied
      ? old.datasetVersion === version ? old : { ...old, datasetVersion: version, paRunId: null, dpdRunId: null }
      : { ...empty(), origin: 'existing', datasetId: id, datasetVersion: version }),
    selectPAReference: id => set(old => old.paRunId === id ? old : { ...old, paRunId: id, dpdRunId: null }),
    invalidateOutput: () => set(old => !old.simulationId && !old.datasetId && !old.paRunId && !old.dpdRunId ? old
      : { ...old, simulationId: null, datasetId: null, paRunId: null, dpdRunId: null }),
    resetPA: () => set(old => ({ ...empty(), origin: old.inputId ? 'generated' : null, inputId: old.inputId, inputIds: old.inputIds, inputConfigs: old.inputConfigs, inputName: old.inputName,
      inputDatasetId: old.inputDatasetId, inputDatasetName: old.inputDatasetName })),
    trackRun: (run, version, paRunId) => set(old => {
      if (!run.dataset_id) return old
      const base = old.datasetId === run.dataset_id && old.datasetVersion === version ? old
        : { ...empty(), origin: 'existing' as const, datasetId: run.dataset_id, datasetVersion: version }
      if (run.task === 'train_pa') return { ...base, paRunId: run.run_id, dpdRunId: null }
      if (run.task === 'train_dpd') return { ...base, dpdRunId: run.run_id, paRunId: paRunId ?? base.paRunId }
      return old
    }),
    reset: () => set(empty()),
  }), [])
  return <Context.Provider value={{ state, paDone, dpdDone, ...actions }}>{children}</Context.Provider>
}

export function StudioWorkflowProvider({ children }: { children: ReactNode }) {
  const caps = useCapabilities()
  if (caps.isPending) return <LoadingState />
  return <ScopedWorkflow key={caps.data?.workspace ?? 'unavailable'} scope={caps.data?.workspace ?? 'unavailable'}>{children}</ScopedWorkflow>
}
