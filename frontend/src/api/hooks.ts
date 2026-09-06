/**
 * Server state lives in TanStack Query; URL state in the router; only
 * transient interaction state is kept in components. Nothing here caches a
 * run status beyond what the server returned.
 */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from './client'
import type {
  ArtifactManifest,
  Capabilities,
  DatasetManifest,
  EvaluationResult,
  ExperimentConfigInput,
  LogPage,
  MetricProfile,
  ModelInfo,
  RecipeInfo,
  ResolvedExperimentConfig,
  RunStatus,
  RunView,
  ValidationReport,
} from './types'
import { isTerminal } from './types'

export const keys = {
  capabilities: ['capabilities'] as const,
  models: ['models'] as const,
  recipes: ['recipes'] as const,
  datasets: ['datasets'] as const,
  dataset: (id: string) => ['datasets', id] as const,
  runs: (status?: RunStatus) => ['runs', status ?? 'all'] as const,
  run: (id: string) => ['run', id] as const,
  runConfig: (id: string) => ['run', id, 'config'] as const,
  runArtifacts: (id: string) => ['run', id, 'artifacts'] as const,
  result: (id: string, profile: string | null = null) => ['result', id, profile ?? 'primary'] as const,
  resultProfiles: (id: string) => ['result', id, 'profiles'] as const,
  metricProfiles: ['metric-profiles'] as const,
}

export const useCapabilities = () =>
  useQuery({ queryKey: keys.capabilities, queryFn: () => api.get<Capabilities>('/system/capabilities'), staleTime: 60_000 })
export const useModels = () => useQuery({ queryKey: keys.models, queryFn: () => api.get<ModelInfo[]>('/models'), staleTime: Infinity })
export const useRecipes = () => useQuery({ queryKey: keys.recipes, queryFn: () => api.get<RecipeInfo[]>('/recipes'), staleTime: Infinity })
export const useDatasets = () => useQuery({ queryKey: keys.datasets, queryFn: () => api.get<DatasetManifest[]>('/datasets') })
export const useDataset = (id: string) =>
  useQuery({ queryKey: keys.dataset(id), queryFn: () => api.get<DatasetManifest>(`/datasets/${encodeURIComponent(id)}`) })

export const useRuns = (status?: RunStatus) =>
  useQuery({
    queryKey: keys.runs(status),
    queryFn: () => api.get<RunView[]>(`/runs?limit=200${status ? `&status=${status}` : ''}`),
    refetchInterval: 5_000,
  })

/** Snapshot of one run; polls slowly while active as a fallback to the event stream. */
export const useRun = (id: string) =>
  useQuery({
    queryKey: keys.run(id),
    queryFn: () => api.get<RunView>(`/runs/${encodeURIComponent(id)}`),
    refetchInterval: (query) => (query.state.data && !isTerminal(query.state.data.status) ? 10_000 : false),
  })

export const useRunConfig = (id: string, enabled = true) =>
  useQuery({ queryKey: keys.runConfig(id), queryFn: () => api.get<ResolvedExperimentConfig>(`/runs/${encodeURIComponent(id)}/config`), enabled })
export const useRunArtifacts = (id: string, enabled = true) =>
  useQuery({ queryKey: keys.runArtifacts(id), queryFn: () => api.get<ArtifactManifest>(`/runs/${encodeURIComponent(id)}/artifacts`), enabled })
export const useResult = (id: string, enabled = true, profile: string | null = null) =>
  useQuery({
    queryKey: keys.result(id, profile),
    queryFn: () => api.get<EvaluationResult>(`/results/${encodeURIComponent(id)}${profile ? `?profile=${encodeURIComponent(profile)}` : ''}`),
    enabled,
    retry: false,
  })
export const useResultProfiles = (id: string) => useQuery({ queryKey: keys.resultProfiles(id), queryFn: () => api.get<string[]>(`/results/${encodeURIComponent(id)}/profiles`) })
export const useMetricProfiles = () => useQuery({ queryKey: keys.metricProfiles, queryFn: () => api.get<MetricProfile[]>('/metrics/profiles'), staleTime: Infinity })

export const fetchLogPage = (id: string, offset: number, limit = 500) =>
  api.get<LogPage>(`/runs/${encodeURIComponent(id)}/logs?offset=${offset}&limit=${limit}`)

export const validateConfig = (config: unknown) => api.post<ValidationReport>('/experiments/validate', { config })

export function useImportBuiltin() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (name: string) => api.post<DatasetManifest>('/datasets/import-builtin', { name }),
    onSuccess: () => void qc.invalidateQueries({ queryKey: keys.datasets }),
  })
}

export interface SubmitRunInput {
  config: ExperimentConfigInput
  name?: string
  idempotency_key: string
}

export function useSubmitRun() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (input: SubmitRunInput) => api.post<RunView>('/runs', input),
    onSuccess: (run) => {
      qc.setQueryData(keys.run(run.run_id), run)
      void qc.invalidateQueries({ queryKey: ['runs'] })
    },
  })
}

export function useCancelRun() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => api.post<RunView>(`/runs/${encodeURIComponent(id)}/cancel`),
    onSuccess: (run) => {
      qc.setQueryData(keys.run(run.run_id), run)
      void qc.invalidateQueries({ queryKey: ['runs'] })
    },
  })
}

export function useRetryRun() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (id: string) => api.post<RunView>(`/runs/${encodeURIComponent(id)}/retry`),
    onSuccess: (run) => {
      qc.setQueryData(keys.run(run.run_id), run)
      void qc.invalidateQueries({ queryKey: ['runs'] })
    },
  })
}
