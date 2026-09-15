import { getLanguage } from '@/i18n'
/**
 * Server state lives in TanStack Query; URL state in the router; only
 * transient interaction state is kept in components. Nothing here caches a
 * run status beyond what the server returned.
 */
import { useMutation, useQuery, keepPreviousData, useQueryClient } from '@tanstack/react-query'
import { api, WEB_MODE } from './client'
import type {
  AdaptationReport,
  DeployExportInfo,
  AdaptationReportSummary,
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
  ComparisonReport,
  ExportInfo,
  HistoryPoint,
  ImportReport,
  RunLineage,
  RunStatus,
  RunView,
  ValidationReport,
  WorkspaceSettings,
  ReviewContext,
} from './types'
import { isTerminal } from './types'

export const keys = {
  capabilities: ['capabilities'] as const,
  settings: ['settings'] as const,
  models: ['models'] as const,
  recipes: ['recipes'] as const,
  datasets: ['datasets'] as const,
  dataset: (id: string) => ['datasets', id] as const,
  runs: (status?: RunStatus, page: RunPage = {}) => ['runs', status ?? 'all', page.q ?? '', page.limit ?? 200, page.offset ?? 0] as const,
  runCount: (status?: RunStatus, q?: string) => ['runs', 'count', status ?? 'all', q ?? ''] as const,
  run: (id: string) => ['run', id] as const,
  runConfig: (id: string) => ['run', id, 'config'] as const,
  runArtifacts: (id: string) => ['run', id, 'artifacts'] as const,
  runLineage: (id: string) => ['run', id, 'lineage'] as const,
  runHistory: (id: string) => ['run', id, 'history'] as const,
  artifactJson: (id: string, artifactId: string) => ['run', id, 'artifact', artifactId] as const,
  compare: (ids: string[], profile: string | null = null) => ['compare', ids.join(','), profile ?? 'primary'] as const,
  result: (id: string, profile: string | null = null) => ['result', id, profile ?? 'primary'] as const,
  resultProfiles: (id: string) => ['result', id, 'profiles'] as const,
  metricProfiles: ['metric-profiles'] as const,
  adaptationReports: ['adaptation', 'reports'] as const,
  adaptationReport: (planSha: string) => ['adaptation', 'reports', planSha] as const,
}

export const useCapabilities = () =>
  useQuery({ queryKey: keys.capabilities, queryFn: ({ signal }: { signal: AbortSignal }) => api.get<Capabilities>('/system/capabilities', signal), staleTime: 60_000 })
/** Closed until the server explicitly enables the retained custom-data workflow. */
export const useCustomDatasetImports = () => useCapabilities().data?.custom_dataset_imports === true
export const useSettings = () => useQuery({ queryKey: keys.settings, queryFn: ({ signal }: { signal: AbortSignal }) => api.get<WorkspaceSettings>('/settings', signal), staleTime: Infinity })

/** Full replacement of the workbench settings; the response is the stored state. */
export function useUpdateSettings() {
  const qc = useQueryClient()
  return useMutation({
    // Toolbar and Settings can both write this file; preserve selection order.
    scope: { id: 'workspace-settings' },
    mutationFn: (settings: WorkspaceSettings) => api.put<WorkspaceSettings>('/settings', settings),
    onSuccess: (saved) => qc.setQueryData(keys.settings, saved),
  })
}
export const useModels = () => useQuery({ queryKey: keys.models, queryFn: ({ signal }: { signal: AbortSignal }) => api.get<ModelInfo[]>('/models', signal), staleTime: Infinity })
export const useRecipes = () => useQuery({ queryKey: keys.recipes, queryFn: ({ signal }: { signal: AbortSignal }) => api.get<RecipeInfo[]>('/recipes', signal), staleTime: Infinity })
export const useDatasets = () => useQuery({ queryKey: keys.datasets, queryFn: ({ signal }: { signal: AbortSignal }) => api.get<DatasetManifest[]>('/datasets', signal) })
export const useDataset = (id: string) =>
  useQuery({ queryKey: keys.dataset(id), queryFn: ({ signal }: { signal: AbortSignal }) => api.get<DatasetManifest>(`/datasets/${encodeURIComponent(id)}`, signal) })

export interface RunPage {
  /** Case-insensitive substring over id, name, dataset and model (server side). */
  q?: string
  limit?: number
  offset?: number
}

function runParams(status?: RunStatus, page: RunPage = {}): string {
  const params = new URLSearchParams({ limit: String(page.limit ?? 200), offset: String(page.offset ?? 0) })
  if (status) params.set('status', status)
  if (page.q) params.set('q', page.q)
  return params.toString()
}

/** One page of runs (newest first); callers that only need "the recent ones" keep the default page of 200. */
export const useRuns = (status?: RunStatus, page: RunPage = {}) =>
  useQuery({
    queryKey: keys.runs(status, page),
    queryFn: ({ signal }: { signal: AbortSignal }) => api.get<RunView[]>(`/runs?${runParams(status, page)}`, signal),
    refetchInterval: query => WEB_MODE ? (query.state.data?.some(run => !isTerminal(run.status)) ? 10_000 : 30_000) : 5_000,
    placeholderData: keepPreviousData,
  })

export const useRunCount = (status?: RunStatus, q?: string) =>
  useQuery({
    queryKey: keys.runCount(status, q),
    queryFn: ({ signal }: { signal: AbortSignal }) => api.get<{ count: number }>(`/runs/count?${runParams(status, { q, limit: 1 }).replace(/&?limit=1&offset=0/, '')}`, signal),
    refetchInterval: WEB_MODE ? 30_000 : 5_000,
  })

/** Snapshot of one run; polls slowly while active as a fallback to the event stream. */
export const useRun = (id: string, enabled = true) =>
  useQuery({
    queryKey: keys.run(id),
    enabled,
    queryFn: ({ signal }) => api.get<RunView>(`/runs/${encodeURIComponent(id)}`, signal),
    refetchInterval: (query) => (query.state.data && !isTerminal(query.state.data.status) ? 10_000 : false),
  })

export const useRunConfig = (id: string, enabled = true) =>
  useQuery({ queryKey: keys.runConfig(id), queryFn: ({ signal }: { signal: AbortSignal }) => api.get<ResolvedExperimentConfig>(`/runs/${encodeURIComponent(id)}/config`, signal), enabled })
export const useRunArtifacts = (id: string, enabled = true) =>
  useQuery({ queryKey: keys.runArtifacts(id), queryFn: ({ signal }: { signal: AbortSignal }) => api.get<ArtifactManifest>(`/runs/${encodeURIComponent(id)}/artifacts`, signal), enabled })
export const useRunLineage = (id: string, enabled = true) =>
  useQuery({ queryKey: keys.runLineage(id), queryFn: ({ signal }: { signal: AbortSignal }) => api.get<RunLineage>(`/runs/${encodeURIComponent(id)}/lineage`, signal), enabled })
export const useRunHistory = (id: string, enabled = true) =>
  useQuery({ queryKey: keys.runHistory(id), queryFn: ({ signal }: { signal: AbortSignal }) => api.get<HistoryPoint[]>(`/runs/${encodeURIComponent(id)}/history`, signal), enabled, retry: false })
/** A registered JSON artifact (plots-v1 data, stored results); cached for the life of the page. */
export const artifactJsonQuery = <T,>(id: string, artifactId: string) => ({
  queryKey: keys.artifactJson(id, artifactId),
  queryFn: ({ signal }: { signal: AbortSignal }) => api.get<T>(`/artifacts/${encodeURIComponent(id)}/${encodeURIComponent(artifactId)}`, signal),
  retry: false,
  staleTime: Infinity,
})
export const useArtifactJson = <T,>(id: string, artifactId: string, enabled = true) => useQuery({ ...artifactJsonQuery<T>(id, artifactId), enabled })
export const useCompare = (ids: string[], profile: string | null = null, profiles: string[] = []) =>
  useQuery({
    queryKey: [...keys.compare(ids, profile), ...profiles],
    queryFn: ({ signal }: { signal: AbortSignal }) => api.get<ComparisonReport>(`/results/compare?${ids.map((i) => `runs=${encodeURIComponent(i)}`).join('&')}${profile ? `&profile=${encodeURIComponent(profile)}` : ''}${profiles.map(p => `&profiles=${encodeURIComponent(p)}`).join('')}`, signal),
    enabled: ids.length >= 2,
    placeholderData: keepPreviousData,
    retry: false,
  })
export const useResult = (id: string, enabled = true, profile: string | null = null) =>
  useQuery({
    queryKey: keys.result(id, profile),
    queryFn: ({ signal }: { signal: AbortSignal }) => api.get<EvaluationResult>(`/results/${encodeURIComponent(id)}${profile ? `?profile=${encodeURIComponent(profile)}` : ''}`, signal),
    enabled,
    placeholderData: (previous, query) => query?.queryKey[1] === id ? previous : undefined,
    retry: false,
  })
export const useResultProfiles = (id: string) => useQuery({ queryKey: keys.resultProfiles(id), queryFn: ({ signal }: { signal: AbortSignal }) => api.get<string[]>(`/results/${encodeURIComponent(id)}/profiles`, signal) })
export const useMetricProfiles = () => useQuery({ queryKey: keys.metricProfiles, queryFn: ({ signal }: { signal: AbortSignal }) => api.get<MetricProfile[]>('/metrics/profiles', signal), staleTime: Infinity })
export const reviewQuery = (runId: string, profileId?: string) => ({
  queryKey: ['review', runId, profileId ?? 'primary'],
  queryFn: ({ signal }: { signal: AbortSignal }) => api.get<ReviewContext>(`/results/${encodeURIComponent(runId)}/review${profileId ? `?profile=${encodeURIComponent(profileId)}` : ''}`, signal),
  staleTime: Infinity, retry: false,
})

export const fetchLogPage = (id: string, offset: number, limit = 500, tail = false, signal?: AbortSignal) =>
  api.get<LogPage>(`/runs/${encodeURIComponent(id)}/logs?offset=${offset}&limit=${limit}${tail ? '&tail=true' : ''}`, signal)

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

/** Write an experiment package into <workspace>/exports; the response carries the download URL and the manifest. */
export function useExportRun() {
  return useMutation({ mutationFn: (input: { run_id: string; kind: 'full' | 'share' }) => api.post<ExportInfo>(`/exports${getLanguage() === 'en' ? '' : `?language=${getLanguage()}`}`, input) })
}

/** Upload a package; every hash is verified server-side before anything is written. */
export function useImportPackage() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (file: File) => {
      const form = new FormData()
      form.append('file', file, file.name)
      return api.upload<ImportReport>('/imports', form)
    },
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: ['runs'] })
      void qc.invalidateQueries({ queryKey: keys.datasets })
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

/** conditions-v1 adaptation reports stored by `opendpd adaptation report` (read only; the GUI never runs a plan). */
export const useAdaptationReports = () =>
  useQuery({ queryKey: keys.adaptationReports, queryFn: ({ signal }: { signal: AbortSignal }) => api.get<AdaptationReportSummary[]>('/adaptation/reports', signal) })
export const useAdaptationReport = (planSha: string, enabled = true) =>
  useQuery({
    queryKey: keys.adaptationReport(planSha),
    queryFn: ({ signal }: { signal: AbortSignal }) => api.get<AdaptationReport>(`/adaptation/reports/${encodeURIComponent(planSha)}`, signal),
    enabled,
    retry: false,
    staleTime: Infinity,
  })

/** fixed-point-v1 deployment package for a finished GRU run (S19); the server verifies the C99 reference bit for bit. */
export const useDeployExport = () =>
  useMutation({ mutationFn: (body: { run_id: string }) => api.post<DeployExportInfo>('/deploy/exports', body) })
