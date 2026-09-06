import type { components } from './schema'

/** Aliases over the generated contract; nothing here is hand-written shape. */
export type Schemas = components['schemas']
export type RunView = Schemas['RunView']
export type RunStatus = Schemas['RunStatus']
export type RunEvent = Schemas['RunEvent']
export type RunEventType = Schemas['RunEventType']
export type RunError = Schemas['RunError']
export type TaskType = Schemas['TaskType']
export type EvidenceType = Schemas['EvidenceType']
export type MetricValue = Schemas['MetricValue']
export type MetricStatus = Schemas['MetricStatus']
export type EvaluationResult = Schemas['EvaluationResult']
export type DatasetManifest = Schemas['DatasetManifest']
export type ArtifactManifest = Schemas['ArtifactManifest']
export type Artifact = Schemas['Artifact']
export type ExperimentConfig = Schemas['ExperimentConfig']
export type ResolvedExperimentConfig = Schemas['ResolvedExperimentConfig']
export type TrainingConfig = Schemas['TrainingConfig']
export type Capabilities = Schemas['Capabilities']
export type DeviceInfo = Schemas['DeviceInfo']
export type ModelInfo = Schemas['ModelInfo']
export type RecipeInfo = Schemas['RecipeInfo']
export type SessionInfo = Schemas['SessionInfo']
export type Device = Schemas['ExecutionConfig']['device']
export type EventPage = Schemas['EventPage']
export type LogPage = Schemas['LogPage']

/** Shape of POST /experiments/validate (a plain dict on the server side). */
export interface ConfigIssue {
  field: string
  message: string
  hint?: string | null
}
export interface ValidationReport {
  ok: boolean
  errors: ConfigIssue[]
  warnings: ConfigIssue[]
  resolved: ResolvedExperimentConfig | null
}

/**
 * Dataset Doctor items are not part of the API until S07; the component is
 * designed now on the contract example (mocks/diagnostics_missing_metadata.json).
 */
export type Severity = 'info' | 'warning' | 'error'
export interface DiagnosticItem {
  code: string
  severity: Severity
  title: string
  message: string
  evidence: Record<string, number | string | boolean>
  suggestion?: string | null
  confidence?: number | null
  blocking: boolean
}
export interface DiagnosticReport {
  report_id: string
  dataset_id: string
  doctor_version: string
  generated_at: string
  items: DiagnosticItem[]
  evaluation_blocked: boolean
}

type DP<T> = T extends object ? DeepPartial<T> : T
type DeepPartial<T> = { [K in keyof T]?: DP<T[K]> }

/**
 * What the form sends: everything the server can default is optional so the
 * resolver (one source of defaults) fills it; responses stay strict.
 */
export type ExperimentConfigInput = DeepPartial<ExperimentConfig> & { task: TaskType; dataset: { id: string }; model: { key: string } }

export const TERMINAL: ReadonlySet<RunStatus> = new Set(['succeeded', 'failed', 'cancelled', 'interrupted'])
export const isTerminal = (s: RunStatus): boolean => TERMINAL.has(s)
