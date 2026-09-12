/** Dataset import / doctor / preprocessing hooks (S07). */
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from './client'
import { keys } from './hooks'
import type { DatasetManifest, DatasetVersion, DiagnosticReport, Schemas } from './types'

export type ImportRootInfo = Schemas['ImportRootInfo']
export type FileEntryInfo = Schemas['FileEntryInfo']
export type SourceInfo = Schemas['SourceInfoOut']
export type ImportRequest = Schemas['ImportRequest']
export type ManifestUpdate = Schemas['ManifestUpdate']
export type PreprocessRequest = Schemas['PreprocessRequest']
export type PreprocessPreview = Schemas['PreprocessPreview']
export type UploadResult = Schemas['UploadResult']
export type DatasetAnalysis = Schemas['DatasetAnalysis']
export type InspectionMeasurement = Schemas['InspectionMeasurement']
export type InspectionReading = Schemas['InspectionReading']
export type BuiltinDatasetInfo = Schemas['BuiltinDatasetInfo']
export type DatasetImportDefaults = Required<Schemas['DatasetImportDefaults']>
export type CsvInspection = Omit<Required<Schemas['CsvInspection']>, 'split'> & { split: DatasetImportDefaults }
export type CsvOptions = Schemas['CsvOptions']
export type CsvPreviewRequest = Schemas['CsvPreviewRequest']
export type CsvCreateRequest = Schemas['CsvCreateRequest']

export const useBuiltinDatasets = () => useQuery({ queryKey: ['builtin-datasets'], queryFn: () => api.get<BuiltinDatasetInfo[]>('/datasets/builtin') })
export const useDatasetImportDefaults = () => useQuery({ queryKey: ['dataset-import-defaults'], queryFn: () => api.get<DatasetImportDefaults>('/datasets/import-defaults') })
export const previewCsv = (body: CsvPreviewRequest) => api.post<CsvInspection>('/datasets/csv/preview', body)

export function useCreateCsvDataset() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: CsvCreateRequest) => api.post<DatasetManifest>('/datasets/csv', body),
    onSuccess: () => void qc.invalidateQueries({ queryKey: keys.datasets }),
  })
}

/** Version names a config may reference: raw-v1 plus every preprocessing version. */
export const versionNames = (d: DatasetManifest): string[] => Array.from(new Set(['raw-v1', ...(d.versions ?? []).map((v) => v.version)]))

export const datasetKeys = {
  roots: ['datasets', 'roots'] as const,
  files: (root: string, path: string) => ['datasets', 'roots', root, path] as const,
  diagnostics: (id: string) => ['datasets', id, 'diagnostics'] as const,
  analysis: (id: string, version: string) => ['datasets', id, 'analysis', version] as const,
}

export const useDatasetAnalysis = (id: string, version: string, enabled = true) =>
  useQuery({
    queryKey: datasetKeys.analysis(id, version),
    queryFn: () => api.get<DatasetAnalysis>(`/datasets/${encodeURIComponent(id)}/analysis?version=${encodeURIComponent(version)}`),
    enabled, staleTime: 60_000,
  })

export const useImportRoots = () => useQuery({ queryKey: datasetKeys.roots, queryFn: () => api.get<ImportRootInfo[]>('/datasets/import-roots') })
export const useRootFiles = (root: string, path: string, enabled = true) =>
  useQuery({
    queryKey: datasetKeys.files(root, path),
    queryFn: () => api.get<FileEntryInfo[]>(`/datasets/import-roots/${encodeURIComponent(root)}/files?path=${encodeURIComponent(path)}`),
    enabled,
  })
export const inspectSource = (root_id: string, path: string) => api.post<SourceInfo>('/datasets/inspect', { root_id, path })
export const useDiagnostics = (id: string) =>
  useQuery({ queryKey: datasetKeys.diagnostics(id), queryFn: () => api.get<DiagnosticReport | null>(`/datasets/${encodeURIComponent(id)}/diagnostics`) })

export function useImportDataset() {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: ImportRequest) => api.post<DatasetManifest>('/datasets/import', body),
    onSuccess: () => void qc.invalidateQueries({ queryKey: keys.datasets }),
  })
}

export function useRunDoctor(id: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (version: string) => api.post<DiagnosticReport>(`/datasets/${encodeURIComponent(id)}/diagnostics?version=${encodeURIComponent(version)}`),
    onSuccess: (report) => qc.setQueryData(datasetKeys.diagnostics(id), report),
  })
}

export function useUpdateManifest(id: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: ManifestUpdate) => api.post<DatasetManifest>(`/datasets/${encodeURIComponent(id)}/manifest`, body),
    onSuccess: (m) => {
      qc.setQueryData(keys.dataset(id), m)
      void qc.invalidateQueries({ queryKey: keys.datasets })
    },
  })
}

export const previewPreprocess = (id: string, body: PreprocessRequest) => api.post<PreprocessPreview>(`/datasets/${encodeURIComponent(id)}/preprocess/preview`, body)

export function useCreateVersion(id: string) {
  const qc = useQueryClient()
  return useMutation({
    mutationFn: (body: PreprocessRequest) => api.post<DatasetVersion>(`/datasets/${encodeURIComponent(id)}/preprocess`, body),
    onSuccess: () => {
      void qc.invalidateQueries({ queryKey: keys.dataset(id) })
      void qc.invalidateQueries({ queryKey: keys.datasets })
    },
  })
}
