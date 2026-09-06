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

/** Version names a config may reference: raw-v1 plus every preprocessing version. */
export const versionNames = (d: DatasetManifest): string[] => Array.from(new Set(['raw-v1', ...(d.versions ?? []).map((v) => v.version)]))

export const datasetKeys = {
  roots: ['datasets', 'roots'] as const,
  files: (root: string, path: string) => ['datasets', 'roots', root, path] as const,
  diagnostics: (id: string) => ['datasets', id, 'diagnostics'] as const,
}

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
