import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from './client'
import type { Schemas } from './types'

export type AnalyzerConfig = Required<Schemas['AnalyzerConfig']>
export type AnalyzerSource = Schemas['AnalyzerSource']
export type AnalyzerSourceInfo = Schemas['AnalyzerSourceInfo']
export type AnalyzerRequest = Schemas['AnalyzerRequest']
export type SignalAnalysis = Schemas['SignalAnalysis']
export type AnalyzerDataset = Schemas['AnalyzerDataset']
export const useAnalyzerDatasets = () => useQuery({ queryKey: ['analyzer-datasets'],
  queryFn: ({ signal }) => api.get<AnalyzerDataset[]>('/signal-analyzer/datasets', signal), staleTime: 0 })
export const useAnalyzerSources = () => useQuery({ queryKey: ['analyzer-sources'],
  queryFn: ({ signal }) => api.get<AnalyzerSourceInfo[]>('/signal-analyzer/sources', signal), staleTime: 0 })
export function useAnalyzeSignal() {
  const qc = useQueryClient()
  return useMutation({ mutationFn: (request: AnalyzerRequest) => api.post<SignalAnalysis>('/signal-analyzer/analyze', request),
    onSuccess: (result, request) => qc.setQueryData(['signal-analyzer-last'], { result, request }) })
}
export function useUploadSignal() {
  const qc = useQueryClient()
  return useMutation({ mutationFn: (file: File) => { const form = new FormData(); form.append('file', file); return api.upload<AnalyzerSourceInfo>('/signal-analyzer/upload', form) },
    onSuccess: () => { void qc.invalidateQueries({ queryKey: ['analyzer-sources'] }); void qc.invalidateQueries({ queryKey: ['analyzer-datasets'] }) } })
}
export function analyzerLink(kind: AnalyzerSource['kind'], id: string, role = 'input', version = 'raw-v1', dataset?: string) {
  return '/signal-analyzer?' + new URLSearchParams({ kind, source: id, role, version, ...(dataset ? { dataset } : {}) }).toString()
}
export const analyzerDefaults: AnalyzerConfig = { sample_rate_hz: 80e6, bandwidth_hz: 20e6, center_hz: 0,
  start_sample: 0, n_samples: 262144, fft_size: 4096, window: 'hann', overlap: .5, occupied_percent: 99,
  adjacent_offset_hz: null, remove_dc: false, frequency_shift_hz: 0, sample_format: 'auto', i_column: 0,
  q_column: 1, samples_per_symbol: 8, symbol_offset: 0, reference_gain_fit: false }
