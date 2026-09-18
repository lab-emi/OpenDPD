import { useMutation, useQueries, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from './client'
import type { Schemas } from './types'

export type GeneratorConfig = Required<Schemas['GeneratorConfig']>
export type GeneratorPreset = Schemas['GeneratorPreset']
export type GeneratedSignal = Schemas['GeneratedSignal']
export const useGeneratorPresets = () => useQuery({ queryKey: ['signal-generator-presets'],
  queryFn: ({ signal }: { signal: AbortSignal }) => api.get<GeneratorPreset[]>('/signal-generator/presets', signal), staleTime: Infinity })
export const useGeneratedSignal = (id: string | null) => useQuery({ queryKey: ['generated-signal', id],
  queryFn: ({ signal }: { signal: AbortSignal }) => api.get<GeneratedSignal>('/signal-generator/signals/' + encodeURIComponent(id!), signal), enabled: !!id, retry: false })
export const useGeneratedSignals = (ids: string[]) => useQueries({ queries: ids.map(id => ({ queryKey: ['generated-signal', id],
  queryFn: ({ signal }: { signal: AbortSignal }) => api.get<GeneratedSignal>('/signal-generator/signals/' + encodeURIComponent(id), signal), retry: false })) })
export function useGenerateSignal() {
  const qc = useQueryClient()
  return useMutation({ mutationFn: (config: GeneratorConfig) => api.post<GeneratedSignal>('/signal-generator/signals', config),
    onSuccess: () => void qc.invalidateQueries({ queryKey: ['pa-inputs'] }) })
}

export function useGenerateBatch() {
  const qc = useQueryClient()
  return useMutation({ mutationFn: (request: Schemas['GeneratorBatchRequest']) => api.post<Schemas['PAInputDataset'][]>('/signal-generator/batches', request),
    onSuccess: () => { void qc.invalidateQueries({ queryKey: ['pa-inputs'] }); void qc.invalidateQueries({ queryKey: ['analyzer-datasets'] }) } })
}
