import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from './client'
import { keys } from './hooks'
import type { Schemas } from './types'

export type GeneratorConfig = Required<Schemas['GeneratorConfig']>
export type GeneratorPreset = Schemas['GeneratorPreset']
export type GeneratedSignal = Schemas['GeneratedSignal']
export type GeneratorDatasetRequest = Required<Schemas['GeneratorDatasetRequest']>
export const useGeneratorPresets = () => useQuery({ queryKey: ['signal-generator-presets'],
  queryFn: () => api.get<GeneratorPreset[]>('/signal-generator/presets'), staleTime: Infinity })
export const useGenerateSignal = () => useMutation({ mutationFn: (config: GeneratorConfig) =>
  api.post<GeneratedSignal>('/signal-generator/signals', config) })
export function useGeneratorDataset() {
  const qc = useQueryClient()
  return useMutation({ mutationFn: ({ signalId, config }: { signalId: string; config: GeneratorDatasetRequest }) =>
    api.post<Schemas['GeneratorDatasetResponse']>(`/signal-generator/signals/${encodeURIComponent(signalId)}/dataset`, config),
  onSuccess: () => void qc.invalidateQueries({ queryKey: keys.datasets }) })
}
