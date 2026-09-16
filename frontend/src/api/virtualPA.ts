import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api } from './client'
import { keys } from './hooks'
import type { Schemas } from './types'
import { getLanguage } from '@/i18n'

export type PAInput = Schemas['PAInputDataset']
export type VirtualPA = Schemas['VirtualPAModel']
export type PAParameter = Schemas['PAParameter']
export type PARequest = Schemas['VirtualPARequest']
export type PASimulation = Schemas['VirtualPASimulation']
export type PairedRequest = Schemas['PairedDatasetRequest']
export const paText = (value: { en: string; zh: string }) => getLanguage() === 'zh' ? value.zh : value.en
export const paDefaults = (model: VirtualPA) => Object.fromEntries(model.parameters.map(p => [p.key, p.default]))
export const parameterKey = (parameters: Record<string, number>) => JSON.stringify(Object.entries(parameters).sort(([a], [b]) => a.localeCompare(b)))
export const usePAInputs = () => useQuery({ queryKey: ['pa-inputs'], queryFn: ({ signal }: { signal: AbortSignal }) => api.get<PAInput[]>('/signal-generator/signals', signal) })
export const useVirtualPAs = () => useQuery({ queryKey: ['virtual-pa-models'], queryFn: ({ signal }: { signal: AbortSignal }) => api.get<VirtualPA[]>('/pa-library/models', signal), staleTime: Infinity })
export const usePASimulation = (id: string | null) => useQuery({ queryKey: ['pa-simulation', id],
  queryFn: ({ signal }: { signal: AbortSignal }) => api.get<PASimulation>('/pa-library/simulations/' + encodeURIComponent(id!), signal), enabled: !!id, retry: false })
export const useSimulatePA = () => useMutation({ mutationFn: (request: PARequest) => api.post<PASimulation>('/pa-library/simulations', request) })
export function usePairedDataset() {
  const qc = useQueryClient()
  return useMutation({ mutationFn: ({ id, request }: { id: string; request: PairedRequest }) =>
    api.post<Schemas['GeneratorDatasetResponse']>('/pa-library/simulations/' + encodeURIComponent(id) + '/dataset', request),
  onSuccess: () => void qc.invalidateQueries({ queryKey: keys.datasets }) })
}

export function useSimulateDataset() {
  const qc = useQueryClient()
  return useMutation({ mutationFn: (request: Schemas['VirtualPADatasetRequest']) =>
    api.post<Schemas['GeneratorDatasetResponse']>('/pa-library/datasets', request),
    onSuccess: () => void qc.invalidateQueries({ queryKey: keys.datasets }) })
}
