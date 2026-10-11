import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, ApiError, WEB_MODE } from './client'

import type { components } from './schema'

type Schemas = components['schemas']
export type MatlabVariable = Schemas['MatlabVariable']
export type MatlabSession = Schemas['MatlinkSession']
export type MatlinkAction = Schemas['MatlinkRequest']['action']
export type MatlinkRequest = Omit<Schemas['MatlinkRequest'], 'payload'> & { payload: Record<string, string | number | boolean> }
export type MatlinkTransfer = Schemas['MatlinkTransfer']
export type MatlinkSnapshot = Schemas['MatlinkState']

export const matlinkKey = ['matlink'] as const

export function useMatlink(enabled = true) {
  return useQuery({
    queryKey: matlinkKey,
    queryFn: ({ signal }) => api.get<MatlinkSnapshot>('/matlink', signal),
    enabled: !WEB_MODE && enabled,
    refetchInterval: 3_000,
    retry: false,
  })
}

export function useMatlinkRequest() {
  const client = useQueryClient()
  return useMutation({
    mutationFn: (request: MatlinkRequest) => api.post<MatlinkTransfer>('/matlink/requests', request),
    // The same request and key are retained if the response was lost.
    retry: (attempt, error) => attempt < 1 && (!(error instanceof ApiError) || error.status >= 500),
    onSuccess: () => void client.invalidateQueries({ queryKey: matlinkKey }),
  })
}
