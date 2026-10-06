import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { api, getQuery } from './client'
import type { Schemas } from './types'

export type ArenaCatalog = Schemas['ArenaCatalog']
export type ArenaBoard = Schemas['ArenaBoard']
export type ArenaLeaderboard = Schemas['ArenaLeaderboard']
export type ArenaRow = Schemas['ArenaRow']
export type ArenaBudget = Schemas['ArenaBudgetResult']
export type ArenaRanking = Schemas['ArenaRanking']
export type ArenaSubmission = Schemas['ArenaSubmission']
export type ArenaSubmissionRequest = Schemas['ArenaSubmissionRequest']

const active = (status: string) => status === 'queued' || status === 'running'
export const useArena = () => useQuery({ queryKey: ['arena', 'catalog'], queryFn: getQuery<ArenaCatalog>('/arena') })
export const useArenaBoard = (id: string) => useQuery({
  queryKey: ['arena', 'boards', id], queryFn: getQuery<ArenaLeaderboard>(`/arena/boards/${encodeURIComponent(id)}`), enabled: !!id,
  // A board carries every case of every sweep; the small submission list below reports live progress.
  refetchInterval: query => query.state.data?.rows.some(row => active(row.status)) ? 15000 : false,
})
export const useArenaSubmissions = () => useQuery({
  queryKey: ['arena', 'submissions'], queryFn: getQuery<ArenaSubmission[]>('/arena/submissions'),
  refetchInterval: query => query.state.data?.some(row => active(row.status)) ? 2500 : false,
})
export function useSubmitArena() {
  const qc = useQueryClient()
  return useMutation({ mutationFn: (request: ArenaSubmissionRequest) => api.post<ArenaSubmission>('/arena/submissions', request),
    onSuccess: () => { void qc.invalidateQueries({ queryKey: ['arena'] }) },
  })
}
