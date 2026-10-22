import { useQuery } from '@tanstack/react-query'
import { api } from './client'
import type { Schemas } from './types'

export type NrTmCatalog = Schemas['NrTmCatalog']
export type NrTmModelInfo = Schemas['NrTmModelInfo']
export type NrTmResolution = Schemas['NrTmResolution']
export type NrTmSummary = Schemas['NrTmSummary']
export type NrTmCarrierSummary = Schemas['NrTmCarrierSummary']
export type NrTmUser = Schemas['NrTmUser']
export type NrTmModelId = Schemas['NrTestModelConfig']['model']
export type NrTmNbIot = Schemas['NrTmNbIot']
/** The wire recipes leave defaulted keys optional; the editor always works on complete objects. */
export type NrTmCarrier = Required<Schemas['NrTmCarrier']>
export type NrTmRecipe = Omit<Required<Schemas['NrTestModelConfig']>, 'carriers'> & { carriers: NrTmCarrier[] }

const CATALOG = '/signal-generator/nr-test-models'

/**
 * The catalogue is static for a server version. A 403 (public Studio before the API was redeployed),
 * 404 or any other failure is not retried: the generator then keeps the generic preset matrix.
 */
export const useNrTestModelCatalog = () => useQuery({ queryKey: ['nr-test-model-catalog'],
  queryFn: ({ signal }: { signal: AbortSignal }) => api.get<NrTmCatalog>(CATALOG, signal), staleTime: Infinity, retry: false })

/** `resolve` stores nothing on the server; callers debounce and pass the signal of the query that owns the request. */
export const resolveNrTestModel = (recipe: NrTmRecipe, signal?: AbortSignal) => api.post<NrTmResolution>(CATALOG + '/resolve', recipe, signal)

/** Query key of a resolution: the recipe is canonicalised so that equal recipes share one request and one cache entry. */
export const resolveKey = (key: string) => ['nr-test-model-resolve', key] as const
