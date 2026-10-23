import { useQuery } from '@tanstack/react-query'
import { api, ApiError } from './client'
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

/** The reply was JSON but not a test-model catalogue this frontend can use (a proxy page, a server of another version). */
export class CatalogShapeError extends Error {
  constructor(what: string) {
    super(`The test-model catalogue has an unexpected shape: ${what}.`)
    this.name = 'CatalogShapeError'
  }
}

const isRecord = (value: unknown): value is Record<string, unknown> => !!value && typeof value === 'object' && !Array.isArray(value)
const LIMITS = ['max_carriers', 'max_samples', 'max_cell_id', 'standard_subframes_fdd', 'standard_subframes_tdd', 'carrier_gap_step_mhz'] as const

/**
 * The minimal shape the panel reads without further checks: the version, a non-empty list of models (with the lists the panel maps over),
 * the limits, the N_RB table, the TDD patterns, the NB-IoT tables and the default recipe and configuration. Anything else is a failure.
 */
export function parseCatalog(raw: unknown): NrTmCatalog {
  if (!isRecord(raw)) throw new CatalogShapeError('not an object')
  if (typeof raw.version !== 'string' || !raw.version.startsWith('nr-test-model-catalog-')) throw new CatalogShapeError('no version')
  if (!Array.isArray(raw.models) || !raw.models.length) throw new CatalogShapeError('no models')
  for (const model of raw.models as unknown[]) {
    if (!isRecord(model) || typeof model.model !== 'string' || typeof model.title !== 'string' || !['used_for', 'modulations', 'users', 'fixed_parameters', 'scs_khz'].every(key => Array.isArray(model[key])))
      throw new CatalogShapeError('a model entry is incomplete')
  }
  if (!isRecord(raw.limits) || !LIMITS.every(key => typeof (raw.limits as Record<string, unknown>)[key] === 'number')) throw new CatalogShapeError('no limits')
  if (!isRecord(raw.resource_blocks) || !Array.isArray(raw.scs_khz) || !Array.isArray(raw.tdd_patterns)) throw new CatalogShapeError('no resource-block or TDD tables')
  const nb = raw.nb_iot
  if (!isRecord(nb) || !Array.isArray(nb.bandwidths_mhz) || !isRecord(nb.eligible_rbs) || !isRecord(nb.lower_rb) || !isRecord(nb.upper_rb)) throw new CatalogShapeError('no NB-IoT tables')
  if (!isRecord(raw.default_recipe) || !Array.isArray(raw.default_recipe.carriers) || !isRecord(raw.default_config) || typeof raw.default_config.preset_id !== 'string') throw new CatalogShapeError('no default setup')
  return raw as unknown as NrTmCatalog
}

/** The catalogue was not offered by this server: a public Studio that predates the endpoints (403) or an older server (404). Every other failure is worth a notice and a retry. */
export const catalogUnavailable = (error: unknown) => error instanceof ApiError && (error.status === 403 || error.status === 404)

/**
 * The catalogue is static for a server version, so it is fetched once and not refetched in the background (neither on a window refocus nor on
 * a reconnect). A failure is not retried automatically (the page offers a retry). A server that does not offer the catalogue (403, 404) is a
 * final answer rather than a failure: the query then holds `null`, so that the page keeps the generic preset matrix without a notice and
 * without any error state that a refetch could flicker through. That answer is asked for again on the next visit of the page.
 */
export const useNrTestModelCatalog = () => useQuery({ queryKey: ['nr-test-model-catalog'],
  queryFn: async ({ signal }: { signal: AbortSignal }) => {
    try { return parseCatalog(await api.get<unknown>(CATALOG, signal)) } catch (error) { if (catalogUnavailable(error)) return null; throw error }
  },
  staleTime: query => query.state.data ? Infinity : 0, retry: false, refetchOnWindowFocus: false, refetchOnReconnect: false })

/** `resolve` stores nothing on the server; callers debounce and pass the signal of the query that owns the request. */
export const resolveNrTestModel = (recipe: NrTmRecipe, signal?: AbortSignal) => api.post<NrTmResolution>(CATALOG + '/resolve', recipe, signal)

/** Query key of a resolution: the recipe is canonicalised so that equal recipes share one request and one cache entry. */
export const resolveKey = (key: string) => ['nr-test-model-resolve', key] as const
