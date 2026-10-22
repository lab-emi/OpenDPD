import { useQueries } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'
import { resolveKey, resolveNrTestModel, type NrTmCatalog, type NrTmRecipe, type NrTmResolution } from '@/api/nrTestModels'
import type { GeneratorConfig } from '@/api/signalGenerator'
import { normalizeRecipe, recipeKey, recipeOf } from '@/utils/nrTestModel'

/** Edits are sent to the server this long after the last one, never per keystroke. */
export const RESOLVE_DEBOUNCE_MS = 400

export type NrTmStatus = 'ready' | 'waiting' | 'resolving' | 'error' | 'duplicate'

function useDebounced<T>(value: T, ms: number) {
  const [debounced, setDebounced] = useState(value)
  useEffect(() => {
    const timer = window.setTimeout(() => setDebounced(value), ms)
    return () => window.clearTimeout(timer)
  }, [value, ms])
  return debounced
}

/** A resolution answers the recipe that was sent and the (normalised) recipe the server echoes in its configuration. */
const keysOf = (data: NrTmResolution, sent: string) => [sent, recipeKey(normalizeRecipe(data.recipe))]
const omit = <T,>(record: Record<string, T>, key: string) => { const { [key]: _removed, ...rest } = record; return rest }

/**
 * Keeps the recipes of the selected test models resolved by the server.
 *
 * `configs` holds the last *resolved* GeneratorConfig of every entry. An edit does not touch it: it becomes a draft recipe, which is
 * resolved (debounced, stale requests aborted) and then handed to `onResolved`, which swaps the entry for the server's configuration
 * (the preset_id follows the recipe). Entries with a draft are `blocked`: they must not be generated until the server accepted them.
 *
 * Equal recipes derive equal preset_ids and one dataset holds each id once, so a draft whose answer names an id that another entry already
 * has is not applied: it stays a draft (status `duplicate`, `duplicateOf` names the other entry) until it is edited or the other entry is removed.
 */
export function useNrTmEditor({ catalog, configs, activeId, onResolved, onCreate }: {
  catalog?: NrTmCatalog
  configs: Record<string, GeneratorConfig>
  activeId: string
  onResolved: (id: string, resolution: NrTmResolution) => void
  onCreate: (config: GeneratorConfig) => void
}) {
  const [drafts, setDrafts] = useState<Record<string, NrTmRecipe>>({})
  const [last, setLast] = useState<Record<string, { data: NrTmResolution; keys: string[] }>>({})
  const settled = useDebounced(drafts, RESOLVE_DEBOUNCE_MS)
  const counter = useRef(0)
  const applied = recipeOf(configs[activeId])
  const latest = useRef({ onResolved, drafts, configs })
  useEffect(() => { latest.current = { onResolved, drafts, configs } })
  const ids = Object.keys(configs).join('\n')

  // Every draft whose debounce has caught up, plus the active entry when it is clean (its summary is shown in the panel).
  const jobs: Array<{ id: string; recipe: NrTmRecipe; key: string }> = []
  if (catalog) {
    for (const [id, draft] of Object.entries(drafts)) {
      const ready = settled[id]
      if (id in configs && ready && recipeKey(ready) === recipeKey(draft)) jobs.push({ id, recipe: draft, key: recipeKey(draft) })
    }
    if (applied && !(activeId in drafts)) jobs.push({ id: activeId, recipe: applied, key: recipeKey(applied) })
  }
  const results = useQueries({ queries: jobs.map(job => ({ queryKey: resolveKey(job.key), staleTime: Infinity, retry: false,
    queryFn: ({ signal }: { signal: AbortSignal }) => resolveNrTestModel(job.recipe, signal) })) })
  const succeeded = jobs.map((job, i) => results[i]?.data ? job.id + '=' + job.key : '').join(';')

  useEffect(() => {
    for (const [i, job] of jobs.entries()) {
      const data = results[i]?.data
      if (!data) continue
      const draft = latest.current.drafts[job.id]
      const target = data.config.preset_id
      if (draft && recipeKey(draft) === job.key && target !== job.id && target in latest.current.configs) {
        setLast(old => old[job.id]?.data === data ? old : { ...old, [job.id]: { data, keys: keysOf(data, job.key) } })
      } else if (draft && recipeKey(draft) === job.key) {
        latest.current.onResolved(job.id, data)
        setDrafts(old => old[job.id] && recipeKey(old[job.id]!) === job.key ? omit(old, job.id) : old)
        setLast(old => ({ ...omit(old, job.id), [target]: { data, keys: keysOf(data, job.key) } }))
      } else if (!draft) setLast(old => old[job.id]?.data === data ? old : { ...old, [job.id]: { data, keys: keysOf(data, job.key) } })
    }
    // `succeeded` names exactly the finished jobs of this render and `ids` the entries that exist; jobs/results are read from the same render.
    // eslint-disable-next-line react-hooks/exhaustive-deps, react/exhaustive-effect-dependencies
  }, [succeeded, ids])

  const index = jobs.findIndex(job => job.id === activeId)
  const query = index >= 0 ? results[index] : undefined
  const draft = drafts[activeId]
  const recipe = draft ?? applied
  const caughtUp = !!draft && !!settled[activeId] && recipeKey(settled[activeId]!) === recipeKey(draft)
  const resolution = last[activeId]?.data
  const fresh = !!recipe && !!last[activeId]?.keys.includes(recipeKey(recipe))
  const target = draft && fresh ? resolution?.config.preset_id : undefined
  const duplicateOf = target && target !== activeId && target in configs ? target : undefined
  const status: NrTmStatus = draft ? (duplicateOf ? 'duplicate' : !caughtUp ? 'waiting' : query?.isError ? 'error' : 'resolving') : 'ready'

  return {
    /** The recipe the panel shows: the draft while there is one, else the active entry's own. */
    recipe, status, resolution, fresh,
    /** The id of the other entry that already has the recipe being edited (status `duplicate`). */
    duplicateOf,
    error: draft && caughtUp ? query?.error : undefined,
    /** The active entry's summary is loading (or being refreshed) from the server. */
    loading: status === 'waiting' || status === 'resolving' || !!query?.isFetching,
    /** True while any entry of the selection waits for the server: generating would use a configuration that is not derived from its recipe. */
    blocked: Object.keys(drafts).some(id => id in configs),
    edit: (next: NrTmRecipe) => setDrafts(old => applied && recipeKey(next) === recipeKey(applied) ? omit(old, activeId) : { ...old, [activeId]: next }),
    /** Appends a new entry that has no derived configuration yet; it is resolved like any other draft. */
    add: (next: NrTmRecipe) => {
      if (!catalog) return
      const id = `nr-tm-new-${++counter.current}`
      setDrafts(old => ({ ...old, [id]: next }))
      onCreate({ ...(catalog.default_config as GeneratorConfig), preset_id: id, nr_test_model: next })
    },
    discard: (id: string) => setDrafts(old => id in old ? omit(old, id) : old),
  }
}
