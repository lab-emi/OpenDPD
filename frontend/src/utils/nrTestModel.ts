import type { NrTmCarrier, NrTmCatalog, NrTmModelId, NrTmModelInfo, NrTmRecipe } from '@/api/nrTestModels'
import type { GeneratorConfig } from '@/api/signalGenerator'
import { t } from '@/i18n'

type WireRecipe = NonNullable<GeneratorConfig['nr_test_model']>

/** Fields the user may still edit on a configuration derived from a test model (everything else follows the recipe). */
export const EDITABLE_FIELDS = ['rms', 'seed', 'carrier_frequency_hz', 'filter_enabled', 'burst_on_samples', 'burst_off_samples', 'burst_ramp_samples',
  'phase_offset_deg', 'phase_noise_rms_deg', 'frequency_offset_hz', 'iq_gain_db', 'iq_phase_deg', 'dc_i', 'dc_q', 'snr_db', 'clip_db'] as const satisfies readonly (keyof GeneratorConfig)[]

export const NB_IOT_DEFAULT = { position: 'lower', boost_db: 6 } as const

/** Recipes from the wire (or from a 2.4.1 workflow state) have optional keys; the editor needs complete objects. */
export function normalizeRecipe(raw: WireRecipe): NrTmRecipe {
  const carriers = raw.carriers?.length ? raw.carriers : [{ scs_khz: 30 as const, bandwidth_mhz: 20, gap_before_mhz: 0 }]
  return { ...raw, carriers: carriers.map(c => ({ scs_khz: c.scs_khz, bandwidth_mhz: c.bandwidth_mhz, n_cell_id: c.n_cell_id ?? null, gap_before_mhz: c.gap_before_mhz ?? 0 })),
    nb_iot: raw.nb_iot ?? null, oversampling: raw.oversampling ?? null, subframes: raw.subframes ?? null }
}

/** `undefined` and `null` both mean "not a test model" (configurations saved by 2.4.0 have no such key). */
export const recipeOf = (config: Pick<GeneratorConfig, 'nr_test_model'> | undefined): NrTmRecipe | null => config?.nr_test_model ? normalizeRecipe(config.nr_test_model) : null

const stable = (value: unknown): unknown => Array.isArray(value) ? value.map(stable)
  : value && typeof value === 'object' ? Object.fromEntries(Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([k, v]) => [k, stable(v)])) : value
/** Canonical text of a recipe: equal recipes give equal keys whatever the key order. */
export const recipeKey = (recipe: NrTmRecipe) => JSON.stringify(stable(recipe))

/** `NR-FR1-TM3.1a` -> `TM3.1a`; `NR-N-TM` is kept whole. */
export const shortModel = (model: string) => model.replace(/^NR-FR1-/, '')

const join = (values: number[]) => [...new Set(values)].join('+')
/** "TM3.1a · 100 MHz · 30 kHz · FDD"; several carriers read "2 × 100 MHz" or "20+100 MHz". */
export function recipeLabel(recipe: NrTmRecipe) {
  const widths = recipe.carriers.map(c => c.bandwidth_mhz)
  const bandwidth = recipe.carriers.length === 1 ? `${widths[0]} MHz` : new Set(widths).size === 1 ? `${widths.length} × ${widths[0]} MHz` : `${widths.join('+')} MHz`
  const parts = [shortModel(recipe.model), bandwidth, `${join(recipe.carriers.map(c => c.scs_khz)).replace(/\+/g, '/')} kHz`, recipe.duplex.toUpperCase()]
  if (recipe.layers === 2) parts.push(t('nrtm.chipLayers'))
  return parts.join(' · ')
}

/** `analysis.allocation[i].nb_iot` of an NR-N-TM signal: what the generator did to the host carrier (all bins are in units of `analysis.subcarrier_spacing_hz`). */
export interface NbIotCarrier { side: 'lower' | 'upper'; rb: number; first_subcarrier: number; subcarriers: number[]; re_per_frame: number; re_per_subframe: number[] }
export interface NbIotAllocation {
  boost_db: number; n_id: number; host_cell_id: number; nr_gain_db: number; nb_iot_gain_db: number
  frame_re: number; nb_iot_re: number; nr_re: number; carriers: NbIotCarrier[]
}
const numbers = (value: unknown): value is number[] => Array.isArray(value) && value.every(item => typeof item === 'number')
/** Allocation entries are open records on the wire; this accepts the NB-IoT block only when every field has the documented type. */
export function nbIotOf(entry: Record<string, unknown> | undefined): NbIotAllocation | null {
  const raw = entry?.nb_iot as Record<string, unknown> | null | undefined
  if (!raw || typeof raw !== 'object') return null
  const scalars = ['boost_db', 'n_id', 'host_cell_id', 'nr_gain_db', 'nb_iot_gain_db', 'frame_re', 'nb_iot_re', 'nr_re'] as const
  if (!scalars.every(key => typeof raw[key] === 'number') || !Array.isArray(raw.carriers)) return null
  const carriers = (raw.carriers as Array<Record<string, unknown>>).filter(c => (c.side === 'lower' || c.side === 'upper') && typeof c.rb === 'number' && typeof c.first_subcarrier === 'number'
    && typeof c.re_per_frame === 'number' && numbers(c.subcarriers) && numbers(c.re_per_subframe)) as unknown as NbIotCarrier[]
  return { ...(raw as unknown as NbIotAllocation), carriers }
}

/** How a signal is named in the Studio's own lists: the recipe label of a test model ("TM3.1a · 100 MHz · 30 kHz · FDD"), else its preset id. */
export function signalName(config: { preset_id: string; nr_test_model?: GeneratorConfig['nr_test_model'] }) {
  const recipe = recipeOf(config)
  return recipe ? recipeLabel(recipe) : config.preset_id
}

export const modelInfo = (catalog: NrTmCatalog, model: NrTmModelId | string) => catalog.models.find(m => m.model === model)

export const scsOptions = (catalog: NrTmCatalog, info?: NrTmModelInfo) => info?.scs_khz.length ? info.scs_khz : catalog.scs_khz

/** Channel bandwidths offered for a subcarrier spacing: the rows of the catalogue's N_RB table (UI filtering only; the server validates). */
export function bandwidthOptions(catalog: NrTmCatalog, info: NrTmModelInfo | undefined, scs: number) {
  const all = Object.keys(catalog.resource_blocks[String(scs)] ?? {}).map(Number).sort((a, b) => a - b)
  return info?.nb_iot ? all.filter(bandwidth => catalog.nb_iot.bandwidths_mhz.includes(bandwidth)) : all
}

const nearest = (options: number[], value: number) => options.reduce((best, option) => Math.abs(option - value) < Math.abs(best - value) ? option : best, options[0] ?? value)

/** Moves a carrier to the closest combination the model and the N_RB table offer. */
export function fitCarrier(catalog: NrTmCatalog, info: NrTmModelInfo | undefined, carrier: NrTmCarrier): NrTmCarrier {
  const spacings = scsOptions(catalog, info)
  const scs = spacings.includes(carrier.scs_khz) ? carrier.scs_khz : spacings[0] ?? carrier.scs_khz
  const widths = bandwidthOptions(catalog, info, scs)
  const bandwidth = !widths.length || widths.includes(carrier.bandwidth_mhz) ? carrier.bandwidth_mhz : nearest(widths, carrier.bandwidth_mhz)
  return { ...carrier, scs_khz: scs as NrTmCarrier['scs_khz'], bandwidth_mhz: bandwidth }
}

/** Switching the model keeps every setting that still applies and repairs the ones the new model does not allow. */
export function convertModel(catalog: NrTmCatalog, recipe: NrTmRecipe, model: NrTmModelId): NrTmRecipe {
  const info = modelInfo(catalog, model)
  const next: NrTmRecipe = { ...recipe, model }
  if (!info) return next
  if (info.fdd_only) next.duplex = 'fdd'
  if (!info.supports_layers) { next.layers = 1; next.output_port = 0 }
  next.nb_iot = info.nb_iot ? recipe.nb_iot ?? { ...NB_IOT_DEFAULT } : null
  next.carriers = recipe.carriers.map(carrier => fitCarrier(catalog, info, carrier))
  return next
}

/** A stepper that grows by cloning the last carrier (default cell ID, contiguous) and shrinks from the top. */
export function resizeCarriers(recipe: NrTmRecipe, count: number, max: number): NrTmRecipe {
  const n = Math.max(1, Math.min(max, Math.round(count)))
  if (n <= recipe.carriers.length) return { ...recipe, carriers: recipe.carriers.slice(0, n) }
  const last = recipe.carriers[recipe.carriers.length - 1]!
  return { ...recipe, carriers: [...recipe.carriers, ...Array.from({ length: n - recipe.carriers.length }, () => ({ ...last, n_cell_id: null, gap_before_mhz: 0 }))] }
}

/** The first model of the catalogue the selection does not use yet: the starting point of "Add another test model". */
export function nextModel(catalog: NrTmCatalog, used: readonly string[], current: string): NrTmModelId {
  const models = catalog.models.map(m => m.model)
  const start = models.indexOf(current as NrTmModelId)
  const ordered = [...models.slice(start + 1), ...models.slice(0, start + 1)]
  return ordered.find(model => !used.includes(model)) ?? ordered[0] ?? 'NR-FR1-TM1.1'
}

/** Keeps what the user edited on the previous configuration when the server re-derives the structural fields. */
export function mergeEditable(resolved: GeneratorConfig, previous: GeneratorConfig | undefined): GeneratorConfig {
  if (!previous) return resolved
  const kept: Partial<GeneratorConfig> = {}
  for (const key of EDITABLE_FIELDS) (kept as Record<string, unknown>)[key] = previous[key]
  return { ...resolved, ...kept }
}

/** Page-level helper: the entries of a record in order with one key replaced (so chips keep their position). */
export function rekey<T>(record: Record<string, T>, from: string, to: string, value: T): Record<string, T> {
  const entries = Object.entries(record)
  if (from !== to && to in record) return Object.fromEntries(entries.filter(([key]) => key !== from))
  return Object.fromEntries(entries.map(([key, old]) => key === from ? [to, value] : [key, old]))
}
