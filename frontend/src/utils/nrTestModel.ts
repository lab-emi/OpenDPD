import type { NrTmCarrier, NrTmCatalog, NrTmModelId, NrTmModelInfo, NrTmRecipe } from '@/api/nrTestModels'
import type { GeneratorConfig } from '@/api/signalGenerator'
import { message as localize, t, type MessageKey } from '@/i18n'
import { describeApiError, plainMessage } from '@/utils/apiErrorText'

type WireRecipe = NonNullable<GeneratorConfig['nr_test_model']>

/** Fields the user may still edit on a configuration derived from a test model (everything else follows the recipe). */
export const EDITABLE_FIELDS = ['rms', 'seed', 'carrier_frequency_hz', 'filter_enabled', 'burst_on_samples', 'burst_off_samples', 'burst_ramp_samples',
  'phase_offset_deg', 'phase_noise_rms_deg', 'frequency_offset_hz', 'iq_gain_db', 'iq_phase_deg', 'dc_i', 'dc_q', 'snr_db', 'clip_db'] as const satisfies readonly (keyof GeneratorConfig)[]

export const NB_IOT_DEFAULT = { position: 'lower', boost_db: 6 } as const

/**
 * The PN23 scope only matters for TDD with the standard convention (the server warns otherwise), and it is part of the preset id: a recipe that cannot use it
 * keeps the default, so that two recipes that make the same signal are the same recipe. Every way into the editor (the controls, a model card, a loaded
 * configuration) goes through here.
 */
export function withPnScope<T extends Pick<NrTmRecipe, 'duplex' | 'data_convention' | 'pn_scope'>>(recipe: T): T {
  return recipe.pn_scope === 'dl_symbols' || (recipe.duplex === 'tdd' && recipe.data_convention === 'standard') ? recipe : { ...recipe, pn_scope: 'dl_symbols' }
}

/** Recipes from the wire (or from a 2.4.1 workflow state) have optional keys; the editor needs complete objects. */
export function normalizeRecipe(raw: WireRecipe): NrTmRecipe {
  const carriers = raw.carriers?.length ? raw.carriers : [{ scs_khz: 30 as const, bandwidth_mhz: 20, gap_before_mhz: 0 }]
  // The first carrier has no gap before it (the field is not offered for it, so a leftover value could never be cleared by the user).
  return withPnScope<NrTmRecipe>({ ...raw, carriers: carriers.map((c, i) => ({ scs_khz: c.scs_khz, bandwidth_mhz: c.bandwidth_mhz, n_cell_id: c.n_cell_id ?? null, gap_before_mhz: i === 0 ? 0 : c.gap_before_mhz ?? 0 })),
    nb_iot: raw.nb_iot ?? null, oversampling: raw.oversampling ?? null, subframes: raw.subframes ?? null })
}

/** The selection holds at most this many signals (GeneratorBatchRequest). */
export const MAX_SELECTED = 16

/** The longest capture the server accepts, in subframes (`subframes` of the recipe). */
export const MAX_SUBFRAMES = 20

/** The NB-IoT power boost X the generator accepts when the catalogue does not say (opendpd/schemas/nr_test_model.py), in dB. */
export const NB_IOT_BOOST = { min: 3, max: 12 } as const

/** `undefined` and `null` both mean "not a test model" (configurations saved by 2.4.0 have no such key). */
export const recipeOf = (config: Pick<GeneratorConfig, 'nr_test_model'> | undefined): NrTmRecipe | null => config?.nr_test_model ? normalizeRecipe(config.nr_test_model) : null

const stable = (value: unknown): unknown => Array.isArray(value) ? value.map(stable)
  : value && typeof value === 'object' ? Object.fromEntries(Object.entries(value).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0).map(([k, v]) => [k, stable(v)])) : value
/** Canonical text of a recipe: equal recipes give equal keys whatever the key order. */
export const recipeKey = (recipe: NrTmRecipe) => JSON.stringify(stable(recipe))

/** `NR-FR1-TM3.1a` -> `TM3.1a`; `NR-N-TM` is kept whole. */
export const shortModel = (model: string) => model.replace(/^NR-FR1-/, '')

const join = (values: number[]) => [...new Set(values)].sort((a, b) => a - b).join('+')
/** "TM3.1a · 100 MHz · 30 kHz · FDD"; several carriers read "2 × 100 MHz" or "20+100 MHz". */
export function recipeLabel(recipe: NrTmRecipe) {
  const widths = recipe.carriers.map(c => c.bandwidth_mhz)
  const bandwidth = recipe.carriers.length === 1 ? `${widths[0]} MHz` : new Set(widths).size === 1 ? `${widths.length} × ${widths[0]} MHz` : `${widths.join('+')} MHz`
  const parts = [shortModel(recipe.model), bandwidth, `${join(recipe.carriers.map(c => c.scs_khz)).replace(/\+/g, '/')} kHz`, recipe.duplex.toUpperCase()]
  if (recipe.layers === 2) parts.push(t('nrtm.chipLayers'))
  return parts.join(' · ')
}

const EDGE_TAG = { lower: 'nrtm.nbiotEdge.lower', upper: 'nrtm.nbiotEdge.upper', both: 'nrtm.nbiotEdge.both' } as const
/** What sets a recipe apart from the default one at a glance: `[attribute, tag]` for every setting the base label does not show (tag null = default). */
function differences(recipe: NrTmRecipe): Array<[string, string | null]> {
  const ids = recipe.carriers.map(c => c.n_cell_id), gaps = recipe.carriers.map(c => c.gap_before_mhz)
  return [
    ['convention', recipe.data_convention === 'standard' ? null : t(`nrtm.convention.${recipe.data_convention}`)],
    ['pn_scope', recipe.pn_scope === 'dl_symbols' ? null : t('nrtm.tag.pnAll')],
    ['cell', ids.every(id => id === null) ? null : t('nrtm.tag.cell', { ids: ids.map(id => id ?? '–').join('/') })],
    ['gap', gaps.every(gap => !gap) ? null : t('nrtm.tag.gap', { gaps: gaps.slice(1).join('/') })],
    ['subframes', recipe.subframes === null ? null : t('nrtm.tag.subframes', { count: recipe.subframes })],
    ['oversampling', recipe.oversampling === null ? null : `${t('nrtm.oversampling')} ${recipe.oversampling}×`],
    ['port', recipe.layers === 2 ? t('nrtm.port', { port: 1000 + recipe.output_port }) : null],
    ['nb_iot', recipe.nb_iot ? t('nrtm.tag.nbiot', { position: t(EDGE_TAG[recipe.nb_iot.position]), boost: recipe.nb_iot.boost_db }) : null],
  ]
}

/**
 * Chip, preview and warning names of the test-model entries of one selection. The base label (`recipeLabel`) is kept as it is; only when
 * another entry would read the same, the settings in which the entries differ are appended ("… · MATLAB 5G Toolbox", "… · cell 5"), and
 * an entry that is still not told apart gets an ordinal. Returns the name of every entry by id.
 */
export function uniqueRecipeLabels(entries: Array<{ id: string; recipe: NrTmRecipe }>): Record<string, string> {
  const groups = new Map<string, Array<{ id: string; recipe: NrTmRecipe; label: string }>>()
  for (const entry of entries) {
    const base = recipeLabel(entry.recipe)
    groups.set(base, [...(groups.get(base) ?? []), { ...entry, label: base }])
  }
  const named: Record<string, string> = {}
  for (const members of groups.values()) {
    if (members.length === 1) { named[members[0]!.id] = members[0]!.label; continue }
    const all = members.map(m => differences(m.recipe))
    const varying = (all[0] ?? []).map((_, k) => new Set(all.map(d => d[k]![1])).size > 1)
    const labels = members.map((m, i) => [m.label, ...all[i]!.flatMap(([, tag], k) => varying[k] && tag ? [tag] : [])].join(' · '))
    const seen = new Map<string, number>()
    members.forEach((m, i) => {
      const same = labels.filter(label => label === labels[i]).length
      const ordinal = (seen.get(labels[i]!) ?? 0) + 1
      seen.set(labels[i]!, ordinal)
      named[m.id] = same > 1 ? `${labels[i]} · #${ordinal}` : labels[i]!
    })
  }
  return named
}

const FIELD_LABEL: Record<string, MessageKey> = {
  model: 'nrtm.model', duplex: 'nrtm.duplex', layers: 'nrtm.layers', output_port: 'nrtm.outputPort', data_convention: 'nrtm.convention', pn_scope: 'nrtm.pnScope',
  oversampling: 'nrtm.oversampling', subframes: 'nrtm.length', carriers: 'nrtm.carriers', nb_iot: 'nrtm.nbiot',
  scs_khz: 'nrtm.scs', bandwidth_mhz: 'nrtm.bandwidth', n_cell_id: 'nrtm.cellId', gap_before_mhz: 'nrtm.gap', position: 'nrtm.nbiotPosition', boost_db: 'nrtm.nbiotBoost',
}
/** The label a form field carries, without the unit in brackets ("Channel bandwidth (MHz)" -> "Channel bandwidth"). */
const known = (name: string) => Object.hasOwn(FIELD_LABEL, name)
const labelOf = (name: string) => t(FIELD_LABEL[name]!).replace(/\s*\((?:kHz|MHz|dB)\)$/, '')

/**
 * A server field path in the words of the form: `carriers.0.n_cell_id` -> "Carrier 1 · Cell ID", `nb_iot.boost_db` -> "NB-IoT power boost X",
 * `subframes` -> "Capture length". The request wrappers (`configs.2.`, `nr_test_model.`) are dropped. A path with a part the form has no field for
 * is returned unchanged, so that no information is lost.
 */
export function nrTmFieldLabel(path: string) {
  const parts = path.split('.').filter(Boolean)
  if (parts[0] === 'configs' && /^\d+$/.test(parts[1] ?? '')) parts.splice(0, 2)
  if (parts[0] === 'nr_test_model') parts.shift()
  if (!parts.length) return ''
  const [head, ...rest] = parts
  if (head === 'carriers' && /^\d+$/.test(rest[0] ?? '')) {
    const field = rest[1]
    if (rest.length > 2 || (field !== undefined && !known(field))) return path
    return t('nrtm.carrierN', { index: Number(rest[0]) + 1 }) + (field ? ` · ${labelOf(field)}` : '')
  }
  if (head === 'nb_iot' && rest.length === 1) return known(rest[0]!) ? labelOf(rest[0]!) : path
  return rest.length || !known(head!) ? path : labelOf(head!)
}

/** One refusal detail as a line: the field in the words of the form, the server's text (the 16-item list limit is explained as the Studio limit it is). */
function nrTmLine(field: string, text: string) {
  const plain = plainMessage(text)
  const label = nrTmFieldLabel(field)
  if (/(?:^|\.)carriers$/.test(field) && /^List should have at most \d+ items/.test(plain)) return t('nrtm.carrierLimit', { max: /at most (\d+)/.exec(plain)![1]! })
  return `${label ? label + ': ' : ''}${localize(plain)}`
}

/** The server's refusal of a recipe (or of a configuration that holds one) as readable lines (both envelopes, without pydantic's "Value error, " prefix). */
export const describeNrTmError = (error: unknown) => describeApiError(error, nrTmLine)

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

/** True for a configuration that holds a test-model recipe (any configuration, including an imported one, may be asked). */
export const isTestModel = (config: { preset_id: string; nr_test_model?: GeneratorConfig['nr_test_model'] }) => !!config.nr_test_model

/** How a signal is named in the Studio's own lists: the recipe label of a test model ("TM3.1a · 100 MHz · 30 kHz · FDD"), else its preset id. */
export function signalName(config: { preset_id: string; nr_test_model?: GeneratorConfig['nr_test_model'] }) {
  const recipe = recipeOf(config)
  return recipe ? recipeLabel(recipe) : config.preset_id
}

/** The names of one selection's entries: `signalName`, with `uniqueRecipeLabels` telling apart test models that would otherwise read the same. */
export function signalNames(configs: Array<{ preset_id: string; nr_test_model?: GeneratorConfig['nr_test_model'] }>): string[] {
  const labels = uniqueRecipeLabels(configs.flatMap(config => { const recipe = recipeOf(config); return recipe ? [{ id: config.preset_id, recipe }] : [] }))
  return configs.map(config => recipeOf(config) ? labels[config.preset_id] ?? signalName(config) : config.preset_id)
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
  if (!info) return withPnScope(next)
  if (info.fdd_only) next.duplex = 'fdd'
  if (!info.supports_layers) { next.layers = 1; next.output_port = 0 }
  next.nb_iot = info.nb_iot ? recipe.nb_iot ?? { ...NB_IOT_DEFAULT } : null
  next.carriers = recipe.carriers.map(carrier => fitCarrier(catalog, info, carrier))
  return withPnScope(next)
}

/** A stepper that grows by cloning the last carrier (default cell ID, contiguous) and shrinks from the top. */
export function resizeCarriers(recipe: NrTmRecipe, count: number, max: number): NrTmRecipe {
  const n = Math.max(1, Math.min(max, Math.round(count)))
  if (n <= recipe.carriers.length) return { ...recipe, carriers: recipe.carriers.slice(0, n) }
  const last = recipe.carriers[recipe.carriers.length - 1]!
  return { ...recipe, carriers: [...recipe.carriers, ...Array.from({ length: n - recipe.carriers.length }, () => ({ ...last, n_cell_id: null, gap_before_mhz: 0 }))] }
}

/** Removing a carrier keeps the others as they are, except that the one that becomes first loses its gap (it has nothing to be a gap before). */
export function removeCarrier(recipe: NrTmRecipe, index: number): NrTmRecipe {
  if (recipe.carriers.length <= 1) return recipe
  return { ...recipe, carriers: recipe.carriers.filter((_, i) => i !== index).map((c, i) => i === 0 ? { ...c, gap_before_mhz: 0 } : c) }
}

/** The carriers that hold NB-IoT resource blocks: the first for `lower`, the last for `upper`, both for `both` (the backend's `_plan_nb_iot`; TS 38.141-1 4.9.2.2.9: L_NB-IoT / R_NB-IoT). */
export function nbIotHosts(recipe: NrTmRecipe): Array<{ index: number; carrier: NrTmCarrier }> {
  if (!recipe.nb_iot) return []
  const first = 0, last = recipe.carriers.length - 1
  const hosts = recipe.nb_iot.position === 'lower' ? [first] : recipe.nb_iot.position === 'upper' ? [last] : [first, last]
  return [...new Set(hosts)].map(index => ({ index, carrier: recipe.carriers[index]! }))
}

/**
 * The NB-IoT boost X the server accepts for this recipe: at least the largest per-host minimum of the catalogue (6 dB up to 20 MHz, 3 dB from 25 MHz;
 * TS 38.104 Table 6.3.4.2-1), at most the Studio limit. A catalogue that lists neither (an API older than this frontend) keeps the bounds of the schema.
 */
export function nbIotBoostBounds(catalog: NrTmCatalog, recipe: NrTmRecipe) {
  const { min_boost_db: minimum, max_boost_db: maximum } = catalog.nb_iot as Partial<NrTmCatalog['nb_iot']>
  const minima = nbIotHosts(recipe).flatMap(({ carrier }) => minimum?.[String(carrier.bandwidth_mhz)] ?? [])
  return { min: minima.length ? Math.max(...minima) : NB_IOT_BOOST.min, max: maximum ?? NB_IOT_BOOST.max }
}

/** One slot of a TDD pattern as the server names it (`TddPattern.labels`): `D`, `U`, `S 10D/2U`, `S 12D`, `S 8U`, or `F` for a slot that is all flexible. */
export function slotLabel(symbols: string) {
  const down = [...symbols].filter(c => c === 'D').length, up = [...symbols].filter(c => c === 'U').length
  if (down === symbols.length) return 'D'
  if (up === symbols.length) return 'U'
  return down || up ? 'S ' + [down ? `${down}D` : '', up ? `${up}U` : ''].filter(Boolean).join('/') : 'F'
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
