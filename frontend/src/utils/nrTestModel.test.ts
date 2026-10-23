import { describe, expect, test } from 'vitest'
import type { NrTmRecipe } from '@/api/nrTestModels'
import type { GeneratorConfig } from '@/api/signalGenerator'
import { fixtureCatalog, fixtureResolve } from '@/test/nrTestModelFixtures'
import { ApiError } from '@/api/client'
import { bandwidthOptions, convertModel, describeNrTmError, fitCarrier, mergeEditable, modelInfo, nbIotHosts, nextModel, normalizeRecipe, nrTmFieldLabel, recipeKey, recipeLabel, recipeOf, rekey, removeCarrier, resizeCarriers, shortModel, signalName, signalNames, slotLabel, uniqueRecipeLabels, withPnScope } from './nrTestModel'

const catalog = fixtureCatalog()
const base = normalizeRecipe(catalog.default_recipe)
const carrier = (scs: 15 | 30 | 60, bandwidth_mhz: number) => ({ scs_khz: scs, bandwidth_mhz, n_cell_id: null, gap_before_mhz: 0 })

describe('recipes', () => {
  test('wire recipes with omitted optional keys become complete, and undefined or null mean "no test model"', () => {
    const wire = { version: 'nr-test-model-v1' as const, model: 'NR-FR1-TM2' as const, duplex: 'tdd' as const, layers: 1 as const, output_port: 0 as const, data_convention: 'standard' as const, pn_scope: 'dl_symbols' as const }
    const recipe = normalizeRecipe(wire)
    expect(recipe.carriers).toEqual([{ scs_khz: 30, bandwidth_mhz: 20, n_cell_id: null, gap_before_mhz: 0 }])
    expect(recipe).toMatchObject({ nb_iot: null, oversampling: null, subframes: null })
    expect(recipeOf(undefined)).toBeNull()
    expect(recipeOf({ nr_test_model: null })).toBeNull()
    expect(recipeOf({ nr_test_model: undefined })).toBeNull()
    expect(recipeOf({ nr_test_model: wire })?.model).toBe('NR-FR1-TM2')
  })

  test('the key does not depend on property order, so equal recipes share one request', () => {
    const shuffled = Object.fromEntries(Object.entries(base).reverse()) as NrTmRecipe
    expect(recipeKey(shuffled)).toBe(recipeKey(base))
    expect(recipeKey({ ...base, duplex: 'tdd' })).not.toBe(recipeKey(base))
  })

  test('labels read "TM3.1a · 100 MHz · 30 kHz · FDD" and describe several carriers', () => {
    expect(shortModel('NR-FR1-TM3.1a')).toBe('TM3.1a')
    expect(shortModel('NR-N-TM')).toBe('NR-N-TM')
    expect(recipeLabel({ ...base, model: 'NR-FR1-TM3.1a', carriers: [carrier(30, 100)] })).toBe('TM3.1a · 100 MHz · 30 kHz · FDD')
    expect(recipeLabel({ ...base, duplex: 'tdd', carriers: [carrier(30, 20), carrier(30, 20), carrier(30, 20)] })).toBe('TM1.1 · 3 × 20 MHz · 30 kHz · TDD')
    expect(recipeLabel({ ...base, carriers: [carrier(15, 5), carrier(30, 100)] })).toBe('TM1.1 · 5+100 MHz · 15/30 kHz · FDD')
    expect(recipeLabel({ ...base, layers: 2 })).toBe('TM1.1 · 20 MHz · 30 kHz · FDD · 2 layers')
  })

  test('signalName reads the recipe label of a test model and falls back to the preset id of a matrix or custom signal', () => {
    const config = fixtureResolve({ ...base, model: 'NR-FR1-TM3.1a', carriers: [carrier(30, 100)] }).config
    expect(config.preset_id).toMatch(/^nr-tm3p1a-fdd-30k-100m-c1-[0-9a-f]{6}$/)
    expect(signalName(config)).toBe('TM3.1a · 100 MHz · 30 kHz · FDD')
    expect(signalName({ preset_id: 'nr-20' })).toBe('nr-20')
    expect(signalName({ preset_id: 'nr-20', nr_test_model: null })).toBe('nr-20')
  })

  test('signalNames tells apart test models that would read the same and leaves matrix and custom signals at their preset id', () => {
    const standard = fixtureResolve(base).config
    const matlab = fixtureResolve({ ...base, data_convention: 'matlab' }).config
    expect(standard.preset_id).not.toBe(matlab.preset_id)
    expect(signalNames([standard, { preset_id: 'nr-20' }, matlab])).toEqual(['TM1.1 · 20 MHz · 30 kHz · FDD', 'nr-20', 'TM1.1 · 20 MHz · 30 kHz · FDD · MATLAB 5G Toolbox'])
    // Alone, a test model keeps its plain name (and the names do not depend on the order of the selection).
    expect(signalNames([matlab])).toEqual(['TM1.1 · 20 MHz · 30 kHz · FDD'])
    expect(signalNames([matlab, standard])).toEqual(['TM1.1 · 20 MHz · 30 kHz · FDD · MATLAB 5G Toolbox', 'TM1.1 · 20 MHz · 30 kHz · FDD'])
  })
})

describe('the PN23 scope exists only where it can matter', () => {
  const all = { ...base, duplex: 'tdd' as const, pn_scope: 'all_symbols' as const }
  test('withPnScope keeps "All symbols" for TDD with the standard convention and resets it everywhere else', () => {
    expect(withPnScope(all)).toBe(all)
    expect(withPnScope({ ...all, duplex: 'fdd' }).pn_scope).toBe('dl_symbols')
    expect(withPnScope({ ...all, data_convention: 'matlab' }).pn_scope).toBe('dl_symbols')
    const downlink = { ...all, pn_scope: 'dl_symbols' as const, duplex: 'fdd' as const }
    expect(withPnScope(downlink)).toBe(downlink)
  })

  test('the shared normaliser applies it: a recipe from the wire, a saved configuration or a server echo cannot carry it into FDD', () => {
    expect(normalizeRecipe({ ...all, duplex: 'fdd' }).pn_scope).toBe('dl_symbols')
    expect(normalizeRecipe(all).pn_scope).toBe('all_symbols')
    expect(recipeOf({ nr_test_model: { ...all, duplex: 'fdd' } })?.pn_scope).toBe('dl_symbols')
    // Without the scope the two recipes of one signal would differ in their key (and in the preset id the server derives from it).
    expect(recipeKey(normalizeRecipe({ ...all, duplex: 'fdd' }))).toBe(recipeKey(normalizeRecipe({ ...all, duplex: 'fdd', pn_scope: 'dl_symbols' })))
  })

  test('a model switch that forces FDD (NR-N-TM) drops the scope, and coming back does not bring it back', () => {
    const ntm = convertModel(catalog, { ...all, model: 'NR-FR1-TM3.2' }, 'NR-N-TM')
    expect(ntm).toMatchObject({ duplex: 'fdd', pn_scope: 'dl_symbols' })
    expect(convertModel(catalog, ntm, 'NR-FR1-TM3.2')).toMatchObject({ duplex: 'fdd', pn_scope: 'dl_symbols' })
    // Between models that keep TDD the choice is kept.
    expect(convertModel(catalog, { ...all, model: 'NR-FR1-TM3.2' }, 'NR-FR1-TM3.3')).toMatchObject({ duplex: 'tdd', pn_scope: 'all_symbols' })
  })
})

describe('select filtering comes from the catalogue', () => {
  test('bandwidth options follow the N_RB table of the chosen subcarrier spacing', () => {
    expect(bandwidthOptions(catalog, undefined, 15)).toEqual([3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    expect(bandwidthOptions(catalog, undefined, 30)).toEqual([5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100])
    expect(bandwidthOptions(catalog, undefined, 60)).toEqual([10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100])
    expect(bandwidthOptions(catalog, undefined, 120)).toEqual([])
  })

  test('NR-N-TM only offers the NB-IoT bandwidths of 15 kHz', () => {
    const info = modelInfo(catalog, 'NR-N-TM')
    expect(info?.fdd_only).toBe(true)
    expect(bandwidthOptions(catalog, info, 15)).toEqual([3, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50])
    expect(bandwidthOptions(catalog, info, 30)).not.toContain(60)
  })

  test('changing the spacing moves the bandwidth to the closest one that exists', () => {
    const info = modelInfo(catalog, 'NR-FR1-TM1.1')
    expect(fitCarrier(catalog, info, carrier(60, 5))).toMatchObject({ scs_khz: 60, bandwidth_mhz: 10 })
    expect(fitCarrier(catalog, info, carrier(30, 3))).toMatchObject({ bandwidth_mhz: 5 })
    expect(fitCarrier(catalog, info, carrier(30, 100))).toMatchObject({ bandwidth_mhz: 100 })
  })
})

describe('switching the model repairs what the new model does not allow', () => {
  test('NR-N-TM forces FDD, 15 kHz and an NB-IoT block; leaving it removes the block again', () => {
    const wide: NrTmRecipe = { ...base, duplex: 'tdd', layers: 2, output_port: 1, carriers: [carrier(60, 100)] }
    const ntm = convertModel(catalog, wide, 'NR-N-TM')
    expect(ntm).toMatchObject({ model: 'NR-N-TM', duplex: 'fdd', layers: 1, output_port: 0, nb_iot: { position: 'lower', boost_db: 6 } })
    expect(ntm.carriers).toEqual([carrier(15, 50)])
    const back = convertModel(catalog, ntm, 'NR-FR1-TM3.1')
    expect(back.nb_iot).toBeNull()
    expect(back.carriers).toEqual([carrier(15, 50)])
  })

  test('rank 2 survives only in TM1.1', () => {
    const rank2 = { ...base, layers: 2 as const, output_port: 1 as const }
    expect(convertModel(catalog, rank2, 'NR-FR1-TM1.1')).toMatchObject({ layers: 2, output_port: 1 })
    expect(convertModel(catalog, rank2, 'NR-FR1-TM3.1')).toMatchObject({ layers: 1, output_port: 0 })
  })

  test('every model of the catalogue converts to a recipe the fixture server accepts', () => {
    for (const { model } of catalog.models) expect(() => fixtureResolve(convertModel(catalog, base, model)), model).not.toThrow()
  })
})

describe('carriers and selection helpers', () => {
  test('growing clones the last carrier with default cell ID and no gap; shrinking removes from the top; the limit holds', () => {
    const two = { ...base, carriers: [carrier(30, 20), { scs_khz: 30 as const, bandwidth_mhz: 40, n_cell_id: 7, gap_before_mhz: 0.6 }] }
    const four = resizeCarriers(two, 4, 16)
    expect(four.carriers).toHaveLength(4)
    expect(four.carriers[3]).toEqual({ scs_khz: 30, bandwidth_mhz: 40, n_cell_id: null, gap_before_mhz: 0 })
    expect(resizeCarriers(four, 1, 16).carriers).toEqual([carrier(30, 20)])
    expect(resizeCarriers(base, 99, 16).carriers).toHaveLength(16)
    expect(resizeCarriers(base, 0, 16).carriers).toHaveLength(1)
  })

  test('"Add another" starts from the next model the selection does not use', () => {
    expect(nextModel(catalog, ['NR-FR1-TM1.1'], 'NR-FR1-TM1.1')).toBe('NR-FR1-TM1.2')
    expect(nextModel(catalog, ['NR-FR1-TM1.1', 'NR-FR1-TM1.2'], 'NR-FR1-TM1.1')).toBe('NR-FR1-TM2')
    expect(nextModel(catalog, ['NR-N-TM'], 'NR-N-TM')).toBe('NR-FR1-TM1.1')
    expect(nextModel(catalog, catalog.models.map(m => m.model), 'NR-FR1-TM3.3')).toBe('NR-N-TM')
  })

  test('re-keying keeps the position of the entry, and merges into an existing entry on collision', () => {
    expect(Object.keys(rekey({ a: 1, b: 2, c: 3 }, 'b', 'x', 9))).toEqual(['a', 'x', 'c'])
    expect(rekey({ a: 1, b: 2, c: 3 }, 'b', 'x', 9)).toEqual({ a: 1, x: 9, c: 3 })
    expect(rekey({ a: 1, b: 2, c: 3 }, 'b', 'c', 9)).toEqual({ a: 1, c: 3 })
  })

  test('a re-derived configuration keeps what the user edited on the previous one', () => {
    const previous = { ...fixtureResolve(base).config, rms: .31, seed: 7, filter_enabled: true, snr_db: 40, clip_db: 6, carrier_frequency_hz: 2.1e9 } as unknown as GeneratorConfig
    const resolved = fixtureResolve({ ...base, carriers: [carrier(30, 40)] }).config as unknown as GeneratorConfig
    const merged = mergeEditable(resolved, previous)
    expect(merged).toMatchObject({ rms: .31, seed: 7, filter_enabled: true, snr_db: 40, clip_db: 6, carrier_frequency_hz: 2.1e9, preset_id: resolved.preset_id, n_samples: resolved.n_samples })
    expect(mergeEditable(resolved, undefined)).toBe(resolved)
  })
})

describe('the first carrier has no gap', () => {
  const gap = (g: number) => ({ ...carrier(30, 10), gap_before_mhz: g })
  test('normalizeRecipe clears a leftover gap on the first carrier and keeps the others', () => {
    const wire = { ...base, carriers: [gap(0.6), gap(0.9)] }
    expect(normalizeRecipe(wire).carriers.map(c => c.gap_before_mhz)).toEqual([0, 0.9])
  })
  test('removing carrier 1 promotes the next one without its gap; removing another carrier changes nothing else; the last carrier cannot be removed', () => {
    const recipe = { ...base, carriers: [carrier(30, 20), gap(0.6), gap(1.2)] }
    expect(removeCarrier(recipe, 0).carriers.map(c => c.gap_before_mhz)).toEqual([0, 1.2])
    expect(removeCarrier(recipe, 1).carriers.map(c => c.gap_before_mhz)).toEqual([0, 1.2])
    expect(removeCarrier(recipe, 2).carriers.map(c => c.gap_before_mhz)).toEqual([0, 0.6])
    expect(removeCarrier({ ...base, carriers: [carrier(30, 20)] }, 0)).toEqual({ ...base, carriers: [carrier(30, 20)] })
  })
})

describe('entries that would read the same are told apart', () => {
  const entry = (id: string, change: Partial<NrTmRecipe> = {}, carrier0: Partial<NrTmRecipe['carriers'][number]> = {}) => ({ id, recipe: { ...base, ...change, carriers: [{ ...base.carriers[0]!, ...carrier0 }] } as NrTmRecipe })
  test('a label that no other entry shares stays exactly as it was', () => {
    expect(uniqueRecipeLabels([entry('a'), entry('b', { model: 'NR-FR1-TM1.2' })])).toEqual({ a: 'TM1.1 · 20 MHz · 30 kHz · FDD', b: 'TM1.2 · 20 MHz · 30 kHz · FDD' })
  })
  test.each([
    ['PN23 convention', {}, { data_convention: 'matlab' as const }, 'MATLAB 5G Toolbox'],
    ['cell ID', {}, { carriers: [{ ...base.carriers[0]!, n_cell_id: 5 }] }, 'cell 5'],
    ['capture length', {}, { subframes: 3 }, '3 subframes'],
    ['oversampling', {}, { oversampling: 4 as const }, 'Oversampling 4×'],
    ['PN scope', { duplex: 'tdd' as const }, { duplex: 'tdd' as const, pn_scope: 'all_symbols' as const }, 'PN23 all symbols'],
  ])('the entry that differs in the %s gets a short tag; the default one keeps the plain label', (_name, first, second, tag) => {
    const plain = recipeLabel({ ...base, ...first })
    const labels = uniqueRecipeLabels([{ id: 'a', recipe: { ...base, ...first } }, { id: 'b', recipe: { ...base, ...second } }])
    expect(labels).toEqual({ a: plain, b: `${plain} · ${tag}` })
  })
  test('two-layer entries on different ports are named after their port; recipes whose labels differ already get no tag', () => {
    const layered = (id: string, output_port: 0 | 1) => ({ id, recipe: { ...base, layers: 2 as const, output_port } })
    expect(uniqueRecipeLabels([layered('a', 0), layered('b', 1)])).toEqual({
      a: 'TM1.1 · 20 MHz · 30 kHz · FDD · 2 layers · Port 1000', b: 'TM1.1 · 20 MHz · 30 kHz · FDD · 2 layers · Port 1001' })
    expect(uniqueRecipeLabels([{ id: 'a', recipe: base }, { id: 'b', recipe: { ...base, layers: 2 } }])).toEqual({ a: 'TM1.1 · 20 MHz · 30 kHz · FDD', b: 'TM1.1 · 20 MHz · 30 kHz · FDD · 2 layers' })
  })
  test('two entries with the same label and the same tags still get distinct names (an ordinal as the last resort)', () => {
    const a = entry('a'), b = entry('b'), c = entry('c', { data_convention: 'matlab' })
    const labels = uniqueRecipeLabels([a, b, c])
    expect(new Set(Object.values(labels)).size).toBe(3)
    expect(labels.a).toBe('TM1.1 · 20 MHz · 30 kHz · FDD · #1')
    expect(labels.b).toBe('TM1.1 · 20 MHz · 30 kHz · FDD · #2')
    expect(labels.c).toBe('TM1.1 · 20 MHz · 30 kHz · FDD · MATLAB 5G Toolbox')
  })
  test('with several carriers the tags list every carrier, and NB-IoT shows its edge and boost', () => {
    const two = (id: string, ids: Array<number | null>) => ({ id, recipe: { ...base, carriers: ids.map(n_cell_id => ({ ...carrier(30, 20), n_cell_id })) } as NrTmRecipe })
    const labels = uniqueRecipeLabels([two('a', [null, null]), two('b', [null, 7]), two('c', [3, 7])])
    expect(labels).toEqual({ a: 'TM1.1 · 2 × 20 MHz · 30 kHz · FDD', b: 'TM1.1 · 2 × 20 MHz · 30 kHz · FDD · cell –/7', c: 'TM1.1 · 2 × 20 MHz · 30 kHz · FDD · cell 3/7' })
    const gapped = (id: string, gap_before_mhz: number) => ({ id, recipe: { ...base, carriers: [carrier(30, 20), { ...carrier(30, 20), gap_before_mhz }] } as NrTmRecipe })
    expect(uniqueRecipeLabels([gapped('a', 0), gapped('b', 0.6)])).toEqual({ a: 'TM1.1 · 2 × 20 MHz · 30 kHz · FDD', b: 'TM1.1 · 2 × 20 MHz · 30 kHz · FDD · gap 0.6 MHz' })
    const ntm = (id: string, position: 'lower' | 'upper' | 'both', boost_db: number) => ({ id, recipe: { ...base, model: 'NR-N-TM' as const, carriers: [carrier(15, 20)].map(c => ({ ...c, scs_khz: 15 as const })), nb_iot: { position, boost_db } } as NrTmRecipe })
    expect(uniqueRecipeLabels([ntm('a', 'lower', 6), ntm('b', 'both', 9)])).toEqual({ a: 'NR-N-TM · 20 MHz · 15 kHz · FDD · NB-IoT Lower, 6 dB', b: 'NR-N-TM · 20 MHz · 15 kHz · FDD · NB-IoT Both, 9 dB' })
  })
})

describe('TDD slot labels and NB-IoT hosts', () => {
  test('the slot labels derived from the catalogue are the ones the server prints for the same recipe', () => {
    for (const [scs, bandwidth] of [[60, 100], [30, 20]] as const) {
      const server = fixtureResolve({ ...base, duplex: 'tdd', carriers: [carrier(scs, bandwidth)] }).summary.tdd_pattern
      expect(catalog.tdd_patterns.find(p => p.scs_khz === scs)!.slot_map.map(slotLabel)).toEqual(server)
    }
    expect(catalog.tdd_patterns.find(p => p.scs_khz === 15)!.slot_map.map(slotLabel)).toEqual(['D', 'D', 'D', 'S 10D/2U', 'U'])
    expect(slotLabel('FFFFFFFFFFFFFF')).toBe('F')
  })
  test('the host of the lower NB-IoT carrier is the first carrier, of the upper one the last, and a single carrier hosts both', () => {
    const ntm = (position: 'lower' | 'upper' | 'both', count: number): NrTmRecipe => ({ ...base, model: 'NR-N-TM', nb_iot: { position, boost_db: 6 }, carriers: Array.from({ length: count }, (_, i) => ({ ...carrier(15, 10 + 10 * i), scs_khz: 15 as const })) })
    expect(nbIotHosts(ntm('lower', 3)).map(h => h.index)).toEqual([0])
    expect(nbIotHosts(ntm('upper', 3)).map(h => h.index)).toEqual([2])
    expect(nbIotHosts(ntm('both', 3)).map(h => h.index)).toEqual([0, 2])
    expect(nbIotHosts(ntm('both', 1)).map(h => h.index)).toEqual([0])
    expect(nbIotHosts(base)).toEqual([])
  })
})

describe('server field paths read as the form labels', () => {
  test.each([
    ['carriers.0.n_cell_id', 'Carrier 1 · Cell ID'],
    ['carriers.2.bandwidth_mhz', 'Carrier 3 · Channel bandwidth'],
    ['carriers.1.gap_before_mhz', 'Carrier 2 · Gap before'],
    ['carriers.0.scs_khz', 'Carrier 1 · Subcarrier spacing'],
    ['nb_iot.boost_db', 'NB-IoT power boost X'],
    ['nb_iot.position', 'Punctured NR RB'],
    ['subframes', 'Capture length'],
    ['oversampling', 'Oversampling'],
    ['data_convention', 'PN23 data convention'],
    ['nr_test_model.carriers.1.n_cell_id', 'Carrier 2 · Cell ID'],
    ['configs.3.nr_test_model.subframes', 'Capture length'],
    ['carriers.0', 'Carrier 1'],
    ['', ''],
    // What the form has no field for stays as the server named it, so nothing is lost.
    ['carriers.0.something_new', 'carriers.0.something_new'],
    ['preset_id', 'preset_id'],
    ['constructor', 'constructor'],
  ])('%s -> %s', (path, label) => { expect(nrTmFieldLabel(path)).toBe(label) })

  test('refusal details carry the label, drop the pydantic prefix, and explain the 16-carrier list limit as a Studio limit', () => {
    const refusal = (details: Array<{ field: string; message: string }>) => new ApiError(422, 'invalid_request', 'request validation failed', details)
    expect(describeNrTmError(refusal([{ field: 'carriers.0.n_cell_id', message: 'Input should be less than or equal to 1007' }]))).toEqual(['Carrier 1 · Cell ID: Input should be less than or equal to 1007'])
    expect(describeNrTmError(refusal([{ field: 'nb_iot.boost_db', message: 'Input should be less than or equal to 12' }, { field: '', message: 'Value error, plain' }]))).toEqual(
      ['NB-IoT power boost X: Input should be less than or equal to 12', 'plain'])
    expect(describeNrTmError(refusal([{ field: 'carriers', message: 'List should have at most 16 items after validation, not 17' }]))).toEqual(['16 carriers is the Studio limit (the standard sets none).'])
    expect(describeNrTmError(refusal([{ field: 'nr_test_model.carriers', message: 'List should have at most 16 items after validation, not 20' }]))).toEqual(['16 carriers is the Studio limit (the standard sets none).'])
  })
})
