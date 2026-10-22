import { describe, expect, test } from 'vitest'
import type { NrTmRecipe } from '@/api/nrTestModels'
import type { GeneratorConfig } from '@/api/signalGenerator'
import { fixtureCatalog, fixtureResolve } from '@/test/nrTestModelFixtures'
import { bandwidthOptions, convertModel, fitCarrier, mergeEditable, modelInfo, nextModel, normalizeRecipe, recipeKey, recipeLabel, recipeOf, rekey, resizeCarriers, shortModel, signalName } from './nrTestModel'

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
