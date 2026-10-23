import fixture from '@mocks/generator_presets.json'
import type { GeneratorConfig } from '@/api/signalGenerator'
import { fixtureCatalog, fixtureResolve } from '@/test/nrTestModelFixtures'
import type { NrTmRecipe } from '@/api/nrTestModels'
import { datasetName, pairedDatasetName, testModelFamily, validDatasetName } from './datasetNames'

test('automatic names describe single specs and mixed spec ranges in a stable order', () => {
  const nr = fixture.data[0]!.config as GeneratorConfig
  const wifi = { ...nr, preset_id: 'wifi7-80-q256-c4', bandwidth_hz: 80e6, sample_rate_hz: 320e6, fft_size: 1024,
    channel_subcarriers: [100, 100, 100, 100], channel_modulations: [256, 256, 256, 256] }
  expect(datasetName([nr])).toBe('syn_pa_in_nr_bw20M_q64_c1_n1')
  const mixed = 'syn_pa_in_nr-w7_bw20-80M_q64-256_c1-4_s30-78p125k_n2'
  expect(datasetName([nr, wifi])).toBe(mixed)
  expect(datasetName([wifi, nr])).toBe(mixed)
  expect(pairedDatasetName(mixed, 'rapp-am-pm')).toBe(mixed.replace('syn_pa_in_', 'syn_pa_inout_') + '_rapp-am-pm')
  expect(datasetName([{ ...nr, waveform: 'psk', psk_order: 8 }])).toBe('syn_pa_in_psk_bw20M_p8_n1')
  expect(datasetName([{ ...nr, bandwidth_hz: 1 }])).toContain('bw1e-6M')
  expect(validDatasetName('syn_pa_in_../../escape', 'in')).toBe(false)
  expect(validDatasetName('syn_pa_out_lab_n1', 'inout')).toBe(false)
  expect(validDatasetName(pairedDatasetName('syn_pa_in_' + 'x'.repeat(86), 'rapp-am-pm'), 'inout')).toBe(true)
})

const recipe = (model: NrTmRecipe['model'], scs: 15 | 30 | 60, bandwidth: number, carriers = 1): NrTmRecipe => ({ version: 'nr-test-model-v1', model, duplex: 'fdd',
  carriers: Array.from({ length: carriers }, () => ({ scs_khz: scs, bandwidth_mhz: bandwidth, n_cell_id: null, gap_before_mhz: 0 })),
  layers: 1, output_port: 0, data_convention: 'standard', pn_scope: 'dl_symbols', nb_iot: model === 'NR-N-TM' ? { position: 'lower', boost_db: 6 } : null, oversampling: null, subframes: null })
/** The configuration the real backend resolves for a recipe (captured by scripts/capture_nr_tm_fixtures.py). */
const resolved = (...args: Parameters<typeof recipe>) => fixtureResolve(recipe(...args)).config as unknown as GeneratorConfig

test('the family token of a standard test model is "nr" + the model id without NR-FR1-/NR-, lower case, "." as "p", "-" dropped', () => {
  // Same table as test_family_token_of_every_model in tests/unit/test_nr_tm_service.py.
  const tokens: Record<string, string> = { 'NR-FR1-TM1.1': 'nrtm1p1', 'NR-FR1-TM1.2': 'nrtm1p2', 'NR-FR1-TM2': 'nrtm2', 'NR-FR1-TM2a': 'nrtm2a', 'NR-FR1-TM2b': 'nrtm2b',
    'NR-FR1-TM3.1': 'nrtm3p1', 'NR-FR1-TM3.1a': 'nrtm3p1a', 'NR-FR1-TM3.1b': 'nrtm3p1b', 'NR-FR1-TM3.2': 'nrtm3p2', 'NR-FR1-TM3.3': 'nrtm3p3', 'NR-N-TM': 'nrntm' }
  for (const [model, token] of Object.entries(tokens)) expect(testModelFamily(model), model).toBe(token)
  expect(Object.keys(tokens)).toEqual(fixtureCatalog().models.map(m => m.model))
})

// The configurations below are what `POST /signal-generator/nr-test-models/resolve` really answered (frontend/src/test/nrTmFixtureResponses.json). The
// bandwidth in the name is bandwidth_hz of the resolved configuration, the symmetric extent of the occupied subcarriers plus one subcarrier spacing:
// 100 MHz at 30 kHz = 3,276 subcarriers + 1 = 98.31 MHz, NR-N-TM 10 MHz at 15 kHz = 624 + 1 = 9.375 MHz, 2 x 20 MHz at 30 kHz = 38.19 MHz. Every
// expected string is also asserted by test_dataset_names_of_test_models in tests/unit/test_nr_tm_service.py, so a drift of either side fails one suite.
test('cross-language: the names of resolved test-model configurations equal the Python names', () => {
  const tm31a = resolved('NR-FR1-TM3.1a', 30, 100)
  const ntm = resolved('NR-N-TM', 15, 10)
  const two = resolved('NR-FR1-TM1.1', 30, 20, 2)
  const matrix = fixture.data[0]!.config as unknown as GeneratorConfig
  expect([tm31a.bandwidth_hz, ntm.bandwidth_hz, two.bandwidth_hz]).toEqual([98_310_000, 9_375_000, 38_190_000])
  expect(datasetName([tm31a])).toBe('syn_pa_in_nrtm3p1a_bw98p31M_q256_c1_n1')
  expect(datasetName([two])).toBe('syn_pa_in_nrtm1p1_bw38p19M_q4_c2_n1')
  expect(datasetName([two], 'inout', 'rapp-am-pm')).toBe('syn_pa_inout_nrtm1p1_bw38p19M_q4_c2_n1_rapp-am-pm')
  expect(datasetName([tm31a, matrix])).toBe('syn_pa_in_nr-nrtm3p1a_bw20-98p31M_q64-256_c1_n2')
  expect(datasetName([matrix, tm31a])).toBe(datasetName([tm31a, matrix]))
  expect(datasetName([tm31a, two, matrix])).toMatch(/^syn_pa_in_mix_/)
  expect(datasetName([matrix])).toBe('syn_pa_in_nr_bw20M_q64_c1_n1')
  expect(datasetName([ntm])).toBe('syn_pa_in_nrntm_bw9p375M_q4_c1_n1')
  expect(datasetName([tm31a, ntm])).toBe('syn_pa_in_nrntm-nrtm3p1a_bw9p375-98p31M_q4-256_c1_s15-30k_n2')
  expect(validDatasetName(datasetName([tm31a, ntm, matrix]), 'in')).toBe(true)
})
