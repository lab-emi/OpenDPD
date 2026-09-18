import fixture from '@mocks/generator_presets.json'
import type { GeneratorConfig } from '@/api/signalGenerator'
import { datasetName, pairedDatasetName, validDatasetName } from './datasetNames'

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
