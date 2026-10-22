import type { GeneratorConfig } from '@/api/signalGenerator'

const number = (value: number) => {
  if (value !== 0 && Math.abs(value) < .001) {
    const [mantissa, exponent] = value.toExponential(2).split('e')
    return mantissa!.replace(/\.?0+$/, '').replace('.', 'p') + 'e' + Number(exponent)
  }
  return value.toFixed(3).replace(/\.?0+$/, '').replace('.', 'p')
}
const range = (values: number[]) => {
  const low = Math.min(...values), high = Math.max(...values)
  return low === high ? number(low) : `${number(low)}-${number(high)}`
}

/**
 * Family token of a standard test model (same rule as opendpd/core/waveforms/dataset_names.py): "nr" followed by the model id without its
 * "NR-FR1-" / "NR-" prefix, lower-cased, "." written "p" and "-" dropped. NR-FR1-TM3.1a -> "nrtm3p1a", NR-N-TM -> "nrntm".
 * The `bw` part is `bandwidth_hz` of the resolved configuration, which for a test model is the symmetric extent of the occupied subcarriers plus
 * one subcarrier spacing: 100 MHz at 30 kHz gives 98,310,000 Hz, so syn_pa_in_nrtm3p1a_bw98p31M_q256_c1_n1; NR-N-TM at 10 MHz and 15 kHz gives
 * 9,375,000 Hz (syn_pa_in_nrntm_bw9p375M_q4_c1_n1); two 20 MHz carriers at 30 kHz give 38,190,000 Hz (syn_pa_in_nrtm1p1_bw38p19M_q4_c2_n1).
 * The same strings are asserted in datasetNames.test.ts and in tests/unit/test_nr_tm_service.py::test_dataset_names_of_test_models.
 */
export const testModelFamily = (model: string) => 'nr' + model.replace(/^NR-(FR1-)?/, '').toLowerCase().replace(/\./g, 'p').replace(/-/g, '')

/** Mirror the server default; immutable signal IDs are independent of these names. */
export function datasetName(configs: GeneratorConfig[], role: 'in' | 'out' | 'inout' = 'in', modelId?: string) {
  if (!configs.length) return `syn_pa_${role}_n0`
  const families = [...new Set(configs.map(c => c.waveform === 'ofdm'
    ? c.nr_test_model ? testModelFamily(c.nr_test_model.model) : c.preset_id.startsWith('nr-') ? 'nr' : c.preset_id.startsWith('wifi6-') ? 'w6' : c.preset_id.startsWith('wifi7-') ? 'w7' : c.waveform : c.waveform))].sort()
  const parts = [families.length <= 2 ? families.join('-') : 'mix', `bw${range(configs.map(c => c.bandwidth_hz / 1e6))}M`]
  const qam = configs.flatMap(c => c.waveform === 'ofdm' ? c.channel_modulations : c.waveform === 'qam' ? [c.modulation_order] : [])
  const psk = configs.filter(c => c.waveform === 'psk').map(c => c.psk_order)
  const ofdm = configs.filter(c => c.waveform === 'ofdm')
  if (qam.length) parts.push('q' + range(qam))
  if (psk.length) parts.push('p' + range(psk))
  if (ofdm.length) {
    parts.push('c' + range(ofdm.map(c => c.channel_subcarriers.length)))
    const spacing = ofdm.map(c => c.sample_rate_hz / (c.fft_size * c.oversampling * 1000))
    if (new Set(spacing).size > 1) parts.push('s' + range(spacing) + 'k')
  }
  parts.push('n' + configs.length)
  if (modelId) parts.push(modelId.slice(0, 20))
  return `syn_pa_${role}_` + parts.join('_')
}

export const validDatasetName = (name: string, role: 'in' | 'inout') => name.length <= 96 && new RegExp(`^${role === 'in' ? '(syn|usr)' : 'syn'}_pa_${role}_[A-Za-z0-9][A-Za-z0-9_.-]*$`).test(name)
export function pairedDatasetName(inputName: string, modelId: string) {
  const suffix = '_' + modelId.slice(0, 20)
  return inputName.replace(/^(syn|usr)_pa_in_/, 'syn_pa_inout_').slice(0, 96 - suffix.length) + suffix
}
