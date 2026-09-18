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

/** Mirror the server default; immutable signal IDs are independent of these names. */
export function datasetName(configs: GeneratorConfig[], role: 'in' | 'out' | 'inout' = 'in', modelId?: string) {
  if (!configs.length) return `syn_pa_${role}_n0`
  const families = [...new Set(configs.map(c => c.waveform === 'ofdm'
    ? c.preset_id.startsWith('nr-') ? 'nr' : c.preset_id.startsWith('wifi6-') ? 'w6' : c.preset_id.startsWith('wifi7-') ? 'w7' : c.waveform : c.waveform))].sort()
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

export const validDatasetName = (name: string, role: 'in' | 'inout') => name.length <= 96 && new RegExp(`^syn_pa_${role}_[A-Za-z0-9][A-Za-z0-9_.-]*$`).test(name)
export function pairedDatasetName(inputName: string, modelId: string) {
  const suffix = '_' + modelId.slice(0, 20)
  return inputName.replace(/^syn_pa_in_/, 'syn_pa_inout_').slice(0, 96 - suffix.length) + suffix
}
