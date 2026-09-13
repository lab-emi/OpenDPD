import { t } from '@/i18n'

export const SIGNAL_NODES = ['dpd_input', 'pa_input', 'pa_output', 'unknown'] as const
export type SignalNode = typeof SIGNAL_NODES[number]
export interface SignalIdentity { name: string; role?: string; stage?: string; signal_node?: string | null; source?: string }

/** Old plots-v1 files used x for both PA input and DPD input. Resolve that
 * ambiguity from the whole capture, never from a display alias or run id. */
export function hasDPD(traces: SignalIdentity[]) {
  return traces.some(tr => tr.signal_node === 'dpd_input' || tr.stage === 'u' || tr.role === 'predistorted' || /dpd|target input/i.test(tr.name))
}
export function signalNode(trace: SignalIdentity, dpd: boolean): SignalNode {
  if (SIGNAL_NODES.includes(trace.signal_node as SignalNode)) return trace.signal_node as SignalNode
  if (trace.stage === 'u' || trace.role === 'predistorted') return 'pa_input'
  if (trace.stage === 'y' || trace.stage === 'reference') return 'pa_output'
  if (trace.stage === 'x' || trace.role === 'input') return dpd ? 'dpd_input' : 'pa_input'
  // Legacy roles identify output comparisons, including the linear target.
  if (['reference', 'primary', 'baseline'].includes(trace.role ?? '')) return 'pa_output'
  if (/^(u =|dpd output|pa input)/i.test(trace.name)) return 'pa_input'
  if (/^(input|target input)/i.test(trace.name)) return dpd ? 'dpd_input' : 'pa_input'
  if (/pa output|with dpd|without dpd/i.test(trace.name)) return 'pa_output'
  return 'unknown'
}
export function nodeTitle(node: SignalNode, dpd = false) {
  return t(node === 'pa_input' && dpd ? 'spectrum.node.dpdOutput' : `spectrum.node.${node}`)
}
export function spectrumGroups<T extends SignalIdentity>(traces: T[], dpd = hasDPD(traces)) {
  return SIGNAL_NODES.map(node => ({ node, traces: traces.filter(trace => signalNode(trace, dpd) === node) })).filter(group => group.traces.length)
}

/** Compact legends; full trace names and source/capture identifiers stay in
 * the review controls and exported data. Synthetic evidence remains explicit. */
export function spectrumLegend(trace: SignalIdentity): string {
  const name = trace.name.toLowerCase(), synthetic = trace.source?.includes('synthetic')
  if (trace.role === 'input') return t('spectrum.trace.input')
  if (trace.role === 'predistorted') return t('spectrum.trace.predistorted')
  if (name.includes('linear target')) return t('spectrum.trace.target')
  if (name.includes('without dpd')) return t(name.includes('surrogate') ? 'spectrum.trace.surrogateWithout' : synthetic ? 'spectrum.trace.syntheticWithout' : 'spectrum.trace.measuredWithout')
  if (name.includes('surrogate') || name.startsWith('with dpd')) return t('spectrum.trace.surrogateWith')
  if (name.includes('with dpd')) return t('spectrum.trace.measuredWith')
  if (name.includes('model output')) return t('spectrum.trace.model')
  if (name.includes('measured pa output')) return t(synthetic ? 'spectrum.trace.synthetic' : 'spectrum.trace.measured')
  return trace.name
}
