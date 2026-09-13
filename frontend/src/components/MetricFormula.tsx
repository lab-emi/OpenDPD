import type { MetricDefinition } from '@/api/types'
import { MathFormula } from './MathFormula'

// Exact definition matches: an unknown/new protocol remains readable as its source text.
const FORMULAS: Record<string, string> = {
  '20 * log10( EVM_RMS / 100 )': String.raw`20\log_{10}\frac{\mathrm{EVM}_{\mathrm{RMS}}}{100}`,
  '10 * log10( P[-29, -11 MHz] / P[-9, +9 MHz] ) from the Welch PSD at the capture rate': String.raw`10\log_{10}\frac{P_{[-29,-11]\,\mathrm{MHz}}}{P_{[-9,+9]\,\mathrm{MHz}}}`,
  '10 * log10( P[+11, +29 MHz] / P[-9, +9 MHz] ) from the Welch PSD at the capture rate': String.raw`10\log_{10}\frac{P_{[+11,+29]\,\mathrm{MHz}}}{P_{[-9,+9]\,\mathrm{MHz}}}`,
  'mean over segments of 10*log10(sum|e|^2 / sum|y|^2)': String.raw`\frac1S\sum_{s=1}^{S}10\log_{10}\frac{\sum_n|e_s[n]|^2}{\sum_n|y_s[n]|^2}`,
  '20*log10(mean over segments of mean over sub-channels of mean|X_pred - X_ref| / mean|X_ref|), FFT of nperseg samples': String.raw`20\log_{10}\!\left[\frac1{SC}\sum_{s,c}\frac{\operatorname{mean}_{k\in c}|X_{\mathrm{pred},s}[k]-X_{\mathrm{ref},s}[k]|}{\operatorname{mean}_{k\in c}|X_{\mathrm{ref},s}[k]|}\right]`,
  '10*log10(P_adjacent_left / max sub-channel power)': String.raw`10\log_{10}\frac{P_{\mathrm{adj},L}}{\max_c P_c}`,
  '10*log10(P_adjacent_right / max sub-channel power)': String.raw`10\log_{10}\frac{P_{\mathrm{adj},R}}{\max_c P_c}`,
  '(ACLR_L + ACLR_R) / 2': String.raw`\frac{\mathrm{ACLR}_L+\mathrm{ACLR}_R}{2}`,
  '10*log10( sum|y - r|^2 / sum|r|^2 ) over the valid range': String.raw`10\log_{10}\frac{\sum_{n\in\mathcal V}|y[n]-r[n]|^2}{\sum_{n\in\mathcal V}|r[n]|^2}`,
  '10*log10( P_main(y - r) / P_main(r) ) with band powers from the Welch PSD': String.raw`10\log_{10}\frac{P_{\mathrm{main}}(y-r)}{P_{\mathrm{main}}(r)}`,
  '10*log10( P_adjacent_left(y) / P_main(y) )': String.raw`10\log_{10}\frac{P_{\mathrm{adj},L}(y)}{P_{\mathrm{main}}(y)}`,
  '10*log10( P_adjacent_right(y) / P_main(y) )': String.raw`10\log_{10}\frac{P_{\mathrm{adj},R}(y)}{P_{\mathrm{main}}(y)}`,
  '100 * sqrt( sum|S_eq - S_ref|^2 / sum|S_ref|^2 ) over occupied subcarriers and complete symbols': String.raw`100\sqrt{\frac{\sum|S_{\mathrm{eq}}-S_{\mathrm{ref}}|^2}{\sum|S_{\mathrm{ref}}|^2}}`,
}
export function MetricFormula({ definition }: { definition: MetricDefinition }) {
  const latex = FORMULAS[definition.formula]
  return latex ? <MathFormula latex={latex} /> : <span>{definition.formula}</span>
}
