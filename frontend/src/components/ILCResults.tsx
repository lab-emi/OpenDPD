import { useMemo } from 'react'
import Alert from '@mui/material/Alert'
import Paper from '@mui/material/Paper'
import Typography from '@mui/material/Typography'
import { t, formatNumber, useLanguage } from '@/i18n'
import { MathFormula } from './MathFormula'
import { PlotlyChart, type PlotTrace } from './PlotlyChart'

const STOP_LABELS = {
  target_reached: 'ilc.stop.target_reached', iteration_limit: 'ilc.stop.iteration_limit',
  no_improving_step: 'ilc.stop.no_improving_step', improvement_below_tolerance: 'ilc.stop.improvement_below_tolerance',
} as const
function stopLabel(reason: unknown) {
  return typeof reason === 'string' && reason in STOP_LABELS ? t(STOP_LABELS[reason as keyof typeof STOP_LABELS]) : String(reason ?? '—')
}

export function ILCResults({ evidence }: { evidence: Record<string, unknown> }) {
  useLanguage()
  const trainingLabel = t('ilc.trainingCurve'), testLabel = t('ilc.testCurve')
  const traces = useMemo<PlotTrace[]>(() => ['training', 'test'].map((split, i) => {
    const rows = (evidence[`${split}_history`] ?? []) as Array<{ iteration: number; nmse_db: number }>
    return { type: 'scatter', mode: 'lines+markers', name: split === 'test' ? testLabel : trainingLabel,
      x: rows.map(r => r.iteration), y: rows.map(r => r.nmse_db), line: { dash: i ? 'dash' : 'solid' } }
  }), [evidence, trainingLabel, testLabel])
  return <Paper sx={{ p: 2.5 }} data-testid="ilc-results">
    <Typography variant="h2" gutterBottom>{t('ilc.benchmark')}</Typography>
    <Alert severity="info">{t('ilc.idealHelp')}</Alert>
    <MathFormula display latex={String.raw`e_k=Gx-\widehat{\mathrm{PA}}(u_k),\quad u_{k+1}=\operatorname{clip}_{A_{\max}}\!\left(u_k+\alpha_k\widehat g^{-1}e_k\right)`} />
    <MathFormula display latex={String.raw`w_{\mathrm{ILA}}=\arg\min_w\|\Phi(y_{\mathrm{ILC,train}}/G)w-u_{\mathrm{ILC,train}}\|_2^2,\quad u_{\mathrm{DPD}}=\Phi(x)w_{\mathrm{ILA}}`} />
    <PlotlyChart title={t('ilc.title')} traces={traces} height={300} layout={{ xaxis: { title: { text: t('ilc.iteration') } }, yaxis: { title: { text: t('ilc.trackingNmse') } }, showlegend: true }} />
    <Typography variant="body2" sx={{ mt: 1 }}>{t('ilc.summary', { trainReason: stopLabel(evidence.training_stop_reason), trainCount: formatNumber(Number(evidence.fit_samples)), testReason: stopLabel(evidence.test_stop_reason), testCount: formatNumber(Number(evidence.test_samples)) })}</Typography>
    <Typography variant="caption" color="text.secondary">{t('ilc.plant', { pa: String(evidence.test_plant_run_id), peak: Number(evidence.peak_limit).toPrecision(4), calls: formatNumber(Number(evidence.test_plant_calls)) })}</Typography>
  </Paper>
}
