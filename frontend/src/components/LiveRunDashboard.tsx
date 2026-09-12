import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Chip from '@mui/material/Chip'
import Grid from '@mui/material/Grid'
import LinearProgress from '@mui/material/LinearProgress'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import { useQuery } from '@tanstack/react-query'
import { useMemo } from 'react'
import { api, WEB_MODE } from '@/api/client'
import type { StreamState } from '@/api/events'
import type { RunView } from '@/api/types'
import { isTerminal } from '@/api/types'
import { formatNumber, formatTime, message, phaseLabel, t } from '@/i18n'
import { IQPreview } from './IQPreview'
import { SpectrumPlot } from './SpectrumPlot'
import type { SpectrumData } from './ResultCharts'
import { MetricHistoryChart } from './MetricHistoryChart'

interface Geometry { batch_size: number; sequence_samples: number; sample_rate_hz: number | null; frame_stride?: number; train_sequences: number; batches_per_epoch: number }
interface LiveSnapshot {
  policy: { min_batches: number; min_seconds: number; overhead_target: number }
  geometry: Geometry | null
  training_geometry?: Geometry
  last_batch?: { phase: string; sequences: number; sequence_samples: number; sample_rate_hz?: number; padded_samples?: number }
  preview_error?: string
  preview: {
    revision: number; updated_at: string; source: string; samples: number; metrics: Record<string, number>; interval_seconds?: number; metric_profile?: string
    units: Record<string, string>
    plots: { spectrum?: SpectrumData; time?: { start: number; traces: Array<{ name: string; i: number[]; q: number[] }> } }
  } | null
}

export const liveKey = (id: string) => ['run', id, 'live'] as const

/** Latest worker snapshot only; structural sharing avoids redraws of unchanged arrays. */
export function LiveRunDashboard({ run, stream, metrics }: { run: RunView; stream: StreamState; metrics: StreamState['metrics'] }) {
  const active = !isTerminal(run.status)
  const query = useQuery({ queryKey: liveKey(run.run_id), queryFn: ({ signal }) => api.get<LiveSnapshot>(`/runs/${encodeURIComponent(run.run_id)}/live`, signal), refetchInterval: active ? (WEB_MODE ? 5000 : 2000) : false, retry: false })
  const snapshot = query.data, preview = snapshot?.preview, geometry = snapshot?.training_geometry ?? snapshot?.geometry
  const progress = stream.batchProgress
  const spec = preview?.plots.spectrum, time = preview?.plots.time
  const spectra = useMemo(() => (spec?.traces ?? []).map((trace) => ({ name: trace.name, psdDb: trace.psd_db })), [spec])
  const series = useMemo(() => time?.traces ?? [], [time])
  const dpd = run.task === 'train_dpd' || run.task === 'run_dpd'
  const training = run.task === 'train_pa' || run.task === 'train_dpd'
  const general = preview?.metric_profile === 'general-spectral-v1'
  const names = general ? (dpd ? ['ACPR_L', 'ACPR_R', 'NMSE', 'IBE'] : ['NMSE', 'IBE']) : dpd ? ['ACLR_AVG', 'NMSE', 'ACLR_L', 'ACLR_R'] : ['NMSE', 'EVM']
  const final = preview?.source === 'final_test'
  const batchSize = geometry?.batch_size
  const length = geometry?.sequence_samples
  const actual = progress?.sequences ? progress : snapshot?.last_batch
  const actualSize = actual?.sequences ?? batchSize
  const actualLength = actual?.sequence_samples ?? length
  const fs = actual?.sample_rate_hz ?? geometry?.sample_rate_hz
  const rate = fs ? `${Number((fs / 1e6).toPrecision(6))} MSa/s` : t('live.unknownRate')
  return <Stack spacing={2} data-testid="live-dashboard">
    <Paper sx={{ p: 2.5 }}>
      <Stack direction="row" sx={{ justifyContent: 'space-between', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
        <Typography variant="h2">{t('live.title')}</Typography>
        <Chip size="small" variant="outlined" label={progress?.phase ? phaseLabel(progress.phase) : active ? t('live.preparing') : message(run.status)} />
      </Stack>
      {batchSize && length ? <>
        <Typography sx={{ mt: 2, fontSize: 20, fontWeight: 650 }}>{t('live.capacity')} · {formatNumber(batchSize)} <Box component="span" sx={{ fontSize: 14, fontWeight: 400 }}>{t('live.sequencesPerBatch')}</Box> × {formatNumber(length)} <Box component="span" sx={{ fontSize: 14, fontWeight: 400 }}>{t('live.samplesPerSequence')}</Box></Typography>
        <Typography color="text.secondary" variant="body2" sx={{ mt: .75 }}>{t('live.actual', { count: actualSize ?? '', length: actualLength ?? '' })} · {t('live.geometry', { samples: formatNumber((actualSize ?? 0) * (actualLength ?? 0)), rate })}{geometry?.frame_stride ? ` · ${t('live.stride', { stride: geometry.frame_stride })}` : ''}</Typography>
        {!!actual?.padded_samples && <Typography variant="caption" color="text.secondary">{t('live.padding', { count: actual.padded_samples })}</Typography>}
      </> : <Typography color="text.secondary" sx={{ mt: 1.5 }}>{t('live.waitingGeometry')}</Typography>}
      {active && progress?.batch !== undefined && progress.total_batches && ['train', 'val', 'test', 'evaluate'].includes(progress.phase) ? <Box sx={{ mt: 2 }}>
        <Stack direction="row" sx={{ justifyContent: 'space-between', mb: .75 }}><Typography variant="body2">{phaseLabel(progress.phase)} · {t('live.batch', { batch: progress.batch, total: progress.total_batches })}</Typography><Typography variant="body2">{Math.round(progress.batch / progress.total_batches * 100)}%</Typography></Stack>
        <LinearProgress variant="determinate" value={progress.batch / progress.total_batches * 100} aria-label={t('live.batch', { batch: progress.batch, total: progress.total_batches })} />
      </Box> : active && <LinearProgress sx={{ mt: 2 }} aria-label={t('live.preparing')} />}
      {training && snapshot?.policy && <Typography variant="caption" color="text.secondary" component="p" sx={{ mt: 1.5 }}>{t('live.cadence', { batches: snapshot.policy.min_batches, seconds: (preview?.interval_seconds ?? snapshot.policy.min_seconds).toFixed(1) })}</Typography>}
    </Paper>
    {dpd && <Alert severity="info">{t('live.surrogate')}</Alert>}
    <Grid container spacing={2}>
      {names.filter((name) => preview?.metrics[name] !== undefined).map((name) => <Grid size={{ xs: 6, md: dpd ? 3 : 6 }} key={name}>
        <Paper sx={{ p: 2 }}><Typography variant="body2" color="text.secondary">{name}</Typography><Typography sx={{ fontSize: 28, fontWeight: 650, mt: .5, fontVariantNumeric: 'tabular-nums' }}>{preview!.metrics[name]!.toFixed(2)} <Box component="span" sx={{ fontSize: 13 }}>{preview?.units?.[name]}</Box></Typography><Typography variant="caption" color="text.secondary">{t(final ? 'live.final' : 'live.probe')}</Typography></Paper>
      </Grid>)}
    </Grid>
    {training && metrics.some((point) => point.values[dpd ? 'ACLR_AVG' : 'NMSE'] !== undefined) && <Paper sx={{ p: 2 }}><MetricHistoryChart points={metrics} metric={dpd ? 'ACLR_AVG' : 'NMSE'} height={285} /></Paper>}
    <Paper sx={{ p: 2.5 }}>
      <Stack direction="row" sx={{ justifyContent: 'space-between', gap: 1, flexWrap: 'wrap', mb: 1 }}><Typography variant="h2">{t('live.signals')}</Typography>{preview && <Typography variant="caption" color="text.secondary">{formatTime(preview.updated_at)}</Typography>}</Stack>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>{preview ? t(final ? 'live.finalHelp' : 'live.probeHelp', { samples: preview.samples, source: phaseLabel(preview.source === 'validation_probe' ? 'validation' : 'test') }) : t('live.waitingSignals')}</Typography>
      {preview?.metric_profile && <Typography variant="caption" color="text.secondary" component="p" sx={{ mb: 1 }}><code>{preview.metric_profile}</code></Typography>}
      {snapshot?.preview_error && <Alert severity="warning">{message(snapshot.preview_error)}</Alert>}
      {query.isError && <Alert severity="warning">{t('live.unavailable')}</Alert>}
      <Grid container spacing={2}>
        {time && <Grid size={{ xs: 12, xl: 6 }}><IQPreview series={series} start={time.start} viewKey={`${run.run_id}:${preview?.source}`} height={285} /></Grid>}
        {spec && <Grid size={{ xs: 12, xl: 6 }}><SpectrumPlot frequencyHz={spec.frequency} axis={spec.axis} traces={spectra} bands={spec.bands ?? undefined} viewKey={`${run.run_id}:${preview?.source}`} height={285} /></Grid>}
      </Grid>
    </Paper>
  </Stack>
}
