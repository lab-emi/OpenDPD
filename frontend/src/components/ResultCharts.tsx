import Grid from '@mui/material/Grid'
import Paper from '@mui/material/Paper'
import Typography from '@mui/material/Typography'
import { useMemo } from 'react'
import { useArtifactJson } from '@/api/hooks'
import { t } from '@/i18n'
import { IQPreview } from './IQPreview'
import { PlotlyChart, type PlotLayout, type PlotTrace } from './PlotlyChart'
import { SpectrumPlot } from './SpectrumPlot'
import { LoadingState } from './StateBlock'

/** plots-v1 payloads written by the worker (see opendpd/core/plots.py); the page only draws them. */
export interface SpectrumData {
  version: string
  axis: 'hz' | 'normalized'
  sample_rate_hz: number | null
  nperseg: number
  n_samples: number
  frequency: number[]
  traces: Array<{ name: string; role: string; psd_db: number[] }>
  bands: { main: [number, number]; adjacent: Array<[number, number]> } | null
  estimator: string
}
export interface TimeData {
  version: string
  start: number
  n: number
  n_samples: number
  traces: Array<{ name: string; role: string; i: number[]; q: number[] }>
}
export interface AmData {
  version: string
  stride: number
  n_points: number
  n_samples: number
  amp_in: number[]
  traces: Array<{ name: string; role: string; amp_out: number[]; phase_deg: number[] }>
  note: string
}

const MARKER = { size: 3, opacity: 0.45 }

export function AmPlots({ data }: { data: AmData }) {
  const am = useMemo<PlotTrace[]>(() => data.traces.map((tr) => ({ x: data.amp_in, y: tr.amp_out, name: tr.name, mode: 'markers', type: 'scatter', marker: MARKER })), [data])
  const pm = useMemo<PlotTrace[]>(() => data.traces.map((tr) => ({ x: data.amp_in, y: tr.phase_deg, name: tr.name, mode: 'markers', type: 'scatter', marker: MARKER })), [data])
  const amLayout = useMemo<PlotLayout>(() => ({ xaxis: { title: { text: t('chart.am.x') } }, yaxis: { title: { text: t('chart.am.y') } }, showlegend: true }), [])
  const pmLayout = useMemo<PlotLayout>(() => ({ xaxis: { title: { text: t('chart.am.x') } }, yaxis: { title: { text: t('chart.pm.y') } }, showlegend: true }), [])
  return (
    <>
      <Grid size={{ xs: 12, md: 6 }}>
        <PlotlyChart title={t('chart.am.title')} traces={am} layout={amLayout} data-testid="am-am-plot" />
      </Grid>
      <Grid size={{ xs: 12, md: 6 }}>
        <PlotlyChart title={t('chart.pm.title')} traces={pm} layout={pmLayout} data-testid="am-pm-plot" />
      </Grid>
    </>
  )
}

/** Spectrum, time window and AM-AM / AM-PM of one result, from its registered plot artifacts. */
export function ResultCharts({ runId }: { runId: string }) {
  const spectrum = useArtifactJson<SpectrumData>(runId, 'plot-spectrum')
  const time = useArtifactJson<TimeData>(runId, 'plot-time')
  const am = useArtifactJson<AmData>(runId, 'plot-amam')
  const spectrumTraces = useMemo(() => (spectrum.data?.traces ?? []).map((tr) => ({ name: tr.name, psdDb: tr.psd_db })), [spectrum.data])
  const series = useMemo(() => (time.data?.traces ?? []).map((tr) => ({ name: tr.name, i: tr.i, q: tr.q })), [time.data])
  const pending = spectrum.isPending || time.isPending || am.isPending
  const nothing = !pending && !spectrum.data && !time.data && !am.data
  return (
    <Paper sx={{ p: 2 }} component="section" aria-label={t('results.charts')}>
      <Typography variant="h3" component="h2" gutterBottom>
        {t('results.charts')}
      </Typography>
      <Typography variant="caption" color="text.secondary" component="p" gutterBottom>
        {t('results.charts.help')}
      </Typography>
      {pending && <LoadingState />}
      {nothing && <Typography variant="body2">{t('results.charts.none')}</Typography>}
      <Grid container spacing={2}>
        {spectrum.data && (
          <Grid size={{ xs: 12 }}>
            <SpectrumPlot frequencyHz={spectrum.data.frequency} axis={spectrum.data.axis} traces={spectrumTraces} bands={spectrum.data.bands ?? undefined} />
            <Typography variant="caption" color="text.secondary">
              {spectrum.data.estimator}
            </Typography>
          </Grid>
        )}
        {time.data && (
          <Grid size={{ xs: 12 }}>
            <IQPreview start={time.data.start} series={series} />
          </Grid>
        )}
        {am.data && <AmPlots data={am.data} />}
      </Grid>
    </Paper>
  )
}
