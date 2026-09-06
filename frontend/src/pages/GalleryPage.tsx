import Alert from '@mui/material/Alert'
import Divider from '@mui/material/Divider'
import Grid from '@mui/material/Grid'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import { useCallback, useMemo, useState } from 'react'
import type { DiagnosticReport, EvaluationResult, MetricValue, ResolvedExperimentConfig, RunEvent, RunStatus, RunView } from '@/api/types'
import { reduceEvent, type StreamState } from '@/api/events'
import { t } from '@/i18n'
import { ConfigDiff } from '@/components/ConfigDiff'
import { DiagnosticItem } from '@/components/DiagnosticItem'
import { EvidenceBadge } from '@/components/EvidenceBadge'
import { IQPreview } from '@/components/IQPreview'
import { MetricCard } from '@/components/MetricCard'
import { MetricHistoryChart } from '@/components/MetricHistoryChart'
import { RunTimeline } from '@/components/RunTimeline'
import { SpectrumPlot } from '@/components/SpectrumPlot'
import { StatusChip } from '@/components/StatusChip'
import { DisconnectedState, EmptyState, ErrorState, LoadingState } from '@/components/StateBlock'
import { ResultView } from '@/pages/ResultDetailPage'
import doctorMock from '@mocks/diagnostics_missing_metadata.json'
import eventsMock from '@mocks/events_running.json'
import resolvedMock from '@mocks/resolved_train_pa_smoke.json'
import naMock from '@mocks/result_metric_not_applicable_mock.json'
import resultMock from '@mocks/result_pa_modeling_mock.json'
import runningMock from '@mocks/run_running.json'
import failedMock from '@mocks/run_failed.json'

/**
 * Real-size synthetic signal for the chart performance probe: the server will
 * send data of this shape (PSD bins per segment, a short I/Q window), never
 * whole captures. Deterministic, no science claimed.
 */
function syntheticSpectrum(nperseg = 2560, fs = 800e6) {
  const f = new Float64Array(nperseg)
  const input = new Float64Array(nperseg)
  const output = new Float64Array(nperseg)
  const dpd = new Float64Array(nperseg)
  for (let k = 0; k < nperseg; k++) {
    const freq = (k / nperseg - 0.5) * fs
    f[k] = freq
    const inBand = Math.abs(freq) < 100e6
    const shoulder = Math.exp(-Math.abs(freq) / 150e6)
    const noise = 3 * Math.sin(k * 12.9898) * Math.cos(k * 78.233)
    input[k] = (inBand ? -20 : -95) + noise * 0.3
    output[k] = (inBand ? -20 : -70 + 30 * shoulder - 25) + noise * 0.4
    dpd[k] = (inBand ? -20 : -85 + 15 * shoulder - 15) + noise * 0.4
  }
  return { f, traces: [{ name: 'input', psdDb: input }, { name: 'PA output', psdDb: output }, { name: 'with DPD', psdDb: dpd }] }
}

function syntheticIQ(n = 20_000) {
  const i = new Float64Array(n)
  const q = new Float64Array(n)
  const oi = new Float64Array(n)
  const oq = new Float64Array(n)
  for (let k = 0; k < n; k++) {
    const env = 0.5 + 0.5 * Math.sin(k / 97)
    i[k] = env * Math.cos(k / 7.3)
    q[k] = env * Math.sin(k / 7.3)
    const comp = 1 - 0.15 * env * env
    oi[k] = 1.4 * comp * i[k]!
    oq[k] = 1.4 * comp * q[k]!
  }
  return [
    { name: 'input', i, q },
    { name: 'output', i: oi, q: oq },
  ]
}

const STATUSES: RunStatus[] = ['queued', 'running', 'cancel_requested', 'cancelled', 'succeeded', 'failed', 'interrupted']

export function GalleryPage() {
  const [perf, setPerf] = useState<{ spectrum?: number; iq?: number }>({})
  const spectrum = useMemo(() => syntheticSpectrum(), [])
  const iq = useMemo(() => syntheticIQ(), [])
  const onSpectrum = useCallback((ms: number) => setPerf((p) => ({ ...p, spectrum: ms })), [])
  const onIQ = useCallback((ms: number) => setPerf((p) => ({ ...p, iq: ms })), [])
  const stream = useMemo(() => (eventsMock.data as unknown as RunEvent[]).reduce<StreamState>(reduceEvent, { connection: 'ended', lastSeq: 0, lastUpdate: null, progress: null, metrics: [], statusEvents: [], heartbeats: 0, lastError: null }), [])
  const running = runningMock.data as unknown as RunView
  const failed = failedMock.data as unknown as RunView
  const result = resultMock.data as unknown as EvaluationResult
  const na = naMock.data as unknown as EvaluationResult
  const resolved = resolvedMock.data as unknown as ResolvedExperimentConfig
  const changed = { ...resolved, training: { ...resolved.training, epochs: 300, learning_rate: 1e-3 }, model: { ...resolved.model, parameters: { ...resolved.model.parameters, hidden_size: 32 } } }
  const doctor = doctorMock.data as unknown as DiagnosticReport
  const ready = perf.spectrum !== undefined && perf.iq !== undefined

  return (
    <Stack spacing={3} data-testid="gallery">
      <Typography variant="h1">{t('gallery.title')}</Typography>
      <Alert severity="warning">{t('gallery.body')}</Alert>
      {ready && (
        <Alert severity="info" data-testid="perf-probe" data-spectrum-ms={perf.spectrum?.toFixed(0)} data-iq-ms={perf.iq?.toFixed(0)}>
          {t('gallery.perf', { psd: spectrum.f.length, traces: spectrum.traces.length, iq: iq[0]!.i.length, ms: Math.round((perf.spectrum ?? 0) + (perf.iq ?? 0)) })}
        </Alert>
      )}
      <Section title="StatusChip">
        <Stack sx={{ flexWrap: 'wrap' }} direction="row" spacing={1} useFlexGap>
          {STATUSES.map((s) => (
            <StatusChip key={s} status={s} />
          ))}
          <StatusChip status="running" stale />
        </Stack>
      </Section>
      <Section title="EvidenceBadge">
        <Stack sx={{ flexWrap: 'wrap' }} direction="row" spacing={1} useFlexGap>
          <EvidenceBadge evidence="pa_modeling" />
          <EvidenceBadge evidence="dpd_surrogate" />
          <EvidenceBadge evidence="dpd_measured" />
          <EvidenceBadge evidence="pa_modeling" mock />
        </Stack>
      </Section>
      <Section title="MetricCard">
        <Grid container spacing={2}>
          {[...result.metrics, ...na.metrics.filter((m) => m.status !== 'ok')].map((m: MetricValue, i) => (
            <Grid key={`${m.name}-${i}`} size={{ xs: 6, md: 2 }}>
              <MetricCard metric={m} />
            </Grid>
          ))}
        </Grid>
      </Section>
      <Section title="SpectrumPlot / IQPreview">
        <Grid container spacing={2}>
          <Grid size={{ xs: 12, lg: 6 }}>
            <SpectrumPlot frequencyHz={spectrum.f} traces={spectrum.traces} bands={{ main: [-100e6, 100e6], adjacent: [[-300e6, -100e6], [100e6, 300e6]] }} onRendered={onSpectrum} />
          </Grid>
          <Grid size={{ xs: 12, lg: 6 }}>
            <IQPreview start={0} series={iq} onRendered={onIQ} />
          </Grid>
          <Grid size={{ xs: 12, lg: 6 }}>
            <MetricHistoryChart points={stream.metrics} metric="NMSE" />
          </Grid>
        </Grid>
      </Section>
      <Section title="DiagnosticItem">
        <Stack spacing={1}>
          {(doctor.items ?? []).map((item) => (
            <DiagnosticItem key={item.code} item={item} />
          ))}
        </Stack>
      </Section>
      <Section title="RunTimeline">
        <RunTimeline run={running} statusEvents={stream.statusEvents} heartbeats={stream.heartbeats} />
      </Section>
      <Section title="ConfigDiff">
        <ConfigDiff left={resolved} right={changed} leftLabel="smoke" rightLabel="research" />
      </Section>
      <Section title="States">
        <Stack spacing={1}>
          <LoadingState />
          <EmptyState body={t('experiments.empty')} />
          <ErrorState error={new Error(failed.error?.message ?? 'worker died')} onRetry={() => undefined} />
          <DisconnectedState lastUpdate={new Date(running.last_heartbeat_at ?? running.created_at)} onRefresh={() => undefined} />
        </Stack>
      </Section>
      <Section title="ResultView (mock)">
        <ResultView result={result} />
      </Section>
    </Stack>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section aria-label={title}>
      <Divider textAlign="left" sx={{ mb: 2 }}>
        <Typography variant="h2" component="h2">
          {title}
        </Typography>
      </Divider>
      {children}
    </section>
  )
}
