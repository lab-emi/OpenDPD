import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import Accordion from '@mui/material/Accordion'
import AccordionDetails from '@mui/material/AccordionDetails'
import AccordionSummary from '@mui/material/AccordionSummary'
import Alert from '@mui/material/Alert'
import Button from '@mui/material/Button'
import Chip from '@mui/material/Chip'
import Grid from '@mui/material/Grid'
import Link from '@mui/material/Link'
import MenuItem from '@mui/material/MenuItem'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import Table from '@mui/material/Table'
import TableBody from '@mui/material/TableBody'
import TableCell from '@mui/material/TableCell'
import TableHead from '@mui/material/TableHead'
import TableRow from '@mui/material/TableRow'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { useState } from 'react'
import { Link as RouterLink, useParams } from 'react-router'
import { API, artifactUrl } from '@/api/client'
import { useDeployExport, useExportRun, useMetricProfiles, useModels, useResult, useResultProfiles } from '@/api/hooks'
import { offeredProfiles } from '@/api/profiles'
import type { BaselineScore, DeploymentManifest, EvaluationResult, ExecutionEvidence, MetricProfile, MetricValue } from '@/api/types'
import { formatNumber, formatDateTime, t, type MessageKey } from '@/i18n'
import { EvidenceBadge } from '@/components/EvidenceBadge'
import { MetricCard } from '@/components/MetricCard'
import { ResultCharts } from '@/components/ResultCharts'
import { ErrorState, LoadingState } from '@/components/StateBlock'

const bytes = (n: number) => (n >= 1 << 20 ? `${(n / (1 << 20)).toFixed(1)} MB` : `${Math.max(1, Math.round(n / 1024))} kB`)

/** Package export (share / full) and report downloads for one run. */
function ExportPanel({ runId }: { runId: string }) {
  const exportRun = useExportRun()
  const info = exportRun.data
  return (
    <Paper sx={{ p: 2 }} component="section" aria-label={t('results.export')}>
      <Typography variant="h3" component="h2" gutterBottom>
        {t('results.export')}
      </Typography>
      <Typography variant="caption" color="text.secondary" component="p" gutterBottom>
        {t('results.export.help')}
      </Typography>
      <Stack direction="row" spacing={1} sx={{ flexWrap: 'wrap' }} useFlexGap>
        <Button variant="contained" size="small" disabled={exportRun.isPending} onClick={() => exportRun.mutate({ run_id: runId, kind: 'share' })}>
          {t('results.export.share')}
        </Button>
        <Button variant="outlined" size="small" disabled={exportRun.isPending} onClick={() => exportRun.mutate({ run_id: runId, kind: 'full' })}>
          {t('results.export.full')}
        </Button>
        <Button size="small" component="a" href={`${API}/results/${encodeURIComponent(runId)}/report?format=html`} download>
          {t('results.report.html')}
        </Button>
        <Button size="small" component="a" href={`${API}/results/${encodeURIComponent(runId)}/report?format=md`} download>
          {t('results.report.md')}
        </Button>
      </Stack>
      {exportRun.isError && <ErrorState error={exportRun.error} />}
      {info && (
        <Alert severity="success" sx={{ mt: 2 }} data-testid="export-ready">
          <strong>{t('results.export.ready', { filename: info.filename, size: bytes(info.size_bytes) })}</strong>{' '}
          <Link href={info.download_url} download={info.filename}>
            {t('results.export.download')}
          </Link>
          {(info.manifest.redaction ?? []).length > 0 && (
            <Typography variant="body2" component="div" sx={{ mt: 1 }}>
              <strong>{t('results.export.redaction')}:</strong>
              <ul style={{ margin: '4px 0 0', paddingLeft: 18 }}>
                {(info.manifest.redaction ?? []).map((line) => (
                  <li key={line}>{line}</li>
                ))}
              </ul>
            </Typography>
          )}
          {(info.manifest.missing ?? []).length > 0 && (
            <Typography variant="body2" component="div" sx={{ mt: 1 }}>
              <strong>{t('results.export.missing')}:</strong>
              <ul style={{ margin: '4px 0 0', paddingLeft: 18 }}>
                {(info.manifest.missing ?? []).map((line) => (
                  <li key={line}>{line}</li>
                ))}
              </ul>
            </Typography>
          )}
          <Typography variant="caption" component="p" sx={{ mt: 1 }}>
            <strong>{t('results.export.retraining')}:</strong> {info.manifest.retraining_note}
          </Typography>
        </Alert>
      )}
    </Paper>
  )
}

const BASELINE: Record<BaselineScore['kind'], MessageKey> = {
  surrogate_without_dpd: 'results.detail.baselines.surrogate_without_dpd',
  measured_without_dpd: 'results.detail.baselines.measured_without_dpd',
}

const fmt = (v: number | null | undefined, digits = 3) => (typeof v === 'number' ? v.toFixed(digits) : t('common.na'))
const score = (m: MetricValue | undefined) => (!m ? t('common.na') : m.status === 'ok' && typeof m.value === 'number' ? `${m.value.toFixed(2)} ${m.unit}` : (m.status ?? 'ok').replace('_', ' '))

/** x → u = DPD(x) → y = PA(u): every stage names its source; simulated stages are marked as such. */
function SignalChain({ result }: { result: EvaluationResult }) {
  const chain = result.signal_chain ?? []
  if (chain.length === 0) return null
  return (
    <Paper sx={{ p: 2 }} component="section" aria-label={t('results.detail.chain')}>
      <Typography variant="h3" component="h2" gutterBottom>
        {t('results.detail.chain')}
      </Typography>
      <Typography variant="caption" color="text.secondary" component="p" gutterBottom>
        {t('results.detail.chain.help')}
      </Typography>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>{t('results.detail.chain.stage')}</TableCell>
            <TableCell>{t('results.detail.chain.role')}</TableCell>
            <TableCell>{t('results.detail.chain.source')}</TableCell>
            <TableCell align="right">{t('results.detail.chain.samples')}</TableCell>
            <TableCell align="right">{t('results.detail.chain.peak')}</TableCell>
            <TableCell align="right">{t('results.detail.chain.rms')}</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {chain.map((s) => (
            <TableRow key={s.symbol} data-stage={s.symbol}>
              <TableCell>
                <code>{s.symbol}</code>
              </TableCell>
              <TableCell>{s.role}</TableCell>
              <TableCell>
                {s.source}
                {s.simulated && <Chip size="small" color="warning" variant="outlined" label={t('results.detail.chain.simulated')} sx={{ ml: 1 }} />}
                {s.artifact_id && result.run_id && (
                  <>
                    {' '}
                    <Link href={artifactUrl(result.run_id, s.artifact_id)} download>
                      {t('results.detail.chain.export')}
                    </Link>
                  </>
                )}
              </TableCell>
              <TableCell align="right">{s.n_samples ?? t('common.na')}</TableCell>
              <TableCell align="right">{fmt(s.peak_abs)}</TableCell>
              <TableCell align="right">{fmt(s.rms)}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Paper>
  )
}

/** dpd_measured: what the operator declared, how each capture was aligned, and the level difference (S16). */
/** fixed-point-v1 export (S19): offered when the registry lists the format for the evaluated model; otherwise the reason. */
function DeploymentPanel({ result }: { result: EvaluationResult }) {
  const models = useModels()
  const deploy = useDeployExport()
  const evaluated = (result.models ?? [])[0]
  const runId = evaluated?.run_id ?? result.run_id
  const descriptor = (models.data ?? []).find((m) => m.key === evaluated?.model.key)
  const supported = (descriptor?.export_formats ?? []).includes('fixed-point-v1')
  const info = deploy.data
  return (
    <Paper sx={{ p: 2 }} component="section" aria-label={t('results.deploy')} data-testid="deployment">
      <Typography variant="h3" component="h2" gutterBottom>
        {t('results.deploy')}
      </Typography>
      <Typography variant="caption" color="text.secondary" component="p" gutterBottom>
        {t('results.deploy.help')}
      </Typography>
      {!supported ? (
        <Alert severity="info" data-testid="deploy-unsupported">
          {t('results.deploy.unsupported', { model: evaluated?.model.key ?? '?', supported: (models.data ?? []).filter((m) => m.export_formats.includes('fixed-point-v1') && !m.weights_from).map((m) => m.key).join(', ') || 'gru' })}
        </Alert>
      ) : (
        <Button variant="contained" size="small" disabled={deploy.isPending || !runId} onClick={() => runId && deploy.mutate({ run_id: runId })}>
          {deploy.isPending ? t('results.deploy.building') : t('results.deploy.export')}
        </Button>
      )}
      {deploy.isError && <ErrorState error={deploy.error} />}
      {info && <DeploymentSummary manifest={info.manifest} filename={info.filename} downloadUrl={info.download_url} />}
    </Paper>
  )
}

function DeploymentSummary({ manifest, filename, downloadUrl }: { manifest: DeploymentManifest; filename: string; downloadUrl: string }) {
  const v = manifest.verification
  const r = manifest.report
  const res = r.resources
  const severity = v.status === 'bit_exact' ? 'success' : v.status === 'mismatch' ? 'error' : 'warning'
  return (
    <Stack spacing={1} sx={{ mt: 2 }} data-testid="deploy-ready">
      <Alert severity={severity} data-testid="deploy-verification">
        <strong>{t(`results.deploy.verification.${v.status}` as MessageKey, { backend: v.backend, cases: v.cases_checked })}</strong>{' '}
        {v.status === 'mismatch' ? t('results.deploy.mismatch', { case: v.mismatch_case ?? '?', step: v.mismatch_step ?? '?', signal: v.mismatch_signal ?? '?' }) : (v.detail ?? '')}
      </Alert>
      <Typography variant="body2">
        <Link href={downloadUrl} download={filename}>
          {t('results.deploy.download', { filename })}
        </Link>
      </Typography>
      <Table size="small" aria-label={t('results.deploy.loss')}>
        <TableHead>
          <TableRow>
            <TableCell>{t('results.deploy.loss.metric')}</TableCell>
            <TableCell align="right">{t('results.deploy.loss.float')}</TableCell>
            <TableCell align="right">{t('results.deploy.loss.fixed')}</TableCell>
            <TableCell align="right">{t('results.deploy.loss.delta')}</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {r.quality_loss.map((d) => (
            <TableRow key={d.name}>
              <TableCell>
                {d.name} ({d.unit})
              </TableCell>
              <TableCell align="right">{d.float_value?.toFixed(2) ?? t('common.na')}</TableCell>
              <TableCell align="right">{d.fixed_value?.toFixed(2) ?? t('common.na')}</TableCell>
              <TableCell align="right">{d.delta === null || d.delta === undefined ? t('common.na') : `${d.delta >= 0 ? '+' : ''}${d.delta.toFixed(2)}`}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <dl style={{ margin: 0, display: 'grid', gridTemplateColumns: 'max-content 1fr', columnGap: 16, rowGap: 4 }} data-testid="deploy-resources">
        <dt style={{ color: '#4B5563' }}>{t('results.deploy.label.theoretical')}</dt>
        <dd style={{ margin: 0 }}>{t('results.deploy.resources', { mac: res.mac_per_sample, lookups: res.table_lookups_per_sample, weights: res.weight_bytes, state: res.state_bytes, tables: res.table_bytes })}</dd>
        <dt style={{ color: '#4B5563' }}>{t('results.deploy.label.measured')}</dt>
        <dd style={{ margin: 0 }}>{r.measured_execution ? t('results.deploy.measured', { rate: formatNumber(Math.round(r.measured_execution.samples_per_second)), what: r.measured_execution.what }) : t('results.deploy.notAvailable')}</dd>
        <dt style={{ color: '#4B5563' }}>{t('results.deploy.label.synthesis')}</dt>
        <dd style={{ margin: 0 }}>{r.synthesis_estimate ?? t('results.deploy.notSynthesised')}</dd>
        <dt style={{ color: '#4B5563' }}>{t('results.deploy.label.power')}</dt>
        <dd style={{ margin: 0 }}>{r.measured_power ?? t('results.deploy.notMeasuredPower')}</dd>
      </dl>
    </Stack>
  )
}

/** How a streaming variant consumed the signal (S18): chunking, look-ahead as samples and time, warm-up, consistency. */
function ExecutionPanel({ result }: { result: EvaluationResult }) {
  const e: ExecutionEvidence | null | undefined = result.execution
  if (!e) return null
  const micro = typeof e.lookahead_s === 'number' ? `${formatNumber(e.lookahead_s * 1e6, { maximumFractionDigits: 4 })} µs` : t('common.na')
  const facts: Array<[string, string]> = [
    [t('results.detail.execution.semantics'), `${e.semantics} · ${t(`results.detail.execution.state.${e.state}` as MessageKey)}`],
    [t('results.detail.execution.chunk'), t('results.detail.execution.samples', { n: e.chunk_samples })],
    [t('results.detail.execution.lookahead'), `${t('results.detail.execution.samples', { n: e.lookahead_samples })} · ${micro}`],
    [t('results.detail.execution.history'), typeof e.history_samples === 'number' ? t('results.detail.execution.samples', { n: e.history_samples }) : t('common.na')],
    [t('results.detail.execution.warmup'), typeof e.warmup_samples === 'number' ? t('results.detail.execution.samples', { n: e.warmup_samples }) : t('results.detail.execution.notMeasured')],
  ]
  const c = e.consistency
  return (
    <Paper sx={{ p: 2 }} component="section" aria-label={t('results.detail.execution')} data-testid="execution">
      <Typography variant="h3" component="h2" gutterBottom>
        {t('results.detail.execution')}
      </Typography>
      <Alert severity={c.within_tolerance ? 'success' : 'error'} sx={{ mb: 2 }} data-testid="chunk-consistency">
        {t(c.within_tolerance ? 'results.detail.execution.consistent' : 'results.detail.execution.inconsistent', { chunk: c.chunk_samples, error: c.max_abs_error.toExponential(2), tolerance: c.tolerance.toExponential(0) })}
      </Alert>
      <dl style={{ margin: 0, display: 'grid', gridTemplateColumns: 'max-content 1fr', columnGap: 16, rowGap: 4 }}>
        {facts.map(([k, v]) => (
          <div key={k} style={{ display: 'contents' }}>
            <dt style={{ color: '#4B5563' }}>{k}</dt>
            <dd style={{ margin: 0 }}>{v}</dd>
          </div>
        ))}
      </dl>
      <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>
        {t('results.detail.execution.note')}
      </Typography>
    </Paper>
  )
}

function MeasurementPanel({ result }: { result: EvaluationResult }) {
  const m = result.measurement
  if (!m) return null
  const c = m.conditions
  const facts: Array<[string, string]> = [
    [t('results.detail.measurement.pa'), c.pa],
    [t('results.detail.measurement.chain'), c.capture_chain],
    [t('results.detail.measurement.rate'), `${formatNumber(c.sample_rate_hz / 1e6, { maximumFractionDigits: 3 })} MS/s`],
    [t('results.detail.measurement.drive'), c.drive],
    [t('results.detail.measurement.gain'), typeof c.gain_db === 'number' ? `${c.gain_db} dB` : t('common.na')],
    [t('results.detail.measurement.calibration'), c.calibration],
    [t('results.detail.measurement.measured_at'), formatDateTime(c.measured_at)],
    [t('results.detail.measurement.temperature'), typeof c.temperature_c === 'number' ? `${c.temperature_c} °C` : t('common.na')],
    [t('results.detail.measurement.operator'), c.operator ?? t('common.na')],
    [t('results.detail.measurement.played'), `${m.apply_run_id} · ${m.played_sha256.slice(0, 12)}`],
  ]
  const level = m.level_difference_db
  return (
    <Paper sx={{ p: 2 }} component="section" aria-label={t('results.detail.measurement')} data-testid="measurement">
      <Typography variant="h3" component="h2" gutterBottom>
        {t('results.detail.measurement')}
      </Typography>
      <Alert severity={result.is_mock ? 'error' : 'warning'} sx={{ mb: 2 }} data-testid="attestation">
        {m.attestation}
      </Alert>
      <dl style={{ margin: 0, display: 'grid', gridTemplateColumns: 'max-content 1fr', columnGap: 16, rowGap: 4 }}>
        {facts.map(([k, v]) => (
          <div key={k} style={{ display: 'contents' }}>
            <dt style={{ color: '#4B5563' }}>{k}</dt>
            <dd style={{ margin: 0 }}>{v}</dd>
          </div>
        ))}
      </dl>
      {c.notes && (
        <Typography variant="body2" sx={{ mt: 1 }}>
          {c.notes}
        </Typography>
      )}
      <Table size="small" sx={{ mt: 2 }} aria-label={t('results.detail.measurement.captures')}>
        <TableHead>
          <TableRow>
            <TableCell>{t('results.detail.measurement.capture')}</TableCell>
            <TableCell>{t('results.detail.measurement.file')}</TableCell>
            <TableCell align="right">{t('results.detail.chain.samples')}</TableCell>
            <TableCell align="right">{t('results.detail.measurement.delay')}</TableCell>
            <TableCell align="right">{t('results.detail.measurement.correlation')}</TableCell>
            <TableCell align="right">{t('results.detail.measurement.fit')}</TableCell>
            <TableCell align="right">{t('results.detail.chain.rms')}</TableCell>
            <TableCell align="right">{t('results.detail.chain.peak')}</TableCell>
            <TableCell align="right">{t('results.detail.measurement.power')}</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {m.captures.map((cap) => (
            <TableRow key={cap.role} data-capture={cap.role}>
              <TableCell>{t(cap.role === 'with_dpd' ? 'results.detail.measurement.with' : 'results.detail.measurement.without')}</TableCell>
              <TableCell>
                {result.run_id ? (
                  <Link href={artifactUrl(result.run_id, cap.artifact_id)} download>
                    {cap.artifact_id}
                  </Link>
                ) : (
                  cap.artifact_id
                )}{' '}
                <code>{cap.raw_sha256.slice(0, 12)}</code>
                {cap.resample_ratio ? ` · ×${cap.resample_ratio[0]}/${cap.resample_ratio[1]}` : ''}
              </TableCell>
              <TableCell align="right">{formatNumber(cap.n_samples_raw)}</TableCell>
              <TableCell align="right">
                {cap.delay_samples}
                {cap.wrapped ? ` (${t('results.detail.measurement.wrapped')})` : ''}
              </TableCell>
              <TableCell align="right">{cap.correlation.toFixed(4)}</TableCell>
              <TableCell align="right">
                {cap.gain_db.toFixed(2)} dB ∠ {cap.gain_phase_deg.toFixed(1)}°
              </TableCell>
              <TableCell align="right">{fmt(cap.rms, 4)}</TableCell>
              <TableCell align="right">{fmt(cap.peak_abs, 4)}</TableCell>
              <TableCell align="right">{typeof cap.declared_output_power_dbm === 'number' ? `${cap.declared_output_power_dbm} dBm` : t('common.na')}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
      {typeof level === 'number' && (
        <Typography variant="body2" sx={{ mt: 1 }} color={Math.abs(level) > 0.5 ? 'warning.main' : 'text.secondary'} data-testid="level-difference">
          {t('results.detail.measurement.level', { db: `${level >= 0 ? '+' : ''}${level.toFixed(2)}` })}
          {Math.abs(level) > 0.5 ? ` ${t('results.detail.measurement.level.warn')}` : ''}
          {typeof m.declared_power_difference_db === 'number' ? ` ${t('results.detail.measurement.level.declared', { db: `${m.declared_power_difference_db >= 0 ? '+' : ''}${m.declared_power_difference_db.toFixed(2)}` })}` : ''}
        </Typography>
      )}
    </Paper>
  )
}

/** The DPD result next to the no-DPD baselines, all scored against the same reference. */
function Baselines({ result }: { result: EvaluationResult }) {
  const baselines = result.baselines ?? []
  if (baselines.length === 0) return null
  return (
    <Paper sx={{ p: 2 }} component="section" aria-label={t('results.detail.baselines')}>
      <Typography variant="h3" component="h2" gutterBottom>
        {t('results.detail.baselines')}
      </Typography>
      <Typography variant="caption" color="text.secondary" component="p" gutterBottom>
        {t('results.detail.baselines.help')}
      </Typography>
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>{t('results.detail.metric')}</TableCell>
            <TableCell align="right">{t('results.detail.baselines.dpd')}</TableCell>
            {baselines.map((b) => (
              <TableCell key={b.kind} align="right">
                {t(BASELINE[b.kind])}
              </TableCell>
            ))}
          </TableRow>
        </TableHead>
        <TableBody>
          {result.metrics.map((m) => (
            <TableRow key={m.name}>
              <TableCell>
                <code>{m.name}</code>
              </TableCell>
              <TableCell align="right">{score(m)}</TableCell>
              {baselines.map((b) => (
                <TableCell key={b.kind} align="right">
                  {score(b.metrics.find((x) => x.name === m.name))}
                </TableCell>
              ))}
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Paper>
  )
}

/** Result view: evidence first, then metrics with their registry definitions, then how they were produced. */
export function ResultView({ result, profile, stored = [], onProfile }: { result: EvaluationResult; profile?: MetricProfile; stored?: string[]; onProfile?: (id: string) => void }) {
  const definitions = new Map((profile?.metrics ?? []).map((m) => [m.name, m]))
  return (
    <Stack spacing={2}>
      <Stack sx={{ alignItems: 'center', flexWrap: 'wrap' }} direction="row" spacing={2} useFlexGap>
        <Typography variant="h1">{result.result_id}</Typography>
        <EvidenceBadge evidence={result.evidence_type} mock={result.is_mock} />
        <Chip size="small" variant="outlined" label={`${result.metric_profile_id} v${result.metric_profile_version}${profile?.frozen ? ` · ${t('results.detail.frozen')}` : ''}`} data-profile={result.metric_profile_id} />
        <Typography variant="body2" color="text.secondary">
          {t('results.columns.run')}:{' '}
          <Link component={RouterLink} to={`/runs/${encodeURIComponent(result.run_id ?? '')}`}>
            {result.run_id}
          </Link>
        </Typography>
        {stored.length > 1 && onProfile && (
          <TextField select size="small" label={t('results.detail.profile')} value={result.metric_profile_id} onChange={(e) => onProfile(e.target.value)} helperText={t('results.detail.profile.help')} sx={{ minWidth: 220, ml: 'auto' }}>
            {stored.map((id) => (
              <MenuItem key={id} value={id}>
                {id}
              </MenuItem>
            ))}
          </TextField>
        )}
      </Stack>
      <Typography variant="body2" color="text.secondary">
        {t('results.detail.protocol')}:{' '}
        {t('results.detail.protocol.body', {
          profile: result.metric_profile_id,
          version: result.metric_profile_version,
          segments: result.n_segments ?? t('common.na'),
          nperseg: result.nperseg ?? t('common.na'),
          epoch: result.selected_epoch ?? t('common.na'),
          device: result.device,
          numeric: result.numeric_mode ?? t('common.na'),
        })}
      </Typography>
      <Grid container spacing={2}>
        {result.metrics.map((m) => (
          <Grid key={m.name} size={{ xs: 6, sm: 4, md: 2.4 }}>
            <MetricCard metric={m} definition={definitions.get(m.name)} />
          </Grid>
        ))}
      </Grid>
      <SignalChain result={result} />
      <ExecutionPanel result={result} />
      <DeploymentPanel result={result} />
      <MeasurementPanel result={result} />
      <Baselines result={result} />
      {result.surrogate_coverage && (
        <Alert severity={result.surrogate_coverage.fraction_above_fitted_peak > 0 ? 'warning' : 'info'} data-testid="surrogate-coverage">
          <strong>{t('results.detail.coverage')}</strong>{' '}
          {t('results.detail.coverage.body', {
            fitted: fmt(result.surrogate_coverage.fitted_peak_abs),
            peak: fmt(result.surrogate_coverage.u_peak_abs),
            fraction: `${(100 * result.surrogate_coverage.fraction_above_fitted_peak).toFixed(2)}%`,
          })}{' '}
          {result.surrogate_coverage.note}
        </Alert>
      )}
      {result.scaling && (
        <Typography variant="body2" color="text.secondary" data-testid="scaling">
          <strong>{t('results.detail.scaling')}</strong>{' '}
          {t('results.detail.scaling.body', { units: result.scaling.amplitude_units, scaling: result.scaling.input_scaling, gain: fmt(result.scaling.reference_gain, 4) })}{' '}
          {!result.scaling.physical_calibration && t('results.detail.scaling.uncalibrated')}
        </Typography>
      )}
      {profile && (
        <Accordion disableGutters>
          <AccordionSummary expandIcon={<ExpandMoreIcon />} aria-controls="definitions-panel" id="definitions-header">
            <Typography>{t('results.detail.definitions')}</Typography>
          </AccordionSummary>
          <AccordionDetails id="definitions-panel">
            <Typography variant="body2" color="text.secondary" gutterBottom>
              {t('results.detail.definitions.help', { profile: profile.profile_id, version: profile.version })} {profile.description}
            </Typography>
            <Table size="small" aria-label={t('results.detail.definitions')}>
              <TableHead>
                <TableRow>
                  <TableCell>{t('results.detail.metric')}</TableCell>
                  <TableCell>{t('metric.formula')}</TableCell>
                  <TableCell>{t('metric.aggregation')}</TableCell>
                  <TableCell>{t('results.detail.unit')}</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {profile.metrics.map((m) => (
                  <TableRow key={m.name}>
                    <TableCell>
                      <code>{m.name}</code> {m.display_name}
                    </TableCell>
                    <TableCell>
                      {m.formula}
                      {m.notes ? (
                        <Typography variant="caption" color="text.secondary" component="div">
                          {m.notes}
                        </Typography>
                      ) : null}
                    </TableCell>
                    <TableCell>{m.aggregation}</TableCell>
                    <TableCell>
                      {m.unit} · {m.better === 'lower' ? t('metric.lowerBetter') : t('metric.higherBetter')}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            <Typography variant="h3" component="h3" sx={{ mt: 2 }} gutterBottom>
              {t('results.detail.parameters')}
            </Typography>
            <dl style={{ margin: 0, display: 'grid', gridTemplateColumns: 'max-content 1fr', columnGap: 16, rowGap: 4 }}>
              {Object.entries(profile.parameters ?? {}).map(([k, v]) => (
                <div key={k} style={{ display: 'contents' }}>
                  <dt style={{ color: '#4B5563' }}>{k}</dt>
                  <dd style={{ margin: 0 }}>{typeof v === 'string' ? v : JSON.stringify(v)}</dd>
                </div>
              ))}
            </dl>
          </AccordionDetails>
        </Accordion>
      )}
      <Grid container spacing={2}>
        <Grid size={{ xs: 12, md: 6 }}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h3" component="h2" gutterBottom>
              {t('results.detail.reference')}
            </Typography>
            <Typography>
              <code>{result.reference.kind}</code> — {result.reference.description}
              {result.reference.gain_rule ? ` (${result.reference.gain_rule}${typeof result.reference.gain_value === 'number' ? ` = ${result.reference.gain_value}` : ''})` : ''}
            </Typography>
          </Paper>
        </Grid>
        <Grid size={{ xs: 12, md: 6 }}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h3" component="h2" gutterBottom>
              {t('results.detail.models')}
            </Typography>
            {(result.models ?? []).map((m) => (
              <Typography key={`${m.role}-${m.run_id}`} variant="body2">
                <strong>{m.role}</strong> {m.model.key} {JSON.stringify(m.model.parameters)} · {m.n_parameters ?? '?'} params · {m.execution_semantics}{m.training_path ? ` · ${m.training_path.replace(/_/g, ' ')}` : ''}
                {typeof m.lookahead_samples === 'number' ? ` · look-ahead ${m.lookahead_samples}` : ''}
              </Typography>
            ))}
          </Paper>
        </Grid>
      </Grid>
      {(result.limitations ?? []).length > 0 && (
        <Alert severity={result.is_mock ? 'warning' : 'info'}>
          <strong>{t('results.detail.limitations')}</strong>
          <ul style={{ margin: '4px 0 0', paddingLeft: 18 }}>
            {(result.limitations ?? []).map((l) => (
              <li key={l}>{l}</li>
            ))}
          </ul>
        </Alert>
      )}
      {result.run_id && <ResultCharts runId={result.run_id} />}
      {result.run_id && !result.is_mock && <ExportPanel runId={result.run_id} />}
    </Stack>
  )
}

export function ResultDetailPage() {
  const { runId = '' } = useParams()
  const [profileId, setProfileId] = useState<string | null>(null)
  const result = useResult(runId, true, profileId)
  const stored = useResultProfiles(runId)
  const profiles = useMetricProfiles()
  if (result.isPending) return <LoadingState />
  if (result.isError) return <ErrorState error={result.error} onRetry={() => void result.refetch()} />
  const profile = profiles.data?.find((p) => p.profile_id === result.data.metric_profile_id)
  // stored results under a profile the GUI does not offer yet stay reachable through the CLI and the API only
  const offered = new Set(offeredProfiles(profiles.data).map((p) => p.profile_id))
  const visible = (stored.data ?? []).filter((id) => offered.has(id) || id === result.data.metric_profile_id)
  return <ResultView result={result.data} profile={profile} stored={visible} onProfile={setProfileId} />
}
