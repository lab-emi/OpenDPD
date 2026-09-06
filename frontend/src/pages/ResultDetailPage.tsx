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
import { useExportRun, useMetricProfiles, useResult, useResultProfiles } from '@/api/hooks'
import type { BaselineScore, EvaluationResult, MetricProfile, MetricValue } from '@/api/types'
import { t, type MessageKey } from '@/i18n'
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
                <strong>{m.role}</strong> {m.model.key} {JSON.stringify(m.model.parameters)} · {m.n_parameters ?? '?'} params · {m.execution_semantics}
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
  return <ResultView result={result.data} profile={profile} stored={stored.data ?? []} onProfile={setProfileId} />
}
