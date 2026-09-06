import Alert from '@mui/material/Alert'
import AlertTitle from '@mui/material/AlertTitle'
import Button from '@mui/material/Button'
import Chip from '@mui/material/Chip'
import Dialog from '@mui/material/Dialog'
import DialogActions from '@mui/material/DialogActions'
import DialogContent from '@mui/material/DialogContent'
import DialogTitle from '@mui/material/DialogTitle'
import Grid from '@mui/material/Grid'
import Link from '@mui/material/Link'
import LinearProgress from '@mui/material/LinearProgress'
import MenuItem from '@mui/material/MenuItem'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import Tab from '@mui/material/Tab'
import Tabs from '@mui/material/Tabs'
import Table from '@mui/material/Table'
import TableBody from '@mui/material/TableBody'
import TableCell from '@mui/material/TableCell'
import TableHead from '@mui/material/TableHead'
import TableRow from '@mui/material/TableRow'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { useQueryClient } from '@tanstack/react-query'
import { useRef, useState } from 'react'
import { Link as RouterLink, useNavigate, useParams, useSearchParams } from 'react-router'
import { artifactUrl } from '@/api/client'
import { useRunStream } from '@/api/events'
import { keys, useCancelRun, useRetryRun, useRun, useRunArtifacts, useRunConfig, useRunLineage, useRuns, useSubmitRun } from '@/api/hooks'
import { isTerminal, type LineageLink, type LineageRelation, type RunView } from '@/api/types'
import { t, type MessageKey } from '@/i18n'
import { LogViewer } from '@/components/LogViewer'
import { MetricHistoryChart } from '@/components/MetricHistoryChart'
import { RunTimeline } from '@/components/RunTimeline'
import { StatusChip } from '@/components/StatusChip'
import { DisconnectedState, EmptyState, ErrorState, LoadingState } from '@/components/StateBlock'

const TABS = ['overview', 'logs', 'artifacts', 'config'] as const
type TabKey = (typeof TABS)[number]

const RELATION: Record<LineageRelation, MessageKey> = {
  pa_surrogate: 'run.lineage.relation.pa_surrogate',
  dpd_model: 'run.lineage.relation.dpd_model',
  retry_of: 'run.lineage.relation.retry_of',
}

const NEXT_STEP: Partial<Record<RunView['status'], MessageKey>> = {
  failed: 'run.next.failed',
  interrupted: 'run.next.interrupted',
  cancelled: 'run.next.cancelled',
  succeeded: 'run.next.succeeded',
  queued: 'run.next.queued',
}

export function RunDetailPage() {
  const { runId = '' } = useParams()
  const [params, setParams] = useSearchParams()
  const tab: TabKey = TABS.includes(params.get('tab') as TabKey) ? (params.get('tab') as TabKey) : 'overview'
  const qc = useQueryClient()
  const run = useRun(runId)
  const active = !!run.data && !isTerminal(run.data.status)
  const stream = useRunStream(runId, !!run.data)
  const cancel = useCancelRun()
  const retry = useRetryRun()
  const [confirmCancel, setConfirmCancel] = useState(false)
  const [applyOpen, setApplyOpen] = useState(false)

  if (run.isPending) return <LoadingState />
  if (run.isError) return <ErrorState error={run.error} onRetry={() => void run.refetch()} />
  const r = run.data
  const progress = stream.progress ?? (typeof r.progress_epoch === 'number' && typeof r.progress_total_epochs === 'number' ? { epoch: r.progress_epoch, total: r.progress_total_epochs } : null)
  const disconnected = active && stream.connection === 'disconnected'
  const nextStep = NEXT_STEP[r.status]

  return (
    <Stack spacing={2}>
      <Stack sx={{ alignItems: 'center', flexWrap: 'wrap' }} direction="row" spacing={2} useFlexGap>
        <Typography variant="h1" sx={{ wordBreak: 'break-all' }}>
          {r.name || r.run_id}
        </Typography>
        <StatusChip status={r.status} stale={r.heartbeat_stale} size="medium" />
        {active && <Chip size="small" variant="outlined" label={stream.connection === 'live' ? t('run.live') : stream.connection === 'ended' ? t('run.ended') : t('run.connecting')} data-testid="stream-state" />}
        <Stack direction="row" spacing={1} sx={{ ml: 'auto' }}>
          {active &&
            (confirmCancel ? (
              <>
                <Typography variant="body2" sx={{ alignSelf: 'center' }}>
                  {t('run.cancel.confirm')}
                </Typography>
                <Button color="error" variant="contained" size="small" onClick={() => cancel.mutate(runId, { onSettled: () => setConfirmCancel(false) })} disabled={cancel.isPending}>
                  {t('run.cancel')}
                </Button>
                <Button size="small" onClick={() => setConfirmCancel(false)}>
                  {t('common.back')}
                </Button>
              </>
            ) : (
              <Button color="error" variant="outlined" size="small" onClick={() => setConfirmCancel(true)} disabled={r.status === 'cancel_requested'}>
                {t('run.cancel')}
              </Button>
            ))}
          {isTerminal(r.status) && r.status !== 'succeeded' && (
            <Button variant="outlined" size="small" onClick={() => retry.mutate(runId)} disabled={retry.isPending}>
              {t('run.retry')}
            </Button>
          )}
          {r.status === 'succeeded' && r.task === 'train_dpd' && (
            <Button variant="outlined" size="small" onClick={() => setApplyOpen(true)}>
              {t('run.apply')}
            </Button>
          )}
          {r.status === 'succeeded' && r.result_id && (
            <Button component={RouterLink} to={`/results/${encodeURIComponent(r.run_id)}`} variant="contained" size="small">
              {t('run.result')}
            </Button>
          )}
        </Stack>
      </Stack>
      {applyOpen && <ApplyDpdDialog run={r} onClose={() => setApplyOpen(false)} />}
      <Typography variant="body2" color="text.secondary">
        <code>{r.run_id}</code> · {r.task} · {r.model_key} · {r.dataset_id} · {r.device}
        {r.parent_run_id && (
          <>
            {' '}
            · {t('run.parent')}{' '}
            <Link component={RouterLink} to={`/runs/${encodeURIComponent(r.parent_run_id)}`}>
              {r.parent_run_id}
            </Link>
          </>
        )}
      </Typography>
      {retry.data && (
        <Alert severity="success">
          <Link component={RouterLink} to={`/runs/${encodeURIComponent(retry.data.run_id)}`}>
            {retry.data.run_id}
          </Link>
        </Alert>
      )}
      {cancel.isError && <ErrorState error={cancel.error} />}
      {retry.isError && <ErrorState error={retry.error} />}
      {disconnected && <DisconnectedState lastUpdate={stream.lastUpdate ?? (r.last_heartbeat_at ? new Date(r.last_heartbeat_at) : null)} onRefresh={() => void qc.invalidateQueries({ queryKey: keys.run(runId) })} />}
      {r.error && (
        <Alert severity="error" data-testid="run-error">
          <AlertTitle>
            {t('run.error.stage', { stage: r.error.stage ?? '?' })}: {r.error.message}
          </AlertTitle>
          {r.error.hint && (
            <Typography variant="body2">
              <strong>{t('run.error.hint')}: </strong>
              {r.error.hint}
            </Typography>
          )}
          {r.error.traceback_tail && (
            <details>
              <summary>{t('run.error.traceback')}</summary>
              <pre style={{ whiteSpace: 'pre-wrap', fontSize: 12 }}>{r.error.traceback_tail}</pre>
            </details>
          )}
        </Alert>
      )}
      {nextStep && !r.error && <Alert severity={r.status === 'succeeded' ? 'success' : 'info'}>{t(nextStep)}</Alert>}
      {r.status_reason && r.status !== 'succeeded' && !r.error && <Typography color="text.secondary">{r.status_reason}</Typography>}
      <Paper sx={{ p: 2 }}>
        <Typography variant="body2" gutterBottom>
          {progress ? t('run.progress', { epoch: progress.epoch + 1, total: progress.total }) : r.status === 'running' ? t('run.progress.none') : ''}
        </Typography>
        {progress && active && <LinearProgress variant="determinate" value={((progress.epoch + 1) / progress.total) * 100} aria-label={t('run.progress', { epoch: progress.epoch + 1, total: progress.total })} />}
      </Paper>
      <Tabs value={tab} onChange={(_, v: TabKey) => setParams(v === 'overview' ? {} : { tab: v })} aria-label={t('run.title')}>
        <Tab value="overview" label={t('run.tabs.overview')} id="tab-overview" aria-controls="panel-overview" />
        <Tab value="logs" label={t('run.tabs.logs')} id="tab-logs" aria-controls="panel-logs" />
        <Tab value="artifacts" label={t('run.tabs.artifacts')} id="tab-artifacts" aria-controls="panel-artifacts" />
        <Tab value="config" label={t('run.tabs.config')} id="tab-config" aria-controls="panel-config" />
      </Tabs>
      <div role="tabpanel" id={`panel-${tab}`} aria-labelledby={`tab-${tab}`}>
        {tab === 'overview' && <OverviewTab run={r} metrics={stream.metrics} statusEvents={stream.statusEvents} heartbeats={stream.heartbeats} />}
        {tab === 'logs' && <LogViewer runId={runId} live={active} />}
        {tab === 'artifacts' && <ArtifactsTab runId={runId} />}
        {tab === 'config' && <ConfigTab runId={runId} run={r} />}
      </div>
    </Stack>
  )
}

function OverviewTab({ run, metrics, statusEvents, heartbeats }: { run: RunView; metrics: ReturnType<typeof useRunStream>['metrics']; statusEvents: ReturnType<typeof useRunStream>['statusEvents']; heartbeats: number }) {
  const names = [...new Set(metrics.flatMap((m) => Object.keys(m.values)))]
  return (
    <Grid container spacing={2}>
      <Grid size={{ xs: 12, lg: 8 }}>
        <Typography variant="h2" gutterBottom>
          {t('run.metrics.title')}
        </Typography>
        {names.length === 0 ? (
          <EmptyState body={t('run.metrics.empty')} />
        ) : (
          <Grid container spacing={2}>
            {names.map((name) => (
              <Grid key={name} size={{ xs: 12, md: 6 }}>
                <MetricHistoryChart points={metrics} metric={name} />
              </Grid>
            ))}
          </Grid>
        )}
      </Grid>
      <Grid size={{ xs: 12, lg: 4 }}>
        <Stack spacing={2}>
          <RunTimeline run={run} statusEvents={statusEvents} heartbeats={heartbeats} />
          <LineageCard runId={run.run_id} />
        </Stack>
      </Grid>
    </Grid>
  )
}

function LineageCard({ runId }: { runId: string }) {
  const lineage = useRunLineage(runId)
  if (lineage.isPending) return <LoadingState />
  if (lineage.isError) return <ErrorState error={lineage.error} onRetry={() => void lineage.refetch()} />
  const parents = lineage.data.parents ?? []
  const children = lineage.data.children ?? []
  const row = (link: LineageLink) => (
    <Typography key={`${link.relation}-${link.run_id}`} variant="body2" component="li">
      {t(RELATION[link.relation])}:{' '}
      <Link component={RouterLink} to={`/runs/${encodeURIComponent(link.run_id)}`}>
        {link.run_id}
      </Link>
      {link.status ? ` · ${link.status}` : ''}
      {link.checkpoint_sha256 ? ` · ${t('run.lineage.weights', { sha: link.checkpoint_sha256.slice(0, 12) })}` : ''}
    </Typography>
  )
  return (
    <Paper sx={{ p: 2 }} component="section" aria-label={t('run.lineage')}>
      <Typography variant="h3" component="h2" gutterBottom>
        {t('run.lineage')}
      </Typography>
      <Typography variant="caption" color="text.secondary" component="p" gutterBottom>
        {t('run.lineage.help')}
      </Typography>
      {parents.length === 0 && children.length === 0 && <Typography variant="body2">{t('run.lineage.empty')}</Typography>}
      {parents.length > 0 && (
        <>
          <Typography variant="subtitle2">{t('run.lineage.parents')}</Typography>
          <ul style={{ margin: '0 0 8px', paddingLeft: 18 }}>{parents.map(row)}</ul>
        </>
      )}
      {children.length > 0 && (
        <>
          <Typography variant="subtitle2">{t('run.lineage.children')}</Typography>
          <ul style={{ margin: 0, paddingLeft: 18 }}>{children.map(row)}</ul>
        </>
      )}
    </Paper>
  )
}

/** run_dpd for a succeeded DPD run: export u = DPD(x) and score it through a surrogate (default: the training one). */
function ApplyDpdDialog({ run, onClose }: { run: RunView; onClose: () => void }) {
  const navigate = useNavigate()
  const lineage = useRunLineage(run.run_id)
  const succeeded = useRuns('succeeded')
  const submit = useSubmitRun()
  const idempotencyKey = useRef(crypto.randomUUID())
  const [paRunId, setPaRunId] = useState('')
  const training = (lineage.data?.parents ?? []).find((p) => p.relation === 'pa_surrogate')?.run_id
  const surrogates = (succeeded.data ?? []).filter((r) => r.task === 'train_pa' && r.dataset_id === run.dataset_id)
  const config = {
    task: 'run_dpd' as const,
    dataset: { id: run.dataset_id ?? '' },
    model: { key: run.model_key ?? 'gru' },
    evaluation: { evidence_type: 'dpd_surrogate' as const },
    dpd_reference: { run_id: run.run_id },
    ...(paRunId ? { pa_reference: { run_id: paRunId } } : {}),
  }
  return (
    <Dialog open onClose={onClose} fullWidth maxWidth="sm" aria-labelledby="apply-dpd-title">
      <DialogTitle id="apply-dpd-title">{t('run.apply.title')}</DialogTitle>
      <DialogContent>
        <Stack spacing={2} sx={{ mt: 1 }}>
          <Typography variant="body2">{t('run.apply.help')}</Typography>
          <TextField select fullWidth label={t('run.apply.surrogate')} value={paRunId} onChange={(e) => setPaRunId(e.target.value)} helperText={t('run.apply.surrogate.help')}>
            <MenuItem value="">{t('run.apply.surrogate.training', { run: training ?? '…' })}</MenuItem>
            {surrogates
              .filter((r) => r.run_id !== training)
              .map((r) => (
                <MenuItem key={r.run_id} value={r.run_id}>
                  {r.name || r.run_id} · {r.model_key}
                </MenuItem>
              ))}
          </TextField>
          {submit.isError && <ErrorState error={submit.error} />}
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>{t('common.back')}</Button>
        <Button variant="contained" disabled={submit.isPending} onClick={() => submit.mutate({ config, idempotency_key: idempotencyKey.current }, { onSuccess: (created) => navigate(`/runs/${encodeURIComponent(created.run_id)}`) })}>
          {t('run.apply.submit')}
        </Button>
      </DialogActions>
    </Dialog>
  )
}

function ArtifactsTab({ runId }: { runId: string }) {
  const artifacts = useRunArtifacts(runId)
  if (artifacts.isPending) return <LoadingState />
  if (artifacts.isError) return <ErrorState error={artifacts.error} onRetry={() => void artifacts.refetch()} />
  const m = artifacts.data
  const items = m.artifacts ?? []
  if (items.length === 0) return <EmptyState body={t('run.artifacts.empty')} />
  return (
    <Stack spacing={1}>
      {!m.complete && <Alert severity="warning">{t('run.artifacts.incomplete')}</Alert>}
      <Table size="small">
        <TableHead>
          <TableRow>
            <TableCell>id</TableCell>
            <TableCell>kind</TableCell>
            <TableCell>path</TableCell>
            <TableCell align="right">bytes</TableCell>
            <TableCell />
          </TableRow>
        </TableHead>
        <TableBody>
          {items.map((a) => (
            <TableRow key={a.artifact_id}>
              <TableCell>
                <code>{a.artifact_id}</code>
              </TableCell>
              <TableCell>{a.kind}</TableCell>
              <TableCell>
                <code>{a.file.path}</code>
              </TableCell>
              <TableCell align="right">{a.file.size_bytes ?? t('common.na')}</TableCell>
              <TableCell>
                <Link href={artifactUrl(runId, a.artifact_id)} download>
                  {t('run.artifacts.download')}
                </Link>
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </Stack>
  )
}

function ConfigTab({ runId, run }: { runId: string; run: RunView }) {
  const cfg = useRunConfig(runId)
  if (cfg.isPending) return <LoadingState />
  if (cfg.isError) return <ErrorState error={cfg.error} onRetry={() => void cfg.refetch()} />
  return (
    <Stack spacing={1}>
      <Typography variant="body2" color="text.secondary">
        {t('run.config.sha')}: <code>{run.config_sha256}</code>
      </Typography>
      <Stack direction="row" spacing={1}>
        <Button variant="outlined" size="small" component="a" href={artifactUrl(runId, 'config-resolved')} download={`${runId}.config.json`}>
          {t('run.config.download')}
        </Button>
        <Button variant="contained" size="small" component={RouterLink} to={`/experiments/new?from=${encodeURIComponent(runId)}`}>
          {t('run.config.rerun')}
        </Button>
      </Stack>
      <Typography variant="h3" component="h2">
        {t('run.config.resolved')}
      </Typography>
      <pre style={{ margin: 0, padding: 12, background: '#F6F7F9', borderRadius: 8, overflow: 'auto', fontSize: 12 }}>{JSON.stringify(cfg.data, null, 2)}</pre>
    </Stack>
  )
}
