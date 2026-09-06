import Alert from '@mui/material/Alert'
import AlertTitle from '@mui/material/AlertTitle'
import Button from '@mui/material/Button'
import Chip from '@mui/material/Chip'
import Grid from '@mui/material/Grid'
import Link from '@mui/material/Link'
import LinearProgress from '@mui/material/LinearProgress'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import Tab from '@mui/material/Tab'
import Tabs from '@mui/material/Tabs'
import Table from '@mui/material/Table'
import TableBody from '@mui/material/TableBody'
import TableCell from '@mui/material/TableCell'
import TableHead from '@mui/material/TableHead'
import TableRow from '@mui/material/TableRow'
import Typography from '@mui/material/Typography'
import { useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { Link as RouterLink, useParams, useSearchParams } from 'react-router'
import { artifactUrl } from '@/api/client'
import { useRunStream } from '@/api/events'
import { keys, useCancelRun, useRetryRun, useRun, useRunArtifacts, useRunConfig } from '@/api/hooks'
import { isTerminal, type RunView } from '@/api/types'
import { t, type MessageKey } from '@/i18n'
import { LogViewer } from '@/components/LogViewer'
import { MetricHistoryChart } from '@/components/MetricHistoryChart'
import { RunTimeline } from '@/components/RunTimeline'
import { StatusChip } from '@/components/StatusChip'
import { DisconnectedState, EmptyState, ErrorState, LoadingState } from '@/components/StateBlock'

const TABS = ['overview', 'logs', 'artifacts', 'config'] as const
type TabKey = (typeof TABS)[number]

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
          {r.status === 'succeeded' && r.result_id && (
            <Button component={RouterLink} to={`/results/${encodeURIComponent(r.run_id)}`} variant="contained" size="small">
              {t('run.result')}
            </Button>
          )}
        </Stack>
      </Stack>
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
        <RunTimeline run={run} statusEvents={statusEvents} heartbeats={heartbeats} />
      </Grid>
    </Grid>
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
      <Typography variant="h3" component="h2">
        {t('run.config.resolved')}
      </Typography>
      <pre style={{ margin: 0, padding: 12, background: '#F6F7F9', borderRadius: 8, overflow: 'auto', fontSize: 12 }}>{JSON.stringify(cfg.data, null, 2)}</pre>
    </Stack>
  )
}
