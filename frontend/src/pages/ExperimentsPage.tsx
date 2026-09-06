import Button from '@mui/material/Button'
import Link from '@mui/material/Link'
import LinearProgress from '@mui/material/LinearProgress'
import Stack from '@mui/material/Stack'
import Table from '@mui/material/Table'
import TableBody from '@mui/material/TableBody'
import TableCell from '@mui/material/TableCell'
import TableHead from '@mui/material/TableHead'
import TableRow from '@mui/material/TableRow'
import ToggleButton from '@mui/material/ToggleButton'
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup'
import Typography from '@mui/material/Typography'
import { Link as RouterLink, useSearchParams } from 'react-router'
import { useRuns } from '@/api/hooks'
import type { RunStatus, RunView } from '@/api/types'
import { t } from '@/i18n'
import { StatusChip, statusLabel } from '@/components/StatusChip'
import { EmptyState, ErrorState, LoadingState } from '@/components/StateBlock'

const FILTERS: Array<RunStatus | 'all'> = ['all', 'running', 'queued', 'succeeded', 'failed']

function progressText(run: RunView): string {
  if (typeof run.progress_epoch === 'number' && typeof run.progress_total_epochs === 'number') {
    return t('run.progress', { epoch: run.progress_epoch + 1, total: run.progress_total_epochs })
  }
  return run.status === 'running' ? t('run.progress.none') : ''
}

export function RunTable({ runs }: { runs: RunView[] }) {
  return (
    <Table size="small" aria-label={t('experiments.title')}>
      <TableHead>
        <TableRow>
          <TableCell>{t('experiments.columns.run')}</TableCell>
          <TableCell>{t('experiments.columns.task')}</TableCell>
          <TableCell>{t('experiments.columns.model')}</TableCell>
          <TableCell>{t('experiments.columns.dataset')}</TableCell>
          <TableCell>{t('experiments.columns.status')}</TableCell>
          <TableCell sx={{ minWidth: 160 }}>{t('experiments.columns.progress')}</TableCell>
          <TableCell>{t('experiments.columns.created')}</TableCell>
        </TableRow>
      </TableHead>
      <TableBody>
        {runs.map((r) => {
          const pct = typeof r.progress_epoch === 'number' && r.progress_total_epochs ? ((r.progress_epoch + 1) / r.progress_total_epochs) * 100 : null
          return (
            <TableRow key={r.run_id} hover data-run-id={r.run_id}>
              <TableCell>
                <Link component={RouterLink} to={`/runs/${encodeURIComponent(r.run_id)}`}>
                  {r.name || r.run_id}
                </Link>
                {r.name && (
                  <Typography sx={{ display: 'block' }} variant="caption" color="text.secondary">
                    {r.run_id}
                  </Typography>
                )}
              </TableCell>
              <TableCell>{r.task}</TableCell>
              <TableCell>{r.model_key ?? t('common.na')}</TableCell>
              <TableCell>{r.dataset_id ?? t('common.na')}</TableCell>
              <TableCell>
                <StatusChip status={r.status} stale={r.heartbeat_stale} />
              </TableCell>
              <TableCell>
                {pct !== null && r.status === 'running' && <LinearProgress variant="determinate" value={pct} aria-label={progressText(r)} sx={{ mb: 0.5 }} />}
                <Typography variant="caption">{progressText(r)}</Typography>
              </TableCell>
              <TableCell>
                <time dateTime={r.created_at}>{new Date(r.created_at).toLocaleString()}</time>
              </TableCell>
            </TableRow>
          )
        })}
      </TableBody>
    </Table>
  )
}

export function ExperimentsPage() {
  const [params, setParams] = useSearchParams()
  const raw = params.get('status')
  const filter: RunStatus | 'all' = FILTERS.includes(raw as RunStatus) ? (raw as RunStatus) : 'all'
  const runs = useRuns(filter === 'all' ? undefined : filter)
  return (
    <Stack spacing={2}>
      <Stack sx={{ alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap' }} direction="row" useFlexGap>
        <Typography variant="h1">{t('experiments.title')}</Typography>
        <Button component={RouterLink} to="/experiments/new" variant="contained">
          {t('experiments.new')}
        </Button>
      </Stack>
      <ToggleButtonGroup
        size="small"
        exclusive
        value={filter}
        aria-label={t('experiments.columns.status')}
        onChange={(_, v: RunStatus | 'all' | null) => {
          if (v) setParams(v === 'all' ? {} : { status: v })
        }}
      >
        {FILTERS.map((f) => (
          <ToggleButton key={f} value={f}>
            {f === 'all' ? t('experiments.filter.all') : statusLabel(f)}
          </ToggleButton>
        ))}
      </ToggleButtonGroup>
      {runs.isPending ? (
        <LoadingState />
      ) : runs.isError ? (
        <ErrorState error={runs.error} onRetry={() => void runs.refetch()} />
      ) : runs.data.length === 0 ? (
        <EmptyState
          body={t('experiments.empty')}
          action={
            <Button component={RouterLink} to="/experiments/new" variant="outlined">
              {t('experiments.new')}
            </Button>
          }
        />
      ) : (
        <RunTable runs={runs.data} />
      )}
    </Stack>
  )
}
