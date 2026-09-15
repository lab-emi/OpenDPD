import Link from '@mui/material/Link'
import LinearProgress from '@mui/material/LinearProgress'
import Table from '@mui/material/Table'
import TableContainer from '@mui/material/TableContainer'
import TableBody from '@mui/material/TableBody'
import TableCell from '@mui/material/TableCell'
import TableHead from '@mui/material/TableHead'
import TableRow from '@mui/material/TableRow'
import Typography from '@mui/material/Typography'
import { Link as RouterLink } from 'react-router'
import type { RunView } from '@/api/types'
import { formatDateTime, t } from '@/i18n'
import { StatusChip } from './StatusChip'
import { taskLabel } from './ExperimentTasks'

function progressText(run: RunView): string {
  if (typeof run.progress_epoch === 'number' && run.progress_epoch > 0 && typeof run.progress_total_epochs === 'number') {
    return t('run.progress', { epoch: run.progress_epoch, total: run.progress_total_epochs })
  }
  return run.status === 'running' ? t('run.progress.none') : ''
}

export function RunTable({ runs }: { runs: RunView[] }) {
  return (
    <TableContainer tabIndex={0} role="region" aria-label={t('experiments.title')}><Table size="small" aria-label={t('experiments.title')}>
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
          const pct = typeof r.progress_epoch === 'number' && r.progress_total_epochs ? (r.progress_epoch / r.progress_total_epochs) * 100 : null
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
              <TableCell>{taskLabel(r.task)}</TableCell>
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
                <time dateTime={r.created_at}>{formatDateTime(r.created_at)}</time>
              </TableCell>
            </TableRow>
          )
        })}
      </TableBody>
    </Table></TableContainer>
  )
}

