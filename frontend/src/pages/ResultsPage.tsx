import Link from '@mui/material/Link'
import Stack from '@mui/material/Stack'
import Table from '@mui/material/Table'
import TableBody from '@mui/material/TableBody'
import TableCell from '@mui/material/TableCell'
import TableHead from '@mui/material/TableHead'
import TableRow from '@mui/material/TableRow'
import Typography from '@mui/material/Typography'
import { Link as RouterLink } from 'react-router'
import { useRuns } from '@/api/hooks'
import { t } from '@/i18n'
import { EvidenceBadge } from '@/components/EvidenceBadge'
import { EmptyState, ErrorState, LoadingState } from '@/components/StateBlock'

/** Results are the succeeded runs that produced a formal result (one per run). */
export function ResultsPage() {
  const runs = useRuns('succeeded')
  if (runs.isPending) return <LoadingState />
  if (runs.isError) return <ErrorState error={runs.error} onRetry={() => void runs.refetch()} />
  const withResult = runs.data.filter((r) => r.result_id)
  return (
    <Stack spacing={2}>
      <Typography variant="h1">{t('results.title')}</Typography>
      {withResult.length === 0 ? (
        <EmptyState body={t('results.empty')} />
      ) : (
        <Table size="small" aria-label={t('results.title')}>
          <TableHead>
            <TableRow>
              <TableCell>{t('results.columns.run')}</TableCell>
              <TableCell>{t('experiments.columns.task')}</TableCell>
              <TableCell>{t('experiments.columns.model')}</TableCell>
              <TableCell>{t('experiments.columns.dataset')}</TableCell>
              <TableCell>{t('results.columns.evidence')}</TableCell>
              <TableCell>{t('experiments.columns.created')}</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {withResult.map((r) => (
              <TableRow key={r.run_id} hover>
                <TableCell>
                  <Link component={RouterLink} to={`/results/${encodeURIComponent(r.run_id)}`}>
                    {r.name || r.run_id}
                  </Link>
                </TableCell>
                <TableCell>{r.task}</TableCell>
                <TableCell>{r.model_key}</TableCell>
                <TableCell>{r.dataset_id}</TableCell>
                <TableCell>
                  <EvidenceBadge evidence={r.task === 'train_pa' ? 'pa_modeling' : 'dpd_surrogate'} />
                </TableCell>
                <TableCell>{new Date(r.created_at).toLocaleString()}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </Stack>
  )
}
