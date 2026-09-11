import Button from '@mui/material/Button'
import Checkbox from '@mui/material/Checkbox'
import Link from '@mui/material/Link'
import Stack from '@mui/material/Stack'
import Table from '@mui/material/Table'
import TableBody from '@mui/material/TableBody'
import TableCell from '@mui/material/TableCell'
import TableHead from '@mui/material/TableHead'
import TableRow from '@mui/material/TableRow'
import Typography from '@mui/material/Typography'
import { useState } from 'react'
import { Link as RouterLink, useNavigate } from 'react-router'
import { useRuns } from '@/api/hooks'
import { formatDateTime, t } from '@/i18n'
import { EvidenceBadge } from '@/components/EvidenceBadge'
import { EmptyState, ErrorState, LoadingState } from '@/components/StateBlock'

/** Results are the succeeded runs that produced a formal result (one per run). */
export function ResultsPage() {
  const runs = useRuns('succeeded')
  const navigate = useNavigate()
  const [selected, setSelected] = useState<string[]>([])
  const toggle = (id: string) => setSelected((s) => (s.includes(id) ? s.filter((x) => x !== id) : [...s, id]))
  if (runs.isPending) return <LoadingState />
  if (runs.isError) return <ErrorState error={runs.error} onRetry={() => void runs.refetch()} />
  const withResult = runs.data.filter((r) => r.result_id)
  return (
    <Stack spacing={2}>
      <Stack direction="row" spacing={2} sx={{ alignItems: 'center', flexWrap: 'wrap' }} useFlexGap>
        <Typography variant="h1">{t('results.title')}</Typography>
        <Button variant="contained" size="small" disabled={selected.length < 2} onClick={() => navigate(`/results/compare?${selected.map((id) => `runs=${encodeURIComponent(id)}`).join('&')}`)} sx={{ ml: 'auto' }}>
          {t('results.compare', { n: selected.length })}
        </Button>
      </Stack>
      {withResult.length === 0 ? (
        <EmptyState body={t('results.empty')} />
      ) : (
        <Table size="small" aria-label={t('results.title')}>
          <TableHead>
            <TableRow>
              <TableCell padding="checkbox" />
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
              <TableRow key={r.run_id} hover selected={selected.includes(r.run_id)}>
                <TableCell padding="checkbox">
                  <Checkbox size="small" checked={selected.includes(r.run_id)} onChange={() => toggle(r.run_id)} slotProps={{ input: { 'aria-label': t('results.select', { run: r.name || r.run_id }) } }} />
                </TableCell>
                <TableCell>
                  <Link component={RouterLink} to={`/results/${encodeURIComponent(r.run_id)}`}>
                    {r.name || r.run_id}
                  </Link>
                </TableCell>
                <TableCell>{r.task}</TableCell>
                <TableCell>{r.model_key}</TableCell>
                <TableCell>{r.dataset_id}</TableCell>
                <TableCell>
                  <EvidenceBadge evidence={r.task === 'train_pa' || r.task === 'evaluate_pa' ? 'pa_modeling' : r.task === 'evaluate_measured' ? 'dpd_measured' : 'dpd_surrogate'} />
                </TableCell>
                <TableCell>{formatDateTime(r.created_at)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </Stack>
  )
}
