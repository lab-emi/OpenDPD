import Button from '@mui/material/Button'
import Link from '@mui/material/Link'
import Stack from '@mui/material/Stack'
import Table from '@mui/material/Table'
import TableBody from '@mui/material/TableBody'
import TableCell from '@mui/material/TableCell'
import TableHead from '@mui/material/TableHead'
import TableRow from '@mui/material/TableRow'
import Tooltip from '@mui/material/Tooltip'
import Typography from '@mui/material/Typography'
import { Link as RouterLink } from 'react-router'
import { useDatasets, useImportBuiltin } from '@/api/hooks'
import { t } from '@/i18n'
import { EmptyState, ErrorState, LoadingState } from '@/components/StateBlock'

export function DatasetsPage() {
  const datasets = useDatasets()
  const importBuiltin = useImportBuiltin()
  const actions = (
    <Stack direction="row" spacing={1}>
      <Button variant="contained" onClick={() => importBuiltin.mutate('DPA_200MHz')} disabled={importBuiltin.isPending}>
        {t('datasets.import')}
      </Button>
      <Tooltip title={t('datasets.importOwn.soon')}>
        <span>
          <Button variant="outlined" disabled>
            {t('datasets.importOwn')}
          </Button>
        </span>
      </Tooltip>
    </Stack>
  )
  return (
    <Stack spacing={2}>
      <Stack sx={{ alignItems: 'center', justifyContent: 'space-between' }} direction="row">
        <Typography variant="h1">{t('datasets.title')}</Typography>
        {actions}
      </Stack>
      {importBuiltin.isError && <ErrorState error={importBuiltin.error} />}
      {datasets.isPending ? (
        <LoadingState />
      ) : datasets.isError ? (
        <ErrorState error={datasets.error} onRetry={() => void datasets.refetch()} />
      ) : datasets.data.length === 0 ? (
        <EmptyState body={t('datasets.empty')} />
      ) : (
        <Table size="small" aria-label={t('datasets.title')}>
          <TableHead>
            <TableRow>
              <TableCell>{t('datasets.columns.id')}</TableCell>
              <TableCell align="right">{t('datasets.columns.samples')}</TableCell>
              <TableCell align="right">{t('datasets.columns.rate')}</TableCell>
              <TableCell>{t('datasets.columns.origin')}</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {datasets.data.map((d) => (
              <TableRow key={d.dataset_id} hover>
                <TableCell>
                  <Link component={RouterLink} to={`/datasets/${encodeURIComponent(d.dataset_id)}`}>
                    {d.display_name}
                  </Link>{' '}
                  <Typography variant="caption" color="text.secondary">
                    {d.dataset_id}
                  </Typography>
                </TableCell>
                <TableCell align="right">{(d.n_samples ?? 0).toLocaleString()}</TableCell>
                <TableCell align="right">{d.signal.sample_rate_hz ? `${(d.signal.sample_rate_hz / 1e6).toFixed(2)} MHz` : t('common.na')}</TableCell>
                <TableCell>{d.origin}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      )}
    </Stack>
  )
}
