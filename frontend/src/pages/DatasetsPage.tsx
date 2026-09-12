import Button from '@mui/material/Button'
import Link from '@mui/material/Link'
import Stack from '@mui/material/Stack'
import Table from '@mui/material/Table'
import TableContainer from '@mui/material/TableContainer'
import TableBody from '@mui/material/TableBody'
import TableCell from '@mui/material/TableCell'
import TableHead from '@mui/material/TableHead'
import TableRow from '@mui/material/TableRow'
import Typography from '@mui/material/Typography'
import { useState } from 'react'
import { Link as RouterLink, useNavigate, useSearchParams } from 'react-router'
import { useCustomDatasetImports, useDatasets } from '@/api/hooks'
import { datasetLabel, formatNumber, message, t } from '@/i18n'
import { ImportDatasetDialog } from '@/components/ImportDatasetDialog'
import { BuiltinDatasetDialog } from '@/components/BuiltinDatasetDialog'
import { CreateDatasetDialog } from '@/components/CreateDatasetDialog'
import { DatasetGuide } from '@/components/DatasetGuide'
import { EmptyState, ErrorState, LoadingState } from '@/components/StateBlock'

export function DatasetsPage() {
  const datasets = useDatasets()
  const customDatasets = useCustomDatasetImports()
  const navigate = useNavigate()
  const [importing, setImporting] = useState(false)
  const [builtin, setBuiltin] = useState(false)
  const [creating, setCreating] = useState(false)
  const [guided, setGuided] = useState(false)
  const [params, setParams] = useSearchParams()
  const guide = params.get('guide') === 'start'
  const closeGuide = () => setParams((old) => { const next = new URLSearchParams(old); next.delete('guide'); return next }, { replace: true })
  const selected = (id: string) => navigate(`/datasets/${encodeURIComponent(id)}${guided ? '?guide=ready' : ''}`)
  const actions = (
    <Stack direction="row" spacing={1} useFlexGap sx={{ flexWrap: 'wrap', justifyContent: 'flex-end' }}>
      <Button variant="outlined" disabled={!customDatasets} onClick={() => { setGuided(false); setCreating(true) }}>
        {t('datasets.create.title')}{!customDatasets && ` · ${t('common.comingSoon')}`}
      </Button>
      <Button variant="contained" onClick={() => { setGuided(false); setBuiltin(true) }}>
        {t('datasets.builtin.title')}
      </Button>
      <Button disabled={!customDatasets} onClick={() => setImporting(true)}>{t('datasets.create.advancedImport')}{!customDatasets && ` · ${t('common.comingSoon')}`}</Button>
    </Stack>
  )
  return (
    <Stack spacing={2}>
      <Stack sx={{ alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 1 }} direction="row">
        <Typography variant="h1">{t('datasets.title')}</Typography>
        {actions}
      </Stack>
      <Button sx={{ alignSelf: 'flex-start' }} size="small" onClick={() => setParams({ guide: 'start' })}>{t('guide.replay')}</Button>
      {datasets.isPending ? (
        <LoadingState />
      ) : datasets.isError ? (
        <ErrorState error={datasets.error} onRetry={() => void datasets.refetch()} />
      ) : datasets.data.length === 0 ? (
        <EmptyState body={t('datasets.empty')} />
      ) : (
        <TableContainer tabIndex={0} role="region" aria-label={t('datasets.title')}><Table size="small" aria-label={t('datasets.title')}>
          <TableHead>
            <TableRow>
              <TableCell>{t('datasets.columns.id')}</TableCell>
              <TableCell align="right">{t('datasets.columns.samples')}</TableCell>
              <TableCell align="right">{t('datasets.columns.rate')}</TableCell>
              <TableCell>{t('datasets.columns.origin')}</TableCell>
              <TableCell align="right">{t('datasets.detail.versions')}</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {datasets.data.map((d) => (
              <TableRow key={d.dataset_id} hover>
                <TableCell>
                  <Link component={RouterLink} to={`/datasets/${encodeURIComponent(d.dataset_id)}`}>
                    {datasetLabel(d)}
                  </Link>{' '}
                  <Typography variant="caption" color="text.secondary">
                    {d.dataset_id}
                  </Typography>
                </TableCell>
                <TableCell align="right">{formatNumber(d.n_samples ?? 0)}</TableCell>
                <TableCell align="right">{d.signal.sample_rate_hz ? `${(d.signal.sample_rate_hz / 1e6).toFixed(2)} MHz` : t('common.na')}</TableCell>
                <TableCell>{message(d.origin)}</TableCell>
                <TableCell align="right">{Math.max(1, d.versions?.length ?? 0)}</TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table></TableContainer>
      )}
      {customDatasets && importing && (
        <ImportDatasetDialog
          onClose={() => setImporting(false)}
          onImported={(id) => {
            setImporting(false)
            navigate(`/datasets/${encodeURIComponent(id)}`)
          }}
        />
      )}
      {guide && <DatasetGuide customDatasets={customDatasets} onSkip={closeGuide} onCsv={() => { closeGuide(); setGuided(true); setCreating(true) }} onBuiltin={() => { closeGuide(); setGuided(true); setBuiltin(true) }} />}
      {builtin && <BuiltinDatasetDialog guided={guided} onSkipGuide={() => setGuided(false)} onClose={() => setBuiltin(false)} onSelected={(id) => { setBuiltin(false); selected(id) }} />}
      {customDatasets && creating && <CreateDatasetDialog guided={guided} onSkipGuide={() => setGuided(false)} onClose={() => setCreating(false)} onImported={(id) => { setCreating(false); selected(id) }} />}
    </Stack>
  )
}
