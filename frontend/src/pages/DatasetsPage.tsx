import Button from '@mui/material/Button'
import Dialog from '@mui/material/Dialog'
import DialogTitle from '@mui/material/DialogTitle'
import DialogContent from '@mui/material/DialogContent'
import DialogActions from '@mui/material/DialogActions'
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
import type { DatasetManifest } from '@/api/types'
import { WEB_MODE } from '@/api/client'
import { datasetLabel, formatNumber, message, t } from '@/i18n'
import { ImportDatasetDialog } from '@/components/ImportDatasetDialog'
import { BuiltinDatasetDialog } from '@/components/BuiltinDatasetDialog'
import { CreateDatasetDialog } from '@/components/CreateDatasetDialog'
import { DatasetGuide } from '@/components/DatasetGuide'
import { SyntheticDatasetDialog } from '@/components/SyntheticDatasetDialog'
import { EmptyState, ErrorState, LoadingState } from '@/components/StateBlock'

const sampleCount = (d: DatasetManifest) => d.captures?.length ? d.captures.reduce((total, c) => total + c.n_samples, 0) : d.n_samples ?? 0
function sampleRates(d: DatasetManifest) {
  const rates = d.captures?.length ? d.captures.map(c => c.sample_rate_hz) : d.signal.sample_rate_hz ? [d.signal.sample_rate_hz] : []
  if (!rates.length) return t('common.na')
  const low = Math.min(...rates) / 1e6, high = Math.max(...rates) / 1e6
  return `${formatNumber(low, { maximumFractionDigits: 2 })}${low === high ? '' : '–' + formatNumber(high, { maximumFractionDigits: 2 })} MS/s`
}
import { useStudioWorkflow } from '@/workflow/StudioWorkflow'

export function DatasetsPage() {
  const datasets = useDatasets()
  const customDatasets = useCustomDatasetImports()
  const navigate = useNavigate()
  const workflow = useStudioWorkflow()
  const [choosingExisting, setChoosingExisting] = useState(false)
  const train = (id: string) => { setChoosingExisting(false); workflow.selectDataset(id, 'raw-v1', true); navigate('/experiments/new?task=train_pa&dataset=' + encodeURIComponent(id)) }
  const [importing, setImporting] = useState(false)
  const [builtin, setBuiltin] = useState(false)
  const [creating, setCreating] = useState(false)
  const [synthetic, setSynthetic] = useState(false)
  const [guided, setGuided] = useState(false)
  const [params, setParams] = useSearchParams()
  const guide = params.get('guide') === 'start'
  const closeGuide = () => setParams((old) => { const next = new URLSearchParams(old); next.delete('guide'); return next }, { replace: true })
  const selected = (id: string, publish = false) => {
    const query = new URLSearchParams()
    if (guided) query.set('guide', 'ready')
    if (publish) query.set('publish', '1')
    navigate(`/datasets/${encodeURIComponent(id)}${query.size ? `?${query}` : ''}`)
  }
  const actions = (
    <Stack direction="row" spacing={1} useFlexGap sx={{ flexWrap: 'wrap', justifyContent: 'flex-end' }}>
      <Button variant="outlined" disabled={!customDatasets} onClick={() => { setGuided(false); setCreating(true) }}>
        {t('datasets.create.title')}{!customDatasets && ` · ${t('common.comingSoon')}`}
      </Button>
      <Button variant="contained" onClick={() => { setGuided(false); setBuiltin(true) }}>
        {t('datasets.builtin.title')}
      </Button>
      {!WEB_MODE && <Button disabled={!customDatasets} onClick={() => setImporting(true)}>{t('datasets.create.advancedImport')}{!customDatasets && ` · ${t('common.comingSoon')}`}</Button>}
      <Button disabled={!customDatasets} onClick={() => setSynthetic(true)}>{t('datasetResearch.syntheticTitle')}</Button>
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
            {datasets.data.filter(d => !d.parent_dataset_id).map((d) => (
              <TableRow key={d.dataset_id} hover>
                <TableCell>
                  <Link component={RouterLink} to={`/datasets/${encodeURIComponent(d.dataset_id)}`}>
                    {datasetLabel(d)}
                  </Link>{' '}
                  <Typography variant="caption" color="text.secondary">
                    {d.dataset_id}
                  </Typography>
                </TableCell>
                <TableCell align="right">{formatNumber(sampleCount(d))}{(d.captures?.length ?? 0) > 1 && <Typography variant="caption" component="div" color="text.secondary">{t('datasets.captureCount', { count: d.captures!.length })}</Typography>}</TableCell>
                <TableCell align="right" sx={{ whiteSpace: 'nowrap' }}>{sampleRates(d)}</TableCell>
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
      {guide && <DatasetGuide customDatasets={customDatasets} onSkip={closeGuide} onGenerator={() => navigate('/signal-generator')} onCsv={() => { closeGuide(); setGuided(true); setCreating(true) }} onBuiltin={() => { closeGuide(); setGuided(true); if (!datasets.data?.length) setBuiltin(true); else setChoosingExisting(true) }} />}
      <Dialog open={choosingExisting} onClose={() => setChoosingExisting(false)} fullWidth maxWidth="sm" aria-labelledby="choose-paired-dataset">
        <DialogTitle id="choose-paired-dataset">{t('paFlow.existingTitle')}</DialogTitle>
        <DialogContent><Typography color="text.secondary" sx={{ mb: 2 }}>{t('paFlow.existingHelp')}</Typography><Stack spacing={1}>
          {datasets.data?.filter(d => !d.parent_dataset_id).map(d => <Button key={d.dataset_id} variant="outlined" sx={{ justifyContent: 'space-between', textAlign: 'left' }} onClick={() => train(d.dataset_id)}>
            <span>{datasetLabel(d)}</span><span>{formatNumber(sampleCount(d))} I/Q</span></Button>)}
        </Stack></DialogContent>
        <DialogActions><Button onClick={() => setChoosingExisting(false)}>{t('common.close')}</Button><Button onClick={() => { setChoosingExisting(false); setBuiltin(true) }}>{t('datasets.builtin.title')}</Button></DialogActions>
      </Dialog>
      {builtin && <BuiltinDatasetDialog guided={guided} onSkipGuide={() => setGuided(false)} onClose={() => setBuiltin(false)} onSelected={(id) => { setBuiltin(false); if (guided) train(id); else selected(id) }} />}
      {customDatasets && creating && <CreateDatasetDialog guided={guided} onSkipGuide={() => setGuided(false)} onClose={() => setCreating(false)} onImported={(id, publish) => { setCreating(false); selected(id, publish) }} />}
      {synthetic && <SyntheticDatasetDialog onClose={() => setSynthetic(false)} />}
    </Stack>
  )
}
