import CheckCircleIcon from '@mui/icons-material/CheckCircle'
import ArrowForwardIcon from '@mui/icons-material/ArrowForward'
import Box from '@mui/material/Box'
import Chip from '@mui/material/Chip'
import Tab from '@mui/material/Tab'
import Tabs from '@mui/material/Tabs'
import Alert from '@mui/material/Alert'
import Button from '@mui/material/Button'
import Grid from '@mui/material/Grid'
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
import { Link as RouterLink, useParams, useSearchParams } from 'react-router'
import { useDatasetAnalysis, useRunDoctor, versionNames } from '@/api/datasets'
import { useDataset } from '@/api/hooks'
import { WEB_MODE } from '@/api/client'
import type { DatasetManifest, PreprocessingParams } from '@/api/types'
import { datasetLabel, formatNumber, formatDateTime, message, t } from '@/i18n'
import { SignalInspection } from '@/components/SignalInspection'
import { DiagnosticItem } from '@/components/DiagnosticItem'
import { ManifestDialog } from '@/components/ManifestDialog'
import { PreprocessDialog } from '@/components/PreprocessDialog'
import { DatasetReadyGuide } from '@/components/DatasetGuide'
import { EmptyState, ErrorState, LoadingState } from '@/components/StateBlock'

function KeyValues({ title, entries }: { title: string; entries: Array<[string, unknown]> }) {
  return (
    <Paper sx={{ p: 2 }}>
      <Typography variant="h3" component="h2" gutterBottom>
        {title}
      </Typography>
      <dl style={{ margin: 0, display: 'grid', gridTemplateColumns: 'max-content 1fr', columnGap: 16, rowGap: 4 }}>
        {entries.map(([k, v]) => (
          <div key={k} style={{ display: 'contents' }}>
            <dt>{k}</dt>
            <dd style={{ margin: 0, wordBreak: 'break-all' }}>
              <code>{v === null || v === undefined ? t('common.na') : typeof v === 'object' ? JSON.stringify(v) : String(v)}</code>
            </dd>
          </div>
        ))}
      </dl>
    </Paper>
  )
}

const describeParams = (p: PreprocessingParams) =>
  [
    p.delay_samples ? `${t('datasets.preprocess.delay')}: ${formatNumber(p.delay_samples)}` : '',
    p.gain_db ? `${t('datasets.preprocess.gain')}: ${formatNumber(p.gain_db)}` : '',
    p.phase_deg ? `${t('datasets.preprocess.phase')}: ${formatNumber(p.phase_deg)}` : '',
    p.interpolate_non_finite ? t('datasets.preprocess.interpolate') : '',
    p.remove_outliers ? t('datasets.preprocess.outliers') : '',
    p.normalize !== 'none' ? message(p.normalize) : '',
  ]
    .filter(Boolean)
    .join(', ') || message('none')

function VersionsTable({ d }: { d: DatasetManifest }) {
  const versions = d.versions ?? []
  if (versions.length === 0) return <Typography color="text.secondary">{t('datasets.detail.versions.empty')}</Typography>
  return (
    <Table size="small" aria-label={t('datasets.detail.versions')}>
      <TableHead>
        <TableRow>
          <TableCell>{t('datasets.versions.columns.version')}</TableCell>
          <TableCell align="right">{t('datasets.versions.columns.samples')}</TableCell>
          <TableCell>{t('datasets.versions.columns.params')}</TableCell>
          <TableCell>{t('datasets.versions.columns.fit')}</TableCell>
          <TableCell>{t('datasets.versions.columns.created')}</TableCell>
        </TableRow>
      </TableHead>
      <TableBody>
        {versions.map((v) => (
          <TableRow key={v.version} hover data-version={v.version}>
            <TableCell>
              <code>{v.version}</code>
              {v.base_version && (
                <Typography variant="caption" color="text.secondary">
                  {' '}
                  ← {v.base_version}
                </Typography>
              )}
            </TableCell>
            <TableCell align="right">{formatNumber(v.n_samples)}</TableCell>
            <TableCell>{v.params ? describeParams(v.params) : t('common.na')}</TableCell>
            <TableCell>{v.fit_range ? `${v.fit_range[0]}–${v.fit_range[1]}` : t('common.na')}</TableCell>
            <TableCell>{v.created_at ? formatDateTime(v.created_at) : t('common.na')}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}

export function DatasetDetailPage() {
  const { datasetId = '' } = useParams()
  const ds = useDataset(datasetId)
  const doctor = useRunDoctor(datasetId)
  const [dialog, setDialog] = useState<'none' | 'manifest' | 'preprocess'>('none')
  const [search, setSearch] = useSearchParams()
  const names = ds.data ? versionNames(ds.data) : ['raw-v1']
  const selectedVersion = search.get('version') ?? 'raw-v1'
  const doctorVersion = names.includes(selectedVersion) ? selectedVersion : 'raw-v1'
  const analysis = useDatasetAnalysis(datasetId, doctorVersion, !!ds.data)
  const tab = ['overview', 'metadata', 'versions', 'doctor'].includes(search.get('tab') ?? '') ? search.get('tab')! : 'overview'
  const select = (key: string, value: string) => setSearch((old) => { const next = new URLSearchParams(old); next.set(key, value); return next })
  const [created, setCreated] = useState<string | null>(null)
  if (ds.isPending) return <LoadingState />
  if (ds.isError) return <ErrorState error={ds.error} onRetry={() => void ds.refetch()} />
  const d = ds.data
  const missing = (['sample_rate_hz', 'bandwidth_hz', 'nperseg', 'n_sub_ch'] as const).filter((k) => d.signal[k] === null || d.signal[k] === undefined)
  const report = analysis.data?.diagnostics ?? (doctor.variables === doctorVersion ? doctor.data : null) ?? null
  const complete = (label: string, ready: boolean) => <Stack direction="row" spacing={.75} sx={{ alignItems: 'center' }}>{label}{ready && <CheckCircleIcon color="success" sx={{ fontSize: 16 }} titleAccess={t('workflow.complete')} />}</Stack>
  return (
    <Stack spacing={1}>
      {search.get('guide') === 'ready' && <DatasetReadyGuide onClose={() => setSearch((old) => { const next = new URLSearchParams(old); next.delete('guide'); return next }, { replace: true })} />}
      <Stack sx={{ alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap' }} direction="row" spacing={1} useFlexGap>
        <Box sx={{ flex: '1 1 200px', minWidth: 0 }}><Typography variant="overline" color="primary" sx={{ fontSize: 11, lineHeight: 1.3, letterSpacing: ".12em", fontWeight: 700 }}>{t('inspection.step')}</Typography><Typography variant="h1" noWrap title={datasetLabel(d)}>{datasetLabel(d)}</Typography></Box>
        <Stack direction="row" spacing={1} sx={{ alignItems: 'center', flexWrap: 'wrap', gap: .5 }}>
          <TextField select size="small" label={t('form.dataVersion')} value={doctorVersion} onChange={(e) => select('version', e.target.value)} sx={{ minWidth: 125 }}>{names.map((v) => <MenuItem key={v} value={v}>{v}</MenuItem>)}</TextField>
          {!WEB_MODE && <Button variant="outlined" onClick={() => setDialog('manifest')}>
            {t('datasets.detail.edit')}
          </Button>}
          <Button variant="outlined" onClick={() => setDialog('preprocess')}>
            {t('datasets.detail.versions.new')}
          </Button>
          <Button variant="contained" endIcon={<ArrowForwardIcon />} component={RouterLink} to={`/experiments/new?dataset=${encodeURIComponent(datasetId)}&version=${encodeURIComponent(doctorVersion)}`} disabled={!analysis.data?.inspection_ready}>{t('inspection.configure')}</Button>
        </Stack>
      </Stack>
      {missing.length > 0 && <Alert severity="warning">{t('datasets.detail.missing', { fields: missing.join(', ') })}</Alert>}
      {created && (
        <Alert severity="success" onClose={() => setCreated(null)}>
          {t('datasets.preprocess.created', { version: created })}
        </Alert>
      )}
      <Paper sx={{ px: 2, py: .65, display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: 2.5 }}>
        {[[t('datasets.columns.samples'), formatNumber(analysis.data?.total_samples ?? d.n_samples ?? 0)], [t('inspection.sampleRate'), d.signal.sample_rate_hz ? `${formatNumber(d.signal.sample_rate_hz / 1e6)} MS/s` : t('common.na')], [t('inspection.bandwidth'), d.signal.bandwidth_hz ? `${formatNumber(d.signal.bandwidth_hz / 1e6)} MHz` : t('common.na')], [t('inspection.modulation'), d.signal.modulation ?? t('common.na')]].map(([label, value]) => <Box key={label}><Typography variant="caption" color="text.secondary">{label}</Typography><Typography variant="body2" sx={{ fontWeight: 650 }}>{value}</Typography></Box>)}
        <Box sx={{ flex: 1 }} />
        <Chip size="small" variant="outlined" label={message(d.origin)} />
        {analysis.data && <Chip size="small" color={analysis.data.inspection_ready ? 'success' : 'warning'} icon={analysis.data.inspection_ready ? <CheckCircleIcon /> : undefined} label={t(analysis.data.inspection_ready ? 'inspection.ready' : 'inspection.needsAttention')} />}
      </Paper>
      <Tabs value={tab} onChange={(_event, value: string) => select('tab', value)} aria-label={t('inspection.tabs')} variant="scrollable" scrollButtons="auto" sx={{ borderBottom: 1, borderColor: 'divider' }}>
        <Tab value="overview" id="dataset-tab-overview" aria-controls="dataset-panel-overview" label={t('inspection.overview')} />
        <Tab value="metadata" id="dataset-tab-metadata" aria-controls="dataset-panel-metadata" label={complete(t('datasets.detail.manifest'), !!analysis.data?.metadata_complete)} />
        <Tab value="versions" id="dataset-tab-versions" aria-controls="dataset-panel-versions" label={t('datasets.detail.versions')} />
        <Tab value="doctor" id="dataset-tab-doctor" aria-controls="dataset-panel-doctor" label={complete(t('datasets.detail.doctor'), !!analysis.data?.inspection_ready)} />
      </Tabs>
      {tab === 'overview' && <Box role="tabpanel" id="dataset-panel-overview" aria-labelledby="dataset-tab-overview">
        {analysis.isPending ? <LoadingState /> : analysis.isError ? <ErrorState error={analysis.error} onRetry={() => void analysis.refetch()} /> : <>
          <SignalInspection data={analysis.data} />
          <Typography variant="caption" color="text.secondary" component="p" sx={{ mt: 1, mb: 0 }}>{t('inspection.window', { start: formatNumber(analysis.data.sample_range[0]), end: formatNumber(analysis.data.sample_range[1]), total: formatNumber(analysis.data.total_samples) })}</Typography>
        </>}
      </Box>}
      {tab === 'metadata' && <Grid container spacing={2} role="tabpanel" id="dataset-panel-metadata" aria-labelledby="dataset-tab-metadata">
        <Grid size={{ xs: 12, md: 6 }}>
          <KeyValues
            title={t('datasets.detail.manifest')}
            entries={[
              ['dataset_id', d.dataset_id],
              ['origin', d.origin],
              ['source', d.source.kind],
              ['n_samples', d.n_samples],
              ['columns', Object.entries(d.columns ?? {}).map(([k, v]) => `${k}: ${v}`).join(', ')],
              ['raw_sha256', d.raw_sha256],
              ['notes', d.notes],
            ]}
          />
        </Grid>
        <Grid size={{ xs: 12, md: 6 }}>
          <KeyValues title={t('datasets.detail.signal')} entries={Object.entries(d.signal)} />
        </Grid>
        <Grid size={{ xs: 12, md: 6 }}>
          <KeyValues
            title={t('datasets.detail.split')}
            entries={[
              ['version', d.split.version],
              ['method', d.split.method],
              ['ratios', d.split.ratios],
              ['guard_samples', d.split.guard_samples],
              ['boundaries', d.split.boundaries],
            ]}
          />
        </Grid>
        <Grid size={{ xs: 12, md: 6 }}>
          <KeyValues title={t('datasets.detail.files')} entries={(d.files ?? []).map((f) => [f.path, `${f.size_bytes ?? '?'} B`])} />
        </Grid>
      </Grid>}
      {tab === 'versions' && <section role="tabpanel" id="dataset-panel-versions" aria-labelledby="dataset-tab-versions">
        <Typography variant="h2" id="versions-title" gutterBottom>
          {t('datasets.detail.versions')}
        </Typography>
        <VersionsTable d={d} />
      </section>}
      {tab === 'doctor' && <section role="tabpanel" id="dataset-panel-doctor" aria-labelledby="dataset-tab-doctor">
        <Stack sx={{ alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', mb: 1 }} direction="row" spacing={1} useFlexGap>
          <Typography variant="h2" id="doctor-title">
            {t('datasets.detail.doctor')}
          </Typography>
          <Stack direction="row" spacing={1}>
            <Button variant="contained" onClick={() => doctor.mutate(doctorVersion, { onSuccess: () => void analysis.refetch() })} disabled={doctor.isPending}>
              {report ? t('datasets.detail.doctor.rerun') : t('datasets.detail.doctor.run')}
            </Button>
          </Stack>
        </Stack>
        {doctor.isError && <ErrorState error={doctor.error} />}
        {analysis.isPending || doctor.isPending ? (
          <LoadingState />
        ) : analysis.isError ? (
          <ErrorState error={analysis.error} onRetry={() => void analysis.refetch()} />
        ) : !report ? (
          <EmptyState body={t('datasets.detail.doctor.empty')} />
        ) : (
          <Stack spacing={1}>
            <Alert severity={report.evaluation_blocked ? 'error' : 'success'}>{report.evaluation_blocked ? t('datasets.detail.doctor.blocked') : t('datasets.detail.doctor.ok')}</Alert>
            <Typography variant="caption" color="text.secondary">
              {t('datasets.detail.doctor.generated', { id: report.report_id, time: report.generated_at ? formatDateTime(report.generated_at) : t('common.na') })}
            </Typography>
            {(report.items ?? []).map((item) => (
              <DiagnosticItem key={item.code} item={item} />
            ))}
          </Stack>
        )}
      </section>}
      {dialog === 'manifest' && <ManifestDialog dataset={d} onClose={() => setDialog('none')} />}
      {dialog === 'preprocess' && (
        <PreprocessDialog
          datasetId={d.dataset_id}
          versions={names}
          report={report}
          onClose={() => setDialog('none')}
          onCreated={(v) => {
            setDialog('none')
            setCreated(v)
            select('version', v)
          }}
        />
      )}
    </Stack>
  )
}
