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
import { useParams } from 'react-router'
import { useDiagnostics, useRunDoctor, versionNames } from '@/api/datasets'
import { useDataset } from '@/api/hooks'
import type { DatasetManifest, PreprocessingParams } from '@/api/types'
import { t } from '@/i18n'
import { DiagnosticItem } from '@/components/DiagnosticItem'
import { ManifestDialog } from '@/components/ManifestDialog'
import { PreprocessDialog } from '@/components/PreprocessDialog'
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
            <dt style={{ color: '#4B5563' }}>{k}</dt>
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
    p.delay_samples ? `delay ${p.delay_samples}` : '',
    p.gain_db ? `gain ${p.gain_db} dB` : '',
    p.phase_deg ? `phase ${p.phase_deg}°` : '',
    p.interpolate_non_finite ? 'interpolate' : '',
    p.remove_outliers ? 'outliers' : '',
    p.normalize !== 'none' ? p.normalize : '',
  ]
    .filter(Boolean)
    .join(', ') || 'none'

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
            <TableCell align="right">{v.n_samples.toLocaleString()}</TableCell>
            <TableCell>{v.params ? describeParams(v.params) : t('common.na')}</TableCell>
            <TableCell>{v.fit_range ? `${v.fit_range[0]}–${v.fit_range[1]}` : t('common.na')}</TableCell>
            <TableCell>{v.created_at ? new Date(v.created_at).toLocaleString() : t('common.na')}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}

export function DatasetDetailPage() {
  const { datasetId = '' } = useParams()
  const ds = useDataset(datasetId)
  const diag = useDiagnostics(datasetId)
  const doctor = useRunDoctor(datasetId)
  const [dialog, setDialog] = useState<'none' | 'manifest' | 'preprocess'>('none')
  const [versionEdit, setVersionEdit] = useState('')
  const [created, setCreated] = useState<string | null>(null)
  if (ds.isPending) return <LoadingState />
  if (ds.isError) return <ErrorState error={ds.error} onRetry={() => void ds.refetch()} />
  const d = ds.data
  const names = versionNames(d)
  const doctorVersion = names.includes(versionEdit) ? versionEdit : (names[names.length - 1] ?? 'raw-v1')
  const missing = (['sample_rate_hz', 'bandwidth_hz', 'nperseg', 'n_sub_ch'] as const).filter((k) => d.signal[k] === null || d.signal[k] === undefined)
  const report = diag.data ?? null
  return (
    <Stack spacing={2}>
      <Stack sx={{ alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap' }} direction="row" spacing={1} useFlexGap>
        <Typography variant="h1">{d.display_name}</Typography>
        <Stack direction="row" spacing={1}>
          <Button variant="outlined" onClick={() => setDialog('manifest')}>
            {t('datasets.detail.edit')}
          </Button>
          <Button variant="contained" onClick={() => setDialog('preprocess')}>
            {t('datasets.detail.versions.new')}
          </Button>
        </Stack>
      </Stack>
      {missing.length > 0 && <Alert severity="warning">{t('datasets.detail.missing', { fields: missing.join(', ') })}</Alert>}
      {created && (
        <Alert severity="success" onClose={() => setCreated(null)}>
          {t('datasets.preprocess.created', { version: created })}
        </Alert>
      )}
      <Grid container spacing={2}>
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
      </Grid>
      <section aria-labelledby="versions-title">
        <Typography variant="h2" id="versions-title" gutterBottom>
          {t('datasets.detail.versions')}
        </Typography>
        <VersionsTable d={d} />
      </section>
      <section aria-labelledby="doctor-title">
        <Stack sx={{ alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', mb: 1 }} direction="row" spacing={1} useFlexGap>
          <Typography variant="h2" id="doctor-title">
            {t('datasets.detail.doctor')}
          </Typography>
          <Stack direction="row" spacing={1}>
            <TextField select size="small" label={t('form.dataVersion')} value={doctorVersion} onChange={(e) => setVersionEdit(e.target.value)} sx={{ minWidth: 160 }}>
              {names.map((v) => (
                <MenuItem key={v} value={v}>
                  {v}
                </MenuItem>
              ))}
            </TextField>
            <Button variant="contained" onClick={() => doctor.mutate(doctorVersion)} disabled={doctor.isPending}>
              {report ? t('datasets.detail.doctor.rerun') : t('datasets.detail.doctor.run')}
            </Button>
          </Stack>
        </Stack>
        {doctor.isError && <ErrorState error={doctor.error} />}
        {diag.isPending || doctor.isPending ? (
          <LoadingState />
        ) : diag.isError ? (
          <ErrorState error={diag.error} onRetry={() => void diag.refetch()} />
        ) : !report ? (
          <EmptyState body={t('datasets.detail.doctor.empty')} />
        ) : (
          <Stack spacing={1}>
            <Alert severity={report.evaluation_blocked ? 'error' : 'success'}>{report.evaluation_blocked ? t('datasets.detail.doctor.blocked') : t('datasets.detail.doctor.ok')}</Alert>
            <Typography variant="caption" color="text.secondary">
              {t('datasets.detail.doctor.generated', { id: report.report_id, time: report.generated_at ? new Date(report.generated_at).toLocaleString() : t('common.na') })}
            </Typography>
            {(report.items ?? []).map((item) => (
              <DiagnosticItem key={item.code} item={item} />
            ))}
          </Stack>
        )}
      </section>
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
          }}
        />
      )}
    </Stack>
  )
}
