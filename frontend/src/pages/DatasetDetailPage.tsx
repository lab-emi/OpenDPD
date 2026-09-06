import Alert from '@mui/material/Alert'
import Grid from '@mui/material/Grid'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import { useParams } from 'react-router'
import { useDataset } from '@/api/hooks'
import type { DiagnosticReport } from '@/api/types'
import { t } from '@/i18n'
import { DiagnosticItem } from '@/components/DiagnosticItem'
import { ErrorState, LoadingState } from '@/components/StateBlock'
import doctorMock from '@mocks/diagnostics_missing_metadata.json'

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

export function DatasetDetailPage() {
  const { datasetId = '' } = useParams()
  const ds = useDataset(datasetId)
  if (ds.isPending) return <LoadingState />
  if (ds.isError) return <ErrorState error={ds.error} onRetry={() => void ds.refetch()} />
  const d = ds.data
  const missing = (['sample_rate_hz', 'bandwidth_hz', 'nperseg', 'n_sub_ch'] as const).filter((k) => d.signal[k] === null || d.signal[k] === undefined)
  const doctor = doctorMock.data as unknown as DiagnosticReport
  return (
    <Stack spacing={2}>
      <Typography variant="h1">{d.display_name}</Typography>
      {missing.length > 0 && <Alert severity="warning">{t('datasets.detail.missing', { fields: missing.join(', ') })}</Alert>}
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
              ['preprocessing_version', d.preprocessing_version],
              ['raw_sha256', d.raw_sha256],
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
      <section aria-labelledby="doctor-title">
        <Typography variant="h2" id="doctor-title" gutterBottom>
          {t('datasets.detail.doctor')}
        </Typography>
        <Alert severity="info" sx={{ mb: 1 }}>
          {t('datasets.detail.doctor.soon')}
        </Alert>
        <Stack spacing={1}>
          {doctor.items.map((item) => (
            <DiagnosticItem key={item.code} item={item} />
          ))}
        </Stack>
      </section>
    </Stack>
  )
}
