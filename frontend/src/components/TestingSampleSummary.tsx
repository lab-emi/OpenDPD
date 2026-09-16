import { getQuery } from '@/api/client'
import { useQuery } from '@tanstack/react-query'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import type { Schemas } from '@/api/types'
import { formatNumber, t } from '@/i18n'

export function TestingSampleSummary({ datasetId, version }: { datasetId: string; version: string }) {
  const counts = useQuery({ queryKey: ['dataset-sample-counts', datasetId, version], enabled: !!datasetId,
    queryFn: getQuery<Schemas['DatasetSampleCounts']>(`/datasets/${encodeURIComponent(datasetId)}/sample-counts?version=${encodeURIComponent(version)}`),
    staleTime: 0, retry: false })
  const n = counts.data?.counts.test
  return <Paper variant="outlined" sx={{ p: 2, borderColor: 'primary.main' }} aria-live="polite" data-testid="testing-samples">
    <Stack direction="row" useFlexGap sx={{ alignItems: 'baseline', gap: 2, flexWrap: 'wrap' }}>
      <Typography variant="h2">{t('modelWorkflow.testSamples')}</Typography>
      <Typography sx={{ fontSize: 28, fontWeight: 750, fontVariantNumeric: 'tabular-nums', color: 'primary.main' }}>{n === undefined ? '—' : formatNumber(n)} <Box component="span" sx={{ fontSize: 14 }}>{t('generator.samplesUnit')}</Box></Typography>
      {n !== undefined && !!counts.data?.sample_rate_hz && <Typography color="text.secondary">{(n / counts.data.sample_rate_hz * 1000).toPrecision(5)} ms</Typography>}
    </Stack>
    <Typography variant="body2" color="text.secondary">{datasetId || t('form.dataset')} · {version} · {t('modelWorkflow.sampleHelp')}</Typography>
    {counts.isFetching && <Typography variant="caption">{t('state.loading')}</Typography>}
    {counts.isError && <Alert severity="warning" sx={{ mt: 1 }}>{t('modelWorkflow.countError')}</Alert>}
  </Paper>
}
