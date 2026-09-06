import Alert from '@mui/material/Alert'
import Grid from '@mui/material/Grid'
import Link from '@mui/material/Link'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import { Link as RouterLink, useParams } from 'react-router'
import { useResult } from '@/api/hooks'
import type { EvaluationResult } from '@/api/types'
import { t } from '@/i18n'
import { EvidenceBadge } from '@/components/EvidenceBadge'
import { MetricCard } from '@/components/MetricCard'
import { ErrorState, LoadingState } from '@/components/StateBlock'

/** Result view: evidence first, then metrics with units, then how they were produced. */
export function ResultView({ result }: { result: EvaluationResult }) {
  return (
    <Stack spacing={2}>
      <Stack sx={{ alignItems: 'center', flexWrap: 'wrap' }} direction="row" spacing={2} useFlexGap>
        <Typography variant="h1">{result.result_id}</Typography>
        <EvidenceBadge evidence={result.evidence_type} mock={result.is_mock} />
        <Typography variant="body2" color="text.secondary">
          {t('results.columns.run')}:{' '}
          <Link component={RouterLink} to={`/runs/${encodeURIComponent(result.run_id ?? '')}`}>
            {result.run_id}
          </Link>
        </Typography>
      </Stack>
      <Typography variant="body2" color="text.secondary">
        {t('results.detail.protocol')}:{' '}
        {t('results.detail.protocol.body', {
          profile: result.metric_profile_id,
          version: result.metric_profile_version,
          segments: result.n_segments ?? t('common.na'),
          nperseg: result.nperseg ?? t('common.na'),
          epoch: result.selected_epoch ?? t('common.na'),
          device: result.device,
          numeric: result.numeric_mode ?? t('common.na'),
        })}
      </Typography>
      <Grid container spacing={2}>
        {result.metrics.map((m) => (
          <Grid key={m.name} size={{ xs: 6, sm: 4, md: 2.4 }}>
            <MetricCard metric={m} />
          </Grid>
        ))}
      </Grid>
      <Grid container spacing={2}>
        <Grid size={{ xs: 12, md: 6 }}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h3" component="h2" gutterBottom>
              {t('results.detail.reference')}
            </Typography>
            <Typography>
              <code>{result.reference.kind}</code> — {result.reference.description}
              {result.reference.gain_rule ? ` (${result.reference.gain_rule}${typeof result.reference.gain_value === 'number' ? ` = ${result.reference.gain_value}` : ''})` : ''}
            </Typography>
          </Paper>
        </Grid>
        <Grid size={{ xs: 12, md: 6 }}>
          <Paper sx={{ p: 2 }}>
            <Typography variant="h3" component="h2" gutterBottom>
              {t('results.detail.models')}
            </Typography>
            {(result.models ?? []).map((m) => (
              <Typography key={`${m.role}-${m.run_id}`} variant="body2">
                <strong>{m.role}</strong> {m.model.key} {JSON.stringify(m.model.parameters)} · {m.n_parameters ?? '?'} params · {m.execution_semantics}
                {typeof m.lookahead_samples === 'number' ? ` · look-ahead ${m.lookahead_samples}` : ''}
              </Typography>
            ))}
          </Paper>
        </Grid>
      </Grid>
      {(result.limitations ?? []).length > 0 && (
        <Alert severity={result.is_mock ? 'warning' : 'info'}>
          <strong>{t('results.detail.limitations')}</strong>
          <ul style={{ margin: '4px 0 0', paddingLeft: 18 }}>
            {(result.limitations ?? []).map((l) => (
              <li key={l}>{l}</li>
            ))}
          </ul>
        </Alert>
      )}
      <Typography variant="caption" color="text.secondary">
        {t('results.detail.charts.soon')} {t('results.export.soon')}
      </Typography>
    </Stack>
  )
}

export function ResultDetailPage() {
  const { runId = '' } = useParams()
  const result = useResult(runId)
  if (result.isPending) return <LoadingState />
  if (result.isError) return <ErrorState error={result.error} onRetry={() => void result.refetch()} />
  return <ResultView result={result.data} />
}
