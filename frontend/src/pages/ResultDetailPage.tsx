import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import Accordion from '@mui/material/Accordion'
import AccordionDetails from '@mui/material/AccordionDetails'
import AccordionSummary from '@mui/material/AccordionSummary'
import Alert from '@mui/material/Alert'
import Chip from '@mui/material/Chip'
import Grid from '@mui/material/Grid'
import Link from '@mui/material/Link'
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
import { Link as RouterLink, useParams } from 'react-router'
import { useMetricProfiles, useResult, useResultProfiles } from '@/api/hooks'
import type { EvaluationResult, MetricProfile } from '@/api/types'
import { t } from '@/i18n'
import { EvidenceBadge } from '@/components/EvidenceBadge'
import { MetricCard } from '@/components/MetricCard'
import { ErrorState, LoadingState } from '@/components/StateBlock'

/** Result view: evidence first, then metrics with their registry definitions, then how they were produced. */
export function ResultView({ result, profile, stored = [], onProfile }: { result: EvaluationResult; profile?: MetricProfile; stored?: string[]; onProfile?: (id: string) => void }) {
  const definitions = new Map((profile?.metrics ?? []).map((m) => [m.name, m]))
  return (
    <Stack spacing={2}>
      <Stack sx={{ alignItems: 'center', flexWrap: 'wrap' }} direction="row" spacing={2} useFlexGap>
        <Typography variant="h1">{result.result_id}</Typography>
        <EvidenceBadge evidence={result.evidence_type} mock={result.is_mock} />
        <Chip size="small" variant="outlined" label={`${result.metric_profile_id} v${result.metric_profile_version}${profile?.frozen ? ` · ${t('results.detail.frozen')}` : ''}`} data-profile={result.metric_profile_id} />
        <Typography variant="body2" color="text.secondary">
          {t('results.columns.run')}:{' '}
          <Link component={RouterLink} to={`/runs/${encodeURIComponent(result.run_id ?? '')}`}>
            {result.run_id}
          </Link>
        </Typography>
        {stored.length > 1 && onProfile && (
          <TextField select size="small" label={t('results.detail.profile')} value={result.metric_profile_id} onChange={(e) => onProfile(e.target.value)} helperText={t('results.detail.profile.help')} sx={{ minWidth: 220, ml: 'auto' }}>
            {stored.map((id) => (
              <MenuItem key={id} value={id}>
                {id}
              </MenuItem>
            ))}
          </TextField>
        )}
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
            <MetricCard metric={m} definition={definitions.get(m.name)} />
          </Grid>
        ))}
      </Grid>
      {profile && (
        <Accordion disableGutters>
          <AccordionSummary expandIcon={<ExpandMoreIcon />} aria-controls="definitions-panel" id="definitions-header">
            <Typography>{t('results.detail.definitions')}</Typography>
          </AccordionSummary>
          <AccordionDetails id="definitions-panel">
            <Typography variant="body2" color="text.secondary" gutterBottom>
              {t('results.detail.definitions.help', { profile: profile.profile_id, version: profile.version })} {profile.description}
            </Typography>
            <Table size="small" aria-label={t('results.detail.definitions')}>
              <TableHead>
                <TableRow>
                  <TableCell>{t('results.detail.metric')}</TableCell>
                  <TableCell>{t('metric.formula')}</TableCell>
                  <TableCell>{t('metric.aggregation')}</TableCell>
                  <TableCell>{t('results.detail.unit')}</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {profile.metrics.map((m) => (
                  <TableRow key={m.name}>
                    <TableCell>
                      <code>{m.name}</code> {m.display_name}
                    </TableCell>
                    <TableCell>
                      {m.formula}
                      {m.notes ? (
                        <Typography variant="caption" color="text.secondary" component="div">
                          {m.notes}
                        </Typography>
                      ) : null}
                    </TableCell>
                    <TableCell>{m.aggregation}</TableCell>
                    <TableCell>
                      {m.unit} · {m.better === 'lower' ? t('metric.lowerBetter') : t('metric.higherBetter')}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
            <Typography variant="h3" component="h3" sx={{ mt: 2 }} gutterBottom>
              {t('results.detail.parameters')}
            </Typography>
            <dl style={{ margin: 0, display: 'grid', gridTemplateColumns: 'max-content 1fr', columnGap: 16, rowGap: 4 }}>
              {Object.entries(profile.parameters ?? {}).map(([k, v]) => (
                <div key={k} style={{ display: 'contents' }}>
                  <dt style={{ color: '#4B5563' }}>{k}</dt>
                  <dd style={{ margin: 0 }}>{typeof v === 'string' ? v : JSON.stringify(v)}</dd>
                </div>
              ))}
            </dl>
          </AccordionDetails>
        </Accordion>
      )}
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
  const [profileId, setProfileId] = useState<string | null>(null)
  const result = useResult(runId, true, profileId)
  const stored = useResultProfiles(runId)
  const profiles = useMetricProfiles()
  if (result.isPending) return <LoadingState />
  if (result.isError) return <ErrorState error={result.error} onRetry={() => void result.refetch()} />
  const profile = profiles.data?.find((p) => p.profile_id === result.data.metric_profile_id)
  return <ResultView result={result.data} profile={profile} stored={stored.data ?? []} onProfile={setProfileId} />
}
