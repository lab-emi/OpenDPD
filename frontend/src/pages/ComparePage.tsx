import { DownloadLink } from '@/components/DownloadLink'
import Alert from '@mui/material/Alert'
import Button from '@mui/material/Button'
import Chip from '@mui/material/Chip'
import Link from '@mui/material/Link'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import Table from '@mui/material/Table'
import TableBody from '@mui/material/TableBody'
import TableCell from '@mui/material/TableCell'
import TableHead from '@mui/material/TableHead'
import TableRow from '@mui/material/TableRow'
import Typography from '@mui/material/Typography'
import { Link as RouterLink, useSearchParams } from 'react-router'
import { API, WEB_MODE } from '@/api/client'
import { useCompare, useRunConfig, useRuns, useMetricProfiles } from '@/api/hooks'
import type { EvaluationResult, MetricValue } from '@/api/types'
import { message, t } from '@/i18n'
import { ConfigDiff } from '@/components/ConfigDiff'
import { EvidenceBadge } from '@/components/EvidenceBadge'
import { SpectrumReview } from '@/components/SpectrumReview'
import { RFFactsPanel } from '@/components/RFFactsPanel'
import { MenuItem, TextField, TableContainer } from '@mui/material'
import { EmptyState, ErrorState, LoadingState } from '@/components/StateBlock'
import { useStudioColors } from '@/theme'

const label = (r: EvaluationResult) => r.run_id ?? r.result_id
const shown = (m: MetricValue | undefined) => (!m ? t('common.na') : m.status === 'ok' && typeof m.value === 'number' ? `${m.value.toFixed(2)} ${m.unit}` : (m.status ?? 'ok').replace('_', ' '))

export function metricDelta(candidate: EvaluationResult, reference: EvaluationResult, name: string, compatible: boolean) {
  const a = candidate.metrics.find(m => m.name === name), b = reference.metrics.find(m => m.name === name)
  if (!compatible || !a || !b || a.unit !== b.unit || a.status !== 'ok' || b.status !== 'ok' || typeof a.value !== 'number' || typeof b.value !== 'number') return t('common.na')
  const delta = a.value - b.value
  return `${delta >= 0 ? '+' : ''}${delta.toFixed(2)} ${a.unit === '%' ? 'pp' : a.unit}`
}

/** Index of the best ok value for a metric, or -1 when there is nothing to rank. */
export function bestIndex(results: EvaluationResult[], name: string): number {
  let best = -1
  let bestValue = 0
  results.forEach((r, i) => {
    const m = r.metrics.find((x) => x.name === name)
    if (!m || m.status !== 'ok' || typeof m.value !== 'number') return
    const better = best === -1 || (m.better === 'lower' ? m.value < bestValue : m.value > bestValue)
    if (better) {
      best = i
      bestValue = m.value
    }
  })
  return best
}

export function ComparePage() {
  const colors = useStudioColors()
  const [params, setParams] = useSearchParams()
  const ids = [...new Set(params.getAll('runs'))]
  const chosenProfiles = params.getAll('profiles')
  const report = useCompare(ids, params.get('profile'), chosenProfiles)
  const available = useRuns('succeeded')
  const profiles = useMetricProfiles()
  const cfgA = useRunConfig(ids[0] ?? '', ids.length === 2)
  const cfgB = useRunConfig(ids[1] ?? '', ids.length === 2)
  const referenceId = ids.includes(params.get('reference') ?? '') ? params.get('reference')! : ids[0] ?? ''
  const mode = params.get('mode') === 'cross_condition' ? 'cross_condition' : 'same_condition'
  const chooseReference = (id: string, selectedMode = mode) => {
    const next = new URLSearchParams(params)
    next.set('reference', id); next.set('mode', selectedMode); setParams(next, { replace: true })
  }

  if (ids.length < 2) {
    return (
      <Stack spacing={2}>
        <Typography variant="h1">{t('compare.title')}</Typography>
        <EmptyState body={t('compare.pick')} />
        <Button component={RouterLink} to="/results" variant="outlined" sx={{ alignSelf: 'flex-start' }}>
          {t('results.title')}
        </Button>
      </Stack>
    )
  }
  if (report.isPending) return <LoadingState />
  if (report.isError) return <ErrorState error={report.error} onRetry={() => void report.refetch()} />
  const rep = report.data
  const results = rep.results
  const reference = results.find(r => label(r) === referenceId) ?? results[0]!
  const names: string[] = []
  for (const r of results) for (const m of r.metrics) if (!names.includes(m.name)) names.push(m.name)
  const csvHref = `${API}/results/compare?${ids.map((i) => `runs=${encodeURIComponent(i)}`).join('&')}&format=csv&reference=${encodeURIComponent(referenceId)}&mode=${mode}${chosenProfiles.map(p => `&profiles=${encodeURIComponent(p)}`).join('')}${params.get('profile') ? `&profile=${encodeURIComponent(params.get('profile')!)}` : ''}`
  const problems = (rep.pairs ?? []).map((p) => ({ ...p, incompatibilities: p.incompatibilities ?? [] })).filter((p) => p.incompatibilities.length > 0)
  return (
    <Stack spacing={2}>
      <Stack direction="row" spacing={2} sx={{ alignItems: 'center', flexWrap: 'wrap' }} useFlexGap>
        <Typography variant="h1">{t('compare.title')}</Typography>
        <DownloadLink button  href={csvHref} download="comparison.csv" variant="outlined" size="small" sx={{ ml: 'auto' }}>
          {t('compare.csv')}
        </DownloadLink>
        {!WEB_MODE && <Button component={RouterLink} to={`/hardware?runs=${ids.join(',')}&profile=${reference.metric_profile_id}`} variant="outlined" size="small">{t('hardware.title')}</Button>}
      </Stack>
      <Stack direction="row" useFlexGap sx={{ flexWrap: 'wrap', gap: 2 }}>
        <TextField select size="small" label={t('review.reference')} value={referenceId} onChange={e => chooseReference(e.target.value)} sx={{ minWidth: 220 }}>
          {results.map(r => <MenuItem key={r.result_id} value={label(r)}>{label(r)}</MenuItem>)}
        </TextField>
        <TextField select size="small" label={t('review.addCandidate')} value="" disabled={ids.length >= 8} onChange={e => { const next = new URLSearchParams(params); next.append('runs', e.target.value); next.set('reference', referenceId); if (chosenProfiles.length) next.append('profiles', reference.metric_profile_id); setParams(next) }} sx={{ minWidth: 220 }}>
          <MenuItem value="">{t('review.addCandidate')}</MenuItem>
          {(available.data ?? []).filter(r => r.result_id && !ids.includes(r.run_id)).map(r => <MenuItem key={r.run_id} value={r.run_id}>{r.name || r.run_id}</MenuItem>)}
        </TextField>
        <TextField select size="small" label={t('review.mode')} value={mode} onChange={e => chooseReference(referenceId, e.target.value as typeof mode)} sx={{ minWidth: 220 }}>
          <MenuItem value="same_condition">{t('review.same')}</MenuItem><MenuItem value="cross_condition">{t('review.cross')}</MenuItem>
        </TextField>
      </Stack>
      <RFFactsPanel result={reference} />
      <Alert severity={rep.comparable ? 'success' : 'warning'} data-testid="compare-verdict">
        <strong>{mode === 'cross_condition' ? t('review.crossHelp') : rep.comparable ? t('compare.comparable') : t('compare.incompatible')}</strong> {mode === 'same_condition' ? rep.note : ''}
        {problems.length > 0 && (
          <ul style={{ margin: '4px 0 0', paddingLeft: 18 }}>
            {problems.map((p) => (
              <li key={`${p.a}-${p.b}`}>
                <code>{p.a}</code> vs <code>{p.b}</code>: {p.incompatibilities.join('; ')}
              </li>
            ))}
          </ul>
        )}
      </Alert>
      <Typography variant="caption">{t('review.deltaHelp')}</Typography>
      <TableContainer tabIndex={0}><Table size="small" aria-label={t('compare.title')}>
        <TableHead>
          <TableRow>
            <TableCell>{t('compare.metric')}</TableCell>
            {results.map((r, i) => (
              <TableCell key={r.result_id} align="right">
                <Stack direction="row" spacing={1} sx={{ justifyContent: 'flex-end', alignItems: 'center' }}>
                  <span aria-hidden style={{ width: 10, height: 10, borderRadius: 2, background: colors.chart[i % colors.chart.length], display: 'inline-block' }} />
                  <Link component={RouterLink} to={`/results/${encodeURIComponent(label(r))}`}>
                    {label(r)}
                  </Link>
                  <EvidenceBadge evidence={r.evidence_type} mock={r.is_mock} />
                </Stack>
                <Typography variant="caption" color="text.secondary" component="div">
                  {r.metric_profile_id} v{r.metric_profile_version} · {r.dataset.dataset_id} {r.dataset.preprocessing_version} · {r.reference.kind}
                </Typography>
              </TableCell>
            ))}
          </TableRow>
        </TableHead>
        <TableBody>
          {names.map((name) => {
            const best = rep.comparable && mode === 'same_condition' ? bestIndex(results, name) : -1
            return (
              <TableRow key={name}>
                <TableCell>
                  <code>{name}</code>
                  <Typography variant="caption" component="div" color="text.secondary">{message(profiles.data?.find(p => p.profile_id === reference.metric_profile_id)?.metrics.find(m => m.name === name)?.display_name ?? (name === 'EVM' ? 'Spectral EVM (repo-specific)' : ''))}</Typography>
                </TableCell>
                {results.map((r, i) => (
                  <TableCell key={r.result_id} align="right" data-best={best === i ? 'true' : undefined}>
                    {shown(r.metrics.find((m) => m.name === name))}
                    {r !== reference && <Typography variant="caption" component="div" color="text.secondary" data-testid="reference-delta">
                      {t('review.delta')}: {metricDelta(r, reference, name, !rep.pairs?.some(p => ((p.a === label(r) && p.b === referenceId) || (p.b === label(r) && p.a === referenceId)) && (p.incompatibilities?.length ?? 0) > 0))}
                    </Typography>}
                    {best === i && <Chip size="small" color="success" variant="outlined" label={t('compare.best')} sx={{ ml: 1 }} />}
                  </TableCell>
                ))}
              </TableRow>
            )
          })}
        </TableBody>
      </Table></TableContainer>
      <SpectrumReview key={ids.join(':')} results={results} referenceRunId={referenceId} mode={mode} onRestore={figure => {
        const next = new URLSearchParams(params); next.set('reference', figure.spec.reference_run_id); next.set('mode', figure.spec.mode ?? 'same_condition')
        next.delete('profile'); next.delete('profiles'); ids.forEach(id => next.append('profiles', figure.spec.profiles[id]!)); setParams(next, { replace: true })
      }} />
      {ids.length === 2 && cfgA.data && cfgB.data && (
        <Paper sx={{ p: 2 }} component="section" aria-label={t('diff.title')}>
          <Typography variant="h3" component="h2" gutterBottom>
            {t('diff.title')}
          </Typography>
          <ConfigDiff left={cfgA.data} right={cfgB.data} leftLabel={ids[0]} rightLabel={ids[1]} />
        </Paper>
      )}
    </Stack>
  )
}
