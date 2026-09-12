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
import { useQueries } from '@tanstack/react-query'
import { useMemo } from 'react'
import { Link as RouterLink, useSearchParams } from 'react-router'
import { API } from '@/api/client'
import { artifactJsonQuery, useCompare, useRunConfig } from '@/api/hooks'
import type { EvaluationResult, MetricValue } from '@/api/types'
import { t } from '@/i18n'
import { ConfigDiff } from '@/components/ConfigDiff'
import { EvidenceBadge } from '@/components/EvidenceBadge'
import type { SpectrumData } from '@/components/ResultCharts'
import { SpectrumPlot, type SpectrumTrace } from '@/components/SpectrumPlot'
import { EmptyState, ErrorState, LoadingState } from '@/components/StateBlock'
import { useStudioColors } from '@/theme'

const label = (r: EvaluationResult) => r.run_id ?? r.result_id
const shown = (m: MetricValue | undefined) => (!m ? t('common.na') : m.status === 'ok' && typeof m.value === 'number' ? `${m.value.toFixed(2)} ${m.unit}` : (m.status ?? 'ok').replace('_', ' '))

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
  const [params] = useSearchParams()
  const ids = params.getAll('runs')
  const report = useCompare(ids)
  const spectra = useQueries({ queries: ids.map((id) => artifactJsonQuery<SpectrumData>(id, 'plot-spectrum')) })
  const cfgA = useRunConfig(ids[0] ?? '', ids.length === 2)
  const cfgB = useRunConfig(ids[1] ?? '', ids.length === 2)
  const overlay = useMemo(() => {
    const loaded = spectra.map((q) => q.data ?? null)
    const first = loaded.find((d) => d !== null)
    if (!first) return null
    const same = loaded.every((d) => d === null || (d.axis === first.axis && d.frequency.length === first.frequency.length && d.sample_rate_hz === first.sample_rate_hz))
    if (!same) return { mismatch: true as const, frequency: first.frequency, axis: first.axis, bands: first.bands, traces: [] as SpectrumTrace[] }
    const traces: SpectrumTrace[] = []
    loaded.forEach((d, i) => {
      if (!d) return
      const primary = d.traces.find((tr) => tr.role === 'primary')
      if (primary) traces.push({ name: `${ids[i]} — ${primary.name}`, psdDb: primary.psd_db, color: colors.chart[i % colors.chart.length] })
    })
    const reference = first.traces.find((tr) => tr.role === 'reference')
    if (reference) traces.push({ name: t('compare.spectrum.reference', { run: ids[loaded.indexOf(first)] ?? '' }), psdDb: reference.psd_db, color: colors.textSecondary })
    return { mismatch: false as const, frequency: first.frequency, axis: first.axis, bands: first.bands, traces }
  }, [spectra, ids, colors])

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
  const names: string[] = []
  for (const r of results) for (const m of r.metrics) if (!names.includes(m.name)) names.push(m.name)
  const csvHref = `${API}/results/compare?${ids.map((i) => `runs=${encodeURIComponent(i)}`).join('&')}&format=csv`
  const problems = (rep.pairs ?? []).map((p) => ({ ...p, incompatibilities: p.incompatibilities ?? [] })).filter((p) => p.incompatibilities.length > 0)
  return (
    <Stack spacing={2}>
      <Stack direction="row" spacing={2} sx={{ alignItems: 'center', flexWrap: 'wrap' }} useFlexGap>
        <Typography variant="h1">{t('compare.title')}</Typography>
        <Button component="a" href={csvHref} download="comparison.csv" variant="outlined" size="small" sx={{ ml: 'auto' }}>
          {t('compare.csv')}
        </Button>
      </Stack>
      <Alert severity={rep.comparable ? 'success' : 'warning'} data-testid="compare-verdict">
        <strong>{rep.comparable ? t('compare.comparable') : t('compare.incompatible')}</strong> {rep.note}
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
      <Table size="small" aria-label={t('compare.title')}>
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
            const best = rep.comparable ? bestIndex(results, name) : -1
            return (
              <TableRow key={name}>
                <TableCell>
                  <code>{name}</code>
                </TableCell>
                {results.map((r, i) => (
                  <TableCell key={r.result_id} align="right" data-best={best === i ? 'true' : undefined}>
                    {shown(r.metrics.find((m) => m.name === name))}
                    {best === i && <Chip size="small" color="success" variant="outlined" label={t('compare.best')} sx={{ ml: 1 }} />}
                  </TableCell>
                ))}
              </TableRow>
            )
          })}
        </TableBody>
      </Table>
      <Paper sx={{ p: 2 }} component="section" aria-label={t('compare.spectrum')}>
        <Typography variant="h3" component="h2" gutterBottom>
          {t('compare.spectrum')}
        </Typography>
        {overlay === null && (spectra.some((q) => q.isPending) ? <LoadingState /> : <Typography variant="body2">{t('results.charts.none')}</Typography>)}
        {overlay?.mismatch && <Alert severity="info">{t('compare.spectrum.mismatch')}</Alert>}
        {overlay && !overlay.mismatch && <SpectrumPlot frequencyHz={overlay.frequency} axis={overlay.axis} traces={overlay.traces} bands={overlay.bands ?? undefined} viewKey={ids.join(':')} />}
      </Paper>
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
