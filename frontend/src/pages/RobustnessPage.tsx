import { DownloadLink } from '@/components/DownloadLink'
import Alert from '@mui/material/Alert'
import Chip from '@mui/material/Chip'
import FormControlLabel from '@mui/material/FormControlLabel'
import Link from '@mui/material/Link'
import MenuItem from '@mui/material/MenuItem'
import Stack from '@mui/material/Stack'
import Switch from '@mui/material/Switch'
import Table from '@mui/material/Table'
import TableBody from '@mui/material/TableBody'
import TableCell from '@mui/material/TableCell'
import TableHead from '@mui/material/TableHead'
import TableRow from '@mui/material/TableRow'
import TextField from '@mui/material/TextField'
import ToggleButton from '@mui/material/ToggleButton'
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup'
import Typography from '@mui/material/Typography'
import { useState } from 'react'
import { Link as RouterLink, useParams } from 'react-router'
import { API } from '@/api/client'
import { useAdaptationReport, useAdaptationReports } from '@/api/hooks'
import type { AdaptationReport, AdaptationReportSummary, CellAggregate } from '@/api/types'
import { EmptyState, ErrorState, LoadingState } from '@/components/StateBlock'
import { formatDateTime, t, type MessageKey } from '@/i18n'

type Task = CellAggregate['task']
const TASKS: Task[] = ['zero_update', 'few_shot', 'full_retrain']

const taskLabel = (task: Task, budget: number | null | undefined) =>
  budget ? t('robustness.task.budget', { task: t(`robustness.task.${task}` as MessageKey), budget }) : t(`robustness.task.${task}` as MessageKey)

const columnKey = (task: Task, budget: number | null | undefined) => `${task}:${budget ?? 0}`

/** Adaptation reports (conditions-v1): a list, and the condition matrix of the selected one. Read only. */
export function RobustnessPage() {
  const { planSha } = useParams<{ planSha: string }>()
  const reports = useAdaptationReports()
  if (reports.isPending) return <LoadingState />
  if (reports.isError) return <ErrorState error={reports.error} onRetry={() => void reports.refetch()} />
  const selected = planSha ?? reports.data[0]?.plan_sha256
  return (
    <Stack spacing={2}>
      <Typography variant="h1">{t('robustness.title')}</Typography>
      <Typography variant="body2" color="text.secondary">
        {t('robustness.intro')}
      </Typography>
      {reports.data.length === 0 ? <EmptyState body={t('robustness.empty')} /> : <ReportList reports={reports.data} selected={selected} />}
      {selected ? <ReportView planSha={selected} /> : null}
    </Stack>
  )
}

function ReportList({ reports, selected }: { reports: AdaptationReportSummary[]; selected?: string }) {
  return (
    <Table size="small" aria-label={t('robustness.list.label')}>
      <TableHead>
        <TableRow>
          <TableCell>{t('robustness.columns.card')}</TableCell>
          <TableCell>{t('robustness.columns.device')}</TableCell>
          <TableCell>{t('robustness.columns.dimension')}</TableCell>
          <TableCell align="right">{t('robustness.columns.conditions')}</TableCell>
          <TableCell>{t('robustness.columns.cells')}</TableCell>
          <TableCell>{t('robustness.columns.bar')}</TableCell>
          <TableCell>{t('robustness.columns.generated')}</TableCell>
        </TableRow>
      </TableHead>
      <TableBody>
        {reports.map((r) => (
          <TableRow key={r.plan_sha256} hover selected={selected !== undefined && r.plan_sha256.startsWith(selected)}>
            <TableCell>
              <Link component={RouterLink} to={`/robustness/${r.plan_sha256.slice(0, 12)}`}>
                {r.set_id}
              </Link>
              <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                {r.plan_sha256.slice(0, 12)}
              </Typography>
            </TableCell>
            <TableCell>{r.device}</TableCell>
            <TableCell>{r.dimension}</TableCell>
            <TableCell align="right">{r.n_conditions}</TableCell>
            <TableCell>{t('robustness.cells.summary', { ok: r.n_cells - r.n_without_number, total: r.n_cells })}</TableCell>
            <TableCell>
              <EvidenceBarChip met={r.evidence_met} />
            </TableCell>
            <TableCell>{formatDateTime(r.generated_at)}</TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}

function EvidenceBarChip({ met }: { met: boolean }) {
  return <Chip size="small" color={met ? 'success' : 'warning'} variant={met ? 'filled' : 'outlined'} label={met ? t('robustness.bar.met') : t('robustness.bar.notMet')} data-testid="evidence-bar" />
}

const fmt = (x: number | null | undefined, digits = 2) => (x === null || x === undefined ? '' : x.toFixed(digits))

function ReportView({ planSha }: { planSha: string }) {
  const report = useAdaptationReport(planSha)
  if (report.isPending) return <LoadingState />
  if (report.isError) return <ErrorState error={report.error} onRetry={() => void report.refetch()} />
  return <ReportBody report={report.data} />
}

function ReportBody({ report }: { report: AdaptationReport }) {
  const entries = [...new Set(report.aggregates.map((a) => a.entry_id))]
  const [entry, setEntry] = useState<string>(entries[0] ?? '')
  const activeEntry = entries.includes(entry) ? entry : (entries[0] ?? '')
  const aggregates = report.aggregates.filter((a) => a.entry_id === activeEntry)
  const metrics = [...new Set(aggregates.flatMap((a) => Object.keys(a.metrics ?? {})))].sort()
  const preferred = report.target && metrics.includes(report.target.metric) ? report.target.metric : (metrics[0] ?? '')
  const [metricChoice, setMetricChoice] = useState<string | null>(null)
  const metric = metricChoice && metrics.includes(metricChoice) ? metricChoice : preferred
  const [tasks, setTasks] = useState<Task[]>(TASKS)
  const [failuresOnly, setFailuresOnly] = useState(false)
  const columns = [...new Map(aggregates.filter((a) => tasks.includes(a.task)).map((a) => [columnKey(a.task, a.budget_samples), a])).values()]
    .map((a) => ({ task: a.task, budget: a.budget_samples ?? null }))
    .sort((a, b) => TASKS.indexOf(a.task) - TASKS.indexOf(b.task) || (a.budget ?? 0) - (b.budget ?? 0))
  const bar = report.evidence_bar
  const rows = report.condition_set.conditions.filter((c) => !failuresOnly || aggregates.some((a) => a.condition_id === c.condition_id && a.n_failed > 0))
  const failuresOf = (agg: CellAggregate) =>
    [...new Set(report.cells.filter((c) => c.status !== 'ok' && c.entry_id === agg.entry_id && c.task === agg.task && c.condition_id === agg.condition_id && (c.budget_samples ?? 0) === (agg.budget_samples ?? 0)).map((c) => c.failure ?? t('robustness.matrix.noNumber')))]
  return (
    <Stack spacing={2} data-testid="adaptation-report">
      <Typography variant="h2">{report.condition_set.set_id}</Typography>
      <Typography variant="body2" color="text.secondary">
        {t('robustness.report.header', { plan: report.plan_sha256.slice(0, 12), profile: report.metric_profile_id, version: report.metric_profile_version, seeds: report.seeds.join(', '), budgets: report.budgets.join(', '), device: report.device })}
      </Typography>
      <Stack direction="row" spacing={1} sx={{ alignItems: 'center', flexWrap: 'wrap' }} useFlexGap>
        <EvidenceBarChip met={bar.met} />
        <Typography variant="body2">{t('robustness.bar.detail', { n: bar.n_conditions, min: bar.min_conditions, independent: String(bar.independent_batches), measured: String(bar.measured_origin) })}</Typography>
      </Stack>
      {!bar.met ? <Alert severity="warning">{t('robustness.bar.rehearsal')}</Alert> : null}
      <Typography variant="body2">
        <strong>{t('robustness.repeats')}:</strong> {report.repeats}
      </Typography>
      <Stack direction="row" spacing={2} sx={{ alignItems: 'center', flexWrap: 'wrap' }} useFlexGap>
        <TextField select size="small" label={t('robustness.filters.entry')} value={activeEntry} onChange={(e) => setEntry(e.target.value)} sx={{ minWidth: 140 }}>
          {entries.map((id) => (
            <MenuItem key={id} value={id}>
              {id}
            </MenuItem>
          ))}
        </TextField>
        <TextField select size="small" label={t('robustness.filters.metric')} value={metric} onChange={(e) => setMetricChoice(e.target.value)} sx={{ minWidth: 140 }} disabled={metrics.length === 0}>
          {metrics.map((m) => (
            <MenuItem key={m} value={m}>
              {m}
            </MenuItem>
          ))}
        </TextField>
        <ToggleButtonGroup size="small" value={tasks} onChange={(_, next: Task[]) => setTasks(next.length ? next : tasks)} aria-label={t('robustness.filters.tasks')}>
          {TASKS.map((task) => (
            <ToggleButton key={task} value={task}>
              {t(`robustness.task.${task}` as MessageKey)}
            </ToggleButton>
          ))}
        </ToggleButtonGroup>
        <FormControlLabel control={<Switch size="small" checked={failuresOnly} onChange={(e) => setFailuresOnly(e.target.checked)} />} label={t('robustness.filters.failuresOnly')} />
      </Stack>
      <Table size="small" aria-label={t('robustness.matrix.label')}>
        <TableHead>
          <TableRow>
            <TableCell>{t('robustness.matrix.condition')}</TableCell>
            {columns.map((c) => (
              <TableCell key={columnKey(c.task, c.budget)}>{taskLabel(c.task, c.budget)}</TableCell>
            ))}
          </TableRow>
        </TableHead>
        <TableBody>
          {rows.map((cond) => (
            <TableRow key={cond.condition_id} data-condition={cond.condition_id}>
              <TableCell>
                <strong>{cond.condition_id}</strong> <Chip size="small" variant="outlined" label={t(cond.role === 'source' ? 'robustness.matrix.source' : 'robustness.matrix.target')} />
                <Typography variant="caption" color="text.secondary" sx={{ display: 'block' }}>
                  {cond.dataset_id} · {Object.entries(cond.values ?? {}).map(([k, v]) => `${k}=${String(v)}`).join(', ')}
                </Typography>
              </TableCell>
              {columns.map((c) => {
                const agg = aggregates.find((a) => a.condition_id === cond.condition_id && a.task === c.task && (a.budget_samples ?? null) === c.budget)
                const stats = agg && metric ? agg.metrics?.[metric] : undefined
                return (
                  <TableCell key={columnKey(c.task, c.budget)} data-cell={`${cond.condition_id}:${columnKey(c.task, c.budget)}`}>
                    {!agg ? (
                      t('robustness.matrix.noCell')
                    ) : (
                      <Stack spacing={0.5}>
                        {stats ? (
                          <span>
                            {fmt(stats.mean)}
                            {stats.std !== null && stats.std !== undefined ? ` ± ${fmt(stats.std)}` : ''} (n={stats.n})
                          </span>
                        ) : agg.n_ok > 0 ? (
                          <span>{t('robustness.matrix.noNumber')}</span>
                        ) : null}
                        {agg.n_failed > 0 ? (
                          <Stack spacing={0.25} data-testid="cell-failure">
                            <Chip size="small" color="error" label={t('robustness.matrix.failed', { failed: agg.n_failed, seeds: agg.n_seeds })} sx={{ alignSelf: 'flex-start' }} />
                            {failuresOf(agg).map((why) => (
                              <Typography key={why} variant="caption" color="error">
                                {why}
                              </Typography>
                            ))}
                          </Stack>
                        ) : null}
                        {agg.target_reached_fraction !== null && agg.target_reached_fraction !== undefined ? (
                          <Typography variant="caption" color="text.secondary">
                            {t('robustness.matrix.target.reached', { pct: `${Math.round(agg.target_reached_fraction * 100)}%` })}
                          </Typography>
                        ) : null}
                      </Stack>
                    )}
                  </TableCell>
                )
              })}
            </TableRow>
          ))}
        </TableBody>
      </Table>
      <Typography variant="h3">{t('robustness.cost.label')}</Typography>
      <Table size="small" aria-label={t('robustness.cost.label')}>
        <TableHead>
          <TableRow>
            <TableCell>{t('robustness.cost.task')}</TableCell>
            <TableCell>{t('robustness.matrix.condition')}</TableCell>
            <TableCell align="right">{t('robustness.cost.newSamples')}</TableCell>
            <TableCell align="right">{t('robustness.cost.wallClock')}</TableCell>
            <TableCell>{t('robustness.cost.device')}</TableCell>
            <TableCell>{t('robustness.cost.seeds')}</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {aggregates
            .filter((a) => tasks.includes(a.task))
            .map((a) => (
              <TableRow key={`${a.condition_id}:${columnKey(a.task, a.budget_samples)}`}>
                <TableCell>{taskLabel(a.task, a.budget_samples)}</TableCell>
                <TableCell>{a.condition_id}</TableCell>
                <TableCell align="right">{a.new_samples}</TableCell>
                <TableCell align="right">{fmt(a.mean_wall_clock_s, 1)}</TableCell>
                <TableCell>{report.device}</TableCell>
                <TableCell>
                  {a.n_ok} / {a.n_failed}
                </TableCell>
              </TableRow>
            ))}
        </TableBody>
      </Table>
      <Typography variant="h3">{t('robustness.limitations')}</Typography>
      <ul data-testid="limitations">
        {(report.limitations ?? []).map((lim) => (
          <li key={lim}>{lim}</li>
        ))}
      </ul>
      <Stack direction="row" spacing={2} sx={{ alignItems: 'center', flexWrap: 'wrap' }} useFlexGap>
        <DownloadLink button size="small"  href={`${API}/adaptation/reports/${encodeURIComponent(report.plan_sha256.slice(0, 12))}?format=md`} download>
          {t('robustness.download')}
        </DownloadLink>
        <Typography variant="caption" color="text.secondary">
          {t('robustness.hashes', { report: report.report_sha256?.slice(0, 12) ?? '', plan: report.plan_sha256.slice(0, 12), card: report.condition_set.card_sha256?.slice(0, 12) ?? '' })}
        </Typography>
      </Stack>
    </Stack>
  )
}
