import Alert from '@mui/material/Alert'
import Button from '@mui/material/Button'
import Link from '@mui/material/Link'
import LinearProgress from '@mui/material/LinearProgress'
import Stack from '@mui/material/Stack'
import Table from '@mui/material/Table'
import TablePagination from '@mui/material/TablePagination'
import TextField from '@mui/material/TextField'
import TableBody from '@mui/material/TableBody'
import TableCell from '@mui/material/TableCell'
import TableHead from '@mui/material/TableHead'
import TableRow from '@mui/material/TableRow'
import ToggleButton from '@mui/material/ToggleButton'
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup'
import Typography from '@mui/material/Typography'
import { useEffect, useState } from 'react'
import { Link as RouterLink, useSearchParams } from 'react-router'
import { useImportPackage, useRunCount, useRuns } from '@/api/hooks'
import type { RunStatus, RunView } from '@/api/types'
import { formatDateTime, t } from '@/i18n'
import { StatusChip, statusLabel } from '@/components/StatusChip'
import { EmptyState, ErrorState, LoadingState } from '@/components/StateBlock'

const FILTERS: Array<RunStatus | 'all'> = ['all', 'running', 'queued', 'succeeded', 'failed']

function progressText(run: RunView): string {
  if (typeof run.progress_epoch === 'number' && run.progress_epoch > 0 && typeof run.progress_total_epochs === 'number') {
    return t('run.progress', { epoch: run.progress_epoch, total: run.progress_total_epochs })
  }
  return run.status === 'running' ? t('run.progress.none') : ''
}

export function RunTable({ runs }: { runs: RunView[] }) {
  return (
    <Table size="small" aria-label={t('experiments.title')}>
      <TableHead>
        <TableRow>
          <TableCell>{t('experiments.columns.run')}</TableCell>
          <TableCell>{t('experiments.columns.task')}</TableCell>
          <TableCell>{t('experiments.columns.model')}</TableCell>
          <TableCell>{t('experiments.columns.dataset')}</TableCell>
          <TableCell>{t('experiments.columns.status')}</TableCell>
          <TableCell sx={{ minWidth: 160 }}>{t('experiments.columns.progress')}</TableCell>
          <TableCell>{t('experiments.columns.created')}</TableCell>
        </TableRow>
      </TableHead>
      <TableBody>
        {runs.map((r) => {
          const pct = typeof r.progress_epoch === 'number' && r.progress_total_epochs ? (r.progress_epoch / r.progress_total_epochs) * 100 : null
          return (
            <TableRow key={r.run_id} hover data-run-id={r.run_id}>
              <TableCell>
                <Link component={RouterLink} to={`/runs/${encodeURIComponent(r.run_id)}`}>
                  {r.name || r.run_id}
                </Link>
                {r.name && (
                  <Typography sx={{ display: 'block' }} variant="caption" color="text.secondary">
                    {r.run_id}
                  </Typography>
                )}
              </TableCell>
              <TableCell>{r.task}</TableCell>
              <TableCell>{r.model_key ?? t('common.na')}</TableCell>
              <TableCell>{r.dataset_id ?? t('common.na')}</TableCell>
              <TableCell>
                <StatusChip status={r.status} stale={r.heartbeat_stale} />
              </TableCell>
              <TableCell>
                {pct !== null && r.status === 'running' && <LinearProgress variant="determinate" value={pct} aria-label={progressText(r)} sx={{ mb: 0.5 }} />}
                <Typography variant="caption">{progressText(r)}</Typography>
              </TableCell>
              <TableCell>
                <time dateTime={r.created_at}>{formatDateTime(r.created_at)}</time>
              </TableCell>
            </TableRow>
          )
        })}
      </TableBody>
    </Table>
  )
}

/** Package import: a hidden file input behind a button; the report says what arrived and what is still missing. */
function ImportPackage() {
  const importPackage = useImportPackage()
  const report = importPackage.data
  const missing = report?.missing ?? []
  const imported = report?.imported_runs ?? []
  return (
    <Stack spacing={1} sx={{ alignItems: 'flex-start' }}>
      <Button component="label" variant="outlined" disabled={importPackage.isPending} title={t('experiments.import.help')}>
        {t('experiments.import')}
        <input
          hidden
          type="file"
          accept="application/zip,.zip"
          data-testid="import-package"
          onChange={(e) => {
            const f = e.target.files?.[0]
            if (f) importPackage.mutate(f)
            e.target.value = ''
          }}
        />
      </Button>
      {importPackage.isError && <ErrorState error={importPackage.error} />}
      {report && (
        <Alert severity={missing.length > 0 ? 'warning' : 'success'} data-testid="import-report">
          {t('experiments.import.done', { runs: imported.join(', '), dataset: report.dataset_id, status: report.dataset_status.replace('_', ' ') })} {report.note}
          {missing.length > 0 && (
            <Typography variant="body2" component="div" sx={{ mt: 1 }}>
              <strong>{t('experiments.import.missing')}:</strong>
              <ul style={{ margin: '4px 0 0', paddingLeft: 18 }}>
                {missing.map((line) => (
                  <li key={line}>{line}</li>
                ))}
              </ul>
              <code>{report.evaluate_command}</code>
            </Typography>
          )}
          <Typography variant="body2" component="div" sx={{ mt: 1 }}>
            <Link component={RouterLink} to={`/runs/${encodeURIComponent(report.run_id)}`}>
              {t('experiments.import.open')}
            </Link>
          </Typography>
        </Alert>
      )}
    </Stack>
  )
}

const PAGE_SIZES = [25, 50, 100]

export function ExperimentsPage() {
  const [params, setParams] = useSearchParams()
  const raw = params.get('status')
  const filter: RunStatus | 'all' = FILTERS.includes(raw as RunStatus) ? (raw as RunStatus) : 'all'
  const q = params.get('q') ?? ''
  const page = Math.max(0, Number(params.get('page') ?? 0) || 0)
  const sizeParam = Number(params.get('size') ?? 50)
  const size = PAGE_SIZES.includes(sizeParam) ? sizeParam : 50
  const [draft, setDraft] = useState(q)
  useEffect(() => setDraft(q), [q])
  const update = (patch: Record<string, string | null>) => {
    const next = new URLSearchParams(params)
    for (const [k, v] of Object.entries(patch)) {
      if (v === null || v === '' || (k === 'page' && v === '0') || (k === 'size' && v === '50') || (k === 'status' && v === 'all')) next.delete(k)
      else next.set(k, v)
    }
    setParams(next)
  }
  // the search is server side; typing is debounced so a 1000-run history is not queried per keystroke
  useEffect(() => {
    if (draft === q) return
    const timer = window.setTimeout(() => update({ q: draft, page: null }), 300)
    return () => window.clearTimeout(timer)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draft])
  const status = filter === 'all' ? undefined : filter
  const runs = useRuns(status, { q: q || undefined, limit: size, offset: page * size })
  const count = useRunCount(status, q || undefined)
  return (
    <Stack spacing={2}>
      <Stack sx={{ alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap' }} direction="row" useFlexGap>
        <Typography variant="h1">{t('experiments.title')}</Typography>
        <Button component={RouterLink} to="/experiments/new" variant="contained">
          {t('experiments.new')}
        </Button>
      </Stack>
      <ImportPackage />
      <Stack direction="row" spacing={2} useFlexGap sx={{ alignItems: 'center', flexWrap: 'wrap' }}>
        <ToggleButtonGroup
          size="small"
          exclusive
          value={filter}
          aria-label={t('experiments.columns.status')}
          onChange={(_, v: RunStatus | 'all' | null) => {
            if (v) update({ status: v, page: null })
          }}
        >
          {FILTERS.map((f) => (
            <ToggleButton key={f} value={f}>
              {f === 'all' ? t('experiments.filter.all') : statusLabel(f)}
            </ToggleButton>
          ))}
        </ToggleButtonGroup>
        <TextField size="small" label={t('experiments.search')} value={draft} onChange={(e) => setDraft(e.target.value)} slotProps={{ htmlInput: { 'aria-label': t('experiments.search') } }} sx={{ minWidth: 260 }} />
      </Stack>
      {runs.isPending ? (
        <LoadingState />
      ) : runs.isError ? (
        <ErrorState error={runs.error} onRetry={() => void runs.refetch()} />
      ) : runs.data.length === 0 && page === 0 ? (
        q ? (
          <EmptyState body={t('experiments.noMatch', { q })} />
        ) : (
          <EmptyState
            body={t('experiments.empty')}
            action={
              <Button component={RouterLink} to="/experiments/new" variant="outlined">
                {t('experiments.new')}
              </Button>
            }
          />
        )
      ) : (
        <>
          <RunTable runs={runs.data} />
          <TablePagination
            component="div"
            count={count.data?.count ?? -1}
            page={page}
            rowsPerPage={size}
            rowsPerPageOptions={PAGE_SIZES}
            onPageChange={(_, p) => update({ page: String(p) })}
            onRowsPerPageChange={(e) => update({ size: e.target.value, page: null })}
            getItemAriaLabel={(type) => t(`experiments.page.${type}` as 'experiments.page.next')}
          />
        </>
      )}
    </Stack>
  )
}
