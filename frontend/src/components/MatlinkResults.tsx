import ArrowOutwardIcon from '@mui/icons-material/ArrowOutward'
import CheckCircleOutlineIcon from '@mui/icons-material/CheckCircleOutlined'
import SaveAltIcon from '@mui/icons-material/SaveAlt'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import ButtonBase from '@mui/material/ButtonBase'
import Chip from '@mui/material/Chip'
import Divider from '@mui/material/Divider'
import Grid from '@mui/material/Grid'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import TablePagination from '@mui/material/TablePagination'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { useEffect, useMemo, useState } from 'react'
import { Link as RouterLink, useSearchParams } from 'react-router'
import { useArtifactJson, useDatasets, useResult, useRun, useRunCount, useRuns } from '@/api/hooks'
import type { MatlabSession, MatlinkAction, MatlinkRequest, MatlinkTransfer } from '@/api/matlink'
import { isTerminal, type DatasetManifest, type RunView } from '@/api/types'
import { EvidenceBadge } from '@/components/EvidenceBadge'
import { taskLabel } from '@/components/ExperimentTasks'
import { MetricCard } from '@/components/MetricCard'
import type { SpectrumData } from '@/components/ResultCharts'
import { SpectrumPanels } from '@/components/SpectrumPanels'
import { ErrorState, LoadingState } from '@/components/StateBlock'
import { StatusChip } from '@/components/StatusChip'
import { formatDateTime, message, t } from '@/i18n'

type Send = (action: MatlinkAction, payload: MatlinkRequest['payload']) => void
const title = (run: RunView) => run.name || `${taskLabel(run.task)} · ${run.model_key ?? ''}`
const keywords = new Set('break case catch classdef continue else elseif end for function global if otherwise parfor persistent return spmd switch try while'.split(' '))
export const validMatlabName = (name: string) => /^[A-Za-z][A-Za-z0-9_]{0,62}$/.test(name) && !keywords.has(name)

export function MatlinkResults({ session, connected, transfers, busy, send }: { session?: MatlabSession; connected: boolean; transfers: MatlinkTransfer[]; busy: boolean; send: Send }) {
  const [params, setParams] = useSearchParams()
  const [draft, setDraft] = useState('')
  const [search, setSearch] = useState('')
  const [page, setPage] = useState(0)
  const pageSize = 6
  useEffect(() => { const timer = window.setTimeout(() => { setSearch(draft); setPage(0) }, 300); return () => window.clearTimeout(timer) }, [draft])
  const runs = useRuns(undefined, { q: search || undefined, limit: pageSize, offset: page * pageSize })
  const count = useRunCount(undefined, search || undefined)
  const datasets = useDatasets()
  const id = params.get('run') ?? runs.data?.[0]?.run_id ?? ''
  const selected = useRun(id, !!id)
  const select = (runId: string) => setParams(previous => { const next = new URLSearchParams(previous); next.set('run', runId); return next }, { replace: true })
  return <Paper component="section" aria-labelledby="matlink-results-heading" variant="outlined" sx={{ p: { xs: 2, md: 3 } }}>
    <Typography variant="overline" color="primary.main">02 · Studio → MATLAB</Typography>
    <Typography id="matlink-results-heading" variant="h2">{t('matlink.resultsTitle')}</Typography>
    <Typography color="text.secondary" sx={{ mt: 1, mb: 3 }}>{t('matlink.reviewHelp')}</Typography>
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: 'minmax(0,1fr)', lg: '300px minmax(0,1fr)' }, gap: 3, alignItems: 'start' }}>
      <Stack spacing={1.5}>
        <TextField size="small" fullWidth label={t('matlink.search')} value={draft} onChange={e => setDraft(e.target.value)} />
        {runs.isPending && <LoadingState />}
        {runs.isError && <ErrorState error={runs.error} onRetry={() => void runs.refetch()} />}
        {runs.data?.length === 0 && <Typography color="text.secondary">{t(search ? 'matlink.noMatches' : 'matlink.noRuns')}</Typography>}
        <Stack spacing={1.5} sx={{ maxHeight: { xs: 280, lg: 660 }, overflowY: 'auto', p: .25 }}>
        {runs.data?.map(run => <ButtonBase key={run.run_id} onClick={() => select(run.run_id)} aria-pressed={id === run.run_id} sx={{ display: 'block', textAlign: 'left', p: 1.75, border: 1, borderRadius: 1.5, borderColor: id === run.run_id ? 'primary.main' : 'divider', bgcolor: id === run.run_id ? 'action.selected' : 'transparent', '&.Mui-focusVisible': { outline: '2px solid', outlineColor: 'primary.main' } }}>
          <Typography sx={{ fontWeight: 650, overflowWrap: 'anywhere' }}>{title(run)}</Typography>
          <Typography variant="caption" color="text.secondary" component="div" sx={{ overflowWrap: 'anywhere', my: .75 }}>{datasets.data?.find(d => d.dataset_id === run.dataset_id)?.display_name ?? run.dataset_id}</Typography>
          <Stack direction="row" useFlexGap sx={{ gap: .75, flexWrap: 'wrap', alignItems: 'center' }}><StatusChip status={run.status} />
            {transfers.some(x => x.action === 'import_result' && x.payload.run_id === run.run_id && x.status === 'succeeded') && <CheckCircleOutlineIcon fontSize="small" color="success" titleAccess={t('matlink.inMatlab')} />}
          </Stack>
          <Typography variant="caption" color="text.secondary">{formatDateTime(run.created_at)}</Typography>
        </ButtonBase>)}
        </Stack>
        {(count.data?.count ?? 0) > pageSize && <TablePagination component="div" rowsPerPageOptions={[pageSize]} count={count.data?.count ?? 0} rowsPerPage={pageSize} page={page} onPageChange={(_, p) => setPage(p)} sx={{ '& .MuiTablePagination-toolbar': { p: 0, flexWrap: 'wrap' } }} />}
      </Stack>
      {selected.isError ? <ErrorState error={selected.error} onRetry={() => void selected.refetch()} /> : selected.data ?
        <ResultPreview key={`${session?.client_id}:${selected.data.run_id}`} run={selected.data} dataset={datasets.data?.find(d => d.dataset_id === selected.data.dataset_id)} session={session} connected={connected} transfers={transfers} busy={busy} send={send} /> :
        <Box sx={{ p: 4, border: '1px dashed', borderColor: 'divider', borderRadius: 2, textAlign: 'center' }}><SaveAltIcon color="primary" sx={{ fontSize: 40, mb: 2 }} /><Typography variant="h3">{t('matlink.resultsEmpty')}</Typography><Typography color="text.secondary" sx={{ mt: 1 }}>{t('matlink.resultsEmptyHelp')}</Typography></Box>}
    </Box>
  </Paper>
}

function ResultPreview({ run, dataset, session, connected, transfers, busy, send }: { run: RunView; dataset?: DatasetManifest; session?: MatlabSession; connected: boolean; transfers: MatlinkTransfer[]; busy: boolean; send: Send }) {
  const ready = run.status === 'succeeded' && !!run.result_id
  const active = !isTerminal(run.status)
  const report = useResult(run.run_id, ready)
  const spectrum = useArtifactJson<SpectrumData>(run.run_id, 'plot-spectrum', ready)
  const traces = useMemo(() => (spectrum.data?.traces ?? []).map(tr => ({ ...tr, psdDb: tr.psd_db })), [spectrum.data])
  const [variable, setVariable] = useState(() => ('opendpd_' + (run.name || [run.task, run.model_key].filter(Boolean).join('_'))).replace(/[^A-Za-z0-9_]/g, '_').slice(0, 63))
  const bundle = !!session?.capabilities?.includes('result_bundle')
  const transfer = transfers.find(item => item.action === 'import_result' && item.payload.run_id === run.run_id &&
    (!bundle || item.payload.variable === variable))
  const waiting = transfer?.status === 'queued' || transfer?.status === 'waiting'
  const saved = transfer?.status === 'succeeded' && typeof transfer.result?.variable === 'string' ? transfer.result.variable : ''
  const save = () => send('import_result', { run_id: run.run_id, ...(bundle ? { variable, bundle: true } : {}) })
  return <Stack spacing={2.5} sx={{ minWidth: 0 }}>
    <Stack spacing={1}>
      <Typography variant="h3" sx={{ overflowWrap: 'anywhere' }}>{title(run)}</Typography>
      <Stack direction="row" useFlexGap sx={{ gap: 1, flexWrap: 'wrap', alignItems: 'center' }}><StatusChip status={run.status} />
        {dataset?.origin === 'synthetic' && <Chip size="small" variant="outlined" color="warning" label={t('matlink.synthetic')} />}
        {report.data && <><EvidenceBadge evidence={report.data.evidence_type} mock={report.data.is_mock} /><Chip size="small" variant="outlined" label={report.data.metric_profile_id} /></>}
        <Button size="small" endIcon={<ArrowOutwardIcon />} component={RouterLink} to={`/${ready ? 'results' : 'runs'}/${encodeURIComponent(run.run_id)}`}>{t('matlink.viewExperiment')}</Button>
      </Stack>
    </Stack>
    {ready && report.isPending && <LoadingState />}
    {report.isError && <ErrorState error={report.error} onRetry={() => void report.refetch()} />}
    {report.data && <Grid container spacing={1.5}>{report.data.metrics.slice(0, 4).map(metric => <Grid key={metric.name} size={{ xs: 6, xl: 3 }}><MetricCard metric={metric} /></Grid>)}</Grid>}
    {ready && spectrum.data && <Box sx={{ minWidth: 0 }}><SpectrumPanels frequencyHz={spectrum.data.frequency} axis={spectrum.data.axis} traces={traces} bands={spectrum.data.bands ?? undefined} /><Typography variant="caption" color="text.secondary">{message(spectrum.data.estimator)}</Typography></Box>}
    {ready && spectrum.isError && <Typography color="text.secondary" variant="body2">{t('matlink.noSpectrum')}</Typography>}
    {active && <Alert severity="info">{t('matlink.runningHelp')}</Alert>}
    {!active && !ready && <Alert severity="warning">{t('matlink.noResult')}</Alert>}
    <Paper variant="outlined" sx={{ p: 2.5, borderColor: saved ? 'success.main' : 'divider', bgcolor: 'action.hover' }}>
      <Stack spacing={2}>
        <Typography variant="h3">{t('matlink.saveTitle')}</Typography>
        <Typography variant="body2" color="text.secondary">{t(bundle ? 'matlink.bundleContents' : 'matlink.resultContents')}</Typography>
        {bundle && <TextField size="small" fullWidth label={t('matlink.destination')} value={variable} error={!validMatlabName(variable)} helperText={t('matlink.destinationHelp')} disabled={busy || waiting} onChange={e => setVariable(e.target.value)} slotProps={{ htmlInput: { maxLength: 63 } }} />}
        {bundle && <Box component="pre" sx={{ m: 0, fontSize: 12, color: 'text.secondary', whiteSpace: 'pre-wrap', overflowWrap: 'anywhere' }}>{`${variable || 'result'}.metrics\n${variable || 'result'}.configuration\n${variable || 'result'}.plots\n${variable || 'result'}.dataset`}</Box>}
        <Divider />
        {saved && <Alert severity="success">{t('matlink.savedAs', { variable: saved })}</Alert>}
        {transfer?.status === 'failed' && <Alert severity="error">{transfer.error}</Alert>}
        <Stack direction="row" useFlexGap sx={{ gap: 1, flexWrap: 'wrap', alignItems: 'center' }}>
          <Button variant={saved ? 'outlined' : 'contained'} startIcon={<SaveAltIcon />} disabled={!connected || busy || waiting || (!ready && !active) || (bundle && !validMatlabName(variable)) || !!report.data?.is_mock} onClick={save}>{t(waiting ? (transfer.status === 'waiting' ? 'matlink.status.waiting' : 'matlink.status.queued') : saved ? 'matlink.sendAgain' : active ? 'matlink.sendWhenReady' : 'matlink.save')}</Button>
          {saved && <Button endIcon={<ArrowOutwardIcon />} disabled={!connected || busy} onClick={() => send('open_variable', { variable: saved })}>{t('matlink.openVariable')}</Button>}
          <Typography variant="caption" color="text.secondary">{connected ? session?.label : t('matlink.reconnect')}</Typography>
        </Stack>
      </Stack>
    </Paper>
  </Stack>
}
