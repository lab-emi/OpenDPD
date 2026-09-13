import { useState } from 'react'
import { Link as RouterLink, useSearchParams } from 'react-router'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Alert, Autocomplete, Box, Button, Checkbox, Chip, Dialog, DialogActions, DialogContent, DialogTitle, FormControlLabel, Link, MenuItem, Paper, Stack, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, TextField, Typography } from '@mui/material'
import { api, API, WEB_MODE } from '@/api/client'
import type { components } from '@/api/schema'
import { useMetricProfiles, useRuns } from '@/api/hooks'
import { DownloadLink } from '@/components/DownloadLink'
import { ErrorState, LoadingState } from '@/components/StateBlock'
import { PlotlyChart, seriesSymbol } from '@/components/PlotlyChart'
import { message, t, type MessageKey } from '@/i18n'

type Report = components['schemas']['HardwareCostReport']
type Entry = components['schemas']['HardwareCostEntry']
type Values = components['schemas']['CostValues']
type Draft = components['schemas']['HardwareCostDraft']
type Attachment = components['schemas']['CostAttachment']
const axes: Array<{ key: keyof Values; unit: string; scale: number }> = [
  { key: 'constant_bytes', unit: 'KiB', scale: 1 / 1024 }, { key: 'mac_per_sample', unit: 'MAC/sample', scale: 1 },
  { key: 'state_bytes', unit: 'bytes', scale: 1 }, { key: 'input_buffer_bytes', unit: 'bytes', scale: 1 },
  { key: 'table_bytes', unit: 'KiB', scale: 1 / 1024 }, { key: 'table_lookups_per_sample', unit: 'lookups/sample', scale: 1 },
  { key: 'dynamic_skip_fraction', unit: '%', scale: 100 }, { key: 'throughput_samples_s', unit: 'MS/s', scale: 1e-6 },
  { key: 'latency_s', unit: 'µs', scale: 1e6 }, { key: 'power_w', unit: 'W', scale: 1 },
  { key: 'energy_j_per_sample', unit: 'nJ/sample', scale: 1e9 }, { key: 'area_mm2', unit: 'mm²', scale: 1 }, { key: 'lut_count', unit: 'LUT', scale: 1 },
]
const sources: Draft['source_kind'][] = ['operation_count', 'cpu_reference_timing', 'fpga_synthesis', 'fpga_board_measurement', 'asic_synthesis', 'post_layout_simulation', 'chip_measurement']
const label = (key: string) => t(`hardware.${key}` as MessageKey)
const axisLabel = (axis: typeof axes[number]) => `${label(axis.key)} (${axis.unit})`
const palette = ['#2c639b', '#b95422', '#298157', '#8047a6', '#a34672', '#487a88']
const numeric = (n: number) => Number.isFinite(n) ? n.toLocaleString(undefined, { maximumSignificantDigits: 6 }) : t('common.na')

export function HardwareCostsPage() {
  const [params, setParams] = useSearchParams()
  const [adding, setAdding] = useState(false)
  const [xKey, setXKey] = useState<keyof Values>('constant_bytes')
  const [metric, setMetric] = useState('')
  const runs = useRuns()
  const profiles = useMetricProfiles()
  const profile = params.get('profile') ?? 'general-spectral-v1'
  const ids = (params.get('runs') ?? '').split(',').filter(Boolean).slice(0, 8)
  const query = new URLSearchParams({ runs: ids.join(','), profile }).toString()
  const report = useQuery({ queryKey: ['hardware-costs', ids.join(','), profile], queryFn: () => api.get<Report>(`/hardware/costs?${query}`), enabled: ids.length > 0 && !WEB_MODE })
  if (WEB_MODE) return null
  const options = runs.data?.filter(r => r.status === 'succeeded' && r.result_id) ?? []
  const definition = profiles.data?.find(p => p.profile_id === profile && p.validation !== 'pending_cross_validation')
  const metrics = definition?.metrics ?? []
  const selectedMetric = metrics.find(m => m.name === metric) ?? metrics.find(m => m.name === 'ACPR_L') ?? metrics[0]
  const entries = report.data?.entries ?? []
  const usableAxes = axes.filter(a => entries.some(e => !e.stale && e.values[a.key] != null))
  const x = usableAxes.find(a => a.key === xKey) ?? usableAxes[0]
  const points = definition && x && selectedMetric ? entries.flatMap((entry, index) => {
    const value = entry.values[x.key]
    const y = entry.metrics.find(m => m.name === selectedMetric.name && m.status === 'ok')?.value
    return value != null && y != null && !entry.stale ? [{ entry, index, x: value * x.scale, y }] : []
  }) : []
  return <Stack spacing={2}>
    <Stack direction="row" sx={{ alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap' }}><Typography variant="h1">{label('title')}</Typography><Button variant="contained" disabled={!ids.length || !definition} onClick={() => setAdding(true)}>{label('addReport')}</Button></Stack>
    <Alert severity="info">{label('intro')}</Alert>
    <Stack direction={{ xs: 'column', md: 'row' }} spacing={1}>
      <Autocomplete multiple sx={{ flex: 1 }} options={options} value={options.filter(r => ids.includes(r.run_id))} getOptionLabel={r => r.name ? `${r.name} · ${r.run_id}` : r.run_id} isOptionEqualToValue={(a, b) => a.run_id === b.run_id} onChange={(_e, selected) => setParams({ runs: selected.slice(0, 8).map(r => r.run_id).join(','), profile })} renderInput={props => <TextField {...props} label={label('runs')} />} />
      <TextField select disabled={!profiles.data} label={t('sweep.profile')} value={definition ? profile : ''} sx={{ minWidth: 230 }} onChange={e => setParams({ runs: ids.join(','), profile: e.target.value })}><MenuItem value="">{t('common.na')}</MenuItem>{profiles.data?.filter(p => p.validation !== 'pending_cross_validation').map(p => <MenuItem key={p.profile_id} value={p.profile_id}>{p.profile_id}</MenuItem>)}</TextField>
    </Stack>
    {!ids.length && <Typography>{label('chooseRuns')}</Typography>}
    {ids.length > 0 && report.isPending && <LoadingState />}{report.isError && <ErrorState error={report.error} />}
    {report.data && <>
      {Object.entries(report.data.missing).map(([id, reason]) => <Alert severity="warning" key={id}>{id}: {reason}</Alert>)}
      {report.data.comparison_notes?.map(note => <Alert key={note} severity="warning">{note}</Alert>)}
      {entries.some(e => e.synthetic) && <Alert severity="warning">{label('syntheticPlotNotice')}</Alert>}
      <Stack direction="row" spacing={1} useFlexGap sx={{ flexWrap: 'wrap' }}>
        <TextField size="small" select label={label('costAxis')} value={x?.key ?? ''} sx={{ minWidth: 245 }} onChange={e => setXKey(e.target.value as keyof Values)}>{usableAxes.map(a => <MenuItem key={a.key} value={a.key}>{axisLabel(a)}</MenuItem>)}</TextField>
        <TextField size="small" select label={label('rfAxis')} value={selectedMetric?.name ?? ''} sx={{ minWidth: 240 }} onChange={e => setMetric(e.target.value)}>{metrics.map(m => <MenuItem key={m.name} value={m.name}>{message(m.display_name)} ({m.unit})</MenuItem>)}</TextField>
        <DownloadLink button href={`${API}/hardware/costs?${query}`} download="hardware-cost-ledger.json">{label('download')}</DownloadLink>
      </Stack>
      {points.length > 0 && x && selectedMetric ? <PlotlyChart title={label('title')} data-testid="hardware-scatter" traces={points.map(p => ({ x: [p.x], y: [p.y], name: `${p.index + 1}. ${p.entry.title}`, mode: 'markers', marker: { size: 12, color: palette[p.index % palette.length], symbol: seriesSymbol(p.index) } }))} layout={{ xaxis: { title: { text: axisLabel(x) } }, yaxis: { title: { text: `${message(selectedMetric.display_name)} (${selectedMetric.unit})` } }, height: 380, showlegend: false }} /> : <Typography>{label('missingAxis')}</Typography>}
      <Typography variant="body2">{label('scatterHelp')}</Typography>
      {entries.map((entry, index) => <CostCard key={entry.entry_id} entry={entry} index={index} />)}
      <Box component="details"><Typography component="summary">{t('sweep.protocol')}</Typography><ul>{report.data.notes.map(note => <li key={note}>{note}</li>)}</ul></Box>
    </>}
    {adding && <AddCostDialog ids={ids} profile={profile} onClose={() => setAdding(false)} />}
  </Stack>
}

function CostCard({ entry, index }: { entry: Entry; index: number }) {
  return <Paper variant="outlined" sx={{ p: 2, borderLeft: 4, borderLeftColor: palette[index % palette.length] }}><Stack spacing={1}>
    <Stack direction="row" spacing={1} useFlexGap sx={{ alignItems: 'center', flexWrap: 'wrap' }}><Typography variant="h3">{index + 1}. {entry.title}</Typography><Chip size="small" label={label(entry.source_kind)} /><Chip size="small" variant="outlined" label={message(entry.rf_evidence_type)} />{entry.synthetic && <Chip size="small" color="warning" label="SYNTHETIC / MOCK" />}{entry.stale && <Chip size="small" color="error" label={label('stale')} />}</Stack>
    <Link component={RouterLink} to={`/results/${entry.run_id}?profile=${entry.profile_id}`}>{entry.run_id}</Link>
    <Typography variant="body2">{entry.metric_basis} · {entry.execution_semantics}</Typography>
    <Typography variant="body2">{entry.target}{entry.process ? ` · ${entry.process}` : ''}{entry.clock_hz ? ` · ${numeric(entry.clock_hz / 1e6)} MHz` : ''} · {label('batch')}: {entry.batch_size ?? t('common.na')}</Typography>
    <Typography variant="body2">{entry.boundary}</Typography>
    <TableContainer role="region" tabIndex={0} aria-label={`${label('costs')} ${index + 1}`}><Table size="small"><TableHead><TableRow><TableCell>{label('quantity')}</TableCell><TableCell align="right">{label('value')}</TableCell></TableRow></TableHead><TableBody>
      {axes.map(a => <TableRow key={a.key}><TableCell>{axisLabel(a)}</TableCell><TableCell align="right">{entry.values[a.key] == null ? t('common.na') : numeric(entry.values[a.key]! * a.scale)}</TableCell></TableRow>)}
    </TableBody></Table></TableContainer>
    <Box component="details"><Typography component="summary">{label('precisionSource')}</Typography><Stack spacing={1} sx={{ pt: 1 }}>
      {entry.precision.map(p => <Typography key={p.module} variant="body2">{p.module}: {p.format}</Typography>)}
      <Typography variant="body2">{label('parameters')}: {entry.stored_parameter_count ?? t('common.na')} · {label('tensorElements')}: {entry.stored_tensor_elements ?? t('common.na')}</Typography>
      <Typography variant="body2">{label('lookahead')}: {entry.lookahead_samples ?? t('common.na')} {t('sweep.samples')} · {entry.lookahead_lower_bound_s == null ? t('common.na') : `${numeric(entry.lookahead_lower_bound_s * 1e9)} ns`} · {label('warmup')}: {entry.warmup_samples ?? t('common.na')} {t('sweep.samples')}</Typography>
      <Typography variant="body2">{label('activity')}: {entry.activity}</Typography>
      <Typography variant="caption" sx={{ overflowWrap: 'anywhere' }}>Weights SHA256: {entry.weights_sha256}<br />Result SHA256: {entry.result_sha256}<br />Source SHA256: {entry.source_file.sha256}</Typography>
      {entry.source_type !== 'checkpoint_shapes' && <DownloadLink href={`${API}/hardware/reports/${entry.source_file.sha256}/download`} download>{label('downloadSource')}</DownloadLink>}
      <ul>{entry.limitations.filter(Boolean).map(note => <li key={note}>{note}</li>)}</ul>
    </Stack></Box>
  </Stack></Paper>
}

function AddCostDialog({ ids, profile, onClose }: { ids: string[]; profile: string; onClose: () => void }) {
  const qc = useQueryClient()
  const [run, setRun] = useState(ids[0] ?? '')
  const [title, setTitle] = useState('')
  const [source, setSource] = useState<Draft['source_kind']>('operation_count')
  const [target, setTarget] = useState('')
  const [process, setProcess] = useState('')
  const [clock, setClock] = useState('')
  const [batch, setBatch] = useState('1')
  const [activity, setActivity] = useState('')
  const [boundary, setBoundary] = useState('')
  const [precision, setPrecision] = useState('')
  const [values, setValues] = useState<Partial<Record<keyof Values, string>>>({})
  const [synthetic, setSynthetic] = useState(false)
  const [receipt, setReceipt] = useState<Attachment>()
  const [notes, setNotes] = useState('')
  const upload = useMutation({ mutationFn: (file: File) => { const form = new FormData(); form.append('file', file); return api.upload<Attachment>('/hardware/reports/upload', form) }, onSuccess: setReceipt })
  const create = useMutation({ mutationFn: () => {
    const modules = precision.split('\n').filter(line => line.trim()).map(line => { const colon = line.indexOf(':'); if (colon <= 0 || !line.slice(colon + 1).trim()) throw new Error(label('precisionHelp')); return { module: line.slice(0, colon).trim(), format: line.slice(colon + 1).trim() } })
    const numbers = [...Object.values(values).filter(value => value?.trim()), batch, ...(clock.trim() ? [clock] : [])]
    if (numbers.some(value => !Number.isFinite(Number(value)))) throw new Error(label('invalidNumber'))
    return api.post<Entry>('/hardware/costs', { run_id: run, profile_id: profile, title, source_kind: source, report_sha256: receipt!.sha256, target, process: process || null, clock_hz: clock.trim() ? Number(clock) * 1e6 : null, batch_size: Number(batch), activity, boundary, precision: modules, synthetic, notes,
      values: Object.fromEntries(axes.filter(a => values[a.key]?.trim()).map(a => [a.key, Number(values[a.key]) / a.scale])) })
  }, onSuccess: () => { void qc.invalidateQueries({ queryKey: ['hardware-costs'] }); onClose() } })
  const busy = upload.isPending || create.isPending
  return <Dialog open onClose={busy ? undefined : onClose} fullWidth maxWidth="md" aria-labelledby="hardware-add-title"><DialogTitle id="hardware-add-title">{label('addReport')}</DialogTitle><DialogContent dividers><Stack spacing={2}>
    <Alert severity="info">{label('reportHelp')}</Alert>
    <Button component="label" variant="outlined" disabled={busy}>{label('upload')}<input hidden type="file" accept=".json,.csv,.txt,.rpt,.pdf" aria-label={label('upload')} onChange={e => { setReceipt(undefined); const file = e.target.files?.[0]; if (file) upload.mutate(file) }} /></Button>
    {receipt && <Typography variant="caption" sx={{ overflowWrap: 'anywhere' }}>SHA256: {receipt.sha256} · {numeric(receipt.size_bytes / 1024)} KiB</Typography>}
    <TextField select size="small" label={label('linkedRun')} value={run} onChange={e => setRun(e.target.value)}>{ids.map(id => <MenuItem key={id} value={id}>{id}</MenuItem>)}</TextField>
    <TextField size="small" label={label('reportTitle')} value={title} onChange={e => setTitle(e.target.value)} />
    <TextField select size="small" label={label('source')} value={source} onChange={e => setSource(e.target.value as Draft['source_kind'])}>{sources.map(s => <MenuItem key={s} value={s}>{label(s)}</MenuItem>)}</TextField>
    <TextField size="small" label={label('target')} value={target} onChange={e => setTarget(e.target.value)} />
    <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1}><TextField size="small" label={label('process')} value={process} onChange={e => setProcess(e.target.value)} /><TextField size="small" label={label('clock')} value={clock} onChange={e => setClock(e.target.value)} /><TextField size="small" label={label('batch')} value={batch} onChange={e => setBatch(e.target.value)} /></Stack>
    <TextField size="small" label={label('activity')} value={activity} onChange={e => setActivity(e.target.value)} />
    <TextField size="small" label={label('boundary')} multiline value={boundary} onChange={e => setBoundary(e.target.value)} />
    <TextField size="small" label={label('precision')} multiline minRows={3} helperText={label('precisionHelp')} value={precision} onChange={e => setPrecision(e.target.value)} />
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 1 }}>{axes.map(a => <TextField key={a.key} size="small" label={axisLabel(a)} value={values[a.key] ?? ''} onChange={e => setValues(v => ({ ...v, [a.key]: e.target.value }))} />)}</Box>
    <FormControlLabel control={<Checkbox checked={synthetic} onChange={e => setSynthetic(e.target.checked)} />} label={label('synthetic')} />
    <TextField size="small" label={label('notes')} multiline value={notes} onChange={e => setNotes(e.target.value)} />
    {upload.isError && <ErrorState error={upload.error} />}{create.isError && <ErrorState error={create.error} />}
  </Stack></DialogContent><DialogActions><Button onClick={onClose} disabled={busy}>{t('form.cancel')}</Button><Button variant="contained" disabled={busy || !receipt || !title.trim() || !target.trim() || !activity.trim() || !boundary.trim() || !precision.trim()} onClick={() => create.mutate()}>{label('record')}</Button></DialogActions></Dialog>
}
