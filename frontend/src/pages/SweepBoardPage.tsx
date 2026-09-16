import { getQuery } from '@/api/client'
import { useState } from 'react'
import { Link as RouterLink, useLocation, useNavigate, useParams } from 'react-router'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Alert, Box, Button, Chip, Dialog, DialogActions, DialogContent, DialogTitle, FormControlLabel, Checkbox, Link, MenuItem, Paper, Stack, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, TextField, Typography } from '@mui/material'
import { api, API, WEB_MODE } from '@/api/client'
import { useDatasets, useMetricProfiles, useRecipes, useRuns } from '@/api/hooks'
import type { components } from '@/api/schema'
import type { ExperimentConfig, RecipeInfo, ResolvedExperimentConfig } from '@/api/types'
import { ErrorState, LoadingState } from '@/components/StateBlock'
import { DownloadLink } from '@/components/DownloadLink'
import { message, t } from '@/i18n'

type Draft = components['schemas']['SweepDraft']
type Preview = components['schemas']['SweepPreview']
type Board = components['schemas']['SweepRecord']
type Cell = components['schemas']['SweepCell']
type Method = { id: string; recipe: string; epochs: string; parameters: string; config?: ExperimentConfig; paEntry: string; modelKey?: string; quantization?: ExperimentConfig['quantization'] }
type Condition = { id: string; dataset: string; batch: string; value: string; phase: string; original?: components['schemas']['Condition'] }
const newMethod = (i: number): Method => ({ id: `method-${i}`, recipe: 'pa-gru-smoke-v1', epochs: '', parameters: '{}', paEntry: '' })
const unused = (prefix: string, items: Array<{ id: string }>) => { let i = 1; while (items.some(x => x.id === `${prefix}-${i}`)) i++; return i }
const numbers = (s: string) => s.split(',').map(x => { if (!/^\d+$/.test(x.trim())) throw new Error('Enter comma-separated integer values'); return Number(x.trim()) })

export function SweepBoardPage() {
  const { sweepId } = useParams<{ sweepId: string }>()
  const navigate = useNavigate()
  const qc = useQueryClient()
  const location = useLocation()
  const conditionSet = (location.state as { conditionSet?: components['schemas']['ConditionSet'] } | null)?.conditionSet
  const [open, setOpen] = useState(!!conditionSet)
  const [copy, setCopy] = useState<Draft | undefined>(conditionSet ? { protocol_id: 'sweep-v1', title: 'Synthetic condition demonstration', mode: 'cross_condition', condition_set: conditionSet, methods: [{ entry_id: 'pa', recipe_id: 'pa-gru-smoke-v1' }], seeds: [0], tasks: ['zero_update', 'full_retrain'], budgets: [2000], metric_profile_id: 'general-spectral-v1', device: 'cpu', max_runs: 12, max_wall_clock_seconds: 3600 } : undefined)
  const records = useQuery({ queryKey: ['sweeps'], queryFn: getQuery<Board[]>('/sweeps'), enabled: !WEB_MODE, refetchInterval: 2000 })
  const selected = records.data?.find(r => r.sweep_id === sweepId) ?? (!sweepId ? records.data?.[0] : undefined)
  const action = useMutation({ mutationFn: ({ id, verb, resume = false }: { id: string; verb: 'start' | 'cancel'; resume?: boolean }) => api.post<Board>(`/sweeps/${id}/${verb}`, verb === 'start' ? { resume_failed: resume } : undefined), onSuccess: () => void qc.invalidateQueries({ queryKey: ['sweeps'] }) })
  if (WEB_MODE) return null
  return <Stack spacing={2}>
    <Stack direction="row" sx={{ justifyContent: 'space-between', flexWrap: 'wrap' }}><Typography variant="h1">{t('sweep.title')}</Typography><Button variant="contained" onClick={() => { setCopy(undefined); setOpen(true) }}>{t('sweep.create')}</Button></Stack>
    <Typography variant="body2">{t('sweep.intro')}</Typography>
    {records.isPending && <LoadingState />}{records.isError && <ErrorState error={records.error} />}{action.isError && <ErrorState error={action.error} />}
    {records.data && <TextField size="small" select label={t('sweep.saved')} value={selected?.sweep_id ?? ''} onChange={e => navigate(`/sweeps/${e.target.value}`)}>
      {!selected && <MenuItem value="">{t('common.na')}</MenuItem>}{records.data.map(r => <MenuItem key={r.sweep_id} value={r.sweep_id}>{r.preview.draft.title} · {r.status}</MenuItem>)}
    </TextField>}
    {selected && <Paper sx={{ p: 2 }}><Stack spacing={2}>
      <Stack direction="row" spacing={1} useFlexGap sx={{ alignItems: 'center', flexWrap: 'wrap' }}>
        <Chip label={selected.status} color={selected.status === 'complete' ? 'success' : selected.status === 'needs_attention' ? 'warning' : 'default'} />
        <Typography variant="body2">{Math.round(selected.elapsed_seconds)} / {selected.preview.draft.max_wall_clock_seconds} s</Typography>
        {selected.status === 'running' ? <Button onClick={() => action.mutate({ id: selected.sweep_id, verb: 'cancel' })}>{t('sweep.cancel')}</Button> : selected.status !== 'complete' && <Button disabled={action.isPending} onClick={() => action.mutate({ id: selected.sweep_id, verb: 'start', resume: selected.status !== 'ready' })}>{t(selected.status === 'ready' ? 'sweep.start' : 'sweep.resume')}</Button>}
        <Button onClick={() => { setCopy(selected.preview.draft); setOpen(true) }}>{t('sweep.copy')}</Button>
        <DownloadLink href={`${API}/sweeps/${selected.sweep_id}`} download>{t('sweep.download')}</DownloadLink>
      </Stack>
      {selected.reason && <Alert severity="warning">{selected.reason}</Alert>}
      <Typography variant="caption" sx={{ overflowWrap: 'anywhere' }}>{selected.preview.plan_sha256}</Typography>
      <Budget preview={selected.preview} />
      <Matrix cells={selected.cells} />
      <SeedSummary board={selected} />
    </Stack></Paper>}
    {open && <DraftDialog initial={copy} onClose={() => setOpen(false)} onSaved={id => { setOpen(false); void qc.invalidateQueries({ queryKey: ['sweeps'] }); navigate(`/sweeps/${id}`) }} />}
  </Stack>
}

function SeedSummary({ board }: { board: Board }) {
  const summary = useQuery({ queryKey: ['sweep-report', board.sweep_id, board.cells.map(c => c.status).join(',')], queryFn: getQuery<components['schemas']['SweepReport']>(`/sweeps/${board.sweep_id}/report`), enabled: board.cells.some(c => c.status === 'succeeded') })
  const profiles = useMetricProfiles()
  const profile = profiles.data?.find(p => p.profile_id === board.preview.draft.metric_profile_id)
  if (!profile || profile.validation === 'pending_cross_validation') return null
  if (!summary.data) return summary.isError ? <ErrorState error={summary.error} /> : null
  return <Stack spacing={1}>
    <Typography variant="h3">{t('sweep.seedSummary')}</Typography>
    <Typography variant="body2">{t('sweep.seedSummary.help')}</Typography>
    <TableContainer tabIndex={0} role="region" aria-label={t('sweep.seedSummary')}><Table size="small" aria-label={t('sweep.seedSummary')}>
      <TableHead><TableRow><TableCell>{t('sweep.method')}</TableCell><TableCell>{t('sweep.condition')}</TableCell><TableCell>{t('sweep.task')}</TableCell><TableCell>{t('sweep.seeds')}</TableCell><TableCell>{t('sweep.metricStats')}</TableCell></TableRow></TableHead>
      <TableBody>{summary.data.aggregates.map(a => <TableRow key={`${a.entry_id}:${a.condition_id}:${a.task}:${a.budget_samples}`}>
        <TableCell>{a.entry_id}</TableCell><TableCell>{a.condition_id}</TableCell><TableCell>{a.task}{a.budget_samples != null ? ` · ${a.budget_samples}` : ''}</TableCell><TableCell>{a.n_succeeded} / {a.n_requested_seeds}</TableCell>
        <TableCell>{Object.entries(a.metrics).map(([name, s]) => <Typography variant="body2" key={name}>{message(profile.metrics.find(m => m.name === name)?.display_name ?? name)}: {s.mean.toFixed(3)} {s.std != null ? `± ${s.std.toFixed(3)}` : ''} {profile.metrics.find(m => m.name === name)?.unit} (n={s.n})</Typography>)}</TableCell>
      </TableRow>)}</TableBody>
    </Table></TableContainer>
    <DownloadLink href={`${API}/sweeps/${board.sweep_id}/report`} download>{t('sweep.downloadSummary')}</DownloadLink>
  </Stack>
}

function Budget({ preview }: { preview: Preview }) {
  return <Stack spacing={1}>
    <Typography variant="body2">{t('sweep.budget.summary', { train: preview.training_runs, evaluate: preview.evaluation_runs, samples: preview.sample_epochs ?? t('common.na'), seconds: preview.draft.max_wall_clock_seconds })}</Typography>
    {(preview.errors ?? []).map(e => <Alert key={e} severity="error">{e}</Alert>)}
    <Box component="details"><Typography component="summary" variant="body2">{t('sweep.protocol')}</Typography><ul>{(preview.warnings ?? []).map(w => <li key={w}>{w}</li>)}</ul></Box>
  </Stack>
}

function Matrix({ cells }: { cells: Cell[] }) {
  return <TableContainer tabIndex={0} role="region" aria-label={t('sweep.matrix')}><Table size="small" aria-label={t('sweep.matrix')}>
    <TableHead><TableRow>{['method', 'condition', 'task', 'seed', 'samples', 'epochs', 'status', 'attempts'].map(k => <TableCell key={k}>{t(`sweep.${k}` as Parameters<typeof t>[0])}</TableCell>)}</TableRow></TableHead>
    <TableBody>{cells.map(c => <TableRow key={c.cell_id}>
      <TableCell>{c.entry_id}</TableCell><TableCell>{c.condition_id}</TableCell><TableCell>{c.task}</TableCell><TableCell>{c.seed}</TableCell><TableCell>{c.train_samples ?? t('common.na')}</TableCell><TableCell>{c.epochs}</TableCell>
      <TableCell>{c.status}{c.reason && <Typography variant="caption" component="div" color="error" sx={{ maxWidth: 300 }}>{c.reason}</Typography>}</TableCell>
      <TableCell>{(c.attempts ?? []).map(id => <Link key={id} component={RouterLink} to={`/runs/${id}`} sx={{ display: 'block', overflowWrap: 'anywhere' }}>{id}</Link>)}</TableCell>
    </TableRow>)}</TableBody>
  </Table></TableContainer>
}

function DraftDialog({ initial, onClose, onSaved }: { initial?: Draft; onClose: () => void; onSaved: (id: string) => void }) {
  const datasets = useDatasets()
  const recipes = useRecipes()
  const profiles = useMetricProfiles()
  const runs = useRuns()
  const [title, setTitle] = useState(initial?.title ?? t('sweep.defaultTitle'))
  const [mode, setMode] = useState<Draft['mode']>(initial?.mode ?? 'same_condition')
  const [dataset, setDataset] = useState(initial?.dataset?.id ?? '')
  const [pa, setPa] = useState(initial?.fixed_pa_run_id ?? '')
  const [seeds, setSeeds] = useState(initial?.seeds?.join(', ') ?? '0, 1, 2')
  const [budgets, setBudgets] = useState(initial?.budgets?.join(', ') ?? '2000')
  const [maxRuns, setMaxRuns] = useState(initial?.max_runs ?? 64)
  const [maxSeconds, setMaxSeconds] = useState(initial?.max_wall_clock_seconds ?? 3600)
  const [profile, setProfile] = useState(initial?.metric_profile_id ?? 'legacy-opendpd-v1')
  const [device, setDevice] = useState(initial?.device ?? 'cpu')
  const [methods, setMethods] = useState<Method[]>(initial?.methods.map(m => ({ id: m.entry_id, recipe: m.recipe_id ?? '__copy__', config: m.config ?? undefined, epochs: '', parameters: '{}', paEntry: m.pa_entry ?? '' })) ?? [newMethod(1)])
  const [cardId, setCardId] = useState(initial?.condition_set?.set_id ?? 'review-conditions')
  const [dut, setDut] = useState(initial?.condition_set?.device ?? '')
  const [dimension, setDimension] = useState(initial?.condition_set?.dimension ?? 'capture_batch')
  const [source, setSource] = useState(initial?.condition_set?.conditions.findIndex(c => c.role === 'source') ?? 0)
  const [conditions, setConditions] = useState<Condition[]>(initial?.condition_set?.conditions.map(c => ({ id: c.condition_id, dataset: c.dataset_id, batch: c.capture_batch, value: String(c.values?.[initial.condition_set!.dimension] ?? ''), phase: String(c.values?.reflection_phase_deg ?? ''), original: c })) ?? [0, 1].map(i => ({ id: `condition-${i+1}`, dataset: '', batch: '', value: '', phase: '' })))
  const [tasks, setTasks] = useState<NonNullable<Draft['tasks']>>(initial?.tasks ?? ['zero_update', 'few_shot', 'full_retrain'])
  const [candidate, setCandidate] = useState<Preview | null>(null)
  const [localError, setLocalError] = useState<unknown>(null)
  const datasetId = dataset || datasets.data?.[0]?.dataset_id || ''
  const edit = (fn: () => void) => { setCandidate(null); setLocalError(null); fn() }
  const setMethod = (i: number, patch: Partial<Method>) => edit(() => setMethods(ms => ms.map((m, j) => j === i ? { ...m, ...patch } : m)))
  const setCondition = (i: number, patch: Partial<Condition>) => edit(() => setConditions(cs => cs.map((c, j) => j === i ? { ...c, ...patch } : c)))

  function configOf(m: Method, recipe?: RecipeInfo) {
    const base = m.config ?? (recipe ? { task: recipe.task, recipe_id: recipe.recipe_id, dataset: { id: datasetId || 'deferred-data' }, model: recipe.model, training: recipe.training, ...(recipe.task === 'train_dpd' ? { pa_reference: { run_id: pa || 'deferred-pa' } } : {}) } : null)
    if (!base) throw new Error('Choose a recipe or copy a configuration')
    const params = JSON.parse(m.parameters)
    if (!params || typeof params !== 'object' || Array.isArray(params)) throw new Error('Model parameters must be an object')
    const quantization = m.quantization ?? ('quantization' in base ? base.quantization : undefined)
    return { ...base, ...(quantization ? { quantization } : {}), training: { ...base.training, ...(m.epochs ? { epochs: Number(m.epochs) } : {}) }, model: { ...base.model, key: m.modelKey ?? base.model.key, parameters: { ...(m.modelKey && m.modelKey !== base.model.key ? {} : base.model.parameters), ...params } } }
  }
  function body() {
    return { title, mode, dataset: mode === 'same_condition' ? { ...initial?.dataset, id: datasetId } : null,
      fixed_pa_run_id: mode === 'same_condition' ? pa || null : null,
      condition_set: mode === 'cross_condition' ? { ...initial?.condition_set, card_sha256: null, set_id: cardId, device: dut, dimension, conditions: conditions.map((c, i) => ({ ...c.original, condition_id: c.id, dataset_id: c.dataset, role: i === source ? 'source' : 'target', capture_batch: c.batch, values: { ...c.original?.values, [dimension]: ['capture_batch', 'mode', 'load'].includes(dimension) ? c.value : c.value.trim() === '' ? null : Number(c.value), ...(dimension === 'vswr' ? { reflection_phase_deg: c.phase.trim() === '' ? null : Number(c.phase) } : {}) } })) } : null,
      methods: methods.map(m => ({ entry_id: m.id, config: configOf(m, recipes.data?.find(r => r.recipe_id === m.recipe)), pa_entry: mode === 'cross_condition' ? m.paEntry || null : null })),
      seeds: numbers(seeds), budgets: numbers(budgets), tasks, metric_profile_id: profile, device, max_runs: maxRuns, max_wall_clock_seconds: maxSeconds }
  }
  const check = useMutation({ mutationFn: () => api.post<Preview>('/sweeps/preview', body()), onSuccess: setCandidate })
  const save = useMutation({ mutationFn: () => api.post<Board>('/sweeps', candidate!.draft), onSuccess: r => onSaved(r.sweep_id) })
  async function copyRun(index: number, id: string) {
    try { const config = await api.get<ResolvedExperimentConfig>(`/runs/${id}/config`); const { resolution: _resolution, ...input } = config; setMethod(index, { recipe: '__copy__', config: input, epochs: '', parameters: '{}', modelKey: undefined, quantization: input.quantization }) } catch (e) { setLocalError(e) }
  }
  return <Dialog open fullWidth maxWidth="xl" onClose={onClose} aria-labelledby="sweep-create-title"><DialogTitle id="sweep-create-title">{t('sweep.create')}</DialogTitle><DialogContent><Stack spacing={2} sx={{ pt: 1 }}>
    <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
      <TextField label={t('sweep.name')} value={title} onChange={e => edit(() => setTitle(e.target.value))} sx={{ flex: 1 }} />
      <TextField select label={t('sweep.mode')} value={mode} onChange={e => edit(() => setMode(e.target.value as Draft['mode']))} sx={{ minWidth: 230 }}>{['same_condition', 'cross_condition'].map(v => <MenuItem key={v} value={v}>{t(`sweep.${v}` as Parameters<typeof t>[0])}</MenuItem>)}</TextField>
    </Stack>
    {mode === 'same_condition' ? <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
      <TextField fullWidth select label={t('sweep.dataset')} value={datasetId} onChange={e => edit(() => setDataset(e.target.value))}>{datasets.data?.map(d => <MenuItem key={d.dataset_id} value={d.dataset_id}>{d.display_name}</MenuItem>)}</TextField>
      <TextField fullWidth select label={t('sweep.fixedPa')} value={pa} onChange={e => edit(() => setPa(e.target.value))}><MenuItem value="">{t('common.na')}</MenuItem>{runs.data?.filter(r => r.task === 'train_pa' && r.status === 'succeeded' && r.dataset_id === datasetId).map(r => <MenuItem key={r.run_id} value={r.run_id}>{r.name ?? r.run_id}</MenuItem>)}</TextField>
    </Stack> : <Stack spacing={1}>
      <Typography variant="body2">{t('sweep.conditions.help')}</Typography>
      <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1}><TextField label={t('sweep.card')} value={cardId} onChange={e => edit(() => setCardId(e.target.value))} /><TextField label={t('sweep.dut')} value={dut} onChange={e => edit(() => setDut(e.target.value))} /><TextField select label={t('sweep.dimension')} value={dimension} onChange={e => edit(() => setDimension(e.target.value))}>{['capture_batch', 'output_power_dbm', 'carrier_frequency_hz', 'bandwidth_hz', 'temperature_c', 'supply_v', 'mode', 'load', 'vswr'].map(d => <MenuItem key={d} value={d}>{d}</MenuItem>)}</TextField></Stack>
      {conditions.map((c, i) => <Paper variant="outlined" sx={{ p: 1 }} key={i}><Stack direction={{ xs: 'column', md: 'row' }} spacing={1}>
        <TextField size="small" label={t('sweep.condition')} value={c.id} onChange={e => setCondition(i, { id: e.target.value })} sx={{ minWidth: 125, flex: 1 }} />
        <TextField size="small" select label={t('sweep.dataset')} value={c.dataset} onChange={e => setCondition(i, { dataset: e.target.value })} sx={{ minWidth: 190, flex: 1 }}>{datasets.data?.map(d => <MenuItem key={d.dataset_id} value={d.dataset_id}>{d.display_name}</MenuItem>)}</TextField>
        <TextField size="small" label={t('sweep.batch')} value={c.batch} onChange={e => setCondition(i, { batch: e.target.value })} sx={{ minWidth: 130, flex: 1 }} />
        <TextField size="small" label={t('sweep.value')} value={c.value} onChange={e => setCondition(i, { value: e.target.value })} sx={{ width: 130 }} />
        {dimension === 'vswr' && <TextField size="small" label={t('sweep.phase')} value={c.phase} onChange={e => setCondition(i, { phase: e.target.value })} sx={{ width: 130 }} />}
        <FormControlLabel control={<Checkbox checked={source === i} onChange={() => edit(() => setSource(i))} />} label={t('sweep.source')} />
        <Button disabled={conditions.length <= 2} onClick={() => edit(() => { setConditions(cs => cs.filter((_, j) => i !== j)); setSource(0) })}>{t('sweep.remove')}</Button>
      </Stack></Paper>)}
      <Button sx={{ alignSelf: 'flex-start' }} disabled={conditions.length >= 12} onClick={() => edit(() => setConditions(cs => [...cs, { id: `condition-${unused('condition', cs)}`, dataset: '', batch: '', value: '', phase: '' }]))}>{t('sweep.addCondition')}</Button>
      <Stack direction="row" useFlexGap sx={{ flexWrap: 'wrap' }}>{(['zero_update', 'few_shot', 'full_retrain'] as NonNullable<Draft['tasks']>).map(task => <FormControlLabel key={task} control={<Checkbox checked={tasks.includes(task)} onChange={e => edit(() => setTasks(ts => e.target.checked ? [...ts, task] : ts.filter(value => value !== task)))} />} label={t(`robustness.task.${task}`)} />)}<TextField size="small" label={t('sweep.budgets')} value={budgets} onChange={e => edit(() => setBudgets(e.target.value))} /></Stack>
    </Stack>}
    <Typography variant="h3">{t('sweep.methods')}</Typography>
    {methods.map((m, i) => <Paper key={i} variant="outlined" sx={{ p: 1 }}><Stack spacing={1}>
      <Stack direction={{ xs: 'column', md: 'row' }} spacing={1}>
        <TextField size="small" label={t('sweep.method')} value={m.id} onChange={e => setMethod(i, { id: e.target.value })} sx={{ width: 160 }} />
        <TextField size="small" select label={t('sweep.recipe')} value={m.recipe} onChange={e => setMethod(i, { recipe: e.target.value, config: undefined, modelKey: undefined, quantization: undefined })} sx={{ minWidth: 200, flex: 1 }}>{m.config && <MenuItem value="__copy__">{t('sweep.copied')}</MenuItem>}{recipes.data?.filter(r => ['train_pa', 'train_dpd'].includes(r.task)).map(r => <MenuItem key={r.recipe_id} value={r.recipe_id}>{r.title}</MenuItem>)}</TextField>
        <TextField size="small" type="number" label={t('sweep.epochs')} value={m.epochs} placeholder={String(m.config?.training?.epochs ?? recipes.data?.find(r => r.recipe_id === m.recipe)?.training.epochs ?? '')} onChange={e => setMethod(i, { epochs: e.target.value })} sx={{ width: 120 }} />
        <TextField size="small" select label={t('sweep.copyRun')} value="" onChange={e => void copyRun(i, e.target.value)} sx={{ minWidth: 180, flex: 1 }}><MenuItem value="">{t('common.na')}</MenuItem>{runs.data?.filter(r => ['train_pa', 'train_dpd'].includes(r.task)).map(r => <MenuItem key={r.run_id} value={r.run_id}>{r.name ?? r.run_id}</MenuItem>)}</TextField>
        {mode === 'cross_condition' && <TextField size="small" label={t('sweep.paEntry')} value={m.paEntry} onChange={e => setMethod(i, { paEntry: e.target.value })} sx={{ width: 180 }} />}
        <Button disabled={methods.length <= 1} onClick={() => edit(() => setMethods(ms => ms.filter((_, j) => j !== i)))}>{t('sweep.remove')}</Button>
      </Stack>
      <Box component="details"><Typography component="summary" variant="body2">{t('sweep.parameters')}</Typography><Stack spacing={1} sx={{ mt: 1 }}>
        <TextField size="small" label={t('sweep.modelKey')} value={m.modelKey ?? m.config?.model.key ?? recipes.data?.find(r => r.recipe_id === m.recipe)?.model.key ?? ''} onChange={e => setMethod(i, { modelKey: e.target.value })} />
        <TextField fullWidth multiline minRows={2} label={t('sweep.parameters')} value={m.parameters} onChange={e => setMethod(i, { parameters: e.target.value })} />
        {m.config?.initialization && <Alert severity="warning">{t('sweep.copiedInitialization')}<Button onClick={() => setMethod(i, { config: { ...m.config!, initialization: null } })}>{t('sweep.clearInitialization')}</Button></Alert>}
        <FormControlLabel control={<Checkbox checked={!!(m.quantization ?? m.config?.quantization)?.enabled} onChange={e => setMethod(i, { quantization: { n_bits_w: 8, n_bits_a: 8, label: '', ...(m.quantization ?? m.config?.quantization), enabled: e.target.checked, ...(e.target.checked ? {} : { pretrained_run_id: null, pretrained_checkpoint_artifact_id: null, pretrained_checkpoint_sha256: null }) } })} />} label={t('sweep.qat')} />
        {(m.quantization ?? m.config?.quantization)?.enabled && <>
          <Alert severity="info">{t('sweep.qat.help')}</Alert>
          <Stack direction={{ xs: 'column', sm: 'row' }} spacing={1}>{(['n_bits_w', 'n_bits_a'] as const).map(key => <TextField key={key} type="number" size="small" label={t(key === 'n_bits_w' ? 'sweep.weightBits' : 'sweep.activationBits')} value={(m.quantization ?? m.config?.quantization)?.[key] ?? 8} onChange={e => setMethod(i, { quantization: { enabled: true, n_bits_w: 8, n_bits_a: 8, label: '', ...(m.quantization ?? m.config?.quantization), [key]: Number(e.target.value) } })} sx={{ minWidth: 180, width: 210 }} slotProps={{ htmlInput: { min: 2, max: 32 } }} />)}</Stack>
          <TextField select size="small" label={t('sweep.floatPretraining')} value={(m.quantization ?? m.config?.quantization)?.pretrained_run_id ?? ''} onChange={e => setMethod(i, { quantization: { enabled: true, n_bits_w: 8, n_bits_a: 8, label: '', ...(m.quantization ?? m.config?.quantization), pretrained_run_id: e.target.value || null, pretrained_checkpoint_artifact_id: null, pretrained_checkpoint_sha256: null } })}><MenuItem value="">{t('sweep.randomInitialization')}</MenuItem>{runs.data?.filter(r => r.task === 'train_dpd' && r.status === 'succeeded').map(r => <MenuItem key={r.run_id} value={r.run_id}>{r.name ?? r.run_id}</MenuItem>)}</TextField>
        </>}
      </Stack></Box>
    </Stack></Paper>)}
    <Button sx={{ alignSelf: 'flex-start' }} disabled={methods.length >= 8} onClick={() => edit(() => setMethods(ms => [...ms, newMethod(unused('method', ms))]))}>{t('sweep.addMethod')}</Button>
    <Stack direction={{ xs: 'column', md: 'row' }} spacing={1}>
      <TextField label={t('sweep.seeds')} value={seeds} onChange={e => edit(() => setSeeds(e.target.value))} />
      <TextField select label={t('sweep.profile')} value={profile} onChange={e => edit(() => setProfile(e.target.value))} sx={{ minWidth: 230 }}>{profiles.data?.filter(p => p.validation !== 'pending_cross_validation').map(p => <MenuItem key={p.profile_id} value={p.profile_id}>{p.profile_id}</MenuItem>)}</TextField>
      <TextField select sx={{ minWidth: 160 }} label={t('sweep.device')} value={device} onChange={e => edit(() => setDevice(e.target.value as Draft['device']))}>{['cpu', 'cuda', 'mps'].map(v => <MenuItem key={v} value={v}>{v}</MenuItem>)}</TextField>
      <TextField type="number" label={t('sweep.maxRuns')} value={maxRuns} onChange={e => edit(() => setMaxRuns(Number(e.target.value)))} />
      <TextField type="number" label={t('sweep.maxSeconds')} value={maxSeconds} onChange={e => edit(() => setMaxSeconds(Number(e.target.value)))} />
    </Stack>
    {[check.error, save.error, localError].filter(Boolean).map((e, i) => <ErrorState key={i} error={e} />)}
    {candidate && <><Budget preview={candidate} /><Matrix cells={candidate.cells} /></>}
  </Stack></DialogContent><DialogActions><Button onClick={onClose}>{t('common.back')}</Button><Button disabled={check.isPending} onClick={() => check.mutate()}>{t('sweep.preview')}</Button><Button variant="contained" disabled={!candidate || (candidate.errors?.length ?? 0) > 0 || save.isPending} onClick={() => save.mutate()}>{t('sweep.register')}</Button></DialogActions></Dialog>
}
