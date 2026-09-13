import DeleteOutlineIcon from '@mui/icons-material/DeleteOutlined'
import { api } from '@/api/client'
import { MathFormula } from '@/components/MathFormula'
import DownloadIcon from '@mui/icons-material/Download'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import RestartAltIcon from '@mui/icons-material/RestartAlt'
import ArrowForwardIcon from '@mui/icons-material/ArrowForward'
import Accordion from '@mui/material/Accordion'
import AccordionDetails from '@mui/material/AccordionDetails'
import AccordionSummary from '@mui/material/AccordionSummary'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import ButtonBase from '@mui/material/ButtonBase'
import Chip from '@mui/material/Chip'
import Grid from '@mui/material/Grid'
import LinearProgress from '@mui/material/LinearProgress'
import MenuItem from '@mui/material/MenuItem'
import Paper from '@mui/material/Paper'
import Slider from '@mui/material/Slider'
import Stack from '@mui/material/Stack'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { useEffect, useRef, useState } from 'react'
import { Link as RouterLink, useNavigate, useSearchParams } from 'react-router'
import { downloadFile } from '@/api/client'
import { useCustomDatasetImports } from '@/api/hooks'
import { paDefaults, parameterKey, paText, usePAInputs, usePASimulation, usePairedDataset, useSimulatePA, useVirtualPAs,
  type PAParameter, type PASimulation, type VirtualPA } from '@/api/virtualPA'
import { SpectrumPanels } from '@/components/SpectrumPanels'
import { PlotlyChart } from '@/components/PlotlyChart'
import { ErrorState, LoadingState } from '@/components/StateBlock'
import { formatNumber, t } from '@/i18n'
import { useStudioColors } from '@/theme'
import { useStudioWorkflow } from '@/workflow/StudioWorkflow'

const CATEGORIES = ['reference', 'static', 'memory', 'dynamics', 'architecture'] as const
const GROUPS = ['gain', 'memory', 'dynamics', 'architecture'] as const
const validParameter = (p: PAParameter, value: number) => Number.isFinite(value) && value >= p.minimum && value <= p.maximum && (!p.integer || Number.isInteger(value))

/** Display-only KaTeX: catalog coefficients become linked controls, never executable formulas. */
export function PAEquations({ model, active, select }: { model: VirtualPA; active: string; select: (key: string) => void }) {
  const scroll = useRef<HTMLDivElement>(null)
  useEffect(() => {
    const box = scroll.current
    const variable = box?.querySelector<HTMLElement>('[data-testid="equation-' + active + '"]')
    if (!box || !variable) return
    const area = box.getBoundingClientRect(), token = variable.getBoundingClientRect()
    if (token.top < area.top || token.bottom > area.bottom) box.scrollTop += token.top - area.top - box.clientHeight / 2 + token.height / 2
    if (token.left < area.left || token.right > area.right) box.scrollLeft += token.left - area.left - box.clientWidth / 2 + token.width / 2
  }, [active])
  return <Box ref={scroll} sx={{ overflow: 'auto', py: 1, maxHeight: { md: '22vh' } }} role="region" tabIndex={0} aria-label={t('paLibrary.equations')}>
    {(model.equations_latex?.length ? model.equations_latex : model.equations).map((equation, index) =>
      <MathFormula key={index} latex={equation} display active={active} onSelect={select}
        variables={model.parameters.map(p => ({ key: p.key, symbol: p.symbol_latex || p.symbol, label: paText(p.label) }))} />)}
  </Box>
}

function ParameterControl({ parameter: p, value, active, select, change, disabled }: { parameter: PAParameter; value: number; active: boolean;
  select: (key: string) => void; change: (key: string, value: number) => void; disabled: boolean }) {
  const colors = useStudioColors()
  const [raw, setRaw] = useState(String(value))
  const focused = useRef(false)
  useEffect(() => { if (!focused.current) setRaw(Number.isFinite(value) ? String(value) : '') }, [value])
  const valid = validParameter(p, value)
  const position = (v: number) => p.logarithmic ? Math.log10(v) : v
  return <Paper variant="outlined" data-testid={'parameter-' + p.key} data-active={active} onFocus={() => select(p.key)} onClick={() => select(p.key)}
    sx={{ p: 1.5, borderColor: active ? 'primary.main' : 'divider', bgcolor: active ? colors.selected : 'background.paper', transition: 'background-color .15s', height: '100%' }}>
    <Stack direction="row" sx={{ alignItems: 'start', gap: 1, justifyContent: 'space-between' }}>
      <Box sx={{ minWidth: 0 }}><Typography variant="body2" sx={{ fontWeight: 650 }}>{paText(p.label)}</Typography>
        <Typography variant="caption" color="text.secondary">{p.symbol}{p.unit ? ' · ' + p.unit : ''}</Typography></Box>
      <TextField id={'pa-param-' + p.key} type="number" size="small" value={raw} disabled={disabled} error={!valid}
        sx={{ width: 110, flexShrink: 0 }} slotProps={{ htmlInput: { 'aria-label': paText(p.label), min: p.minimum, max: p.maximum, step: p.integer ? 1 : 'any' } }}
        onFocus={() => { focused.current = true; select(p.key) }} onBlur={() => { focused.current = false; setRaw(Number.isFinite(value) ? String(value) : '') }}
        onChange={e => { setRaw(e.target.value); change(p.key, e.target.value.trim() === '' ? NaN : Number(e.target.value)) }} />
    </Stack>
    <Box sx={{ px: 1 }}><Slider disabled={disabled} aria-label={paText(p.label)} min={position(p.minimum)} max={position(p.maximum)}
      step={p.logarithmic ? .01 : p.step} value={position(valid ? value : p.default)} valueLabelDisplay="auto"
      valueLabelFormat={v => (p.logarithmic ? 10 ** v : v).toPrecision(4)} onFocus={() => select(p.key)}
      getAriaValueText={v => (p.logarithmic ? 10 ** v : v).toPrecision(4) + (p.unit ? ' ' + p.unit : '')}
      onChange={(_, rawValue) => { const v = Array.isArray(rawValue) ? rawValue[0]! : rawValue; const next = p.logarithmic ? Number((10 ** v).toPrecision(5)) : Number(v.toFixed(8)); setRaw(String(next)); change(p.key, next) }} /></Box>
    <Typography variant="caption" color={valid ? 'text.secondary' : 'error'} sx={{ display: 'block' }}>{valid ? paText(p.description) : t('paLibrary.range', { min: p.minimum, max: p.maximum })}</Typography>
  </Paper>
}

export function PALibraryPage() {
  const models = useVirtualPAs()
  const [query] = useSearchParams()
  if (models.isPending) return <LoadingState />
  if (models.isError) return <ErrorState error={models.error} onRetry={() => void models.refetch()} />
  return <Library key={query.get('input') ?? ''} models={models.data} />
}

function Library({ models }: { models: VirtualPA[] }) {
  const colors = useStudioColors()
  const inputs = usePAInputs()
  const workflow = useStudioWorkflow()
  const { selectInput, configurePA, simulated } = workflow
  const [query] = useSearchParams()
  const navigate = useNavigate()
  const initialModel = models.find(m => m.model_id === workflow.state.modelId) ?? models.find(m => m.model_id === 'rapp-am-pm') ?? models[0]!
  const [modelId, setModelId] = useState(initialModel.model_id)
  const [values, setValues] = useState<Record<string, number>>(() => ({ ...paDefaults(initialModel), ...(initialModel.model_id === workflow.state.modelId ? workflow.state.parameters : {}) }))
  const [inputId, setInputId] = useState(query.get('input') ?? (workflow.state.origin === 'generated' ? workflow.state.inputId ?? '' : ''))
  const [active, setActive] = useState(initialModel.parameters[0]!.key)
  const [error, setError] = useState<unknown>(null)
  const [removed, setRemoved] = useState<string | null>(null)
  const [removing, setRemoving] = useState(false)
  const [downloading, setDownloading] = useState(false)
  const simulate = useSimulatePA()
  const create = usePairedDataset()
  const allowed = useCustomDatasetImports()
  const model = models.find(m => m.model_id === modelId)!
  const input = inputs.data?.find(entry => entry.signal_id === inputId)
  const valid = model.parameters.every(p => validParameter(p, values[p.key]!))
  const signature = parameterKey(values)
  useEffect(() => {
    if (input && valid) { selectInput(input.signal_id, input.name); configurePA(modelId, values) }
  }, [input, modelId, valid, selectInput, configurePA, values])
  const saved = usePASimulation(workflow.state.simulationId)
  const candidate = simulate.data ?? saved.data
  const preview = valid && candidate?.config.input_signal_id === inputId && candidate.config.model_id === modelId
    && parameterKey(candidate.config.parameters ?? {}) === signature ? candidate : null
  const selectModel = (entry: VirtualPA) => { setModelId(entry.model_id); setValues(paDefaults(entry)); setActive(entry.parameters[0]!.key); setError(null); simulate.reset(); create.reset() }
  const selectVariable = (key: string) => { setActive(key) }
  const download = (url: string) => { setDownloading(true); setError(null); void downloadFile(url).catch(setError).finally(() => setDownloading(false)) }
  const run = () => {
    if (!input || !valid) return
    setError(null)
    simulate.mutate({ input_signal_id: input.signal_id, model_id: modelId, parameters: values }, { onSuccess: simulated })
  }
  return <Stack spacing={2.5}>
    <Stack direction="row" sx={{ justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 1 }}>
      <Box><Typography variant="overline" color="primary">{t('paLibrary.eyebrow')}</Typography><Typography variant="h1">{t('paLibrary.title')}</Typography></Box>
      <Chip label={t('paLibrary.synthetic')} color="info" variant="outlined" />
    </Stack>
    {removed && <Alert severity="success" action={<Button disabled={removing} onClick={() => {
      setRemoving(true); void api.post(`/signal-generator/signals/${removed}/restore`, {}).then(() => { setInputId(removed); setRemoved(null); return inputs.refetch() }).catch(setError).finally(() => setRemoving(false))
    }}>{t('common.undo')}</Button>}>{t('paInput.removed')}</Alert>}
    <Typography color="text.secondary" sx={{ maxWidth: 980 }}>{t('paLibrary.intro')}</Typography>
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: '270px minmax(0, 1fr)' }, gap: 2.5, alignItems: 'start' }}>
      <Paper variant="outlined" sx={{ p: 2 }}><Typography variant="h3" sx={{ mb: 2 }}>{t('paLibrary.chooseModel')}</Typography>
        <Stack spacing={2}>{CATEGORIES.map(category => <Box key={category}>
          <Typography variant="overline" color="text.secondary">{t(`paLibrary.category.${category}`)}</Typography>
          <Stack spacing={.75} sx={{ mt: .5 }}>{models.filter(m => m.category === category).map(entry => <ButtonBase key={entry.model_id} disabled={create.isPending || simulate.isPending}
            aria-pressed={entry.model_id === modelId} onClick={() => selectModel(entry)} sx={{ display: 'block', textAlign: 'left', p: 1.25, borderRadius: 1.5,
              border: '1px solid', borderColor: entry.model_id === modelId ? 'primary.main' : 'divider', bgcolor: entry.model_id === modelId ? colors.selected : 'transparent',
              '&:focus-visible': { outline: '2px solid ' + colors.primary, outlineOffset: 2 } }}>
            <Typography variant="body2" sx={{ fontWeight: 650 }}>{paText(entry.name)}</Typography><Typography variant="caption" color="text.secondary">{entry.technology}</Typography>
          </ButtonBase>)}</Stack>
        </Box>)}</Stack>
      </Paper>
      <Stack spacing={2} sx={{ minWidth: 0 }}>
        <Paper sx={{ p: 2.5 }}><Stack spacing={2}>
          <Typography variant="h2">{t('paLibrary.feed')}</Typography>
          {inputs.isPending ? <LoadingState /> : inputs.isError ? <ErrorState error={inputs.error} onRetry={() => void inputs.refetch()} /> :
            <TextField select fullWidth label={t('paLibrary.input')} value={input ? inputId : ''} disabled={create.isPending || simulate.isPending}
              onChange={e => { setInputId(e.target.value); simulate.reset(); create.reset() }}>
              <MenuItem value="" disabled>{t('paLibrary.chooseInput')}</MenuItem>
              {inputs.data.map(entry => <MenuItem key={entry.signal_id} value={entry.signal_id}>{entry.name} · {formatNumber(entry.n_samples)} I/Q · {(entry.sample_rate_hz / 1e6).toFixed(2)} MHz · {entry.signal_id.slice(3, 11)}</MenuItem>)}
            </TextField>}
          {input ? <Typography variant="body2" color="text.secondary">{t('paLibrary.inputSummary', { count: formatNumber(input.n_samples), duration: (1000 * input.n_samples / input.sample_rate_hz).toPrecision(5) })}</Typography>
            : <Alert severity="info">{t('paLibrary.noInput')}</Alert>}
          <Stack direction="row" spacing={1} useFlexGap sx={{ flexWrap: 'wrap' }}>
            <Button variant="contained" startIcon={<PlayArrowIcon />} endIcon={<ArrowForwardIcon />} onClick={run} disabled={!input || !valid || simulate.isPending || create.isPending}>{t(simulate.isPending ? 'paLibrary.simulating' : 'paLibrary.simulate')}</Button>
            <Button color="inherit" startIcon={<DeleteOutlineIcon />} disabled={!input || removing || simulate.isPending || create.isPending} onClick={() => {
              if (!input) return
              const id = input.signal_id; setRemoving(true)
              void api.post(`/signal-generator/signals/${id}/archive`, {}).then(() => {
                setRemoved(id); setInputId(''); simulate.reset(); create.reset(); workflow.reset(); return inputs.refetch()
              }).catch(setError).finally(() => setRemoving(false))
            }}>{t('paInput.remove')}</Button>
            <Button component={RouterLink} to="/signal-generator">{t('paLibrary.generateInput')}</Button>
          </Stack>
          {simulate.isPending && <LinearProgress />}
          {candidate && !preview && !simulate.isPending && <Alert severity="info">{t('paLibrary.stale')}</Alert>}
          {(simulate.isError || saved.isError || !!error) && <ErrorState error={error || simulate.error || saved.error} />}
        </Stack></Paper>
        <Paper sx={{ p: { xs: 2, md: 2.5 }, minWidth: 0 }}>
          <Typography variant="h2">{paText(model.name)}</Typography><Typography color="text.secondary" sx={{ mt: 1 }}>{paText(model.description)}</Typography>
          <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>{paText(model.limitations)}</Typography>
          <Box sx={{ my: 2, px: 2, py: 1, borderRadius: 2, bgcolor: colors.surfaceMuted, border: '1px solid', borderColor: 'divider', position: { md: 'sticky' }, top: 162, zIndex: 2 }}>
            <Typography variant="overline" color="primary">{t('paLibrary.equations')}</Typography>
            <PAEquations key={modelId} model={model} active={active} select={selectVariable} />
            <Typography variant="caption" color="text.secondary">{t('paLibrary.equationHelp')}</Typography>
          </Box>
          <Stack direction="row" sx={{ justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
            <Typography variant="h3">{t('paLibrary.parameters')}</Typography><Button size="small" startIcon={<RestartAltIcon />} disabled={create.isPending || simulate.isPending}
              onClick={() => { setValues(paDefaults(model)); setActive(model.parameters[0]!.key); simulate.reset(); create.reset() }}>{t('paLibrary.defaults')}</Button>
          </Stack>
          {GROUPS.map(group => {
            const params = model.parameters.filter(p => (p.group ?? 'gain') === group)
            return params.length ? <Box key={group} sx={{ mb: 2 }}><Typography variant="overline" color="text.secondary">{t(`paLibrary.group.${group}`)}</Typography>
              <Grid container spacing={1.25} sx={{ mt: .5 }}>{params.map(p => <Grid key={modelId + ':' + p.key} size={{ xs: 12, md: 6 }}>
                <ParameterControl parameter={p} value={values[p.key]!} active={active === p.key} select={selectVariable} disabled={create.isPending || simulate.isPending}
                  change={(key, value) => { workflow.invalidateOutput(); setValues(old => ({ ...old, [key]: value })); simulate.reset(); create.reset() }} />
              </Grid>)}</Grid></Box> : null
          })}
        </Paper>

      </Stack>
    </Box>
    {preview && <>
      <Paper sx={{ p: 2.5 }}><Stack spacing={2}>
        <Stack direction="row" spacing={1} useFlexGap sx={{ flexWrap: 'wrap' }}>
          <Button startIcon={<DownloadIcon />} variant="outlined" disabled={downloading} onClick={() => download(preview.output_csv_url)}>{t('paLibrary.outputCsv')}</Button>
          <Button startIcon={<DownloadIcon />} variant="outlined" disabled={downloading} onClick={() => download(preview.paired_csv_url)}>{t('paLibrary.pairedCsv')}</Button>
          <Button startIcon={<DownloadIcon />} variant="outlined" disabled={downloading} onClick={() => download(preview.metadata_url)}>{t('paLibrary.metadata')}</Button>
        </Stack>
        <PairedDatasetForm key={preview.simulation_id} result={preview} allowed={allowed} pending={create.isPending} error={create.error} onCreate={request => {
          create.mutate({ id: preview.simulation_id, request }, { onSuccess: result => {
            workflow.paired(result.dataset.dataset_id, preview.simulation_id)
            navigate('/experiments/new?task=train_pa&dataset=' + encodeURIComponent(result.dataset.dataset_id))
          } })
        }} />
      </Stack></Paper>
      <PAOutputPlots result={preview} />
    </>}
  </Stack>
}

function PairedDatasetForm({ result, allowed, pending, error, onCreate }: { result: PASimulation; allowed: boolean; pending: boolean; error: unknown;
  onCreate: (request: { dataset_id: string; display_name: string; guard_samples: number; train_ratio: number; val_ratio: number }) => void }) {
  const [id, setId] = useState('virtual-pa-' + result.simulation_id.slice(4, 16))
  const [name, setName] = useState(paText(result.model.name) + ' · synthetic pair')
  const [guard, setGuard] = useState('256'), [train, setTrain] = useState('60'), [val, setVal] = useState('20')
  const g = Number(guard), tr = Number(train) / 100, vr = Number(val) / 100
  const n = result.analysis.n_samples
  const usable = n - 2 * g
  const nTrain = Math.floor(usable * tr), nVal = Math.floor(usable * vr), nTest = usable - nTrain - nVal
  const valid = /^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$/.test(id) && name.trim().length > 0 && name.length <= 180
    && guard.trim() !== '' && Number.isInteger(g) && g >= 0 && g <= 10000 && tr > 0 && vr > 0 && tr + vr < 1
    && Math.min(nTrain, nVal, nTest) >= 256 && n >= 8192
  return <Stack spacing={2} component="form" onSubmit={e => { e.preventDefault(); if (valid && allowed && !pending) onCreate({ dataset_id: id, display_name: name.trim(), guard_samples: g, train_ratio: tr, val_ratio: vr }) }}>
    <Typography variant="h2">{t('paLibrary.pairTitle')}</Typography><Typography color="text.secondary">{t('paLibrary.pairHelp')}</Typography>
    <Button type="submit" variant="contained" endIcon={<ArrowForwardIcon />} disabled={!allowed || !valid || pending} sx={{ alignSelf: 'flex-start' }}>{t(pending ? 'paLibrary.creating' : 'paLibrary.create')}</Button>
    <Grid container spacing={2}><Grid size={{ xs: 12, md: 6 }}><TextField label={t('paLibrary.datasetId')} fullWidth value={id} disabled={pending} onChange={e => setId(e.target.value)} /></Grid>
      <Grid size={{ xs: 12, md: 6 }}><TextField label={t('paLibrary.datasetName')} fullWidth value={name} disabled={pending} onChange={e => setName(e.target.value)} /></Grid></Grid>
    <Accordion disableGutters variant="outlined"><AccordionSummary expandIcon={<ExpandMoreIcon />}>{t('paLibrary.splits')}</AccordionSummary>
      <AccordionDetails><Grid container spacing={2}>{([{ label: 'paLibrary.guard', value: guard, set: setGuard }, { label: 'paLibrary.trainRatio', value: train, set: setTrain }, { label: 'paLibrary.valRatio', value: val, set: setVal }] as const).map(field =>
        <Grid size={{ xs: 12, md: 4 }} key={field.label}><TextField type="number" label={t(field.label)} fullWidth disabled={pending} value={field.value} onChange={e => field.set(e.target.value)} /></Grid>)}</Grid>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>{t('paLibrary.splitHelp')}</Typography>
      </AccordionDetails></Accordion>
    <Typography variant="body2">{valid ? t('paLibrary.splitCounts', { train: formatNumber(nTrain), val: formatNumber(nVal), test: formatNumber(nTest) }) : t('paLibrary.splitInvalid')}</Typography>
    {!allowed && <Alert severity="info">{t('paLibrary.importDisabled')}</Alert>}
    {!!error && <ErrorState error={error} />}

  </Stack>
}

function PAOutputPlots({ result }: { result: PASimulation }) {
  const colors = useStudioColors()
  const a = result.analysis
  const axis = (text: string) => ({ title: { text } })
  const pair = [colors.primary, colors.status.success]
  const legend = { legend: { orientation: 'h' as const, x: 0, y: -.28 }, margin: { l: 60, r: 16, t: 42, b: 90 } }
  const number = (v: number | null) => v == null ? '—' : v.toFixed(2)
  return <Stack spacing={2} data-testid="pa-output-preview">
    <Stack direction="row" sx={{ gap: 1, alignItems: 'center', flexWrap: 'wrap' }}><Typography variant="h2">{t('paLibrary.outputTitle')}</Typography><Chip size="small" label={t('paLibrary.synthetic')} color="info" variant="outlined" /></Stack>
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: 'repeat(2, minmax(0,1fr))', md: 'repeat(4, minmax(0,1fr))' }, gap: 1.5 }}>
      {[[t('paLibrary.samples'), formatNumber(a.n_samples)], [t('paLibrary.rmsGain'), number(a.rms_gain_db) + ' dB'],
        [t('paLibrary.inputPapr'), number(a.input_papr_db) + ' dB'], [t('paLibrary.outputPapr'), number(a.output_papr_db) + ' dB']].map(([label, value]) =>
        <Paper key={label} sx={{ p: 2 }}><Typography variant="caption" color="text.secondary">{label}</Typography><Typography variant="h2" sx={{ mt: .5 }}>{value}</Typography></Paper>)}
    </Box>
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: 'repeat(2, minmax(0,1fr))' }, gap: 2 }}>
      <Paper sx={{ p: 1.5, minWidth: 0 }}><PlotlyChart title="AM / AM" viewKey={result.simulation_id} height={270} traces={[{ x: a.am_input, y: a.am_output, mode: 'markers', name: 'y(x)', marker: { size: 3, opacity: .55, color: pair[0] } }]}
        layout={{ xaxis: axis('|x[n]|'), yaxis: axis('|y[n]|'), showlegend: false }} /></Paper>
      <Paper sx={{ p: 1.5, minWidth: 0 }}><PlotlyChart title="AM / PM" viewKey={result.simulation_id} height={270} traces={[{ x: a.am_input, y: a.am_pm_deg, mode: 'markers', name: '∠y − ∠x', marker: { size: 3, opacity: .55, color: pair[0] } }]}
        layout={{ xaxis: axis('|x[n]|'), yaxis: axis('∠y − ∠x (°)'), showlegend: false }} /></Paper>
      <Box sx={{ gridColumn: '1 / -1' }}><SpectrumPanels frequencyHz={a.frequency_mhz.map(f => f * 1e6)} viewKey={result.simulation_id} height={300} traces={[
        { name: 'PA input x', role: 'input', signal_node: 'pa_input', psdDb: a.input_psd, color: pair[0] },
        { name: 'Virtual PA · synthetic', role: 'primary', signal_node: 'pa_output', psdDb: a.output_psd, color: pair[1] },
      ]} /></Box>
      <Paper sx={{ p: 1.5, minWidth: 0 }}><PlotlyChart title={t('paLibrary.envelopes')} viewKey={result.simulation_id} height={280} traces={[
        { x: a.time_us, y: a.input_envelope, name: '|x[n]|', line: { color: pair[0] } }, { x: a.time_us, y: a.output_envelope, name: '|y[n]|', line: { color: pair[1] } }]}
        layout={{ ...legend, xaxis: axis('µs'), yaxis: axis(t('generator.amplitude')) }} /></Paper>
      {Object.entries(a.states).map(([name, values]) => <Paper key={name} sx={{ p: 1.5, minWidth: 0 }}><PlotlyChart title={name.replaceAll('_', ' ')} viewKey={result.simulation_id} height={230}
        traces={[{ x: a.time_us, y: values, name, line: { color: pair[1] } }]} layout={{ xaxis: axis('µs'), showlegend: false }} /></Paper>)}
    </Box>
    <Typography variant="caption" color="text.secondary">{t('paLibrary.plotHelp', { count: a.time_us.length, duration: a.duration_ms.toPrecision(5), input: a.input_rms.toPrecision(5), output: a.output_rms.toPrecision(5) })}</Typography>
    {a.notes.map(note => <Typography key={note} variant="caption" color="text.secondary">{note}</Typography>)}
  </Stack>
}
