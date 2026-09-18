import DeleteOutlineIcon from '@mui/icons-material/DeleteOutlined'
import { useGeneratedSignals, type GeneratorConfig } from '@/api/signalGenerator'
import { pairedDatasetName, validDatasetName } from '@/utils/datasetNames'
import { useAnalyzerDatasets } from '@/api/signalAnalyzer'
import { SignalDatasetSelect } from '@/components/SignalDatasetSelect'
import { api } from '@/api/client'
import { MathFormula } from '@/components/MathFormula'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import RestartAltIcon from '@mui/icons-material/RestartAlt'
import ArrowForwardIcon from '@mui/icons-material/ArrowForward'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import ButtonBase from '@mui/material/ButtonBase'
import Chip from '@mui/material/Chip'
import Grid from '@mui/material/Grid'
import LinearProgress from '@mui/material/LinearProgress'
import Paper from '@mui/material/Paper'
import Slider from '@mui/material/Slider'
import Stack from '@mui/material/Stack'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { useEffect, useMemo, useRef, useState } from 'react'
import { Link as RouterLink, useNavigate, useSearchParams } from 'react-router'
import { useCustomDatasetImports } from '@/api/hooks'
import { paDefaults, paText, useSimulateDataset, useVirtualPAs,
  type PAParameter, type VirtualPA } from '@/api/virtualPA'
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
  return <Library key={(query.get('dataset') ?? '') + ':' + (query.get('input') ?? '')} models={models.data} />
}

function Library({ models }: { models: VirtualPA[] }) {
  const colors = useStudioColors()
  const collections = useAnalyzerDatasets()
  const workflow = useStudioWorkflow()
  const { selectInputs, configurePA } = workflow
  const [query] = useSearchParams()
  const navigate = useNavigate()
  const initialModel = models.find(m => m.model_id === workflow.state.modelId) ?? models.find(m => m.model_id === 'rapp-am-pm') ?? models[0]!
  const [modelId, setModelId] = useState(initialModel.model_id)
  const [values, setValues] = useState<Record<string, number>>(() => ({ ...paDefaults(initialModel), ...(initialModel.model_id === workflow.state.modelId ? workflow.state.parameters : {}) }))
  const [datasetId, setDatasetId] = useState(query.get('dataset') ?? (workflow.state.origin === 'generated' ? workflow.state.inputDatasetId ?? '' : ''))
  const requestedInput = query.get('input') ?? (workflow.state.origin === 'generated' ? workflow.state.inputId : null)
  const [active, setActive] = useState(initialModel.parameters[0]!.key)
  const [error, setError] = useState<unknown>(null)
  const [removed, setRemoved] = useState<{ id: string; endpoint: string } | null>(null)
  const [removing, setRemoving] = useState(false)
  const create = useSimulateDataset()
  const allowed = useCustomDatasetImports()
  const model = models.find(m => m.model_id === modelId)!
  const datasets = (collections.data ?? []).filter(d => d.kind === 'pa_input')
  const input = datasetId ? (removed?.id === datasetId ? undefined : datasets.find(d => d.dataset_id === datasetId))
    : datasets.find(d => d.signals.some(s => s.source.source_id === requestedInput))
  const batchIds = useMemo(() => input?.signals.map(s => s.source.source_id) ?? [], [input])
  const savedConfigs = workflow.state.inputDatasetId === input?.dataset_id && workflow.state.inputConfigs?.length === batchIds.length ? workflow.state.inputConfigs : undefined
  const inputSignals = useGeneratedSignals(savedConfigs ? [] : batchIds)
  const configs = savedConfigs ?? inputSignals.flatMap(q => q.data ? [q.data.config as GeneratorConfig] : [])
  const [customName, setCustomName] = useState<string | null>(null)
  const automaticName = input ? pairedDatasetName(input.name, modelId) : ''
  const name = customName ?? automaticName
  const validName = validDatasetName(name, 'inout') && configs.length === batchIds.length && batchIds.length > 0
  const valid = model.parameters.every(p => validParameter(p, values[p.key]!))
  useEffect(() => {
    if (input && valid && configs.length === batchIds.length) {
      if (workflow.state.inputDatasetId !== input.dataset_id || workflow.state.inputIds?.join() !== batchIds.join()) {
        selectInputs(batchIds, configs, { id: input.dataset_id, name: input.name })
      }
      configurePA(modelId, values)
    }
  }, [input, modelId, valid, selectInputs, configurePA, values, configs, batchIds, workflow.state.inputDatasetId, workflow.state.inputIds])
  const selectModel = (entry: VirtualPA) => { setModelId(entry.model_id); setValues(paDefaults(entry)); setActive(entry.parameters[0]!.key); setError(null); create.reset() }
  const selectVariable = (key: string) => { setActive(key) }
  const run = () => {
    if (!input || !valid || !validName || !allowed) return
    setError(null)
    create.mutate({ input_signal_ids: batchIds, model_id: modelId, parameters: values, dataset_name: name }, { onSuccess: result => {
      workflow.completeDataset(result.dataset.dataset_id, String(result.dataset.simulation?.simulation_id ?? ''))
      navigate('/datasets/' + encodeURIComponent(result.dataset.dataset_id))
    } })
  }
  return <Stack spacing={2.5}>
    <Stack direction="row" sx={{ justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 1 }}>
      <Box><Typography variant="overline" color="primary">{t('paLibrary.eyebrow')}</Typography><Typography variant="h1">{t('paLibrary.title')}</Typography></Box>
      <Chip label={t('paLibrary.synthetic')} color="info" variant="outlined" />
    </Stack>
    {removed && <Alert severity="success" action={<Button disabled={removing} onClick={() => {
      setRemoving(true); void api.post(removed.endpoint + '/restore', {}).then(async () => { await collections.refetch(); setDatasetId(removed.id); setRemoved(null) }).catch(setError).finally(() => setRemoving(false))
    }}>{t('common.undo')}</Button>}>{t('paInput.removed')}</Alert>}
    <Typography color="text.secondary" sx={{ maxWidth: 980 }}>{t('paLibrary.intro')}</Typography>
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: '270px minmax(0, 1fr)' }, gap: 2.5, alignItems: 'start' }}>
      <Paper variant="outlined" sx={{ p: 2 }}><Typography variant="h3" sx={{ mb: 2 }}>{t('paLibrary.chooseModel')}</Typography>
        <Stack spacing={2}>{CATEGORIES.map(category => <Box key={category}>
          <Typography variant="overline" color="text.secondary">{t(`paLibrary.category.${category}`)}</Typography>
          <Stack spacing={.75} sx={{ mt: .5 }}>{models.filter(m => m.category === category).map(entry => <ButtonBase key={entry.model_id} disabled={create.isPending}
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
          {collections.isPending ? <LoadingState /> : collections.isError ? <ErrorState error={collections.error} onRetry={() => void collections.refetch()} /> :
            <SignalDatasetSelect datasets={datasets} value={input} label={t('paLibrary.input')} disabled={create.isPending || removing}
              onChange={entry => { setDatasetId(entry.dataset_id); setCustomName(null); create.reset() }} />}
          {input ? <Typography variant="body2" color="text.secondary">{t('paLibrary.datasetSummary', { signals: formatNumber(batchIds.length), count: formatNumber(input.signals.reduce((n, s) => n + s.sample_count, 0)) })}</Typography>
            : <Alert severity="info">{t('paLibrary.noInput')}</Alert>}
          {batchIds.length > 1 && <Alert severity="info">{t('paLibrary.batch', { count: batchIds.length })}</Alert>}
          {!allowed && <Alert severity="info">{t('paLibrary.importDisabled')}</Alert>}
          <Typography variant="caption" color="text.secondary">{t('paLibrary.autoDataset')}</Typography>
          <TextField fullWidth label={t('paLibrary.datasetName')} value={name} onChange={e => setCustomName(e.target.value)} disabled={create.isPending || !input} error={!!input && !validDatasetName(name, 'inout')}
            helperText={t('paLibrary.datasetNameHelp')} slotProps={{ htmlInput: { maxLength: 96 } }} />
          {customName !== null && <Button sx={{ alignSelf: 'start' }} onClick={() => setCustomName(null)} disabled={create.isPending}>{t('generator.autoName')}</Button>}
          {input && <Typography variant="caption" color="text.secondary" sx={{ overflowWrap: 'anywhere' }}>{t('paLibrary.outputDatasetName', { name: name.replace(/^syn_pa_inout_/, 'syn_pa_out_') })}</Typography>}
          {inputSignals.some(q => q.isError) && <ErrorState error={inputSignals.find(q => q.isError)?.error} />}
          <Stack direction="row" spacing={1} useFlexGap sx={{ flexWrap: 'wrap' }}>
            <Button variant="contained" startIcon={<PlayArrowIcon />} endIcon={<ArrowForwardIcon />} onClick={run} disabled={!input || !valid || !validName || !allowed || create.isPending}>{t(create.isPending ? 'paLibrary.simulating' : 'paLibrary.simulate')}</Button>
            <Button color="inherit" startIcon={<DeleteOutlineIcon />} disabled={!input || removing || create.isPending} onClick={() => {
              if (!input) return
              const id = input.dataset_id
              const endpoint = id.startsWith('sds-') ? `/signal-generator/datasets/${id}` : `/signal-generator/signals/${batchIds[0]}`
              setRemoving(true)
              void api.post(endpoint + '/archive', {}).then(() => {
                setRemoved({ id, endpoint }); setDatasetId(id); create.reset(); workflow.reset(); return collections.refetch()
              }).catch(setError).finally(() => setRemoving(false))
            }}>{t('paInput.remove')}</Button>
            <Button component={RouterLink} to="/signal-generator">{t('paLibrary.generateInput')}</Button>
          </Stack>
          {create.isPending && <LinearProgress />}
          {(create.isError || !!error) && <ErrorState error={error || create.error} />}
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
            <Typography variant="h3">{t('paLibrary.parameters')}</Typography><Button size="small" startIcon={<RestartAltIcon />} disabled={create.isPending}
              onClick={() => { setValues(paDefaults(model)); setActive(model.parameters[0]!.key); create.reset() }}>{t('paLibrary.defaults')}</Button>
          </Stack>
          {GROUPS.map(group => {
            const params = model.parameters.filter(p => (p.group ?? 'gain') === group)
            return params.length ? <Box key={group} sx={{ mb: 2 }}><Typography variant="overline" color="text.secondary">{t(`paLibrary.group.${group}`)}</Typography>
              <Grid container spacing={1.25} sx={{ mt: .5 }}>{params.map(p => <Grid key={modelId + ':' + p.key} size={{ xs: 12, md: 6 }}>
                <ParameterControl parameter={p} value={values[p.key]!} active={active === p.key} select={selectVariable} disabled={create.isPending}
                  change={(key, value) => { workflow.invalidateOutput(); setValues(old => ({ ...old, [key]: value })); create.reset() }} />
              </Grid>)}</Grid></Box> : null
          })}
        </Paper>

      </Stack>
    </Box>

  </Stack>
}
