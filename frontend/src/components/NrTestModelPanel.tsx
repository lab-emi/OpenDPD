import AddIcon from '@mui/icons-material/Add'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import Accordion from '@mui/material/Accordion'
import AccordionDetails from '@mui/material/AccordionDetails'
import AccordionSummary from '@mui/material/AccordionSummary'
import Alert from '@mui/material/Alert'
import AlertTitle from '@mui/material/AlertTitle'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import ButtonBase from '@mui/material/ButtonBase'
import Chip from '@mui/material/Chip'
import LinearProgress from '@mui/material/LinearProgress'
import MenuItem from '@mui/material/MenuItem'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import TextField from '@mui/material/TextField'
import ToggleButton from '@mui/material/ToggleButton'
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup'
import Typography from '@mui/material/Typography'
import type { Ref } from 'react'
import type { NrTmCatalog, NrTmModelId, NrTmRecipe, NrTmResolution } from '@/api/nrTestModels'
import { formatNumber, t } from '@/i18n'
import { useStudioColors } from '@/theme'
import { convertModel, describeNrTmError, MAX_SUBFRAMES, modelInfo, NB_IOT_BOOST, NB_IOT_DEFAULT, nbIotBoostBounds, nbIotHosts, nextModel, normalizeRecipe, shortModel, withPnScope } from '@/utils/nrTestModel'
import { NrTmCarriers } from './NrTmCarriers'
import { NumberInput } from './NrTmFields'
import { NrTmStandard } from './NrTmStandard'
import type { NrTmStatus } from './useNrTmEditor'

export interface NrTestModelPanelProps {
  catalog: NrTmCatalog
  /** The active entry's recipe (draft while editing), or null when the active entry is not a test model. */
  recipe: NrTmRecipe | null
  status: NrTmStatus
  error?: unknown
  /** The last server resolution of the active entry; `fresh` says it belongs to the recipe shown. */
  resolution?: NrTmResolution
  fresh: boolean
  loading: boolean
  /** Label of the other entry that already holds the recipe being edited (status `duplicate`). */
  duplicateOf?: string
  disabled: boolean
  /** Models already in the selection, so that "Add another" starts from one that is not. */
  selectedModels: string[]
  canAdd: boolean
  /** The panel's heading takes focus when the page brings an entry here ("Show it"), so that keyboard and screen-reader users land in the panel. */
  headingRef?: Ref<HTMLHeadingElement>
  onChange: (recipe: NrTmRecipe) => void
  onAdd: (recipe: NrTmRecipe) => void
}

export function NrTestModelPanel({ catalog, recipe, status, error, resolution, fresh, loading, duplicateOf, disabled, selectedModels, canAdd, headingRef, onChange, onAdd }: NrTestModelPanelProps) {
  const colors = useStudioColors()
  const info = recipe ? modelInfo(catalog, recipe.model) : undefined
  const summary = resolution?.summary
  const current = fresh ? summary : undefined
  const standardSubframes = recipe?.duplex === 'tdd' ? catalog.limits.standard_subframes_tdd : catalog.limits.standard_subframes_fdd
  // The server accepts 1 to 20 subframes whatever the duplex mode; the standard's own length is only the default.
  const maxSubframes = Math.max(MAX_SUBFRAMES, catalog.limits.standard_subframes_tdd)
  // Without an active test model a card adds an entry, and the selection is full at 16: say so instead of offering a button that would break the limit.
  const full = !recipe && !canAdd
  const boost = recipe?.nb_iot ? nbIotBoostBounds(catalog, recipe) : NB_IOT_BOOST
  const choose = (model: NrTmModelId) => recipe ? onChange(convertModel(catalog, recipe, model)) : canAdd && onAdd(convertModel(catalog, normalizeRecipe(catalog.default_recipe), model))
  // The PN23 scope only matters for TDD with the standard convention (the server warns otherwise), so any other combination keeps the default.
  const set = (change: Partial<NrTmRecipe>) => { if (recipe) onChange(withPnScope({ ...recipe, ...change })) }

  return <Stack spacing={1.5} data-testid="nrtm-panel" aria-busy={loading}>
    <Stack spacing={.5}>
      <Stack direction="row" useFlexGap sx={{ alignItems: 'center', flexWrap: 'wrap', gap: 1 }}>
        <Typography variant="h3" component="h3" ref={headingRef} tabIndex={-1} sx={{ '&:focus-visible': { outline: '3px solid', outlineColor: 'primary.main', outlineOffset: 2, borderRadius: .5 } }}>{t('nrtm.title')}</Typography>
        <Chip size="small" variant="outlined" label={catalog.standard} sx={{ height: 'auto', maxWidth: '100%', py: .25, '& .MuiChip-label': { whiteSpace: 'normal', overflowWrap: 'anywhere' } }} />
      </Stack>
      <Typography variant="caption" color="text.secondary">{t(recipe ? 'nrtm.help' : 'nrtm.helpEmpty')}</Typography>
    </Stack>

    <Box role="group" aria-label={t('nrtm.model')} sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill,minmax(108px,1fr))', gap: 1 }}>
      {catalog.models.map(m => {
        const selected = recipe?.model === m.model
        return <ButtonBase key={m.model} disabled={disabled || full} aria-pressed={selected} aria-label={`${shortModel(m.model)}, ${m.modulations.join(', ')}`} title={m.title} data-testid={'nrtm-model-' + m.model} onClick={() => choose(m.model)}
          sx={{ display: 'block', textAlign: 'left', p: 1.25, border: 1, borderRadius: 1.5, borderColor: selected ? 'primary.main' : 'divider', bgcolor: selected ? colors.selected : 'background.paper',
            '&:hover': { borderColor: 'primary.main' }, '&.Mui-focusVisible': { outline: '3px solid', outlineColor: 'primary.main', outlineOffset: 2 } }}>
          <Typography component="span" sx={{ display: 'block', fontWeight: 750, fontSize: 16 }}>{shortModel(m.model) + ' '}</Typography>
          <Typography component="span" variant="caption" color="text.secondary" sx={{ display: 'block' }}>{m.modulations.join(' · ')}</Typography>
        </ButtonBase>
      })}
    </Box>
    {full && <Typography variant="caption" color="warning.main" role="status" data-testid="nrtm-selection-full">{t('generator.selectionLimit')}</Typography>}

    {recipe && info && <Paper sx={{ p: 1.5 }} data-testid="nrtm-model-details"><Stack spacing={.75}>
      <Typography variant="subtitle2" component="h4">{info.title}</Typography>
      <Typography variant="caption" color="text.secondary">{t('nrtm.clause', { clause: info.clause })} · {info.users.map(u => `nRNTI ${u.rnti}: ${u.modulation}`).join(' · ')}</Typography>
      <Typography variant="body2">{info.description}</Typography>
      <Accordion disableGutters variant="outlined" slotProps={{ heading: { component: 'h4' } }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}><Typography variant="body2" sx={{ fontWeight: 600 }}>{t('nrtm.usedFor', { count: info.used_for.length })}</Typography></AccordionSummary>
        <AccordionDetails><Box component="ul" sx={{ m: 0, pl: 2.5 }}>{info.used_for.map(use => <Typography component="li" variant="body2" key={use}>{use}</Typography>)}</Box></AccordionDetails>
      </Accordion>
    </Stack></Paper>}

    {recipe && <>
      <Stack direction="row" useFlexGap sx={{ gap: 2, flexWrap: 'wrap', alignItems: 'flex-start' }}>
        <Stack spacing={.5}>
          <Typography variant="caption" color="text.secondary" id="nrtm-duplex-label">{t('nrtm.duplex')}</Typography>
          <ToggleButtonGroup exclusive size="small" value={recipe.duplex} aria-labelledby="nrtm-duplex-label" onChange={(_, value: 'fdd' | 'tdd' | null) => { if (value) set({ duplex: value }) }}>
            <ToggleButton value="fdd" disabled={disabled}>FDD</ToggleButton>
            <ToggleButton value="tdd" disabled={disabled || !!info?.fdd_only}>TDD</ToggleButton>
          </ToggleButtonGroup>
          {info?.fdd_only && <Typography variant="caption" color="text.secondary">{t('nrtm.fddOnly', { model: shortModel(recipe.model) })}</Typography>}
        </Stack>
        {info?.supports_layers && <Stack spacing={.5}>
          <Typography variant="caption" color="text.secondary" id="nrtm-layers-label">{t('nrtm.layers')}</Typography>
          <ToggleButtonGroup exclusive size="small" value={recipe.layers} aria-labelledby="nrtm-layers-label"
            onChange={(_, value: 1 | 2 | null) => { if (value) set({ layers: value, ...(value === 1 ? { output_port: 0 as const } : {}) }) }}>
            <ToggleButton value={1} disabled={disabled}>1</ToggleButton>
            <ToggleButton value={2} disabled={disabled}>2</ToggleButton>
          </ToggleButtonGroup>
        </Stack>}
        {info?.supports_layers && recipe.layers === 2 && <TextField select label={t('nrtm.outputPort')} value={recipe.output_port} disabled={disabled} sx={{ minWidth: 200 }}
          onChange={e => set({ output_port: Number(e.target.value) as 0 | 1 })}>
          <MenuItem value={0}>{t('nrtm.port', { port: 1000 })}</MenuItem><MenuItem value={1}>{t('nrtm.port', { port: 1001 })}</MenuItem>
        </TextField>}
      </Stack>
      {info?.supports_layers && <Typography variant="caption" color="text.secondary">{t('nrtm.layersHelp')}</Typography>}

      <NrTmCarriers catalog={catalog} info={info} recipe={recipe} summary={current} disabled={disabled} onChange={onChange} />

      <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(230px,1fr))', gap: 1.5, alignItems: 'start' }}>
        <TextField select label={t('nrtm.convention')} value={recipe.data_convention} disabled={disabled} helperText={t(`nrtm.convention.${recipe.data_convention}.help`)}
          onChange={e => set({ data_convention: e.target.value as NrTmRecipe['data_convention'] })}>
          {(['standard', 'matlab'] as const).map(value => <MenuItem key={value} value={value}>{t(`nrtm.convention.${value}`)}</MenuItem>)}
        </TextField>
        {recipe.duplex === 'tdd' && recipe.data_convention === 'standard' && <TextField select label={t('nrtm.pnScope')} value={recipe.pn_scope} disabled={disabled} helperText={t('nrtm.pnScopeHelp')}
          onChange={e => set({ pn_scope: e.target.value as NrTmRecipe['pn_scope'] })}>
          {(['dl_symbols', 'all_symbols'] as const).map(value => <MenuItem key={value} value={value}>{t(`nrtm.pnScope.${value}`)}</MenuItem>)}
        </TextField>}
        <TextField select label={t('nrtm.oversampling')} value={recipe.oversampling === null ? 'auto' : String(recipe.oversampling)} disabled={disabled} helperText={t('nrtm.oversamplingHelp')}
          onChange={e => set({ oversampling: e.target.value === 'auto' ? null : Number(e.target.value) as 1 | 2 | 4 | 8 })}>
          <MenuItem value="auto">{t('nrtm.auto')}</MenuItem>{[1, 2, 4, 8].map(n => <MenuItem key={n} value={String(n)}>{n}×</MenuItem>)}
        </TextField>
        <Stack spacing={.75}>
          <Typography variant="caption" color="text.secondary" id="nrtm-length-label">{t('nrtm.length')}</Typography>
          <ToggleButtonGroup exclusive size="small" fullWidth value={recipe.subframes === null ? 'standard' : 'custom'} aria-labelledby="nrtm-length-label"
            onChange={(_, value: 'standard' | 'custom' | null) => { if (value) set({ subframes: value === 'standard' ? null : summary?.subframes ?? standardSubframes }) }}>
            <ToggleButton value="standard" disabled={disabled}>{t('nrtm.lengthStandard', { count: standardSubframes })}</ToggleButton>
            <ToggleButton value="custom" disabled={disabled}>{t('nrtm.lengthCustom')}</ToggleButton>
          </ToggleButtonGroup>
          {recipe.subframes !== null && <NumberInput label={t('nrtm.subframes')} value={recipe.subframes} min={1} max={maxSubframes} disabled={disabled}
            onCommit={value => set({ subframes: value === null ? null : Math.trunc(value) })} />}
          <Typography variant="caption" color="text.secondary">{t('nrtm.lengthHelp', { max: maxSubframes, limit: formatNumber(catalog.limits.max_samples) })}</Typography>
        </Stack>
      </Box>

      {info?.nb_iot && recipe.nb_iot && <Paper sx={{ p: 1.5 }} component="section" aria-label={t('nrtm.nbiot')}><Stack spacing={1}>
        <Typography variant="subtitle2" component="h4">{t('nrtm.nbiot')}</Typography>
        <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit,minmax(200px,1fr))', gap: 1.5 }}>
          <TextField select label={t('nrtm.nbiotPosition')} value={recipe.nb_iot.position} disabled={disabled}
            onChange={e => set({ nb_iot: { ...(recipe.nb_iot ?? NB_IOT_DEFAULT), position: e.target.value as 'lower' | 'upper' | 'both' } })}>
            {(['lower', 'upper', 'both'] as const).map(value => <MenuItem key={value} value={value}>{t(`nrtm.nbiotPosition.${value}`)}</MenuItem>)}
          </TextField>
          <NumberInput label={t('nrtm.nbiotBoost')} value={recipe.nb_iot.boost_db} min={boost.min} max={boost.max} step="any" disabled={disabled} helperText={t('nrtm.nbiotBoostHelp', boost)}
            onCommit={value => value !== null && set({ nb_iot: { ...(recipe.nb_iot ?? NB_IOT_DEFAULT), boost_db: value } })} />
        </Box>
        <Stack data-testid="nrtm-nbiot-rbs">{nbIotRbs(catalog, recipe).map(line => <Typography key={line} variant="caption" color="text.secondary">{line}</Typography>)}</Stack>
      </Stack></Paper>}

      <Stack spacing={.75} aria-live="polite" data-testid="nrtm-status">
        {(status === 'waiting' || status === 'resolving') && <><Typography variant="caption" color="text.secondary" role="status">{t('nrtm.resolving')}</Typography><LinearProgress aria-label={t('nrtm.resolving')} sx={{ height: 3 }} /></>}
        {status === 'duplicate' && <Alert severity="warning" data-testid="nrtm-duplicate">{t('nrtm.duplicate', { name: duplicateOf ?? '' })}</Alert>}
        {status === 'error' && <Alert severity="error"><AlertTitle>{t('nrtm.invalid')}</AlertTitle>{describeNrTmError(error).map(line => <Typography key={line} variant="body2">{line}</Typography>)}</Alert>}
        {summary && status !== 'error' && <Stack direction="row" useFlexGap sx={{ gap: 1, alignItems: 'center', flexWrap: 'wrap', opacity: fresh ? 1 : .55 }} data-testid="nrtm-derived">
          <Typography variant="body2">{t('nrtm.derived', { fs: formatNumber(summary.sample_rate_hz / 1e6, { maximumFractionDigits: 4 }), os: summary.oversampling, samples: formatNumber(summary.n_samples),
            duration: formatNumber(summary.duration_ms, { maximumFractionDigits: 3 }), subframes: summary.subframes })}</Typography>
          {summary.truncated && <Chip size="small" color="warning" label={t('nrtm.truncated', { used: summary.subframes, standard: summary.standard_subframes })} />}
        </Stack>}
        {status !== 'error' && !!summary?.warnings?.length && <Stack spacing={.75} sx={{ opacity: fresh ? 1 : .55 }} data-testid="nrtm-warnings">
          {summary.warnings.map(warning => <Alert key={warning} severity="warning" sx={{ overflowWrap: 'anywhere' }}>{warning}</Alert>)}
        </Stack>}
      </Stack>
    </>}

    {recipe && <Button size="small" variant="outlined" startIcon={<AddIcon />} disabled={disabled || !canAdd} sx={{ alignSelf: 'flex-start' }}
      onClick={() => onAdd(convertModel(catalog, recipe, nextModel(catalog, selectedModels, recipe.model)))}>{t('nrtm.addAnother')}</Button>}

    {recipe && <NrTmStandard catalog={catalog} info={info} recipe={recipe} summary={summary} stale={!fresh} />}
  </Stack>
}

/** One line per host carrier of the NB-IoT carriers (first for lower, last for upper, both for both): the eligible NR RBs of that carrier's bandwidth. */
function nbIotRbs(catalog: NrTmCatalog, recipe: NrTmRecipe) {
  return nbIotHosts(recipe).map(({ index, carrier }) => {
    const key = String(carrier.bandwidth_mhz)
    const eligible = catalog.nb_iot.eligible_rbs[key]
    if (!eligible?.length) return t('nrtm.nbiotNone', { index: index + 1 })
    return t('nrtm.nbiotEligible', { index: index + 1, bandwidth: key, count: eligible.length, lower: catalog.nb_iot.lower_rb[key] ?? eligible[0]!, upper: catalog.nb_iot.upper_rb[key] ?? eligible.at(-1)! })
  })
}
