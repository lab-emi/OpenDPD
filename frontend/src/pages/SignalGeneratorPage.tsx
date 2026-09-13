import ArrowForwardIcon from '@mui/icons-material/ArrowForward'
import Checkbox from '@mui/material/Checkbox'
import AddIcon from '@mui/icons-material/Add'
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutlined'
import DownloadIcon from '@mui/icons-material/Download'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import GraphicEqIcon from '@mui/icons-material/GraphicEq'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import UploadFileIcon from '@mui/icons-material/UploadFile'
import Accordion from '@mui/material/Accordion'
import AccordionDetails from '@mui/material/AccordionDetails'
import AccordionSummary from '@mui/material/AccordionSummary'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import ButtonBase from '@mui/material/ButtonBase'
import Chip from '@mui/material/Chip'
import FormControlLabel from '@mui/material/FormControlLabel'
import Grid from '@mui/material/Grid'
import IconButton from '@mui/material/IconButton'
import LinearProgress from '@mui/material/LinearProgress'
import MenuItem from '@mui/material/MenuItem'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import Switch from '@mui/material/Switch'
import TextField from '@mui/material/TextField'
import ToggleButton from '@mui/material/ToggleButton'
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup'
import Typography from '@mui/material/Typography'
import { useEffect, useRef, useState } from 'react'
import { Link as RouterLink } from 'react-router'
import { api, downloadFile } from '@/api/client'
import { useGenerateSignal, useGeneratedSignal, useGeneratorPresets, type GeneratedSignal, type GeneratorConfig, type GeneratorPreset } from '@/api/signalGenerator'
import { useStudioWorkflow } from '@/workflow/StudioWorkflow'
import { SignalGeneratorPlots } from '@/components/SignalGeneratorPlots'
import { ErrorState, LoadingState } from '@/components/StateBlock'
import { formatNumber, t, type MessageKey } from '@/i18n'
import { useStudioColors } from '@/theme'

const FAMILIES = ['nr', 'wifi6', 'wifi7', 'wifi8', 'custom'] as const
const ORDERS = [2, 4, 16, 64, 256, 1024, 4096]
const modulation = (m: number) => m === 2 ? 'BPSK' : m === 4 ? 'QPSK' : `${m}-QAM`
function NumberField({ label, value, onChange, unit = '', help }: { label: MessageKey; value: number; onChange: (value: number) => void; unit?: string; help?: string }) {
  const [raw, setRaw] = useState(String(value))
  const focused = useRef(false)
  useEffect(() => { if (!focused.current) setRaw(Number.isFinite(value) ? String(value) : '') }, [value])
  return <TextField fullWidth type="number" label={`${t(label)}${unit ? ` (${unit})` : ''}`} value={raw} onFocus={() => { focused.current = true }} onBlur={() => { focused.current = false; setRaw(Number.isFinite(value) ? String(value) : '') }} onChange={e => { setRaw(e.target.value); onChange(e.target.value === '' ? NaN : Number(e.target.value)) }} error={!Number.isFinite(value)} helperText={help} slotProps={{ inputLabel: { shrink: true }, htmlInput: { step: 'any' } }} />
}

export function SignalGeneratorPage() {
  const presets = useGeneratorPresets()
  const workflow = useStudioWorkflow()
  // Returning to this tab restores x instead of silently generating a new source.
  const [restoreId] = useState(workflow.state.origin === 'generated' ? workflow.state.inputId : null)
  const saved = useGeneratedSignal(restoreId)
  if (presets.isPending || (restoreId && saved.isPending)) return <LoadingState />
  if (presets.isError) return <ErrorState error={presets.error} onRetry={() => void presets.refetch()} />
  return <Generator presets={presets.data} saved={saved.data} />
}

function Generator({ presets, saved }: { presets: GeneratorPreset[]; saved?: GeneratedSignal }) {
  const colors = useStudioColors()
  const { selectInput } = useStudioWorkflow()
  const generate = useGenerateSignal()
  const [config, setConfig] = useState<GeneratorConfig>(() => (saved?.config ?? presets[0]!.config) as GeneratorConfig)
  const [advanced, setAdvanced] = useState(false)
  const [error, setError] = useState<unknown>(null)
  const [downloading, setDownloading] = useState(false)
  const [pilotText, setPilotText] = useState('')
  const preset = presets.find(p => p.preset_id === config.preset_id)
  const family = preset?.family ?? 'custom'
  const shared = config.shared_channel_settings ?? [config.channel_subcarriers, config.channel_modulations, config.channel_power_db].every(values => new Set(values).size === 1)
  const ofdm = config.waveform === 'ofdm'
  const result = generate.data ?? saved
  const stale = !!result && JSON.stringify(result.config) !== JSON.stringify(config)
  useEffect(() => {
    if (result && !stale) selectInput(result.signal_id, result.config.preset_id)
  }, [result, stale, selectInput])
  const count = config.length_mode === 'samples' ? config.n_samples : Math.floor(config.sample_rate_hz * config.duration_ms / 1000 + .5)
  const duration = count / config.sample_rate_hz * 1000
  const spacing = config.sample_rate_hz / (config.fft_size * config.oversampling)
  const validNumbers = Object.values(config).every(value => typeof value !== 'number' || Number.isFinite(value))
    && [...config.channel_subcarriers, ...config.channel_power_db, ...config.pilot_indices].every(Number.isFinite)
  const change = <K extends keyof GeneratorConfig>(key: K, value: GeneratorConfig[K]) => setConfig(old => ({ ...old, [key]: value }))
  const select = (entry: GeneratorPreset) => {
    setConfig({ ...entry.config, seed: config.seed, length_mode: config.length_mode, n_samples: config.n_samples, duration_ms: config.duration_ms } as GeneratorConfig)
    setPilotText((entry.config.pilot_indices ?? []).join(', ')); setError(null)
  }
  const numeric = (key: keyof GeneratorConfig, label: MessageKey, unit = '', scale = 1, help?: string) => <NumberField label={label} unit={unit} value={Number(config[key]) / scale} onChange={value => change(key, value * scale as never)} help={help} />
  const exportConfig = () => {
    const blob = URL.createObjectURL(new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' }))
    const link = document.createElement('a'); link.href = blob; link.download = 'signal-config.json'; link.click(); window.setTimeout(() => URL.revokeObjectURL(blob), 1000)
  }
  const uploadConfig = async (file: File) => {
    try {
      if (file.size > 65536) throw new Error(t('generator.configTooLarge'))
      const checked = await api.post<GeneratorConfig>('/signal-generator/validate', JSON.parse(await file.text()))
      setConfig(checked); setPilotText(checked.pilot_indices.join(', ')); setAdvanced(true); setError(null)
    } catch (e) { setError(e) }
  }
  return <Stack spacing={2.5} sx={{ maxWidth: 1600, mx: 'auto' }}>
    <Stack direction="row" useFlexGap sx={{ alignItems: 'center', justifyContent: 'space-between', gap: 1.5, flexWrap: 'wrap' }}>
      <Box><Typography variant="h1">{t('generator.title')}</Typography><Typography variant="body2" color="text.secondary" sx={{ mt: .75 }}>{t('generator.intro')}</Typography></Box>
      <Stack direction="row" spacing={1}><Button component="label" variant="outlined" size="small" startIcon={<UploadFileIcon />} disabled={generate.isPending}>{t('generator.importConfig')}<input type="file" hidden accept=".json,application/json" data-testid="generator-config-upload" onChange={e => { const file = e.target.files?.[0]; if (file) void uploadConfig(file); e.target.value = '' }} /></Button><Button size="small" onClick={exportConfig} startIcon={<DownloadIcon />}>{t('generator.exportConfig')}</Button></Stack>
    </Stack>
    <Box component="nav" aria-label={t('generator.choose')} sx={{ display: 'grid', gridTemplateColumns: { xs: 'repeat(2, minmax(0, 1fr))', sm: 'repeat(5, minmax(0, 1fr))' }, gap: 1 }}>
      {FAMILIES.map((key, index) => <ButtonBase key={key} onClick={() => { const match = presets.find(p => p.family === key); if (match) select(match) }} disabled={generate.isPending} aria-pressed={family === key} sx={{ textAlign: 'left', display: 'block', p: { xs: 1.5, lg: 2 }, border: 1, borderColor: family === key ? 'primary.main' : 'divider', borderRadius: 1.5, bgcolor: family === key ? colors.selected : 'background.paper', '&:hover': { borderColor: 'primary.main' }, '&.Mui-focusVisible': { outline: '3px solid', outlineColor: 'primary.main', outlineOffset: 2 } }}>
        <Typography variant="caption" sx={{ color: 'text.secondary', display: 'block', mb: .5 }}>0{index + 1} · {t(`generator.familyHint.${key}`)}</Typography>
        <Typography sx={{ fontWeight: 750, fontSize: { xs: 14, lg: 18 } }}>{t(`generator.family.${key}`)}</Typography>
      </ButtonBase>)}
    </Box>
    <Typography variant="body2" color="text.secondary">{t(family === 'wifi8' ? 'generator.wifi8Scope' : 'generator.scopeHelp')}</Typography>
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: 'minmax(0, 1fr)', lg: '350px minmax(0, 1fr)' }, alignItems: 'start', gap: 2.5 }}>
      <Stack spacing={1.5} component="fieldset" disabled={generate.isPending} sx={{ m: 0, p: 0, border: 0, minWidth: 0 }}>
        {result && <Paper sx={{ p: 2 }}><Stack spacing={1.5}><Typography variant="h3">{t('generator.next')}</Typography>
          <Typography variant="body2" color="text.secondary">{t('paInput.help')}</Typography>
          <Button variant="contained" endIcon={<ArrowForwardIcon />} disabled={stale} component={RouterLink} to={'/pa-library?input=' + encodeURIComponent(result.signal_id)}>{t('paInput.next')}</Button>
          <Button variant="outlined" startIcon={<DownloadIcon />} disabled={stale || downloading} onClick={() => { setDownloading(true); void downloadFile('/api/v1/signal-generator/signals/' + result.signal_id + '/input.csv').catch(setError).finally(() => setDownloading(false)) }}>{t('paInput.csv')}</Button>
          <Button variant="outlined" startIcon={<DownloadIcon />} disabled={stale || downloading} onClick={() => { setDownloading(true); void downloadFile('/api/v1/signal-generator/signals/' + result.signal_id + '/metadata.json').catch(setError).finally(() => setDownloading(false)) }}>{t('paInput.metadata')}</Button>
          <Button variant="outlined" startIcon={<DownloadIcon />} disabled={stale || downloading} onClick={() => { setDownloading(true); void downloadFile(result.download_url).catch(setError).finally(() => setDownloading(false)) }}>{t('generator.exportIq')}</Button>
          <Button component={RouterLink} to="/datasets?guide=start">{t('generator.useMeasured')}</Button>
        </Stack></Paper>}
        <Paper sx={{ p: 2.25 }}><Stack spacing={2.25}>
          <Typography variant="h2">{t('generator.setup')}</Typography>
          <TextField select fullWidth label={t('generator.preset')} value={preset?.preset_id ?? ''} onChange={e => { const entry = presets.find(p => p.preset_id === e.target.value); if (entry) select(entry) }}>
            {presets.filter(p => p.family === family).map(p => <MenuItem key={p.preset_id} value={p.preset_id}>{p.label}</MenuItem>)}
          </TextField>
          <Stack direction="row" useFlexGap sx={{ gap: .75, flexWrap: 'wrap' }}><Chip size="small" variant="outlined" label={`${config.sample_rate_hz / 1e6} MS/s`} /><Chip size="small" variant="outlined" label={`${config.bandwidth_hz / 1e6} MHz BW`} />{ofdm && <Chip size="small" variant="outlined" label={`${(spacing / 1000).toPrecision(4)} kHz SCS`} />}</Stack>
          <ToggleButtonGroup exclusive value={config.length_mode} fullWidth size="small" aria-label={t('generator.lengthMode')} onChange={(_, value: 'samples' | 'duration' | null) => { if (value) change('length_mode', value) }}><ToggleButton value="samples">{t('generator.bySamples')}</ToggleButton><ToggleButton value="duration">{t('generator.byDuration')}</ToggleButton></ToggleButtonGroup>
          {config.length_mode === 'samples' ? numeric('n_samples', 'generator.samples') : numeric('duration_ms', 'generator.duration', 'ms')}
          <Typography variant="body2" color="text.secondary" aria-live="polite" data-testid="generator-length">{Number.isFinite(count) ? formatNumber(count) : '—'} {t('generator.samplesUnit')} · {Number.isFinite(duration) ? duration.toPrecision(5) : '—'} ms</Typography>
          {(count < 256 || count > 1_000_000) && <Alert severity="error">{t('generator.lengthLimit')}</Alert>}
          <Button variant="contained" size="large" startIcon={<PlayArrowIcon />} disabled={generate.isPending || !validNumbers || count < 256 || count > 1_000_000} onClick={() => { setError(null); generate.mutate(config) }} sx={{ minHeight: 48 }}>{t(generate.isPending ? 'generator.generating' : 'generator.generate')}</Button>
          {generate.isPending && <LinearProgress aria-label={t('generator.generating')} />}
        </Stack></Paper>
        <Accordion expanded={advanced} onChange={(_, value) => setAdvanced(value)} disableGutters>
          <AccordionSummary expandIcon={<ExpandMoreIcon />}><Typography sx={{ fontWeight: 650 }}>{t('generator.advanced')}</Typography></AccordionSummary>
          <AccordionDetails><Stack spacing={2.5}>
            <Typography variant="body2" color="text.secondary">{t('generator.advancedHelp')}</Typography>
            {numeric('sample_rate_hz', 'generator.fs', 'MS/s', 1e6)}
            {numeric('bandwidth_hz', 'generator.bandwidth', 'MHz', 1e6)}
            {numeric('carrier_frequency_hz', 'generator.carrier', 'GHz', 1e9, t('generator.carrierHelp'))}
            {numeric('rms', 'generator.rms')}{numeric('seed', 'generator.seed')}
            {ofdm && <>
              <Typography variant="h3">OFDM / OFDMA</Typography>
              {numeric('fft_size', 'generator.fft')}
              <NumberField label="generator.spacing" unit="kHz" value={spacing / 1000} onChange={value => change('sample_rate_hz', value * 1000 * config.fft_size * config.oversampling)} help={t('generator.spacingHelp')} />
              <TextField select label={t('generator.oversampling')} value={config.oversampling} onChange={e => change('oversampling', Number(e.target.value) as GeneratorConfig['oversampling'])} helperText={t('generator.oversamplingHelp')}>{[1, 2, 4, 8].map(n => <MenuItem key={n} value={n}>{n}×</MenuItem>)}</TextField>
              <TextField select label={t('generator.cpMode')} value={config.cp_mode} onChange={e => change('cp_mode', e.target.value as GeneratorConfig['cp_mode'])}>{(['fixed', 'nr_normal', 'nr_extended'] as const).map(mode => <MenuItem key={mode} value={mode}>{t(`generator.cp.${mode}`)}</MenuItem>)}</TextField>
              {config.cp_mode === 'fixed' && numeric('cp_samples', 'generator.cpSamples', '', 1, `${config.cp_samples * config.oversampling} ${t('generator.outputSamples')}`)}
              <FormControlLabel label={t('generator.dcNull')} control={<Switch checked={config.dc_null} onChange={(_, value) => change('dc_null', value)} />} />
              <Typography variant="h3">{t('generator.channels')} · {config.channel_subcarriers.length}</Typography>
              <Typography variant="caption" color="text.secondary">{t('generator.channelsHelp')}</Typography>
              <FormControlLabel label={t('generator.sharedChannels')} control={<Checkbox checked={shared} onChange={(_, checked) => setConfig(old => ({ ...old, shared_channel_settings: checked,
                ...(checked ? { channel_subcarriers: old.channel_subcarriers.map(() => old.channel_subcarriers[0]!), channel_modulations: old.channel_modulations.map(() => old.channel_modulations[0]!), channel_power_db: old.channel_power_db.map(() => old.channel_power_db[0]!) } : {}) }))} />} />
              {(shared ? config.channel_subcarriers.slice(0, 1) : config.channel_subcarriers).map((carrierCount, index) => <Paper variant="outlined" key={index} sx={{ p: 1.5 }}><Stack spacing={1.75}>
                <Stack direction="row" sx={{ alignItems: 'center', justifyContent: 'space-between' }}><Typography variant="subtitle2">{shared ? t('generator.allChannels') : `${t('generator.channel')} ${index + 1}`}</Typography><IconButton size="small" disabled={config.channel_subcarriers.length === 1} aria-label={`${t('generator.removeChannel')} ${index + 1}`} onClick={() => setConfig(old => ({ ...old, channel_subcarriers: old.channel_subcarriers.filter((_, i) => i !== (shared ? old.channel_subcarriers.length - 1 : index)), channel_modulations: old.channel_modulations.filter((_, i) => i !== (shared ? old.channel_modulations.length - 1 : index)), channel_power_db: old.channel_power_db.filter((_, i) => i !== (shared ? old.channel_power_db.length - 1 : index)) }))}><DeleteOutlineIcon fontSize="small" /></IconButton></Stack>
                <NumberField label="generator.subcarriers" value={carrierCount} onChange={value => change('channel_subcarriers', config.channel_subcarriers.map((v, i) => shared || i === index ? value : v))} />
                <TextField select label={t('generator.modulation')} value={config.channel_modulations[index]} onChange={e => change('channel_modulations', config.channel_modulations.map((v, i) => shared || i === index ? Number(e.target.value) : v))}>{ORDERS.map(m => <MenuItem key={m} value={m}>{modulation(m)}</MenuItem>)}</TextField>
                <NumberField label="generator.channelPower" unit="dB" value={config.channel_power_db[index]!} onChange={value => change('channel_power_db', config.channel_power_db.map((v, i) => shared || i === index ? value : v))} />
              </Stack></Paper>)}
              <Button startIcon={<AddIcon />} variant="outlined" disabled={config.channel_subcarriers.length >= 16} onClick={() => setConfig(old => ({ ...old, shared_channel_settings: shared, channel_subcarriers: [...old.channel_subcarriers, shared ? old.channel_subcarriers[0]! : 26], channel_modulations: [...old.channel_modulations, shared ? old.channel_modulations[0]! : 64], channel_power_db: [...old.channel_power_db, shared ? old.channel_power_db[0]! : 0] }))}>{t('generator.addChannel')}</Button>
              {numeric('channel_gap_bins', 'generator.channelGap')}
              <TextField select label={t('generator.pilotMode')} value={config.pilot_mode} onChange={e => change('pilot_mode', e.target.value as GeneratorConfig['pilot_mode'])}>{(['comb', 'explicit', 'none'] as const).map(mode => <MenuItem key={mode} value={mode}>{t(`generator.pilot.${mode}`)}</MenuItem>)}</TextField>
              {config.pilot_mode === 'comb' && numeric('pilot_spacing', 'generator.pilotSpacing')}
              {config.pilot_mode === 'explicit' && <TextField label={t('generator.pilotIndices')} value={pilotText} helperText={t('generator.pilotIndicesHelp')} onChange={e => { setPilotText(e.target.value); change('pilot_indices', e.target.value.trim() ? e.target.value.trim().split(/[\s,]+/).map(Number) : []) }} />}
              {config.pilot_mode !== 'none' && numeric('pilot_boost_db', 'generator.pilotBoost', 'dB')}
            </>}
            {config.waveform === 'qam' && <>
              <TextField select label={t('generator.modulation')} value={config.modulation_order} onChange={e => change('modulation_order', Number(e.target.value) as GeneratorConfig['modulation_order'])}>{ORDERS.map(m => <MenuItem key={m} value={m}>{modulation(m)}</MenuItem>)}</TextField>
              {numeric('samples_per_symbol', 'generator.samplesPerSymbol')}{numeric('rrc_rolloff', 'generator.rolloff')}{numeric('rrc_span_symbols', 'generator.rrcSpan')}
            </>}
            {config.waveform === 'tone' && numeric('tone_frequency_hz', 'generator.toneFrequency', 'MHz', 1e6)}
            {config.waveform === 'multitone' && numeric('tone_count', 'generator.tones')}
            <Typography variant="h3">{t('generator.impairments')}</Typography>
            {numeric('frequency_offset_hz', 'generator.frequencyOffset', 'kHz', 1000)}
            <Grid container spacing={1.5}><Grid size={6}>{numeric('iq_gain_db', 'generator.iqGain', 'dB')}</Grid><Grid size={6}>{numeric('iq_phase_deg', 'generator.iqPhase', '°')}</Grid></Grid>
            <Grid container spacing={1.5}><Grid size={6}>{numeric('dc_i', 'generator.dcI')}</Grid><Grid size={6}>{numeric('dc_q', 'generator.dcQ')}</Grid></Grid>
            <FormControlLabel label={t('generator.awgn')} control={<Switch checked={config.snr_db !== null} onChange={(_, enabled) => change('snr_db', enabled ? 35 : null)} />} />
            {config.snr_db !== null && numeric('snr_db', 'generator.snr', 'dB')}
            <FormControlLabel label={t('generator.clipping')} control={<Switch checked={config.clip_db !== null} onChange={(_, enabled) => change('clip_db', enabled ? 8 : null)} />} />
            {config.clip_db !== null && numeric('clip_db', 'generator.clipLevel', 'dB')}
            <Button variant="contained" startIcon={<PlayArrowIcon />} disabled={generate.isPending || !validNumbers || count < 256 || count > 1_000_000} onClick={() => generate.mutate(config)}>{t('generator.applyGenerate')}</Button>
          </Stack></AccordionDetails>
        </Accordion>

        {(generate.isError || !!error) && <ErrorState error={error || generate.error} />}
      </Stack>
      <Box sx={{ minWidth: 0 }}>{result ? <SignalGeneratorPlots result={result} stale={stale} /> : <Paper sx={{ p: 5, minHeight: 450, display: 'grid', placeContent: 'center', textAlign: 'center' }}><GraphicEqIcon sx={{ fontSize: 60, color: 'primary.main', mx: 'auto', mb: 2 }} /><Typography variant="h2">{t(generate.isPending ? 'generator.generating' : 'generator.generate')}</Typography><Typography color="text.secondary" sx={{ mt: 1 }}>{t(generate.isError ? 'generator.checkParameters' : 'generator.firstPreview')}</Typography></Paper>}</Box>
    </Box>
  </Stack>
}
