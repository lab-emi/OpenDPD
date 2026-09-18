import ArrowForwardIcon from '@mui/icons-material/ArrowForward'
import DownloadIcon from '@mui/icons-material/Download'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import InsightsIcon from '@mui/icons-material/Insights'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import UploadFileIcon from '@mui/icons-material/UploadFile'
import Accordion from '@mui/material/Accordion'
import AccordionDetails from '@mui/material/AccordionDetails'
import AccordionSummary from '@mui/material/AccordionSummary'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Chip from '@mui/material/Chip'
import FormControlLabel from '@mui/material/FormControlLabel'
import LinearProgress from '@mui/material/LinearProgress'
import MenuItem from '@mui/material/MenuItem'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import Switch from '@mui/material/Switch'
import Tab from '@mui/material/Tab'
import Tabs from '@mui/material/Tabs'
import Table from '@mui/material/Table'
import TableBody from '@mui/material/TableBody'
import TableCell from '@mui/material/TableCell'
import TableContainer from '@mui/material/TableContainer'
import TableRow from '@mui/material/TableRow'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { Link as RouterLink, useSearchParams } from 'react-router'
import { analyzerDefaults, useAnalyzerDatasets, useAnalyzeSignal, useUploadSignal, type AnalyzerConfig, type AnalyzerDataset, type AnalyzerRequest, type AnalyzerSource, type AnalyzerSourceInfo, type SignalAnalysis } from '@/api/signalAnalyzer'
import { PlotlyChart, type PlotTrace } from '@/components/PlotlyChart'
import { SignalSpectrogram } from '@/components/SignalSpectrogram'
import { downloadFile } from '@/api/client'
import { MathFormula } from '@/components/MathFormula'
import { ErrorState, LoadingState } from '@/components/StateBlock'
import { formatNumber, t, type MessageKey } from '@/i18n'

const sourceKey = (s: AnalyzerSource) => `${s.kind}:${s.source_id}:${s.role ?? 'input'}`
function bandwidthText(value: number) {
  const scale = value >= 1e6 ? 1e6 : value >= 1e3 ? 1e3 : 1
  return `${(value / scale).toPrecision(4)} ${scale === 1e6 ? 'MHz' : scale === 1e3 ? 'kHz' : 'Hz'}`
}
function saveFile(name: string, value: unknown, csv = false) {
  const url = URL.createObjectURL(new Blob([csv ? String(value) : JSON.stringify(value, null, 2)], { type: csv ? 'text/csv' : 'application/json' }))
  const link = document.createElement('a'); link.href = url; link.download = name; link.click()
  window.setTimeout(() => URL.revokeObjectURL(url), 1000)
}

export function SignalAnalyzerPage() {
  const sources = useAnalyzerDatasets()
  const [params] = useSearchParams()
  if (sources.isPending) return <LoadingState />
  if (sources.isError) return <ErrorState error={sources.error} onRetry={() => void sources.refetch()} />
  return <Analyzer key={params.toString()} datasets={sources.data} />
}

function Analyzer({ datasets }: { datasets: AnalyzerDataset[] }) {
  const [params] = useSearchParams()
  const qc = useQueryClient()
  const [previous] = useState(() => qc.getQueryData<{ request: AnalyzerRequest; result: SignalAnalysis }>(['signal-analyzer-last']))
  const [selection, setSelection] = useState(() => params.get('source') ? `${params.get('kind')}:${params.get('source')}:${params.get('role') ?? 'input'}` : previous?.request.source ? sourceKey(previous.request.source) : datasets[0]?.signals[0] ? sourceKey(datasets[0].signals[0].source) : '')
  const [datasetId, setDatasetId] = useState(() => params.get('dataset') ?? datasets.find(d => d.signals.some(s => sourceKey(s.source) === selection))?.dataset_id ?? '')
  const [version, setVersion] = useState(() => params.get('source') ? params.get('version') ?? 'raw-v1' : previous?.request.source.version ?? 'raw-v1')
  const [uploaded, setUploaded] = useState<AnalyzerSourceInfo | null>(null)
  const availableDatasets: AnalyzerDataset[] = uploaded && !datasets.some(d => d.dataset_id === uploaded.source.source_id)
    ? [...datasets, { dataset_id: uploaded.source.source_id, name: uploaded.label, kind: 'upload', signals: [uploaded] }] : datasets
  const dataset = availableDatasets.find(d => d.dataset_id === datasetId)
  const available = [...new Map(availableDatasets.flatMap(d => d.signals).map(s => [sourceKey(s.source), s])).values()]
  const selected = dataset?.signals.find(s => sourceKey(s.source) === selection)
  const [config, setConfig] = useState<AnalyzerConfig>(() => previous && !params.get('source') ? { ...analyzerDefaults, ...previous.request.config } : { ...analyzerDefaults,
    sample_rate_hz: selected?.sample_rate_hz ?? analyzerDefaults.sample_rate_hz,
    bandwidth_hz: selected?.bandwidth_hz ?? Math.min(20e6, (selected?.sample_rate_hz ?? 80e6) / 3),
    n_samples: selected?.sample_count ? Math.min(262144, Math.max(256, selected.sample_count)) : 262144 })
  const [reference, setReference] = useState(previous?.request.reference && !params.get('source') ? sourceKey(previous.request.reference) : '')
  const [tab, setTab] = useState('spectrum')
  const [downloading, setDownloading] = useState(false)
  const [downloadError, setDownloadError] = useState<unknown>(null)
  const analyze = useAnalyzeSignal()
  const upload = useUploadSignal()
  const result = analyze.data ?? (!params.get('source') ? previous?.result : undefined)
  const ref = available.find(s => sourceKey(s.source) === reference)
  const request: AnalyzerRequest | null = selected ? { source: { ...selected.source, version }, config, reference: ref?.source ?? null } : null
  const lastRequest = analyze.variables ?? previous?.request
  const stale = !!result && JSON.stringify(lastRequest) !== JSON.stringify(request)
  const pending = analyze.isPending || upload.isPending
  const count = selected?.sample_count ? Math.max(0, Math.min(selected.sample_count - config.start_sample, config.n_samples)) : config.n_samples
  const finite = Object.values(config).every(v => typeof v !== 'number' || Number.isFinite(v))
  const change = <K extends keyof AnalyzerConfig>(key: K, value: AnalyzerConfig[K]) => setConfig(c => ({ ...c, [key]: value }))
  const choose = (value: string, item = available.find(s => sourceKey(s.source) === value)) => {
    setSelection(value)
    setVersion(item?.source.version ?? 'raw-v1')
    if (item) setConfig(c => ({ ...c, start_sample: 0, i_column: 0, q_column: 1, sample_format: 'auto',
      sample_rate_hz: item.sample_rate_hz ?? c.sample_rate_hz, bandwidth_hz: item.bandwidth_hz ?? Math.min(c.bandwidth_hz, (item.sample_rate_hz ?? c.sample_rate_hz) / 3),
      n_samples: item.sample_count ? Math.min(262144, Math.max(256, item.sample_count)) : c.n_samples }))
  }
  const numeric = (key: keyof AnalyzerConfig, label: MessageKey, scale = 1, unit = '') => <TextField fullWidth type="number" label={`${t(label)}${unit ? ` (${unit})` : ''}`} value={Number.isFinite(Number(config[key])) ? Number(config[key]) / scale : ''} onChange={e => change(key, (e.target.value === '' ? NaN : Number(e.target.value) * scale) as never)} slotProps={{ inputLabel: { shrink: true }, htmlInput: { step: 'any' } }} />
  return <Stack spacing={2.5} sx={{ maxWidth: 1600, mx: 'auto' }}>
    <Stack direction="row" useFlexGap sx={{ justifyContent: 'space-between', alignItems: 'center', gap: 1, flexWrap: 'wrap' }}>
      <Box><Typography variant="h1">{t('analyzer.title')}</Typography><Typography variant="body2" color="text.secondary" sx={{ mt: .75 }}>{t('analyzer.intro')}</Typography></Box>
      <Button component="label" variant="contained" startIcon={<UploadFileIcon />} disabled={pending}>{t('analyzer.upload')}<input type="file" hidden accept=".csv,text/csv" data-testid="analyzer-upload" onChange={e => { const file = e.target.files?.[0]; if (file) upload.mutate(file, { onSuccess: info => { setUploaded(info); setDatasetId(info.source.source_id); choose(sourceKey(info.source), info) } }); e.target.value = '' }} /></Button>
    </Stack>
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: 'minmax(0,1fr)', lg: '330px minmax(0,1fr)' }, gap: 2.5, alignItems: 'start' }}>
      <Stack spacing={1.5} component="fieldset" disabled={pending} sx={{ m: 0, p: 0, border: 0, minWidth: 0 }}>
        <Paper sx={{ p: 2 }}><Stack spacing={2}>
          <Typography variant="h2">{t('analyzer.dataset')}</Typography>
          <TextField select fullWidth label={t('analyzer.dataset')} value={dataset?.dataset_id ?? ''} sx={{ '& .MuiSelect-select': { whiteSpace: 'normal', overflowWrap: 'anywhere' } }} onChange={e => {
            setDatasetId(e.target.value)
            const first = availableDatasets.find(d => d.dataset_id === e.target.value)?.signals[0]
            choose(first ? sourceKey(first.source) : '', first)
          }}>{availableDatasets.map(item => <MenuItem key={item.dataset_id} value={item.dataset_id} sx={{ whiteSpace: 'normal', overflowWrap: 'anywhere', maxWidth: 600 }}>{item.name}</MenuItem>)}</TextField>
          <TextField select fullWidth label={t('analyzer.datasetSignal')} value={selected ? selection : ''} disabled={!dataset} onChange={e => choose(e.target.value, dataset?.signals.find(s => sourceKey(s.source) === e.target.value))}>
            {dataset?.signals.map(item => <MenuItem key={sourceKey(item.source)} value={sourceKey(item.source)} sx={{ whiteSpace: 'normal', maxWidth: 600 }}>{item.label}</MenuItem>)}
          </TextField>
          {dataset?.download_url && <Button startIcon={<DownloadIcon />} disabled={downloading} onClick={() => {
            setDownloading(true); setDownloadError(null)
            void downloadFile(dataset.download_url!).catch(setDownloadError).finally(() => setDownloading(false))
          }}>{t('generator.downloadDataset')}</Button>}
          {!!downloadError && <ErrorState error={downloadError} />}
          {selection && !selected && <Alert severity="warning">{t('analyzer.sourceMissing')}</Alert>}
          {selected?.source.kind === 'upload' && <Typography variant="body2" color="text.secondary">{t('analyzer.csvHelp')}</Typography>}
          {selected && <Stack direction="row" useFlexGap sx={{ gap: .75, flexWrap: 'wrap' }}><Chip size="small" label={selected.origin.toUpperCase()} /><Chip size="small" variant="outlined" label={`${formatNumber(selected.sample_count)} ${t('analyzer.samples')}`} /></Stack>}
          <Button variant="contained" size="large" startIcon={<PlayArrowIcon />} disabled={!request || !finite || count < 256 || pending} onClick={() => { if (request) analyze.mutate(request) }}>{t('analyzer.analyze')}</Button>
          <Typography aria-live="polite" data-testid="analyzer-count" variant="body2">{formatNumber(count)} {t('analyzer.samples')} · {(count / config.sample_rate_hz * 1000).toPrecision(5)} ms</Typography>
          {selected?.source.kind === 'generated' && <Button component={RouterLink} to={'/pa-library?input=' + selected.source.source_id} endIcon={<ArrowForwardIcon />}>{t('paInput.next')}</Button>}
          {selected?.source.kind === 'dataset' && <Button component={RouterLink} to={'/datasets/' + selected.source.source_id} endIcon={<ArrowForwardIcon />}>{t('analyzer.openDataset')}</Button>}
          {!selected && <Button component={RouterLink} to="/signal-generator" endIcon={<ArrowForwardIcon />}>{t('generator.title')}</Button>}
          {pending && <LinearProgress />}
          {analyze.isError && <ErrorState error={analyze.error} />}{upload.isError && <ErrorState error={upload.error} />}
        </Stack></Paper>
        <Paper sx={{ p: 2 }}><Stack spacing={2}>
          {numeric('sample_rate_hz', 'generator.fs', 1e6, 'MS/s')}{numeric('bandwidth_hz', 'analyzer.bandwidth', 1e6, 'MHz')}
          {!selected?.sample_rate_hz && <Typography variant="caption" color="warning.main">{t('analyzer.rateRequired')}</Typography>}
          {selected?.source.kind === 'upload' && <>
            <TextField select label={t('analyzer.format')} value={config.sample_format} onChange={e => change('sample_format', e.target.value as AnalyzerConfig['sample_format'])}>{(['auto', 'real', 'complex', 'iq'] as const).map(f => <MenuItem key={f} value={f}>{t(`analyzer.format.${f}`)}</MenuItem>)}</TextField>
            <TextField select label={t('analyzer.iColumn')} value={config.i_column} onChange={e => change('i_column', Number(e.target.value))}>{selected.columns?.map((name, i) => <MenuItem key={i} value={i}>{i + 1} · {name}</MenuItem>)}</TextField>
            {(config.sample_format === 'iq' || (config.sample_format === 'auto' && !selected.complex_columns?.length && (selected.columns?.length ?? 0) > 1)) && <TextField select label={t('analyzer.qColumn')} value={config.q_column} onChange={e => change('q_column', Number(e.target.value))}>{selected.columns?.map((name, i) => <MenuItem key={i} value={i}>{i + 1} · {name}</MenuItem>)}</TextField>}
          </>}
          {numeric('start_sample', 'analyzer.start')}{numeric('n_samples', 'analyzer.length')}
        </Stack></Paper>
        <Accordion disableGutters><AccordionSummary expandIcon={<ExpandMoreIcon />}><Typography sx={{ fontWeight: 650 }}>{t('analyzer.measurementSettings')}</Typography></AccordionSummary><AccordionDetails><Stack spacing={2}>
          <TextField select label={t('analyzer.fft')} value={config.fft_size} onChange={e => change('fft_size', Number(e.target.value))}>{[256, 512, 1024, 2048, 4096, 8192, 16384].map(v => <MenuItem value={v} key={v}>{v}</MenuItem>)}</TextField>
          <TextField select label={t('analyzer.window')} value={config.window} onChange={e => change('window', e.target.value as AnalyzerConfig['window'])}>{['hann', 'hamming', 'blackman', 'boxcar'].map(v => <MenuItem key={v} value={v}>{v}</MenuItem>)}</TextField>
          <TextField select label={t('analyzer.overlap')} value={config.overlap} onChange={e => change('overlap', Number(e.target.value) as AnalyzerConfig['overlap'])}>{[0, .5, .75].map(v => <MenuItem key={v} value={v}>{v * 100}%</MenuItem>)}</TextField>
          {numeric('occupied_percent', 'analyzer.occupied', 1, '%')}{numeric('center_hz', 'analyzer.center', 1e6, 'MHz')}
          <TextField type="number" label={t('analyzer.adjacentOffset')} value={config.adjacent_offset_hz === null ? '' : config.adjacent_offset_hz / 1e6} placeholder={String(config.bandwidth_hz / 1e6)} helperText={t('analyzer.adjacentHelp')} onChange={e => change('adjacent_offset_hz', e.target.value === '' ? null : Number(e.target.value) * 1e6)} slotProps={{ inputLabel: { shrink: true } }} />
          <FormControlLabel label={t('analyzer.removeDc')} control={<Switch checked={config.remove_dc} onChange={(_, v) => change('remove_dc', v)} />} />
          {numeric('frequency_shift_hz', 'analyzer.shift', 1000, 'kHz')}
          {numeric('samples_per_symbol', 'generator.samplesPerSymbol')}{numeric('symbol_offset', 'analyzer.symbolOffset')}
          <TextField select label={t('analyzer.reference')} value={ref ? reference : ''} onChange={e => setReference(e.target.value)}><MenuItem value="">{t('analyzer.noReference')}</MenuItem>{available.map(item => <MenuItem key={sourceKey(item.source)} value={sourceKey(item.source)}>{item.label}</MenuItem>)}</TextField>
          <Typography variant="caption">{t('analyzer.referenceHelp')}</Typography>
          <FormControlLabel label={t('analyzer.gainFit')} control={<Switch disabled={!ref} checked={config.reference_gain_fit} onChange={(_, v) => change('reference_gain_fit', v)} />} />
        </Stack></AccordionDetails></Accordion>
      </Stack>
      <Stack spacing={1.5} sx={{ minWidth: 0 }}>
        {result ? <>
          {stale && <Alert severity="warning">{t('analyzer.stale')}</Alert>}
          <Stack direction="row" useFlexGap sx={{ alignItems: 'center', flexWrap: 'wrap', gap: 1 }}><Typography variant="h2" sx={{ flex: 1 }}>{result.source.label}</Typography><Button startIcon={<DownloadIcon />} onClick={() => saveFile('signal-analysis.json', result)}>{t('analyzer.exportJson')}</Button><Button startIcon={<DownloadIcon />} onClick={() => saveFile('signal-psd.csv', 'frequency_hz,psd_dbfs_hz\n' + result.frequency_hz.map((f, i) => `${f},${result.psd_dbfs_hz[i]}`).join('\n'), true)}>{t('analyzer.exportPsd')}</Button></Stack>
          <Typography variant="body2" color="text.secondary">{formatNumber(result.sample_count)} {t('analyzer.samples')} · [{result.sample_range.join(', ')}) · {result.real_signal ? t('analyzer.format.real') : t('analyzer.format.complex')}</Typography>
          <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(3,minmax(0,1fr))', gap: 1 }}>{['papr', 'obw', 'rms'].map(key => { const m = result.measurements.find(v => v.key === key); return <Paper key={key} sx={{ p: 1.5 }}><Typography variant="caption" color="text.secondary">{m?.label}</Typography><Typography sx={{ fontSize: { xs: 16, lg: 22 }, fontWeight: 700 }}>{m?.value == null ? '—' : key === 'obw' ? bandwidthText(m.value) : `${m.value.toPrecision(4)} ${m.unit}`}</Typography></Paper> })}</Box>
          <Tabs value={tab} onChange={(_, v: string) => setTab(v)} variant="scrollable" scrollButtons="auto" aria-label={t('generator.visualizations')}>{(['spectrum', 'time', 'distribution', 'eye', 'metrics'] as const).map(v => <Tab key={v} value={v} label={t(`analyzer.tab.${v}`)} />)}</Tabs>
          <AnalyzerPlots result={result} tab={tab} />
        </> : <Paper sx={{ p: 5, minHeight: 450, display: 'grid', placeContent: 'center', textAlign: 'center' }}><InsightsIcon sx={{ color: 'primary.main', fontSize: 64, mx: 'auto', mb: 2 }} /><Typography variant="h2">{t('analyzer.first')}</Typography><Typography color="text.secondary" sx={{ mt: 1 }}>{t('analyzer.intro')}</Typography></Paper>}
      </Stack>
    </Box>
  </Stack>
}

function AnalyzerPlots({ result: a, tab }: { result: SignalAnalysis; tab: string }) {
  const axis = (text: string) => ({ title: { text } })
  const key = a.source_sha256 + JSON.stringify(a.config)
  const chart = (title: string, traces: PlotTrace[], x: string, y: string, legend = false, equalAspect = false) => <Paper sx={{ p: 1.5, minWidth: 0 }}><PlotlyChart title={title} traces={traces} height={310} viewKey={key} layout={{ xaxis: axis(x), yaxis: { ...axis(y), ...(equalAspect ? { scaleanchor: 'x', scaleratio: 1 } : {}) }, showlegend: legend }} /></Paper>
  if (tab === 'spectrum') return <Stack spacing={1.5}>{chart(`${a.source.label} · PSD`, [{ x: a.frequency_hz.map(v => v / 1e6), y: a.psd_dbfs_hz, name: 'PSD' }], t('generator.frequencyAxis'), 'dBFS/Hz')}<Paper sx={{ p: 2, minWidth: 0 }}><SignalSpectrogram result={a} /></Paper></Stack>
  if (tab === 'time') return <Stack spacing={1.5}>{chart(t('generator.time'), [{ x: a.time_s.map(v => v * 1e6), y: a.time_i, name: 'I' }, { x: a.time_s.map(v => v * 1e6), y: a.time_q, name: 'Q', line: { dash: 'dot' } }, { x: a.time_s.map(v => v * 1e6), y: a.envelope, name: '|I+jQ|', visible: 'legendonly' }], 'µs', t('generator.amplitude'), true)}{chart(t('analyzer.instantFrequency'), [{ x: a.time_s.flatMap((v, i) => a.instantaneous_frequency_hz[i] === null ? [] : [v * 1e6]), y: a.instantaneous_frequency_hz.filter((v): v is number => v !== null).map(v => v / 1e6), mode: 'markers', marker: { size: 2 } }], 'µs', 'MHz')}<Typography variant="caption">{t('analyzer.timeHelp')}</Typography></Stack>
  if (tab === 'distribution') return <Stack spacing={1.5}>{chart(t('analyzer.scatter'), [{ x: a.scatter_i, y: a.scatter_q, mode: 'markers', marker: { size: 3, opacity: .55 } }], 'I', 'Q', false, true)}{chart(t('generator.ccdf'), [{ x: a.ccdf_db.filter((_, i) => a.ccdf_probability[i]! > 0), y: a.ccdf_probability.filter(v => v > 0).map(v => Math.log10(v)) }], 'P / Pavg (dB)', 'log₁₀ Pr(P/Pavg > x)')}{chart(t('analyzer.histogram'), [{ x: a.histogram_amplitude, y: a.histogram_probability }], '|I+jQ|', t('analyzer.probability'))}</Stack>
  if (tab === 'eye') return <Stack spacing={1.5}><Alert severity="info">{t('analyzer.eyeHelp')}</Alert>{(['i', 'q'] as const).map(component => chart(`${t('analyzer.eye')} · ${component.toUpperCase()}`, a[component === 'i' ? 'eye_i' : 'eye_q'].map(y => ({ x: y.map((_, i) => i / (a.config.samples_per_symbol ?? 8)), y, line: { width: 1, color: component === 'i' ? '#16758C' : '#C47722' } })), t('analyzer.symbolPeriods'), t('generator.amplitude')))}</Stack>
  return <Stack spacing={2}><Paper><TableContainer><Table size="small"><TableBody>{a.measurements.map(m => <TableRow key={m.key}><TableCell component="th">{m.label}{m.reason && <Typography variant="caption" sx={{ display: 'block' }} color="text.secondary">{m.reason}</Typography>}</TableCell><TableCell align="right">{m.value === null ? '—' : `${m.value.toPrecision(7)} ${m.unit}`}</TableCell></TableRow>)}</TableBody></Table></TableContainer></Paper>
    <Paper sx={{ p: 2 }}><Typography variant="h3">{t('analyzer.definitions')}</Typography>{[String.raw`P_{\mathrm{avg}}=\frac{1}{N}\sum_{n=0}^{N-1}|x[n]|^2`, String.raw`\mathrm{PAPR}=10\log_{10}\frac{\max_n|x[n]|^2}{P_{\mathrm{avg}}}`, String.raw`\mathrm{ACPR}_{L,R}=10\log_{10}\frac{\int_{B_{L,R}}S_{xx}(f)\,df}{\int_{B_0}S_{xx}(f)\,df}`, String.raw`\mathrm{ENBW}=f_s\frac{\sum_n w[n]^2}{(\sum_n w[n])^2}`, String.raw`\epsilon_{\mathrm{rms}}=100\sqrt{\frac{\sum_n|x[n]-g r[n]|^2}{\sum_n|g r[n]|^2}}\,\%`].map(latex => <MathFormula key={latex} latex={latex} display />)}<Typography variant="body2">{t('analyzer.formulaHelp')}</Typography></Paper>
    {a.notes.map(note => <Typography key={note} variant="body2" color="text.secondary">{note}</Typography>)}<Typography variant="caption" sx={{ overflowWrap: 'anywhere' }}>SHA-256 · {a.source_sha256}</Typography>
  </Stack>
}
