import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Chip from '@mui/material/Chip'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import Tab from '@mui/material/Tab'
import Tabs from '@mui/material/Tabs'
import Table from '@mui/material/Table'
import TableBody from '@mui/material/TableBody'
import TableCell from '@mui/material/TableCell'
import TableContainer from '@mui/material/TableContainer'
import TableRow from '@mui/material/TableRow'
import Typography from '@mui/material/Typography'
import { useState } from 'react'
import type { GeneratedSignal } from '@/api/signalGenerator'
import { formatNumber, t } from '@/i18n'
import { PlotlyChart, type PlotTrace } from './PlotlyChart'

const PALETTE = ['#16758C', '#C47722', '#8262BB', '#3E8754', '#BB556C', '#5580BE']

export function SignalGeneratorPlots({ result, stale }: { result: GeneratedSignal; stale: boolean }) {
  const [tab, setTab] = useState('overview')
  const a = result.analysis
  const axis = (text: string) => ({ title: { text } })
  const legendSpace = { margin: { l: 50, r: 12, t: 68, b: 42 }, legend: { orientation: 'h' as const, x: 0, y: 1.02, yanchor: 'bottom' as const, font: { size: 11 } } }
  const ccdf = a.ccdf_probability.flatMap((probability, index) => probability > 0 ? [{ x: a.ccdf_db[index]!, y: Math.log10(probability) }] : [])
  const allocations: PlotTrace[] = a.allocation.flatMap((entry, index) => {
    const bins = entry.subcarriers as number[]
    const pilots = entry.pilots as number[]
    const selected = new Set(pilots)
    const data = bins.filter(bin => !selected.has(bin))
    return [
      { x: data, y: data.map(() => index + 1), mode: 'markers' as const, name: `${t('generator.channel')} ${index + 1}`, marker: { color: PALETTE[index % PALETTE.length], size: 5 } },
      { x: pilots, y: pilots.map(() => index + 1), mode: 'markers' as const, name: `${t('generator.pilots')} ${index + 1}`, marker: { color: '#B45C21', size: 9, symbol: 'cross' as const } },
    ]
  })
  const metrics = [
    [t('generator.samples'), formatNumber(a.sample_count)],
    [t('generator.duration'), `${a.duration_ms.toPrecision(6)} ms`],
    [t('generator.fs'), `${a.sample_rate_hz / 1e6} MHz`],
    [t('generator.carrier'), `${result.config.carrier_frequency_hz / 1e9} GHz`],
    [t('generator.spacing'), a.subcarrier_spacing_hz === null ? '—' : `${a.subcarrier_spacing_hz / 1000} kHz`],
    [t('generator.symbolDuration'), a.useful_symbol_us === null ? '—' : `${a.useful_symbol_us.toPrecision(6)} µs`],
    [t('generator.cpPattern'), a.cp_lengths_samples.length ? [...new Set(a.cp_lengths_samples)].join(', ') : '—'],
    [t('generator.completeSymbols'), formatNumber(a.complete_symbols)],
    [t('generator.trailing'), formatNumber(a.trailing_samples)],
    [t('generator.dataPilots'), `${a.data_carriers} / ${a.pilot_carriers}`],
    ['PAPR', `${a.papr_db.toFixed(3)} dB`],
    [t('generator.meanPower'), `${a.mean_power_dbfs.toFixed(3)} dBFS`],
    ['RMS / peak', `${a.rms.toPrecision(5)} / ${a.peak.toPrecision(5)}`],
    [t('generator.occupied'), `${(a.occupied_bandwidth_99_hz / 1e6).toPrecision(5)} MHz`],
    [t('generator.dc'), a.dc_magnitude.toPrecision(5)],
    [t('generator.evm'), a.evm_percent === null ? '—' : `${a.evm_percent.toPrecision(4)} % (${a.evm_symbols} ${t('generator.symbols')})`],
  ]
  return <Stack spacing={1.5} data-testid="signal-generator-results">
    <Stack direction="row" useFlexGap sx={{ gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
      <Typography variant="h2" sx={{ flex: 1 }}>{t('generator.preview')}</Typography>
      <Chip size="small" variant="outlined" label="SYNTHETIC" color="warning" />
      <Chip size="small" label={t(`generator.coverage.${result.coverage}`)} />
    </Stack>
    {stale && <Alert severity="warning">{t('generator.stale')}</Alert>}
    <Typography variant="caption" color="text.secondary">{result.config.preset_id} · {formatNumber(a.sample_count)} {t('generator.samplesUnit')} · {a.duration_ms.toPrecision(4)} ms</Typography>
    <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(3, minmax(0, 1fr))', gap: 1 }}>
      {[
        ['PAPR', `${a.papr_db.toFixed(2)} dB`],
        [t('generator.occupiedShort'), `${(a.occupied_bandwidth_99_hz / 1e6).toFixed(2)} MHz`],
        [t('generator.rms'), a.rms.toFixed(4)],
      ].map(([label, value]) => <Paper key={label} sx={{ p: 1.5 }}><Typography variant="caption" color="text.secondary">{label}</Typography><Typography sx={{ fontSize: { xs: 17, lg: 22 }, fontWeight: 700, fontVariantNumeric: 'tabular-nums' }}>{value}</Typography></Paper>)}
    </Box>
    <Tabs value={tab} onChange={(_, value: string) => setTab(value)} variant="scrollable" scrollButtons="auto" aria-label={t('generator.visualizations')}>
      <Tab value="overview" label={t('generator.overview')} />
      <Tab value="allocation" label={t('generator.allocation')} disabled={!allocations.length} />
      <Tab value="metrics" label={t('generator.metrics')} />
    </Tabs>
    {tab === 'overview' && <Box sx={{ display: 'grid', gridTemplateColumns: { xs: 'minmax(0, 1fr)', lg: 'repeat(2, minmax(0, 1fr))' }, gap: 1.5 }}>
      <Paper sx={{ p: 1.5, minWidth: 0 }}><PlotlyChart title={t('generator.spectrum')} height={270} viewKey={result.signal_id} traces={[{ x: a.frequency_mhz, y: a.psd_dbfs_hz, name: 'PSD', line: { color: PALETTE[0], width: 1.5 } }]} layout={{ xaxis: axis(t('generator.frequencyAxis')), yaxis: axis('dBFS / Hz'), showlegend: false }} />
        <Typography variant="caption" color="text.secondary">{t('generator.psdHelp')}</Typography></Paper>
      <Paper sx={{ p: 1.5, minWidth: 0 }}><PlotlyChart title={t('generator.time')} height={270} viewKey={result.signal_id} traces={[
        { x: a.time_us, y: a.time_i, name: 'I', line: { color: PALETTE[0], width: 1 } },
        { x: a.time_us, y: a.time_q, name: 'Q', line: { color: PALETTE[1], width: 1, dash: 'dot' } },
        { x: a.time_us, y: a.time_envelope, name: '|I+jQ|', line: { color: PALETTE[2], width: 1 }, visible: 'legendonly' },
      ]} layout={{ ...legendSpace, xaxis: axis('µs'), yaxis: axis(t('generator.amplitude')) }} /><Typography variant="caption" color="text.secondary">{t('generator.timeHelp', { count: a.time_us.length })}</Typography></Paper>
      <Paper sx={{ p: 1.5, minWidth: 0 }}>{a.constellation_i.length ? <PlotlyChart title={t('generator.constellation')} height={270} viewKey={result.signal_id} traces={[
        { x: a.reference_i, y: a.reference_q, mode: 'markers', name: t('generator.reference'), marker: { size: 7, opacity: .4, color: PALETTE[1], symbol: 'cross' } },
        { x: a.constellation_i, y: a.constellation_q, mode: 'markers', name: t(result.config.waveform === 'ofdm' ? 'generator.recovered' : 'generator.transmitted'), marker: { size: 4, opacity: .65, color: PALETTE[0] } },
      ]} layout={{ ...legendSpace, xaxis: axis('I'), yaxis: { ...axis('Q'), scaleanchor: 'x', scaleratio: 1 } }} /> : <Stack sx={{ minHeight: 270, justifyContent: 'center', p: 2 }}><Typography variant="h3">{t('generator.constellation')}</Typography><Typography color="text.secondary">{t('generator.noConstellation')}</Typography></Stack>}
        <Typography variant="caption" color="text.secondary">{t(result.config.waveform === 'qam' ? 'generator.qamConstellationHelp' : 'generator.constellationHelp')}</Typography></Paper>
      <Paper sx={{ p: 1.5, minWidth: 0 }}><PlotlyChart title={t('generator.ccdf')} height={270} viewKey={result.signal_id} traces={[{ x: ccdf.map(p => p.x), y: ccdf.map(p => p.y), name: 'CCDF', mode: 'lines+markers', marker: { size: 2, color: PALETTE[2] }, line: { color: PALETTE[2], width: 2 } }]} layout={{ xaxis: axis(t('generator.ccdfAxis')), yaxis: axis('log₁₀ Pr(P / Pavg > x)'), showlegend: false }} />
        <Typography variant="caption" color="text.secondary">{t('generator.ccdfHelp')}</Typography></Paper>
    </Box>}
    {tab === 'allocation' && <Paper sx={{ p: 2, minWidth: 0 }}><PlotlyChart title={t('generator.allocation')} height={360} viewKey={result.signal_id} traces={allocations} layout={{ xaxis: axis(t('generator.binIndex')), yaxis: axis(t('generator.channel')), showlegend: false }} /><Typography variant="body2" color="text.secondary">{t('generator.allocationHelp')}</Typography></Paper>}
    {tab === 'metrics' && <Stack spacing={2}>
      <Paper><TableContainer><Table size="small"><TableBody>{metrics.map(([key, value]) => <TableRow key={key}><TableCell component="th">{key}</TableCell><TableCell align="right" sx={{ fontVariantNumeric: 'tabular-nums' }}>{value}</TableCell></TableRow>)}</TableBody></Table></TableContainer></Paper>
      <Typography variant="h3">{t('generator.scope')}</Typography>
      <Typography variant="body2">{t('generator.scopeHelp')}</Typography>
      {a.notes.map(note => <Typography key={note} variant="body2" color="text.secondary">{note}</Typography>)}
      <Typography variant="caption" sx={{ overflowWrap: 'anywhere' }}>IQ SHA-256 · {result.iq_sha256}</Typography>
    </Stack>}
  </Stack>
}
