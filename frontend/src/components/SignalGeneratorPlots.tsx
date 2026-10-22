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
import TableHead from '@mui/material/TableHead'
import TableRow from '@mui/material/TableRow'
import Typography from '@mui/material/Typography'
import { useState } from 'react'
import type { GeneratedSignal } from '@/api/signalGenerator'
import { formatNumber, t } from '@/i18n'
import { nbIotOf, shortModel, signalName, type NbIotAllocation } from '@/utils/nrTestModel'
import { Facts, Scroll } from './NrTmFacts'
import { PlotlyChart, type PlotTrace } from './PlotlyChart'
import { useStudioColors } from '@/theme'

export function SignalGeneratorPlots({ result, stale }: { result: GeneratedSignal; stale: boolean }) {
  const PALETTE = useStudioColors().chart
  const [tab, setTab] = useState('overview')
  const a = result.analysis
  const axis = (text: string) => ({ title: { text } })
  const legendSpace = { margin: { l: 50, r: 12, t: 68, b: 42 }, legend: { orientation: 'h' as const, x: 0, y: 1.02, yanchor: 'bottom' as const, font: { size: 12 } } }
  const ccdf = a.ccdf_probability.flatMap((probability, index) => probability > 0 ? [{ x: a.ccdf_db[index]!, y: Math.log10(probability) }] : [])
  const summary = a.nr_test_model
  // Allocation bins and EVM subcarrier indices of every signal are absolute: signed bins from baseband DC in units of `analysis.subcarrier_spacing_hz`
  // (the lowest spacing of a multi-numerology test model). A test model therefore plots at bin x spacing on one shared frequency axis (MHz); the
  // centre offsets of its carriers are already contained in the bins and must not be added. Matrix signals keep the bin index.
  const binHz = summary && a.subcarrier_spacing_hz ? a.subcarrier_spacing_hz : null
  const placed = binHz !== null
  const place = (bins: number[]) => binHz === null ? bins : bins.map(bin => bin * binHz / 1e6)
  const nbIot = a.allocation.map(entry => nbIotOf(entry))
  // Wide test-model captures list one subcarrier per resource block (`subcarrier_step` 12) to keep the stored analysis short.
  const step = Math.max(1, ...a.allocation.map(entry => typeof entry.subcarrier_step === 'number' ? entry.subcarrier_step : 1))
  const allocations: PlotTrace[] = a.allocation.flatMap((entry, index) => {
    const bins = entry.subcarriers as number[]
    const pilots = entry.pilots as number[]
    const selected = new Set(pilots)
    const data = bins.filter(bin => !selected.has(bin))
    return [
      { x: place(data), y: data.map(() => index + 1), mode: 'markers' as const, name: placed ? t('nrtm.carrierN', { index: index + 1 }) : `${t('generator.channel')} ${index + 1}`, marker: { color: PALETTE[index % PALETTE.length], size: 5 } },
      { x: place(pilots), y: pilots.map(() => index + 1), mode: 'markers' as const, name: `${t('generator.pilots')} ${index + 1}`, marker: { color: PALETTE[1], size: 9, symbol: 'cross' as const } },
      // The punctured NR RBs of NR-N-TM, drawn over their host carrier.
      ...(nbIot[index]?.carriers ?? []).map(nb => ({ x: place(nb.subcarriers), y: nb.subcarriers.map(() => index + 1), mode: 'markers' as const, name: `${t('nrtm.nbiot')} · RB ${nb.rb}`,
        marker: { color: PALETTE[3 % PALETTE.length], size: 8, symbol: 'diamond' as const } })),
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
    ...(summary ? [
      [t('nrtm.model'), shortModel(summary.model)],
      [t('nrtm.nativeRate'), `${summary.native_sample_rate_hz / 1e6} MHz × ${summary.oversampling}`],
      [t('nrtm.captured'), t('nrtm.subframesOf', { used: summary.subframes, standard: summary.standard_subframes })],
      [t('nrtm.aggregated'), `${(summary.aggregated_bandwidth_hz / 1e6).toPrecision(5)} MHz`],
      ...(summary.nb_iot_rbs?.length ? [[t('nrtm.nbiot'), `RB ${summary.nb_iot_rbs.join(', ')}`]] : []),
    ] : []),
  ]
  return <Stack spacing={1.5} data-testid="signal-generator-results">
    <Stack direction="row" useFlexGap sx={{ gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
      <Typography variant="h2" sx={{ flex: 1 }}>{t('paInput.title')}</Typography>
      <Chip size="small" variant="outlined" label={result.origin === 'uploaded' ? t('signalImport.uploaded') : 'SYNTHETIC'} color="warning" />
      <Chip size="small" label={t(`generator.coverage.${result.coverage}`)} />
    </Stack>
    {stale && <Alert severity="warning">{t('generator.stale')}</Alert>}
    {summary?.warnings?.map(warning => <Alert key={warning} severity="warning" sx={{ overflowWrap: 'anywhere' }} data-testid="nrtm-result-warning">{warning}</Alert>)}
    <Typography variant="body2" color="text.secondary">{t('paInput.help')}</Typography>
    <Typography variant="caption" color="text.secondary" data-testid="generator-result-name"><span title={result.config.preset_id}>{signalName(result.config)}</span> · {formatNumber(a.sample_count)} {t('generator.samplesUnit')} · {a.duration_ms.toPrecision(4)} ms</Typography>
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: 'minmax(0, 1fr)', sm: 'repeat(3, minmax(0, 1fr))' }, gap: 1 }}>
      {[
        ['PAPR', `${a.papr_db.toFixed(2)} dB`],
        [t('generator.occupiedShort'), `${(a.occupied_bandwidth_99_hz / 1e6).toFixed(2)} MHz`],
        [t('generator.rms'), a.rms.toFixed(4)],
      ].map(([label, value]) => <Paper key={label} sx={{ p: 1.5, display: { xs: 'flex', sm: 'block' }, alignItems: 'center', justifyContent: 'space-between', gap: 2 }}><Typography variant="caption" color="text.secondary">{label}</Typography><Typography sx={{ fontSize: { xs: 17, lg: 22 }, fontWeight: 700, fontVariantNumeric: 'tabular-nums', whiteSpace: 'nowrap' }}>{value}</Typography></Paper>)}
    </Box>
    <Tabs value={tab} onChange={(_, value: string) => setTab(value)} variant="scrollable" scrollButtons="auto" aria-label={t('generator.visualizations')}>
      <Tab value="overview" label={t('generator.overview')} />
      <Tab value="allocation" label={t('generator.allocation')} disabled={!allocations.length} />
      <Tab value="evm" label={t('generator.evmTrends')} disabled={!a.evm_per_symbol_percent?.length} />
      <Tab value="metrics" label={t('generator.metrics')} />
    </Tabs>
    {tab === 'overview' && <Box sx={{ display: 'grid', gridTemplateColumns: { xs: 'minmax(0, 1fr)', lg: 'repeat(2, minmax(0, 1fr))' }, gap: 1.5 }}>
      <Paper sx={{ p: 1.5, minWidth: 0 }}><PlotlyChart title={`${t('spectrum.node.pa_input')} · PSD`} height={270} viewKey={result.signal_id} traces={[{ x: a.frequency_mhz, y: a.psd_dbfs_hz, name: 'PSD', line: { color: PALETTE[0], width: 1.5 } }]} layout={{ xaxis: axis(t('generator.frequencyAxis')), yaxis: axis('dBFS / Hz'), showlegend: false }} />
        <Typography variant="caption" color="text.secondary">{t('generator.psdHelp')}</Typography></Paper>
      <Paper sx={{ p: 1.5, minWidth: 0 }}><PlotlyChart title={t('generator.time')} height={270} viewKey={result.signal_id} traces={[
        { x: a.time_us, y: a.time_i, name: 'I', line: { color: PALETTE[0], width: 1 } },
        { x: a.time_us, y: a.time_q, name: 'Q', line: { color: PALETTE[1], width: 1, dash: 'dot' } },
        { x: a.time_us, y: a.time_envelope, name: '|I+jQ|', line: { color: PALETTE[2], width: 1 }, visible: 'legendonly' },
      ]} layout={{ ...legendSpace, xaxis: axis('µs'), yaxis: axis(t('generator.amplitude')) }} /><Typography variant="caption" color="text.secondary">{t('generator.timeHelp', { count: a.time_us.length })}</Typography></Paper>
      <Paper sx={{ p: 1.5, minWidth: 0 }}>{a.constellation_i.length ? <PlotlyChart title={t('generator.constellation')} height={270} viewKey={result.signal_id} traces={[
        { x: a.reference_i, y: a.reference_q, mode: 'markers', name: t('generator.reference'), marker: { size: 7, opacity: .4, color: PALETTE[1], symbol: 'cross' } },
        { x: a.constellation_i, y: a.constellation_q, mode: 'markers', name: t('generator.recovered'), marker: { size: 4, opacity: .65, color: PALETTE[0] } },
      ]} layout={{ ...legendSpace, xaxis: axis('I'), yaxis: { ...axis('Q'), scaleanchor: 'x', scaleratio: 1 } }} /> : <Stack sx={{ minHeight: 270, justifyContent: 'center', p: 2 }}><Typography variant="h3">{t('generator.constellation')}</Typography><Typography color="text.secondary">{t('generator.noConstellation')}</Typography></Stack>}
        <Typography variant="caption" color="text.secondary">{t(['qam', 'psk'].includes(result.config.waveform) ? 'generator.qamConstellationHelp' : 'generator.constellationHelp')}</Typography></Paper>
      <Paper sx={{ p: 1.5, minWidth: 0 }}><PlotlyChart title={t('generator.ccdf')} height={270} viewKey={result.signal_id} traces={[{ x: ccdf.map(p => p.x), y: ccdf.map(p => p.y), name: 'CCDF', mode: 'lines+markers', marker: { size: 2, color: PALETTE[2] }, line: { color: PALETTE[2], width: 2 } }]} layout={{ xaxis: axis(t('generator.ccdfAxis')), yaxis: axis('log₁₀ Pr(P / Pavg > x)'), showlegend: false }} />
        <Typography variant="caption" color="text.secondary">{t('generator.ccdfHelp')}</Typography></Paper>
    </Box>}
    {tab === 'allocation' && <Paper sx={{ p: 2, minWidth: 0 }}><PlotlyChart title={t('generator.allocation')} height={360} viewKey={result.signal_id} traces={allocations} layout={{ xaxis: axis(t(placed ? 'generator.frequencyAxis' : 'generator.binIndex')), yaxis: { ...axis(t(placed ? 'nrtm.carrier' : 'generator.channel')), dtick: 1 }, showlegend: false }} /><Typography variant="body2" color="text.secondary">{t('generator.allocationHelp')}{placed && step > 1 ? ' ' + t('nrtm.allocationThinned', { step }) : ''}</Typography></Paper>}
    {tab === 'evm' && <Stack spacing={1.5}>
      <Alert severity="info">{t('generator.evmTrendHelp')}</Alert>
      <Paper sx={{ p: 2 }}><PlotlyChart title={t('generator.evmSymbols')} height={290} viewKey={result.signal_id} traces={[{ x: (a.evm_per_symbol_percent ?? []).map((_, i) => i), y: a.evm_per_symbol_percent ?? [], mode: 'lines+markers' }]} layout={{ showlegend: false, xaxis: axis(t('generator.symbols')), yaxis: axis('EVM (%)') }} /></Paper>
      <Paper sx={{ p: 2 }}><PlotlyChart title={t('generator.evmCarriers')} height={290} viewKey={result.signal_id} traces={[{ x: place(a.evm_subcarrier_indices ?? []), y: a.evm_per_subcarrier_percent ?? [] }]} layout={{ showlegend: false, xaxis: axis(t(placed ? 'generator.frequencyAxis' : 'generator.binIndex')), yaxis: axis('EVM (%)') }} /></Paper>
    </Stack>}
    {tab === 'metrics' && <Stack spacing={2}>
      <Paper><TableContainer><Table size="small"><TableBody>{metrics.map(([key, value]) => <TableRow key={key}><TableCell component="th">{key}</TableCell><TableCell align="right" sx={{ fontVariantNumeric: 'tabular-nums' }}>{value}</TableCell></TableRow>)}</TableBody></Table></TableContainer></Paper>
      {summary && <Paper><TableContainer role="region" tabIndex={0} aria-label={t('nrtm.carriers')} sx={{ overflowX: 'auto' }} data-testid="nrtm-result-carriers"><Table size="small">
        <TableHead><TableRow>{[t('nrtm.carrier'), t('nrtm.scs'), t('nrtm.bandwidth'), 'N_RB', 'FFT', t('nrtm.cellId'), t('nrtm.centerOffset'), t('nrtm.occupied'), 'EVM (%)'].map(head => <TableCell key={head} sx={{ whiteSpace: 'nowrap' }}>{head}</TableCell>)}</TableRow></TableHead>
        <TableBody>{summary.carriers.map(c => <TableRow key={c.index}>
          <TableCell>{c.index + 1}</TableCell><TableCell>{c.scs_khz}</TableCell><TableCell>{c.bandwidth_mhz}</TableCell><TableCell>{formatNumber(c.nrb)}</TableCell><TableCell>{formatNumber(c.n_fft)}</TableCell>
          <TableCell>{c.n_cell_id}</TableCell><TableCell>{formatNumber(c.center_offset_hz / 1e6, { maximumFractionDigits: 3 })} MHz</TableCell><TableCell>{formatNumber(c.occupied_bandwidth_hz / 1e6, { maximumFractionDigits: 3 })} MHz</TableCell>
          <TableCell>{typeof a.allocation[c.index]?.evm_percent === 'number' ? (a.allocation[c.index]!.evm_percent as number).toPrecision(3) : '—'}</TableCell>
        </TableRow>)}</TableBody>
      </Table></TableContainer></Paper>}
      {nbIot.some(Boolean) && <NbIotDetails entries={nbIot} />}
      <Typography variant="h3">{t('generator.scope')}</Typography>
      <Typography variant="body2">{t(summary ? 'nrtm.scopeHelp' : 'generator.scopeHelp')}</Typography>
      {a.notes.filter(note => !summary?.warnings?.includes(note)).map(note => <Typography key={note} variant="body2" color="text.secondary">{note}</Typography>)}
      <Typography variant="caption" sx={{ overflowWrap: 'anywhere' }}>IQ SHA-256 · {result.iq_sha256}</Typography>
    </Stack>}
  </Stack>
}

const level = (db: number) => `${formatNumber(db, { signDisplay: 'exceptZero', minimumFractionDigits: 2, maximumFractionDigits: 2 })} dB`

/** NR-N-TM: what the generator did to each host carrier, read from `analysis.allocation[i].nb_iot` (nothing is recomputed here). */
function NbIotDetails({ entries }: { entries: Array<NbIotAllocation | null> }) {
  return <Paper component="section" aria-label={t('nrtm.nbiot')} sx={{ p: 1.5 }} data-testid="nrtm-result-nbiot"><Stack spacing={2}>
    <Typography variant="h3">{t('nrtm.nbiot')}</Typography>
    {entries.map((nb, index) => nb && <Stack key={index} spacing={.75} component="section" aria-label={t('nrtm.carrierN', { index: index + 1 })} data-testid={`nrtm-result-nbiot-${index}`}>
      <Typography variant="subtitle2">{t('nrtm.nbiotHost', { index: index + 1, nid: nb.n_id, host: nb.host_cell_id })}</Typography>
      <Facts items={[
        [t('nrtm.nbiotBoost'), formatNumber(nb.boost_db, { maximumFractionDigits: 2 })],
        [t('nrtm.nbiotLevel'), level(nb.nb_iot_gain_db)],
        [t('nrtm.nrLevel'), level(nb.nr_gain_db)],
        [t('nrtm.nbiotRePerFrame'), t('nrtm.nbiotReSplit', { nr: formatNumber(nb.nr_re), nb: formatNumber(nb.nb_iot_re), frame: formatNumber(nb.frame_re) })],
      ]} />
      <Scroll label={`${t('nrtm.carrierN', { index: index + 1 })} · ${t('nrtm.nbiot')}`}><Table size="small">
        <TableHead><TableRow>{[t('nrtm.nbiotEdge'), t('nrtm.nbiotRb'), t('nrtm.nbiotFirstSubcarrier'), t('nrtm.nbiotRePerFrameShort'), t('nrtm.nbiotRePerSubframe')].map(head => <TableCell key={head} sx={{ whiteSpace: 'nowrap' }}>{head}</TableCell>)}</TableRow></TableHead>
        <TableBody>{nb.carriers.map(c => <TableRow key={c.side}>
          <TableCell>{t(`nrtm.nbiotEdge.${c.side}`)}</TableCell><TableCell>{c.rb}</TableCell><TableCell>{c.first_subcarrier}</TableCell><TableCell>{formatNumber(c.re_per_frame)}</TableCell>
          <TableCell sx={{ fontVariantNumeric: 'tabular-nums', minWidth: 260 }}>{c.re_per_subframe.join(', ')}</TableCell>
        </TableRow>)}</TableBody>
      </Table></Scroll>
    </Stack>)}
  </Stack></Paper>
}
