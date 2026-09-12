import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import IconButton from '@mui/material/IconButton'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import Table from '@mui/material/Table'
import TableBody from '@mui/material/TableBody'
import TableCell from '@mui/material/TableCell'
import TableHead from '@mui/material/TableHead'
import TableRow from '@mui/material/TableRow'
import ToggleButton from '@mui/material/ToggleButton'
import ToggleButtonGroup from '@mui/material/ToggleButtonGroup'
import Tooltip from '@mui/material/Tooltip'
import Typography from '@mui/material/Typography'
import useMediaQuery from '@mui/material/useMediaQuery'
import { useMemo, useState } from 'react'
import type { DatasetAnalysis, InspectionReading } from '@/api/datasets'
import { formatNumber, message, t, type MessageKey } from '@/i18n'
import { tokens, useStudioColors } from '@/theme'
import { IQPreview } from './IQPreview'
import { PlotlyChart, seriesSymbol, type PlotLayout, type PlotTrace } from './PlotlyChart'
import { SpectrumPlot } from './SpectrumPlot'

const LABELS: Record<string, MessageKey> = {
  PAPR: 'inspection.metric.papr', RMS: 'inspection.metric.rms', PEAK: 'inspection.metric.peak',
  ACPR_L: 'inspection.metric.acprL', ACPR_R: 'inspection.metric.acprR', DELAY: 'inspection.metric.delay',
  GAIN: 'inspection.metric.gain', PHASE: 'inspection.metric.phase', BLA_NMSE: 'inspection.metric.bla',
  EVM_RMS: 'inspection.metric.evm', ACLR_L: 'inspection.metric.aclrL', ACLR_R: 'inspection.metric.aclrR',
}

function Reading({ reading }: { reading?: InspectionReading | null }) {
  if (!reading) return <Box component="span" sx={{ color: 'text.secondary' }}>—</Box>
  if (reading.status === 'ok' && reading.value != null) return <>{formatNumber(reading.value, { maximumFractionDigits: 2, minimumFractionDigits: 2 })}</>
  return <Tooltip title={message(reading.reason) || t('common.na')}><Box component="span" tabIndex={0} aria-label={message(reading.reason) || t('common.na')} sx={{ color: reading.status === 'invalid' ? 'error.main' : 'text.secondary', borderBottom: '1px dotted', cursor: 'help', fontSize: 12 }}>
    {reading.status === 'review_required' ? t('inspection.review') : t('common.na')}
  </Box></Tooltip>
}

function MeasurementTable({ data }: { data: DatasetAnalysis }) {
  return (
    <Paper component="section" aria-label={t('inspection.measurements')} sx={{ minWidth: 0, overflow: 'hidden', alignSelf: 'stretch' }}>
      <Stack direction="row" sx={{ px: 1.5, py: .75, alignItems: 'center', justifyContent: 'space-between' }}>
        <Typography variant="h3" component="h2">{t('inspection.measurements')}</Typography>
        <Tooltip title={(data.notes ?? []).map(message).join(' ')}><IconButton size="small" aria-label={t('inspection.conditions')}><InfoOutlinedIcon fontSize="small" /></IconButton></Tooltip>
      </Stack>
      <Table size="small" aria-label={t('inspection.measurements')} sx={{ '& .MuiTableCell-root': { px: 1.25, py: .55, fontSize: 13 }, '& .MuiTableCell-head': { fontSize: 12 } }}>
        <TableHead><TableRow>
          <TableCell>{t('inspection.metric')}</TableCell>
          <TableCell align="right">{t('inspection.input')}</TableCell>
          <TableCell align="right">{t('inspection.output')}</TableCell>
        </TableRow></TableHead>
        <TableBody>{data.measurements.map((m) => (
          <TableRow key={m.name}>
            <TableCell component="th" scope="row">
              <Tooltip describeChild title={message(m.method)}><Box component="span" tabIndex={0} sx={{ cursor: 'help' }}>{LABELS[m.name] ? t(LABELS[m.name]!) : m.label}</Box></Tooltip>
              <Box component="span" sx={{ color: 'text.secondary', ml: .5, fontSize: 12 }}> {m.unit === 'unknown' || m.unit === 'normalized' ? t('inspection.au') : message(m.unit)}</Box>
            </TableCell>
            <TableCell align="right" sx={{ fontFamily: tokens.typography.monoFamily, color: 'info.main' }}><Reading reading={m.input} /></TableCell>
            <TableCell align="right" sx={{ fontFamily: tokens.typography.monoFamily }}><Reading reading={m.output} /></TableCell>
          </TableRow>
        ))}</TableBody>
      </Table>
      <Typography variant="caption" component="p" color="text.secondary" sx={{ px: 1.5, mb: 0, mt: 1 }}>{t('inspection.measurementNote')}</Typography>
    </Paper>
  )
}

/** A first-viewport instrument layout; all signal processing stays in the service. */
export function SignalInspection({ data }: { data: DatasetAnalysis }) {
  const colors = useStudioColors()
  const [amMode, setAmMode] = useState<'am' | 'pm'>('am')
  const [iqMode, setIqMode] = useState<'symbols' | 'samples'>('symbols')
  const hasSymbols = data.constellation?.status === 'ok'
  const showSymbols = hasSymbols && iqMode === 'symbols'
  const tallViewport = useMediaQuery('(min-height: 1000px)')
  const spectrumTraces = useMemo(() => (data.spectrum?.traces ?? []).map((tr) => ({ name: tr.name, psdDb: tr.psd_db })), [data.spectrum])
  const inputLabel = t('inspection.input'), outputLabel = t('inspection.output'), equalizedLabel = t('inspection.outputEq')
  const iqTraces = useMemo<PlotTrace[]>(() => (showSymbols ? data.constellation?.traces ?? [] : data.iq?.traces ?? []).map((tr, i) => ({
    x: tr.i, y: tr.q, name: tr.role === 'input' ? inputLabel : showSymbols ? equalizedLabel : outputLabel, type: 'scatter', mode: 'markers',
    marker: { size: 2.5, opacity: .4, symbol: seriesSymbol(i) },
  })), [data.iq, data.constellation, showSymbols, inputLabel, outputLabel, equalizedLabel])
  const amTraces = useMemo<PlotTrace[]>(() => (data.am?.traces ?? []).map((tr) => ({
    x: data.am!.amp_in, y: amMode === 'am' ? tr.amp_out : tr.phase_deg, name: tr.name, type: 'scatter', mode: 'markers',
    marker: { size: 2.5, opacity: .4, color: colors.chart[1] },
  })), [data.am, amMode, colors])
  const iqLayout = useMemo<PlotLayout>(() => ({ xaxis: { title: { text: 'I' }, constrain: 'domain' }, yaxis: { title: { text: 'Q' }, scaleanchor: 'x', scaleratio: 1 }, showlegend: true }), [])
  const amXTitle = t('chart.am.x'), amYTitle = t(amMode === 'am' ? 'chart.am.y' : 'chart.pm.y')
  const amLayout = useMemo<PlotLayout>(() => ({ xaxis: { title: { text: amXTitle } }, yaxis: { title: { text: amYTitle } }, showlegend: true }), [amXTitle, amYTitle])
  const height = tallViewport ? 276 : 174
  const panel = { p: 1.25, minWidth: 0, overflow: 'hidden' }
  const constellationNote = showSymbols
    ? `${t('inspection.constellationNote')} ${data.constellation!.demodulator}; ${t('inspection.symbolWindow')}: ${data.constellation!.sample_range?.join('–')}. ${message(data.constellation!.note)}`
    : `${t('inspection.iqNote')} ${hasSymbols ? '' : message(data.constellation?.reason)}`
  const iqTitle = showSymbols ? [data.constellation?.modulation, t('inspection.constellation')].filter(Boolean).join(' · ') : t('inspection.iq')
  const viewKey = `${data.dataset_id}/${data.data_version}/${data.sample_range.join(':')}`
  return (
    <Box data-testid="signal-workbench" sx={{ display: 'grid', gridTemplateColumns: { xs: 'minmax(0, 1fr)', lg: 'minmax(0, 1fr) 330px', xl: 'minmax(0, 1fr) 360px' }, gap: 1.5 }}>
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: 'minmax(0, 1fr)', md: 'repeat(2, minmax(0, 1fr))' }, gap: 1.5, alignContent: 'start' }}>
        {!data.spectrum && <Alert severity="warning" sx={{ gridColumn: '1 / -1' }}>{t('inspection.noPlots')}</Alert>}
        {data.spectrum && <Paper sx={panel}>
          <SpectrumPlot frequencyHz={data.spectrum.frequency} axis={data.spectrum.axis} traces={spectrumTraces} bands={data.spectrum.bands ?? undefined} height={height} title={t('inspection.frequency')} viewKey={viewKey} />
        </Paper>}
        {data.time && <Paper sx={panel}>
          <IQPreview start={data.time.start} series={data.time.traces} height={height} title={t('inspection.time')} viewKey={viewKey} />
        </Paper>}
        {(data.iq || hasSymbols) && <Paper sx={panel}>
          <PlotlyChart title={iqTitle} traces={iqTraces} layout={iqLayout} height={height} viewKey={`${viewKey}/${showSymbols ? 'symbols' : 'samples'}`} data-testid="constellation-plot" actions={<Box sx={{ display: 'flex', alignItems: 'center', gap: .5, flexShrink: 0 }}>
            <Tooltip title={constellationNote}><InfoOutlinedIcon titleAccess={constellationNote} tabIndex={0} sx={{ fontSize: 16, color: 'text.secondary' }} /></Tooltip>
            {hasSymbols && <ToggleButtonGroup size="small" exclusive value={showSymbols ? 'symbols' : 'samples'} onChange={(_e, v: 'symbols' | 'samples' | null) => v && setIqMode(v)} aria-label={t('inspection.iqMode')} sx={{ '& button': { fontSize: 11, px: .75, py: .25 } }}>
              <ToggleButton value="symbols">{t('inspection.symbols')}</ToggleButton><ToggleButton value="samples">{t('inspection.samples')}</ToggleButton>
            </ToggleButtonGroup>}
          </Box>} />
        </Paper>}
        {data.am && <Paper sx={panel}>
          <PlotlyChart title={t('inspection.transfer')} traces={amTraces} layout={amLayout} height={height} viewKey={`${viewKey}/${amMode}`} data-testid="dataset-am-plot" actions={
            <ToggleButtonGroup size="small" exclusive value={amMode} onChange={(_e, v: 'am' | 'pm' | null) => v && setAmMode(v)} aria-label={t('inspection.amMode')} sx={{ '& button': { fontSize: 11, px: .75, py: .25 } }}>
              <ToggleButton value="am" aria-label="AM/AM">AM/AM</ToggleButton><ToggleButton value="pm" aria-label="AM/PM">AM/PM</ToggleButton>
            </ToggleButtonGroup>
          } />
        </Paper>}
      </Box>
      <MeasurementTable data={data} />
    </Box>
  )
}
