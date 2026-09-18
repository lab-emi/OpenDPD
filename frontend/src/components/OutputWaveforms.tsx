import { Box, Stack, TextField, Typography, useMediaQuery } from '@mui/material'
import { useMemo, useState } from 'react'
import { t, useLanguage } from '@/i18n'
import { useStudioColors } from '@/theme'
import { PlotlyChart, type PlotLayout, type PlotTrace } from './PlotlyChart'
import { spectrumLegend, type SignalIdentity } from './spectrumNodes'
import type { IQSeries } from './IQPreview'

export type OutputWaveform = IQSeries & SignalIdentity

/** Roles are authoritative; exact names keep old plots-v1 artifacts readable. */
function outputPair(series: OutputWaveform[]) {
  const reference = series.find(s => s.role === 'reference') ?? series.find(s => !s.role && /^(measured pa output|linear target(?: gain\*x)?)$/i.test(s.name))
  const output = series.find(s => s.role === 'primary') ?? series.find(s => !s.role && /^(pa model output|DPD → PA surrogate|with DPD: PA_surrogate\(u\))$/i.test(s.name))
  return reference && output && reference !== output ? [reference, output] : undefined
}

/** A focused comparison, never an overlay of every signal and both components.
 * All server samples remain available to pan/zoom; the initial window is short.
 */
export function OutputWaveforms({ start, series, viewKey = '' }: { start: number; series: OutputWaveform[]; viewKey?: string }) {
  useLanguage()
  const colors = useStudioColors()
  const narrow = useMediaQuery('(max-width: 600px)')
  const [selection, setSelection] = useState('comparison')
  const [window, setWindow] = useState('detail')
  const pair = useMemo(() => outputPair(series), [series])
  const fallback = pair ? 'comparison' : `signal:${series[0]?.name ?? ''}`
  const selected = selection === 'comparison' ? fallback : series.some(s => `signal:${s.name}` === selection) ? selection : fallback
  const drawn = useMemo(() => selected === 'comparison' ? pair! : series.filter(s => `signal:${s.name}` === selected), [selected, pair, series])
  const count = drawn.length ? Math.min(...drawn.flatMap(s => [s.i.length, s.q.length])) : 0
  const shown = window === 'detail' ? Math.min(128, count) : count
  const labels = JSON.stringify(drawn.map(s => narrow ? spectrumLegend(s).replace(' · ', ' · <br>') : spectrumLegend(s)))
  const data = useMemo(() => {
    const names = JSON.parse(labels) as string[]
    return (['i', 'q'] as const).map(component => drawn.map((s, index): PlotTrace => ({
      x: Float64Array.from({ length: s[component].length }, (_, k) => start + k),
      y: s[component], name: names[index], type: 'scatter', mode: 'lines',
      line: { width: 1.8, color: colors.chart[selected === 'comparison' && index === 0 ? 1 : 0], dash: selected === 'comparison' && index === 0 ? 'dash' : 'solid' },
    })))
  }, [drawn, labels, selected, start, colors])
  const amplitude = useMemo(() => {
    let peak = 0
    for (const s of drawn) for (const component of [s.i, s.q]) for (let k = 0; k < component.length; k++) {
      if (Number.isFinite(component[k])) peak = Math.max(peak, Math.abs(component[k]!))
    }
    return peak ? peak * 1.08 : 1
  }, [drawn])
  const xTitle = t('chart.iq.x'), yTitle = t('chart.iq.y')
  const layout = useMemo<PlotLayout>(() => ({
    xaxis: { title: { text: xTitle, font: { size: 14 } }, tickfont: { size: 13 }, range: [start, start + Math.max(1, shown - 1)] },
    yaxis: { title: { text: yTitle, font: { size: 14 } }, tickfont: { size: 13 }, range: [-amplitude, amplitude] },
    margin: { l: 62, r: 16, t: narrow ? 122 : 104, b: 56 },
    legend: { orientation: 'h', x: 0, y: 1.02, yanchor: 'bottom', maxheight: narrow ? 80 : 56, font: { size: 14 } },
    showlegend: true,
  }), [xTitle, yTitle, start, shown, amplitude, narrow])
  if (!series.length) return null
  return <Box data-testid="output-waveforms">
    <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2} sx={{ mb: 1.5 }}>
      <TextField select size="small" label={t('waveform.view')} value={selected} onChange={event => setSelection(event.target.value)} slotProps={{ select: { native: true } }} sx={{ minWidth: 0, width: { xs: '100%', sm: 300 } }}>
        {pair && <option value="comparison">{t('live.signals')}</option>}
        <optgroup label={t('waveform.individual')}>
          {series.map(s => <option key={s.name} value={`signal:${s.name}`}>{spectrumLegend(s)}</option>)}
        </optgroup>
      </TextField>
      {count > 128 && <TextField select size="small" label={t('waveform.window')} value={window} onChange={event => setWindow(event.target.value)} slotProps={{ select: { native: true } }} sx={{ minWidth: 0, width: { xs: '100%', sm: 225 } }}>
        <option value="detail">{t('waveform.detail')}</option>
        <option value="full">{t('waveform.full', { count })}</option>
      </TextField>}
    </Stack>
    <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>{t('waveform.help')}</Typography>
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: 'minmax(0, 1fr)', md: 'repeat(2, minmax(0, 1fr))' }, gap: 2 }}>
      {(['i', 'q'] as const).map((component, index) => <Box key={component} sx={{ minWidth: 0 }}>
        <PlotlyChart title={t(component === 'i' ? 'waveform.inPhase' : 'waveform.quadrature')} traces={data[index]!} layout={layout} height={narrow ? 378 : 360}
          viewKey={`${viewKey}:${start}:${selected}:${window}:${component}`} data-testid={`output-waveform-${component}`} />
      </Box>)}
    </Box>
  </Box>
}
