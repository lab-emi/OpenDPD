import { getQuery } from '@/api/client'
import { useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { Alert, Box, Button, Checkbox, Dialog, DialogActions, DialogContent, DialogTitle, FormControlLabel, MenuItem, Paper, Stack, TextField, Typography } from '@mui/material'
import { API, api } from '@/api/client'
import type { components } from '@/api/schema'
import type { FigureSpec, SavedFigure } from '@/api/types'
import { t, type MessageKey } from '@/i18n'
import { DownloadLink } from './DownloadLink'
import { ErrorState } from './StateBlock'
import { PlotlyChart, seriesDash, seriesSymbol, type PlotTrace } from './PlotlyChart'
import { SpectrumPanels } from './SpectrumPanels'
import { hasDPD, nodeTitle, spectrumLegend } from './spectrumNodes'
import type { PlotViewport } from './plotInteractions'

type Sources = components['schemas']['FigureSources']
type Preview = components['schemas']['FigurePreview']
type Panel = FigureSpec['panels'][number]
type Kind = NonNullable<Panel['kind']>
type RawTrace = { name: string; role: string; source?: string; stage?: string; signal_node?: string; psd_db?: number[]; amp_out?: number[]; phase_deg?: number[]; y?: number[]; y_unit?: string; y_label?: string }
type RawPlot = { axis?: string; frequency?: number[]; amp_in?: number[]; x?: number[]; x_label?: string; x_unit?: string; y_label?: string; y_unit?: string; traces: RawTrace[]; note?: string }
const kinds: Kind[] = ['spectrum', 'amam', 'ampm', 'power_scan', 'error_distribution']
const label = (key: string) => t(`publication.${key}` as MessageKey)
const palette = ['#2563EB', '#B45309', '#047857', '#9333EA', '#BE185D', '#0E7490']
const identity = (value: unknown) => JSON.stringify(value)

export function PublicationFigureDialog({ initial, onClose }: { initial: FigureSpec; onClose: () => void }) {
  const qc = useQueryClient()
  const [spec, setSpec] = useState<FigureSpec>({ ...initial, version: 'figure-v2' })
  const [nextKind, setNextKind] = useState<Kind>('error_distribution')
  const [saved, setSaved] = useState<SavedFigure>()
  const [preview, setPreview] = useState<Preview>()
  const viewports = useRef<Record<number, PlotViewport>>({})
  const ids = Object.keys(spec.profiles)
  const source = useQuery({ queryKey: ['figure-sources', spec.profiles], queryFn: () => api.post<Sources>('/figure-sources', { profiles: spec.profiles }) })
  const figures = useQuery({ queryKey: ['figures', ...ids], queryFn: getQuery<SavedFigure[]>(`/figures?${ids.map(id => `runs=${encodeURIComponent(id)}`).join('&')}`) })
  const offered = figures.data?.filter(f => f.bindings.every(b => b.review.profile.validation !== 'pending_cross_validation')) ?? []
  const change = (next: FigureSpec) => { setSpec(next); setSaved(undefined) }
  const updatePanel = (index: number, patch: Partial<Panel>) => change({ ...spec, panels: spec.panels.map((p, i) => i === index ? { ...p, ...patch } : p) })
  const inspect = useMutation({ mutationFn: () => api.post<Preview>('/figure-preview', spec), onSuccess: value => { setSpec(value.figure.spec); setPreview(value); viewports.current = {} } })
  const save = useMutation({ mutationFn: () => api.post<SavedFigure>('/figures', spec), onSuccess: async value => { setSaved(value); await qc.invalidateQueries({ queryKey: ['figures'] }) } })
  const restore = useMutation({ mutationFn: (id: string) => api.get<SavedFigure>(`/figures/${id}`), onSuccess: async value => { const next = await api.post<Preview>('/figure-preview', value.spec); setSpec(next.figure.spec); setSaved(identity(next.figure.spec) === identity(value.spec) ? value : undefined); setPreview(next); viewports.current = {} } })
  const ready = preview && identity(preview.figure.spec) === identity(spec)
  const add = () => {
    let choices = source.data?.sources.filter(s => s.kind === nextKind && s.role === 'primary') ?? []
    if (nextKind === 'power_scan') {
      const metric = choices.find(s => s.trace_name === 'ACPR_L')?.trace_name ?? choices[0]?.trace_name
      choices = choices.filter(s => s.trace_name === metric)
    }
    const panel: Panel = { kind: nextKind, show_bands: nextKind === 'spectrum', traces: choices.map((s, i) => ({ run_id: s.run_id, trace_name: s.trace_name, visible: true, color: palette[i % palette.length]!, dash: seriesDash(i) })) }
    change({ ...spec, version: 'figure-v2', panels: [...spec.panels, panel] })
  }
  const ranges = (index: number, v: PlotViewport) => {
    const previous = viewports.current[index]
    viewports.current[index] = v
    if (!previous || identity(previous) === identity(v)) return
    const next = { ...spec, panels: spec.panels.map((p, i) => i === index ? { ...p, x_range: v.autoX ? null : v.x, y_range: v.autoY ? null : v.y } : p) }
    change(next)
    if (ready) setPreview({ ...preview, figure: { ...preview.figure, spec: next } })
  }
  const visibility = (index: number, visible: boolean[]) => {
    const next = { ...spec, panels: spec.panels.map((p, i) => i === index ? { ...p, traces: p.traces.map((tr, j) => ({ ...tr, visible: visible[j] ?? true })) } : p) }
    change(next)
    if (ready) setPreview({ ...preview, figure: { ...preview.figure, spec: next } })
  }
  return <Dialog open onClose={onClose} fullWidth maxWidth="lg" aria-labelledby="publication-title"><DialogTitle id="publication-title">{label('title')}</DialogTitle><DialogContent dividers><Stack spacing={2}>
    <Alert severity="info">{label('intro')}</Alert>
    {preview?.figure.bindings.some(b => b.review.facts.some(f => f.key === 'dataset_origin' && f.value === 'synthetic')) && <Alert severity="warning">{t('datasetResearch.syntheticNotice')}</Alert>}
    <Stack direction={{ xs: 'column', md: 'row' }} spacing={1}>
      <TextField size="small" label={t('review.viewTitle')} value={spec.title} sx={{ flex: 1 }} onChange={e => change({ ...spec, title: e.target.value })} />
      <TextField size="small" select label={t('review.saved')} value={saved?.figure_id ?? ''} sx={{ minWidth: 200, maxWidth: 350 }} onChange={e => { if (e.target.value) restore.mutate(e.target.value) }}><MenuItem value="">{t('review.current')}</MenuItem>{offered.map(f => <MenuItem key={f.figure_id} value={f.figure_id}>{f.spec.title}</MenuItem>)}</TextField>
      <TextField size="small" select label={t('review.width')} value={spec.width ?? 'double_column'} onChange={e => change({ ...spec, width: e.target.value as FigureSpec['width'] })}><MenuItem value="single_column">{t('review.single')}</MenuItem><MenuItem value="double_column">{t('review.double')}</MenuItem></TextField>
    </Stack>
    <Typography variant="caption" sx={{ overflowWrap: 'anywhere' }}>{label('bound')}: {Object.entries(spec.profiles).map(([run, p]) => `${run} / ${p}`).join(' · ')} · {label('reference')}: {spec.reference_run_id} · {spec.mode}</Typography>
    {spec.panels.map((panel, index) => <Paper variant="outlined" key={index} sx={{ p: 1.5 }}><Stack spacing={1}>
      <Stack direction="row" sx={{ alignItems: 'center', justifyContent: 'space-between' }}><Typography variant="h3">{index + 1}. {panel.signal_node ? `${nodeTitle(panel.signal_node)} · PSD` : label(panel.kind ?? 'spectrum')}</Typography><Button disabled={spec.panels.length === 1} onClick={() => change({ ...spec, panels: spec.panels.filter((_p, i) => i !== index) })}>{label('remove')}</Button></Stack>
      <Box component="details"><Typography component="summary">{t('review.trace')}</Typography><Stack>
        {source.data?.sources.filter(s => s.kind === panel.kind && (!panel.signal_node || s.signal_node === panel.signal_node) && (panel.kind !== 'power_scan' || s.trace_name === panel.traces[0]?.trace_name)).map((s, i) => {
          const trace = panel.traces.find(tr => tr.run_id === s.run_id && tr.trace_name === s.trace_name)
          return <Stack direction="row" key={`${s.run_id}:${s.trace_name}`} spacing={1} sx={{ alignItems: 'center', flexWrap: 'wrap' }}>
            <FormControlLabel sx={{ flex: 1, minWidth: 230 }} label={`${s.run_id} · ${s.trace_name} · ${s.source}`} control={<Checkbox checked={!!trace && trace.visible !== false} onChange={e => updatePanel(index, { traces: trace ? panel.traces.map(tr => tr === trace ? { ...tr, visible: e.target.checked } : tr) : [...panel.traces, { run_id: s.run_id, trace_name: s.trace_name, visible: true, color: palette[i % palette.length]!, dash: seriesDash(i) }] })} />} />
            {trace && <><TextField size="small" type="color" label={label('color')} value={trace.color} sx={{ width: 90 }} onChange={e => updatePanel(index, { traces: panel.traces.map(tr => tr === trace ? { ...tr, color: e.target.value } : tr) })} /><TextField size="small" select label={label('dash')} value={trace.dash ?? 'solid'} sx={{ width: 125 }} onChange={e => updatePanel(index, { traces: panel.traces.map(tr => tr === trace ? { ...tr, dash: e.target.value as typeof tr.dash } : tr) })}>{['solid', 'dash', 'dot', 'dashdot', 'longdash'].map(d => <MenuItem key={d} value={d}>{d}</MenuItem>)}</TextField></>}
          </Stack>
        })}
      </Stack></Box>
      {panel.kind === 'power_scan' && <TextField size="small" select label={label('powerMetric')} value={panel.traces[0]?.trace_name ?? ''} onChange={e => updatePanel(index, { traces: (source.data?.sources.filter(s => s.kind === 'power_scan' && s.trace_name === e.target.value) ?? []).map((s, i) => ({ run_id: s.run_id, trace_name: s.trace_name, visible: true, color: palette[i % palette.length]!, dash: seriesDash(i) })) })}>{[...new Set(source.data?.sources.filter(s => s.kind === 'power_scan').map(s => s.trace_name))].map(name => <MenuItem key={name} value={name}>{name}</MenuItem>)}</TextField>}
      {panel.kind === 'spectrum' && <FormControlLabel control={<Checkbox checked={panel.show_bands ?? true} onChange={e => updatePanel(index, { show_bands: e.target.checked })} />} label={t('review.bands')} />}
      {ready && <PanelPreview index={index} value={preview} onViewport={v => ranges(index, v)} onVisibility={v => visibility(index, v)} />}
    </Stack></Paper>)}
    <Stack direction="row" spacing={1}><TextField size="small" select label={label('kind')} value={nextKind} sx={{ minWidth: 220 }} onChange={e => setNextKind(e.target.value as Kind)}>{kinds.map(k => <MenuItem key={k} value={k}>{label(k)}</MenuItem>)}</TextField><Button variant="outlined" disabled={spec.panels.length >= 4 || !source.data?.sources.some(s => s.kind === nextKind && s.role === 'primary')} onClick={add}>{label('add')}</Button></Stack>
    {source.data?.missing.length ? <Box component="details"><Typography component="summary">{label('missing')}</Typography><ul>{source.data.missing.map(m => <li key={m}>{m}</li>)}</ul></Box> : null}
    {[source, inspect, save, restore].filter(q => q.isError).map((q, i) => <ErrorState key={i} error={q.error} />)}
    {saved && <Stack spacing={1}><DownloadLink button href={`${API}/figures/${saved.figure_id}/export`} download>{t('review.export')}</DownloadLink><Alert severity="info">{label('private')}</Alert><DownloadLink button href={`${API}/figures/${saved.figure_id}/reproduction`} download>{label('reproduce')}</DownloadLink></Stack>}
  </Stack></DialogContent><DialogActions><Button onClick={onClose}>{t('common.back')}</Button><Button variant="outlined" disabled={inspect.isPending || !spec.title.trim()} onClick={() => inspect.mutate()}>{label('preview')}</Button><Button variant="contained" disabled={!ready || save.isPending} onClick={() => save.mutate()}>{t('review.save')}</Button></DialogActions></Dialog>
}

function PanelPreview({ index, value, onViewport, onVisibility }: { index: number; value: Preview; onViewport: (v: PlotViewport) => void; onVisibility: (visible: boolean[]) => void }) {
  const panel = value.figure.spec.panels[index]!
  const kind = panel.kind ?? 'spectrum'
  const series = panel.traces.map(trace => {
    const plot = value.plots[`${trace.run_id}/${kind === 'ampm' ? 'amam' : kind}`] as unknown as RawPlot
    const row = plot.traces.find(r => r.name === trace.trace_name)!
    const spectrum = kind === 'spectrum', am = kind === 'amam' || kind === 'ampm'
    const x = spectrum ? plot.frequency!.map(v => plot.axis === 'hz' ? v / 1e6 : v) : am ? plot.amp_in! : plot.x!
    const y = spectrum ? row.psd_db! : kind === 'amam' ? row.amp_out! : kind === 'ampm' ? row.phase_deg! : row.y!
    const xlabel = spectrum ? `Frequency (${plot.axis === 'hz' ? 'MHz' : 'cycles/sample'})` : am ? 'Input amplitude (stored units)' : `${plot.x_label} (${plot.x_unit})`
    const ylabel = spectrum ? `PSD (dB re amplitude²/${plot.axis === 'hz' ? 'Hz' : '(cycles/sample)'})` : am ? kind === 'amam' ? 'Output amplitude (stored units)' : 'Phase difference (degrees)' : `${row.y_label ?? plot.y_label} (${row.y_unit ?? plot.y_unit})`
    const result = value.figure.bindings.find(b => b.run_id === trace.run_id)?.review.result
    const synthetic = value.figure.bindings.find(b => b.run_id === trace.run_id)?.review.facts.some(f => f.key === 'dataset_origin' && f.value === 'synthetic')
    const plotTrace: PlotTrace = { x, y, name: `${trace.run_id} · ${trace.trace_name} [${synthetic ? 'SYNTHETIC / ' : ''}${result?.is_mock ? 'MOCK' : row.source ?? result?.evidence_type ?? row.role}]`, mode: spectrum || kind === 'error_distribution' ? 'lines' : 'markers', visible: trace.visible === false ? 'legendonly' : true, line: { color: trace.color, dash: trace.dash, width: 1.2 }, marker: { color: trace.color, size: am ? 3 : 9, symbol: seriesSymbol(['solid', 'dash', 'dot', 'dashdot', 'longdash'].indexOf(trace.dash ?? 'solid')) } }
    return { plotTrace, xlabel, ylabel }
  })
  const traces = series.map(s => s.plotTrace)
  const { xlabel = '', ylabel = '' } = series[0] ?? {}
  const context = value.figure.bindings.find(b => b.run_id === value.figure.spec.reference_run_id)?.review
  const bands = kind === 'spectrum' && panel.show_bands ? context?.bands?.filter(b => b.role !== 'subchannel').map(b => ({ type: 'rect' as const, x0: b.edges_hz[0]! / 1e6, x1: b.edges_hz[1]! / 1e6, y0: 0, y1: 1, yref: 'paper' as const, fillcolor: b.role === 'main' ? '#2563EB11' : '#D9770611', line: { width: 0 } })) : []
  const cursor = panel.cursor_x == null ? [] : [{ type: 'line' as const, x0: panel.cursor_x, x1: panel.cursor_x, y0: 0, y1: 1, yref: 'paper' as const, line: { width: 1, color: '#64748B', dash: 'dot' as const } }]
  if (kind === 'spectrum') {
    const firstPlot = value.plots[`${panel.traces[0]!.run_id}/spectrum`] as unknown as RawPlot
    const main = context?.bands?.find(b => b.role === 'main')
    return <Box data-testid={`publication-panel-${index}`}><SpectrumPanels
      frequencyHz={firstPlot.frequency!} axis={firstPlot.axis as 'hz' | 'normalized'}
      dpd={hasDPD(firstPlot.traces)} height={340} viewKey={`${index}:${kind}`}
      xRange={panel.x_range as [number, number] | undefined} yRange={panel.y_range as [number, number] | undefined}
      cursorX={panel.cursor_x ?? undefined} onViewportChange={onViewport} onVisibilityChange={onVisibility}
      bands={panel.show_bands && main ? { main: main.edges_hz as [number, number], adjacent: (context?.bands ?? []).filter(b => b.role === 'adjacent').map(b => b.edges_hz as [number, number]) } : undefined}
      traces={panel.traces.map(tr => {
        const plot = value.plots[`${tr.run_id}/spectrum`] as unknown as RawPlot
        const raw = plot.traces.find(r => r.name === tr.trace_name)!
        const binding = value.figure.bindings.find(b => b.run_id === tr.run_id)
        const synthetic = binding?.review.facts.some(f => f.key === 'dataset_origin' && f.value === 'synthetic')
        return { ...raw, signal_node: panel.signal_node ?? raw.signal_node, frequencyHz: plot.frequency, psdDb: raw.psd_db!, color: tr.color, dash: tr.dash, visible: tr.visible,
          legendName: `R${value.figure.bindings.findIndex(b => b.run_id === tr.run_id) + 1} · ${spectrumLegend(raw)}${binding?.review.result.is_mock ? ' · MOCK' : synthetic && !raw.source?.includes('synthetic') ? ' · SYNTHETIC' : ''}` }
      })} /></Box>
  }
  return <PlotlyChart title={label(kind)} data-testid={`publication-panel-${index}`} traces={traces} viewKey={`${index}:${kind}`} onViewportChange={onViewport} onVisibilityChange={onVisibility} layout={{ xaxis: { title: { text: xlabel }, ...(panel.x_range ? { range: panel.x_range as [number, number] } : {}) }, yaxis: { title: { text: ylabel }, ...(panel.y_range ? { range: panel.y_range as [number, number] } : {}) }, shapes: [...(bands ?? []), ...cursor], height: 340 }} />
}
