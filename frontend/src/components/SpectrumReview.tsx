import { Accordion, AccordionDetails, AccordionSummary, Alert, Box, Button, Checkbox, FormControlLabel, MenuItem, Paper, Stack, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, TextField, Typography } from '@mui/material'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import { useMutation, useQueries, useQuery, useQueryClient } from '@tanstack/react-query'
import { useMemo, useRef, useState } from 'react'
import { API, api, WEB_MODE } from '@/api/client'
import { artifactJsonQuery, reviewQuery } from '@/api/hooks'
import type { EvaluationResult, FigureSpec, FigureTrace, SavedFigure } from '@/api/types'
import { t } from '@/i18n'
import { useStudioColors } from '@/theme'
import { DownloadLink } from './DownloadLink'
import type { PlotViewport } from './plotInteractions'
import type { SpectrumData } from './ResultCharts'
import { SpectrumPlot } from './SpectrumPlot'
import { PublicationFigureDialog } from './PublicationFigureDialog'

/** A cursor reads the actual nearest saved bin, with a binary search on its own frequency axis. */
export function nearestBin(values: number[], requested: number): number {
  if (!values.length || requested < values[0]! || requested > values[values.length - 1]!) return -1
  let lo = 0, hi = values.length - 1
  while (lo < hi) { const mid = Math.floor((lo + hi) / 2); if (values[mid]! < requested) lo = mid + 1; else hi = mid }
  return lo > 0 && requested - values[lo - 1]! <= values[lo]! - requested ? lo - 1 : lo
}

export function SpectrumReview({ results, referenceRunId, mode = 'same_condition', onReference, onRestore }: {
  results: EvaluationResult[]; referenceRunId: string; mode?: 'same_condition' | 'cross_condition'; onReference?: (id: string, mode: 'same_condition' | 'cross_condition') => void; onRestore?: (figure: SavedFigure) => void
}) {
  const colors = useStudioColors()
  const qc = useQueryClient()
  const ids = results.map(r => r.run_id!).filter(Boolean)
  const spectra = useQueries({ queries: ids.map(id => artifactJsonQuery<SpectrumData>(id, 'plot-spectrum')) })
  const reviews = useQueries({ queries: results.filter(r => r.run_id).map(r => reviewQuery(r.run_id!, r.metric_profile_id)) })
  const figures = useQuery({ queryKey: ['figures', ...ids], queryFn: () => api.get<SavedFigure[]>(`/figures?${ids.map(id => `runs=${encodeURIComponent(id)}`).join('&')}`), enabled: !WEB_MODE })
  const [title, setTitle] = useState(t('chart.spectrum.title'))
  const [showBands, setShowBands] = useState(true)
  const [cursor, setCursor] = useState('')
  const [width, setWidth] = useState<'single_column' | 'double_column'>('double_column')
  const [options, setOptions] = useState<Record<string, FigureTrace>>({})
  const [loaded, setLoaded] = useState<SavedFigure | null>(null)
  const [savedId, setSavedId] = useState('')
  const [publication, setPublication] = useState<FigureSpec>()
  const viewport = useRef<PlotViewport | undefined>(undefined)
  const referenceIndex = ids.indexOf(referenceRunId)
  const context = reviews[referenceIndex]?.data
  const first = spectra.find(q => q.data)?.data
  const mismatch = spectra.some(q => q.data && q.data.axis !== first?.axis)
  const traceRows = useMemo(() => spectra.flatMap((q, i) => (q.data?.traces ?? []).map((tr, j) => {
    const result = results.find(r => r.run_id === ids[i])!
    const id = `${ids[i]}:${tr.name}`
    const measured = tr.source?.startsWith('measured') || ['measured PA without DPD', 'measured PA output'].includes(tr.name)
    const opt = options[id] ?? { run_id: ids[i]!, trace_name: tr.name, color: colors.chart[(ids.length === 1 ? j : i) % colors.chart.length]!, dash: result.is_mock || tr.role === 'reference' ? 'dot' : measured ? 'solid' : tr.role === 'input' ? 'longdash' : tr.role === 'baseline' ? 'dashdot' : ['dpd_surrogate', 'pa_modeling'].includes(result.evidence_type) ? 'dash' : 'solid', visible: ids.length === 1 || tr.role === 'primary' || (ids[i] === referenceRunId && tr.role === 'reference') }
    return { id, opt: opt as FigureTrace, trace: tr, data: q.data!, result }
  })), [spectra, results, ids, options, colors, referenceRunId])
  const visibleRows = useMemo(() => traceRows.filter(r => r.opt.visible !== false), [traceRows])
  const traces = useMemo(() => visibleRows.map(({ opt, trace, data, result }) => ({
    name: `${opt.run_id} · ${trace.name} [${result.is_mock ? 'MOCK' : trace.source ?? (trace.role === 'primary' ? result.evidence_type : trace.role)}]`,
    psdDb: trace.psd_db, frequencyHz: data.frequency, color: opt.color, dash: opt.dash,
    width: opt.run_id === referenceRunId ? 2 : 1.2, visible: opt.visible,
  })), [visibleRows, referenceRunId])
  const bands = useMemo(() => {
    const main = context?.bands?.find(b => b.role === 'main')
    if (!showBands || !main) return undefined
    return { main: main.edges_hz as [number, number], adjacent: (context?.bands ?? []).filter(b => b.role === 'adjacent').map(b => b.edges_hz as [number, number]) }
  }, [context, showBands])
  const panel = loaded?.spec.panels[0]
  const save = useMutation({
    mutationFn: () => {
      const v = viewport.current
      const spec: FigureSpec = { version: 'figure-v1', title, reference_run_id: referenceRunId, mode, width,
        profiles: Object.fromEntries(results.filter(r => r.run_id).map(r => [r.run_id!, r.metric_profile_id])),
        panels: [{ kind: 'spectrum', traces: traceRows.map(r => r.opt), show_bands: showBands,
          x_range: v ? v.autoX ? null : v.x : panel?.x_range, y_range: v ? v.autoY ? null : v.y : panel?.y_range,
          cursor_x: cursor.trim() && Number.isFinite(Number(cursor)) ? Number(cursor) : null }],
      }
      return api.post<SavedFigure>('/figures', spec)
    },
    onSuccess: async figure => { setSavedId(figure.figure_id); await qc.invalidateQueries({ queryKey: ['figures'] }) },
  })
  const restore = useMutation({
    mutationFn: (id: string) => api.get<SavedFigure>(`/figures/${encodeURIComponent(id)}`),
    onSuccess: figure => {
      const p = figure.spec.panels[0]!
      setLoaded(figure); setSavedId(figure.figure_id); setTitle(figure.spec.title); setShowBands(p.show_bands ?? true)
      setCursor(p.cursor_x === null || p.cursor_x === undefined ? '' : String(p.cursor_x)); setWidth(figure.spec.width ?? 'double_column')
      setOptions(Object.fromEntries(p.traces.map(tr => [`${tr.run_id}:${tr.trace_name}`, tr])))
      viewport.current = undefined
      onReference?.(figure.spec.reference_run_id, figure.spec.mode ?? 'same_condition')
      onRestore?.(figure)
    },
  })
  const cursorValue = cursor.trim() && Number.isFinite(Number(cursor)) ? Number(cursor) : undefined
  const dirty = () => setSavedId('')
  const selectedFigure = figures.data?.find(f => f.figure_id === savedId)
  const offeredFigures = (figures.data ?? []).filter(f => f.spec.panels.length === 1 && f.spec.panels[0]?.kind === 'spectrum' && f.bindings.every(b => b.review.profile.validation !== 'pending_cross_validation'))
  const matchesSaved = !!selectedFigure && selectedFigure.spec.reference_run_id === referenceRunId && (selectedFigure.spec.mode ?? 'same_condition') === mode && results.every(r => selectedFigure.spec.profiles[r.run_id!] === r.metric_profile_id)
  if (!first) return spectra.some(q => q.isPending) ? null : <Typography variant="body2">{t('results.charts.none')}</Typography>
  return <Paper component="section" aria-label={t('compare.spectrum')} sx={{ p: 2 }} data-testid="spectrum-review">
    <Stack spacing={1.5}>
      <Typography variant="h3" component="h2">{t('compare.spectrum')}</Typography>
      {!WEB_MODE && <Stack direction="row" useFlexGap sx={{ flexWrap: 'wrap', gap: 1, alignItems: 'center' }}>
        <TextField size="small" label={t('review.viewTitle')} value={title} onChange={e => { setTitle(e.target.value); dirty() }} sx={{ minWidth: 180, flex: 1 }} />
        <TextField size="small" select label={t('review.saved')} value={matchesSaved ? savedId : ''} onChange={e => { if (e.target.value) restore.mutate(e.target.value) }} sx={{ minWidth: 180, maxWidth: 280 }}>
          <MenuItem value="">{t('review.current')}</MenuItem>{offeredFigures.map(f => <MenuItem key={f.figure_id} value={f.figure_id}>{f.spec.title} · {f.created_at?.slice(0, 16)}</MenuItem>)}
        </TextField>
        <TextField select size="small" label={t('review.width')} value={width} onChange={e => { setWidth(e.target.value as typeof width); dirty() }}>
          <MenuItem value="single_column">{t('review.single')}</MenuItem><MenuItem value="double_column">{t('review.double')}</MenuItem>
        </TextField>
        <Button variant="outlined" disabled={save.isPending || !title.trim() || mismatch || spectra.some(q => q.isPending) || !traceRows.some(r => r.opt.visible !== false)} onClick={() => save.mutate()}>{t('review.save')}</Button>
        {matchesSaved && <DownloadLink button href={`${API}/figures/${encodeURIComponent(savedId)}/export`} download>{t('review.export')}</DownloadLink>}
        <Button variant="outlined" disabled={mismatch || !traceRows.some(r => r.opt.visible !== false)} onClick={() => setPublication({ version: 'figure-v2', title, reference_run_id: referenceRunId, mode, width, profiles: Object.fromEntries(results.filter(r => r.run_id).map(r => [r.run_id!, r.metric_profile_id])), panels: [{ kind: 'spectrum', traces: traceRows.map(r => r.opt), show_bands: showBands, x_range: viewport.current?.autoX ? null : viewport.current?.x ?? panel?.x_range, y_range: viewport.current?.autoY ? null : viewport.current?.y ?? panel?.y_range, cursor_x: cursorValue ?? null }] })}>{t('publication.title')}</Button>
      </Stack>}
      {(save.isError || restore.isError) && <Alert severity="error">{save.error?.message ?? restore.error?.message}</Alert>}
      <Stack direction="row" useFlexGap sx={{ alignItems: 'center', flexWrap: 'wrap', gap: 2 }}>
        <FormControlLabel control={<Checkbox checked={showBands} onChange={e => { setShowBands(e.target.checked); dirty() }} />} label={t('review.bands')} />
        <TextField size="small" type="number" label={`${t('review.cursor')} (${first.axis === 'hz' ? 'MHz' : 'cycles/sample'})`} value={cursor} onChange={e => { setCursor(e.target.value); dirty() }} slotProps={{ htmlInput: { step: 'any' } }} />
      </Stack>
      {mismatch ? <Alert severity="info">{t('compare.spectrum.mismatch')}</Alert> : <SpectrumPlot
        frequencyHz={first.frequency} axis={first.axis} traces={traces} bands={bands} cursorX={cursorValue}
        xRange={panel?.x_range as [number, number] | undefined} yRange={panel?.y_range as [number, number] | undefined}
        viewKey={`${ids.join(':')}:${loaded?.figure_id ?? 'current'}`} onViewportChange={v => {
          if (viewport.current && JSON.stringify(viewport.current) !== JSON.stringify(v)) dirty()
          viewport.current = v
        }}
        onVisibilityChange={visible => { setOptions(Object.fromEntries(traceRows.map(r => [r.id, { ...r.opt, visible: visibleRows.includes(r) ? visible[visibleRows.indexOf(r)] ?? true : false }]))); dirty() }}
      />}
      <Typography variant="caption" color="text.secondary">{context?.band_note ?? first.estimator}</Typography>
      <Accordion disableGutters elevation={0}><AccordionSummary expandIcon={<ExpandMoreIcon />}>{t('review.trace')}</AccordionSummary><AccordionDetails>
        <Stack>{traceRows.map(r => <FormControlLabel key={r.id} control={<Checkbox size="small" checked={r.opt.visible !== false} onChange={e => { setOptions({ ...options, [r.id]: { ...r.opt, visible: e.target.checked } }); dirty() }} />} label={`${r.opt.run_id} · ${r.trace.name} · ${r.trace.source ?? r.trace.role}${r.trace.capture_id ? ` · ${r.trace.capture_id}` : ''}`} />)}</Stack>
      </AccordionDetails></Accordion>
      {(context?.bands ?? []).some(b => !b.available) && <Alert severity="warning">{context?.bands?.filter(b => !b.available).map(b => `${b.label}: ${b.reason}`).join('; ')}</Alert>}
      {context?.bands?.length ? <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>{context.bands.filter(b => b.role !== 'subchannel').map(b => <Typography key={b.label} variant="caption">{b.label}: [{(b.edges_hz[0]! / 1e6).toFixed(3)}, {(b.edges_hz[1]! / 1e6).toFixed(3)}) MHz</Typography>)}</Box> : null}
      {cursorValue !== undefined && <>
        <Typography variant="caption">{t('review.cursorHelp')}</Typography>
        <TableContainer tabIndex={0}><Table size="small" aria-label={t('review.cursor')}><TableHead><TableRow><TableCell>{t('review.trace')}</TableCell><TableCell>MHz / cycles/sample</TableCell><TableCell>PSD (dB)</TableCell></TableRow></TableHead><TableBody>
          {traceRows.filter(r => r.opt.visible !== false).map(r => {
            const index = nearestBin(r.data.frequency, cursorValue * (first.axis === 'hz' ? 1e6 : 1))
            return <TableRow key={r.id}><TableCell>{r.opt.run_id} · {r.trace.name} · {r.trace.role}</TableCell><TableCell>{index < 0 ? t('review.missing') : (r.data.frequency[index]! / (first.axis === 'hz' ? 1e6 : 1)).toFixed(5)}</TableCell><TableCell>{index < 0 ? t('review.missing') : r.trace.psd_db[index]?.toFixed(3)}</TableCell></TableRow>
          })}
        </TableBody></Table></TableContainer>
      </>}
    </Stack>
    {publication && <PublicationFigureDialog initial={publication} onClose={() => setPublication(undefined)} />}
  </Paper>
}
