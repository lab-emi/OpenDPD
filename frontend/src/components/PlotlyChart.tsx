import FullscreenIcon from '@mui/icons-material/Fullscreen'
import CloseIcon from '@mui/icons-material/Close'
import HelpOutlineIcon from '@mui/icons-material/HelpOutlineOutlined'
import Box from '@mui/material/Box'
import Dialog from '@mui/material/Dialog'
import DialogContent from '@mui/material/DialogContent'
import DialogTitle from '@mui/material/DialogTitle'
import IconButton from '@mui/material/IconButton'
import FormControlLabel from '@mui/material/FormControlLabel'
import MenuItem from '@mui/material/MenuItem'
import Popover from '@mui/material/Popover'
import Stack from '@mui/material/Stack'
import Switch from '@mui/material/Switch'
import TextField from '@mui/material/TextField'
import Tooltip from '@mui/material/Tooltip'
import Typography from '@mui/material/Typography'
import { useEffect, useId, useRef, useState, type ReactNode, type RefObject } from 'react'
import { languageInfo, message, phaseLabel, t, useLanguage } from '@/i18n'
import { plotLayoutFor, useStudioColors } from '@/theme'
import { attachPlotInteractions, readViewport, viewportUpdate, type PlotElement, type PlotInteractionApi, type PlotViewport } from './plotInteractions'
import { DEFAULT_PLOT_RECOVERY, PLOT_RECOVERY_IDLE_MS, type PlotRecoverySettings, type PlotRecoveryState } from './plotRecovery'

export type SeriesDash = 'solid' | 'dash' | 'dot' | 'dashdot' | 'longdash'
export type SeriesSymbol = 'circle' | 'triangle-up' | 'square' | 'diamond' | 'cross'
/** Line dash and marker symbol by series index: every trace differs in shape, not only in colour (UX spec §3). */
export const SERIES_DASHES: readonly SeriesDash[] = ['solid', 'dash', 'dot', 'dashdot', 'longdash']
export const SERIES_SYMBOLS: readonly SeriesSymbol[] = ['circle', 'triangle-up', 'square', 'diamond', 'cross']
export const seriesDash = (i: number): SeriesDash => SERIES_DASHES[i % SERIES_DASHES.length] ?? 'solid'
export const seriesSymbol = (i: number): SeriesSymbol => SERIES_SYMBOLS[i % SERIES_SYMBOLS.length] ?? 'circle'

/** The subset of Plotly this app uses (typed locally on purpose). */
export interface PlotTrace {
  x: ArrayLike<number>
  y: ArrayLike<number>
  name?: string
  mode?: 'lines' | 'markers' | 'lines+markers'
  type?: 'scatter' | 'scattergl'
  line?: { width?: number; dash?: SeriesDash; color?: string }
  marker?: { size?: number; opacity?: number; color?: string; symbol?: SeriesSymbol }
  hoverinfo?: 'x+y+name' | 'skip'
}
export interface PlotLayout {
  title?: { text: string }
  xaxis?: { title?: { text: string }; range?: [number, number]; constrain?: 'domain' }
  yaxis?: { title?: { text: string }; range?: [number, number]; scaleanchor?: string; scaleratio?: number }
  shapes?: Array<{ type: 'rect'; x0: number; x1: number; y0: number; y1: number; yref: 'paper'; fillcolor: string; line: { width: number } }>
  showlegend?: boolean
  height?: number
}
interface PlotlyModule extends PlotInteractionApi {
  react: (el: HTMLElement, data: PlotTrace[], layout: object, config: object) => Promise<unknown>
  purge: (el: HTMLElement) => void
}

let svgPromise: Promise<PlotlyModule> | null = null
let webGLPromise: Promise<PlotlyModule> | null = null
/** Load WebGL only for dense scatter; the strict bundle preserves the app's CSP. */
function loadPlotly(accelerated: boolean): Promise<PlotlyModule> {
  if (accelerated) {
    webGLPromise ??= import('plotly.js-strict-dist-min').then((m) => (m.default ?? m) as unknown as PlotlyModule)
    return webGLPromise
  }
  svgPromise ??= import('plotly.js-basic-dist-min').then((m) => (m.default ?? m) as unknown as PlotlyModule)
  return svgPromise
}

let webGLAvailable: boolean | undefined
function supportsWebGL() {
  if (typeof WebGLRenderingContext === 'undefined') return false
  if (webGLAvailable === undefined) {
    try {
      const context = document.createElement('canvas').getContext('webgl')
      webGLAvailable = !!context
      context?.getExtension('WEBGL_lose_context')?.loseContext()
    } catch { webGLAvailable = false }
  }
  return webGLAvailable
}

export interface PlotlyChartProps {
  title: string
  traces: PlotTrace[]
  layout?: PlotLayout
  height?: number
  /** Called once the plot is drawn (used by the performance probe). */
  onRendered?: (ms: number) => void
  enlargeable?: boolean
  /** Change when switching to a different signal/coordinate system, not on live updates. */
  viewKey?: string | number
  actions?: ReactNode
  'data-testid'?: string
}

interface ViewMemory { key: string | number; viewport?: PlotViewport; recovery?: PlotRecoveryState }

const MODEBAR_TEXT = ['Download plot as a PNG', 'Zoom', 'Pan', 'Zoom in', 'Zoom out', 'Autoscale', 'Reset axes', 'Toggle Spike Lines', 'Show closest data on hover', 'Compare data on hover', 'Taking snapshot - this may take a few seconds', 'Snapshot succeeded', 'Snapshot failed', 'Double-click on legend to isolate one trace']
function traceName(name: string | undefined) {
  if (!name) return name
  const history = /^(train|val|test|validation|train_probe|val_probe|validation_probe|test_probe) (.+)$/.exec(name)
  if (history) return languageInfo().code === 'en' && !history[1]!.endsWith('_probe') ? name : `${phaseLabel(history[1]!)} ${history[2]}`
  const component = /^(.*) ([IQ])$/.exec(name)
  return component ? `${message(component[1])} ${component[2]}` : message(name)
}

function Plot({ traces, layout, height, title, onRendered, viewKey = 0, memory, active = true, descriptionId, recovery }: PlotlyChartProps & { memory: RefObject<ViewMemory>; active?: boolean; descriptionId: string; recovery: PlotRecoverySettings }) {
  const language = useLanguage()
  const colors = useStudioColors()
  const ref = useRef<PlotElement>(null)
  const revision = useRef(0)
  const rendering = useRef<Promise<unknown>>(Promise.resolve())
  const library = useRef<PlotlyModule | null>(null)
  const [failed, setFailed] = useState<string | null>(null)
  const [useSVG, setUseSVG] = useState(false)
  // The callback is read through a ref so a new function identity never redraws the plot
  // (a redraw that reports back into parent state would otherwise loop).
  const onRenderedRef = useRef(onRendered)
  useEffect(() => {
    onRenderedRef.current = onRendered
  })
  useEffect(() => {
    const el = ref.current
    if (!el || !active) return
    let cancelled = false
    let inputs: ReturnType<typeof attachPlotInteractions> | undefined
    let observer: ResizeObserver | undefined
    const contextLost = () => { if (!cancelled) setUseSVG(true) }
    el.addEventListener('webglcontextlost', contextLost, true)
    const started = performance.now()
    const denseMarkers = !useSVG && traces.some((trace) => trace.mode === 'markers' && trace.x.length >= 1000)
    const accelerated = denseMarkers && supportsWebGL()
    // Serialize source/layout changes with the previous draw of this instance.
    rendering.current = rendering.current.catch(() => undefined).then(() => loadPlotly(accelerated))
      .then(async (Plotly) => {
        if (cancelled) return
        if (library.current && library.current !== Plotly) library.current.purge(el)
        library.current = Plotly
        if (memory.current.key !== viewKey) memory.current = { key: viewKey }
        const saved = memory.current.viewport
        const view = saved ? { ...saved, x: [...saved.x] as [number, number], y: [...saved.y] as [number, number] } : undefined
        // Keep every point. Only the renderer changes for dense marker traces;
        // line paths and short training histories are already efficient as SVG.
        const renderTraces = traces.map<PlotTrace>((trace) => ({ ...trace, name: traceName(trace.name), type: accelerated && trace.mode === 'markers' ? 'scattergl' : 'scatter' }))
        const plotLayoutBase = plotLayoutFor(colors)
        try {
          await Plotly.react(el, renderTraces, {
            ...plotLayoutBase, ...layout, height, autosize: true,
            dragmode: view?.dragmode ?? 'pan',
            uirevision: String(viewKey),
            // Recompute bounds on layout-only updates too (for example a language
            // change), instead of reusing calcdata with newly defaulted axis ranges.
            datarevision: ++revision.current,
            xaxis: { ...plotLayoutBase.xaxis, ...layout?.xaxis, ...(view ? { range: [...view.x], autorange: view.autoX } : {}) },
            yaxis: { ...plotLayoutBase.yaxis, ...layout?.yaxis, ...(view ? { range: [...view.y], autorange: view.autoY } : {}) },
          }, {
            displaylogo: false, responsive: false, scrollZoom: false,
            locale: languageInfo(language).tag,
            locales: { [languageInfo(language).tag]: { dictionary: Object.fromEntries(MODEBAR_TEXT.map((text) => [text, message(text)])) } },
            // Wheel/pinch are handled once by the shared controller; native mouse
            // dragging, box zoom, fit/reset, legend controls and export stay Plotly's.
            modeBarButtonsToRemove: ['select2d', 'lasso2d'],
            doubleClick: 'autosize',
          })
        } catch (error) {
          if (accelerated && !cancelled) { setUseSVG(true); return }
          throw error
        }
        if (cancelled) return
        // Plotly.react can restore an old range or autorange state when an inline
        // plot resumes. Reconcile both, including a fit performed in the dialog.
        const drawn = readViewport(el)
        if (view && drawn && (view.autoX !== drawn.autoX || view.autoY !== drawn.autoY || (!view.autoX && drawn.x.some((value, i) => value !== view.x[i])) || (!view.autoY && drawn.y.some((value, i) => value !== view.y[i])))) {
          await Plotly.relayout(el, viewportUpdate(view))
          if (cancelled) return
        }
        const recoveryState = memory.current.recovery ??= {}
        inputs = attachPlotInteractions(el, Plotly, (viewport) => {
          memory.current = { ...memory.current, key: viewKey, viewport }
        }, (error) => setFailed(error instanceof Error ? error.message : String(error)), {
          settings: recovery, state: recoveryState, traces: () => el['_fullData'] ?? traces,
        })
        if (typeof ResizeObserver !== 'undefined') {
          let width = el.clientWidth, size = el.clientHeight
          observer = new ResizeObserver(() => {
            if (el.clientWidth !== width || el.clientHeight !== size) {
              width = el.clientWidth; size = el.clientHeight
              inputs?.resize()
            }
          })
          observer.observe(el)
        }
        onRenderedRef.current?.(performance.now() - started)
      })
      .catch((err: unknown) => { if (!cancelled) setFailed(err instanceof Error ? err.message : String(err)) })
    return () => {
      cancelled = true
      el.removeEventListener('webglcontextlost', contextLost, true)
      observer?.disconnect()
      const stopped = inputs?.dispose()
      if (stopped) rendering.current = Promise.all([rendering.current, stopped])
    }
  }, [traces, layout, height, active, viewKey, memory, useSVG, recovery, language, colors])
  useEffect(() => {
    const el = ref.current
    return () => {
      if (el) rendering.current = rendering.current.catch(() => undefined)
        .then(() => { library.current?.purge(el); library.current = null })
    }
  }, [])
  if (failed) return <Typography color="error">{message(failed)}</Typography>
  // "figure", not "img": Plotly's mode bar inside the plot is focusable, and an image role may not contain controls.
  return <Box ref={ref} role="figure" aria-label={title} aria-describedby={descriptionId} tabIndex={active ? 0 : -1} sx={{ width: '100%', height: height ?? '100%', minHeight: height ?? 0, '&:focus-visible': { outline: '2px solid', outlineColor: 'primary.main', outlineOffset: -2 } }} />
}

/** Plotly wrapper: tokens-derived layout, lazy bundle, and an enlarge dialog (UX spec §6).
 * The title row sits above the plot so it never overlaps Plotly's own mode bar. */
export function PlotlyChart(props: PlotlyChartProps) {
  const { title, enlargeable = true, height = 320, viewKey = 0, actions } = props
  const [open, setOpen] = useState(false)
  const [helpAnchor, setHelpAnchor] = useState<HTMLButtonElement | null>(null)
  const [recovery, setRecovery] = useState<PlotRecoverySettings>(() => ({ ...DEFAULT_PLOT_RECOVERY }))
  const memory = useRef<ViewMemory>({ key: viewKey })
  const id = useId()
  const descriptionId = `${id}-controls`
  const close = () => { setHelpAnchor(null); setOpen(false) }
  const help = <Tooltip title={t('chart.controls')}><IconButton size="small" aria-label={t('chart.controls')} onClick={(event) => setHelpAnchor(event.currentTarget)}><HelpOutlineIcon fontSize="small" /></IconButton></Tooltip>
  return (
    <Box data-testid={props['data-testid']}>
      <Stack direction="row" sx={{ alignItems: 'center', gap: .5, minHeight: 32 }}>
        <Typography variant="subtitle2" component="h3" noWrap title={title} sx={{ flex: 1, minWidth: 0 }}>
          {title}
        </Typography>
        {actions}
        {help}
        {enlargeable && (
          <IconButton aria-label={`${t('chart.enlarge')}: ${title}`} size="small" onClick={() => setOpen(true)}>
            <FullscreenIcon fontSize="small" />
          </IconButton>
        )}
      </Stack>
      <Box id={descriptionId} sx={{ position: 'absolute', width: '1px', height: '1px', overflow: 'hidden', clipPath: 'inset(50%)', whiteSpace: 'nowrap' }}>{t('chart.controls.summary')}</Box>
      <Plot {...props} height={height} active={!open} memory={memory} descriptionId={descriptionId} recovery={recovery} />
      {open && (
        <Dialog open fullWidth maxWidth={false} onClose={close} aria-labelledby={`${id}-title`} slotProps={{ paper: { sx: { width: 'calc(100vw - 48px)', height: 'calc(100dvh - 48px)', maxHeight: 'none', m: 3 } } }}>
          <DialogTitle id={`${id}-title`} sx={{ display: 'flex', alignItems: 'center', gap: 1, py: 1, flexShrink: 0 }}>
            <Box component="span" sx={{ flex: 1, minWidth: 0 }}>{title}</Box>
            {actions}{help}
            <IconButton aria-label={t('chart.close')} onClick={close}><CloseIcon /></IconButton>
          </DialogTitle>
          <DialogContent sx={{ display: 'flex', flexDirection: 'column', minHeight: 0, overflow: 'hidden', p: 1 }}>
            <Typography id={`${id}-large-controls`} variant="caption" color="text.secondary" sx={{ px: 1, flexShrink: 0 }}>{t('chart.controls.summary')}</Typography>
            <Box sx={{ flex: 1, minHeight: 0 }}><Plot {...props} height={undefined} memory={memory} descriptionId={`${id}-large-controls`} recovery={recovery} /></Box>
          </DialogContent>
        </Dialog>
      )}
      <Popover open={!!helpAnchor} anchorEl={helpAnchor} onClose={() => setHelpAnchor(null)} anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }} transformOrigin={{ vertical: 'top', horizontal: 'right' }}>
        <Box role="dialog" aria-label={t('chart.controls')} sx={{ p: 2, maxWidth: 380 }}>
          <Stack direction="row" sx={{ alignItems: 'center', justifyContent: 'space-between', mb: 1 }}><Typography variant="subtitle2">{t('chart.controls')}</Typography><IconButton size="small" aria-label={t('chart.controls.close')} onClick={() => setHelpAnchor(null)}><CloseIcon fontSize="small" /></IconButton></Stack>
          <Stack spacing={1}>{(['chart.controls.pan', 'chart.controls.zoom', 'chart.controls.mouse', 'chart.controls.keyboard', 'chart.controls.page'] as const).map((key) => <Typography key={key} variant="body2">{t(key)}</Typography>)}</Stack>
          <Stack spacing={1.5} sx={{ mt: 2, pt: 1.5, borderTop: '1px solid', borderColor: 'divider' }}>
            <FormControlLabel control={<Switch checked={recovery.enabled} onChange={(_event, enabled) => setRecovery((old) => ({ ...old, enabled }))} />} label={t('chart.recovery.enabled')} />
            <TextField select label={t('chart.recovery.threshold')} value={recovery.emptyThreshold * 100} disabled={!recovery.enabled} onChange={(event) => setRecovery((old) => ({ ...old, emptyThreshold: Number(event.target.value) / 100 }))}>
              {[65, 75, 85, 95].map((percent) => <MenuItem key={percent} value={percent}>{percent}%</MenuItem>)}
            </TextField>
            <Typography variant="body2" color="text.secondary">{t('chart.recovery.note', { threshold: recovery.emptyThreshold * 100, delay: PLOT_RECOVERY_IDLE_MS / 1000 })}</Typography>
          </Stack>
        </Box>
      </Popover>
    </Box>
  )
}
