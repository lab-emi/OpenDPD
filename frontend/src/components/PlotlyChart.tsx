import FullscreenIcon from '@mui/icons-material/Fullscreen'
import Box from '@mui/material/Box'
import Dialog from '@mui/material/Dialog'
import DialogContent from '@mui/material/DialogContent'
import DialogTitle from '@mui/material/DialogTitle'
import IconButton from '@mui/material/IconButton'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import { useEffect, useRef, useState } from 'react'
import { t } from '@/i18n'
import { plotLayoutBase } from '@/theme'

/** The subset of the Plotly basic bundle this app uses (typed locally on purpose). */
export interface PlotTrace {
  x: ArrayLike<number>
  y: ArrayLike<number>
  name?: string
  mode?: 'lines' | 'markers' | 'lines+markers'
  type?: 'scatter'
  line?: { width?: number; dash?: 'solid' | 'dot' | 'dash'; color?: string }
  marker?: { size?: number; opacity?: number; color?: string }
  hoverinfo?: 'x+y+name' | 'skip'
}
export interface PlotLayout {
  title?: { text: string }
  xaxis?: { title?: { text: string }; range?: [number, number] }
  yaxis?: { title?: { text: string }; range?: [number, number] }
  shapes?: Array<{ type: 'rect'; x0: number; x1: number; y0: number; y1: number; yref: 'paper'; fillcolor: string; line: { width: number } }>
  showlegend?: boolean
  height?: number
}
interface PlotlyModule {
  react: (el: HTMLElement, data: PlotTrace[], layout: object, config: object) => Promise<unknown>
  purge: (el: HTMLElement) => void
}

let plotlyPromise: Promise<PlotlyModule> | null = null
/** Plotly (~1 MB) is loaded on first use so pages without charts stay light. */
function loadPlotly(): Promise<PlotlyModule> {
  plotlyPromise ??= import('plotly.js-basic-dist-min').then((m) => (m.default ?? m) as unknown as PlotlyModule)
  return plotlyPromise
}

export interface PlotlyChartProps {
  title: string
  traces: PlotTrace[]
  layout?: PlotLayout
  height?: number
  /** Called once the plot is drawn (used by the performance probe). */
  onRendered?: (ms: number) => void
  enlargeable?: boolean
  'data-testid'?: string
}

function Plot({ traces, layout, height, title, onRendered }: PlotlyChartProps) {
  const ref = useRef<HTMLDivElement>(null)
  const [failed, setFailed] = useState<string | null>(null)
  // The callback is read through a ref so a new function identity never redraws the plot
  // (a redraw that reports back into parent state would otherwise loop).
  const onRenderedRef = useRef(onRendered)
  useEffect(() => {
    onRenderedRef.current = onRendered
  })
  useEffect(() => {
    const el = ref.current
    if (!el) return
    let cancelled = false
    const started = performance.now()
    loadPlotly()
      .then((Plotly) => {
        if (cancelled) return
        return Plotly.react(el, traces, { ...plotLayoutBase, height, ...layout }, { displaylogo: false, responsive: true }).then(() => {
          if (!cancelled) onRenderedRef.current?.(performance.now() - started)
        })
      })
      .catch((err: unknown) => setFailed(err instanceof Error ? err.message : String(err)))
    return () => {
      cancelled = true
    }
  }, [traces, layout, height])
  useEffect(() => {
    const el = ref.current
    return () => {
      if (el) void loadPlotly().then((Plotly) => Plotly.purge(el))
    }
  }, [])
  if (failed) return <Typography color="error">{failed}</Typography>
  return <div ref={ref} role="img" aria-label={title} style={{ width: '100%', minHeight: height }} />
}

/** Plotly wrapper: tokens-derived layout, lazy bundle, and an enlarge dialog (UX spec §6).
 * The title row sits above the plot so it never overlaps Plotly's own mode bar. */
export function PlotlyChart(props: PlotlyChartProps) {
  const { title, enlargeable = true, height = 320 } = props
  const [open, setOpen] = useState(false)
  return (
    <Box data-testid={props['data-testid']}>
      <Stack direction="row" sx={{ alignItems: 'center', justifyContent: 'space-between', minHeight: 32 }}>
        <Typography variant="subtitle2" component="h3">
          {title}
        </Typography>
        {enlargeable && (
          <IconButton aria-label={`${t('chart.enlarge')}: ${title}`} size="small" onClick={() => setOpen(true)}>
            <FullscreenIcon fontSize="small" />
          </IconButton>
        )}
      </Stack>
      <Plot {...props} height={height} />
      {open && (
        <Dialog open fullWidth maxWidth="xl" onClose={() => setOpen(false)} aria-labelledby="chart-dialog-title">
          <DialogTitle id="chart-dialog-title">{title}</DialogTitle>
          <DialogContent>
            <Plot {...props} height={Math.max(480, Math.floor(window.innerHeight * 0.7))} enlargeable={false} />
          </DialogContent>
        </Dialog>
      )}
    </Box>
  )
}
