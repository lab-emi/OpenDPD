import { useMemo } from 'react'
import { t } from '@/i18n'
import { useStudioColors } from '@/theme'
import { PlotlyChart, seriesDash, type PlotLayout, type PlotTrace, type SeriesDash } from './PlotlyChart'
import type { PlotViewport } from './plotInteractions'

export interface SpectrumTrace {
  name: string
  color?: string
  frequencyHz?: ArrayLike<number>
  dash?: SeriesDash
  width?: number
  visible?: boolean
  /** PSD in dB, one value per frequency bin (already decimated by the server). */
  psdDb: ArrayLike<number>
}
export interface SpectrumBands {
  /** Main channel edges in Hz, relative to carrier. */
  main: [number, number]
  /** Adjacent channel edges in Hz (left and right). */
  adjacent: Array<[number, number]>
}

export interface SpectrumPlotProps {
  /** Frequency axis in Hz (same length as every trace). */
  frequencyHz: ArrayLike<number>
  /** 'hz' (default, drawn in MHz) or 'normalized' (cycles per sample, no bands). */
  axis?: 'hz' | 'normalized'
  traces: SpectrumTrace[]
  bands?: SpectrumBands
  title?: string
  height?: number
  /** Shared space for wrapping legends in adjacent signal-chain panels. */
  legendRows?: number
  onRendered?: (ms: number) => void
  viewKey?: string
  xRange?: [number, number]
  yRange?: [number, number]
  cursorX?: number
  onViewportChange?: (viewport: PlotViewport) => void
  onVisibilityChange?: (visible: boolean[]) => void
}

/** PSD traces on a dB axis with the ACLR integration bands shaded (UX spec §5). */
export function SpectrumPlot({ frequencyHz, axis = 'hz', traces, bands, title = t('chart.spectrum.title'), height, legendRows: sharedLegendRows, onRendered, viewKey = '', xRange, yRange, cursorX, onViewportChange, onVisibilityChange }: SpectrumPlotProps) {
  const colors = useStudioColors()
  // Reserve enough space for wrapped legend rows even in a narrow grid cell.
  // Keep the legend clear of both the mode bar and frequency-axis labels.
  const legendRows = Math.min(sharedLegendRows ?? traces.length, 6)
  const mhz = useMemo(() => Float64Array.from(frequencyHz, (f) => (axis === 'hz' ? f / 1e6 : f)), [frequencyHz, axis])
  const data = useMemo<PlotTrace[]>(
    () => traces.map((tr, i) => ({ x: tr.frequencyHz ? Float64Array.from(tr.frequencyHz, (f) => axis === 'hz' ? f / 1e6 : f) : mhz, y: tr.psdDb, name: tr.name, visible: tr.visible === false ? 'legendonly' : true, mode: 'lines', type: 'scatter', hovertemplate: `%{x:.4f} ${axis === 'hz' ? 'MHz' : 'cycles/sample'}<br>%{y:.3f} dB<extra>%{fullData.name}</extra>`, line: { width: tr.width ?? 1.2, dash: tr.dash ?? seriesDash(i), ...(tr.color ? { color: tr.color } : {}) } })),
    [traces, mhz, axis],
  )
  // Keyed by value so an inline `bands` literal does not redraw on every render.
  const bandsKey = JSON.stringify(bands ?? null)
  const xTitle = t(axis === 'hz' ? 'chart.spectrum.x' : 'chart.spectrum.x.normalized')
  const yTitle = t(axis === 'hz' ? 'chart.spectrum.y' : 'chart.spectrum.y.normalized')
  const layout = useMemo<PlotLayout>(() => {
    const parsed = JSON.parse(bandsKey) as SpectrumBands | null
    const shapes: NonNullable<PlotLayout['shapes']> = []
    if (parsed && axis === 'hz') {
      const shade = (edges: [number, number], color: string) =>
        shapes.push({ type: 'rect', x0: edges[0] / 1e6, x1: edges[1] / 1e6, y0: 0, y1: 1, yref: 'paper', fillcolor: color, line: { width: 0 } })
      shade(parsed.main, `${colors.primary}14`)
      for (const adj of parsed.adjacent) shade(adj, `${colors.status.warning}14`)
    }
    if (cursorX !== undefined) shapes.push({ type: 'line', x0: cursorX, x1: cursorX, y0: 0, y1: 1, yref: 'paper', line: { width: 1, dash: 'dot', color: colors.textSecondary } })
    return {
      xaxis: { title: { text: xTitle, font: { size: 14 } }, tickfont: { size: 13 }, ...(xRange ? { range: xRange } : {}) },
      yaxis: { title: { text: yTitle, font: { size: 14 } }, tickfont: { size: 13 }, ...(yRange ? { range: yRange } : {}) },
      shapes, showlegend: true,
      margin: { l: 62, r: 16, t: 52 + 24 * legendRows, b: 56 },
      legend: { orientation: 'h', x: 0, y: 1.02, yanchor: 'bottom', maxheight: Math.max(32, 24 * legendRows + 8), font: { size: 14 } },
    }
  }, [bandsKey, xTitle, yTitle, colors, axis, xRange, yRange, cursorX, legendRows])
  return <PlotlyChart title={title} traces={data} layout={layout} height={Math.max(height ?? 320, 328 + 24 * legendRows)} onRendered={onRendered} onViewportChange={onViewportChange} onVisibilityChange={onVisibilityChange} viewKey={`${viewKey}:${axis}`} data-testid="spectrum-plot" />
}
