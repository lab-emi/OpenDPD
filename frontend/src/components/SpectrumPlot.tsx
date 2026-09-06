import { useMemo } from 'react'
import { t } from '@/i18n'
import { tokens } from '@/theme'
import { PlotlyChart, type PlotLayout, type PlotTrace } from './PlotlyChart'

export interface SpectrumTrace {
  name: string
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
  traces: SpectrumTrace[]
  bands?: SpectrumBands
  title?: string
  height?: number
  onRendered?: (ms: number) => void
}

/** PSD traces on a dB axis with the ACLR integration bands shaded (UX spec §5). */
export function SpectrumPlot({ frequencyHz, traces, bands, title = t('chart.spectrum.title'), height, onRendered }: SpectrumPlotProps) {
  const mhz = useMemo(() => Float64Array.from(frequencyHz, (f) => f / 1e6), [frequencyHz])
  const data = useMemo<PlotTrace[]>(
    () => traces.map((tr) => ({ x: mhz, y: tr.psdDb, name: tr.name, mode: 'lines', type: 'scatter', line: { width: 1.2 } })),
    [traces, mhz],
  )
  // Keyed by value so an inline `bands` literal does not redraw on every render.
  const bandsKey = JSON.stringify(bands ?? null)
  const layout = useMemo<PlotLayout>(() => {
    const parsed = JSON.parse(bandsKey) as SpectrumBands | null
    const shapes: NonNullable<PlotLayout['shapes']> = []
    if (parsed) {
      const shade = (edges: [number, number], color: string) =>
        shapes.push({ type: 'rect', x0: edges[0] / 1e6, x1: edges[1] / 1e6, y0: 0, y1: 1, yref: 'paper', fillcolor: color, line: { width: 0 } })
      shade(parsed.main, `${tokens.color.primary}14`)
      for (const adj of parsed.adjacent) shade(adj, `${tokens.color.status.warning}14`)
    }
    return { xaxis: { title: { text: t('chart.spectrum.x') } }, yaxis: { title: { text: t('chart.spectrum.y') } }, shapes, showlegend: true }
  }, [bandsKey])
  return <PlotlyChart title={title} traces={data} layout={layout} height={height} onRendered={onRendered} data-testid="spectrum-plot" />
}
