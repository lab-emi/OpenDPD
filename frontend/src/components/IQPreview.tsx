import { useMemo } from 'react'
import { t } from '@/i18n'
import { PlotlyChart, type PlotLayout, type PlotTrace } from './PlotlyChart'

export interface IQSeries {
  name: string
  i: ArrayLike<number>
  q: ArrayLike<number>
}

export interface IQPreviewProps {
  /** Sample index of the first shown sample (window start). */
  start: number
  series: IQSeries[]
  title?: string
  height?: number
  onRendered?: (ms: number) => void
}

/** Short time-domain window; the server chooses the window, the browser never receives whole captures. */
export function IQPreview({ start, series, title = t('chart.iq.title'), height, onRendered }: IQPreviewProps) {
  const data = useMemo<PlotTrace[]>(() => {
    const out: PlotTrace[] = []
    for (const s of series) {
      const n = s.i.length
      const x = Float64Array.from({ length: n }, (_, k) => start + k)
      out.push({ x, y: s.i, name: `${s.name} I`, mode: 'lines', type: 'scatter', line: { width: 1 } })
      out.push({ x, y: s.q, name: `${s.name} Q`, mode: 'lines', type: 'scatter', line: { width: 1, dash: 'dot' } })
    }
    return out
  }, [series, start])
  const layout = useMemo<PlotLayout>(() => ({ xaxis: { title: { text: t('chart.iq.x') } }, yaxis: { title: { text: t('chart.iq.y') } }, showlegend: true }), [])
  return <PlotlyChart title={title} traces={data} layout={layout} height={height} onRendered={onRendered} data-testid="iq-preview" />
}
