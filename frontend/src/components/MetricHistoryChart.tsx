import { useMemo } from 'react'
import type { MetricPoint } from '@/api/events'
import { t } from '@/i18n'
import { PlotlyChart, seriesDash, seriesSymbol, type PlotLayout, type PlotTrace } from './PlotlyChart'

/** Training curves from `metric` events: one trace per (split, metric). */
export function MetricHistoryChart({ points, metric, height = 280 }: { points: MetricPoint[]; metric: string; height?: number }) {
  const traces = useMemo<PlotTrace[]>(() => {
    const bySplit = new Map<string, { x: number[]; y: number[] }>()
    for (const p of points) {
      const v = p.values[metric]
      if (v === undefined) continue
      const acc = bySplit.get(p.split) ?? { x: [], y: [] }
      acc.x.push(p.epoch)
      acc.y.push(v)
      bySplit.set(p.split, acc)
    }
    return [...bySplit.entries()].map(([split, s], i) => ({ x: s.x, y: s.y, name: `${split} ${metric}`, mode: 'lines+markers', type: 'scatter', line: { dash: seriesDash(i) }, marker: { symbol: seriesSymbol(i) } }))
  }, [points, metric])
  const layout = useMemo<PlotLayout>(() => ({ xaxis: { title: { text: t('chart.history.x') } }, yaxis: { title: { text: metric } }, showlegend: true }), [metric])
  return <PlotlyChart title={`${metric} per epoch`} traces={traces} layout={layout} height={height} data-testid={`history-${metric}`} />
}
