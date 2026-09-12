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
      // Epoch-end events are zero-based; batch probes already carry fractional completed epochs.
      acc.x.push(p.split.endsWith('_probe') ? p.epoch : p.epoch + 1)
      acc.y.push(v)
      bySplit.set(p.split, acc)
    }
    return [...bySplit.entries()].map(([split, s], i) => ({ x: s.x, y: s.y, name: `${split} ${metric}`, mode: 'lines+markers', type: 'scatter', line: { dash: seriesDash(i) }, marker: { symbol: seriesSymbol(i) } }))
  }, [points, metric])
  const xTitle = t('chart.history.x')
  const layout = useMemo<PlotLayout>(() => ({ xaxis: { title: { text: xTitle } }, yaxis: { title: { text: metric } }, showlegend: true }), [xTitle, metric])
  return <PlotlyChart title={t('chart.history.title', { metric })} traces={traces} layout={layout} height={height} viewKey={metric} data-testid={`history-${metric}`} />
}
