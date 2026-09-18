import { useMemo } from 'react'
import type { MetricPoint } from '@/api/events'
import { t } from '@/i18n'
import { PlotlyChart, seriesDash, seriesSymbol, type PlotLayout, type PlotTrace } from './PlotlyChart'

/** Training curves from `metric` events: one trace per (split, metric). */
export function MetricHistoryChart({ points, metric, height = 320 }: { points: MetricPoint[]; metric: string; height?: number }) {
  const traces = useMemo<PlotTrace[]>(() => {
    const bySplit = new Map<string, { x: number[]; y: number[] }>()
    for (const p of points) {
      if (p.split.endsWith('_probe')) continue
      const v = p.values[metric]
      if (v === undefined) continue
      const acc = bySplit.get(p.split) ?? { x: [], y: [] }
      acc.x.push(p.epoch + 1)
      acc.y.push(v)
      bySplit.set(p.split, acc)
    }
    return [...bySplit.entries()].map(([split, s], i) => ({ x: s.x, y: s.y, name: `${split} ${metric}`, mode: 'lines+markers', type: 'scatter', line: { dash: seriesDash(i) }, marker: { symbol: seriesSymbol(i) } }))
  }, [points, metric])
  const xTitle = t('chart.history.x')
  const layout = useMemo<PlotLayout>(() => ({
    xaxis: { title: { text: xTitle, font: { size: 14 } }, tickfont: { size: 13 } },
    yaxis: { title: { text: metric, font: { size: 14 } }, tickfont: { size: 13 } },
    margin: { l: 60, r: 18, t: 92, b: 54 },
    legend: { orientation: 'h', x: 0, xanchor: 'left', y: 1.03, yanchor: 'bottom', font: { size: 14 } },
    showlegend: true,
  }), [xTitle, metric])
  return <PlotlyChart title={t('chart.history.title', { metric })} traces={traces} layout={layout} height={height} viewKey={metric} data-testid={`history-${metric}`} />
}
