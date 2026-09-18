import { Box, Typography, useMediaQuery } from '@mui/material'
import { useMemo } from 'react'
import { t, useLanguage } from '@/i18n'
import { SpectrumPlot, type SpectrumPlotProps, type SpectrumTrace } from './SpectrumPlot'
import type { PlotViewport } from './plotInteractions'
import { hasDPD, nodeTitle, spectrumGroups, spectrumLegend, type SignalIdentity, type SignalNode } from './spectrumNodes'

export type NodeSpectrumTrace = SpectrumTrace & SignalIdentity & { legendName?: string }
export type SpectrumViews = Partial<Record<SignalNode, { x_range?: number[] | null; y_range?: number[] | null }>>
interface Props extends Omit<SpectrumPlotProps, 'traces' | 'onViewportChange'> {
  traces: NodeSpectrumTrace[]
  dpd?: boolean
  views?: SpectrumViews
  onViewportChange?: (viewport: PlotViewport, node: SignalNode) => void
}

/** One physical signal-chain location per chart. Only traces at that location
 * share a legend. Each chart keeps its own zoom; initial scales match. */
export function SpectrumPanels({ traces, views, title, dpd: dpdContext, onViewportChange, onVisibilityChange, ...props }: Props) {
  const narrow = useMediaQuery('(max-width: 600px)')
  useLanguage()
  const dpd = dpdContext ?? hasDPD(traces)
  const legendKey = JSON.stringify(traces.map(trace => trace.legendName ?? spectrumLegend(trace)))
  // Stable arrays matter: a parent's render-timing callback can update state.
  // Rebuilding trace arrays on that update would start another Plotly render.
  const groups = useMemo(() => {
    const names = JSON.parse(legendKey) as string[]
    return spectrumGroups(traces).map(group => ({ ...group,
      drawn: group.traces.map(trace => ({ ...trace, name: names[traces.indexOf(trace)] ?? trace.name })),
    }))
  }, [traces, legendKey])
  const rendered = useMemo(() => new Map(groups.map(group => [group.node, undefined as number | undefined])), [groups])
  const range = useMemo<[number, number] | undefined>(() => {
    let lo = Infinity, hi = -Infinity
    for (const trace of traces) for (let i = 0; i < trace.psdDb.length; i++) {
      const v = trace.psdDb[i]!
      if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v) }
    }
    return Number.isFinite(lo) ? [Math.floor((lo - 3) / 10) * 10, Math.ceil((hi + 3) / 10) * 10] : undefined
  }, [traces])
  return <Box data-testid="spectrum-panels">
    {title && <Typography variant="subtitle2" sx={{ mb: 1 }}>{title}</Typography>}
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: 'minmax(0, 1fr)', md: 'repeat(2, minmax(0, 1fr))', xl: `repeat(${Math.min(3, groups.length)}, minmax(0, 1fr))` }, gap: 2 }}>
      {groups.map(({ node, traces: rows, drawn }, i) => <Box key={node} data-signal-node={node} sx={{ minWidth: 0, p: 1, border: '1px solid', borderColor: 'divider', borderRadius: 1, gridColumn: { md: groups.length % 2 && i === groups.length - 1 ? '1 / -1' : 'auto', xl: 'auto' } }}>
        <SpectrumPlot {...props} title={`${nodeTitle(node, dpd)} · PSD`} height={narrow ? Math.max(380, props.height ?? 330) : props.height ?? 330}
          legendRows={Math.max(...groups.map(group => group.traces.length))}
          traces={drawn} onRendered={ms => { rendered.set(node, ms); const times = [...rendered.values()]; if (times.every(time => time !== undefined)) props.onRendered?.(Math.max(...times)) }}
          xRange={views?.[node]?.x_range as [number, number] | undefined ?? props.xRange}
          yRange={views?.[node]?.y_range as [number, number] | undefined ?? props.yRange ?? range}
          viewKey={`${props.viewKey ?? ''}:${node}`} onViewportChange={v => onViewportChange?.(v, node)}
          onVisibilityChange={visible => onVisibilityChange?.(traces.map(trace => rows.includes(trace) ? visible[rows.indexOf(trace)] ?? true : trace.visible !== false))} />
        {node === 'pa_output' && dpd && rows.some(trace => /surrogate/i.test(`${trace.name} ${trace.source ?? ''}`)) &&
          <Typography variant="body2" color="text.secondary" data-testid="pa-model-explanation" sx={{ mt: 1, px: 1, lineHeight: 1.65 }}>{t('spectrum.paModelHelp')}</Typography>}
      </Box>)}
    </Box>
    {groups.length > 1 && <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>{t('spectrum.separated')}</Typography>}
  </Box>
}
