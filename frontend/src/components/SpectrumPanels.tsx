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
  const language = useLanguage()
  const dpd = dpdContext ?? hasDPD(traces)
  // Stable arrays matter: a parent's render-timing callback can update state.
  // Rebuilding trace arrays on that update would start another Plotly render.
  const groups = useMemo(() => spectrumGroups(traces).map(group => ({ ...group,
    drawn: group.traces.map(trace => ({ ...trace, name: trace.legendName ?? spectrumLegend(trace) })),
  })), [traces, language])
  const rendered = useMemo(() => new Map<SignalNode, number>(), [groups, props.viewKey])
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
          traces={drawn} onRendered={ms => { rendered.set(node, ms); if (rendered.size === groups.length) props.onRendered?.(Math.max(...rendered.values())) }}
          xRange={views?.[node]?.x_range as [number, number] | undefined ?? props.xRange}
          yRange={views?.[node]?.y_range as [number, number] | undefined ?? props.yRange ?? range}
          viewKey={`${props.viewKey ?? ''}:${node}`} onViewportChange={v => onViewportChange?.(v, node)}
          onVisibilityChange={visible => onVisibilityChange?.(traces.map(trace => rows.includes(trace) ? visible[rows.indexOf(trace)] ?? true : trace.visible !== false))} />
      </Box>)}
    </Box>
    {groups.length > 1 && <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 1 }}>{t('spectrum.separated')}</Typography>}
  </Box>
}
