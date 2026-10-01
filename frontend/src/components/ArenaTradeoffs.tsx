import { Box, MenuItem, Stack, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, TextField, Typography } from '@mui/material'
import { useState } from 'react'
import type { ArenaRow } from '@/api/arena'
import { formatNumber, t } from '@/i18n'
import { useStudioColors } from '@/theme'
import { configurationPoints, configurationScore, paretoFront, type ArenaPoint } from '@/utils/arenaPareto'
import { PlotlyChart, seriesSymbol, type PlotTrace } from './PlotlyChart'

const number = (value: number) => formatNumber(value, { maximumFractionDigits: 2 })
// Write complete cost values on logarithmic axes, including minor ticks.
const costTicks = Array.from({ length: 8 }, (_, power) => [1, 2, 5].map(n => n * 10 ** power)).flat()
const escape = (value: string) => value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
const label = (p: ArenaPoint) => `${p.row.display_name} · ${number(p.parameters)} P`

export function ArenaConfigurationTable({ rows, protocol, ranking }: { rows: ArenaRow[]; protocol: string; ranking: string }) {
  const points = configurationPoints(rows, protocol)
    .map(point => ({ point, score: configurationScore(point, ranking) }))
    .filter((p): p is { point: ArenaPoint; score: number } => p.score !== null)
    .sort((a, b) => b.score - a.score || a.point.parameters - b.point.parameters || a.point.id.localeCompare(b.point.id))
  return <Stack spacing={1}>
    <Typography variant="h3">{t('arena.configurations')}</Typography>
    <Typography variant="body2" color="text.secondary">{t('arena.configurationHelp')}</Typography>
    <TableContainer tabIndex={0} role="region" aria-label={t('arena.configurations')} sx={{ maxHeight: 430 }}>
      <Table stickyHeader size="small" aria-label={t('arena.configurations')}>
        <TableHead><TableRow>{(['arena.rank', 'arena.configuration', 'arena.parameters', 'arena.ops', 'arena.scoreColumn', 'arena.evm', 'arena.aclr', 'arena.qualityStd'] as const).map(key => <TableCell key={key}>{t(key)}</TableCell>)}</TableRow></TableHead>
        <TableBody>{points.map(({ point: p, score }, index) => <TableRow key={p.id}>
          <TableCell>{index && points[index - 1]?.score === score ? points.findIndex(other => other.score === score) + 1 : index + 1}</TableCell>
          <TableCell>{label(p)}</TableCell><TableCell>{number(p.parameters)}</TableCell><TableCell>{number(p.ops)}</TableCell>
          <TableCell sx={{ fontWeight: 700, color: 'primary.main' }}>{number(score)}</TableCell>
          <TableCell sx={{ whiteSpace: 'nowrap' }}>{number(100 * 10 ** (p.evm / 20))} % · {number(p.evm)} dB</TableCell>
          <TableCell sx={{ whiteSpace: 'nowrap' }}>{number(p.aclr)} dBc</TableCell>
          <TableCell>±{number(p.budget.quality_std_db ?? 0)} dB</TableCell>
        </TableRow>)}</TableBody>
      </Table>
    </TableContainer>
    {!points.length && <Typography color="text.secondary">{t('arena.noValidConfigurations')}</Typography>}
  </Stack>
}

export function ArenaTradeoffs({ rows, protocol }: { rows: ArenaRow[]; protocol: string }) {
  const colors = useStudioColors()
  const conditions = [...new Set(rows.flatMap(row => (row.cases ?? []).map(item => item.condition_id).filter((v): v is string => typeof v === 'string')))].sort()
  const [chosen, setChosen] = useState('')
  const condition = conditions.includes(chosen) ? chosen : conditions[0]
  const points = configurationPoints(rows, protocol, condition)
  const backbones = [...new Set(rows.map(row => row.backbone))].sort()
  return <Stack spacing={1.5} sx={{ my: 2, minWidth: 0 }}>
    <Stack direction={{ xs: 'column', sm: 'row' }} sx={{ gap: 2, alignItems: { sm: 'center' }, justifyContent: 'space-between' }}>
      <Typography variant="h3">{t('arena.tradeoffs')}</Typography>
      {conditions.length > 1 && <TextField select size="small" label={t('arena.plotCondition')} value={condition ?? ''} onChange={event => setChosen(event.target.value)}>{conditions.map(id => <MenuItem key={id} value={id}>{id}</MenuItem>)}</TextField>}
    </Stack>
    <Typography variant="body2" color="text.secondary">{t('arena.paretoHelp')}</Typography>
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: 'minmax(0, 1fr)', lg: 'repeat(2, minmax(0, 1fr))' }, gap: 2, '& > *': { minWidth: 0 } }}>
      {(['evm', 'aclr'] as const).flatMap(metric => (['parameters', 'ops'] as const).map(cost => {
        const front = paretoFront(points, cost, metric)
        const names = new Set(points.map(p => p.row.backbone))
        const traces: PlotTrace[] = backbones.filter(name => names.has(name)).map(name => {
          const samples = points.filter(p => p.row.backbone === name && p.eligible)
          const index = backbones.indexOf(name)
          return { x: samples.map(p => p[cost]), y: samples.map(p => p[metric]), name, type: 'scatter', mode: 'markers',
            marker: { size: 7, color: colors.chart[index % colors.chart.length], symbol: seriesSymbol(index), opacity: .65 },
            error_y: { type: 'data', array: samples.map(p => metric === 'evm' ? p.evmStd : p.aclrStd), visible: true, thickness: 1, width: 2 },
            text: samples.map(p => `${escape(label(p))}<br>P: ${p.parameters} · OPs: ${p.ops}<br>EVM: ${number(p.evm)} dB · ACLR: ${number(p.aclr)} dBc<br>FoM: ${number(p.budget.score ?? 0)} dB`),
            hovertemplate: '%{text}<extra></extra>' }
        })
        const invalid = points.filter(p => !p.eligible)
        traces.push({ x: invalid.map(p => p[cost]), y: invalid.map(p => p[metric]), name: t('arena.unranked'), mode: 'markers', marker: { size: 6, symbol: 'cross', color: '#888888' }, text: invalid.map(p => escape(label(p))), hovertemplate: '%{text}<extra></extra>' })
        traces.push({ x: front.map(p => p[cost]), y: front.map(p => p[metric]), name: t('arena.paretoFront'), mode: 'lines+markers',
          line: { color: colors.chart[0], width: 2, dash: 'dash' }, marker: { size: 11, symbol: 'diamond', color: colors.chart[0] },
          text: front.map(p => escape(label(p))), hovertemplate: '%{text}<br>%{x} · %{y:.2f}<extra></extra>' })
        const xTitle = t(cost === 'parameters' ? 'arena.parameters' : 'arena.opsPerSample')
        const yTitle = metric === 'evm' ? 'EVM (dB)' : 'ACLR (dBc)'
        return <PlotlyChart key={`${metric}-${cost}`} data-testid={`arena-${metric}-${cost}`} title={`${yTitle} vs. ${xTitle}`} traces={traces} height={370}
          viewKey={`${protocol}:${condition}:${rows[0]?.execution_semantics}:${metric}:${cost}`}
          layout={{ xaxis: { type: 'log', title: { text: xTitle }, tickmode: 'array', tickvals: costTicks, ticktext: costTicks.map(number) }, yaxis: { title: { text: yTitle } }, legend: { orientation: 'h', x: 0, y: -.28, maxheight: .2, font: { size: 10 } }, margin: { l: 65, r: 15, t: 20, b: 110 } }} />
      }))}
    </Box>
  </Stack>
}
