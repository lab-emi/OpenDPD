import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward'
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward'
import Card from '@mui/material/Card'
import CardContent from '@mui/material/CardContent'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import type { MetricValue } from '@/api/types'
import { t } from '@/i18n'

export function formatMetric(m: MetricValue): string {
  if (m.status === 'ok' && typeof m.value === 'number') return `${m.value.toFixed(2)} ${m.unit}`.trim()
  return t(
    m.status === 'not_applicable'
      ? 'metric.notApplicable'
      : m.status === 'missing_reference'
        ? 'metric.missingReference'
        : m.status === 'invalid'
          ? 'metric.invalid'
          : 'metric.failed',
  )
}

/** One MetricValue: value with unit, better-direction, and the reason when there is no value (never "0"). */
export function MetricCard({ metric }: { metric: MetricValue }) {
  const ok = metric.status === 'ok' && typeof metric.value === 'number'
  const direction = metric.better === 'lower' ? t('metric.lowerBetter') : t('metric.higherBetter')
  const Arrow = metric.better === 'lower' ? ArrowDownwardIcon : ArrowUpwardIcon
  return (
    <Card component="section" aria-label={metric.name} data-metric={metric.name} data-status={metric.status} sx={{ minWidth: 150 }}>
      <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
        <Typography variant="overline" component="h3" sx={{ lineHeight: 1.4 }}>
          {metric.name}
        </Typography>
        <Typography variant="h2" component="p" sx={{ fontVariantNumeric: 'tabular-nums', color: ok ? 'text.primary' : 'text.secondary' }}>
          {formatMetric(metric)}
        </Typography>
        <Stack sx={{ alignItems: 'center' }} direction="row" spacing={0.5}>
          <Arrow sx={{ fontSize: 14 }} aria-hidden />
          <Typography variant="caption" color="text.secondary">
            {direction}
          </Typography>
        </Stack>
        {!ok && metric.reason && (
          <Typography variant="caption" color="text.secondary" component="p" sx={{ mt: 0.5 }}>
            {metric.reason}
          </Typography>
        )}
      </CardContent>
    </Card>
  )
}
