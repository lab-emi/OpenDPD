import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward'
import ArrowUpwardIcon from '@mui/icons-material/ArrowUpward'
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined'
import Card from '@mui/material/Card'
import CardContent from '@mui/material/CardContent'
import IconButton from '@mui/material/IconButton'
import Stack from '@mui/material/Stack'
import Tooltip from '@mui/material/Tooltip'
import Typography from '@mui/material/Typography'
import type { MetricDefinition, MetricValue } from '@/api/types'
import { message, t } from '@/i18n'

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

/** One MetricValue: value with unit, better-direction, the registry definition, and the reason when there is no value (never "0"). */
export function MetricCard({ metric, definition }: { metric: MetricValue; definition?: MetricDefinition }) {
  const ok = metric.status === 'ok' && typeof metric.value === 'number'
  const direction = metric.better === 'lower' ? t('metric.lowerBetter') : t('metric.higherBetter')
  const Arrow = metric.better === 'lower' ? ArrowDownwardIcon : ArrowUpwardIcon
  return (
    <Card component="section" aria-label={metric.name} data-metric={metric.name} data-status={metric.status} sx={{ minWidth: 0, height: '100%' }}>
      <CardContent sx={{ py: 1.5, '&:last-child': { pb: 1.5 } }}>
        <Stack direction="row" sx={{ alignItems: 'center', justifyContent: 'space-between' }}>
          <Typography variant="overline" component="h3" sx={{ lineHeight: 1.4, minWidth: 0, overflowWrap: 'anywhere' }}>
            {message(definition?.display_name ?? metric.name)}
          </Typography>
          {definition && (
            <Tooltip
              title={
                <span>
                  <strong>{t('metric.formula')}:</strong> {message(definition.formula)}
                  <br />
                  <strong>{t('metric.aggregation')}:</strong> {message(definition.aggregation)}
                  {definition.notes ? (
                    <>
                      <br />
                      {message(definition.notes)}
                    </>
                  ) : null}
                </span>
              }
            >
              <IconButton size="small" sx={{ flexShrink: 0 }} aria-label={t('metric.definition', { name: metric.name })}>
                <InfoOutlinedIcon fontSize="inherit" />
              </IconButton>
            </Tooltip>
          )}
        </Stack>
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
            {message(metric.reason)}
          </Typography>
        )}
      </CardContent>
    </Card>
  )
}
