import ErrorOutlineIcon from '@mui/icons-material/ErrorOutlineOutlined'
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined'
import WarningAmberIcon from '@mui/icons-material/WarningAmber'
import Chip from '@mui/material/Chip'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import type { DiagnosticItem as Item, Severity } from '@/api/types'
import { formatNumber, message, t } from '@/i18n'
import { useStudioColors } from '@/theme'

const SEVERITY: Record<Severity, { Icon: typeof InfoOutlinedIcon }> = {
  info: { Icon: InfoOutlinedIcon },
  warning: { Icon: WarningAmberIcon },
  error: { Icon: ErrorOutlineIcon },
}

/** One Dataset Doctor finding: severity, evidence numbers, suggestion and confidence. */
export function DiagnosticItem({ item }: { item: Item }) {
  const color = useStudioColors().status[item.severity]
  const { Icon } = SEVERITY[item.severity]
  const evidence = Object.entries(item.evidence ?? {})
  return (
    <Paper component="article" aria-label={message(item.title)} data-severity={item.severity} data-code={item.code} sx={{ p: 2, borderLeft: `4px solid ${color}` }}>
      <Stack sx={{ alignItems: 'center', flexWrap: 'wrap' }} direction="row" spacing={1} useFlexGap>
        <Icon sx={{ color }} aria-hidden />
        <Typography variant="h3" component="h4">
          {message(item.title)}
        </Typography>
        <Chip size="small" label={message(item.severity)} sx={{ color, borderColor: color }} variant="outlined" />
        {item.blocking && <Chip size="small" color="error" label={t('diag.blocking')} />}
        {typeof item.confidence === 'number' && <Chip size="small" variant="outlined" label={t('diag.confidence', { value: item.confidence.toFixed(2) })} />}
        <Typography variant="caption" color="text.secondary" sx={{ ml: 'auto' }}>
          <code>{item.code}</code>
        </Typography>
      </Stack>
      <Typography sx={{ mt: 1 }}>{message(item.message)}</Typography>
      {evidence.length > 0 && (
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }} component="p">
          <strong>{t('diag.evidence')}: </strong>
          {evidence.map(([k, v]) => (
            <span key={k} style={{ marginRight: 12 }} data-evidence-key={k}>
              <span style={{ fontWeight: 600 }}>{message(k)}</span>={typeof v === 'number' ? formatNumber(v, { maximumSignificantDigits: 6 }) : message(String(v))}
            </span>
          ))}
        </Typography>
      )}
      {item.suggestion && (
        <Typography variant="body2" sx={{ mt: 1 }}>
          <strong>{t('diag.suggestion')}: </strong>
          {message(item.suggestion)}
        </Typography>
      )}
    </Paper>
  )
}
