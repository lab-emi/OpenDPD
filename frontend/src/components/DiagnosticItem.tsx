import ErrorOutlineIcon from '@mui/icons-material/ErrorOutlineOutlined'
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined'
import WarningAmberIcon from '@mui/icons-material/WarningAmber'
import Chip from '@mui/material/Chip'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import type { DiagnosticItem as Item, Severity } from '@/api/types'
import { t } from '@/i18n'
import { tokens } from '@/theme'

const SEVERITY: Record<Severity, { color: string; Icon: typeof InfoOutlinedIcon }> = {
  info: { color: tokens.color.status.info, Icon: InfoOutlinedIcon },
  warning: { color: tokens.color.status.warning, Icon: WarningAmberIcon },
  error: { color: tokens.color.status.error, Icon: ErrorOutlineIcon },
}

/** One Dataset Doctor finding: severity, evidence numbers, suggestion and confidence. */
export function DiagnosticItem({ item }: { item: Item }) {
  const { color, Icon } = SEVERITY[item.severity]
  const evidence = Object.entries(item.evidence)
  return (
    <Paper component="article" aria-label={item.title} data-severity={item.severity} data-code={item.code} sx={{ p: 2, borderLeft: `4px solid ${color}` }}>
      <Stack sx={{ alignItems: 'center', flexWrap: 'wrap' }} direction="row" spacing={1} useFlexGap>
        <Icon sx={{ color }} aria-hidden />
        <Typography variant="h3" component="h4">
          {item.title}
        </Typography>
        <Chip size="small" label={item.severity} sx={{ color, borderColor: color }} variant="outlined" />
        {item.blocking && <Chip size="small" color="error" label={t('diag.blocking')} />}
        {typeof item.confidence === 'number' && <Chip size="small" variant="outlined" label={t('diag.confidence', { value: item.confidence.toFixed(2) })} />}
        <Typography variant="caption" color="text.secondary" sx={{ ml: 'auto' }}>
          <code>{item.code}</code>
        </Typography>
      </Stack>
      <Typography sx={{ mt: 1 }}>{item.message}</Typography>
      {evidence.length > 0 && (
        <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }} component="dl">
          <strong>{t('diag.evidence')}: </strong>
          {evidence.map(([k, v]) => (
            <span key={k} style={{ marginRight: 12 }}>
              <dt style={{ display: 'inline' }}>{k}</dt>=<dd style={{ display: 'inline', margin: 0 }}>{String(v)}</dd>
            </span>
          ))}
        </Typography>
      )}
      {item.suggestion && (
        <Typography variant="body2" sx={{ mt: 1 }}>
          <strong>{t('diag.suggestion')}: </strong>
          {item.suggestion}
        </Typography>
      )}
    </Paper>
  )
}
