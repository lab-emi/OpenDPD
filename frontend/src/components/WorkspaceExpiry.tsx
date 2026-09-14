import AutoDeleteOutlinedIcon from '@mui/icons-material/AutoDeleteOutlined'
import Box from '@mui/material/Box'
import Tooltip from '@mui/material/Tooltip'
import Typography from '@mui/material/Typography'
import { formatDateTime, t } from '@/i18n'

/** A single timestamp, with a timezone, in the persistent top bar. */
export function WorkspaceExpiry({ expiresAt }: { expiresAt: string }) {
  const date = new Date(expiresAt)
  if (!Number.isFinite(date.getTime())) return null
  const utc = date.toISOString().replace('T', ' ').replace(/\.\d{3}Z$/, ' UTC')
  return <Tooltip title={`${t('web.expiryHelp')} ${t('web.localTime', { date: formatDateTime(date) })}`}>
    <Box sx={{ display: 'flex', alignItems: 'center', gap: .75, minWidth: 0, flexBasis: { xs: '100%', sm: 'auto' }, order: { xs: 2, sm: 0 }, pb: { xs: 1, sm: 0 } }} data-testid="workspace-expiry">
      <AutoDeleteOutlinedIcon sx={{ fontSize: 18, flexShrink: 0, color: 'text.secondary' }} />
      <Box sx={{ minWidth: 0, display: { xs: 'flex', sm: 'block' }, alignItems: 'center', gap: .75 }}>
        <Typography variant="caption" component="div" color="text.secondary" sx={{ lineHeight: 1.2, whiteSpace: 'nowrap', display: { xs: 'none', sm: 'block' } }}>{t('web.cleanupAt')}</Typography>
        <Typography variant="caption" component="div" color="text.secondary" sx={{ whiteSpace: 'nowrap', display: { xs: 'block', sm: 'none' } }}>{t('web.cleanupShort')}</Typography>
        <Typography component="time" dateTime={date.toISOString()} sx={{ fontSize: 12, fontVariantNumeric: 'tabular-nums', whiteSpace: 'nowrap' }}>{utc}</Typography>
      </Box>
    </Box>
  </Tooltip>
}
