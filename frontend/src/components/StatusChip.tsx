import Chip from '@mui/material/Chip'
import Tooltip from '@mui/material/Tooltip'
import BoltIcon from '@mui/icons-material/Bolt'
import CheckIcon from '@mui/icons-material/Check'
import ErrorOutlineIcon from '@mui/icons-material/ErrorOutlineOutlined'
import HourglassEmptyIcon from '@mui/icons-material/HourglassEmpty'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import StopIcon from '@mui/icons-material/Stop'
import StopCircleOutlinedIcon from '@mui/icons-material/StopCircleOutlined'
import WifiOffIcon from '@mui/icons-material/WifiOff'
import type { RunStatus } from '@/api/types'
import { t } from '@/i18n'
import { useStudioColors, type StatusTone } from '@/theme'

/** Status label + icon + colour (UX spec §3): colour is never the only encoding. */
const STATUS: Record<RunStatus, { tone: StatusTone; Icon: typeof CheckIcon }> = {
  queued: { tone: 'neutral', Icon: HourglassEmptyIcon },
  running: { tone: 'info', Icon: PlayArrowIcon },
  cancel_requested: { tone: 'warning', Icon: StopCircleOutlinedIcon },
  cancelled: { tone: 'neutral', Icon: StopIcon },
  succeeded: { tone: 'success', Icon: CheckIcon },
  failed: { tone: 'error', Icon: ErrorOutlineIcon },
  interrupted: { tone: 'warning', Icon: BoltIcon },
}

export function statusLabel(status: RunStatus): string {
  return t(`status.${status}`)
}

export function StatusChip({ status, stale = false, size = 'small' }: { status: RunStatus; stale?: boolean; size?: 'small' | 'medium' }) {
  const { tone, Icon } = STATUS[status]
  const colors = useStudioColors()
  const color = colors.status[tone]
  const chip = (
    <Chip
      size={size}
      icon={stale ? <WifiOffIcon fontSize="small" aria-hidden /> : <Icon fontSize="small" aria-hidden />}
      label={statusLabel(status)}
      data-status={status}
      data-stale={stale ? 'true' : undefined}
      sx={{ color, borderColor: color, fontWeight: 600, '& .MuiChip-icon': { color } }}
      variant="outlined"
    />
  )
  return stale ? <Tooltip title={t('status.stale')}>{chip}</Tooltip> : chip
}
