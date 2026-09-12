import List from '@mui/material/List'
import ListItem from '@mui/material/ListItem'
import ListItemText from '@mui/material/ListItemText'
import Typography from '@mui/material/Typography'
import type { RunEvent, RunStatus, RunView } from '@/api/types'
import { formatTime, message, t } from '@/i18n'
import { StatusChip } from './StatusChip'

const STATUSES: ReadonlySet<string> = new Set<RunStatus>(['queued', 'running', 'succeeded', 'failed', 'cancel_requested', 'cancelled', 'interrupted'])

/** Status transitions (from `status` events) with timestamps, plus heartbeat count. */
export function RunTimeline({ run, statusEvents, heartbeats }: { run: RunView; statusEvents: RunEvent[]; heartbeats: number }) {
  const transitions = statusEvents
    .map((e) => ({ seq: e.seq, ts: e.ts, to: String(e.payload?.['to'] ?? ''), reason: e.payload?.['reason'] }))
    .filter((x) => STATUSES.has(x.to))
  const items = transitions.length > 0 ? transitions : [{ seq: 0, ts: run.created_at, to: run.status, reason: run.status_reason }]
  return (
    <section aria-label={t('run.timeline')}>
      <Typography variant="h3" component="h2" gutterBottom>
        {t('run.timeline')}
      </Typography>
      <List dense disablePadding>
        {items.map((x) => (
          <ListItem key={`${x.seq}-${x.to}`} disableGutters sx={{ gap: 1.5 }}>
            <Typography variant="caption" color="text.secondary" sx={{ minWidth: 88, fontVariantNumeric: 'tabular-nums' }}>
              {formatTime(x.ts)}
            </Typography>
            <StatusChip status={x.to as RunStatus} />
            {typeof x.reason === 'string' && x.reason && <ListItemText secondary={message(x.reason)} sx={{ m: 0 }} />}
          </ListItem>
        ))}
        {heartbeats > 0 && (
          <ListItem disableGutters>
            <Typography variant="caption" color="text.secondary">
              {heartbeats} × {t('timeline.heartbeat')}
              {run.last_heartbeat_at ? ` · ${t('timeline.last', { time: formatTime(run.last_heartbeat_at) })}` : ''}
            </Typography>
          </ListItem>
        )}
      </List>
    </section>
  )
}
