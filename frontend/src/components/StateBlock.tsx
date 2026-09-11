import Alert from '@mui/material/Alert'
import AlertTitle from '@mui/material/AlertTitle'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import CircularProgress from '@mui/material/CircularProgress'
import Paper from '@mui/material/Paper'
import Typography from '@mui/material/Typography'
import type { ReactNode } from 'react'
import { ApiError } from '@/api/client'
import { formatTime, t } from '@/i18n'

/** The four non-happy states every list/detail must render (UX spec §2). */

export function LoadingState({ label = t('state.loading') }: { label?: string }) {
  return (
    <Box role="status" aria-live="polite" sx={{ display: 'flex', alignItems: 'center', gap: 1.5, p: 3 }}>
      <CircularProgress size={20} aria-hidden />
      <Typography color="text.secondary">{label}</Typography>
    </Box>
  )
}

export function EmptyState({ title = t('state.empty.title'), body, action }: { title?: string; body?: string; action?: ReactNode }) {
  return (
    <Paper sx={{ p: 3, textAlign: 'center' }} data-state="empty">
      <Typography variant="h3" component="p" gutterBottom>
        {title}
      </Typography>
      {body && (
        <Typography color="text.secondary" sx={{ mb: action ? 2 : 0 }}>
          {body}
        </Typography>
      )}
      {action}
    </Paper>
  )
}

function describeError(error: unknown): { message: string; hint: string | null } {
  if (error instanceof ApiError) return { message: `${error.message} (${error.code})`, hint: error.hint }
  if (error instanceof Error) return { message: error.message, hint: null }
  return { message: String(error), hint: null }
}

export function ErrorState({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const { message, hint } = describeError(error)
  return (
    <Alert
      severity="error"
      data-state="error"
      action={
        onRetry && (
          <Button color="inherit" size="small" onClick={onRetry}>
            {t('state.error.retry')}
          </Button>
        )
      }
    >
      <AlertTitle>{t('state.error.title')}</AlertTitle>
      {message}
      {hint && <Typography variant="body2">{hint}</Typography>}
    </Alert>
  )
}

export function DisconnectedState({ lastUpdate, onRefresh }: { lastUpdate: Date | null; onRefresh: () => void }) {
  const time = lastUpdate ? formatTime(lastUpdate) : t('common.na')
  return (
    <Alert
      severity="warning"
      role="status"
      data-state="disconnected"
      action={
        <Button color="inherit" size="small" onClick={onRefresh}>
          {t('state.disconnected.refresh')}
        </Button>
      }
    >
      <AlertTitle>{t('state.disconnected.title')}</AlertTitle>
      {t('state.disconnected.body', { time })}
    </Alert>
  )
}
