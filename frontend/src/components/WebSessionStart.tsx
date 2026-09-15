import HourglassEmptyIcon from '@mui/icons-material/HourglassEmpty'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import LinearProgress from '@mui/material/LinearProgress'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import { useQueryClient } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { ApiError, cancelWebQueue, clearWebQueue, createWebSession, hasQueuedWebSession, type WebQueueInfo } from '@/api/client'
import { ErrorState } from '@/components/StateBlock'
import { ReportBugsButton } from '@/components/ReportBugsButton'
import { formatDateTime, formatNumber, t } from '@/i18n'

export function WebSessionStart() {
  const qc = useQueryClient()
  const [running, setRunning] = useState(hasQueuedWebSession)
  const [queue, setQueue] = useState<WebQueueInfo | null>(null)
  const [error, setError] = useState<unknown>(null)
  useEffect(() => {
    if (!running) return
    let stopped = false
    let queued = hasQueuedWebSession()
    let timer: number | undefined
    let request: AbortController | undefined
    const poll = async () => {
      request = new AbortController()
      const timeout = window.setTimeout(() => request?.abort(), 30_000)
      let delay = 5_000
      try {
        const info = await createWebSession(request.signal)
        if (stopped) return
        setError(null)
        if (info.authenticated) {
          qc.removeQueries({ predicate: query => query.queryKey[0] !== 'session' })
          qc.setQueryData(['session'], info)
          return
        }
        if ('status' in info && info.status === 'queued') {
          setQueue(info)
          queued = true
          delay = Math.max(5_000, Math.min(60_000, info.retry_after_seconds * 1_000))
        }
      } catch (failure) {
        if (stopped) return
        if (failure instanceof ApiError && failure.code === 'queue_expired') {
          clearWebQueue()
        } else {
          setError(failure)
          const transient = failure instanceof ApiError && [408, 429, 500, 502, 503, 504, 507].includes(failure.status)
          const refused = failure instanceof ApiError && (!transient || ['session_quota', 'queue_quota'].includes(failure.code))
          if ((!queued && !transient) || refused) {
            setRunning(false)
            return
          }
          delay = Math.max(10_000, failure instanceof ApiError ? failure.retryAfterMs : 0)
        }
      } finally {
        window.clearTimeout(timeout)
      }
      if (!stopped) timer = window.setTimeout(() => void poll(), delay)
    }
    void poll()
    return () => { stopped = true; window.clearTimeout(timer); request?.abort() }
  }, [running, qc])

  return <Paper sx={{ maxWidth: 600, mx: 'auto', my: '10vh', p: { xs: 2.5, sm: 3 } }}>
    <Stack spacing={2}>
      <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}><ReportBugsButton /></Box>
      <Typography variant="h1">{t('web.welcome')}</Typography>
      <Typography>{t('web.description')}</Typography>
      <Alert severity="info">{t('web.temporary')}</Alert>
      {running && <Paper variant="outlined" sx={{ p: 2.5 }} data-testid="workspace-queue">
        <Stack spacing={1.5} role="status" aria-live="polite">
          <Stack direction="row" spacing={1} sx={{ alignItems: 'center' }}><HourglassEmptyIcon color="primary" /><Typography variant="h2">{t(queue ? 'web.queue.title' : 'web.queue.joining')}</Typography></Stack>
          {queue && <>
            <Typography sx={{ fontSize: 40, fontWeight: 650, fontVariantNumeric: 'tabular-nums' }}>{t('web.queue.position', { position: formatNumber(queue.queue_position) })}</Typography>
            <Typography>{t(`web.queue.${queue.reason}`)}</Typography>
            {queue.admission_resumes_at && <Typography variant="body2">{t('web.queue.resumes', { date: formatDateTime(queue.admission_resumes_at) })}</Typography>}
            <Typography variant="body2" color="text.secondary">{t('web.queue.help')}</Typography>
          </>}
          <LinearProgress aria-label={t('web.queue.joining')} />
        </Stack>
      </Paper>}
      {error != null && (running ? <Alert severity="warning">{t('web.queue.retrying')}</Alert> : <ErrorState error={error} />)}
      {running ? <Button variant="outlined" onClick={() => {
        setRunning(false); setQueue(null); setError(null)
        void cancelWebQueue().catch(setError)
      }}>{t('web.queue.leave')}</Button> : <Button variant="contained" onClick={() => { setError(null); setRunning(true) }}>{t(error != null ? 'state.error.retry' : 'web.start')}</Button>}
    </Stack>
  </Paper>
}
