import Alert from '@mui/material/Alert'
import Button from '@mui/material/Button'
import Box from '@mui/material/Box'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useState, type FormEvent, type ReactNode } from 'react'
import { WEB_MODE, bootstrapSession, clearWebSession, loadSession } from '@/api/client'
import { WebSessionStart } from '@/components/WebSessionStart'
import { t } from '@/i18n'
import { ErrorState, LoadingState } from '@/components/StateBlock'
import { ReportBugsButton } from '@/components/ReportBugsButton'
import { useWebActivity } from '@/components/useWebActivity'

/** Renders children only with a valid local session; otherwise explains how to get one. */
export function SessionGate({ children }: { children: ReactNode }) {
  const qc = useQueryClient()
  const session = useQuery({ queryKey: ['session'], queryFn: ({ signal }) => loadSession(signal), retry: false, staleTime: Infinity })
  useWebActivity(session.data?.authenticated ?? false)
  const [token, setToken] = useState('')
  const [invalid, setInvalid] = useState(false)
  const [busy, setBusy] = useState(false)
  const expiresAt = session.data?.expires_at
  const idleExpiresAt = session.data?.idle_expires_at
  const serverTime = session.data?.server_time
  useEffect(() => {
    if (!WEB_MODE) return
    let stopped = false
    let timer: number | undefined
    const controller = new AbortController()
    const reset = () => {
      clearWebSession()
      void qc.cancelQueries()
      qc.removeQueries({ predicate: (query) => query.queryKey[0] !== 'session' })
      qc.setQueryData(['session'], { authenticated: false, mode: 'web', version: '' })
    }
    window.addEventListener('opendpd-session-expired', reset)
    const deadlines = [expiresAt, idleExpiresAt].filter(Boolean).map(value => Date.parse(value!)).filter(Number.isFinite)
    // Another workspace on this IP may have renewed the shared deadline. Ask
    // the server before discarding credentials; a read itself does not renew it.
    const verify = async () => {
      try {
        const info = await loadSession(controller.signal)
        if (!stopped) { if (info.authenticated) qc.setQueryData(['session'], info); else reset() }
      } catch {
        if (!stopped) timer = window.setTimeout(() => void verify(), 30_000)
      }
    }
    if (deadlines.length) {
      const now = serverTime ? Date.parse(serverTime) : Date.now()
      timer = window.setTimeout(() => void verify(), Math.max(0, Math.min(...deadlines) - now))
    }
    return () => { stopped = true; controller.abort(); window.removeEventListener('opendpd-session-expired', reset); window.clearTimeout(timer) }
  }, [qc, expiresAt, idleExpiresAt, serverTime])

  if (session.isPending) return <LoadingState />
  const reportBugs = <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}><ReportBugsButton /></Box>
  if (session.isError) return <Stack sx={{ maxWidth: 560, mx: 'auto', p: 3 }} spacing={2}>{reportBugs}<ErrorState error={session.error} onRetry={() => void session.refetch()} /></Stack>
  if (session.data.authenticated) return <>{children}</>

  if (WEB_MODE) return <WebSessionStart />

  const submit = async (e: FormEvent) => {
    e.preventDefault()
    setBusy(true)
    try {
      const info = await bootstrapSession(token.trim())
      qc.setQueryData(['session'], info)
      setInvalid(false)
    } catch {
      setInvalid(true)
    } finally {
      setBusy(false)
    }
  }
  return (
    <Paper component="form" onSubmit={(e) => void submit(e)} sx={{ maxWidth: 560, m: '10vh auto', p: 3 }}>
      <Stack spacing={2}>
        {reportBugs}
        <Typography variant="h1">{t('session.required.title')}</Typography>
        <Typography>{t('session.required.body')}</Typography>
        {invalid && <Alert severity="error">{t('session.token.invalid')}</Alert>}
        <TextField label={t('session.token.label')} value={token} onChange={(e) => setToken(e.target.value)} required autoFocus slotProps={{ htmlInput: { 'aria-label': t('session.token.label') } }} />
        <Button type="submit" variant="contained" disabled={busy || !token.trim()}>
          {t('session.token.submit')}
        </Button>
      </Stack>
    </Paper>
  )
}
