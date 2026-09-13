import Alert from '@mui/material/Alert'
import Button from '@mui/material/Button'
import Box from '@mui/material/Box'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useEffect, useState, type FormEvent, type ReactNode } from 'react'
import { WEB_MODE, bootstrapSession, clearWebSession, createWebSession, loadSession } from '@/api/client'
import { t } from '@/i18n'
import { ErrorState, LoadingState } from '@/components/StateBlock'
import { ReportBugsButton } from '@/components/ReportBugsButton'

/** Renders children only with a valid local session; otherwise explains how to get one. */
export function SessionGate({ children }: { children: ReactNode }) {
  const qc = useQueryClient()
  const session = useQuery({ queryKey: ['session'], queryFn: loadSession, retry: false, staleTime: Infinity })
  const [token, setToken] = useState('')
  const [invalid, setInvalid] = useState(false)
  const [busy, setBusy] = useState(false)
  const [webError, setWebError] = useState<unknown>(null)
  const expiresAt = session.data?.expires_at
  useEffect(() => {
    if (!WEB_MODE) return
    const reset = () => {
      clearWebSession()
      void qc.cancelQueries()
      qc.removeQueries({ predicate: (query) => query.queryKey[0] !== 'session' })
      qc.setQueryData(['session'], { authenticated: false, mode: 'web', version: '' })
    }
    window.addEventListener('opendpd-session-expired', reset)
    const delay = expiresAt ? new Date(expiresAt).getTime() - Date.now() : null
    const timer = delay === null ? undefined : window.setTimeout(reset, Math.max(0, delay))
    return () => { window.removeEventListener('opendpd-session-expired', reset); window.clearTimeout(timer) }
  }, [qc, expiresAt])

  if (session.isPending) return <LoadingState />
  const reportBugs = <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}><ReportBugsButton /></Box>
  if (session.isError) return <Stack sx={{ maxWidth: 560, mx: 'auto', p: 3 }} spacing={2}>{reportBugs}<ErrorState error={session.error} onRetry={() => void session.refetch()} /></Stack>
  if (session.data.authenticated) return <>{children}</>

  if (WEB_MODE) return <Paper sx={{ maxWidth: 560, mx: 'auto', my: '10vh', p: 3 }}>
    <Stack spacing={2}>
      {reportBugs}
      <Typography variant="h1">{t('web.welcome')}</Typography>
      <Typography>{t('web.description')}</Typography>
      <Alert severity="info">{t('web.temporary')}</Alert>
      {webError != null && <ErrorState error={webError} />}
      <Button variant="contained" disabled={busy} onClick={() => {
        setBusy(true); setWebError(null)
        void createWebSession().then((info) => {
          qc.removeQueries({ predicate: (query) => query.queryKey[0] !== 'session' })
          qc.setQueryData(['session'], info)
        })
          .catch(setWebError).finally(() => setBusy(false))
      }}>{t(webError != null ? 'state.error.retry' : 'web.start')}</Button>
    </Stack>
  </Paper>

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
