import Alert from '@mui/material/Alert'
import Button from '@mui/material/Button'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useState, type FormEvent, type ReactNode } from 'react'
import { bootstrapSession, loadSession } from '@/api/client'
import { t } from '@/i18n'
import { ErrorState, LoadingState } from '@/components/StateBlock'

/** Renders children only with a valid local session; otherwise explains how to get one. */
export function SessionGate({ children }: { children: ReactNode }) {
  const qc = useQueryClient()
  const session = useQuery({ queryKey: ['session'], queryFn: loadSession, retry: false, staleTime: Infinity })
  const [token, setToken] = useState('')
  const [invalid, setInvalid] = useState(false)
  const [busy, setBusy] = useState(false)

  if (session.isPending) return <LoadingState />
  if (session.isError) return <ErrorState error={session.error} onRetry={() => void session.refetch()} />
  if (session.data.authenticated) return <>{children}</>

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
