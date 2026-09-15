import LogoutIcon from '@mui/icons-material/Logout'
import Button from '@mui/material/Button'
import Dialog from '@mui/material/Dialog'
import DialogActions from '@mui/material/DialogActions'
import DialogContent from '@mui/material/DialogContent'
import DialogContentText from '@mui/material/DialogContentText'
import DialogTitle from '@mui/material/DialogTitle'
import { useState } from 'react'
import { endWebSession } from '@/api/client'
import { ErrorState } from '@/components/StateBlock'
import { t } from '@/i18n'

export function EndWorkspaceButton() {
  const [open, setOpen] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<unknown>(null)
  return <>
    <Button variant="outlined" startIcon={<LogoutIcon />} onClick={() => setOpen(true)}>{t('web.end.title')}</Button>
    <Dialog open={open} onClose={() => { if (!busy) setOpen(false) }} aria-labelledby="end-workspace-title">
      <DialogTitle id="end-workspace-title">{t('web.end.title')}</DialogTitle>
      <DialogContent><DialogContentText>{t('web.end.body')}</DialogContentText>{error != null && <ErrorState error={error} />}</DialogContent>
      <DialogActions>
        <Button disabled={busy} onClick={() => setOpen(false)}>{t('web.end.keep')}</Button>
        <Button color="error" variant="contained" disabled={busy} onClick={() => {
          setBusy(true); setError(null)
          void endWebSession().catch(setError).finally(() => setBusy(false))
        }}>{t('web.end.confirm')}</Button>
      </DialogActions>
    </Dialog>
  </>
}
