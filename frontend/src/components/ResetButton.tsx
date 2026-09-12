import RestartAltIcon from '@mui/icons-material/RestartAlt'
import SettingsBackupRestoreIcon from '@mui/icons-material/SettingsBackupRestore'
import Alert from '@mui/material/Alert'
import Button from '@mui/material/Button'
import Dialog from '@mui/material/Dialog'
import DialogActions from '@mui/material/DialogActions'
import DialogContent from '@mui/material/DialogContent'
import DialogContentText from '@mui/material/DialogContentText'
import DialogTitle from '@mui/material/DialogTitle'
import Snackbar from '@mui/material/Snackbar'
import Tooltip from '@mui/material/Tooltip'
import IconButton from '@mui/material/IconButton'
import { useId, useRef, useState } from 'react'
import { t } from '@/i18n'

/** The prompt stays available during a save, but confirmation waits for it to finish. */
export function ResetButton({ onReset, disabled = false, scope = 'page', detail, compact = false }: { onReset: () => void; disabled?: boolean; scope?: 'page' | 'studio'; detail?: string; compact?: boolean }) {
  const [open, setOpen] = useState(false)
  const [completed, setCompleted] = useState(false)
  const trigger = useRef<HTMLButtonElement>(null)
  const id = useId()
  const label = t(scope === 'studio' ? 'reset.studio' : 'reset.page')
  const Icon = scope === 'studio' ? SettingsBackupRestoreIcon : RestartAltIcon
  const show = (event: React.MouseEvent<HTMLButtonElement>) => { event.currentTarget.blur(); setCompleted(false); setOpen(true) }
  const restoreFocus = () => {
    // A global reset opens the dataset guide. Keep focus in that new dialog.
    if (trigger.current && !trigger.current.closest('[aria-hidden="true"]')) trigger.current.focus()
  }
  return <>
    {compact ? <Tooltip title={label}><IconButton ref={trigger} aria-label={label} onClick={show} sx={{ flexShrink: 0, width: 40, height: 40 }}><Icon fontSize="small" /></IconButton></Tooltip>
      : <Button ref={trigger} size="small" color="inherit" startIcon={<Icon />} onClick={show} sx={{ flexShrink: 0 }}>{label}</Button>}
    <Dialog open={open} onClose={() => setOpen(false)} maxWidth="sm" fullWidth disableRestoreFocus slotProps={{ transition: { onExited: restoreFocus } }} aria-labelledby={`${id}-title`} aria-describedby={`${id}-body`}>
      <DialogTitle id={`${id}-title`}>{t(scope === 'studio' ? 'reset.studio.title' : 'reset.page.title')}</DialogTitle>
      <DialogContent id={`${id}-body`}>
        <Alert severity="warning" sx={{ mb: 2 }}>{t('reset.warning')}</Alert>
        <DialogContentText>{t(scope === 'studio' ? 'reset.studio.body' : 'reset.page.body')}</DialogContentText>
        {detail && <DialogContentText sx={{ mt: 1 }}>{detail}</DialogContentText>}
        {disabled && <Alert severity="info" sx={{ mt: 2 }}>{t('reset.busy')}</Alert>}
      </DialogContent>
      <DialogActions>
        <Button onClick={() => setOpen(false)} autoFocus>{t('form.cancel')}</Button>
        <Button variant="contained" color="warning" disabled={disabled} onClick={() => { setOpen(false); onReset(); setCompleted(true) }}>{t(scope === 'page' ? 'workflow.next' : 'reset.confirm')}</Button>
      </DialogActions>
    </Dialog>
    <Snackbar open={completed} autoHideDuration={4500} onClose={() => setCompleted(false)} anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}>
      <Alert severity="success" onClose={() => setCompleted(false)}>{t('reset.done')}</Alert>
    </Snackbar>
  </>
}
