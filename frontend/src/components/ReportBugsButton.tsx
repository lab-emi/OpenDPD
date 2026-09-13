import BugReportOutlinedIcon from '@mui/icons-material/BugReportOutlined'
import OpenInNewIcon from '@mui/icons-material/OpenInNew'
import Button from '@mui/material/Button'
import Dialog from '@mui/material/Dialog'
import DialogActions from '@mui/material/DialogActions'
import DialogContent from '@mui/material/DialogContent'
import DialogContentText from '@mui/material/DialogContentText'
import DialogTitle from '@mui/material/DialogTitle'
import IconButton from '@mui/material/IconButton'
import TextField from '@mui/material/TextField'
import Tooltip from '@mui/material/Tooltip'
import { useId, useRef, useState } from 'react'
import { version } from '../../package.json'
import { t } from '@/i18n'

const NEW_ISSUE = 'https://github.com/lab-emi/OpenDPD/issues/new'

/** Only the user's description and the public app version leave Studio. */
function issueUrl(description: string): string {
  const url = new URL(NEW_ISSUE)
  url.searchParams.set('title', '[Bug] OpenDPD Studio')
  url.searchParams.set('body', `${description.trim()}\n\n---\nOpenDPD Studio ${version}`)
  return url.href
}

export function ReportBugsButton({ compact = false }: { compact?: boolean }) {
  const [open, setOpen] = useState(false)
  const [description, setDescription] = useState('')
  const trigger = useRef<HTMLButtonElement>(null)
  const id = useId()
  const href = issueUrl(description)
  // Account for encoded Unicode, not just the textarea's character count.
  const tooLong = href.length > 7500
  const ready = description.trim().length > 0 && !tooLong
  const show = (event: React.MouseEvent<HTMLButtonElement>) => { event.currentTarget.blur(); setOpen(true) }
  const label = t('bugs.report')
  return <>
    {compact ? <Tooltip title={label}><IconButton ref={trigger} aria-label={label} onClick={show} color="primary" sx={{ width: 40, height: 40, flexShrink: 0 }}><BugReportOutlinedIcon fontSize="small" /></IconButton></Tooltip>
      : <Button ref={trigger} size="small" variant="outlined" startIcon={<BugReportOutlinedIcon />} onClick={show} sx={{ whiteSpace: 'nowrap', flexShrink: 0 }}>{label}</Button>}
    <Dialog open={open} onClose={() => setOpen(false)} maxWidth="sm" fullWidth disableRestoreFocus aria-labelledby={`${id}-title`} aria-describedby={`${id}-help`} slotProps={{ transition: { onExited: () => trigger.current?.focus() } }}>
      <DialogTitle id={`${id}-title`}>{label}</DialogTitle>
      <DialogContent>
        <DialogContentText id={`${id}-help`} sx={{ mb: 2 }}>{t('bugs.help')}</DialogContentText>
        <TextField autoFocus fullWidth multiline minRows={6} maxRows={12} label={t('bugs.description')} placeholder={t('bugs.placeholder')} value={description} onChange={(event) => setDescription(event.target.value)} error={tooLong} helperText={tooLong ? t('bugs.tooLong') : undefined} />
      </DialogContent>
      <DialogActions sx={{ flexWrap: 'wrap', gap: 1, px: 3, pb: 2 }}>
        <Button onClick={() => setOpen(false)}>{t('form.cancel')}</Button>
        <Button component="a" href={ready ? href : undefined} target="_blank" rel="noopener noreferrer" variant="contained" disabled={!ready} endIcon={<OpenInNewIcon />} onClick={() => setOpen(false)}>{t('bugs.continue')}</Button>
      </DialogActions>
    </Dialog>
  </>
}
