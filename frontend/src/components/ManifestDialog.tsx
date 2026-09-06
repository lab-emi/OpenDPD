import Button from '@mui/material/Button'
import Dialog from '@mui/material/Dialog'
import DialogActions from '@mui/material/DialogActions'
import DialogContent from '@mui/material/DialogContent'
import DialogTitle from '@mui/material/DialogTitle'
import Grid from '@mui/material/Grid'
import MenuItem from '@mui/material/MenuItem'
import Stack from '@mui/material/Stack'
import TextField from '@mui/material/TextField'
import { useState } from 'react'
import { useUpdateManifest } from '@/api/datasets'
import type { DatasetManifest, DatasetOrigin } from '@/api/types'
import { t } from '@/i18n'
import { SignalFields, signalFormFrom, signalSpecFrom, type SignalForm } from '@/components/SignalFields'
import { ErrorState } from '@/components/StateBlock'

export const ORIGINS: DatasetOrigin[] = ['measured', 'synthetic', 'unknown']

/** Edits the human-owned part of a manifest (name, origin, notes, signal metadata); mount it fresh per opening. */
export function ManifestDialog({ dataset, onClose }: { dataset: DatasetManifest; onClose: () => void }) {
  const [displayName, setDisplayName] = useState(dataset.display_name)
  const [origin, setOrigin] = useState<DatasetOrigin>(dataset.origin)
  const [notes, setNotes] = useState(dataset.notes ?? '')
  const [signal, setSignal] = useState<SignalForm>(() => signalFormFrom(dataset.signal))
  const update = useUpdateManifest(dataset.dataset_id)
  const save = () => update.mutate({ display_name: displayName.trim() || null, origin, notes: notes.trim() || null, signal: signalSpecFrom(signal, dataset.signal) }, { onSuccess: onClose })
  return (
    <Dialog open onClose={onClose} fullWidth maxWidth="md" aria-labelledby="manifest-title">
      <DialogTitle id="manifest-title">{t('datasets.detail.edit')}</DialogTitle>
      <DialogContent dividers>
        <Stack spacing={2} sx={{ pt: 1 }}>
          <Grid container spacing={2}>
            <Grid size={{ xs: 12, md: 8 }}>
              <TextField fullWidth size="small" label={t('datasets.import.name')} value={displayName} onChange={(e) => setDisplayName(e.target.value)} />
            </Grid>
            <Grid size={{ xs: 12, md: 4 }}>
              <TextField select fullWidth size="small" label={t('datasets.import.origin')} value={origin} onChange={(e) => setOrigin(e.target.value as DatasetOrigin)}>
                {ORIGINS.map((o) => (
                  <MenuItem key={o} value={o}>
                    {o}
                  </MenuItem>
                ))}
              </TextField>
            </Grid>
          </Grid>
          <SignalFields value={signal} onChange={setSignal} />
          <TextField fullWidth multiline minRows={2} size="small" label={t('datasets.detail.notes')} value={notes} onChange={(e) => setNotes(e.target.value)} />
          {update.isError && <ErrorState error={update.error} />}
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>{t('form.cancel')}</Button>
        <Button variant="contained" onClick={save} disabled={update.isPending}>
          {t('datasets.detail.save')}
        </Button>
      </DialogActions>
    </Dialog>
  )
}
