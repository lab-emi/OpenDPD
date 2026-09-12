import Alert from '@mui/material/Alert'
import Button from '@mui/material/Button'
import Chip from '@mui/material/Chip'
import Dialog from '@mui/material/Dialog'
import DialogActions from '@mui/material/DialogActions'
import DialogContent from '@mui/material/DialogContent'
import DialogTitle from '@mui/material/DialogTitle'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import { useBuiltinDatasets } from '@/api/datasets'
import { useDatasets, useImportBuiltin } from '@/api/hooks'
import { formatNumber, message, t } from '@/i18n'
import { ErrorState, LoadingState } from './StateBlock'
import { GUIDE_BACKDROP } from './DatasetGuide'

export function BuiltinDatasetDialog({ onClose, onSelected, guided = false, onSkipGuide }: { onClose: () => void; onSelected: (id: string) => void; guided?: boolean; onSkipGuide?: () => void }) {
  const catalog = useBuiltinDatasets()
  const datasets = useDatasets()
  const register = useImportBuiltin()
  return <Dialog open onClose={register.isPending ? undefined : onClose} fullWidth maxWidth="md" aria-labelledby="builtin-title" slotProps={{ backdrop: guided ? GUIDE_BACKDROP : undefined }}>
    <DialogTitle id="builtin-title">{t('datasets.builtin.title')}</DialogTitle>
    <DialogContent dividers>
      {guided && <Alert severity="info" sx={{ mb: 2 }}>{t('guide.builtin.help')}</Alert>}
      <Typography color="text.secondary" sx={{ mb: 2 }}>{t('datasets.builtin.help')}</Typography>
      {catalog.isPending ? <LoadingState /> : catalog.isError ? <ErrorState error={catalog.error} onRetry={() => void catalog.refetch()} /> : <Stack spacing={1.5}>
        {catalog.data.map((entry) => {
          const registered = datasets.data?.find((d) => d.source.kind === 'builtin' && d.source.name === entry.name && d.raw_sha256 === entry.raw_sha256)
          return <Stack key={entry.name} direction="row" spacing={2} sx={{ p: 2, border: 1, borderColor: 'divider', borderRadius: 1.5, alignItems: 'center' }}>
            <Stack spacing={0.5} sx={{ flex: 1, minWidth: 0 }}>
              <Stack direction="row" spacing={1} sx={{ alignItems: 'center' }}><Typography sx={{ fontWeight: 700 }}>{entry.name}</Typography><Chip size="small" label={message(entry.origin)} color={entry.origin === 'synthetic' ? 'warning' : 'default'} /></Stack>
              {entry.origin === 'synthetic' && <Typography variant="body2">{message(entry.description)}</Typography>}
              <Typography variant="caption" color="text.secondary">{formatNumber(entry.n_samples ?? 0)} {t('datasets.columns.samples')} · {entry.signal.sample_rate_hz ? `${entry.signal.sample_rate_hz / 1e6} MHz` : t('common.na')} · {entry.signal.modulation ?? t('common.na')} · {entry.dataset_format}</Typography>
              {entry.problem && <Alert severity="error">{entry.problem}</Alert>}
            </Stack>
            <Button variant="outlined" disabled={!!entry.problem || register.isPending} aria-label={`${registered ? t('datasets.builtin.open') : t('datasets.builtin.add')} ${entry.name}`} onClick={() => registered ? onSelected(registered.dataset_id) : register.mutate(entry.name, { onSuccess: (d) => onSelected(d.dataset_id) })}>
              {registered ? t('datasets.builtin.open') : t('datasets.builtin.add')}
            </Button>
          </Stack>
        })}
      </Stack>}
      {register.isError && <ErrorState error={register.error} />}
    </DialogContent>
    <DialogActions>{guided && <Button onClick={onSkipGuide} disabled={register.isPending}>{t('guide.skip')}</Button>}<Button onClick={onClose} disabled={register.isPending}>{t('form.cancel')}</Button></DialogActions>
  </Dialog>
}
