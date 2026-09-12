import ArrowForwardIcon from '@mui/icons-material/ArrowForward'
import CheckCircleIcon from '@mui/icons-material/CheckCircle'
import DatasetIcon from '@mui/icons-material/Dataset'
import UploadFileIcon from '@mui/icons-material/UploadFile'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Dialog from '@mui/material/Dialog'
import DialogActions from '@mui/material/DialogActions'
import DialogContent from '@mui/material/DialogContent'
import DialogTitle from '@mui/material/DialogTitle'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import { t } from '@/i18n'
import type { Theme } from '@mui/material/styles'
import { colorsFor, useStudioColors } from '@/theme'

export const GUIDE_BACKDROP = { sx: { backdropFilter: 'blur(6px)', backgroundColor: (theme: Theme) => colorsFor(theme.palette.mode).backdrop } }

export function DatasetGuide({ onCsv, onBuiltin, onSkip, customDatasets = false }: { onCsv: () => void; onBuiltin: () => void; onSkip: () => void; customDatasets?: boolean }) {
  const colors = useStudioColors()
  return <Dialog open onClose={onSkip} fullWidth maxWidth="sm" aria-labelledby="dataset-guide-title" slotProps={{ backdrop: GUIDE_BACKDROP, paper: { sx: { borderRadius: 3, boxShadow: `0 24px 90px ${colors.backdrop}` } } }}>
    <DialogTitle id="dataset-guide-title" sx={{ pb: 1 }}><Typography component="span" variant="overline" color="primary" sx={{ display: 'block' }}>{t('guide.eyebrow')}</Typography>{t('guide.title')}</DialogTitle>
    <DialogContent>
      <Typography color="text.secondary" sx={{ mb: 2.5 }}>{t(customDatasets ? 'guide.intro' : 'guide.builtin.intro')}</Typography>
      <Stack spacing={1.5} sx={{ mb: 3 }}>
        {(customDatasets ? ['guide.file', 'guide.split', 'guide.review'] as const : ['guide.builtin.choose', 'guide.builtin.inspect', 'guide.builtin.configure'] as const).map((key, index) => <Stack key={key} direction="row" spacing={1.5} sx={{ alignItems: 'center' }}>
          <Box sx={{ width: 28, height: 28, flexShrink: 0, borderRadius: '50%', bgcolor: colors.selected, color: 'primary.main', display: 'grid', placeItems: 'center', fontWeight: 700 }}>{index + 1}</Box><Typography variant="body2">{t(key)}</Typography>
        </Stack>)}
      </Stack>
      <Stack spacing={1.25}>
        <Button variant={customDatasets ? 'contained' : 'outlined'} disabled={!customDatasets} size="large" startIcon={<UploadFileIcon />} endIcon={<ArrowForwardIcon />} onClick={onCsv}>{t('guide.csv')}{!customDatasets && ` · ${t('common.comingSoon')}`}</Button>
        <Button variant={customDatasets ? 'outlined' : 'contained'} size="large" startIcon={<DatasetIcon />} onClick={onBuiltin}>{t('guide.builtin')}</Button>
      </Stack>
    </DialogContent>
    <DialogActions><Button onClick={onSkip}>{t('guide.skip')}</Button></DialogActions>
  </Dialog>
}

export function DatasetReadyGuide({ onClose }: { onClose: () => void }) {
  return <Dialog open onClose={onClose} fullWidth maxWidth="sm" aria-labelledby="dataset-ready-title" slotProps={{ backdrop: GUIDE_BACKDROP }}>
    <DialogTitle id="dataset-ready-title"><Stack component="span" direction="row" spacing={1} sx={{ alignItems: 'center' }}><CheckCircleIcon color="success" /><span>{t('guide.ready.title')}</span></Stack></DialogTitle>
    <DialogContent><Typography>{t('guide.ready.body')}</Typography></DialogContent>
    <DialogActions><Button onClick={onClose}>{t('guide.skip')}</Button><Button onClick={onClose} variant="contained" endIcon={<ArrowForwardIcon />}>{t('guide.ready.action')}</Button></DialogActions>
  </Dialog>
}
