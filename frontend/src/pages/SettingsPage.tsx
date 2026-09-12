import Chip from '@mui/material/Chip'
import Grid from '@mui/material/Grid'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import { useCapabilities } from '@/api/hooks'
import { t } from '@/i18n'
import { ErrorState, LoadingState } from '@/components/StateBlock'

export function SettingsPage() {
  const caps = useCapabilities()
  if (caps.isPending) return <LoadingState />
  if (caps.isError) return <ErrorState error={caps.error} onRetry={() => void caps.refetch()} />
  const c = caps.data
  return (
    <Stack spacing={2}>
      <Typography variant="h1">{t('settings.title')}</Typography>
      <Paper sx={{ p: 2 }}>
        <Typography variant="h2" gutterBottom>
          {t('settings.workspace')}
        </Typography>
        <code style={{ wordBreak: 'break-all' }}>{c.workspace}</code>
      </Paper>
      <Paper sx={{ p: 2 }}>
        <Typography variant="h2" gutterBottom>
          {t('settings.devices')}
        </Typography>
        <Typography variant="body2" color="text.secondary" gutterBottom>
          {t('settings.devices.note')}
        </Typography>
        <Grid container spacing={2}>
          {c.devices.map((d) => (
            <Grid key={d.device} size={{ xs: 12, md: 4 }}>
              <Paper sx={{ p: 2 }} data-device={d.device}>
                <Stack direction="row" spacing={1} sx={{ alignItems: 'center', mb: 1 }}>
                  <Typography variant="h3" component="h3">
                    {d.device}
                  </Typography>
                  <Chip size="small" color={d.detected ? 'success' : 'default'} variant="outlined" label={d.detected ? t('settings.devices.detected') : t('settings.devices.notDetected')} />
                </Stack>
                {d.name && <Typography variant="body2">{d.name}</Typography>}
                <Typography variant="caption" color="text.secondary">
                  {t('settings.devices.tested')}: {(d.tested_models ?? []).length > 0 ? (d.tested_models ?? []).join(', ') : t('common.na')}
                </Typography>
              </Paper>
            </Grid>
          ))}
        </Grid>
      </Paper>
      <Paper sx={{ p: 2 }}>
        <Typography variant="h2" gutterBottom>
          {t('settings.about')}
        </Typography>
        <Typography>
          {t('settings.version')}: <code>{c.version}</code>
        </Typography>
        <Typography variant="body2" color="text.secondary">
          {t('settings.contract')}
        </Typography>
      </Paper>
    </Stack>
  )
}
