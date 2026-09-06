import Button from '@mui/material/Button'
import Card from '@mui/material/Card'
import CardActions from '@mui/material/CardActions'
import CardContent from '@mui/material/CardContent'
import Chip from '@mui/material/Chip'
import Grid from '@mui/material/Grid'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import { Link as RouterLink } from 'react-router'
import { useCapabilities, useDatasets, useImportBuiltin, useRuns } from '@/api/hooks'
import { t } from '@/i18n'
import { EmptyState, ErrorState, LoadingState } from '@/components/StateBlock'
import { RunTable } from '@/pages/ExperimentsPage'

const EXAMPLE = { name: 'DPA_200MHz', id: 'dpa-200mhz' }

export function HomePage() {
  const caps = useCapabilities()
  const datasets = useDatasets()
  const runs = useRuns()
  const importBuiltin = useImportBuiltin()
  const registered = datasets.data?.some((d) => d.dataset_id === EXAMPLE.id) ?? false
  return (
    <Stack spacing={3}>
      <Typography variant="h1">{t('home.title')}</Typography>
      <Grid container spacing={2}>
        <Grid size={{ xs: 12, md: 6 }}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Typography variant="h2" gutterBottom>
                {t('home.workspace')}
              </Typography>
              {caps.isPending ? <LoadingState /> : caps.isError ? <ErrorState error={caps.error} onRetry={() => void caps.refetch()} /> : <code style={{ wordBreak: 'break-all' }}>{caps.data.workspace}</code>}
            </CardContent>
          </Card>
        </Grid>
        <Grid size={{ xs: 12, md: 6 }}>
          <Card sx={{ height: '100%' }}>
            <CardContent>
              <Stack direction="row" spacing={1} sx={{ alignItems: 'center', mb: 1 }}>
                <Typography variant="h2">{t('home.example.title')}</Typography>
                <Chip size="small" color="success" variant="outlined" label={t('home.example.badge')} />
              </Stack>
              <Typography color="text.secondary">{t('home.example.body')}</Typography>
              {importBuiltin.isError && <ErrorState error={importBuiltin.error} />}
            </CardContent>
            <CardActions>
              {registered ? (
                <Button component={RouterLink} to="/experiments/new" variant="contained">
                  {t('home.newExperiment')}
                </Button>
              ) : (
                <Button variant="contained" onClick={() => importBuiltin.mutate(EXAMPLE.name)} disabled={importBuiltin.isPending || datasets.isPending}>
                  {t('home.example.action')}
                </Button>
              )}
              {registered && <Chip size="small" label={t('home.example.registered')} />}
            </CardActions>
          </Card>
        </Grid>
      </Grid>
      <section aria-labelledby="recent-runs">
        <Typography variant="h2" id="recent-runs" gutterBottom>
          {t('home.recentRuns')}
        </Typography>
        {runs.isPending ? (
          <LoadingState />
        ) : runs.isError ? (
          <ErrorState error={runs.error} onRetry={() => void runs.refetch()} />
        ) : runs.data.length === 0 ? (
          <EmptyState body={t('home.noRuns')} />
        ) : (
          <RunTable runs={runs.data.slice(0, 8)} />
        )}
      </section>
    </Stack>
  )
}
