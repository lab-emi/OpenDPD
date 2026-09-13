import ArrowForwardIcon from '@mui/icons-material/ArrowForward'
import FolderOpenOutlinedIcon from '@mui/icons-material/FolderOpenOutlined'
import GitHubIcon from '@mui/icons-material/GitHub'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Link from '@mui/material/Link'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import { Link as RouterLink } from 'react-router'
import { useCapabilities, useRuns } from '@/api/hooks'
import { t } from '@/i18n'
import { EmptyState, ErrorState, LoadingState } from '@/components/StateBlock'
import { RunTable } from '@/pages/ExperimentsPage'
import { StudioLogo } from '@/components/StudioLogo'
import emi from '@/assets/emi-logo.svg'
import signal from '@/assets/home-signal.svg'

const WORKFLOW = ['data', 'pa', 'dpd'] as const

export function HomePage() {
  const caps = useCapabilities()
  const runs = useRuns()
  return (
    <Stack spacing={3} sx={{ maxWidth: 1360, mx: 'auto' }}>
      <Stack direction={{ xs: 'column', sm: 'row' }} sx={{ gap: 1, alignItems: { sm: 'center' }, justifyContent: 'space-between', minWidth: 0 }}>
        <Typography variant="h1">{t('home.title')}</Typography>
        {caps.isPending ? <Typography variant="caption" color="text.secondary">{t('state.loading')}</Typography> : caps.isError ? (
          <ErrorState error={caps.error} onRetry={() => void caps.refetch()} />
        ) : (
          <Stack direction="row" spacing={1} sx={{ alignItems: 'center', minWidth: 0, maxWidth: { sm: '70%' }, color: 'text.secondary' }}>
            <FolderOpenOutlinedIcon sx={{ fontSize: 16, flexShrink: 0 }} />
            <Typography component="span" variant="caption" sx={{ flexShrink: 0 }}>{t('home.workspace')}</Typography>
            <Typography component="code" variant="caption" noWrap title={caps.data.workspace}>{caps.data.workspace}</Typography>
          </Stack>
        )}
      </Stack>

      <Paper component="section" aria-labelledby="home-introduction" sx={{ overflow: 'hidden', borderRadius: 2, bgcolor: '#0D2233', borderColor: '#243D4F', color: '#F4F9FC' }}>
        <Box sx={{ position: 'relative', p: { xs: 3, sm: 4, lg: 5 }, minHeight: { md: 390 } }}>
          <Box component="img" src={signal} alt="" aria-hidden="true" sx={{
            position: 'absolute', right: '-5%', top: 0, width: '57%', height: '100%', objectFit: 'cover',
            pointerEvents: 'none', display: { xs: 'none', md: 'block' },
          }} />
          <Box sx={{ position: 'relative', maxWidth: { xs: '100%', md: '60%' } }}>
            <Stack direction="row" spacing={2} sx={{ alignItems: 'center', mb: 3.5 }}>
              <Box sx={{ bgcolor: '#F4F9FC', px: 1.25, py: .75, borderRadius: .75, display: 'flex' }}>
                <Box component="img" src={emi} alt="EMI Lab" sx={{ height: 32, width: 103 }} />
              </Box>
              <Box sx={{ borderLeft: '1px solid #466172', pl: 2 }}>
                <Typography sx={{ fontSize: 13, fontWeight: 650, letterSpacing: '.08em' }}>TU Delft</Typography>
                <Typography sx={{ color: '#AAC3D4', fontSize: 11, mt: .25 }}>Efficient Machine Intelligence Lab</Typography>
              </Box>
            </Stack>
            <Box component="h2" id="home-introduction" sx={{ m: 0, width: '100%', maxWidth: 520, aspectRatio: '600 / 168' }}><StudioLogo inverse /></Box>
            <Typography sx={{ mt: 2, maxWidth: 475, fontSize: { xs: 14, sm: 15.5 }, lineHeight: 1.8, color: '#C2D4E0', textWrap: 'pretty' }}>
              {t('home.intro')}
            </Typography>
            <Stack direction="row" useFlexGap sx={{ mt: 3, gap: 1.5, alignItems: 'center', flexWrap: 'wrap' }}>
              <Button component={RouterLink} to="/datasets?guide=start" variant="contained" endIcon={<ArrowForwardIcon />} sx={{
                minHeight: 54, minWidth: { xs: 0, sm: 194 }, width: { xs: '100%', sm: 'auto' }, px: 3, gap: 2, bgcolor: '#A1E8F5', color: '#0D2233', fontSize: 16, fontWeight: 750,
                '&:hover': { bgcolor: '#C4F3FB', boxShadow: '0 4px 24px #72D7EE26' },
                '&:focus-visible': { outline: '3px solid #FFFFFF', outlineOffset: 5 },
              }}>{t('home.start.action')}</Button>
              <Button href="https://github.com/lab-emi/OpenDPD" target="_blank" rel="noopener noreferrer" variant="outlined" startIcon={<GitHubIcon />} sx={{
                minHeight: 48, width: { xs: '100%', sm: 'auto' }, px: 2.5, color: '#C2D4E0', borderColor: '#5A788C', fontWeight: 600,
                '&:hover': { borderColor: '#A9C3D3', bgcolor: '#FFFFFF0A', color: '#FFFFFF' },
                '&:focus-visible': { outline: '3px solid #FFFFFF', outlineOffset: 5 },
              }}>OpenDPD GitHub</Button>
            </Stack>
            <Typography sx={{ mt: 1.5, color: '#A9C3D3', fontSize: 12.5 }}>{t(caps.data?.custom_dataset_imports ? 'home.start.hint' : 'home.start.builtinHint')}</Typography>
          </Box>
        </Box>
        <Box component="ol" aria-label={t('home.workflow.label')} sx={{
          display: 'grid', gridTemplateColumns: { xs: '1fr', sm: 'repeat(3, minmax(0, 1fr))' }, m: 0,
          px: { xs: 3, sm: 4, lg: 5 }, py: 2.5, listStyle: 'none', borderTop: '1px solid #2B4455', bgcolor: '#132C3E', gap: { xs: 2, sm: 3 },
        }}>
          {WORKFLOW.map((step, index) => (
            <Box component="li" key={step} sx={{ display: 'flex', gap: 1.5, minWidth: 0 }}>
              <Typography component="span" aria-hidden="true" sx={{ fontFamily: 'monospace', color: '#8ACAD9', fontSize: 12, pt: .3 }}>0{index + 1}</Typography>
              <Box>
                <Typography sx={{ fontSize: 14, fontWeight: 650 }}>{t(`home.workflow.${step}`)}</Typography>
                <Typography sx={{ fontSize: 12, lineHeight: 1.6, color: '#A9C3D3', mt: .5 }}>{t(step === 'data' && !caps.data?.custom_dataset_imports ? 'home.workflow.data.builtinHelp' : `home.workflow.${step}.help`)}</Typography>
              </Box>
            </Box>
          ))}
        </Box>
      </Paper>

      <section aria-labelledby="recent-runs">
        <Stack direction="row" sx={{ alignItems: 'center', justifyContent: 'space-between', gap: 2, mb: 1.5 }}>
          <Typography variant="h2" id="recent-runs">{t('home.recentRuns')}</Typography>
          {!!runs.data?.length && <Link component={RouterLink} to="/experiments" underline="hover" sx={{ fontSize: 13 }}>{t('home.runs.all')}</Link>}
        </Stack>
        {runs.isPending ? (
          <LoadingState />
        ) : runs.isError ? (
          <ErrorState error={runs.error} onRetry={() => void runs.refetch()} />
        ) : runs.data.length === 0 ? (
          <EmptyState title={t('home.runs.empty')} body={t('home.noRuns')} />
        ) : (
          <Paper sx={{ overflowX: 'auto' }}><RunTable runs={runs.data.slice(0, 8)} /></Paper>
        )}
      </section>
    </Stack>
  )
}
