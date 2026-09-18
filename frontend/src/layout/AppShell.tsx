import { runIdFromPath } from '@/api/client'
import { useSession } from '@/api/hooks'
import DatasetIcon from '@mui/icons-material/Dataset'
import GraphicEqIcon from '@mui/icons-material/GraphicEq'
import PlayArrowIcon from '@mui/icons-material/PlayArrow'
import PreviewIcon from '@mui/icons-material/Preview'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import ChevronRightIcon from '@mui/icons-material/ChevronRight'
import ElectricalServicesIcon from '@mui/icons-material/ElectricalServices'
import HomeIcon from '@mui/icons-material/Home'
import InsightsIcon from '@mui/icons-material/Insights'
import ScienceIcon from '@mui/icons-material/Science'
import SettingsIcon from '@mui/icons-material/Settings'
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined'
import DnsOutlinedIcon from '@mui/icons-material/DnsOutlined'
import AppBar from '@mui/material/AppBar'
import Box from '@mui/material/Box'
import Chip from '@mui/material/Chip'
import Drawer from '@mui/material/Drawer'
import List from '@mui/material/List'
import ListItemButton from '@mui/material/ListItemButton'
import ListItemIcon from '@mui/material/ListItemIcon'
import ListItemText from '@mui/material/ListItemText'
import Toolbar from '@mui/material/Toolbar'
import Tooltip from '@mui/material/Tooltip'
import Typography from '@mui/material/Typography'
import useMediaQuery from '@mui/material/useMediaQuery'
import { useIsMutating, useQueryClient } from '@tanstack/react-query'
import { useEffect, useState } from 'react'
import { Link, useLocation, useNavigate } from 'react-router'
import { useCapabilities, useRun, useRuns } from '@/api/hooks'
import { WEB_MODE } from '@/api/client'
import { RouteContent } from '@/components/RouteContent'
import { LanguageMenu } from '@/components/LanguageMenu'
import { ResetButton } from '@/components/ResetButton'
import { ReportBugsButton } from '@/components/ReportBugsButton'
import { StudioLogo } from '@/components/StudioLogo'
import { ExperimentTerminal } from '@/components/ExperimentTerminal'
import { WorkflowProgress } from '@/components/WorkflowProgress'
import { WorkspaceExpiry } from '@/components/WorkspaceExpiry'
import { useStudioWorkflow } from '@/workflow/StudioWorkflow'
import { isExperimentTask } from '@/components/ExperimentTasks'
import { t, type MessageKey } from '@/i18n'
import { tokens, useStudioColors } from '@/theme'

const NAV: Array<{ to: string; key: MessageKey; Icon: typeof HomeIcon }> = [
  { to: '/', key: 'nav.home', Icon: HomeIcon },
  { to: '/signal-generator', key: 'generator.title', Icon: GraphicEqIcon },
  { to: '/signal-analyzer', key: 'analyzer.title', Icon: InsightsIcon },
  { to: '/pa-library', key: 'paLibrary.title', Icon: ElectricalServicesIcon },
  { to: '/datasets', key: 'nav.datasets', Icon: DatasetIcon },
  { to: '/experiments', key: 'nav.experiments', Icon: ScienceIcon },
  { to: '/results', key: 'nav.results', Icon: InsightsIcon },
  { to: '/server', key: 'server.title', Icon: DnsOutlinedIcon },
  { to: '/settings', key: 'nav.settings', Icon: SettingsIcon },
  { to: '/about', key: 'about.title', Icon: InfoOutlinedIcon },
]

/** Route-derived selection also covers experiment run detail pages. */
export function AppShell() {
  const colors = useStudioColors()
  const workflow = useStudioWorkflow()
  const caps = useCapabilities()
  const running = useRuns('running')
  const { pathname, search } = useLocation()
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const { data: session } = useSession()
  const mutating = useIsMutating() > 0
  const [revision, setRevision] = useState(0)
  const runId = runIdFromPath(pathname)
  const run = useRun(runId, !!runId)
  const requestedTask = runId ? run.data?.task ?? null : new URLSearchParams(search).get('task')
  const task = isExperimentTask(requestedTask) ? requestedTask : null
  const datasetDetail = pathname.startsWith('/datasets/')
  const generatorActive = pathname === '/signal-generator' || pathname.startsWith('/signal-generator/')
  const generatorPreview = pathname === '/signal-generator/preview'
  const resultDetail = pathname.startsWith('/results/')
  const resetDetail = datasetDetail ? t('reset.page.dataset') : runId ? t('reset.page.experiment') : resultDetail ? t('reset.page.results') : undefined
  useEffect(() => {
    window.scrollTo({ top: 0, left: 0, behavior: 'instant' })
    // Route changes and explicit same-page resets are the viewport triggers.
    // oxlint-disable-next-line react/exhaustive-effect-dependencies
  }, [pathname, revision])
  const showWorkflow = !['/settings', '/about', '/server', '/hardware', '/robustness', '/sweeps'].some(path => pathname === path || pathname.startsWith(path + '/'))
  const reset = (global: boolean) => {
    // Drop selections encoded in detail URLs as well as drafts in React state.
    // A run restarts at its task's setup, never at the old progress dashboard.
    setRevision((value) => value + 1)
    if (global) { workflow.reset(); navigate('/datasets?guide=start', { replace: true }) }
    else {
      if (generatorActive) workflow.reset()
      if (pathname === '/pa-library') workflow.resetPA()
      const destination = generatorActive ? '/signal-generator' : datasetDetail ? '/datasets' : resultDetail ? '/results' : runId ? (task ? '/experiments/new' : '/experiments') : pathname.startsWith('/robustness/') ? '/robustness' : pathname
      navigate({ pathname: destination, search: destination === '/experiments/new' && task ? `?task=${task}` : '', hash: '' }, { replace: true })
    }
    void queryClient.invalidateQueries()
  }
  const count = running.data?.length ?? 0
  const wideNavigation = useMediaQuery('(min-width: 900px)')
  const compactToolbar = useMediaQuery('(max-width: 1099px)')
  const width = { xs: 64, md: tokens.layout.navWidth }
  const active = (to: string) => to === '/' ? pathname === '/' : pathname === to || pathname.startsWith(`${to}/`) || (to === '/experiments' && pathname.startsWith('/runs/'))
  const current = NAV.find(({ to }) => active(to))
  return (
    <Box sx={{ display: 'flex', minHeight: '100dvh' }}>
      <a href="#main" style={{ position: 'absolute', left: -9999, top: 8, background: colors.surface, color: colors.primary, padding: 8, zIndex: 2000 }} onFocus={(e) => (e.currentTarget.style.left = '8px')} onBlur={(e) => (e.currentTarget.style.left = '-9999px')}>
        {t('app.skipToContent')}
      </a>
      <Drawer variant="permanent" sx={{ width, flexShrink: 0, '& .MuiDrawer-paper': { width, boxSizing: 'border-box', bgcolor: colors.navBackground, color: colors.navText, border: 0, borderRight: '1px solid', borderColor: 'divider' } }}>
        <Box sx={{ minHeight: 72, display: 'flex', alignItems: 'center', justifyContent: 'center', p: 1 }}>
          <Box component={Link} to="/" aria-label="OpenDPD Studio" sx={{ display: 'block', width: '100%', height: { xs: 48, md: 56 }, borderRadius: 1, '&:focus-visible': { outline: `2px solid ${colors.primary}`, outlineOffset: 2 } }}>
            <StudioLogo compact={!wideNavigation} />
          </Box>
        </Box>
        <List component="nav" aria-label={t('nav.primary')} sx={{ px: 1, pt: 1 }}>
          {NAV.map(({ to, key, Icon }) => <Box key={to}>
            <Tooltip title={t(key)} placement="right" disableHoverListener={wideNavigation}>
              <ListItemButton component={Link} to={to} selected={active(to)} aria-current={active(to) && to !== '/signal-generator' ? 'page' : undefined} aria-label={t(key)}
                aria-expanded={to === '/signal-generator' ? generatorActive : undefined} aria-controls={to === '/signal-generator' && generatorActive ? 'generator-navigation' : undefined} sx={{
                minHeight: 44, px: 1.5, mb: .75, borderRadius: 1, color: 'inherit', position: 'relative',
                ...(to === '/settings' ? { mt: 3 } : {}),
                '&:hover': { bgcolor: colors.surfaceMuted },
                '&.Mui-selected': { bgcolor: colors.selected, color: 'primary.main', '&:hover': { bgcolor: colors.selectedHover }, '&::before': { content: '""', position: 'absolute', width: 3, height: 22, left: 0, borderRadius: 2, bgcolor: 'primary.main' } },
              }}>
                <ListItemIcon sx={{ minWidth: { xs: 24, md: to === '/signal-generator' ? 24 : 32 }, color: active(to) ? 'primary.main' : 'inherit' }}><Icon fontSize="small" /></ListItemIcon>
                <ListItemText primary={t(key)} sx={{ display: { xs: 'none', md: 'block' } }} slotProps={{ primary: { sx: { fontSize: 14, fontWeight: active(to) ? 650 : 450 } } }} />
                {to === '/signal-generator' && (generatorActive ? <ExpandMoreIcon sx={{ fontSize: 16, display: { xs: 'none', md: 'block' } }} /> : <ChevronRightIcon sx={{ fontSize: 16, display: { xs: 'none', md: 'block' } }} />)}
              </ListItemButton>
            </Tooltip>
            {to === '/signal-generator' && generatorActive && <List id="generator-navigation" component="div" role="group" aria-label={t('generator.navigation')} disablePadding sx={{ mb: 1, ml: { xs: 0, md: 2 }, borderLeft: { md: '1px solid' }, borderColor: 'divider' }}>
              {[
                { path: '/signal-generator', label: 'generator.page.generate' as const, ChildIcon: PlayArrowIcon, selected: !generatorPreview },
                { path: '/signal-generator/preview', label: 'generator.page.preview' as const, ChildIcon: PreviewIcon, selected: generatorPreview },
              ].map(({ path, label, ChildIcon, selected }) => <ListItemButton key={path} component={Link} to={path} selected={selected} aria-current={selected ? 'page' : undefined} aria-label={t(label)}
                sx={{ minHeight: 44, px: { xs: .25, md: 1.25 }, py: .75, mb: .5, borderRadius: 1, flexDirection: { xs: 'column', md: 'row' }, gap: { xs: .25, md: 1 }, color: selected ? 'primary.main' : 'inherit', '&.Mui-selected': { bgcolor: colors.selected }, '&:hover': { bgcolor: colors.surfaceMuted } }}>
                <ChildIcon sx={{ fontSize: 18 }} />
                <Typography component="span" sx={{ fontSize: { xs: 9, md: 13 }, lineHeight: 1.4, fontWeight: selected ? 650 : 450 }}>{t(label)}</Typography>
              </ListItemButton>)}
            </List>}
          </Box>)}
        </List>
        <Box sx={{ mt: 'auto', px: 2.5, py: 2, display: { xs: 'none', md: 'block' }, borderTop: '1px solid', borderColor: 'divider' }}>
          <Typography variant="caption">{WEB_MODE ? t('web.welcome') : t('shell.localWorkspace')}</Typography>
          <Typography variant="body2" sx={{ color: 'text.primary', mt: .5 }}>{caps.data?.version ? `v${caps.data.version}` : 'OpenDPD'}</Typography>
        </Box>
      </Drawer>
      <AppBar position="fixed" color="default" elevation={0} sx={{ width: { xs: 'calc(100% - 64px)', md: `calc(100% - ${tokens.layout.navWidth}px)` }, border: 0, borderBottom: `1px solid ${colors.border}`, bgcolor: 'background.paper' }}>
        <Toolbar sx={{ minHeight: { xs: WEB_MODE ? '88px !important' : '56px !important', sm: '56px !important' }, flexWrap: { xs: WEB_MODE ? 'wrap' : 'nowrap', sm: 'nowrap' }, columnGap: { xs: .5, sm: 1, lg: 1.5 }, px: { xs: '8px !important', sm: '16px !important', lg: '20px !important' } }}>
          <Typography variant="body2" noWrap sx={{ fontWeight: 600, minWidth: 0, display: { xs: 'none', sm: WEB_MODE ? 'none' : 'block', lg: 'block' } }}>{current ? t(current.key) : t('app.title')}</Typography>
          <Box sx={{ height: 16, borderLeft: `1px solid ${colors.border}`, display: { xs: 'none', lg: 'block' } }} />
          {WEB_MODE && session?.expires_at ? <WorkspaceExpiry expiresAt={session.expires_at} idleExpiresAt={session.idle_expires_at} /> : <Typography variant="body2" color="text.secondary" noWrap sx={{ maxWidth: 380, minWidth: 0, display: { xs: 'none', lg: 'block' } }} title={caps.data?.workspace}>
            {caps.data?.workspace.split(/[\\/]/).filter(Boolean).at(-1) ?? '…'}
          </Typography>}
          <Box sx={{ flex: 1 }} />
          <ReportBugsButton compact={compactToolbar} />
          <ResetButton compact={compactToolbar} disabled={mutating} detail={resetDetail} onReset={() => reset(false)} />
          <ResetButton compact={compactToolbar} scope="studio" disabled={mutating} onReset={() => reset(true)} />
          <Chip size="small" sx={{ display: { xs: 'none', sm: 'flex' }, flexShrink: 0 }} color={count > 0 ? 'info' : 'default'} variant="outlined" label={count > 0 ? t('topbar.nowRunning', { count }) : t('topbar.idle')} data-testid="now-running" />
          <LanguageMenu />
        </Toolbar>
      </AppBar>
      <Box key={revision} component="main" id="main" tabIndex={-1} sx={{ flex: 1, px: { xs: 1.5, md: 2.5 }, pb: 2, pt: { xs: WEB_MODE ? '108px' : '76px', sm: '76px' }, minWidth: 0 }}>
        <Box sx={{ maxWidth: tokens.layout.maxContent, mx: 'auto', minWidth: 0 }}>
          {showWorkflow && <WorkflowProgress />}
          <RouteContent />
          {(pathname.startsWith('/experiments') || pathname.startsWith('/runs/')) && <ExperimentTerminal />}
        </Box>
      </Box>
    </Box>
  )
}
