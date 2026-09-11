import DatasetIcon from '@mui/icons-material/Dataset'
import GridOnIcon from '@mui/icons-material/GridOn'
import HomeIcon from '@mui/icons-material/Home'
import InsightsIcon from '@mui/icons-material/Insights'
import ScienceIcon from '@mui/icons-material/Science'
import SettingsIcon from '@mui/icons-material/Settings'
import WidgetsIcon from '@mui/icons-material/Widgets'
import AppBar from '@mui/material/AppBar'
import Box from '@mui/material/Box'
import Chip from '@mui/material/Chip'
import Drawer from '@mui/material/Drawer'
import List from '@mui/material/List'
import ListItemButton from '@mui/material/ListItemButton'
import ListItemIcon from '@mui/material/ListItemIcon'
import ListItemText from '@mui/material/ListItemText'
import Toolbar from '@mui/material/Toolbar'
import Typography from '@mui/material/Typography'
import { NavLink, Outlet } from 'react-router'
import { useCapabilities, useRuns } from '@/api/hooks'
import { LanguageMenu } from '@/components/LanguageMenu'
import { t, type MessageKey } from '@/i18n'
import { tokens } from '@/theme'

const NAV: Array<{ to: string; key: MessageKey; Icon: typeof HomeIcon }> = [
  { to: '/', key: 'nav.home', Icon: HomeIcon },
  { to: '/datasets', key: 'nav.datasets', Icon: DatasetIcon },
  { to: '/experiments', key: 'nav.experiments', Icon: ScienceIcon },
  { to: '/results', key: 'nav.results', Icon: InsightsIcon },
  { to: '/robustness', key: 'nav.robustness', Icon: GridOnIcon },
  { to: '/settings', key: 'nav.settings', Icon: SettingsIcon },
  { to: '/gallery', key: 'nav.gallery', Icon: WidgetsIcon },
]

/** Fixed left navigation + top bar with the workspace and "what is running now". */
export function AppShell() {
  const caps = useCapabilities()
  const running = useRuns('running')
  const count = running.data?.length ?? 0
  const width = tokens.layout.navWidth
  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <a
        href="#main"
        style={{ position: 'absolute', left: -9999, top: 8, background: '#fff', padding: 8, zIndex: 2000 }}
        onFocus={(e) => (e.currentTarget.style.left = '8px')}
        onBlur={(e) => (e.currentTarget.style.left = '-9999px')}
      >
        {t('app.skipToContent')}
      </a>
      <AppBar position="fixed" color="default" elevation={0} sx={{ zIndex: (th) => th.zIndex.drawer + 1, borderBottom: `1px solid ${tokens.color.border}` }}>
        <Toolbar variant="dense" sx={{ gap: 2 }}>
          <Typography variant="h1" component="span" sx={{ fontSize: '1.05rem' }}>
            {t('app.title')}
          </Typography>
          <Typography variant="body2" color="text.secondary" sx={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={caps.data?.workspace}>
            {t('topbar.workspace')}: <code>{caps.data?.workspace ?? '…'}</code>
          </Typography>
          <Box sx={{ flex: 1 }} />
          <LanguageMenu />
          <Chip size="small" color={count > 0 ? 'info' : 'default'} variant={count > 0 ? 'filled' : 'outlined'} label={count > 0 ? t('topbar.nowRunning', { count }) : t('topbar.idle')} data-testid="now-running" />
        </Toolbar>
      </AppBar>
      <Drawer variant="permanent" sx={{ width, flexShrink: 0, '& .MuiDrawer-paper': { width, boxSizing: 'border-box' } }}>
        <Toolbar variant="dense" />
        <List component="nav" aria-label="primary">
          {NAV.map(({ to, key, Icon }) => (
            <ListItemButton key={to} component={NavLink} to={to} end={to === '/'} sx={{ '&.active': { bgcolor: `${tokens.color.primary}14`, fontWeight: 600 } }}>
              <ListItemIcon sx={{ minWidth: 36 }}>
                <Icon fontSize="small" />
              </ListItemIcon>
              <ListItemText primary={t(key)} />
            </ListItemButton>
          ))}
        </List>
      </Drawer>
      <Box component="main" id="main" tabIndex={-1} sx={{ flex: 1, p: 3, maxWidth: tokens.layout.maxContent, minWidth: 0 }}>
        <Toolbar variant="dense" />
        <Outlet />
      </Box>
    </Box>
  )
}
