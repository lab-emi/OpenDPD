import GitHubIcon from '@mui/icons-material/GitHub'
import RefreshIcon from '@mui/icons-material/Refresh'
import OpenInNewIcon from '@mui/icons-material/OpenInNew'
import Avatar from '@mui/material/Avatar'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Chip from '@mui/material/Chip'
import Grid from '@mui/material/Grid'
import Link from '@mui/material/Link'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/api/client'
import { formatDateTime, formatNumber, t, type MessageKey } from '@/i18n'
import emi from '@/assets/emi-logo.svg'
import tudelft from '@/assets/tudelft-logo.svg'
import changGao from '@/assets/chang-gao.webp'
import yizhuoWu from '@/assets/yizhuo-wu.webp'
import { StudioLogo } from '@/components/StudioLogo'
import { useStudioColors } from '@/theme'

interface ProjectInfo {
  version: string; repository: string; local_commit: string | null; updated_at: string | null; status: string
  contributors: Array<{ login: string; contributions: number; url: string }>
  commits: Array<{ sha: string; url: string; message: string; author: string; date: string }>
  contributors_truncated?: boolean
}

const REPO = 'https://github.com/lab-emi/OpenDPD'
const LEADERS = [
  { name: 'Chang Gao', role: 'about.leader' as MessageKey, login: 'gaochangw', portrait: changGao },
  { name: 'Yizhuo Wu', role: 'about.developer' as MessageKey, login: 'yizhuo990', portrait: yizhuoWu },
]

export function AboutPage() {
  const colors = useStudioColors()
  // Partner artwork keeps its official colors on an intentional light brand plate.
  const partnerPlate = { display: 'flex', alignItems: 'center', p: 1.5, borderRadius: 1, bgcolor: '#F9FBFD' }
  const info = useQuery({ queryKey: ['system', 'about'], queryFn: () => api.get<ProjectInfo>('/system/about'), refetchInterval: 300_000, staleTime: 300_000, retry: false })
  const data = info.data
  return <Stack spacing={3} sx={{ maxWidth: 1180, mx: 'auto' }}>
    <Stack direction="row" sx={{ alignItems: 'center', justifyContent: 'space-between', gap: 2 }}><Typography variant="h1">{t('about.title')}</Typography><Chip variant="outlined" size="small" label={data?.version ? `v${data.version}` : 'OpenDPD Studio'} /></Stack>
    <Paper sx={{ p: { xs: 3, md: 4 }, background: `linear-gradient(125deg, ${colors.selected}, ${colors.surface} 72%)` }}>
      <Box sx={{ width: 'min(100%, 480px)', aspectRatio: '600 / 168', mb: 3 }}><StudioLogo /></Box>
      <Stack direction="row" sx={{ alignItems: 'center', gap: 4, mb: 3, flexWrap: 'wrap' }}>
        <Link href="https://www.tudemi.com/" target="_blank" rel="noopener noreferrer" sx={partnerPlate}><Box component="img" src={emi} alt="EMI Lab" sx={{ height: 52, maxWidth: 210 }} /></Link>
        <Box sx={{ height: 48, borderLeft: '1px solid', borderColor: 'divider' }} />
        <Link href="https://www.tudelft.nl/en/" target="_blank" rel="noopener noreferrer" sx={partnerPlate}><Box component="img" src={tudelft} alt="TU Delft" sx={{ height: 62, maxWidth: 170 }} /></Link>
      </Stack>
      <Typography color="text.secondary" sx={{ mt: 1, maxWidth: 740 }}>{t('about.description')}</Typography>
      <Stack direction="row" spacing={1.5} sx={{ mt: 3, flexWrap: 'wrap' }} useFlexGap><Button href="https://www.tudemi.com/" target="_blank" rel="noopener noreferrer" variant="contained" endIcon={<OpenInNewIcon />}>{t('about.lab')}</Button><Button href={REPO} target="_blank" rel="noopener noreferrer" variant="outlined" startIcon={<GitHubIcon />}>GitHub</Button></Stack>
    </Paper>
    <Box><Typography variant="h2" sx={{ mb: 2 }}>{t('about.people')}</Typography><Grid container spacing={2}>
      {LEADERS.map((person) => <Grid size={{ xs: 12, sm: 6 }} key={person.login}><Paper sx={{ p: 2.5 }}><Stack direction="row" spacing={2} sx={{ alignItems: 'center' }}><Avatar src={person.portrait} alt={person.name} sx={{ width: 64, height: 64, bgcolor: colors.selected, color: 'primary.main', fontWeight: 700 }}>{person.name.split(' ').map((word) => word[0]).join('')}</Avatar><Box><Link href={`https://github.com/${person.login}`} target="_blank" rel="noopener noreferrer" underline="hover" sx={{ fontWeight: 700, fontSize: 18 }}>{person.name}</Link><Typography variant="body2" color="text.secondary">{t(person.role)}</Typography></Box></Stack></Paper></Grid>)}
    </Grid></Box>
    <Paper sx={{ p: 3 }}>
      <Stack direction="row" sx={{ alignItems: 'center', gap: 1, mb: 1, flexWrap: 'wrap' }}><GitHubIcon fontSize="small" /><Typography variant="h2">{t('about.activity')}</Typography><Button size="small" sx={{ ml: 'auto' }} startIcon={<RefreshIcon />} disabled={info.isFetching} onClick={() => void info.refetch()}>{t('about.refresh')}</Button></Stack>
      <Typography variant="caption" color="text.secondary" component="p" sx={{ mb: 2 }}>{t('about.refreshNote')}{data?.updated_at ? ` · ${formatDateTime(data.updated_at)}` : ''}</Typography>
      {(info.isError || data?.status === 'unavailable' || data?.status === 'stale') && <Alert severity="info" sx={{ mb: 2 }}>{t('about.offline')}</Alert>}
      {info.isPending && <Typography color="text.secondary">{t('about.loading')}</Typography>}
      <Grid container spacing={3}>
        <Grid size={{ xs: 12, md: 5 }}><Typography variant="h3" sx={{ mb: 1.5 }}>{t('about.contributors')}</Typography><Stack spacing={1}>
          {(data?.contributors ?? []).map((person) => <Stack key={person.login} direction="row" sx={{ justifyContent: 'space-between', py: .75, gap: 1 }}><Link href={person.url} target="_blank" rel="noopener noreferrer" underline="hover">{LEADERS.find((leader) => leader.login === person.login)?.name ?? person.login}</Link><Typography variant="body2" color="text.secondary">{t('about.commits', { count: formatNumber(person.contributions) })}</Typography></Stack>)}
          <Link href={`${REPO}/graphs/contributors`} target="_blank" rel="noopener noreferrer" sx={{ fontSize: 13 }}>{t('about.allContributors')}</Link>
        </Stack></Grid>
        <Grid size={{ xs: 12, md: 7 }}><Typography variant="h3" sx={{ mb: 1.5 }}>{t('about.latest')}</Typography><Typography variant="caption" color="text.secondary">{t('about.original')}</Typography><Stack spacing={2}>{(data?.commits ?? []).map((commit) => <Box key={commit.sha}><Link href={commit.url} target="_blank" rel="noopener noreferrer" underline="hover" sx={{ fontSize: 14 }}>{commit.message}</Link><Typography variant="caption" color="text.secondary" component="p" sx={{ mt: .4 }}><code>{commit.sha.slice(0, 8)}</code> · {commit.author} · {formatDateTime(commit.date)}</Typography></Box>)}</Stack></Grid>
      </Grid>
    </Paper>
    <Typography variant="caption" color="text.secondary">Apache-2.0 · {t('about.localCommit')}: <code>{data?.local_commit?.slice(0, 12) ?? '—'}</code></Typography>
  </Stack>
}
