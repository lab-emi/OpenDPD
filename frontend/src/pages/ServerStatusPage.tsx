import DnsOutlinedIcon from '@mui/icons-material/DnsOutlined'
import MemoryOutlinedIcon from '@mui/icons-material/MemoryOutlined'
import PeopleOutlineIcon from '@mui/icons-material/People'
import RefreshIcon from '@mui/icons-material/Refresh'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Chip from '@mui/material/Chip'
import Grid from '@mui/material/Grid'
import LinearProgress from '@mui/material/LinearProgress'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/api/client'
import type { components } from '@/api/schema'
import { ErrorState, LoadingState } from '@/components/StateBlock'
import { formatDateTime, formatNumber, t } from '@/i18n'

type ServerStatus = components['schemas']['ServerStatus']
type ResourceStatus = components['schemas']['ResourceStatus']

function Meter({ label, value, detail }: { label: string; value: number | null | undefined; detail?: string }) {
  return <Box>
    <Stack direction="row" sx={{ justifyContent: 'space-between', gap: 2, mb: .75 }}>
      <Typography variant="body2">{label}</Typography>
      <Typography variant="body2" sx={{ fontWeight: 650, fontVariantNumeric: 'tabular-nums' }}>{value == null ? '—' : `${formatNumber(value, { maximumFractionDigits: 1 })}%`}</Typography>
    </Stack>
    <LinearProgress aria-label={label} aria-hidden={value == null} variant="determinate" value={value ?? 0} color={value != null && value >= 90 ? 'warning' : 'primary'} sx={{ height: 8, borderRadius: 2, opacity: value == null ? .3 : 1 }} />
    {detail && <Typography variant="caption" color="text.secondary">{detail}</Typography>}
  </Box>
}

function memoryDetail(used?: number | null, total?: number | null) {
  return used != null && total != null ? `${formatNumber(used / 2**30, { maximumFractionDigits: 1 })} / ${formatNumber(total / 2**30, { maximumFractionDigits: 1 })} GiB` : undefined
}

function ResourcePanel({ title, description, resource, gpu = false }: { title: string; description: string; resource: ResourceStatus; gpu?: boolean }) {
  const load: components['schemas']['MachineLoad'] = resource.load ?? {}
  const stale = resource.stale
  const device = load.gpu
  const vram = device?.memory_used_bytes != null && device.memory_total_bytes ? 100 * device.memory_used_bytes / device.memory_total_bytes : null
  return <Paper sx={{ p: 2.5, height: '100%' }}>
    <Stack spacing={2.5}>
      <Stack direction="row" sx={{ alignItems: 'center', gap: 1 }}>
        {gpu ? <MemoryOutlinedIcon color="primary" /> : <DnsOutlinedIcon color="primary" />}
        <Typography variant="h2" sx={{ flex: 1 }}>{title}</Typography>
        <Chip size="small" variant="outlined" color={stale ? 'warning' : 'success'} label={t(stale ? 'server.stale' : 'server.live')} />
      </Stack>
      <Typography variant="body2" color="text.secondary">{description}</Typography>
      {stale && <Alert severity="warning">{t('server.staleBody')}</Alert>}
      <Meter label={t('server.cpu')} value={stale ? null : load.cpu_percent} />
      <Meter label={t('server.memory')} value={stale ? null : load.memory_percent} detail={stale ? undefined : memoryDetail(load.memory_used_bytes, load.memory_total_bytes)} />
      {gpu && <>
        <Meter label={t('server.gpu')} value={stale ? null : device?.utilization_percent} />
        <Meter label={t('server.vram')} value={stale ? null : vram} detail={stale ? undefined : memoryDetail(device?.memory_used_bytes, device?.memory_total_bytes)} />
        {!device && !stale && <Typography variant="body2" color="text.secondary">{t('server.gpuUnavailable')}</Typography>}
      </>}
      <Typography variant="caption" color="text.secondary">{load.sampled_at ? t('server.sampled', { date: formatDateTime(load.sampled_at) }) : t('server.waiting')}</Typography>
    </Stack>
  </Paper>
}

export function ServerStatusPage() {
  const status = useQuery({ queryKey: ['system', 'status'], queryFn: ({ signal }) => api.get<ServerStatus>('/system/status', signal),
    staleTime: 4_000, refetchInterval: (query) => query.state.error ? 30_000 : 5_000,
    refetchIntervalInBackground: false, retry: false })
  const data = status.data
  return <Stack spacing={3} sx={{ maxWidth: 1180, mx: 'auto' }}>
    <Stack direction="row" sx={{ justifyContent: 'space-between', alignItems: 'center', gap: 2, flexWrap: 'wrap' }}>
      <Box><Typography variant="h1">{t('server.title')}</Typography><Typography color="text.secondary" sx={{ mt: .75 }}>{t('server.description')}</Typography></Box>
      <Button variant="outlined" startIcon={<RefreshIcon />} disabled={status.isFetching} onClick={() => void status.refetch()}>{t('server.refresh')}</Button>
    </Stack>
    {status.isPending && <LoadingState />}
    {status.isError && <ErrorState error={status.error} onRetry={() => void status.refetch()} />}
    {data && <>
      <Grid container spacing={2}>
        {[
          { label: t('server.users'), value: data.active_sessions, note: data.mode === 'web' ? t('server.usersHelp') : t('server.localUsers') },
          { label: t('server.running'), value: data.running_jobs, note: data.parallel_capacity ? t('server.capacity', { count: data.parallel_capacity }) : t('server.localJobs') },
          { label: t('server.queued'), value: data.queued_jobs, note: t('server.queueHelp') },
        ].map(({ label, value, note }) => <Grid size={{ xs: 12, sm: 4 }} key={label}><Paper sx={{ p: 2.5, height: '100%' }}>
          <Typography variant="body2" color="text.secondary">{label}</Typography>
          <Typography sx={{ fontSize: 40, fontWeight: 650, my: .5, fontVariantNumeric: 'tabular-nums' }}>{status.isError ? '—' : value == null ? '—' : formatNumber(value)}</Typography>
          <Typography variant="caption" color="text.secondary">{note}</Typography>
        </Paper></Grid>)}
      </Grid>
      {data.mode === 'web' && <Stack direction="row" sx={{ gap: 1, alignItems: 'center' }}><PeopleOutlineIcon fontSize="small" color="action" /><Typography variant="body2" color="text.secondary">{t('server.workspaces', { count: data.workspaces, capacity: data.workspace_capacity ?? '—' })}</Typography></Stack>}
      <Grid container spacing={2}>
        <Grid size={{ xs: 12, md: data.mode === 'web' ? 6 : 12 }}><ResourcePanel title={t(data.mode === 'web' ? 'server.api' : 'server.local')} description={t(data.mode === 'web' ? 'server.apiHelp' : 'server.localHelp')} resource={{ ...data.api, stale: data.api.stale || status.isError }} gpu={data.mode === 'local'} /></Grid>
        {data.compute && <Grid size={{ xs: 12, md: 6 }}><ResourcePanel title={t('server.compute')} description={t('server.computeHelp')} resource={{ ...data.compute, stale: data.compute.stale || status.isError }} gpu /></Grid>}
      </Grid>
      <Typography variant="caption" color="text.secondary">{t('server.privacy')}</Typography>
    </>}
  </Stack>
}
