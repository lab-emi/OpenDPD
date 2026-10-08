import GraphicEqIcon from '@mui/icons-material/GraphicEq'
import DataObjectIcon from '@mui/icons-material/DataObject'
import ButtonBase from '@mui/material/ButtonBase'
import { MatlinkResults } from '@/components/MatlinkResults'
import ArrowDownwardIcon from '@mui/icons-material/ArrowDownward'
import ArrowOutwardIcon from '@mui/icons-material/ArrowOutward'
import LinkIcon from '@mui/icons-material/Link'
import RefreshIcon from '@mui/icons-material/Refresh'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Chip from '@mui/material/Chip'
import Divider from '@mui/material/Divider'
import Grid from '@mui/material/Grid'
import MenuItem from '@mui/material/MenuItem'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { useQueryClient } from '@tanstack/react-query'
import { useEffect, useRef, useState } from 'react'
import { Link as RouterLink, useNavigate } from 'react-router'
import { ApiError, WEB_MODE } from '@/api/client'
import { keys } from '@/api/hooks'
import { useMatlink, useMatlinkRequest, type MatlinkAction, type MatlinkRequest, type MatlinkTransfer } from '@/api/matlink'
import { ErrorState, LoadingState } from '@/components/StateBlock'
import { formatDateTime, t } from '@/i18n'

const pending = (transfer: MatlinkTransfer) => transfer.status === 'queued' || transfer.status === 'waiting'
const transferColor = (status: MatlinkTransfer['status']) => status === 'succeeded' ? 'success' : status === 'failed' ? 'error' : 'default'
const sessionLabel = ({ label, release }: { label: string; release: string }) => label.toLowerCase().includes(release.replace(/^r/i, '').toLowerCase()) ? label : `${label} · ${release}`

export function MatlinkPage() {
  // Hosted Studio has no authority over a desktop MATLAB session.
  return WEB_MODE ? <Stack spacing={2}>
    <Typography variant="h1">{t('matlink.title')}</Typography>
    <Alert severity="info">{t('matlink.localOnly')}</Alert>
    <Box component="pre" sx={{ p: 2, bgcolor: 'action.hover', borderRadius: 1 }}>opendpd.studio()</Box>
  </Stack> : <LocalMatlinkPage />
}

function LocalMatlinkPage() {
  const navigate = useNavigate()
  const [source, setSource] = useState<'generator' | 'workspace'>('generator')
  const snapshot = useMatlink()
  const request = useMatlinkRequest()
  const client = useQueryClient()
  const [sessionChoice, setSessionChoice] = useState('')
  const [input, setInput] = useState('')
  const [output, setOutput] = useState('')
  const [name, setName] = useState('')
  const [sampleRate, setSampleRate] = useState('')
  const [bandwidth, setBandwidth] = useState('')
  const [origin, setOrigin] = useState('unknown')
  const [importRequest, setImportRequest] = useState('')
  const handled = useRef(new Set<string>())
  const sessions = snapshot.data?.sessions ?? []
  const online = sessions.filter(item => item.connected)
  const session = sessionChoice ? sessions.find(item => item.client_id === sessionChoice) : online.find(item => item.capabilities?.includes('dataset_export')) ?? online[0]
  const choices = online.some(item => item.client_id === sessionChoice) || !session ? online : [session, ...online]
  const connected = !!session?.connected && !snapshot.isError
  const transfers = (snapshot.data?.transfers ?? []).filter(item => item.client_id === session?.client_id)
  const variables = session?.variables.filter(item => item.eligible) ?? []
  const inputVar = variables.find(item => item.name === input)
  const outputVar = variables.find(item => item.name === output)
  const pairValid = !!inputVar && !!outputVar && input !== output && inputVar.n_samples === outputVar.n_samples
  const metadataValid = (name.trim() === '' || /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(name.trim())) && Number.isFinite(Number(sampleRate)) && Number(sampleRate) > 0 && Number.isFinite(Number(bandwidth)) && Number(bandwidth) > 0 && Number(bandwidth) <= Number(sampleRate)
  const latestImport = transfers.find(item => item.action === 'import_iq' && item.status === 'succeeded' && item.result?.dataset_id)
  const uncertain = request.isError && (!(request.error instanceof ApiError) || request.error.status >= 500)
  const actionPending = (action: MatlinkAction) => request.isPending || transfers.some(item => item.action === action && pending(item))

  useEffect(() => {
    // Pin the first external bridge selection; a disconnect must not switch MATLAB sessions.
    // oxlint-disable-next-line react/set-state-in-effect
    if (!sessionChoice && session) setSessionChoice(session.client_id)
  }, [sessionChoice, session])

  useEffect(() => {
    const imported = snapshot.data?.transfers.find(transfer => transfer.request_id === importRequest && transfer.status === 'succeeded')
    if (imported?.action === 'import_iq' && imported.result?.dataset_id) {
      navigate('/experiments/new?' + new URLSearchParams({ dataset: String(imported.result.dataset_id), version: 'raw-v1', task: 'train_pa', matlink: imported.client_id }))
    }
    for (const transfer of snapshot.data?.transfers ?? []) {
      if (transfer.status !== 'succeeded' || handled.current.has(transfer.request_id)) continue
      handled.current.add(transfer.request_id)
      if (transfer.action === 'import_iq') void client.invalidateQueries({ queryKey: ['datasets'] })
      if (transfer.action === 'import_result') {
        void client.invalidateQueries({ queryKey: ['runs'] })
        if (typeof transfer.payload.run_id === 'string') void client.invalidateQueries({ queryKey: keys.run(transfer.payload.run_id) })
      }
    }
  }, [snapshot.data, importRequest, client, navigate])

  const send = (action: MatlinkAction, payload: MatlinkRequest['payload'] = {}) => {
    if (!connected || !session || request.isPending) return
    request.mutate({ client_id: session.client_id, action, payload, idempotency_key: crypto.randomUUID() }, {
      onSuccess: transfer => { if (action === 'import_iq') setImportRequest(transfer.request_id) },
    })
  }
  const retry = () => { if (request.variables && connected) request.mutate(request.variables) }

  return <Stack spacing={3}>
    <Stack direction="row" spacing={2} sx={{ alignItems: 'center', flexWrap: 'wrap' }} useFlexGap>
      <Box sx={{ p: 1.25, display: 'flex', color: 'primary.main', bgcolor: 'action.selected', borderRadius: 2 }}><LinkIcon /></Box>
      <Box sx={{ flex: 1 }}><Typography variant="h1">{t('matlink.title')}</Typography><Typography color="text.secondary">{t('matlink.subtitle')}</Typography></Box>
      <Button startIcon={<RefreshIcon />} disabled={snapshot.isFetching} onClick={() => void snapshot.refetch()}>{t('matlink.refresh')}</Button>
    </Stack>

    {snapshot.isPending && <LoadingState />}
    {snapshot.isError && <ErrorState error={snapshot.error} onRetry={() => void snapshot.refetch()} />}
    <Paper component="section" aria-label={t('matlink.session')} variant="outlined" sx={{ p: { xs: 2, md: 3 }, borderTop: '3px solid', borderTopColor: connected ? 'success.main' : 'divider' }}>
      {session || online.length > 0 ? <Stack spacing={1.5}>
        <Stack direction="row" spacing={2} sx={{ alignItems: 'center', flexWrap: 'wrap' }} useFlexGap>
          {(!session || choices.length > 1) ? <TextField select size="small" label={t('matlink.session')} value={session?.client_id ?? ''} onChange={event => { setSessionChoice(event.target.value); setInput(''); setOutput(''); request.reset() }} sx={{ minWidth: 240 }}>
            {choices.map(item => <MenuItem key={item.client_id} value={item.client_id}>{sessionLabel(item)}{!item.connected && ` · ${t('matlink.offline')}`}</MenuItem>)}
          </TextField> : <Typography variant="h3" component="h2">{session && sessionLabel(session)}</Typography>}
          <Chip size="small" color={connected ? 'success' : 'default'} variant="outlined" label={t(connected ? 'matlink.connected' : 'matlink.offline')} />
          {!!session?.pending_count && <Chip size="small" label={t('matlink.pending', { n: session.pending_count })} />}
        </Stack>
        <Typography variant="caption" color="text.secondary">{t('matlink.sessionHelp')}</Typography>
        {snapshot.data?.workspace && <Typography variant="caption" color="text.secondary" sx={{ overflowWrap: 'anywhere' }}>{snapshot.data.workspace}</Typography>}
        {!connected && <Alert severity="warning">{t('matlink.reconnect')}</Alert>}
      </Stack> : <Stack spacing={1.5}>
        <Typography variant="h3" component="h2">{t('matlink.connectTitle')}</Typography>
        <Typography color="text.secondary">{t('matlink.connectHelp')}</Typography>
        <Box component="pre" sx={{ m: 0, p: 2, bgcolor: 'action.hover', borderRadius: 1, width: 'fit-content' }}>opendpd.studio()</Box>
      </Stack>}
    </Paper>

    {request.isError && <Alert severity="error" action={uncertain ? <Button color="inherit" disabled={!connected} onClick={retry}>{t('matlink.retry')}</Button> : undefined}>
      <Typography variant="body2">{request.error.message}</Typography>{uncertain && <Typography variant="caption">{t('matlink.retryHelp')}</Typography>}
    </Alert>}

    <Paper component="section" aria-labelledby="matlink-import-heading" variant="outlined" sx={{ p: { xs: 2, md: 3 } }}>
      <Stack spacing={2.5}>
        <Box><Typography variant="overline" color="primary.main">01 · MATLAB → Studio</Typography><Typography id="matlink-import-heading" variant="h2">{t('matlink.sourceTitle')}</Typography><Typography color="text.secondary" variant="body2" sx={{ mt: .5 }}>{t('matlink.sourceHelp')}</Typography></Box>
        <Grid container spacing={2}>
          {(['generator', 'workspace'] as const).map(option => <Grid key={option} size={{ xs: 12, md: 6 }}><ButtonBase onClick={() => setSource(option)} aria-pressed={source === option} sx={{ width: '100%', textAlign: 'left', alignItems: 'flex-start', justifyContent: 'flex-start', p: 2.5, gap: 2, border: 1, borderRadius: 2, borderColor: source === option ? 'primary.main' : 'divider', bgcolor: source === option ? 'action.selected' : 'transparent', '&.Mui-focusVisible': { outline: '2px solid', outlineColor: 'primary.main' } }}>
            {option === 'generator' ? <GraphicEqIcon color="primary" /> : <DataObjectIcon color="primary" />}<Box><Typography sx={{ fontWeight: 650 }}>{t(`matlink.source.${option}`)}</Typography><Typography variant="body2" color="text.secondary" sx={{ mt: .5 }}>{t(`matlink.source.${option}Help`)}</Typography></Box>
          </ButtonBase></Grid>)}
        </Grid>
        {source === 'generator' ? <Stack spacing={2}>
          <Typography variant="body2" color="text.secondary">{t('matlink.generatorFlow')}</Typography>
          {connected && !session?.capabilities?.includes('dataset_export') && <Alert severity="info">{t('matlink.upgrade')}</Alert>}
          <Button variant="contained" endIcon={<ArrowOutwardIcon />} sx={{ alignSelf: 'flex-start' }} disabled={!connected || !session?.capabilities?.includes('dataset_export')} component={RouterLink} to={'/signal-generator?' + new URLSearchParams({ matlink: session?.client_id ?? '' })}>{t('matlink.openGenerator')}</Button>
        </Stack> : <>
        {latestImport && <Alert severity="success" action={<Button component={RouterLink} to={`/datasets/${encodeURIComponent(String(latestImport.result?.dataset_id))}`}>{t('matlink.viewDataset')}</Button>}>{t('matlink.datasetReady')} <strong>{String(latestImport.result?.display_name ?? latestImport.result?.dataset_id)}</strong></Alert>}
        {connected && variables.length === 0 && <Alert severity="info">{t('matlink.noVariables')}</Alert>}
        <Grid container spacing={2}>
          {(['input', 'output'] as const).map(kind => <Grid key={kind} size={{ xs: 12, md: 6 }}>
            <TextField select fullWidth label={t(kind === 'input' ? 'matlink.input' : 'matlink.output')} disabled={!connected} value={variables.some(item => item.name === (kind === 'input' ? input : output)) ? (kind === 'input' ? input : output) : ''} onChange={event => (kind === 'input' ? setInput : setOutput)(event.target.value)}>
              <MenuItem value="" disabled>{t('matlink.selectVariable')}</MenuItem>
              {variables.map(item => <MenuItem value={item.name} key={item.name}>{item.name} · {item.size.join(' × ')} · {item.class_name}{item.complex ? ' · complex' : ''}</MenuItem>)}
            </TextField>
          </Grid>)}
          <Grid size={{ xs: 12, md: 4 }}><TextField fullWidth label={t('matlink.datasetName')} helperText={t('matlink.datasetNameHelp')} slotProps={{ htmlInput: { maxLength: 128 } }} value={name} disabled={!connected} onChange={event => setName(event.target.value)} /></Grid>
          <Grid size={{ xs: 12, sm: 6, md: 2.5 }}><TextField fullWidth label={t('matlink.sampleRate')} type="number" value={sampleRate} disabled={!connected} onChange={event => setSampleRate(event.target.value)} slotProps={{ htmlInput: { min: 0, step: 'any' } }} /></Grid>
          <Grid size={{ xs: 12, sm: 6, md: 2.5 }}><TextField fullWidth label={t('matlink.bandwidth')} type="number" value={bandwidth} disabled={!connected} onChange={event => setBandwidth(event.target.value)} slotProps={{ htmlInput: { min: 0, step: 'any' } }} /></Grid>
          <Grid size={{ xs: 12, md: 3 }}><TextField fullWidth select label={t('matlink.origin')} value={origin} disabled={!connected} onChange={event => setOrigin(event.target.value)}><MenuItem value="unknown">{t('matlink.unknown')}</MenuItem><MenuItem value="measured">{t('matlink.measured')}</MenuItem><MenuItem value="synthetic">{t('matlink.synthetic')}</MenuItem></TextField></Grid>
        </Grid>
        {inputVar && outputVar && !pairValid && <Alert severity="warning">{t('matlink.mismatch')}</Alert>}
        {pairValid && !metadataValid && <Typography variant="body2" color="text.secondary">{t('matlink.metadataInvalid')}</Typography>}
        <Stack direction="row" spacing={2} sx={{ alignItems: 'center', flexWrap: 'wrap' }} useFlexGap>
          <Button variant="contained" startIcon={<ArrowDownwardIcon />} disabled={!connected || !pairValid || !metadataValid || actionPending('import_iq')} onClick={() => send('import_iq', { input, output, name: name.trim(), sample_rate_mhz: Number(sampleRate), bandwidth_mhz: Number(bandwidth), origin })}>{t(actionPending('import_iq') && !request.isPending ? 'matlink.importBusy' : 'matlink.import')}</Button>
          <Typography color="text.secondary" variant="caption">{t('matlink.signalHint')}</Typography>
        </Stack>
        </>}
      </Stack>
    </Paper>

    <MatlinkResults session={session} connected={connected} transfers={transfers} busy={request.isPending} send={send} />

    <Paper component="section" aria-labelledby="matlink-transfers-heading" variant="outlined" sx={{ p: { xs: 2, md: 3 } }}>
      <Typography id="matlink-transfers-heading" variant="h2" gutterBottom>{t('matlink.transfers')}</Typography>
      <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>{t('matlink.busyHint')}</Typography>
      {transfers.length === 0 ? <Typography color="text.secondary">{t('matlink.noTransfers')}</Typography> : <Stack divider={<Divider />} spacing={1.5} aria-live="polite" aria-relevant="additions text">
        {transfers.slice(0, 12).map(transfer => <Stack key={transfer.request_id} direction={{ xs: 'column', sm: 'row' }} spacing={1.5} sx={{ justifyContent: 'space-between', alignItems: { sm: 'center' } }}>
          <Box sx={{ minWidth: 0 }}><Stack direction="row" spacing={1} useFlexGap sx={{ alignItems: 'center', flexWrap: 'wrap' }}><Typography variant="body2" sx={{ fontWeight: 600 }}>{t(`matlink.transfer.${transfer.action}`)}</Typography><Chip size="small" color={transferColor(transfer.status)} variant="outlined" label={t(`matlink.status.${transfer.status}`)} /></Stack>
            <Typography variant="caption" color="text.secondary">{formatDateTime(transfer.created_at)}</Typography>
            {transfer.error && <Typography variant="body2" color="error.main" sx={{ overflowWrap: 'anywhere' }}>{transfer.error}</Typography>}
            {transfer.status === 'succeeded' && (transfer.action === 'import_result' || transfer.action === 'import_dataset') && typeof transfer.result?.variable === 'string' && <Typography variant="body2" sx={{ fontFamily: 'monospace', overflowWrap: 'anywhere' }}>{t('matlink.savedAs', { variable: transfer.result.variable })}</Typography>}
          </Box>
          <Stack direction="row" spacing={1} useFlexGap sx={{ flexWrap: 'wrap' }}>
            {transfer.status === 'succeeded' && transfer.action === 'import_iq' && typeof transfer.result?.dataset_id === 'string' && <>
              <Button component={RouterLink} to={`/datasets/${encodeURIComponent(transfer.result.dataset_id)}`}>{t('matlink.viewDataset')}</Button>
              <Button component={RouterLink} to={`/experiments/new?dataset=${encodeURIComponent(transfer.result.dataset_id)}`}>{t('matlink.newExperiment')}</Button>
            </>}
            {transfer.status === 'succeeded' && (transfer.action === 'import_result' || transfer.action === 'import_dataset') && typeof transfer.result?.variable === 'string' && <Button disabled={!connected || request.isPending} onClick={() => send('open_variable', { variable: String(transfer.result?.variable) })} endIcon={<ArrowOutwardIcon />}>{t('matlink.openVariable')}</Button>}
          </Stack>
        </Stack>)}
      </Stack>}
    </Paper>
  </Stack>
}
