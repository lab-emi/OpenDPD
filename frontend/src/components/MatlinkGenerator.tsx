import Alert from '@mui/material/Alert'
import Button from '@mui/material/Button'
import Grid from '@mui/material/Grid'
import LinearProgress from '@mui/material/LinearProgress'
import MenuItem from '@mui/material/MenuItem'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { useEffect, useRef, useState } from 'react'
import { Link as RouterLink, useNavigate, useSearchParams } from 'react-router'
import { WEB_MODE } from '@/api/client'
import { useMatlink, useMatlinkRequest } from '@/api/matlink'
import { paDefaults, paText, useSimulateDataset, useVirtualPAs } from '@/api/virtualPA'
import { ErrorState } from '@/components/StateBlock'
import { t } from '@/i18n'
import { pairedDatasetName } from '@/utils/datasetNames'
import { useStudioWorkflow } from '@/workflow/StudioWorkflow'

/** A generated input becomes an explicitly simulated pair before training. */
export function useMatlinkGenerator() {
  const [params, setParams] = useSearchParams()
  const navigate = useNavigate()
  const clientId = WEB_MODE ? '' : params.get('matlink') ?? ''
  const datasetId = clientId ? params.get('paired') ?? '' : ''
  const active = !!clientId
  const snapshot = useMatlink(active)
  const models = useVirtualPAs(active)
  const pair = useSimulateDataset()
  const request = useMatlinkRequest()
  const workflow = useStudioWorkflow()
  const [modelChoice, setModelChoice] = useState('')
  const [edited, setEdited] = useState<Record<string, number> | null>(null)
  const model = models.data?.find(m => m.model_id === (modelChoice || 'rapp-am-pm')) ?? models.data?.[0]
  const parameters = edited ?? (model ? paDefaults(model) : {})
  const session = snapshot.data?.sessions.find(s => s.client_id === clientId)
  const supported = !!session?.capabilities?.includes('dataset_export')
  const connected = !!session?.connected && !snapshot.isError
  const transfer = snapshot.data?.transfers.find(r => r.client_id === clientId && r.action === 'import_dataset' && r.payload.dataset_id === datasetId)
  const attempt = useRef('')
  const retry = () => request.mutate({ client_id: clientId, action: 'import_dataset', payload: { dataset_id: datasetId }, idempotency_key: transfer?.status === 'failed' ? crypto.randomUUID() : `signal:${clientId}:${datasetId}` })
  useEffect(() => {
    if (!datasetId || !connected || !supported || transfer || request.isPending || request.isError || attempt.current === datasetId) return
    attempt.current = datasetId
    request.mutate({ client_id: clientId, action: 'import_dataset', payload: { dataset_id: datasetId }, idempotency_key: `signal:${clientId}:${datasetId}` })
  }, [datasetId, connected, supported, transfer, clientId, request])
  useEffect(() => {
    if (!datasetId || transfer?.status !== 'succeeded') return
    navigate('/experiments/new?' + new URLSearchParams({ dataset: datasetId, version: 'raw-v1', task: 'train_pa', matlink: clientId,
      matlab_variable: String(transfer.result?.variable ?? '') }), { replace: true })
  }, [datasetId, transfer, clientId, navigate])

  const finish = async (ids: string[], name: string) => {
    if (!active || !model) return
    workflow.configurePA(model.model_id, parameters)
    const result = await pair.mutateAsync({ input_signal_ids: ids, model_id: model.model_id, parameters, dataset_name: pairedDatasetName(name, model.model_id) })
    workflow.completeDataset(result.dataset.dataset_id, String(result.dataset.simulation?.simulation_id ?? ''))
    setParams(old => { const next = new URLSearchParams(old); next.set('paired', result.dataset.dataset_id); return next }, { replace: true })
  }
  const busy = active && (pair.isPending || request.isPending || !!datasetId)
  const canGenerate = !active || (connected && supported && !!model && !busy && model.parameters.every(p => Number.isFinite(parameters[p.key]) && parameters[p.key]! >= p.minimum && parameters[p.key]! <= p.maximum && (!p.integer || Number.isInteger(parameters[p.key]))))
  const panel = !active ? null : <Paper variant="outlined" sx={{ p: 2.5, borderColor: 'primary.main' }}>
    <Stack spacing={2}>
      <Typography variant="h2">{t('matlink.generatorTitle')}</Typography>
      <Typography color="text.secondary">{t('matlink.generatorFlow')}</Typography>
      {!connected && <Alert severity="warning">{t('matlink.reconnect')}</Alert>}
      {connected && !supported && <Alert severity="warning">{t('matlink.upgrade')}</Alert>}
      <Alert severity="info">{t('matlink.virtualPAHelp')}</Alert>
      {models.isError && <ErrorState error={models.error} onRetry={() => void models.refetch()} />}
      {model && <Grid container spacing={2}>
        <Grid size={{ xs: 12, md: 4 }}><TextField select fullWidth label={t('matlink.virtualPA')} value={model.model_id} disabled={busy} onChange={e => { setModelChoice(e.target.value); setEdited(null) }}>
          {models.data?.map(m => <MenuItem key={m.model_id} value={m.model_id}>{paText(m.name)}</MenuItem>)}
        </TextField></Grid>
        {model.parameters.map(p => <Grid key={p.key} size={{ xs: 6, md: 2 }}><TextField fullWidth type="number" label={paText(p.label)} value={Number.isFinite(parameters[p.key]) ? parameters[p.key] : ''} disabled={busy} slotProps={{ htmlInput: { min: p.minimum, max: p.maximum, step: p.step } }} onChange={e => setEdited({ ...parameters, [p.key]: e.target.value === '' ? NaN : Number(e.target.value) })} /></Grid>)}
      </Grid>}
      {!!datasetId && transfer?.status !== 'failed' && !request.isError && <><LinearProgress aria-label={t('matlink.savingSignals')} /><Typography>{t('matlink.savingSignals')}</Typography><Typography variant="caption">{t('matlink.busyHint')}</Typography></>}
      {pair.isError && <ErrorState error={pair.error} />}
      {request.isError && <ErrorState error={request.error} onRetry={connected ? retry : undefined} />}
      {transfer?.status === 'failed' && <Alert severity="error" action={<Button disabled={!connected} onClick={retry}>{t('matlink.retry')}</Button>}>{transfer.error}</Alert>}
      <Button component={RouterLink} to="/matlink" sx={{ alignSelf: 'flex-start' }}>{t('matlink.back')}</Button>
    </Stack>
  </Paper>
  return { active, busy, canGenerate, panel, finish }
}
