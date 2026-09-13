import Alert from '@mui/material/Alert'
import Button from '@mui/material/Button'
import Dialog from '@mui/material/Dialog'
import DialogActions from '@mui/material/DialogActions'
import DialogContent from '@mui/material/DialogContent'
import DialogTitle from '@mui/material/DialogTitle'
import Grid from '@mui/material/Grid'
import Stack from '@mui/material/Stack'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { useState } from 'react'
import { useNavigate } from 'react-router'
import { useGeneratorDataset, type GeneratedSignal, type GeneratorDatasetRequest } from '@/api/signalGenerator'
import { formatNumber, t } from '@/i18n'
import { ErrorState } from './StateBlock'

export function GeneratorDatasetDialog({ result, onClose }: { result: GeneratedSignal; onClose: () => void }) {
  const navigate = useNavigate()
  const create = useGeneratorDataset()
  const [config, setConfig] = useState<GeneratorDatasetRequest>({ dataset_id: `signal-${result.signal_id.slice(3, 13)}`, display_name: `${result.config.preset_id} · synthetic PA`, pa_gain: 1.6, compression: .7, am_pm: .1, memory: .08, noise_db: -60, guard_samples: 256, train_ratio: .6, val_ratio: .2 })
  const usable = result.analysis.sample_count - 2 * config.guard_samples
  const test = usable - Math.floor(usable * config.train_ratio) - Math.floor(usable * config.val_ratio)
  return <Dialog open fullWidth maxWidth="sm" onClose={create.isPending ? undefined : onClose} aria-labelledby="generator-dataset-title">
    <DialogTitle id="generator-dataset-title">{t('generator.createDataset')}</DialogTitle>
    <DialogContent><Stack spacing={2} sx={{ pt: 1 }}>
      <Alert severity="info">{t('generator.syntheticPaHelp')}</Alert>
      <TextField label={t('generator.datasetId')} value={config.dataset_id} onChange={e => setConfig(old => ({ ...old, dataset_id: e.target.value }))} />
      <TextField label={t('generator.datasetName')} value={config.display_name} onChange={e => setConfig(old => ({ ...old, display_name: e.target.value }))} />
      <Grid container spacing={2}>{([
        ['pa_gain', 'generator.paGain'], ['compression', 'generator.compression'], ['am_pm', 'generator.amPm'],
        ['memory', 'generator.memory'], ['noise_db', 'generator.noise'], ['guard_samples', 'generator.guard'],
        ['train_ratio', 'generator.trainRatio'], ['val_ratio', 'generator.valRatio'],
      ] as const).map(([key, label]) => <Grid key={key} size={6}><TextField fullWidth type="number" label={t(label)} value={Number.isFinite(config[key]) ? config[key] : ''} onChange={e => setConfig(old => ({ ...old, [key]: e.target.value === '' ? NaN : Number(e.target.value) }))} slotProps={{ inputLabel: { shrink: true }, htmlInput: { step: 'any' } }} /></Grid>)}</Grid>
      <Typography aria-live="polite">{t('modelWorkflow.testSamples')}: <strong>{Number.isFinite(test) && test > 0 ? formatNumber(test) : '—'}</strong> {t('generator.samplesUnit')}</Typography>
      <Typography variant="body2" color="text.secondary">{t('generator.privateHelp')}</Typography>
      {create.isError && <ErrorState error={create.error} />}
    </Stack></DialogContent>
    <DialogActions><Button onClick={onClose} disabled={create.isPending}>{t('form.cancel')}</Button><Button variant="contained" disabled={create.isPending || !config.dataset_id || !Number.isFinite(test) || test < 256} onClick={() => create.mutate({ signalId: result.signal_id, config }, { onSuccess: response => navigate(`/experiments/new?task=train_pa&dataset=${encodeURIComponent(response.dataset.dataset_id)}`) })}>{t(create.isPending ? 'state.loading' : 'generator.createTrain')}</Button></DialogActions>
  </Dialog>
}
