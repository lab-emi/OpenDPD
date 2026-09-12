import Alert from '@mui/material/Alert'
import Button from '@mui/material/Button'
import Dialog from '@mui/material/Dialog'
import DialogActions from '@mui/material/DialogActions'
import DialogContent from '@mui/material/DialogContent'
import DialogTitle from '@mui/material/DialogTitle'
import Grid from '@mui/material/Grid'
import MenuItem from '@mui/material/MenuItem'
import Stack from '@mui/material/Stack'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { useRef, useState } from 'react'
import { useNavigate } from 'react-router'
import { api } from '@/api/client'
import { useDataset, useSubmitRun } from '@/api/hooks'
import type { ExperimentConfigInput, RunView } from '@/api/types'
import type { components } from '@/api/schema'
import { t } from '@/i18n'
import { ErrorState } from '@/components/StateBlock'

type UploadResult = components['schemas']['UploadResult']

const ACCEPT = '.csv,.npy,.npz'

/** Local wall-clock time for a datetime-local input, computed once when the dialog opens. */
const localNow = () => {
  const now = new Date()
  return new Date(now.getTime() - now.getTimezoneOffset() * 60_000).toISOString().slice(0, 16)
}

interface Form {
  pa: string
  capture_chain: string
  sample_rate_hz: string
  drive: string
  gain_db: string
  calibration: string
  measured_at: string
  temperature_c: string
  operator: string
  power_with: string
  power_without: string
  notes: string
  playback: 'loop' | 'single'
}

const number = (s: string): number | null => (s.trim() === '' ? null : Number(s))

/** One capture file input: a labelled button, the chosen file name next to it. */
function CaptureInput({ label, help, file, onFile }: { label: string; help: string; file: File | null; onFile: (f: File | null) => void }) {
  return (
    <Stack spacing={0.5}>
      <Stack direction="row" spacing={1} sx={{ alignItems: 'center' }}>
        <Button component="label" variant="outlined" size="small">
          {label}
          <input type="file" hidden accept={ACCEPT} aria-label={label} onChange={(e) => onFile(e.target.files?.[0] ?? null)} />
        </Button>
        <Typography variant="body2" color={file ? 'text.primary' : 'text.secondary'}>
          {file ? `${file.name} (${Math.max(1, Math.round(file.size / 1024))} kB)` : t('run.measure.file.none')}
        </Typography>
      </Stack>
      <Typography variant="caption" color="text.secondary">
        {help}
      </Typography>
    </Stack>
  )
}

/**
 * evaluate_measured for a succeeded run_dpd run (plan S16): upload the analyser captures of the PA output while
 * this run's export was played, declare the conditions, and submit. The service aligns and scores the captures;
 * the result is labelled as operator-provided, not independently verified.
 */
export function MeasurementDialog({ run, onClose }: { run: RunView; onClose: () => void }) {
  const navigate = useNavigate()
  const dataset = useDataset(run.dataset_id ?? '')
  const submit = useSubmitRun()
  const idempotencyKey = useRef(crypto.randomUUID())
  const [withFile, setWithFile] = useState<File | null>(null)
  const [withoutFile, setWithoutFile] = useState<File | null>(null)
  const [uploading, setUploading] = useState(false)
  const [uploadError, setUploadError] = useState<unknown>(null)
  const [form, setForm] = useState<Form>({
    pa: '',
    capture_chain: '',
    sample_rate_hz: '',
    drive: '',
    gain_db: '',
    calibration: 'none',
    measured_at: localNow(),
    temperature_c: '',
    operator: '',
    power_with: '',
    power_without: '',
    notes: '',
    playback: 'loop',
  })
  // the dataset rate is the default until the operator types a rate (derived during render, no effect needed)
  const datasetRate = dataset.data?.signal.sample_rate_hz
  const sampleRate = form.sample_rate_hz !== '' ? form.sample_rate_hz : datasetRate ? String(datasetRate) : ''
  const set = (key: keyof Form) => (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => setForm((f) => ({ ...f, [key]: e.target.value }))
  const ready = !!withFile && form.pa.trim() !== '' && form.capture_chain.trim() !== '' && form.drive.trim() !== '' && Number(sampleRate) > 0 && !uploading && !submit.isPending

  async function start() {
    if (!withFile) return
    setUploading(true)
    setUploadError(null)
    try {
      const upload = async (file: File) => {
        const fd = new FormData()
        fd.append('file', file)
        return api.upload<UploadResult>('/datasets/upload', fd)
      }
      const withUpload = await upload(withFile)
      const withoutUpload = withoutFile ? await upload(withoutFile) : null
      const config: ExperimentConfigInput = {
        task: 'evaluate_measured',
        dataset: { id: run.dataset_id ?? '' },
        model: { key: run.model_key ?? 'gru' },
        evaluation: { evidence_type: 'dpd_measured' },
        measurement: {
          apply_run_id: run.run_id,
          with_dpd: { path: withUpload.path, declared_output_power_dbm: number(form.power_with) },
          without_dpd: withoutUpload ? { path: withoutUpload.path, declared_output_power_dbm: number(form.power_without) } : null,
          conditions: {
            pa: form.pa.trim(),
            capture_chain: form.capture_chain.trim(),
            sample_rate_hz: Number(sampleRate),
            drive: form.drive.trim(),
            gain_db: number(form.gain_db),
            calibration: form.calibration.trim() || 'none',
            measured_at: new Date(form.measured_at).toISOString(),
            temperature_c: number(form.temperature_c),
            operator: form.operator.trim() || null,
            notes: form.notes.trim() || null,
          },
          source: 'manual',
          playback: form.playback,
        },
      }
      submit.mutate(
        { config, idempotency_key: idempotencyKey.current },
        {
          onSuccess: (created) => {
            onClose()
            navigate(`/runs/${encodeURIComponent(created.run_id)}`)
          },
        },
      )
    } catch (err) {
      setUploadError(err)
    } finally {
      setUploading(false)
    }
  }

  return (
    <Dialog open onClose={onClose} fullWidth maxWidth="md" aria-labelledby="measure-title">
      <DialogTitle id="measure-title">{t('run.measure.title')}</DialogTitle>
      <DialogContent>
        <Stack spacing={2} sx={{ mt: 1 }}>
          <Typography variant="body2">{t('run.measure.help')}</Typography>
          <Alert severity="info">{t('run.measure.attestation')}</Alert>
          <CaptureInput label={t('run.measure.with')} help={t('run.measure.with.help')} file={withFile} onFile={setWithFile} />
          <CaptureInput label={t('run.measure.without')} help={t('run.measure.without.help')} file={withoutFile} onFile={setWithoutFile} />
          <Grid container spacing={2}>
            <Grid size={{ xs: 12, sm: 6 }}>
              <TextField fullWidth required label={t('run.measure.pa')} value={form.pa} onChange={set('pa')} />
            </Grid>
            <Grid size={{ xs: 12, sm: 6 }}>
              <TextField fullWidth required label={t('run.measure.drive')} value={form.drive} onChange={set('drive')} helperText={t('run.measure.drive.help')} />
            </Grid>
            <Grid size={{ xs: 12 }}>
              <TextField fullWidth required label={t('run.measure.chain')} value={form.capture_chain} onChange={set('capture_chain')} helperText={t('run.measure.chain.help')} />
            </Grid>
            <Grid size={{ xs: 12, sm: 4 }}>
              <TextField fullWidth required type="number" label={t('run.measure.rate')} value={sampleRate} onChange={set('sample_rate_hz')} helperText={t('run.measure.rate.help')} slotProps={{ htmlInput: { min: 1, step: 'any' } }} />
            </Grid>
            <Grid size={{ xs: 12, sm: 4 }}>
              <TextField fullWidth type="number" label={t('run.measure.power_with')} value={form.power_with} onChange={set('power_with')} slotProps={{ htmlInput: { step: 'any' } }} />
            </Grid>
            <Grid size={{ xs: 12, sm: 4 }}>
              <TextField fullWidth type="number" label={t('run.measure.power_without')} value={form.power_without} onChange={set('power_without')} slotProps={{ htmlInput: { step: 'any' } }} />
            </Grid>
            <Grid size={{ xs: 12, sm: 4 }}>
              <TextField fullWidth type="number" label={t('run.measure.gain')} value={form.gain_db} onChange={set('gain_db')} slotProps={{ htmlInput: { step: 'any' } }} />
            </Grid>
            <Grid size={{ xs: 12, sm: 4 }}>
              <TextField fullWidth type="number" label={t('run.measure.temperature')} value={form.temperature_c} onChange={set('temperature_c')} slotProps={{ htmlInput: { step: 'any' } }} />
            </Grid>
            <Grid size={{ xs: 12, sm: 4 }}>
              <TextField fullWidth select label={t('run.measure.playback')} value={form.playback} onChange={set('playback')} helperText={t('run.measure.playback.help')}>
                <MenuItem value="loop">{t('run.measure.playback.loop')}</MenuItem>
                <MenuItem value="single">{t('run.measure.playback.single')}</MenuItem>
              </TextField>
            </Grid>
            <Grid size={{ xs: 12, sm: 6 }}>
              <TextField fullWidth label={t('run.measure.calibration')} value={form.calibration} onChange={set('calibration')} helperText={t('run.measure.calibration.help')} />
            </Grid>
            <Grid size={{ xs: 12, sm: 3 }}>
              <TextField fullWidth type="datetime-local" label={t('run.measure.measured_at')} value={form.measured_at} onChange={set('measured_at')} slotProps={{ inputLabel: { shrink: true } }} />
            </Grid>
            <Grid size={{ xs: 12, sm: 3 }}>
              <TextField fullWidth label={t('run.measure.operator')} value={form.operator} onChange={set('operator')} />
            </Grid>
            <Grid size={{ xs: 12 }}>
              <TextField fullWidth multiline minRows={2} label={t('run.measure.notes')} value={form.notes} onChange={set('notes')} />
            </Grid>
          </Grid>
          {uploadError != null && <ErrorState error={uploadError} />}
          {submit.isError && <ErrorState error={submit.error} />}
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>{t('common.back')}</Button>
        <Button variant="contained" disabled={!ready} onClick={() => void start()}>
          {uploading ? t('run.measure.uploading') : t('run.measure.submit')}
        </Button>
      </DialogActions>
    </Dialog>
  )
}
