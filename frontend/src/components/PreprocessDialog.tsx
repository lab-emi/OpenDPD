import Button from '@mui/material/Button'
import Checkbox from '@mui/material/Checkbox'
import Dialog from '@mui/material/Dialog'
import DialogActions from '@mui/material/DialogActions'
import DialogContent from '@mui/material/DialogContent'
import DialogTitle from '@mui/material/DialogTitle'
import FormControlLabel from '@mui/material/FormControlLabel'
import FormGroup from '@mui/material/FormGroup'
import Grid from '@mui/material/Grid'
import MenuItem from '@mui/material/MenuItem'
import Stack from '@mui/material/Stack'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { useMutation } from '@tanstack/react-query'
import { useState } from 'react'
import { previewPreprocess, useCreateVersion, type PreprocessRequest } from '@/api/datasets'
import type { DiagnosticReport, PreprocessingParams } from '@/api/types'
import { t } from '@/i18n'
import { DiagnosticItem } from '@/components/DiagnosticItem'
import { ErrorState } from '@/components/StateBlock'

interface Fields {
  delay: string
  gainDb: string
  phaseDeg: string
  interpolate: boolean
  outliers: boolean
  normalize: boolean
  base: string
  version: string
}

const num = (s: string, fallback: number) => {
  const v = Number(s)
  return s.trim() === '' || !Number.isFinite(v) ? fallback : v
}
const asNumber = (v: unknown) => (typeof v === 'number' && Number.isFinite(v) ? v : 0)
const fmt = (v: number) => String(Math.round(v * 1000) / 1000)

/** Mirrors opendpd.core.doctor.estimates_from_report: the numbers a user can accept as corrections. */
function estimatesFrom(report: DiagnosticReport) {
  const by = new Map((report.items ?? []).map((i) => [i.code, i]))
  const align = by.get('time_misalignment') ?? by.get('alignment_ok')
  const gain = by.get('linear_gain_phase')
  return {
    delay: asNumber(align?.evidence?.['delay_samples']),
    gainDb: asNumber(gain?.evidence?.['gain_db']),
    phaseDeg: asNumber(gain?.evidence?.['phase_deg']),
    nonFinite: by.has('non_finite_samples'),
    outliers: by.has('output_outliers'),
  }
}

export function PreprocessDialog({ datasetId, versions, report, onClose, onCreated }: { datasetId: string; versions: string[]; report: DiagnosticReport | null; onClose: () => void; onCreated: (version: string) => void }) {
  const [fields, setFields] = useState<Fields>({ delay: '0', gainDb: '0', phaseDeg: '0', interpolate: false, outliers: false, normalize: false, base: '', version: '' })
  const base = versions.includes(fields.base) ? fields.base : (versions[0] ?? 'raw-v1')
  const preview = useMutation({ mutationFn: (body: PreprocessRequest) => previewPreprocess(datasetId, body) })
  const create = useCreateVersion(datasetId)
  const estimates = report ? estimatesFrom(report) : null
  const params = (): PreprocessingParams => ({
    delay_samples: num(fields.delay, 0),
    gain_db: num(fields.gainDb, 0),
    phase_deg: num(fields.phaseDeg, 0),
    interpolate_non_finite: fields.interpolate,
    remove_outliers: fields.outliers,
    normalize: fields.normalize ? 'peak_input' : 'none',
  })
  // Any edit invalidates the preview: what is shown must be what would be written.
  const edit = (patch: Partial<Fields>) => {
    preview.reset()
    setFields((f) => ({ ...f, ...patch }))
  }
  const text = (k: 'delay' | 'gainDb' | 'phaseDeg' | 'version') => (e: { target: { value: string } }) => edit({ [k]: e.target.value })
  const check = (k: 'interpolate' | 'outliers' | 'normalize') => (_: unknown, checked: boolean) => edit({ [k]: checked })
  const useEstimates = () => estimates && edit({ delay: fmt(estimates.delay), gainDb: fmt(estimates.gainDb), phaseDeg: fmt(estimates.phaseDeg), interpolate: estimates.nonFinite, outliers: estimates.outliers })
  const version = fields.version.trim()
  return (
    <Dialog open onClose={onClose} fullWidth maxWidth="md" aria-labelledby="preprocess-title">
      <DialogTitle id="preprocess-title">{t('datasets.preprocess.title')}</DialogTitle>
      <DialogContent dividers>
        <Stack spacing={2} sx={{ pt: 1 }}>
          <Typography variant="body2" color="text.secondary">
            {t('datasets.preprocess.help')}
          </Typography>
          <Grid container spacing={2}>
            <Grid size={{ xs: 12, md: 3 }}>
              <TextField select fullWidth size="small" label={t('datasets.preprocess.base')} value={base} onChange={(e) => edit({ base: e.target.value })}>
                {versions.map((v) => (
                  <MenuItem key={v} value={v}>
                    {v}
                  </MenuItem>
                ))}
              </TextField>
            </Grid>
            <Grid size={{ xs: 4, md: 3 }}>
              <TextField fullWidth size="small" label={t('datasets.preprocess.delay')} value={fields.delay} onChange={text('delay')} slotProps={{ htmlInput: { inputMode: 'decimal' } }} />
            </Grid>
            <Grid size={{ xs: 4, md: 3 }}>
              <TextField fullWidth size="small" label={t('datasets.preprocess.gain')} value={fields.gainDb} onChange={text('gainDb')} slotProps={{ htmlInput: { inputMode: 'decimal' } }} />
            </Grid>
            <Grid size={{ xs: 4, md: 3 }}>
              <TextField fullWidth size="small" label={t('datasets.preprocess.phase')} value={fields.phaseDeg} onChange={text('phaseDeg')} slotProps={{ htmlInput: { inputMode: 'decimal' } }} />
            </Grid>
          </Grid>
          <FormGroup row>
            <FormControlLabel control={<Checkbox checked={fields.interpolate} onChange={check('interpolate')} />} label={t('datasets.preprocess.interpolate')} />
            <FormControlLabel control={<Checkbox checked={fields.outliers} onChange={check('outliers')} />} label={t('datasets.preprocess.outliers')} />
            <FormControlLabel control={<Checkbox checked={fields.normalize} onChange={check('normalize')} />} label={t('datasets.preprocess.normalize')} />
          </FormGroup>
          <Stack direction="row" spacing={1}>
            <Button variant="outlined" onClick={useEstimates} disabled={!estimates}>
              {t('datasets.preprocess.fromDoctor')}
            </Button>
            <Button variant="outlined" onClick={() => preview.mutate({ params: params(), base_version: base, version: null })} disabled={preview.isPending}>
              {t('datasets.preprocess.preview')}
            </Button>
          </Stack>
          {preview.isError && <ErrorState error={preview.error} />}
          {preview.data && (
            <section aria-label={t('datasets.preprocess.preview')}>
              <Typography gutterBottom>{t('datasets.preprocess.previewResult', { before: preview.data.n_samples_before.toLocaleString(), after: preview.data.n_samples_after.toLocaleString() })}</Typography>
              <Stack spacing={1}>
                {(preview.data.report_after.items ?? []).map((item) => (
                  <DiagnosticItem key={item.code} item={item} />
                ))}
              </Stack>
            </section>
          )}
          <TextField size="small" label={t('datasets.preprocess.version')} value={fields.version} onChange={text('version')} helperText={t('datasets.preprocess.version.help')} />
          {create.isError && <ErrorState error={create.error} />}
        </Stack>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>{t('form.cancel')}</Button>
        <Button variant="contained" disabled={!version || create.isPending} onClick={() => create.mutate({ params: params(), base_version: base, version }, { onSuccess: (v) => onCreated(v.version) })}>
          {t('datasets.preprocess.create')}
        </Button>
      </DialogActions>
    </Dialog>
  )
}
