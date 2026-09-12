import Grid from '@mui/material/Grid'
import MenuItem from '@mui/material/MenuItem'
import TextField from '@mui/material/TextField'
import type { SignalSpec } from '@/api/types'
import { t, type MessageKey } from '@/i18n'

/** String-typed form model of SignalSpec so partially typed numbers never become NaN in the request. */
export interface SignalForm {
  sample_rate_hz: string
  bandwidth_hz: string
  sub_channel_bandwidth_hz: string
  n_sub_ch: string
  nperseg: string
  amplitude_units: SignalSpec['amplitude_units']
}

const NUMERIC = ['sample_rate_hz', 'bandwidth_hz', 'sub_channel_bandwidth_hz', 'n_sub_ch', 'nperseg'] as const
const LABEL: Record<(typeof NUMERIC)[number], MessageKey> = {
  sample_rate_hz: 'signal.sample_rate_hz',
  bandwidth_hz: 'signal.bandwidth_hz',
  sub_channel_bandwidth_hz: 'signal.sub_channel_bandwidth_hz',
  n_sub_ch: 'signal.n_sub_ch',
  nperseg: 'signal.nperseg',
}
const UNITS: SignalSpec['amplitude_units'][] = ['unknown', 'normalized', 'volts']

const str = (v: number | null | undefined) => (v === null || v === undefined ? '' : String(v))
const numOrNull = (s: string) => {
  const v = Number(s)
  return s.trim() === '' || !Number.isFinite(v) ? null : v
}

export const emptySignalForm = (): SignalForm => ({ sample_rate_hz: '', bandwidth_hz: '', sub_channel_bandwidth_hz: '', n_sub_ch: '', nperseg: '', amplitude_units: 'unknown' })
export const signalFormFrom = (s: SignalSpec): SignalForm => ({
  sample_rate_hz: str(s.sample_rate_hz),
  bandwidth_hz: str(s.bandwidth_hz),
  sub_channel_bandwidth_hz: str(s.sub_channel_bandwidth_hz),
  n_sub_ch: str(s.n_sub_ch),
  nperseg: str(s.nperseg),
  amplitude_units: s.amplitude_units,
})
/** Fields not shown in the form (modulation, standard) are carried over from `base`. */
export const signalSpecFrom = (f: SignalForm, base?: SignalSpec): SignalSpec => ({
  ...base,
  amplitude_units: f.amplitude_units,
  sample_rate_hz: numOrNull(f.sample_rate_hz),
  bandwidth_hz: numOrNull(f.bandwidth_hz),
  sub_channel_bandwidth_hz: numOrNull(f.sub_channel_bandwidth_hz),
  n_sub_ch: numOrNull(f.n_sub_ch),
  nperseg: numOrNull(f.nperseg),
})

export function SignalFields({ value, onChange }: { value: SignalForm; onChange: (next: SignalForm) => void }) {
  const set = (k: keyof SignalForm) => (e: { target: { value: string } }) => onChange({ ...value, [k]: e.target.value })
  return (
    <Grid container spacing={2}>
      {NUMERIC.map((k) => (
        <Grid key={k} size={{ xs: 6, md: 4 }}>
          <TextField fullWidth size="small" label={t(LABEL[k])} value={value[k]} onChange={set(k)} slotProps={{ htmlInput: { inputMode: 'decimal' } }} />
        </Grid>
      ))}
      <Grid size={{ xs: 6, md: 4 }}>
        <TextField select fullWidth size="small" label={t('datasets.import.units')} value={value.amplitude_units} onChange={set('amplitude_units')}>
          {UNITS.map((u) => (
            <MenuItem key={u} value={u}>
              {u}
            </MenuItem>
          ))}
        </TextField>
      </Grid>
    </Grid>
  )
}
