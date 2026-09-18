import Autocomplete from '@mui/material/Autocomplete'
import TextField from '@mui/material/TextField'
import type { AnalyzerDataset } from '@/api/signalAnalyzer'
import { t } from '@/i18n'

/** Search names without losing the dataset identity behind a reused signal. */
export function SignalDatasetSelect({ datasets, value, onChange, label, disabled = false }: {
  datasets: AnalyzerDataset[]; value?: AnalyzerDataset; onChange: (dataset: AnalyzerDataset) => void; label: string; disabled?: boolean
}) {
  const options = [...datasets].sort((a, b) => a.kind.localeCompare(b.kind) || a.name.localeCompare(b.name))
  return <Autocomplete options={options} value={value ?? null} disableClearable={!!value} disabled={disabled} fullWidth size="small"
    getOptionLabel={option => option.name} getOptionKey={option => option.dataset_id} isOptionEqualToValue={(a, b) => a.dataset_id === b.dataset_id}
    groupBy={option => t(`signalDataset.kind.${option.kind}`)}
    onChange={(_, option) => { if (option) onChange(option) }}
    renderOption={(props, option) => {
      const { key, ...rest } = props
      return <li key={key} {...rest} style={{ whiteSpace: 'normal', overflowWrap: 'anywhere' }}>{option.name}</li>
    }}
    renderInput={params => <TextField {...params} label={label} title={value?.name} placeholder={t('signalDataset.search')} />}
  />
}
