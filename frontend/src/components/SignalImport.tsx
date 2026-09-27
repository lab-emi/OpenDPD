import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import UploadFileIcon from '@mui/icons-material/UploadFile'
import ArrowForwardIcon from '@mui/icons-material/ArrowForward'
import Accordion from '@mui/material/Accordion'
import AccordionDetails from '@mui/material/AccordionDetails'
import AccordionSummary from '@mui/material/AccordionSummary'
import Alert from '@mui/material/Alert'
import Button from '@mui/material/Button'
import LinearProgress from '@mui/material/LinearProgress'
import MenuItem from '@mui/material/MenuItem'
import Stack from '@mui/material/Stack'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { useState } from 'react'
import { useNavigate } from 'react-router'
import { useUploadSignal, type AnalyzerSourceInfo } from '@/api/signalAnalyzer'
import { useImportSignal } from '@/api/signalGenerator'
import { validDatasetName } from '@/utils/datasetNames'
import { formatNumber, t } from '@/i18n'
import { ErrorState } from './StateBlock'

export function SignalImport({ disabled = false }: { disabled?: boolean }) {
  const navigate = useNavigate()
  const upload = useUploadSignal()
  const admit = useImportSignal()
  const [source, setSource] = useState<AnalyzerSourceInfo | null>(null)
  const [name, setName] = useState('usr_pa_in_custom_n1')
  const [format, setFormat] = useState<'iq' | 'complex' | 'real'>('iq')
  const [iColumn, setIColumn] = useState(0)
  const [qColumn, setQColumn] = useState(1)
  const [rate, setRate] = useState('80')
  const [bandwidth, setBandwidth] = useState('20')
  const busy = disabled || upload.isPending || admit.isPending
  const columns = source?.columns ?? []
  const fs = Number(rate) * 1e6, bw = Number(bandwidth) * 1e6
  const validRate = Number.isFinite(fs) && fs >= 1 && fs <= 2e9
  const validBandwidth = Number.isFinite(bw) && bw > 0 && bw <= fs
  const validColumns = !!source && iColumn < columns.length && (format !== 'iq' || (qColumn < columns.length && iColumn !== qColumn))
    && (format === 'complex' || !source.complex_columns?.includes(iColumn))
    && (format !== 'iq' || !source.complex_columns?.includes(qColumn))
  const receive = (file: File) => {
    setSource(null); admit.reset()
    upload.mutate(file, { onSuccess: info => {
      setSource(info)
      const firstComplex = info.complex_columns?.[0]
      setFormat(firstComplex !== undefined ? 'complex' : info.columns?.length === 1 ? 'real' : 'iq')
      setIColumn(firstComplex ?? 0); setQColumn(1)
      const stem = file.name.replace(/\.csv$/i, '').replace(/[^A-Za-z0-9_.-]+/g, '-').replace(/^[^A-Za-z0-9]+/, '').slice(0, 64) || 'custom'
      setName(`usr_pa_in_${stem}_n1`)
    } })
  }
  return <Accordion disabled={disabled} data-testid="signal-import">
    <AccordionSummary expandIcon={<ExpandMoreIcon />}><Stack><Typography sx={{ fontWeight: 700 }}>{t('signalImport.title')}</Typography>
      <Typography variant="body2" color="text.secondary">{t('signalImport.summary')}</Typography></Stack></AccordionSummary>
    <AccordionDetails><Stack spacing={2}>
      <Typography variant="body2" color="text.secondary">{t('signalImport.help')}</Typography>
      <Button component="label" variant="outlined" startIcon={<UploadFileIcon />} disabled={busy} sx={{ alignSelf: 'start' }}>{t('signalImport.file')}
        <input hidden type="file" accept=".csv,text/csv" data-testid="signal-import-upload" onChange={e => {
          const file = e.target.files?.[0]; if (file) receive(file); e.target.value = ''
        }} /></Button>
      {source && <>
        <Typography sx={{ overflowWrap: 'anywhere' }}>{source.label} · {formatNumber(source.sample_count)} {t('generator.samplesUnit')}</Typography>
        <TextField label={t('generator.datasetName')} value={name} disabled={busy} error={!validDatasetName(name, 'in')}
          helperText={t('signalImport.nameHelp')} onChange={e => setName(e.target.value)} slotProps={{ htmlInput: { maxLength: 96 } }} />
        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
          <TextField fullWidth type="number" label={t('generator.fs') + ' (MHz)'} value={rate} disabled={busy} error={!validRate}
            onChange={e => setRate(e.target.value)} slotProps={{ htmlInput: { min: .000001, max: 2000, step: 'any' } }} />
          <TextField fullWidth type="number" label={t('generator.bandwidth') + ' (MHz)'} value={bandwidth} disabled={busy} error={!validBandwidth}
            onChange={e => setBandwidth(e.target.value)} slotProps={{ htmlInput: { min: .000001, max: 2000, step: 'any' } }} />
        </Stack>
        <Stack direction={{ xs: 'column', sm: 'row' }} spacing={2}>
          <TextField select fullWidth label={t('analyzer.format')} value={format} disabled={busy} onChange={e => setFormat(e.target.value as typeof format)}>
            {(['iq', 'complex', 'real'] as const).map(value => <MenuItem key={value} value={value}>{t(`analyzer.format.${value}`)}</MenuItem>)}
          </TextField>
          <TextField select fullWidth label={t('analyzer.iColumn')} value={iColumn} disabled={busy} onChange={e => setIColumn(Number(e.target.value))}>
            {columns.map((label, index) => <MenuItem key={index} value={index}>{label}</MenuItem>)}
          </TextField>
          {format === 'iq' && <TextField select fullWidth label={t('analyzer.qColumn')} value={qColumn < columns.length ? qColumn : ''} disabled={busy} onChange={e => setQColumn(Number(e.target.value))}>
            {columns.map((label, index) => <MenuItem key={index} value={index}>{label}</MenuItem>)}
          </TextField>}
        </Stack>
        {!validColumns && <Alert severity="warning">{t('signalImport.columns')}</Alert>}
        {source.sample_count < 8192 && <Alert severity="warning">{t('signalImport.short')}</Alert>}
        <Button variant="contained" endIcon={<ArrowForwardIcon />} disabled={busy || !validRate || !validBandwidth || !validColumns || !validDatasetName(name, 'in')}
          sx={{ alignSelf: 'start' }} onClick={() => admit.mutate({ upload_id: source.source.source_id, dataset_name: name,
            sample_rate_hz: fs, bandwidth_hz: bw, carrier_frequency_hz: 0, sample_format: format, i_column: iColumn, q_column: qColumn }, {
            onSuccess: result => navigate('/pa-library?' + new URLSearchParams({ input: result.signal_id, dataset: result.dataset_id! })),
          })}>{t('signalImport.use')}</Button>
      </>}
      {(upload.isPending || admit.isPending) && <LinearProgress />}
      {upload.isError && <ErrorState error={upload.error} />}
      {admit.isError && <ErrorState error={admit.error} />}
    </Stack></AccordionDetails>
  </Accordion>
}
