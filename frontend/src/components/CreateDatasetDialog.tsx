import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Dialog from '@mui/material/Dialog'
import DialogActions from '@mui/material/DialogActions'
import DialogContent from '@mui/material/DialogContent'
import DialogTitle from '@mui/material/DialogTitle'
import Grid from '@mui/material/Grid'
import MenuItem from '@mui/material/MenuItem'
import Stack from '@mui/material/Stack'
import Table from '@mui/material/Table'
import TableBody from '@mui/material/TableBody'
import TableCell from '@mui/material/TableCell'
import TableHead from '@mui/material/TableHead'
import TableRow from '@mui/material/TableRow'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { useMutation } from '@tanstack/react-query'
import { useState } from 'react'
import { api } from '@/api/client'
import { previewCsv, useCreateCsvDataset, useDatasetImportDefaults, type CsvInspection, type CsvOptions, type CsvPreviewRequest, type DatasetImportDefaults, type UploadResult } from '@/api/datasets'
import type { DatasetOrigin } from '@/api/types'
import { message, formatNumber, getLanguage, t, type MessageKey } from '@/i18n'
import { ORIGINS } from './ManifestDialog'
import { SignalFields, emptySignalForm, signalSpecFrom } from './SignalFields'
import { ErrorState, LoadingState } from './StateBlock'
import { WorkflowSteps } from './WorkflowSteps'
import { GUIDE_BACKDROP } from './DatasetGuide'
import { ResetButton } from './ResetButton'

const SPLITS = ['train', 'val', 'test'] as const
const INITIAL_OPTIONS: CsvOptions = { format: 'auto', header: 'auto', mapping: {} }
const stamp = (source: CsvPreviewRequest['source'] | null, options: CsvOptions) => JSON.stringify({ source, options })
const FIX_KEYS: Record<string, MessageKey> = {
  numeric: 'datasets.create.fixSamples', non_finite: 'datasets.create.fixSamples',
  columns: 'datasets.create.fixFormat', encoding: 'datasets.create.fixFormat', csv_syntax: 'datasets.create.fixFormat', file: 'datasets.create.fixFormat', read: 'datasets.create.fixFormat',
  row_width: 'datasets.create.fixRow', header: 'datasets.create.fixMapping', mapping: 'datasets.create.fixMapping',
  split: 'datasets.create.fixSplit', empty: 'datasets.create.fixEmpty',
}

/** Whole-file validation precedes split review; changing a setting invalidates its confirmation. */
export function CreateDatasetDialog({ onClose, onImported, guided = false, onSkipGuide }: { onClose: () => void; onImported: (id: string) => void; guided?: boolean; onSkipGuide?: () => void }) {
  const defaults = useDatasetImportDefaults()
  const [active, setActive] = useState(0)
  const [source, setSource] = useState<CsvPreviewRequest['source'] | null>(null)
  const [filename, setFilename] = useState('')
  const [options, setOptions] = useState<CsvOptions>(INITIAL_OPTIONS)
  const [report, setReport] = useState<CsvInspection | null>(null)
  const [csvStamp, setCsvStamp] = useState('')
  const [reviewStamp, setReviewStamp] = useState('')
  const [ratios, setRatios] = useState<Partial<Record<(typeof SPLITS)[number], string>>>({})
  const [guard, setGuard] = useState<string | null>(null)
  const [datasetId, setDatasetId] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [origin, setOrigin] = useState<DatasetOrigin>('unknown')
  const [signal, setSignal] = useState(emptySignalForm)
  const [fileError, setFileError] = useState('')
  const create = useCreateCsvDataset()
  const percentages = Object.fromEntries(SPLITS.map((key) => [key, ratios[key] ?? (defaults.data ? String((defaults.data.ratios[key] ?? 0) * 100) : '')]))
  const guardValue = guard ?? String(defaults.data?.guard_samples ?? '')
  const split: DatasetImportDefaults = { ratios: Object.fromEntries(SPLITS.map((key) => [key, Number(percentages[key]) / 100])), guard_samples: Number(guardValue) }
  const numbersValid = SPLITS.every((key) => (percentages[key] ?? '').trim() !== '' && Number.isFinite(split.ratios[key]) && (split.ratios[key] ?? 0) > 0) && guardValue.trim() !== '' && Number.isSafeInteger(split.guard_samples) && split.guard_samples >= 0
  const signalValid = Object.entries(signal).every(([key, value]) => key === 'amplitude_units' || value.trim() === '' || (Number.isFinite(Number(value)) && Number(value) > 0 && (!['n_sub_ch', 'nperseg'].includes(key) || Number.isInteger(Number(value)))))
  const identityValid = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/.test(datasetId) && displayName.trim() !== ''
  const formStamp = JSON.stringify({ source, options, split, datasetId, displayName, origin, signal })
  const csvConfirmed = !!report?.data_valid && csvStamp === stamp(source, options)
  const reviewed = !!report?.valid && formStamp === reviewStamp
  const scan = useMutation({
    mutationFn: ({ body, review }: { body: CsvPreviewRequest; review: boolean }) => previewCsv(body).then((result) => ({ result, body, review })),
    onSuccess: ({ result, body, review }) => {
      setReport(result)
      setOptions(result.options)
      setCsvStamp(stamp(body.source, result.options))
      if (review && result.valid) {
        setReviewStamp(JSON.stringify({ source: body.source, options: result.options, split: body.split, datasetId, displayName, origin, signal }))
        setActive(2)
      }
    },
  })
  const upload = useMutation({
    mutationFn: (file: File) => {
      const data = new FormData()
      data.append('file', file, file.name)
      return api.upload<UploadResult>('/datasets/upload', data)
    },
    onSuccess: (uploaded) => {
      const selected = { root_id: uploaded.root_id, path: uploaded.path }
      setSource(selected)
      if (defaults.data) scan.mutate({ body: { source: selected, options: INITIAL_OPTIONS, split: defaults.data }, review: false })
    },
  })
  const busy = upload.isPending || scan.isPending || create.isPending
  const validate = (review = false) => {
    if (source && defaults.data) scan.mutate({ body: { source, options, split: numbersValid ? split : defaults.data, signal: signalSpecFrom(signal) }, review })
  }
  const chooseFile = (file: File) => {
    setFileError('')
    setReport(null)
    setSource(null)
    setOptions(INITIAL_OPTIONS)
    setActive(0)
    upload.reset(); scan.reset(); create.reset()
    if (!file.name.toLowerCase().endsWith('.csv')) { setFileError(t('datasets.create.csvOnly')); return }
    if (file.size > 25 * 1024 * 1024) { setFileError(t('datasets.upload.limit')); return }
    const name = file.name.replace(/\.csv$/i, '')
    setFilename(file.name)
    setDatasetId(name.replace(/[^A-Za-z0-9._-]+/g, '-').replace(/^[^A-Za-z0-9]+/, '').slice(0, 64) || 'my-dataset')
    setDisplayName(name)
    upload.mutate(file)
  }
  const reset = () => {
    setActive(0); setSource(null); setFilename(''); setOptions(INITIAL_OPTIONS)
    setReport(null); setCsvStamp(''); setReviewStamp(''); setRatios({}); setGuard(null)
    setDatasetId(''); setDisplayName(''); setOrigin('unknown'); setSignal(emptySignalForm); setFileError('')
    upload.reset(); scan.reset(); create.reset()
  }
  return <Dialog open onClose={busy ? undefined : onClose} fullWidth maxWidth="md" aria-labelledby="create-dataset-title" slotProps={{ backdrop: guided ? GUIDE_BACKDROP : undefined }}>
    <DialogTitle id="create-dataset-title">{t('datasets.create.title')}</DialogTitle>
    <DialogContent dividers>
      {defaults.isPending ? <LoadingState /> : defaults.isError ? <ErrorState error={defaults.error} onRetry={() => void defaults.refetch()} /> : <Stack spacing={2}>
        <WorkflowSteps active={active} ariaLabel={t('datasets.create.title')} labels={[t('datasets.create.fileStep'), t('datasets.create.splitStep'), t('datasets.create.reviewStep')]} completed={[csvConfirmed, reviewed, false]} onChange={setActive} canOpen={(step) => !busy && (step === 0 || (step === 1 && csvConfirmed) || (step === 2 && reviewed))} />
        {guided && <Alert severity="info">{t(active === 0 ? 'guide.csv.file' : active === 1 ? 'guide.csv.split' : 'guide.csv.review')}</Alert>}
        <Box role="tabpanel" id={`workflow-panel-${active}`} aria-labelledby={`workflow-step-${active}`}>
          {active === 0 && <Stack spacing={2}>
            <Typography>{t('datasets.create.help')}</Typography>
            <Grid container spacing={2}>
              {(['complex_pair', 'iq_columns'] as const).map((kind) => <Grid key={kind} size={{ xs: 12, sm: 6 }}>
                <Box sx={{ border: 1, borderColor: 'divider', borderRadius: 1.5, p: 1.5, bgcolor: 'background.default' }}>
                  <Typography variant="body2" sx={{ fontWeight: 700 }}>{t(kind === 'complex_pair' ? 'datasets.create.complex' : 'datasets.create.real')}</Typography>
                  <Box component="pre" sx={{ fontSize: 12, mb: 0, overflowX: 'auto' }}>{kind === 'complex_pair' ? 'input,output\n0.1+0.2j,0.3-0.4i' : 'I_in,Q_in,I_out,Q_out\n0.1,0.2,0.3,-0.4'}</Box>
                </Box>
              </Grid>)}
            </Grid>
            <Stack direction="row" spacing={2} sx={{ alignItems: 'center' }}>
              <Button component="label" variant="contained" disabled={busy}>{t('datasets.create.choose')}<input aria-label={t('datasets.create.choose')} hidden type="file" accept=".csv,text/csv" onChange={(event) => { const file = event.target.files?.[0]; if (file) chooseFile(file); event.target.value = '' }} /></Button>
              <Typography variant="body2" sx={{ overflowWrap: 'anywhere' }}>{filename}</Typography>
            </Stack>
            <Typography color="text.secondary" variant="caption">{t('datasets.create.formatHelp')}</Typography>
            <Typography color="text.secondary" variant="caption">{t('datasets.upload.limit')}</Typography>
            {fileError && <Alert severity="error">{fileError}</Alert>}
            {upload.isPending && <Alert severity="info" role="status">{t('datasets.upload.checking')}</Alert>}
            {upload.data?.validation?.status === 'passed' && <Alert severity="success" role="status" data-testid="csv-upload-validated">{t('datasets.upload.passed', { samples: formatNumber(upload.data.validation.n_samples) })}</Alert>}
            {source && <>
              <Grid container spacing={2}>
                <Grid size={{ xs: 12, sm: 6 }}><TextField select fullWidth size="small" label={t('datasets.create.layout')} value={options.format} disabled={busy} onChange={(e) => setOptions({ ...options, format: e.target.value as CsvOptions['format'], mapping: {} })}>
                  <MenuItem value="auto">{t('datasets.create.auto')}</MenuItem><MenuItem value="complex_pair">{t('datasets.create.complex')}</MenuItem><MenuItem value="iq_columns">{t('datasets.create.real')}</MenuItem>
                </TextField></Grid>
                <Grid size={{ xs: 12, sm: 6 }}><TextField select fullWidth size="small" label={t('datasets.create.header')} value={options.header} disabled={busy} onChange={(e) => setOptions({ ...options, header: e.target.value as CsvOptions['header'], mapping: {} })}>
                  <MenuItem value="auto">{t('datasets.create.auto')}</MenuItem><MenuItem value="present">{t('datasets.create.headerYes')}</MenuItem><MenuItem value="absent">{t('datasets.create.headerNo')}</MenuItem>
                </TextField></Grid>
              </Grid>
              {!!report?.columns.length && <Grid container spacing={1}>
                {(options.format === 'complex_pair' ? ['input', 'output'] : ['I_in', 'Q_in', 'I_out', 'Q_out']).map((role) => <Grid key={role} size={{ xs: 6, sm: 3 }}><TextField select fullWidth size="small" label={role} disabled={busy} value={options.mapping?.[role] ?? ''} onChange={(e) => setOptions({ ...options, mapping: { ...options.mapping, [role]: Number(e.target.value) } })}>
                  {report.columns.map((column, index) => <MenuItem key={index} value={index}>{index + 1}. {column}</MenuItem>)}
                </TextField></Grid>)}
              </Grid>}
              {!!report?.preview.length && <Box sx={{ overflowX: 'auto' }}><Table size="small" aria-label={t('datasets.import.preview')}><TableHead><TableRow>{report.columns.map((column, i) => <TableCell key={i}>{column}</TableCell>)}</TableRow></TableHead><TableBody>{report.preview.map((row, i) => <TableRow key={i}>{row.map((cell, j) => <TableCell key={j} sx={{ fontFamily: 'monospace', whiteSpace: 'nowrap' }}>{cell}</TableCell>)}</TableRow>)}</TableBody></Table></Box>}
              <Button variant="outlined" onClick={() => validate()} disabled={busy}>{t('datasets.create.validate')}</Button>
            </>}
            {csvConfirmed && <Alert severity="success">{t('datasets.create.valid', { count: formatNumber(report!.n_samples) })}</Alert>}
          </Stack>}
          {active === 1 && <Stack spacing={2}>
            <Typography>{t('datasets.create.splitHelp')}</Typography>
            <Grid container spacing={2}>{SPLITS.map((key) => <Grid key={key} size={{ xs: 4 }}><TextField fullWidth size="small" label={`${key} (%)`} value={percentages[key]} disabled={busy} onChange={(e) => setRatios({ ...ratios, [key]: e.target.value })} slotProps={{ htmlInput: { inputMode: 'decimal' } }} /></Grid>)}</Grid>
            <Typography variant="body2">{t('datasets.create.total', { value: SPLITS.reduce((sum, key) => sum + (Number(percentages[key]) || 0), 0) })}</Typography>
            {!numbersValid && <Alert severity="error">{t('datasets.create.numbers')}</Alert>}
            <Box component="details"><Box component="summary" sx={{ cursor: 'pointer', color: 'primary.main' }}>{t('datasets.create.guardSettings')}</Box><Stack spacing={1} sx={{ pt: 1 }}><TextField size="small" label={t('datasets.import.guard')} value={guardValue} disabled={busy} onChange={(e) => setGuard(e.target.value)} slotProps={{ htmlInput: { inputMode: 'numeric' } }} /><Typography variant="caption" color="text.secondary">{t('datasets.create.guardHelp', { count: defaults.data.guard_samples })}</Typography></Stack></Box>
            <Grid container spacing={2}>
              <Grid size={{ xs: 12, sm: 6 }}><TextField size="small" fullWidth label={t('datasets.import.name')} value={displayName} disabled={busy} onChange={(e) => setDisplayName(e.target.value)} /></Grid>
              <Grid size={{ xs: 12, sm: 6 }}><TextField size="small" fullWidth label={t('datasets.import.id')} value={datasetId} disabled={busy} onChange={(e) => setDatasetId(e.target.value)} helperText={t('datasets.create.idHelp')} /></Grid>
            </Grid>
            <TextField select size="small" label={t('datasets.import.origin')} value={origin} disabled={busy} onChange={(e) => setOrigin(e.target.value as DatasetOrigin)}>{ORIGINS.map((value) => <MenuItem key={value} value={value}>{value}</MenuItem>)}</TextField>
            <Box component="details"><Box component="summary" sx={{ cursor: 'pointer', color: 'primary.main' }}>{t('datasets.create.signalOptional')}</Box><Stack spacing={1} sx={{ pt: 2 }}><Typography variant="caption" color="text.secondary">{t('datasets.create.signalHelp')}</Typography><Box component="fieldset" disabled={busy} sx={{ border: 0, p: 0, m: 0 }}><SignalFields value={signal} onChange={setSignal} /></Box></Stack></Box>
            {!signalValid && <Alert severity="error">{t('datasets.create.signalInvalid')}</Alert>}
          </Stack>}
          {active === 2 && report && <Stack spacing={2}>
            <Alert severity="success">{t('datasets.create.ready', { count: formatNumber(report.n_samples) })}</Alert>
            <Typography variant="h3">{displayName}</Typography><Typography variant="body2">{filename} · {datasetId} · {origin}</Typography>
            <Table size="small" aria-label={t('datasets.create.splitStep')}><TableHead><TableRow><TableCell>{t('datasets.create.partition')}</TableCell><TableCell align="right">%</TableCell><TableCell align="right">{t('datasets.columns.samples')}</TableCell><TableCell align="right">{t('datasets.create.range')}</TableCell></TableRow></TableHead><TableBody>{SPLITS.map((key) => <TableRow key={key}><TableCell>{key}</TableCell><TableCell align="right">{percentages[key]}</TableCell><TableCell align="right">{formatNumber(report.split_counts?.[key] ?? 0)}</TableCell><TableCell align="right">[{report.boundaries?.[key]?.join(', ')})</TableCell></TableRow>)}</TableBody></Table>
            <Typography variant="body2" color="text.secondary">{t('datasets.create.guardReview', { count: report.split.guard_samples, total: report.split.guard_samples * 2 })}</Typography>
            <Typography variant="body2" color="text.secondary">{t('datasets.create.afterCreate')}</Typography>
          </Stack>}
        </Box>
        {!!report?.issues.length && active !== 2 && <Alert severity="error"><Typography sx={{ fontWeight: 700 }}>{t('datasets.create.problems', { count: report.issue_count })}</Typography><Stack spacing={1} sx={{ mt: 1 }}>{report.issues.map((problem, i) => <Box key={i}><Typography variant="body2" sx={{ fontWeight: 600 }}>{problem.line ? `${t('datasets.create.line')} ${problem.line}` : ''}{problem.column ? ` · ${problem.column}` : ''}{problem.line || problem.column ? ': ' : ''}{message(problem.message)}</Typography><Typography variant="body2">{t('datasets.create.fix')}: {getLanguage() !== 'en' && FIX_KEYS[problem.code] ? t(FIX_KEYS[problem.code]!) : message(problem.fix)}</Typography></Box>)}</Stack></Alert>}
        {(upload.isPending || scan.isPending) && <LoadingState label={t('datasets.create.scanning')} />}
        {upload.isError && <ErrorState error={upload.error} />}{scan.isError && <ErrorState error={scan.error} />}{create.isError && <ErrorState error={create.error} />}
      </Stack>}
    </DialogContent>
    <DialogActions>
      <ResetButton onReset={reset} disabled={busy} />
      {guided && <Button onClick={onSkipGuide} disabled={busy}>{t('guide.skip')}</Button>}
      <Box sx={{ flex: 1 }} />
      <Button onClick={onClose} disabled={busy}>{t('form.cancel')}</Button>
      {active > 0 && <Button disabled={busy} onClick={() => setActive(active - 1)}>{t('workflow.back')}</Button>}
      {active === 0 && <Button variant="contained" disabled={!csvConfirmed || busy} onClick={() => setActive(1)}>{t('workflow.next')}</Button>}
      {active === 1 && <Button variant="contained" disabled={!csvConfirmed || !numbersValid || !signalValid || !identityValid || busy} onClick={() => validate(true)}>{t('datasets.create.reviewStep')}</Button>}
      {active === 2 && <Button variant="contained" disabled={!reviewed || busy} onClick={() => { if (source && report?.sha256) create.mutate({ source, options, split, dataset_id: datasetId, display_name: displayName, origin, signal: signalSpecFrom(signal), expected_sha256: report.sha256 }, { onSuccess: (result) => onImported(result.dataset_id) }) }}>{create.isPending ? t('datasets.import.importing') : t('datasets.create.submit')}</Button>}
    </DialogActions>
  </Dialog>
}
