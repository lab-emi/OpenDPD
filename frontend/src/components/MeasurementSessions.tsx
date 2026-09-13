import { Accordion, AccordionDetails, AccordionSummary, Alert, Box, Button, Dialog, DialogActions, DialogContent, DialogTitle, MenuItem, Paper, Stack, Table, TableBody, TableCell, TableContainer, TableHead, TableRow, TextField, Typography } from '@mui/material'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import { useMutation, useQueries, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { Link as RouterLink } from 'react-router'
import { API, api } from '@/api/client'
import { useRuns } from '@/api/hooks'
import type { EvaluationResult, MeasurementSession, MeasurementSessionSpec } from '@/api/types'
import { formatNumber, message, t } from '@/i18n'
import { DownloadLink } from './DownloadLink'

type DraftCapture = { run_id: string; role: 'with_dpd' | 'without_dpd'; acquisition_id: string; acquired_at: string; calibration_id: string; excluded_reason: string }
const newCapture = (run: string): DraftCapture => ({ run_id: run, role: 'with_dpd', acquisition_id: '', acquired_at: '', calibration_id: '', excluded_reason: '' })

function CreateSession({ result, close, saved }: { result: EvaluationResult; close: () => void; saved: (session: MeasurementSession) => void }) {
  const runs = useRuns()
  const [title, setTitle] = useState('')
  const [plane, setPlane] = useState(result.rf_conditions?.power_reference_plane ?? '')
  const [tolerance, setTolerance] = useState('')
  const [interval, setInterval] = useState<'none' | 'student_t_95'>('none')
  const [captures, setCaptures] = useState<DraftCapture[]>([newCapture(result.run_id!)])
  const [records, setRecords] = useState('{"instruments": [], "calibrations": []}')
  const details = useQueries({ queries: captures.map(c => ({ queryKey: ['result', c.run_id, result.metric_profile_id], queryFn: () => api.get<EvaluationResult>(`/results/${encodeURIComponent(c.run_id)}?profile=${encodeURIComponent(result.metric_profile_id)}`), retry: false })) })
  const change = (index: number, values: Partial<DraftCapture>) => setCaptures(captures.map((c, i) => i === index ? { ...c, ...values } : c))
  const create = useMutation({
    mutationFn: () => {
      const extra: unknown = JSON.parse(records)
      if (!extra || Array.isArray(extra) || typeof extra !== 'object') throw new Error('Expected an object with instruments and calibrations')
      const spec: MeasurementSessionSpec = {
        ...extra, version: 'measurement-session-v1', title, dut: result.measurement!.conditions.pa, source: result.is_mock ? 'mock' : 'measured', profile_id: result.metric_profile_id,
        reference_plane: plane || null, operator: result.measurement!.conditions.operator,
        power_tolerance_db: tolerance.trim() ? Number(tolerance) : null, interval,
        captures: captures.map((c, i) => ({
          capture_id: `capture-${i + 1}`, acquisition_id: c.acquisition_id.trim(), run_id: c.run_id, role: c.role,
          acquired_at: c.acquired_at ? new Date(c.acquired_at).toISOString() : details[i]?.data?.measurement?.conditions.measured_at ?? (c.run_id === result.run_id ? result.measurement?.conditions.measured_at : null) ?? '',
          calibration_id: c.calibration_id || null, excluded_reason: c.excluded_reason || null,
        })),
      }
      return api.post<MeasurementSession>('/measurement-sessions', spec)
    },
    onSuccess: saved,
  })
  return <Dialog open fullWidth maxWidth="lg" onClose={create.isPending ? undefined : close}>
    <DialogTitle>{t('session.create')}</DialogTitle>
    <DialogContent dividers><Stack spacing={2}>
      <Alert severity={result.is_mock ? 'warning' : 'info'}>{result.is_mock ? 'MOCK · ' : ''}{t('session.independenceHelp')}</Alert>
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', md: '2fr 2fr 1fr 1fr' }, gap: 2 }}>
        <TextField label={t('session.title')} required value={title} onChange={e => setTitle(e.target.value)} size="small" />
        <TextField label={t('session.plane')} value={plane} onChange={e => setPlane(e.target.value)} size="small" />
        <TextField label={t('session.powerTolerance')} type="number" value={tolerance} onChange={e => setTolerance(e.target.value)} size="small" slotProps={{ htmlInput: { min: 0, max: 10, step: .1 } }} />
        <TextField select label={t('session.interval')} value={interval} onChange={e => setInterval(e.target.value as typeof interval)} size="small"><MenuItem value="none">{t('session.noInterval')}</MenuItem><MenuItem value="student_t_95">Student t · 95%</MenuItem></TextField>
      </Box>
      {captures.map((capture, index) => <Paper variant="outlined" key={`${capture.run_id}-${index}`} sx={{ p: 2 }}>
        <Stack direction="row" useFlexGap sx={{ flexWrap: 'wrap', gap: 1 }}>
          <Typography variant="body2" sx={{ flexBasis: '100%' }}>{capture.run_id} · {details[index]?.data?.measurement?.conditions.measured_at ?? t('review.missing')}</Typography>
          <TextField label={t('session.acquisition')} value={capture.acquisition_id} required onChange={e => change(index, { acquisition_id: e.target.value })} size="small" />
          <TextField label={t('session.role')} select value={capture.role} onChange={e => change(index, { role: e.target.value as DraftCapture['role'] })} size="small"><MenuItem value="with_dpd">with DPD</MenuItem><MenuItem value="without_dpd">without DPD</MenuItem></TextField>
          <TextField label={t('session.time')} type="datetime-local" value={capture.acquired_at} onChange={e => change(index, { acquired_at: e.target.value })} size="small" slotProps={{ inputLabel: { shrink: true } }} />
          <TextField label={t('session.calibration')} value={capture.calibration_id} onChange={e => change(index, { calibration_id: e.target.value })} size="small" />
          <TextField label={t('session.exclusion')} value={capture.excluded_reason} onChange={e => change(index, { excluded_reason: e.target.value })} size="small" />
          <Button onClick={() => setCaptures(captures.filter((_, i) => i !== index))} disabled={captures.length === 1}>{t('session.remove')}</Button>
        </Stack>
      </Paper>)}
      <TextField select label={t('session.addCapture')} value="" onChange={e => setCaptures([...captures, newCapture(e.target.value)])} size="small">
        <MenuItem value="">{t('session.addCapture')}</MenuItem>{(runs.data ?? []).filter(r => r.task === 'evaluate_measured').map(r => <MenuItem key={r.run_id} value={r.run_id}>{r.name || r.run_id} · {r.status}</MenuItem>)}
      </TextField>
      <Accordion><AccordionSummary expandIcon={<ExpandMoreIcon />}>{t('session.records')}</AccordionSummary><AccordionDetails>
        <TextField fullWidth multiline minRows={8} label={t('session.records')} value={records} onChange={e => setRecords(e.target.value)} helperText="instruments: instrument_id, role, model, gain_db, bandwidth_hz, noise_floor_dbm, compression_input_dbm; calibrations: calibration_id, version, performed_at, method, reference_plane; uncertainty_budget" />
      </AccordionDetails></Accordion>
      {create.isError && <Alert severity="error">{create.error.message}</Alert>}
    </Stack></DialogContent>
    <DialogActions><Button disabled={create.isPending} onClick={close}>{t('form.cancel')}</Button><Button variant="contained" disabled={create.isPending || !title.trim() || captures.some((c, i) => !c.acquisition_id.trim() || (!c.acquired_at && !details[i]?.data?.measurement?.conditions.measured_at && c.run_id !== result.run_id))} onClick={() => create.mutate()}>{t('session.create')}</Button></DialogActions>
  </Dialog>
}

export function MeasurementSessions({ result }: { result: EvaluationResult }) {
  const qc = useQueryClient()
  const list = useQuery({ queryKey: ['measurement-sessions', result.run_id], queryFn: () => api.get<MeasurementSession[]>(`/measurement-sessions?run_id=${encodeURIComponent(result.run_id!)}`) })
  const [open, setOpen] = useState(false)
  const [chosen, setChosen] = useState('')
  const sessionId = chosen || list.data?.[0]?.session_id || ''
  const current = useQuery({ queryKey: ['measurement-session', sessionId], queryFn: () => api.get<MeasurementSession>(`/measurement-sessions/${encodeURIComponent(sessionId)}`), enabled: !!sessionId })
  const session = current.data
  const runs = [...new Set(session?.spec.captures.map(c => c.run_id) ?? [])]
  const value = (n: number | null | undefined) => typeof n === 'number' ? formatNumber(n, { maximumFractionDigits: 4 }) : t('review.missing')
  return <Paper component="section" aria-label={t('session.heading')} sx={{ p: 2 }}><Stack spacing={2}>
    <Stack direction="row" useFlexGap sx={{ alignItems: 'center', flexWrap: 'wrap', gap: 2 }}><Typography variant="h3" component="h2">{t('session.heading')}</Typography><Button onClick={() => setOpen(true)}>{t('session.create')}</Button>
      {sessionId && <TextField select size="small" label={t('session.heading')} value={list.data?.some(s => s.session_id === sessionId) ? sessionId : ''} onChange={e => setChosen(e.target.value)} sx={{ minWidth: 180 }}>{(list.data ?? []).map(s => <MenuItem key={s.session_id} value={s.session_id}>{s.spec.title}</MenuItem>)}</TextField>}
    </Stack>
    {!sessionId && <Typography color="text.secondary">{t('session.empty')}</Typography>}
    {session && <>
      <Typography variant="body2">{session.spec.source === 'mock' ? 'MOCK · ' : ''}{session.spec.dut} · {session.spec.reference_plane ?? t('review.missing')} · {session.spec.profile_id}</Typography>
      <Alert severity="info">{session.power_matching}</Alert>
      {(session.warnings ?? []).map(w => <Alert key={w} severity="warning">{message(w)}</Alert>)}
      <TableContainer tabIndex={0}><Table size="small" aria-label={t('session.repeats')}><TableHead><TableRow>
        <TableCell>{t('session.role')}</TableCell><TableCell>{t('compare.metric')}</TableCell><TableCell>n ({t('session.acquisition')})</TableCell><TableCell>{t('session.seeds')}</TableCell><TableCell>{t('session.mean')}</TableCell><TableCell>{t('session.sd')}</TableCell><TableCell>{t('session.drift')}</TableCell><TableCell>{t('session.interval')}</TableCell>
      </TableRow></TableHead><TableBody>{session.repeats.map(r => <TableRow key={`${r.role}:${r.metric}`}>
        <TableCell>{r.role}</TableCell><TableCell>{r.metric} ({r.unit})</TableCell><TableCell>{r.n_independent_captures}</TableCell><TableCell>{r.n_seeds ?? t('review.missing')}</TableCell><TableCell>{value(r.mean)}</TableCell><TableCell>{value(r.sample_std)}</TableCell><TableCell>{value(r.first_to_last_drift)}</TableCell><TableCell>{r.ci95 ? `[${r.ci95.map(value).join(', ')}]` : t('review.missing')}</TableCell>
      </TableRow>)}</TableBody></Table></TableContainer>
      <Typography variant="caption">{session.repeats[0]?.method ?? t('session.independenceHelp')}</Typography>
      <Accordion><AccordionSummary expandIcon={<ExpandMoreIcon />}>{t('session.records')}</AccordionSummary><AccordionDetails>
        <Stack spacing={1} sx={{ mb: 2 }}>
          {(session.spec.instruments ?? []).map(instrument => <Typography key={instrument.instrument_id} variant="body2">
            {instrument.instrument_id} · {instrument.role} · {instrument.model} · {instrument.serial ?? t('review.missing')}<br />
            {message('Gain')}: {value(instrument.gain_db)} dB · BW: {value(instrument.bandwidth_hz)} Hz · {message('Noise floor')}: {value(instrument.noise_floor_dbm)} dBm · {message('Compression input')}: {value(instrument.compression_input_dbm)} dBm
          </Typography>)}
          {(session.spec.calibrations ?? []).map(calibration => <Typography key={calibration.calibration_id} variant="body2">
            {calibration.calibration_id} · {calibration.version} · {calibration.performed_at} · {calibration.reference_plane}<br />
            {calibration.method} · {calibration.fixture ?? t('review.missing')} · {calibration.deembedding ?? t('review.missing')}
          </Typography>)}
          {session.spec.uncertainty_budget && <Typography variant="body2">{session.spec.uncertainty_budget}</Typography>}
        </Stack>
        <Stack spacing={1}>{session.captures.map(c => <Box key={c.capture.capture_id}>
          <Typography>{c.capture.acquisition_id} · {c.capture.role} · {c.status} {c.reason ?? ''}</Typography>
          <Typography variant="caption" sx={{ overflowWrap: 'anywhere' }}>SHA-256: {c.raw_sha256 ?? t('review.missing')}</Typography>
          <Typography variant="caption" component="div">{JSON.stringify(c.processing)}</Typography>
          {typeof c.processing?.artifact_id === 'string' && <DownloadLink href={`${API}/artifacts/${encodeURIComponent(c.capture.run_id)}/${encodeURIComponent(c.processing.artifact_id)}`} download>{c.capture.run_id} · {c.capture.role}</DownloadLink>}
        </Box>)}</Stack>
      </AccordionDetails></Accordion>
      <Stack direction="row" spacing={2}><DownloadLink href={`${API}/measurement-sessions/${session.session_id}`} download={`${session.session_id}.json`}>{t('session.export')}</DownloadLink>
        {runs.length >= 2 && runs.length <= 8 && <Button component={RouterLink} to={`/results/compare?${runs.map(r => `runs=${encodeURIComponent(r)}`).join('&')}&profile=${encodeURIComponent(session.spec.profile_id)}`}>{t('compare.title')}</Button>}
      </Stack>
    </>}
    {current.isError && <Alert severity="error">{current.error.message}</Alert>}
    {open && <CreateSession result={result} close={() => setOpen(false)} saved={s => { qc.setQueryData(['measurement-session', s.session_id], s); qc.setQueryData<MeasurementSession[]>(['measurement-sessions', result.run_id], old => [s, ...(old ?? [])]); setChosen(s.session_id); setOpen(false) }} />}
  </Stack></Paper>
}
