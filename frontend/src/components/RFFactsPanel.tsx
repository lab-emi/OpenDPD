import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import { Accordion, AccordionDetails, AccordionSummary, Alert, Box, Button, Dialog, DialogActions, DialogContent, DialogTitle, Paper, Stack, TextField, Typography } from '@mui/material'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'
import { api, WEB_MODE } from '@/api/client'
import { reviewQuery } from '@/api/hooks'
import type { EvaluationResult, RFConditions, ReviewContext } from '@/api/types'
import { formatNumber, message, t } from '@/i18n'

const CORE = ['dut', 'carrier_frequency_hz', 'bandwidth_hz', 'sample_rate_hz', 'average_output_power_dbm']
export function factValue(fact: ReviewContext['facts'][number]) {
  if (fact.value === null || fact.value === undefined) return t('review.missing')
  const numeric = Number(fact.value)
  if (fact.unit === 'Hz' && Number.isFinite(numeric)) return `${formatNumber(numeric / 1e6, { maximumFractionDigits: 3 })} ${fact.key === 'sample_rate_hz' ? 'MS/s' : 'MHz'}`
  return `${fact.unit && Number.isFinite(numeric) ? formatNumber(numeric, { maximumFractionDigits: 4 }) : fact.value}${fact.unit ? ` ${fact.unit}` : ''}`
}

const FIELDS = [
  ['dut', 'PA / DUT', false], ['carrier_frequency_hz', 'Carrier frequency (MHz)', true],
  ['average_output_power_dbm', 'Average output power (dBm)', true], ['input_power_dbm', 'Input power (dBm)', true],
  ['pa_dc_power_w', 'PA DC power (W)', true], ['power_reference_plane', 'Power reference plane', false],
  ['temperature_c', 'Temperature (°C)', true], ['supply_v', 'Supply (V)', true],
  ['mode', 'Operating mode', false], ['calibration_id', 'Calibration record', false],
] as const

function ConditionsEditor({ result, onClose }: { result: EvaluationResult; onClose: () => void }) {
  const qc = useQueryClient()
  const [draft, setDraft] = useState<Record<string, string>>(() => Object.fromEntries(FIELDS.map(([key]) => {
    const value = result.rf_conditions?.[key]
    return [key, value === null || value === undefined ? '' : String(key === 'carrier_frequency_hz' ? Number(value) / 1e6 : value)]
  })))
  const [note, setNote] = useState(result.rf_conditions?.note ?? '')
  const [extra, setExtra] = useState(() => JSON.stringify(Object.fromEntries(Object.entries(result.rf_conditions ?? {}).filter(([key]) => !FIELDS.some(([k]) => k === key) && !['version', 'source', 'note', 'recorded_at'].includes(key))), null, 2))
  const save = useMutation({
    mutationFn: async () => {
      const advanced: unknown = JSON.parse(extra)
      if (!advanced || Array.isArray(advanced) || typeof advanced !== 'object') throw new Error('Expected a JSON object')
      const fields = Object.fromEntries(FIELDS.map(([key, , numeric]) => [key, draft[key]?.trim() ? numeric ? Number(draft[key]) * (key === 'carrier_frequency_hz' ? 1e6 : 1) : draft[key] : null]))
      return api.post<RFConditions>(`/results/${encodeURIComponent(result.run_id!)}/rf-conditions`, { ...advanced, ...fields, note, source: 'user_declared' })
    },
    onSuccess: async () => {
      await Promise.all(['review', 'result', 'compare'].map(key => qc.invalidateQueries({ queryKey: [key] })))
      onClose()
    },
  })
  return <Dialog open onClose={save.isPending ? undefined : onClose} fullWidth maxWidth="md">
    <DialogTitle>{t('review.edit')}</DialogTitle>
    <DialogContent dividers><Stack spacing={2}>
      <Alert severity="info">{t('review.declaration')}</Alert>
      <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 2 }}>
        {FIELDS.map(([key, label, numeric]) => <TextField key={key} label={message(label)} size="small" type={numeric ? 'number' : 'text'} value={draft[key]} onChange={e => setDraft({ ...draft, [key]: e.target.value })} slotProps={{ htmlInput: numeric ? { step: 'any' } : {} }} />)}
      </Box>
      <TextField label={t('review.note')} required value={note} onChange={e => setNote(e.target.value)} />
      <Accordion><AccordionSummary expandIcon={<ExpandMoreIcon />}>{t('review.advanced')}</AccordionSummary><AccordionDetails>
        <TextField fullWidth multiline minRows={5} label={t('review.advanced')} value={extra} onChange={e => setExtra(e.target.value)} helperText="dc_rails_w, included_rails, bias, load, vswr + reflection_phase_deg, fixture, deembedding, backoff_reference" />
      </AccordionDetails></Accordion>
      {save.isError && <Alert severity="error">{save.error.message}</Alert>}
    </Stack></DialogContent>
    <DialogActions><Button onClick={onClose} disabled={save.isPending}>{t('form.cancel')}</Button><Button variant="contained" disabled={save.isPending || !note.trim()} onClick={() => save.mutate()}>{t('review.saveConditions')}</Button></DialogActions>
  </Dialog>
}

export function RFFactsPanel({ result }: { result: EvaluationResult }) {
  const review = useQuery({ ...reviewQuery(result.run_id ?? '', result.metric_profile_id), enabled: !!result.run_id })
  const [editing, setEditing] = useState(false)
  const facts = review.data?.facts ?? CORE.map(key => ({ key, label: key.replaceAll('_', ' '), value: null, source: 'not recorded' }))
  return <Paper component="section" aria-label={t('review.facts')} sx={{ p: 2 }} data-testid="rf-facts">
    <Stack direction="row" sx={{ justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
      <Typography variant="h3" component="h2">{t('review.facts')}</Typography>
      {!WEB_MODE && result.run_id && <Button size="small" onClick={() => setEditing(true)}>{t('review.edit')}</Button>}
    </Stack>
    {facts.some(f => f.key === 'dataset_origin' && f.value === 'synthetic') && <Alert severity="warning" sx={{ mb: 1 }}>{t('datasetResearch.syntheticNotice')}</Alert>}
    <Box component="dl" sx={{ m: 0, display: 'grid', gridTemplateColumns: { xs: '1fr 1fr', md: 'repeat(5, minmax(0,1fr))' }, gap: 1.5 }}>
      {CORE.map(key => facts.find(f => f.key === key)).filter(f => !!f).map(f => <Box key={f.key}>
        <Typography component="dt" variant="caption" color="text.secondary">{message(f.label)}</Typography>
        <Typography component="dd" sx={{ m: 0, overflowWrap: 'anywhere' }}>{factValue(f)}</Typography>
      </Box>)}
    </Box>
    <Accordion disableGutters elevation={0} sx={{ mt: 1 }}><AccordionSummary expandIcon={<ExpandMoreIcon />}>{t('review.details')}</AccordionSummary><AccordionDetails>
      {review.data && <Stack spacing={2}>
        <Typography variant="body2">{review.data.signal_source}. {review.data.profile.description}</Typography>
        <Box component="dl" sx={{ m: 0, display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 2fr' }, gap: 1, overflowWrap: 'anywhere' }}>
          {facts.map(f => <div key={f.key} style={{ display: 'contents' }}><dt>{message(f.label)}</dt><dd style={{ margin: 0 }}>{factValue(f)}<Typography variant="caption" component="div" color="text.secondary">{f.source}</Typography></dd></div>)}
          {Object.entries(review.data.provenance ?? {}).map(([key, value]) => <div key={key} style={{ display: 'contents' }}><dt>{key}</dt><dd style={{ margin: 0 }}><code>{value ?? t('review.missing')}</code></dd></div>)}
        </Box>
      </Stack>}
      {review.isError && <Alert severity="info">{review.error.message}</Alert>}
    </AccordionDetails></Accordion>
    {editing && <ConditionsEditor result={review.data?.result ?? result} onClose={() => setEditing(false)} />}
  </Paper>
}
