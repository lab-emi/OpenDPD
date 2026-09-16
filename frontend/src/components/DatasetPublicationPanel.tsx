import { getQuery } from '@/api/client'
import { useState } from 'react'
import { Alert, Box, Button, Checkbox, Chip, Dialog, DialogActions, DialogContent, DialogTitle, FormControlLabel, Link, MenuItem, Paper, Stack, TextField, Typography } from '@mui/material'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { useSearchParams } from 'react-router'
import { api, API, safePullRequestUrl } from '@/api/client'
import type { components } from '@/api/schema'
import type { DatasetManifest } from '@/api/types'
import { t, type MessageKey } from '@/i18n'
import { DownloadLink } from './DownloadLink'
import { ErrorState, LoadingState } from './StateBlock'

type Publication = components['schemas']['DatasetPublication']
const active = (r?: Publication) => !!r && ['queued', 'branch', 'push', 'pull_request'].includes(r.status)

export function DatasetPublicationPanel({ dataset }: { dataset: DatasetManifest }) {
  const [search, setSearch] = useSearchParams()
  const [open, setOpen] = useState(search.get('publish') === '1')
  const close = () => { setOpen(false); setSearch(old => { const next = new URLSearchParams(old); next.delete('publish'); return next }, { replace: true }) }
  return <Paper sx={{ p: 1.5 }}><Stack spacing={.75}>
    <Stack direction="row" spacing={1} sx={{ alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap' }}><Typography variant="h3">{t('datasetResearch.sharing')}</Typography><Button onClick={() => setOpen(true)}>{t('datasetResearch.reviewPublic')}</Button></Stack>
    <Typography variant="body2">{t('datasetResearch.privateNotice')}</Typography>
    {open && <PublicationDialog dataset={dataset} onClose={close} />}
  </Stack></Paper>
}

function PublicationDialog({ dataset, onClose }: { dataset: DatasetManifest; onClose: () => void }) {
  const qc = useQueryClient()
  const queryKey = ['dataset-publications', dataset.dataset_id]
  const [description, setDescription] = useState('')
  const [attribution, setAttribution] = useState('')
  const [license, setLicense] = useState('')
  const [consent, setConsent] = useState(false)
  const [review, setReview] = useState<Publication>()
  const [stamp, setStamp] = useState('')
  const formStamp = JSON.stringify({ description, attribution, license })
  const capability = useQuery({ queryKey: ['dataset-publication-capability'], queryFn: getQuery<components['schemas']['PublicationCapability']>('/dataset-publications/capability'), staleTime: 15000 })
  const history = useQuery({ queryKey, queryFn: getQuery<Publication[]>(`/dataset-publications?dataset_id=${encodeURIComponent(dataset.dataset_id)}`), refetchInterval: q => q.state.data?.some(active) ? 1000 : false })
  const selected = (review && history.data?.find(r => r.publication_id === review.publication_id)) ?? review
  const prepare = useMutation({ mutationFn: () => api.post<Publication>('/dataset-publications/prepare', { dataset_id: dataset.dataset_id, description, attribution, license }), onSuccess: r => { setReview(r); setStamp(formStamp); setConsent(false); void qc.invalidateQueries({ queryKey }) } })
  const submit = useMutation({ mutationFn: (record: Publication) => api.post<Publication>(`/dataset-publications/${encodeURIComponent(record.publication_id)}/submit`, { package_sha256: record.package_sha256, publish_publicly: true, rights_confirmed: true }), onSuccess: r => { setReview(r); void qc.invalidateQueries({ queryKey }) } })
  const busy = prepare.isPending || submit.isPending || active(selected)
  const fresh = !!review && stamp === formStamp
  return <Dialog open fullWidth maxWidth="md" onClose={prepare.isPending || submit.isPending ? undefined : onClose} aria-labelledby="publication-title">
    <DialogTitle id="publication-title">{t('datasetResearch.reviewPublic')}</DialogTitle>
    <DialogContent dividers><Stack spacing={2}>
      <Alert severity="info">{t('datasetResearch.humanReview')} <Link href="mailto:emi.lab@outlook.com">emi.lab@outlook.com</Link></Alert>
      <Alert severity="warning">{t('datasetResearch.publicNotice')}</Alert>
      {capability.isPending && <LoadingState />}{capability.isError && <ErrorState error={capability.error} />}
      {capability.data && !capability.data.available && <Alert severity="warning">{capability.data.reason}<Button size="small" onClick={() => void capability.refetch()}>{t('datasetResearch.refresh')}</Button></Alert>}
      <TextField label={t('datasetResearch.description')} multiline minRows={2} value={description} disabled={busy} onChange={e => setDescription(e.target.value)} slotProps={{ htmlInput: { maxLength: 2000 } }} />
      <TextField label={t('datasetResearch.attribution')} value={attribution} disabled={busy} onChange={e => setAttribution(e.target.value)} slotProps={{ htmlInput: { maxLength: 200 } }} />
      <TextField select label={t('datasetResearch.license')} value={license} disabled={busy} onChange={e => setLicense(e.target.value)}><MenuItem value="">{t('datasetResearch.chooseLicense')}</MenuItem>{['CC0-1.0', 'CC-BY-4.0'].map(l => <MenuItem key={l} value={l}>{l}</MenuItem>)}</TextField>
      <Button variant="outlined" disabled={busy || !description.trim() || !attribution.trim() || !license} onClick={() => prepare.mutate()}>{t('datasetResearch.preview')}</Button>
      {fresh && selected && <Paper variant="outlined" sx={{ p: 2 }}><Stack spacing={1}>
        <Typography variant="h3">{dataset.display_name}</Typography><Chip sx={{ alignSelf: 'flex-start' }} label={t(`datasetResearch.status.${selected.status}` as MessageKey)} />
        <Typography variant="body2" sx={{ overflowWrap: 'anywhere' }}>{selected.repository} · {selected.directory}</Typography>
        <Typography variant="caption" sx={{ overflowWrap: 'anywhere' }}>SHA256: {selected.package_sha256}</Typography>
        {selected.files.map(f => <Typography variant="body2" key={f.path}>{f.path} · {f.size_bytes == null ? t('common.na') : `${(f.size_bytes / 1024).toFixed(1)} KiB`}</Typography>)}
        <Box component="details"><Typography component="summary">{t('datasetResearch.metadata')}</Typography><Box component="pre" sx={{ whiteSpace: 'pre-wrap', overflowWrap: 'anywhere', fontSize: 12 }}>{JSON.stringify(selected.catalog, null, 2)}</Box></Box>
        <DownloadLink href={`${API}/dataset-publications/${encodeURIComponent(selected.publication_id)}/download`} download>{t('datasetResearch.download')}</DownloadLink>
        {selected.error && <Alert severity="error">{selected.error}</Alert>}
        {selected.pull_request_url && <Alert severity="success"><Link href={safePullRequestUrl(selected.pull_request_url)} target="_blank" rel="noreferrer">{t('datasetResearch.openPR')}</Link> · {selected.pull_request_state}</Alert>}
        {selected.status !== 'submitted' && <FormControlLabel control={<Checkbox checked={consent} disabled={busy} onChange={e => setConsent(e.target.checked)} />} label={t('datasetResearch.consent')} />}
      </Stack></Paper>}
      {prepare.isError && <ErrorState error={prepare.error} />}{submit.isError && <ErrorState error={submit.error} />}
      {!!history.data?.length && <Box component="details"><Typography component="summary">{t('datasetResearch.history')}</Typography><Stack spacing={1} sx={{ pt: 1 }}>{history.data.map(r => <Stack key={r.publication_id} direction="row" spacing={1} sx={{ alignItems: 'center', flexWrap: 'wrap' }}><Typography variant="body2">{r.created_at} · {t(`datasetResearch.status.${r.status}` as MessageKey)}</Typography>{r.pull_request_url ? <Link href={safePullRequestUrl(r.pull_request_url)} target="_blank" rel="noreferrer">{t('datasetResearch.openPR')}</Link> : <Button onClick={() => { setDescription(r.catalog.description); setAttribution(r.catalog.attribution); setLicense(r.catalog.license); setReview(r); setStamp(JSON.stringify({ description: r.catalog.description, attribution: r.catalog.attribution, license: r.catalog.license })); setConsent(false) }}>{t('datasetResearch.reopen')}</Button>}</Stack>)}</Stack></Box>}
      {history.isError && <ErrorState error={history.error} />}
    </Stack></DialogContent>
    <DialogActions><Button onClick={onClose} disabled={prepare.isPending || submit.isPending}>{t('common.back')}</Button><Button variant="contained" disabled={busy || !fresh || !consent || !capability.data?.available || selected?.status === 'submitted'} onClick={() => selected && submit.mutate(selected)}>{t('datasetResearch.submit')}</Button></DialogActions>
  </Dialog>
}
