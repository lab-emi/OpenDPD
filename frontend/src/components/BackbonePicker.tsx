import { useRef, useState } from 'react'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import Alert from '@mui/material/Alert'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Checkbox from '@mui/material/Checkbox'
import Chip from '@mui/material/Chip'
import Dialog from '@mui/material/Dialog'
import DialogActions from '@mui/material/DialogActions'
import DialogContent from '@mui/material/DialogContent'
import DialogTitle from '@mui/material/DialogTitle'
import FormControlLabel from '@mui/material/FormControlLabel'
import Link from '@mui/material/Link'
import MenuItem from '@mui/material/MenuItem'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import UploadIcon from '@mui/icons-material/Upload'
import RefreshIcon from '@mui/icons-material/Refresh'
import { API, api, getQuery, safePullRequestUrl } from '@/api/client'
import type { Schemas } from '@/api/types'
import { t } from '@/i18n'
import { DownloadLink } from './DownloadLink'
import { ErrorState } from './StateBlock'

export type BackboneEntry = Schemas['BackboneEntry']
type Upload = Schemas['BackboneUpload']
type Catalog = Schemas['BackboneCatalog']
type Capability = Schemas['BackboneCapability']
const uploadsKey = ['backbone-uploads']
const catalogKey = ['backbone-catalog']
const active = (status: Upload['status']) => ['queued', 'branch', 'push', 'pull_request'].includes(status)

export function BackbonePicker({ selected, onSelect }: { selected: BackboneEntry | null; onSelect: (entry: BackboneEntry | null) => void }) {
  const qc = useQueryClient()
  const [dialog, setDialog] = useState<'new' | Upload | null>(null)
  const uploads = useQuery({ queryKey: uploadsKey, queryFn: getQuery<Upload[]>('/backbones/uploads'),
    refetchInterval: query => query.state.data?.some(r => active(r.status)) ? 1500 : false })
  const catalog = useQuery({ queryKey: catalogKey, queryFn: getQuery<Catalog>('/backbones/catalog'), staleTime: 60_000 })
  const refresh = useMutation({ mutationFn: () => api.post<Catalog>('/backbones/catalog/refresh', {}),
    onSuccess: result => qc.setQueryData(catalogKey, result) })
  const mine = uploads.data?.find(r => r.backbone_id === selected?.backbone_id)
  const entries = catalog.data?.entries ?? []
  return <Paper variant="outlined" sx={{ p: 2, width: '100%' }}>
    <Stack spacing={1.5}>
      <Stack direction="row" sx={{ alignItems: 'center', gap: 1, flexWrap: 'wrap', justifyContent: 'space-between' }}>
        <Box><Typography variant="h3">{t('backbones.title')}</Typography><Typography variant="body2" color="text.secondary">{t('backbones.intro')}</Typography></Box>
        <Stack direction="row" sx={{ gap: 1, flexWrap: 'wrap' }}>
          <DownloadLink href={`${API}/backbones/template`} button variant="outlined">{t('backbones.download')}</DownloadLink>
          <Button variant="contained" startIcon={<UploadIcon />} onClick={() => setDialog('new')}>{t('backbones.upload')}</Button>
        </Stack>
      </Stack>
      <Stack direction={{ xs: 'column', md: 'row' }} spacing={2}>
        <TextField select fullWidth label={t('backbones.mine')} value={selected?.origin === 'private' && mine ? selected.backbone_id : ''}
          slotProps={{ select: { displayEmpty: true }, inputLabel: { shrink: true } }}
          onChange={e => onSelect(uploads.data?.find(r => r.backbone_id === e.target.value) ?? null)}
          helperText={t('backbones.mineHelp')}>
          <MenuItem value="">{t('backbones.choose')}</MenuItem>
          {(uploads.data ?? []).map(r => <MenuItem value={r.backbone_id} key={r.backbone_id}>{r.name} · {r.source_sha256.slice(0, 8)}</MenuItem>)}
        </TextField>
        <TextField select fullWidth label="User Uploaded Backbones" value={selected?.origin === 'community' ? selected.backbone_id : ''}
          slotProps={{ select: { displayEmpty: true }, inputLabel: { shrink: true } }}
          onChange={e => onSelect(entries.find(r => r.backbone_id === e.target.value) ?? null)}
          helperText={entries.length ? t('backbones.communityHelp') : t('backbones.communityEmpty')}>
          <MenuItem value="">{t('backbones.choose')}</MenuItem>
          {entries.map(r => <MenuItem value={r.backbone_id} key={r.backbone_id}>{r.name} · {r.source_sha256.slice(0, 8)}</MenuItem>)}
        </TextField>
      </Stack>
      <Stack direction="row" sx={{ gap: 1, alignItems: 'center', flexWrap: 'wrap' }}>
        <Button size="small" startIcon={<RefreshIcon />} disabled={refresh.isPending} onClick={() => refresh.mutate()}>{t(refresh.isPending ? 'backbones.refreshing' : 'backbones.refresh')}</Button>
        {selected && <Button size="small" onClick={() => onSelect(null)}>{t('backbones.builtin')}</Button>}
      </Stack>
      {selected && <Box sx={{ borderLeft: 3, borderColor: 'primary.main', pl: 1.5 }}>
        <Stack direction="row" sx={{ gap: 1, alignItems: 'center', flexWrap: 'wrap' }}><Typography sx={{ fontWeight: 600 }}>{selected.name}</Typography>
          <Chip size="small" label={t('backbones.summary', { nodes: selected.node_count, count: selected.parameter_count })} />
          {mine?.pull_request_url && <Link href={safePullRequestUrl(mine.pull_request_url)} target="_blank" rel="noreferrer">{t('backbones.openPR')}</Link>}
          {mine && mine.status !== 'submitted' && !active(mine.status) && <Button size="small" onClick={() => setDialog(mine)}>{t('backbones.contribute')}</Button>}
        </Stack>
        <Typography variant="body2" sx={{ mt: .5 }}>{selected.description}</Typography>
        <Typography variant="caption" color="text.secondary">{t('backbones.contract')}</Typography>
        {mine && active(mine.status) && <Typography role="status">{t('backbones.submitting')}</Typography>}
        {mine?.error && <Alert severity="error" sx={{ mt: 1 }}>{mine.error}</Alert>}
      </Box>}
      {catalog.data?.warning && <Alert severity="warning">{catalog.data.warning}</Alert>}
      {(uploads.isError || catalog.isError) && <Alert severity="warning">{t('backbones.loadError')} <Button onClick={() => { void uploads.refetch(); void catalog.refetch() }}>{t('state.error.retry')}</Button></Alert>}
      {refresh.isError && <ErrorState error={refresh.error} />}
    </Stack>
    {dialog !== null && <UploadDialog existing={dialog === 'new' ? null : dialog} onClose={() => setDialog(null)} onAdded={onSelect} />}
  </Paper>
}

function UploadDialog({ existing, onClose, onAdded }: { existing: Upload | null; onClose: () => void; onAdded: (entry: BackboneEntry) => void }) {
  const qc = useQueryClient()
  const generation = useRef(0)
  const [file, setFile] = useState<{ filename: string; source: string } | null>(null)
  const [reading, setReading] = useState(false)
  const [fileError, setFileError] = useState<string | null>(null)
  const [share, setShare] = useState(false)
  const [saved, setSaved] = useState<Upload | null>(existing)
  const capability = useQuery({ queryKey: ['backbone-capability'], queryFn: getQuery<Capability>('/backbones/capability'), staleTime: 0 })
  const save = useMutation({ mutationFn: async () => {
    const record = saved ?? await api.post<Upload>('/backbones/uploads', file)
    setSaved(record)
    onAdded(record)
    void qc.invalidateQueries({ queryKey: uploadsKey })
    if (share) {
      const submitted = await api.post<Upload>(`/backbones/uploads/${record.publication_id}/submit`, {
        source_sha256: record.source_sha256, package_sha256: record.package_sha256, publish_publicly: true, rights_confirmed: true,
      })
      setSaved(submitted)
      void qc.invalidateQueries({ queryKey: uploadsKey })
    }
    return record
  }, onSuccess: () => onClose() })
  const busy = reading || save.isPending
  const choose = async (picked?: File) => {
    const current = ++generation.current
    setFile(null); setSaved(null); setFileError(null); setShare(false); save.reset()
    if (!picked) return
    if (!/^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\.py$/.test(picked.name) || picked.name.includes('..') || picked.size > 32768 || picked.size === 0) {
      setFileError(t('backbones.fileError')); return
    }
    setReading(true)
    try {
      // Preserve a UTF-8 BOM in the posted bytes so the displayed SHA256 also
      // identifies the file on disk, not a silently normalized version of it.
      const source = new TextDecoder('utf-8', { fatal: true, ignoreBOM: true }).decode(await picked.arrayBuffer())
      if (generation.current === current) setFile({ filename: picked.name, source })
    } catch { if (generation.current === current) setFileError(t('backbones.fileError')) }
    finally { if (generation.current === current) setReading(false) }
  }
  return <Dialog open onClose={busy ? undefined : onClose} fullWidth maxWidth="sm" aria-labelledby="backbone-upload-title">
    <DialogTitle id="backbone-upload-title">{t(existing ? 'backbones.contribute' : 'backbones.upload')}</DialogTitle>
    <DialogContent dividers><Stack spacing={2}>
      <Typography>{t('backbones.templateHelp')}</Typography>
      {!existing && <>
        <DownloadLink href={`${API}/backbones/template`} button variant="outlined">{t('backbones.download')}</DownloadLink>
        <Button component="label" variant="outlined" disabled={busy} startIcon={<UploadIcon />}>{file?.filename ?? t('backbones.chooseFile')}
          <input type="file" accept=".py" hidden aria-label={t('backbones.chooseFile')} disabled={busy} onChange={e => { void choose(e.target.files?.[0]); e.target.value = '' }} />
        </Button>
      </>}
      {file && <Box component="details"><Typography component="summary">{t('backbones.sourcePreview')}</Typography><Box component="pre" sx={{ maxHeight: 230, overflow: 'auto', p: 1.5, bgcolor: 'action.hover', fontSize: 12 }}>{file.source}</Box></Box>}
      {saved && <Alert severity="success"><Typography sx={{ fontWeight: 600 }}>{saved.name}</Typography>{t('backbones.validated')}<Typography variant="caption" component="p" sx={{ overflowWrap: 'anywhere' }}>SHA256: {saved.source_sha256}</Typography>
        <DownloadLink href={`${API}/backbones/uploads/${saved.publication_id}/source`}>{t('backbones.sourcePreview')}</DownloadLink>
      </Alert>}
      <Paper variant="outlined" sx={{ p: 1.5 }}>
        <FormControlLabel sx={{ alignItems: 'flex-start', m: 0 }} control={<Checkbox checked={share} disabled={busy || !capability.data?.publication_available} onChange={e => setShare(e.target.checked)} />} label={t('backbones.share')} />
        <Typography variant="body2" color="text.secondary" sx={{ pl: 4.5 }}>{t('backbones.shareHelp')}</Typography>
      </Paper>
      {capability.data?.reason && <Alert severity="info">{capability.data.reason}</Alert>}
      {capability.isError && <ErrorState error={capability.error} onRetry={() => void capability.refetch()} />}
      {fileError && <Alert severity="error">{fileError}</Alert>}
      {save.isError && <ErrorState error={save.error} />}
    </Stack></DialogContent>
    <DialogActions><Button disabled={busy} onClick={onClose}>{t('common.back')}</Button><Button variant="contained" disabled={busy || (!file && !saved) || (!!existing && !share) || (share && !capability.data?.publication_available)} onClick={() => save.mutate()}>
      {t(save.isPending ? 'backbones.saving' : share ? 'backbones.addAndSubmit' : 'backbones.addPrivate')}
    </Button></DialogActions>
  </Dialog>
}
