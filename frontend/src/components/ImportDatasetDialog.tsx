import Alert from '@mui/material/Alert'
import AlertTitle from '@mui/material/AlertTitle'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Dialog from '@mui/material/Dialog'
import DialogActions from '@mui/material/DialogActions'
import DialogContent from '@mui/material/DialogContent'
import DialogTitle from '@mui/material/DialogTitle'
import Grid from '@mui/material/Grid'
import List from '@mui/material/List'
import ListItemButton from '@mui/material/ListItemButton'
import ListItemText from '@mui/material/ListItemText'
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
import { inspectSource, useImportDataset, useImportRoots, useRootFiles, type ImportRequest, type SourceInfo, type UploadResult } from '@/api/datasets'
import type { DatasetOrigin } from '@/api/types'
import { t } from '@/i18n'
import { ORIGINS } from '@/components/ManifestDialog'
import { SignalFields, emptySignalForm, signalSpecFrom, type SignalForm } from '@/components/SignalFields'
import { ErrorState, LoadingState } from '@/components/StateBlock'

const LOGICAL = ['I_in', 'Q_in', 'I_out', 'Q_out'] as const

interface Picked {
  root_id: string
  path: string
  info: SourceInfo
}

const basename = (p: string) => p.split('/').pop() ?? p
const parent = (p: string) => (p.includes('/') ? p.slice(0, p.lastIndexOf('/')) : '')
const slug = (name: string) =>
  name
    .replace(/\.[^.]+$/, '')
    .replace(/[^A-Za-z0-9._-]+/g, '-')
    .replace(/^[^A-Za-z0-9]+/, '')
    .slice(0, 64)

/**
 * Import wizard over the authorised import roots: browse or upload → inspect
 * (server reads headers/shape) → confirm mapping + metadata → import.
 * Absolute paths never leave the browser; only (root_id, relative path).
 */
export function ImportDatasetDialog({ onClose, onImported }: { onClose: () => void; onImported: (datasetId: string) => void }) {
  const roots = useImportRoots()
  const [rootEdit, setRoot] = useState('')
  const rootId = rootEdit || roots.data?.[0]?.root_id || ''
  const [dir, setDir] = useState('')
  const files = useRootFiles(rootId, dir, rootId !== '')
  const [picked, setPicked] = useState<Picked | null>(null)
  const [mapping, setMapping] = useState<Record<string, string>>({})
  const [datasetId, setDatasetId] = useState('')
  const [displayName, setDisplayName] = useState('')
  const [origin, setOrigin] = useState<DatasetOrigin>('measured')
  const [guard, setGuard] = useState('256')
  const [signal, setSignal] = useState<SignalForm>(emptySignalForm)
  const inspect = useMutation({
    mutationFn: ({ root_id, path }: { root_id: string; path: string }) => inspectSource(root_id, path).then((info): Picked => ({ root_id, path, info })),
    onSuccess: (p) => {
      setPicked(p)
      setMapping(p.info.suggested_mapping)
      setDatasetId((id) => id || slug(basename(p.path)))
    },
  })
  const upload = useMutation({
    mutationFn: (file: File) => {
      const form = new FormData()
      form.append('file', file, file.name)
      return api.upload<UploadResult>('/datasets/upload', form)
    },
    onSuccess: (r) => {
      setRoot(r.root_id)
      setDir(parent(r.path))
      inspect.mutate({ root_id: r.root_id, path: r.path })
    },
  })
  const importDs = useImportDataset()

  const info = picked?.info ?? null
  const mappingNeeded = info?.kind === 'csv_import'
  const mappingComplete = !mappingNeeded || LOGICAL.every((k) => !!mapping[k])
  const canImport = !!info && info.problems.length === 0 && mappingComplete && !importDs.isPending
  const submit = () => {
    if (!picked) return
    const g = Number(guard)
    const body: ImportRequest = {
      source: { root_id: picked.root_id, path: picked.path },
      dataset_id: datasetId.trim() || null,
      display_name: displayName.trim() || null,
      mapping: mappingNeeded ? mapping : {},
      signal: signalSpecFrom(signal),
      origin,
      guard_samples: guard.trim() === '' || !Number.isFinite(g) ? 256 : g,
      notes: null,
    }
    importDs.mutate(body, { onSuccess: (m) => onImported(m.dataset_id) })
  }

  return (
    <Dialog open onClose={onClose} fullWidth maxWidth="lg" aria-labelledby="import-title">
      <DialogTitle id="import-title">{t('datasets.import.title')}</DialogTitle>
      <DialogContent dividers>
        <Grid container spacing={2} sx={{ pt: 1 }}>
          <Grid size={{ xs: 12, md: 4 }}>
            <Stack spacing={1}>
              <TextField
                select
                fullWidth
                size="small"
                label={t('datasets.import.root')}
                value={rootId}
                onChange={(e) => {
                  setRoot(e.target.value)
                  setDir('')
                }}
              >
                {(roots.data ?? []).map((r) => (
                  <MenuItem key={r.root_id} value={r.root_id}>
                    {r.root_id} — {r.path}
                  </MenuItem>
                ))}
              </TextField>
              <Button component="label" variant="outlined" disabled={upload.isPending}>
                {upload.isPending ? t('datasets.import.uploading') : t('datasets.import.upload')}
                <input
                  hidden
                  type="file"
                  accept=".csv,.npy,.npz"
                  onChange={(e) => {
                    const f = e.target.files?.[0]
                    if (f) upload.mutate(f)
                    e.target.value = ''
                  }}
                />
              </Button>
              <Typography variant="caption" color="text.secondary">
                {t('datasets.import.upload.help')}
              </Typography>
              {upload.isError && <ErrorState error={upload.error} />}
              <Typography variant="h3" component="h3">
                {t('datasets.import.files')} <code>/{dir}</code>
              </Typography>
              {files.isPending ? (
                <LoadingState />
              ) : files.isError ? (
                <ErrorState error={files.error} onRetry={() => void files.refetch()} />
              ) : (
                <List dense aria-label={t('datasets.import.files')} sx={{ maxHeight: 360, overflowY: 'auto' }}>
                  {dir !== '' && (
                    <ListItemButton onClick={() => setDir(parent(dir))}>
                      <ListItemText primary=".." />
                    </ListItemButton>
                  )}
                  {files.data.length === 0 && (
                    <Typography color="text.secondary" sx={{ p: 1 }}>
                      {t('datasets.import.noFiles')}
                    </Typography>
                  )}
                  {files.data.map((f) => (
                    <ListItemButton key={f.path} selected={picked?.path === f.path} onClick={() => (f.kind === 'dir' ? setDir(f.path) : inspect.mutate({ root_id: rootId, path: f.path }))}>
                      <ListItemText primary={basename(f.path)} secondary={f.kind === 'dir' ? t('datasets.import.folder') : `${f.size_bytes.toLocaleString()} B`} />
                    </ListItemButton>
                  ))}
                </List>
              )}
            </Stack>
          </Grid>
          <Grid size={{ xs: 12, md: 8 }}>
            {inspect.isPending ? (
              <LoadingState label={t('datasets.import.inspecting')} />
            ) : inspect.isError ? (
              <ErrorState error={inspect.error} />
            ) : !info || !picked ? (
              <Typography color="text.secondary">{t('datasets.import.pick')}</Typography>
            ) : (
              <Stack spacing={2}>
                <Typography variant="body2">
                  <code>{picked.path}</code> · {info.kind}
                  {info.n_rows !== null && info.n_rows !== undefined ? ` · ${t('datasets.rows', { count: info.n_rows.toLocaleString() })}` : ''}
                  {info.legacy_files.length > 0 ? ` · ${t('datasets.import.legacy', { count: info.legacy_files.length })}` : ''}
                </Typography>
                {info.problems.length > 0 && (
                  <Alert severity="error">
                    <AlertTitle>{t('datasets.import.problems')}</AlertTitle>
                    <ul style={{ margin: 0, paddingLeft: 20 }}>
                      {info.problems.map((p) => (
                        <li key={p}>{p}</li>
                      ))}
                    </ul>
                  </Alert>
                )}
                {mappingNeeded && (
                  <section aria-label={t('datasets.import.mapping')}>
                    <Typography variant="h3" component="h3" gutterBottom>
                      {t('datasets.import.mapping')}
                    </Typography>
                    <Typography variant="body2" color="text.secondary" gutterBottom>
                      {t('datasets.import.mapping.help')}
                    </Typography>
                    <Grid container spacing={1}>
                      {LOGICAL.map((k) => (
                        <Grid key={k} size={{ xs: 6, md: 3 }}>
                          <TextField select fullWidth size="small" label={k} value={mapping[k] ?? ''} error={!mapping[k]} onChange={(e) => setMapping((m) => ({ ...m, [k]: e.target.value }))}>
                            {info.columns.map((c) => (
                              <MenuItem key={c} value={c}>
                                {c}
                              </MenuItem>
                            ))}
                          </TextField>
                        </Grid>
                      ))}
                    </Grid>
                  </section>
                )}
                {Object.keys(info.arrays).length > 0 && (
                  <Typography variant="body2">
                    {t('datasets.import.arrays')}:{' '}
                    {Object.entries(info.arrays)
                      .map(([k, v]) => `${k} ${JSON.stringify(v)}`)
                      .join('; ')}
                  </Typography>
                )}
                {info.preview.length > 0 && (
                  <Box sx={{ overflowX: 'auto' }}>
                    <Table size="small" aria-label={t('datasets.import.preview')}>
                      <TableHead>
                        <TableRow>
                          {info.columns.map((c) => (
                            <TableCell key={c}>{c}</TableCell>
                          ))}
                        </TableRow>
                      </TableHead>
                      <TableBody>
                        {info.preview.map((row, i) => (
                          <TableRow key={i}>
                            {info.columns.map((c) => (
                              <TableCell key={c}>{row[c]}</TableCell>
                            ))}
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </Box>
                )}
                <Grid container spacing={2}>
                  <Grid size={{ xs: 12, md: 4 }}>
                    <TextField fullWidth size="small" label={t('datasets.import.id')} value={datasetId} onChange={(e) => setDatasetId(e.target.value)} />
                  </Grid>
                  <Grid size={{ xs: 12, md: 4 }}>
                    <TextField fullWidth size="small" label={t('datasets.import.name')} value={displayName} onChange={(e) => setDisplayName(e.target.value)} />
                  </Grid>
                  <Grid size={{ xs: 6, md: 2 }}>
                    <TextField select fullWidth size="small" label={t('datasets.import.origin')} value={origin} onChange={(e) => setOrigin(e.target.value as DatasetOrigin)}>
                      {ORIGINS.map((o) => (
                        <MenuItem key={o} value={o}>
                          {o}
                        </MenuItem>
                      ))}
                    </TextField>
                  </Grid>
                  <Grid size={{ xs: 6, md: 2 }}>
                    <TextField fullWidth size="small" label={t('datasets.import.guard')} value={guard} onChange={(e) => setGuard(e.target.value)} slotProps={{ htmlInput: { inputMode: 'numeric' } }} />
                  </Grid>
                </Grid>
                <section aria-label={t('datasets.import.signal')}>
                  <Typography variant="h3" component="h3" gutterBottom>
                    {t('datasets.import.signal')}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" gutterBottom>
                    {t('datasets.import.signal.help')}
                  </Typography>
                  <SignalFields value={signal} onChange={setSignal} />
                </section>
                {importDs.isError && <ErrorState error={importDs.error} />}
              </Stack>
            )}
          </Grid>
        </Grid>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>{t('form.cancel')}</Button>
        <Button variant="contained" onClick={submit} disabled={!canImport}>
          {importDs.isPending ? t('datasets.import.importing') : t('datasets.import.submit')}
        </Button>
      </DialogActions>
    </Dialog>
  )
}
