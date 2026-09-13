import { useState } from 'react'
import { Alert, Button, Dialog, DialogActions, DialogContent, DialogTitle, Link, MenuItem, Stack, TextField, Typography } from '@mui/material'
import { Link as RouterLink } from 'react-router'
import { useMutation, useQueryClient } from '@tanstack/react-query'
import { api, WEB_MODE } from '@/api/client'
import { keys } from '@/api/hooks'
import type { components } from '@/api/schema'
import { t } from '@/i18n'
import { ErrorState } from './StateBlock'

export function SyntheticDatasetDialog({ onClose }: { onClose: () => void }) {
  const qc = useQueryClient()
  const [prefix, setPrefix] = useState('synthetic-research')
  const [seed, setSeed] = useState('20260913')
  const [samples, setSamples] = useState(16384)
  const [repeats, setRepeats] = useState(2)
  const generate = useMutation({ mutationFn: () => api.post<components['schemas']['SyntheticSuite']>('/datasets/synthetic', { prefix, seed: Number(seed), samples_per_capture: samples, repeats }), onSuccess: () => void qc.invalidateQueries({ queryKey: keys.datasets }) })
  const valid = /^[a-z0-9][a-z0-9-]{0,35}$/.test(prefix) && /^\d+$/.test(seed) && Number(seed) <= 2 ** 32 - 1
  return <Dialog open fullWidth maxWidth="sm" onClose={generate.isPending ? undefined : onClose} aria-labelledby="synthetic-title">
    <DialogTitle id="synthetic-title">{t('datasetResearch.syntheticTitle')}</DialogTitle>
    <DialogContent dividers><Stack spacing={2}>
      <Alert severity="warning">{t('datasetResearch.syntheticNotice')}</Alert>
      <Typography>{t('datasetResearch.syntheticRecipe')}</Typography>
      {!generate.data && <>
        <TextField size="small" label={t('datasetResearch.prefix')} value={prefix} disabled={generate.isPending} onChange={e => setPrefix(e.target.value)} />
        <TextField size="small" label={t('sweep.seed')} value={seed} disabled={generate.isPending} onChange={e => setSeed(e.target.value)} slotProps={{ htmlInput: { inputMode: 'numeric' } }} />
        <TextField select size="small" label={t('datasetResearch.samples')} value={samples} disabled={generate.isPending} onChange={e => setSamples(Number(e.target.value))}>{[8192, 16384, 32768, 65536, 131072].map(n => <MenuItem key={n} value={n}>{n.toLocaleString()}</MenuItem>)}</TextField>
        <TextField select size="small" label={t('datasetResearch.repeats')} value={repeats} disabled={generate.isPending} onChange={e => setRepeats(Number(e.target.value))}>{[1, 2, 3, 4].map(n => <MenuItem key={n} value={n}>{n}</MenuItem>)}</TextField>
      </>}
      {generate.data && <>
        <Alert severity="success">{t('datasetResearch.generated', { count: generate.data.datasets.length })}</Alert>
        {generate.data.datasets.map(d => <Link key={d.dataset_id} component={RouterLink} to={`/datasets/${d.dataset_id}`} onClick={onClose}>{d.display_name}</Link>)}
        {!WEB_MODE && <Button component={RouterLink} to="/sweeps" state={{ conditionSet: generate.data.condition_set }}>{t('datasetResearch.useConditions')}</Button>}
      </>}
      {generate.isError && <ErrorState error={generate.error} />}
    </Stack></DialogContent>
    <DialogActions><Button onClick={onClose} disabled={generate.isPending}>{t('form.cancel')}</Button>{!generate.data && <Button variant="contained" disabled={!valid || generate.isPending} onClick={() => generate.mutate()}>{t('datasetResearch.generate')}</Button>}</DialogActions>
  </Dialog>
}
