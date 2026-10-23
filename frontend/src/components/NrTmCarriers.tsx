import AddIcon from '@mui/icons-material/Add'
import DeleteOutlineIcon from '@mui/icons-material/DeleteOutlined'
import RemoveIcon from '@mui/icons-material/Remove'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import IconButton from '@mui/material/IconButton'
import MenuItem from '@mui/material/MenuItem'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import type { NrTmCarrier, NrTmCatalog, NrTmModelInfo, NrTmRecipe, NrTmSummary } from '@/api/nrTestModels'
import { formatNumber, t } from '@/i18n'
import { bandwidthOptions, fitCarrier, removeCarrier, resizeCarriers, scsOptions } from '@/utils/nrTestModel'
import { NumberInput } from './NrTmFields'

/**
 * Carrier rows of a test-model recipe: subcarrier spacing and channel bandwidth (filtered by the N_RB table of the catalogue),
 * cell ID, and the gap before every carrier after the first. The numbers beside the selects come from the server's summary
 * when it matches the recipe, else from the catalogue's N_RB table; nothing here is computed from the standard.
 */
export function NrTmCarriers({ catalog, info, recipe, summary, disabled, onChange }: {
  catalog: NrTmCatalog; info?: NrTmModelInfo; recipe: NrTmRecipe; summary?: NrTmSummary; disabled: boolean; onChange: (recipe: NrTmRecipe) => void
}) {
  const max = catalog.limits.max_carriers
  const count = recipe.carriers.length
  const spacings = scsOptions(catalog, info)
  const patch = (index: number, change: Partial<NrTmCarrier>) => onChange({ ...recipe, carriers: recipe.carriers.map((c, i) => i === index ? { ...c, ...change } : c) })
  return <Stack spacing={1} role="group" aria-label={t('nrtm.carriers')}>
    <Stack direction="row" useFlexGap sx={{ alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 1 }}>
      <Typography variant="subtitle2" component="h4">{t('nrtm.carriers')} · {count}</Typography>
      <Stack direction="row" spacing={.5} sx={{ alignItems: 'center', flexWrap: 'wrap' }} useFlexGap role="group" aria-label={t('nrtm.carrierCount')}>
        <IconButton size="small" aria-label={t('nrtm.fewerCarriers')} disabled={disabled || count <= 1} onClick={() => onChange(resizeCarriers(recipe, count - 1, max))}><RemoveIcon fontSize="small" /></IconButton>
        <Typography component="output" aria-live="polite" sx={{ minWidth: 24, textAlign: 'center', fontWeight: 650 }} data-testid="nrtm-carrier-count">{count}</Typography>
        <IconButton size="small" aria-label={t('nrtm.moreCarriers')} disabled={disabled || count >= max} onClick={() => onChange(resizeCarriers(recipe, count + 1, max))}><AddIcon fontSize="small" /></IconButton>
        {count >= max && <Typography variant="caption" color="text.secondary" data-testid="nrtm-carrier-limit">{t('nrtm.carrierLimit', { max })}</Typography>}
      </Stack>
    </Stack>
    {recipe.carriers.map((carrier, index) => {
      const widths = bandwidthOptions(catalog, info, carrier.scs_khz)
      const row = summary?.carriers[index]
      const nrb = row?.nrb ?? catalog.resource_blocks[String(carrier.scs_khz)]?.[String(carrier.bandwidth_mhz)]
      const name = t('nrtm.carrierN', { index: index + 1 })
      return <Paper key={index} role="group" aria-label={name} sx={{ p: 1.25 }}>
        <Box sx={{ display: 'grid', gap: 1.25, alignItems: 'start', gridTemplateColumns: { xs: 'minmax(0,1fr)', sm: 'repeat(2,minmax(0,1fr))', md: 'minmax(0,.55fr) minmax(0,1fr) minmax(0,1fr) minmax(0,1.1fr) minmax(0,1fr) minmax(0,1fr) auto' } }}>
          <Typography variant="body2" sx={{ fontWeight: 650, alignSelf: 'center' }}>{name}</Typography>
          <TextField select label={t('nrtm.scs')} value={carrier.scs_khz} disabled={disabled} onChange={e => patch(index, fitCarrier(catalog, info, { ...carrier, scs_khz: Number(e.target.value) as NrTmCarrier['scs_khz'] }))}>
            {spacings.map(scs => <MenuItem key={scs} value={scs}>{scs}</MenuItem>)}
          </TextField>
          <TextField select label={t('nrtm.bandwidth')} value={carrier.bandwidth_mhz} disabled={disabled} onChange={e => patch(index, { bandwidth_mhz: Number(e.target.value) })}>
            {(widths.includes(carrier.bandwidth_mhz) ? widths : [...widths, carrier.bandwidth_mhz].sort((a, b) => a - b)).map(bandwidth => <MenuItem key={bandwidth} value={bandwidth}>{bandwidth}</MenuItem>)}
          </TextField>
          <Typography variant="body2" color="text.secondary" sx={{ alignSelf: 'center' }} data-testid={`nrtm-nrb-${index}`}>
            {nrb === undefined ? '—' : t('nrtm.nrb', { nrb: formatNumber(nrb), subcarriers: formatNumber(row?.subcarriers ?? 12 * nrb) })}
          </Typography>
          <NumberInput label={t('nrtm.cellId')} value={carrier.n_cell_id} allowEmpty min={0} max={catalog.limits.max_cell_id} disabled={disabled}
            placeholder={row ? t('nrtm.cellIdAuto', { id: row.n_cell_id }) : t('nrtm.auto')} onCommit={value => patch(index, { n_cell_id: value === null ? null : Math.trunc(value) })} />
          {index > 0 ? <NumberInput label={t('nrtm.gap')} value={carrier.gap_before_mhz} min={0} step={catalog.limits.carrier_gap_step_mhz} disabled={disabled}
            onCommit={value => patch(index, { gap_before_mhz: value ?? 0 })} /> : <span />}
          <IconButton aria-label={t('nrtm.removeCarrier', { index: index + 1 })} disabled={disabled || count <= 1} onClick={() => onChange(removeCarrier(recipe, index))}
            sx={{ justifySelf: 'end', gridColumn: { xs: 1, sm: 2, md: 'auto' }, gridRow: { xs: 1, md: 'auto' } }}><DeleteOutlineIcon fontSize="small" /></IconButton>
        </Box>
      </Paper>
    })}
    {count > 1 && <Typography variant="caption" color="text.secondary">{t('nrtm.gapHelp', { step: catalog.limits.carrier_gap_step_mhz })}</Typography>}
    <Button size="small" variant="outlined" startIcon={<AddIcon />} disabled={disabled || count >= max} onClick={() => onChange(resizeCarriers(recipe, count + 1, max))} sx={{ alignSelf: 'flex-start' }}>{t('nrtm.addCarrier')}</Button>
  </Stack>
}
