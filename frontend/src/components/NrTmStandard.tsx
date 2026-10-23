import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import Accordion from '@mui/material/Accordion'
import AccordionDetails from '@mui/material/AccordionDetails'
import AccordionSummary from '@mui/material/AccordionSummary'
import Box from '@mui/material/Box'
import Chip from '@mui/material/Chip'
import Stack from '@mui/material/Stack'
import Table from '@mui/material/Table'
import TableBody from '@mui/material/TableBody'
import TableCell from '@mui/material/TableCell'
import TableHead from '@mui/material/TableHead'
import TableRow from '@mui/material/TableRow'
import Typography from '@mui/material/Typography'
import type { NrTmCarrierSummary, NrTmCatalog, NrTmModelInfo, NrTmRecipe, NrTmSummary } from '@/api/nrTestModels'
import { formatNumber, t } from '@/i18n'
import { slotLabel } from '@/utils/nrTestModel'
import { Facts, Scroll } from './NrTmFacts'

const level = (db: number) => `${formatNumber(db, { signDisplay: 'exceptZero', minimumFractionDigits: 2, maximumFractionDigits: 2 })} dB`
const ranges = (list: number[][]) => list.map(([first, last]) => first === last ? String(first) : `${first}–${last}`).join(', ')

function CarrierBlock({ carrier }: { carrier: NrTmCarrierSummary }) {
  const title = t('nrtm.carrierSummary', { index: carrier.index + 1, scs: carrier.scs_khz, bandwidth: carrier.bandwidth_mhz })
  const facts: Array<[string, string]> = [
    ['N_RB', formatNumber(carrier.nrb)], [t('nrtm.subcarriers'), formatNumber(carrier.subcarriers)], ['FFT', formatNumber(carrier.n_fft)], [t('nrtm.cellId'), String(carrier.n_cell_id)],
    [t('nrtm.centerOffset'), `${formatNumber(carrier.center_offset_hz / 1e6, { maximumFractionDigits: 3 })} MHz`],
    [t('nrtm.occupied'), `${formatNumber(carrier.occupied_bandwidth_hz / 1e6, { maximumFractionDigits: 3 })} MHz`],
    [t('nrtm.slots'), `${formatNumber(carrier.slots)} / ${formatNumber(carrier.slots_per_frame)}`],
  ]
  const boosted = carrier.boosted_rbgs ?? [], users = carrier.users ?? []
  if (carrier.rbg_size) facts.push(['RBG P', String(carrier.rbg_size)], ['N_RBG', String(carrier.n_rbg ?? '—')], [t('nrtm.boostedRbgs'), boosted.length ? boosted.join(', ') : '—'])
  return <Stack spacing={.75} component="section" aria-label={title} data-testid={`nrtm-carrier-summary-${carrier.index}`}>
    <Typography variant="subtitle2" component="h5">{title}</Typography>
    <Facts items={facts} />
    {!!users.length && <Scroll label={`${title} · ${t('nrtm.users')}`}><Table size="small">
      <TableHead><TableRow>{['nRNTI', t('nrtm.role'), t('nrtm.modulation'), t('nrtm.prbs'), t('nrtm.allocation'), t('nrtm.startSymbol'), t('nrtm.level'), 'EVM'].map(head => <TableCell key={head} sx={{ whiteSpace: 'nowrap' }}>{head}</TableCell>)}</TableRow></TableHead>
      <TableBody>{users.map(user => <TableRow key={user.rnti}>
        <TableCell>{user.rnti}</TableCell><TableCell sx={{ minWidth: 150 }}>{user.role}</TableCell><TableCell>{user.modulation}</TableCell><TableCell>{formatNumber(user.prbs)}</TableCell>
        <TableCell sx={{ minWidth: 110, overflowWrap: 'anywhere' }}>{ranges(user.rb_ranges)}</TableCell><TableCell>{user.start_symbol}</TableCell><TableCell sx={{ whiteSpace: 'nowrap' }}>{level(user.level_db)}</TableCell>
        <TableCell>{user.evm_measured ? t('nrtm.evmMeasured') : '—'}</TableCell>
      </TableRow>)}</TableBody>
    </Table></Scroll>}
  </Stack>
}

/** "Defined by the standard": what the test model fixes (from the catalogue) and what it resolves to for the current recipe (from the server). */
export function NrTmStandard({ catalog, info, recipe, summary, stale }: { catalog: NrTmCatalog; info?: NrTmModelInfo; recipe: NrTmRecipe; summary?: NrTmSummary; stale: boolean }) {
  const spacings = [...new Set(recipe.carriers.map(c => c.scs_khz))]
  return <Accordion disableGutters data-testid="nrtm-standard" slotProps={{ heading: { component: 'h4' } }}>
    <AccordionSummary expandIcon={<ExpandMoreIcon />}><Typography sx={{ fontWeight: 650 }}>{t('nrtm.defined')}</Typography></AccordionSummary>
    <AccordionDetails><Stack spacing={1.5} aria-busy={stale} sx={{ opacity: stale ? .55 : 1, minWidth: 0 }}>
      <Typography variant="caption" color="text.secondary">{t('nrtm.definedHelp')}</Typography>
      {!!info?.fixed_parameters.length && <Box component="section" aria-label={t('nrtm.fixed')} data-testid="nrtm-fixed">
        <Box component="dl" sx={{ m: 0 }}>{info.fixed_parameters.map(row => <Box key={row.name} sx={{ display: 'grid', gridTemplateColumns: { xs: 'minmax(0,1fr)', sm: 'minmax(120px,30%) minmax(0,1fr)' }, columnGap: 2, rowGap: .25, py: .75, borderBottom: 1, borderColor: 'divider' }}>
          <Typography component="dt" variant="body2" sx={{ fontWeight: 600, overflowWrap: 'anywhere' }}>{row.name}</Typography>
          <Typography component="dd" variant="body2" sx={{ m: 0, overflowWrap: 'anywhere' }}>{row.value}</Typography>
        </Box>)}</Box>
      </Box>}
      {summary ? summary.carriers.map(carrier => <CarrierBlock key={carrier.index} carrier={carrier} />) : <Typography variant="body2" color="text.secondary">{t('nrtm.resolving')}</Typography>}
      {recipe.duplex === 'tdd' && <Stack spacing={1} component="section" aria-label={t('nrtm.tdd')}>
        <Typography variant="subtitle2" component="h5">{t('nrtm.tdd')}</Typography>
        {spacings.map(scs => {
          const pattern = catalog.tdd_patterns.find(p => p.scs_khz === scs)
          if (!pattern) return null
          const heading = `${scs} kHz · ${recipe.carriers.flatMap((c, i) => c.scs_khz === scs ? [t('nrtm.carrierN', { index: i + 1 })] : []).join(', ')}`
          return <Stack spacing={.75} key={scs} data-testid={`nrtm-tdd-${scs}`}>
            {spacings.length > 1 && <Typography variant="body2" sx={{ fontWeight: 600 }}>{heading}</Typography>}
            <Typography variant="body2" color="text.secondary">{t('nrtm.tddHelp', { period: pattern.period_ms, dlSlots: pattern.dl_slots, dlSymbols: pattern.dl_symbols, ulSymbols: pattern.ul_symbols, ulSlots: pattern.ul_slots })}</Typography>
            <Stack direction="row" useFlexGap sx={{ gap: .5, flexWrap: 'wrap' }} role="list" aria-label={spacings.length > 1 ? `${t('nrtm.tddSlots')} · ${scs} kHz` : t('nrtm.tddSlots')}>
              {pattern.slot_map.map(slotLabel).map((slot, i) => <Chip key={i} role="listitem" size="small" label={slot} variant={slot === 'D' ? 'filled' : 'outlined'} color={slot === 'D' ? 'primary' : slot.startsWith('S') ? 'warning' : 'default'} />)}
            </Stack>
          </Stack>
        })}
      </Stack>}
      {summary?.nb_iot_rbs?.length ? <Typography variant="body2">{t('nrtm.nbiotPunctured', { rbs: summary.nb_iot_rbs.join(', ') })}</Typography> : null}
    </Stack></AccordionDetails>
  </Accordion>
}
