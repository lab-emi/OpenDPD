import CheckIcon from '@mui/icons-material/Check'
import AddIcon from '@mui/icons-material/Add'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import ButtonBase from '@mui/material/ButtonBase'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import { useState } from 'react'
import type { GeneratorPreset } from '@/api/signalGenerator'
import { t } from '@/i18n'
import { useStudioColors } from '@/theme'

export function PresetMatrix({ presets, selected, disabled, toggle }: {
  presets: GeneratorPreset[]; selected: string[]; disabled: boolean; toggle: (preset: GeneratorPreset) => void
}) {
  const colors = useStudioColors()
  const groups = [...new Set(presets.map(p => p.numerology ?? 'OFDM'))]
  const [choice, setChoice] = useState(groups[0])
  const group = groups.includes(choice ?? '') ? choice : groups[0]
  const visible = presets.filter(p => (p.numerology ?? 'OFDM') === group)
  const channels = [...new Set(visible.map(p => p.channel_count ?? p.config.channel_subcarriers?.length ?? 1))].sort((a, b) => a-b)
  return <Stack spacing={1.25}>
    {groups.length > 1 && <Stack direction="row" useFlexGap sx={{ gap: .75, flexWrap: 'wrap' }} aria-label={t('generator.numerology')}>
      {groups.map(g => <Button key={g} size="small" variant={g === group ? 'contained' : 'outlined'} aria-pressed={g === group} disabled={disabled} onClick={() => setChoice(g)}>{g}</Button>)}
    </Stack>}
    <Typography variant="caption" color="text.secondary">{t('generator.matrixHelp')}</Typography>
    <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', lg: 'repeat(2,minmax(0,1fr))' }, gap: 1.5 }}>
      {channels.map(count => {
        const entries = visible.filter(p => (p.channel_count ?? p.config.channel_subcarriers?.length ?? 1) === count)
        const bands = [...new Set(entries.map(p => p.config.bandwidth_hz))].sort((a,b) => (a ?? 0)-(b ?? 0))
        const orders = [...new Set(entries.map(p => p.config.channel_modulations?.[0] ?? 64))].sort((a,b) => a-b)
        const label = t('generator.channelGroup', { count })
        return <Box key={count} sx={{ border: 1, borderColor: 'divider', borderRadius: 1.5, p: 1, minWidth: 0 }}>
          <Typography variant="subtitle2" sx={{ mb: .75 }}>{label}</Typography>
          <Box role="region" tabIndex={0} aria-label={label} sx={{ overflowX: 'auto' }}>
            <Box component="table" aria-label={label + ' · ' + group} sx={{ width: '100%', borderSpacing: '3px', tableLayout: 'fixed', minWidth: 80 + bands.length * 31 }}>
              <caption style={{ textAlign: 'right', fontSize: 11 }}>{t('generator.bandwidthAxis')} →</caption>
              <thead><tr><Box component="th" scope="col" sx={{ width: 65, fontSize: 10, textAlign: 'left', color: 'text.secondary' }}>QAM ↓</Box>{bands.map(b => <Box component="th" scope="col" key={b} sx={{ fontSize: 11, fontWeight: 650 }}>{(b ?? 0) / 1e6}</Box>)}</tr></thead>
              <tbody>{orders.map(order => <tr key={order}>
                <Box component="th" scope="row" sx={{ fontSize: 11, fontWeight: 550, textAlign: 'left' }}>{order === 2 ? 'BPSK' : order === 4 ? 'QPSK' : order}</Box>
                {bands.map(band => {
                  const p = entries.find(e => e.config.bandwidth_hz === band && (e.config.channel_modulations?.[0] ?? 64) === order)
                  return <td key={band}>{p ? <ButtonBase disabled={disabled || (!selected.includes(p.preset_id) && selected.length >= 16)}
                    aria-label={p.label + ' · ' + group} title={p.description} aria-pressed={selected.includes(p.preset_id)}
                    data-testid={'preset-' + p.preset_id} onClick={() => toggle(p)} sx={{ width: '100%', height: 28, border: 1, borderRadius: .75,
                      borderColor: selected.includes(p.preset_id) ? 'primary.main' : 'divider',
                      color: selected.includes(p.preset_id) ? 'primary.main' : 'text.secondary', bgcolor: selected.includes(p.preset_id) ? colors.selected : 'background.default',
                      '&:hover': { borderColor: 'primary.main', bgcolor: colors.selected }, '&:focus-visible': { outline: '2px solid', outlineColor: 'primary.main', outlineOffset: 1 } }}>
                    {selected.includes(p.preset_id) ? <CheckIcon sx={{ fontSize: 16 }} /> : <AddIcon sx={{ fontSize: 13, opacity: .65 }} />}
                  </ButtonBase> : <span aria-label={t('common.na')}>—</span>}</td>
                })}
              </tr>)}</tbody>
            </Box>
          </Box>
        </Box>
      })}
    </Box>
  </Stack>
}
