import Chip from '@mui/material/Chip'
import Tooltip from '@mui/material/Tooltip'
import type { EvidenceType } from '@/api/types'
import { t } from '@/i18n'
import { useStudioColors } from '@/theme'

/** Evidence type of a result, plus the MOCK stripe that disables export (UX spec §3). */
export function EvidenceBadge({ evidence, mock = false }: { evidence: EvidenceType; mock?: boolean }) {
  const colors = useStudioColors()
  const color = mock ? colors.evidence.mock : colors.evidence[evidence]
  const label = mock ? `${t('evidence.mock')} · ${t(`evidence.${evidence}`)}` : t(`evidence.${evidence}`)
  const hint = mock ? t('evidence.mock.hint') : t(`evidence.${evidence}.hint`)
  return (
    <Tooltip title={hint}>
      <Chip
        size="small"
        label={label}
        data-evidence={evidence}
        data-mock={mock ? 'true' : undefined}
        sx={{
          color,
          borderColor: color,
          fontWeight: 700,
          letterSpacing: 0.3,
          backgroundImage: mock ? `repeating-linear-gradient(135deg, transparent 0 6px, ${color}22 6px 12px)` : undefined,
        }}
        variant="outlined"
      />
    </Tooltip>
  )
}
