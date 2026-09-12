import CheckCircleIcon from '@mui/icons-material/CheckCircle'
import Box from '@mui/material/Box'
import Tab from '@mui/material/Tab'
import Tabs from '@mui/material/Tabs'
import { t } from '@/i18n'
import { useStudioColors } from '@/theme'

/** A numbered, keyboard-accessible step strip. Completion is supplied by the validated form. */
export function WorkflowSteps({ active, labels, completed, onChange, canOpen, ariaLabel }: {
  active: number; labels: string[]; completed: boolean[]; onChange: (step: number) => void; canOpen: (step: number) => boolean; ariaLabel?: string
}) {
  const colors = useStudioColors()
  return <Tabs value={active} onChange={(_event, step: number) => onChange(step)} aria-label={ariaLabel ?? t('workflow.steps')} variant="fullWidth" sx={{ border: 1, borderColor: 'divider', borderRadius: 1.5, bgcolor: 'background.paper', '& .MuiTabs-indicator': { display: 'none' } }}>
    {labels.map((label, index) => <Tab key={label} disabled={!canOpen(index)} id={`workflow-step-${index}`} aria-controls={`workflow-panel-${index}`} sx={{ minHeight: 58, '&.Mui-selected': { bgcolor: colors.selected, boxShadow: `inset 0 -3px ${colors.primary}` } }} label={<Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
      {completed[index] ? <CheckCircleIcon color="success" titleAccess={t('workflow.complete')} data-testid={`step-${index}-complete`} /> : <Box component="span" sx={{ width: 24, height: 24, display: 'grid', placeItems: 'center', borderRadius: '50%', fontSize: 12, border: 1, borderColor: active === index ? 'primary.main' : 'divider', bgcolor: active === index ? 'primary.main' : 'transparent', color: active === index ? 'primary.contrastText' : 'inherit' }}>{index + 1}</Box>}
      {label}
    </Box>} />)}
  </Tabs>
}
