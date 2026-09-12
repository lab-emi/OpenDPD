import ArrowForwardIcon from '@mui/icons-material/ArrowForward'
import CheckCircleIcon from '@mui/icons-material/CheckCircle'
import ModelTrainingIcon from '@mui/icons-material/ModelTraining'
import Box from '@mui/material/Box'
import ButtonBase from '@mui/material/ButtonBase'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import { Link as RouterLink } from 'react-router'
import { t } from '@/i18n'
import { useStudioColors } from '@/theme'

export const EXPERIMENT_TASKS = ['train_pa', 'evaluate_pa', 'train_dpd', 'run_dpd'] as const
export type ExperimentTask = (typeof EXPERIMENT_TASKS)[number]
export const isExperimentTask = (value: string | null): value is ExperimentTask => EXPERIMENT_TASKS.includes(value as ExperimentTask)
export const taskLabel = (task: string) => isExperimentTask(task) ? t(`tasks.${task}`) : task.replaceAll('_', ' ')

export function ExperimentTasks({ active, dataset, version, compact = false }: { active?: string; dataset?: string; version?: string; compact?: boolean }) {
  const colors = useStudioColors()
  return <Box component="nav" aria-label={t('tasks.choose')} sx={{ display: 'grid', gridTemplateColumns: { xs: compact ? 'repeat(2, minmax(0, 1fr))' : 'minmax(0, 1fr)', sm: 'repeat(2, minmax(0, 1fr))', lg: 'repeat(4, minmax(0, 1fr))' }, gap: 1.25, '& > a': { minWidth: 0, overflowWrap: 'anywhere' } }}>
    {EXPERIMENT_TASKS.map((task, index) => {
      const params = new URLSearchParams({ task })
      if (dataset) params.set('dataset', dataset)
      if (version) params.set('version', version)
      const selected = active === task
      const Icon = task === 'evaluate_pa' || task === 'run_dpd' ? CheckCircleIcon : ModelTrainingIcon
      return <ButtonBase component={RouterLink} to={`/experiments/new?${params}`} key={task} aria-current={selected ? 'page' : undefined} sx={{ p: compact ? 1.5 : 2, display: 'block', textAlign: 'left', height: '100%', border: 1, borderColor: selected ? 'primary.main' : 'divider', borderRadius: 1.5, bgcolor: selected ? colors.selected : 'background.paper', boxShadow: selected ? `inset 0 -3px ${colors.primary}` : 'none', '&:hover': { borderColor: 'primary.main', bgcolor: colors.surfaceMuted }, '&.Mui-focusVisible': { outline: `3px solid ${colors.primary}`, outlineOffset: 2 } }}>
        <Stack direction="row" spacing={1} sx={{ alignItems: 'center', color: selected ? 'primary.main' : 'text.secondary', mb: .75 }}><Icon fontSize="small" /><Typography variant="caption">0{index + 1}</Typography><Box sx={{ flex: 1 }} /><ArrowForwardIcon sx={{ fontSize: 16 }} /></Stack>
        <Typography sx={{ fontSize: compact ? 13 : 15, fontWeight: 700 }}>{taskLabel(task)}</Typography>
        {!compact && <Typography variant="body2" color="text.secondary" sx={{ mt: .75 }}>{t(`tasks.${task}.help`)}</Typography>}
      </ButtonBase>
    })}
  </Box>
}
