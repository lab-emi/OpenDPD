import TerminalIcon from '@mui/icons-material/Terminal'
import StopCircleOutlinedIcon from '@mui/icons-material/StopCircleOutlined'
import Alert from '@mui/material/Alert'
import Button from '@mui/material/Button'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import Box from '@mui/material/Box'
import ButtonBase from '@mui/material/ButtonBase'
import Chip from '@mui/material/Chip'
import Collapse from '@mui/material/Collapse'
import MenuItem from '@mui/material/MenuItem'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import Tab from '@mui/material/Tab'
import Tabs from '@mui/material/Tabs'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { useState } from 'react'
import { Link as RouterLink, useLocation } from 'react-router'
import { useCancelRun, useRun, useRuns } from '@/api/hooks'
import { isTerminal } from '@/api/types'
import { message, t } from '@/i18n'
import { useStudioColors } from '@/theme'
import { isExperimentTask, taskLabel, type ExperimentTask } from './ExperimentTasks'
import { LogViewer } from './LogViewer'

const TASKS: ExperimentTask[] = ['train_pa', 'evaluate_pa', 'train_dpd', 'run_dpd']

/** The console follows navigation once; changing its tabs never navigates the workspace. */
export function ExperimentTerminal() {
  const colors = useStudioColors()
  const { pathname, search } = useLocation()
  const runs = useRuns()
  const cancel = useCancelRun()
  const id = pathname.startsWith('/runs/') ? decodeURIComponent(pathname.split('/')[2] ?? '') : ''
  const detail = useRun(id, !!id)
  const routeRun = detail.data ?? runs.data?.find((run) => run.run_id === id)
  const records = routeRun && !runs.data?.some((run) => run.run_id === routeRun.run_id) ? [routeRun, ...(runs.data ?? [])] : runs.data ?? []
  const requested = new URLSearchParams(search).get('task')
  const contextTask = isExperimentTask(routeRun?.task ?? null) ? routeRun!.task as ExperimentTask : isExperimentTask(requested) ? requested : 'train_pa'
  const context = `${pathname}:${contextTask}:${id}`
  const [manualTab, setManualTab] = useState<{ context: string; task: ExperimentTask } | null>(null)
  const tab = manualTab?.context === context ? manualTab.task : contextTask
  const [expanded, setExpanded] = useState(false)
  const [selection, setSelection] = useState<Partial<Record<ExperimentTask, { id: string; context: string }>>>({})
  const candidates = records.filter((run) => run.task === tab)
  const selectedId = selection[tab]?.context === context ? selection[tab]?.id : tab === contextTask ? id : ''
  const chosen = candidates.find((run) => run.run_id === selectedId) ?? candidates[0]
  const running = (runs.data ?? []).filter((run) => !isTerminal(run.status)).length
  if (!pathname.startsWith('/experiments') && !pathname.startsWith('/runs/')) return null
  return <Paper component="section" aria-label={t('terminal.title')} data-testid="experiment-terminal" sx={{ mt: 3, overflow: 'hidden', borderColor: running ? 'primary.main' : 'divider' }}>
    <ButtonBase onClick={() => setExpanded((old) => !old)} aria-expanded={expanded} aria-controls="experiment-terminal-panel"
      sx={{ width: '100%', p: 1.6, justifyContent: 'flex-start', gap: 1.3, bgcolor: running ? colors.selected : colors.surfaceMuted, color: running ? 'primary.main' : 'text.secondary', '&:hover': { bgcolor: colors.selectedHover } }}>
      <TerminalIcon sx={{ p: .4, borderRadius: 1, boxSizing: 'content-box', bgcolor: running ? colors.selectedHover : colors.border }} />
      <Typography sx={{ fontWeight: 700, color: 'inherit' }}>{t('terminal.title')}</Typography>
      {running > 0 && <Chip size="small" color="primary" label={`${t('status.running')} · ${running}`} />}
      <Typography variant="body2" sx={{ ml: 1, display: { xs: 'none', md: 'block' } }}>{taskLabel(tab)}</Typography>
      <ExpandMoreIcon sx={{ ml: 'auto', transform: expanded ? 'rotate(180deg)' : undefined, transition: 'transform 150ms' }} />
    </ButtonBase>
    <Collapse in={expanded} unmountOnExit>
      <Box id="experiment-terminal-panel" sx={{ p: 2 }}>
        <Tabs value={tab} onChange={(_, value: ExperimentTask) => setManualTab({ context, task: value })} variant="scrollable" scrollButtons="auto" aria-label={t('terminal.tabs')}>
          {TASKS.map((task) => <Tab key={task} value={task} id={`terminal-tab-${task}`} aria-controls="terminal-log-panel" label={taskLabel(task)} />)}
        </Tabs>
        <Stack id="terminal-log-panel" role="tabpanel" aria-labelledby={`terminal-tab-${tab}`} spacing={1.5} sx={{ pt: 2 }}>
          <Typography variant="caption" color="text.secondary">{t('terminal.independent')}</Typography>
          {chosen ? <>
            <TextField select fullWidth size="small" label={t('terminal.run')} value={chosen.run_id} onChange={(event) => setSelection((old) => ({ ...old, [tab]: { id: event.target.value, context } }))}>
              {candidates.map((run) => <MenuItem key={run.run_id} value={run.run_id}>{run.name || run.run_id} · {message(run.status)} · {run.model_key}</MenuItem>)}
            </TextField>
            <Stack direction="row" spacing={1}>
              {!isTerminal(chosen.status) && <Button color="error" variant="outlined" startIcon={<StopCircleOutlinedIcon />} disabled={cancel.isPending || chosen.status === 'cancel_requested'} onClick={() => cancel.mutate(chosen.run_id)}>{t(chosen.status === 'cancel_requested' ? 'terminal.stopping' : 'terminal.stop')}</Button>}
              <Button component={RouterLink} to={`/runs/${encodeURIComponent(chosen.run_id)}`}>{t('terminal.openRun')}</Button>
            </Stack>
            {cancel.isError && cancel.variables === chosen.run_id && <Alert severity="error">{message(cancel.error.message)}</Alert>}
            <Typography variant="caption" color="text.secondary">{t('terminal.readOnly')}</Typography>
            <LogViewer key={chosen.run_id} runId={chosen.run_id} live={!isTerminal(chosen.status)} height={300} tail />
          </> : <Typography color="text.secondary" sx={{ py: 3 }}>{t('terminal.empty')}</Typography>}
        </Stack>
      </Box>
    </Collapse>
  </Paper>
}
