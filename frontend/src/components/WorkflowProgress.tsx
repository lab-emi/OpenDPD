import AccountTreeIcon from '@mui/icons-material/AccountTree'
import OpenInFullIcon from '@mui/icons-material/OpenInFull'
import CloseIcon from '@mui/icons-material/Close'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Dialog from '@mui/material/Dialog'
import DialogContent from '@mui/material/DialogContent'
import DialogTitle from '@mui/material/DialogTitle'
import IconButton from '@mui/material/IconButton'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import { useState } from 'react'
import { useLocation, useNavigate } from 'react-router'
import { t } from '@/i18n'
import { useStudioColors } from '@/theme'
import { useStudioWorkflow } from '@/workflow/StudioWorkflow'

export function WorkflowProgress() {
  const workflow = useStudioWorkflow()
  const { state } = workflow
  const colors = useStudioColors()
  const { pathname, search } = useLocation()
  const navigate = useNavigate()
  const [expanded, setExpanded] = useState(false)
  const supplied = state.origin === 'existing' && !!state.datasetId
  const pair = !!state.datasetId
  const datasetRoute = state.datasetId ? '/datasets/' + encodeURIComponent(state.datasetId) : '/datasets?guide=start'
  const steps = [
    { key: 'input', label: t('paFlow.input'), symbol: 'x[n]', done: supplied || !!state.inputId, route: supplied ? datasetRoute : '/signal-generator' },
    { key: 'virtual', label: t('paFlow.virtual'), symbol: 'PA', done: supplied || !!state.modelId, route: supplied ? datasetRoute : '/pa-library' },
    { key: 'output', label: t('paFlow.output'), symbol: 'y[n]', done: pair || !!state.simulationId, route: pair ? datasetRoute : '/pa-library' },
    { key: 'pa', label: t('paFlow.trainPA'), symbol: 'PÂ', done: workflow.paDone,
      route: '/experiments/new?task=train_pa' + (state.datasetId ? '&dataset=' + encodeURIComponent(state.datasetId) : '') },
    { key: 'dpd', label: t('paFlow.trainDPD'), symbol: 'DPD', done: workflow.dpdDone,
      route: '/experiments/new?task=train_dpd' + (state.datasetId ? '&dataset=' + encodeURIComponent(state.datasetId) : '') + (state.paRunId ? '&paRun=' + encodeURIComponent(state.paRunId) : '') },
  ]
  const active = pathname === '/signal-generator' || pathname.startsWith('/signal-generator/') ? 0 : pathname === '/pa-library' ? (state.simulationId ? 2 : 1)
    : pathname.startsWith('/datasets') ? 2 : new URLSearchParams(search).get('task')?.includes('dpd') ? 4
      : pathname.startsWith('/experiments') || pathname.startsWith('/runs') ? (state.dpdRunId ? 4 : 3) : -1
  const currentStep = steps[active] ?? steps.find(step => !step.done) ?? steps[4]!
  const open = (index: number) => {
    setExpanded(false)
    const step = steps[index]!
    if (index >= 3 && !pair) return
    navigate(step.route)
  }
  const diagram = (large: boolean) => {
    const width = 990, height = large ? 350 : 86
    const positions = large ? [[100, 91], [490, 91], [880, 91], [280, 258], [680, 258]]
      : [[70, 43], [290, 43], [510, 43], [715, 43], [920, 43]]
    return <svg viewBox={'0 0 ' + width + ' ' + height} role="group" aria-label={t('paFlow.title')}
      style={{ display: 'block', width: '100%', minWidth: large ? 780 : 580, maxHeight: large ? undefined : 80 }}>
      <defs><marker id={large ? 'workflow-arrow-large' : 'workflow-arrow'} viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M 0 0 L 10 5 L 0 10 z" fill={colors.border} /></marker></defs>
      <rect x="3" y="1" width={large ? 984 : 588} height={large ? 151 : 83} rx="10" fill={colors.selected} opacity=".55" />
      <rect x={large ? 3 : 606} y={large ? 186 : 1} width={large ? 984 : 381} height={large ? 158 : 83} rx="10" fill={colors.surfaceMuted} />
      <text x="16" y={large ? 25 : 17} fill={colors.primary} fontSize={large ? 14 : 11} fontWeight="700">{t('paFlow.making')}</text>
      <text x={large ? 16 : 620} y={large ? 211 : 17} fill={colors.textSecondary} fontSize={large ? 14 : 11} fontWeight="700">{t('paFlow.learning')}</text>
      {steps.slice(1).map((_, index) => {
        const [x1, y1] = positions[index]!, [x2] = positions[index+1]!
        const path = large && index === 2 ? 'M 924 91 H 955 V 167 H 190 V 258 H 235'
          : 'M ' + (x1!+44) + ' ' + y1 + ' H ' + (x2!-45)
        return <path key={index} d={path} fill="none" stroke={colors.border} strokeWidth="1.8" markerEnd={'url(#' + (large ? 'workflow-arrow-large' : 'workflow-arrow') + ')'} />
      })}
      {steps.map((step, index) => {
        const [x, y] = positions[index]!
        const color = step.done ? colors.primary : colors.textSecondary
        const fill = step.done ? colors.selected : colors.surface
        const enabled = index < 3 || pair
        return <g key={step.key} role="button" tabIndex={enabled ? 0 : -1} aria-label={step.label + ' · ' + t(step.done ? 'paFlow.complete' : 'paFlow.pending')}
          aria-disabled={!enabled} data-testid={'workflow-' + step.key} data-complete={step.done}
          onClick={() => open(index)} onKeyDown={e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); open(index) } }}
          style={{ cursor: enabled ? 'pointer' : 'default', outlineColor: colors.primary }}>
          <title>{step.label + ': ' + t(step.done ? 'paFlow.complete' : 'paFlow.pending')}</title>
          {active === index && <rect x={x!-49} y={y!-24} width="98" height="49" rx="9" fill="none" stroke={colors.primary} strokeWidth="1.5" strokeDasharray="4 3" />}
          {index === 1 || index === 3 ? <polygon points={(x!-32) + ',' + (y!-20) + ' ' + (x!-32) + ',' + (y!+20) + ' ' + (x!+39) + ',' + y}
            fill={fill} stroke={color} strokeWidth="1.5" /> : <>
            <rect x={x!-40} y={y!-16} width="80" height="34" rx={index === 4 ? 4 : 7} fill={fill} stroke={color} strokeWidth="1.5" />
          </>}
          <text x={x!-(index === 1 || index === 3 ? 7 : 0)} y={y!+5} textAnchor="middle" fill={color} fontFamily="ui-monospace, monospace" fontWeight="700" fontSize="15">{step.symbol}</text>
          {step.done && <g><circle cx={x!+36} cy={y!-17} r="8" fill={colors.primary} /><path d={'M ' + (x!+32) + ' ' + (y!-17) + ' l 3 3 5 -6'} fill="none" stroke={colors.primaryContrast} strokeWidth="1.7" /></g>}
          <text x={x} y={y!+37} textAnchor="middle" fill={color} fontSize={large ? 14 : 10.5} fontWeight={active === index ? 700 : 500}>{step.label}</text>
        </g>
      })}
      {large && <text x="480" y="318" textAnchor="middle" fill={colors.textSecondary} fontSize="12">{t('paFlow.trainingHelp')}</text>}
    </svg>
  }
  return <>
    <Paper variant="outlined" sx={{ px: 1.5, py: .5, mb: 2, borderRadius: 2, position: 'sticky', top: 64, zIndex: 5 }} data-testid="studio-workflow">
      <Stack direction="row" sx={{ alignItems: 'center', gap: 1 }}>
        <AccountTreeIcon sx={{ color: 'primary.main', fontSize: 19, display: { xs: 'none', md: 'block' } }} />
        <Box sx={{ flex: 1, minWidth: 0, overflow: 'hidden', display: { xs: 'none', md: 'block' } }}>{diagram(false)}</Box>
        <Typography variant="body2" noWrap sx={{ flex: 1, display: { xs: 'block', md: 'none' } }}>{currentStep.label} · {steps.filter(step => step.done).length}/{steps.length}</Typography>
        <Button size="small" onClick={() => setExpanded(true)} startIcon={<OpenInFullIcon />} sx={{ flexShrink: 0 }}>{t('paFlow.expand')}</Button>
      </Stack>
    </Paper>
    <Dialog open={expanded} onClose={() => setExpanded(false)} fullWidth maxWidth="lg" aria-labelledby="workflow-title">
      <DialogTitle id="workflow-title"><Stack direction="row" sx={{ alignItems: 'center', justifyContent: 'space-between' }}>{t('paFlow.title')}<IconButton aria-label={t('common.close')} onClick={() => setExpanded(false)}><CloseIcon /></IconButton></Stack></DialogTitle>
      <DialogContent><Typography color="text.secondary" sx={{ mb: 2 }}>{t(supplied ? 'paFlow.existingHelp' : 'paFlow.generatedHelp')}</Typography>
        <Box role="region" tabIndex={0} aria-label={t('paFlow.title')} sx={{ overflowX: 'auto' }}>{diagram(true)}</Box>
        <Typography variant="body2" color="text.secondary" sx={{ mt: 2 }}>{(state.datasetId ?? state.inputName) || t('paFlow.begin')}</Typography>
      </DialogContent>
    </Dialog>
  </>
}
