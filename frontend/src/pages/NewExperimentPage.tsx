import Accordion from '@mui/material/Accordion'
import AccordionDetails from '@mui/material/AccordionDetails'
import AccordionSummary from '@mui/material/AccordionSummary'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import Box from '@mui/material/Box'
import Chip from '@mui/material/Chip'
import ArrowForwardIcon from '@mui/icons-material/ArrowForward'
import ArrowBackIcon from '@mui/icons-material/ArrowBack'
import Alert from '@mui/material/Alert'
import AlertTitle from '@mui/material/AlertTitle'
import Button from '@mui/material/Button'
import Grid from '@mui/material/Grid'
import MenuItem from '@mui/material/MenuItem'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { useEffect, useRef, useState, type FormEvent } from 'react'
import { Link as RouterLink, useNavigate, useSearchParams } from 'react-router'
import { versionNames } from '@/api/datasets'
import { useCapabilities, useDatasets, useMetricProfiles, useModels, useRecipes, useRunConfig, useRuns, useSubmitRun, validateConfig } from '@/api/hooks'
import { offeredProfiles } from '@/api/profiles'
import type { ConfigIssue, Device, ExperimentConfigInput, ModelInfo, RecipeInfo, ValidationReport } from '@/api/types'
import { datasetLabel, message, phaseLabel, t } from '@/i18n'
import { WorkflowSteps } from '@/components/WorkflowSteps'
import { ExperimentTasks, isExperimentTask, taskLabel, type ExperimentTask } from '@/components/ExperimentTasks'
import { JsonConfigDialog, importableConfig } from '@/components/JsonConfigDialog'
import { ErrorState, LoadingState } from '@/components/StateBlock'

interface FormState {
  recipeId: string
  datasetId: string
  dataVersion: string
  paRunId: string
  device: string
  seed: string
  name: string
  epochs: string
  batchSize: string
  learningRate: string
  frameLength: string
  frameStride: string
  params: Record<string, string>
  profileId: string
  sourceRunId: string
  variantKey: string
  numThreads: string
  chunkSamples: string
  optimizer: string
  loss: string
  previewMode: string
  previewBatches: string
}

function num(s: string, fallback: number): number {
  const v = Number(s)
  return s.trim() === '' || !Number.isFinite(v) ? fallback : v
}

type ParamSpec = ModelInfo['params'][number]

/** Coerces a text edit by the registry's declared type; empty or unparsable edits keep the recipe value. */
function coerceParam(spec: ParamSpec, raw: string): number | boolean | string | undefined {
  const s = raw.trim()
  if (s === '') return undefined
  if (spec.type === 'bool') return s === 'true'
  if (spec.type === 'str') return s
  const v = Number(s)
  // Preserve fractional edits so the server can reject invalid integer parameters.
  return Number.isFinite(v) ? v : undefined
}

/** Builds the ExperimentConfig sent to validate/submit: recipe defaults + user edits. */
function buildConfig(recipe: RecipeInfo, f: FormState, specs: ParamSpec[]): ExperimentConfigInput {
  const training = {
    ...recipe.training,
    epochs: num(f.epochs, recipe.training.epochs),
    batch_size: num(f.batchSize, recipe.training.batch_size),
    learning_rate: num(f.learningRate, recipe.training.learning_rate),
    frame_length: num(f.frameLength, recipe.training.frame_length),
    frame_stride: num(f.frameStride, recipe.training.frame_stride),
    seed: num(f.seed, recipe.training.seed),
    optimizer: (f.optimizer || recipe.training.optimizer) as RecipeInfo['training']['optimizer'],
    loss: (f.loss || recipe.training.loss) as RecipeInfo['training']['loss'],
  }
  const parameters = { ...recipe.model.parameters }
  for (const spec of specs) {
    const v = coerceParam(spec, f.params[spec.name] ?? '')
    if (v !== undefined) parameters[spec.name] = v
  }
  const config: ExperimentConfigInput = {
    task: recipe.task,
    recipe_id: recipe.recipe_id,
    name: f.name.trim() || null,
    dataset: f.dataVersion ? { id: f.datasetId, preprocessing_version: f.dataVersion } : { id: f.datasetId },
    model: { key: recipe.model.key, parameters },
    training,
    execution: { device: f.device as Device, ...(f.numThreads.trim() ? { num_threads: num(f.numThreads, 0) } : {}),
      preview_every_batches: f.previewMode === 'batch' ? num(f.previewBatches, 100) : null },
  }
  if (recipe.task === 'train_dpd' && f.paRunId) config.pa_reference = { run_id: f.paRunId }
  if (f.profileId) config.evaluation = { profile_id: f.profileId }
  return config
}

const FIELD_MAP: Record<string, keyof FormState> = {
  'training.epochs': 'epochs',
  'training.batch_size': 'batchSize',
  'training.learning_rate': 'learningRate',
  'training.frame_length': 'frameLength',
  'training.frame_stride': 'frameStride',
  'training.seed': 'seed',
  'dataset.id': 'datasetId',
  'dataset.preprocessing_version': 'dataVersion',
  'pa_reference': 'paRunId',
  'pa_reference.run_id': 'paRunId',
  'execution.device': 'device',
  'model.key': 'recipeId',
  'evaluation.profile_id': 'profileId',
  'execution.num_threads': 'numThreads',
  'evaluation.chunk_samples': 'chunkSamples',
  'training.optimizer': 'optimizer',
  'training.loss': 'loss',
  'execution.preview_every_batches': 'previewBatches',
}

export function NewExperimentPage() {
  const [params] = useSearchParams()
  const requested = params.get('task')
  const task = isExperimentTask(requested) ? requested : 'train_pa'
  return <ExperimentForm key={task} task={task} />
}

function ExperimentForm({ task }: { task: ExperimentTask }) {
  const navigate = useNavigate()
  const recipes = useRecipes()
  const datasets = useDatasets()
  const models = useModels()
  const caps = useCapabilities()
  const metricProfiles = useMetricProfiles()
  const succeeded = useRuns('succeeded')
  const submit = useSubmitRun()
  const [params, setParams] = useSearchParams()
  const fromRunId = params.get('from') ?? ''
  const fromRun = useRunConfig(fromRunId, fromRunId !== '')
  const [importedFile, setImportedFile] = useState<{ config: ExperimentConfigInput; source: string } | null>(null)
  const [jsonOpen, setJsonOpen] = useState(false)
  const idempotencyKey = useRef(crypto.randomUUID())
  const [edits, setEdits] = useState<FormState>({ recipeId: '', datasetId: '', dataVersion: '', paRunId: '', device: '', seed: '', name: '', epochs: '', batchSize: '', learningRate: '', frameLength: '', frameStride: '', params: {}, profileId: '', sourceRunId: '', variantKey: '', numThreads: '', chunkSamples: '', optimizer: '', loss: '', previewMode: 'epoch', previewBatches: '' })
  // The report is stored with the config it validated, so "checking" is derived, not duplicated state.
  const [validated, setValidated] = useState<{ configJson: string; attempt: number; report: ValidationReport | null; error: unknown } | null>(null)
  const [validationAttempt, setValidationAttempt] = useState(0)
  const [step, setStep] = useState(0)
  const [confirmed, setConfirmed] = useState<Record<number, string>>({})

  const testing = task === 'evaluate_pa' || task === 'run_dpd'
  const taskRecipes = (recipes.data ?? []).filter((entry) => entry.task === task)
  const sourceRunId = edits.sourceRunId || params.get('modelRun') || ''
  const source = useRunConfig(sourceRunId, testing && !!sourceRunId)
  const sourceRuns = (succeeded.data ?? []).filter((run) => run.task === (task === 'evaluate_pa' ? 'train_pa' : 'train_dpd'))
  const sourceRun = sourceRuns.find((run) => run.run_id === sourceRunId)
  // Every training starting point comes from the shared service. Testing binds
  // the checkpoint's weights and training settings through that same service.
  const datasetId = edits.datasetId || params.get('dataset') || source.data?.dataset.id || sourceRun?.dataset_id || datasets.data?.[0]?.dataset_id || ''
  const dataset = datasets.data?.find((d) => d.dataset_id === datasetId)
  const versions = dataset ? versionNames(dataset) : ['raw-v1']
  // '' means "server default (raw-v1)"; a version that no longer exists for the chosen dataset falls back too.
  const requestedVersion = edits.dataVersion || (datasetId === params.get('dataset') ? params.get('version') : '') || (testing && source.data?.dataset.id === datasetId ? source.data.dataset.preprocessing_version : '') || ''
  // Capability discovery is asynchronous. Derive the initial device until the
  // user chooses one; later refetches must never overwrite that explicit choice.
  const defaultDevice = caps.data?.devices.some((d) => d.device === 'cuda' && d.detected) ? 'cuda' : 'cpu'
  const form: FormState = { ...edits, device: edits.device || defaultDevice, recipeId: edits.recipeId || taskRecipes[0]?.recipe_id || '', datasetId, dataVersion: versions.includes(requestedVersion) ? requestedVersion : '' }
  const recipe = taskRecipes.find((r) => r.recipe_id === form.recipeId) ?? null

  const paRuns = (succeeded.data ?? []).filter((r) => r.task === 'train_pa' && r.dataset_id === datasetId)
  const modelKey = testing ? (form.variantKey || source.data?.model.key) : recipe?.model.key
  const model = models.data?.find((m) => m.key === modelKey)
  const leastSquares = model?.training_method === 'least_squares'
  const streaming = model?.execution_semantics === 'streaming_stateful'
  const specs = model?.params ?? []
  // An imported configuration (file or `?from=<run>`) is derived during render, never copied into form state.
  const imported = importedFile ?? (fromRunId && fromRun.data ? { config: importableConfig(fromRun.data), source: fromRunId } : null)
  const testConfig: ExperimentConfigInput | null = testing && source.data && sourceRunId && form.datasetId ? {
    task, name: form.name.trim() || null,
    dataset: { id: form.datasetId, ...(form.dataVersion ? { preprocessing_version: form.dataVersion } : {}) },
    model: { ...source.data.model, key: modelKey! },
    execution: { device: form.device as Device, ...(form.numThreads.trim() ? { num_threads: num(form.numThreads, 0) } : {}) },
    evaluation: { ...(form.profileId ? { profile_id: form.profileId } : {}), ...(streaming && form.chunkSamples.trim() ? { chunk_samples: num(form.chunkSamples, 0) } : {}) },
    ...(task === 'evaluate_pa' ? { pa_reference: { run_id: sourceRunId } } : { dpd_reference: { run_id: sourceRunId }, ...(form.paRunId ? { pa_reference: { run_id: form.paRunId } } : {}) }),
  } : null
  const config = imported ? { ...imported.config, name: form.name.trim() || imported.config.name || null } : !fromRunId ? (testing ? testConfig : recipe && form.datasetId ? buildConfig(recipe, form, specs) : null) : null
  const configJson = config ? JSON.stringify(config) : ''
  const currentValidation = validated?.configJson === configJson && validated.attempt === validationAttempt
  const report = currentValidation ? validated.report : null
  const validationError = currentValidation ? validated.error : null
  const checking = !!configJson && !currentValidation

  // Server-side validation, debounced; the server is the only source of rules.
  useEffect(() => {
    if (!configJson) return
    let active = true
    const timer = window.setTimeout(() => {
      validateConfig(JSON.parse(configJson))
        .then((r) => active && setValidated({ configJson, attempt: validationAttempt, report: r, error: null }))
        .catch((error: unknown) => active && setValidated({ configJson, attempt: validationAttempt, report: null, error }))
    }, 300)
    return () => {
      active = false
      window.clearTimeout(timer)
    }
  }, [configJson, validationAttempt])

  if (fromRunId && !importedFile && fromRun.isPending) return <LoadingState />
  if (fromRunId && !importedFile && fromRun.isError) return <ErrorState error={fromRun.error} onRetry={() => void fromRun.refetch()} />
  if (recipes.isPending || datasets.isPending || models.isPending) return <LoadingState />
  if (recipes.isError) return <ErrorState error={recipes.error} onRetry={() => void recipes.refetch()} />
  if (datasets.isError) return <ErrorState error={datasets.error} onRetry={() => void datasets.refetch()} />
  if (models.isError) return <ErrorState error={models.error} onRetry={() => void models.refetch()} />

  const issuesFor = (key: keyof FormState): ConfigIssue[] => (report?.errors ?? []).filter((e) => FIELD_MAP[e.field] === key)
  const errorText = (key: keyof FormState) => issuesFor(key).map((e) => (e.hint ? `${message(e.message)} (${message(e.hint)})` : message(e.message))).join(' ')
  const paramIssues = (name: string) => (report?.errors ?? []).filter((e) => e.field === `model.parameters.${name}`)
  const set = (key: keyof FormState) => (e: { target: { value: string } }) => setEdits((f) => ({ ...f, [key]: e.target.value }))
  const setParam = (name: string) => (e: { target: { value: string } }) => setEdits((f) => ({ ...f, params: { ...f.params, [name]: e.target.value } }))
  const untested = model && !model.devices_tested.includes(form.device)
  const discardImport = () => {
    setImportedFile(null)
    setConfirmed({}); setStep(0)
    if (fromRunId) setParams({ task })
  }
  const selectStartingPoint = (entry: RecipeInfo) => setEdits((old) => ({ ...old, recipeId: entry.recipe_id, params: {}, epochs: '', batchSize: '', learningRate: '', frameLength: '', frameStride: '', seed: '', optimizer: '', loss: '' }))
  const purposeLabel = (purpose: string) => t(purpose === 'smoke' ? 'tasks.quick' : purpose === 'research' ? 'tasks.full' : 'tasks.fit')
  const dataSignature = JSON.stringify(config && { dataset: config.dataset, recipe: config.recipe_id, task: config.task, model: testing && !imported ? source.data?.model.key : config.model.key, pa: config.pa_reference, dpd: config.dpd_reference })
  const signatures = [dataSignature, JSON.stringify(config && { dataSignature, training: config.training, model: config.model, execution: config.execution, evaluation: config.evaluation })]
  const dataErrors = (report?.errors ?? []).filter((e) => /^(dataset|pa_reference|dpd_reference|recipe_id|task|model\.key)(\.|$)/.test(e.field))
  const completed = [confirmed[0] === signatures[0] && !!report && dataErrors.length === 0 && !checking, confirmed[1] === signatures[1] && !!report?.ok && !checking]
  const canOpen = (next: number) => next === 0 || (next === 1 ? completed[0]! : completed.every(Boolean))
  const dataReady = !!imported || (!!form.datasetId && (testing ? !!source.data : task !== 'train_dpd' || !!form.paRunId))
  const canNext = dataReady && !!config && !!report && !checking && (step === 0 ? dataErrors.length === 0 : completed[0] && report.ok)
  const canSubmit = step === 2 && completed.every(Boolean) && !!config && !!report?.ok && !checking && !submit.isPending
  const next = () => { if (canNext) { setConfirmed((old) => ({ ...old, [step]: signatures[step]! })); setStep(step + 1) } }

  const onSubmit = (e: FormEvent) => {
    e.preventDefault()
    if (!config || !canSubmit) return
    submit.mutate({ config, name: form.name.trim() || undefined, idempotency_key: idempotencyKey.current }, { onSuccess: (run) => navigate(`/runs/${encodeURIComponent(run.run_id)}`) })
  }

  return (
    <Stack component="form" spacing={2} onSubmit={onSubmit} noValidate aria-labelledby="form-title" sx={{ maxWidth: 1120, mx: 'auto' }}>
      <Stack direction="row" sx={{ alignItems: 'center', justifyContent: 'space-between', gap: 1, flexWrap: 'wrap' }}>
        <Typography variant="h1" id="form-title">{taskLabel(config?.task ?? task)}</Typography>
        <Button variant="outlined" size="small" onClick={() => setJsonOpen(true)} disabled={submit.isPending}>{t('json.open')}</Button>
      </Stack>
      <ExperimentTasks active={config?.task ?? task} dataset={form.datasetId} version={form.dataVersion} compact />
      <WorkflowSteps active={step} labels={[t('workflow.data'), t(testing ? 'workflow.testConfigure' : 'workflow.configure'), t('workflow.review')]} completed={completed} onChange={setStep} canOpen={canOpen} />
      {recipe?.purpose === 'smoke' && !imported && <Alert severity="info">{t('form.smokeBanner')}</Alert>}
      {jsonOpen && <JsonConfigDialog config={config} onClose={() => setJsonOpen(false)} onApply={(edited, label) => { setImportedFile({ config: edited, source: label }); setJsonOpen(false); setStep(0); setConfirmed({}) }} />}
      {imported && (
        <Alert severity="info" action={<Button color="inherit" size="small" onClick={discardImport}>{t('form.import.discard')}</Button>} data-testid="imported-banner">
          {t('form.imported', { source: imported.source })}
          <Typography variant="body2" sx={{ mt: 1 }}>{taskLabel(config!.task)} · {config?.dataset.id} · {config?.model.key}</Typography>
        </Alert>
      )}
      <Paper role="tabpanel" id="workflow-panel-0" aria-labelledby="workflow-step-0" sx={{ p: 2.5, display: step === 0 && !imported ? 'block' : 'none' }}>
        <Grid container spacing={2}>
          {!imported && <>
          {testing && <Grid size={{ xs: 12 }}>
            <Typography color="text.secondary" sx={{ mb: 2 }}>{t(task === 'evaluate_pa' ? 'tasks.evaluate_pa.help' : 'tasks.run_dpd.help')}</Typography>
            <TextField select fullWidth required label={t('testing.model')} value={sourceRunId} onChange={(event) => {
              setEdits((old) => ({ ...old, sourceRunId: event.target.value, datasetId: '', dataVersion: '', variantKey: '', paRunId: '', chunkSamples: '' }))
              setParams({ task })
            }} helperText={sourceRuns.length ? t('testing.model.help') : t('testing.model.none')}>
              {sourceRuns.map((run) => <MenuItem key={run.run_id} value={run.run_id}>{run.name || run.run_id} · {run.model_key}</MenuItem>)}
            </TextField>
            {succeeded.isError && <ErrorState error={succeeded.error} onRetry={() => void succeeded.refetch()} />}
            {sourceRunId && source.isPending && <LoadingState />}
            {sourceRunId && source.isError && <ErrorState error={source.error} onRetry={() => void source.refetch()} />}
            {!succeeded.isPending && sourceRuns.length === 0 && <Button component={RouterLink} to={`/experiments/new?task=${task === 'evaluate_pa' ? 'train_pa' : 'train_dpd'}`} sx={{ mt: 1 }}>{taskLabel(task === 'evaluate_pa' ? 'train_pa' : 'train_dpd')}</Button>}
          </Grid>}
          <Grid size={{ xs: 12, md: 6 }}>
            <TextField select fullWidth required label={t('form.dataset')} value={form.datasetId} onChange={set('datasetId')} error={issuesFor('datasetId').length > 0} helperText={errorText('datasetId') || (datasets.data.length === 0 ? t('form.dataset.none') : ' ')}>
              {datasets.data.filter((d) => task !== 'run_dpd' || !source.data || d.dataset_id === source.data.dataset.id).map((d) => (
                <MenuItem key={d.dataset_id} value={d.dataset_id}>
                  {datasetLabel(d)}
                </MenuItem>
              ))}
            </TextField>
          </Grid>
          {!testing && <>
            <Grid size={{ xs: 12, md: 6 }}>
              <TextField select fullWidth required label={t('tasks.model')} value={recipe?.model.key ?? ''} onChange={(event) => { const entry = taskRecipes.find((r) => r.model.key === event.target.value && r.purpose === recipe?.purpose) ?? taskRecipes.find((r) => r.model.key === event.target.value); if (entry) selectStartingPoint(entry) }} helperText={errorText('recipeId') || t('tasks.model.help')} error={issuesFor('recipeId').length > 0}>
                {[...new Set(taskRecipes.map((entry) => entry.model.key))].map((key) => <MenuItem key={key} value={key}>{message(models.data.find((entry) => entry.key === key)?.display_name ?? key)}</MenuItem>)}
              </TextField>
            </Grid>
            <Grid size={{ xs: 12, md: 6 }}>
              <TextField select fullWidth label={t('tasks.startingSettings')} value={form.recipeId} onChange={(event) => { const entry = taskRecipes.find((r) => r.recipe_id === event.target.value); if (entry) selectStartingPoint(entry) }} helperText={t('tasks.startingSettings.help')}>
                {taskRecipes.filter((entry) => entry.model.key === recipe?.model.key).map((entry) => <MenuItem key={entry.recipe_id} value={entry.recipe_id}>{purposeLabel(entry.purpose)}</MenuItem>)}
              </TextField>
            </Grid>
          </>}
          {versions.length > 1 && (
            <Grid size={{ xs: 12, md: 6 }}>
              <TextField select fullWidth label={t('form.dataVersion')} value={form.dataVersion || 'raw-v1'} onChange={(e) => setEdits((f) => ({ ...f, dataVersion: e.target.value }))} error={issuesFor('dataVersion').length > 0} helperText={errorText('dataVersion') || ' '}>
                {versions.map((v) => (
                  <MenuItem key={v} value={v}>
                    {v}
                  </MenuItem>
                ))}
              </TextField>
            </Grid>
          )}
          {(task === 'train_dpd' || task === 'run_dpd') && (
            <Grid size={{ xs: 12, md: 6 }}>
              <TextField select fullWidth required={task === 'train_dpd'} label={t(task === 'run_dpd' ? 'testing.surrogate' : 'form.paRun')} value={form.paRunId} onChange={set('paRunId')} slotProps={task === 'run_dpd' ? { select: { displayEmpty: true }, inputLabel: { shrink: true } } : undefined} error={issuesFor('paRunId').length > 0} helperText={errorText('paRunId') || (task === 'run_dpd' ? t('testing.surrogate.help') : paRuns.length === 0 ? t('form.paRun.none') : t('form.paRun.help'))}>
                {task === 'run_dpd' && <MenuItem value="">{t('testing.trainingSurrogate')}</MenuItem>}
                {paRuns.map((r) => (
                  <MenuItem key={r.run_id} value={r.run_id}>
                    {r.name || r.run_id}
                  </MenuItem>
                ))}
              </TextField>
              {task === 'train_dpd' && paRuns.length === 0 && <Button component={RouterLink} to={`/experiments/new?task=train_pa&dataset=${encodeURIComponent(form.datasetId)}`}>{taskLabel('train_pa')}</Button>}
            </Grid>
          )}
          </>}
        </Grid>
      </Paper>
      <Box role="tabpanel" id="workflow-panel-1" aria-labelledby="workflow-step-1" sx={{ display: step === 1 ? 'block' : 'none' }}>
      <Paper sx={{ p: 2.5, mb: 1.5 }}><Grid container spacing={2}>
          {!imported && <>
          <Grid size={{ xs: 12, md: testing || leastSquares ? 6 : 4 }}>
            <TextField select fullWidth label={t('form.device')} value={form.device} onChange={set('device')} error={issuesFor('device').length > 0} helperText={errorText('device') || (untested ? t('form.device.untested', { device: form.device }) : ' ')}>
              {(caps.data?.devices ?? [{ device: 'cpu', detected: true, count: 1, tested_models: [] }]).map((d) => (
                <MenuItem key={d.device} value={d.device} disabled={!d.detected}>
                  {d.device}
                  {d.name ? ` · ${d.name}` : ''}
                  {!d.detected ? ` (${t('settings.devices.notDetected')})` : ''}
                </MenuItem>
              ))}
            </TextField>
          </Grid>
          {!testing && !leastSquares && <Grid size={{ xs: 6, md: 4 }}>
            <TextField fullWidth type="number" label={t('form.seed')} value={form.seed} onChange={set('seed')} placeholder={String(recipe?.training.seed ?? 0)} slotProps={{ inputLabel: { shrink: true } }} error={issuesFor('seed').length > 0} helperText={errorText('seed') || ' '} />
          </Grid>}
          </>}
          <Grid size={{ xs: 12, md: testing || leastSquares || imported ? 6 : 4 }}>
            <TextField fullWidth label={t('form.name')} value={form.name} onChange={set('name')} helperText=" " />
          </Grid>
      </Grid></Paper>
      {!imported && !testing && <Paper sx={{ p: 2.5 }}>
          <Typography variant="h2" sx={{ mb: 2 }}>{t(leastSquares ? 'tasks.fitParameters' : 'tasks.hyperparameters')}</Typography>
          <Grid container spacing={2}>
            {!leastSquares && (
              [
                ['epochs', 'form.epochs', recipe?.training.epochs],
                ['batchSize', 'form.batchSize', recipe?.training.batch_size],
                ['learningRate', 'form.learningRate', recipe?.training.learning_rate],
                ['frameLength', 'form.frameLength', recipe?.training.frame_length],
                ['frameStride', 'form.frameStride', recipe?.training.frame_stride],
              ] as const
            ).map(([key, label, placeholder]) => (
              <Grid key={key} size={{ xs: 6, md: 4 }}>
                <TextField fullWidth type="number" label={t(label)} value={form[key]} onChange={set(key)} placeholder={placeholder === undefined ? '' : String(placeholder)} error={issuesFor(key).length > 0} helperText={errorText(key) || ' '} slotProps={{ inputLabel: { shrink: true }, htmlInput: { step: key === 'learningRate' ? 'any' : 1 } }} />
              </Grid>
            ))}
            {!leastSquares && <>
              <Grid size={{ xs: 6, md: 4 }}><TextField select fullWidth label={t('tasks.optimizer')} value={form.optimizer || recipe?.training.optimizer || ''} onChange={set('optimizer')}>{(['adamw', 'adam', 'sgd', 'rmsprop'] as const).map((value) => <MenuItem key={value} value={value}>{value}</MenuItem>)}</TextField></Grid>
              <Grid size={{ xs: 6, md: 4 }}><TextField select fullWidth label={t('tasks.loss')} value={form.loss || recipe?.training.loss || ''} onChange={set('loss')}><MenuItem value="l2">L2</MenuItem><MenuItem value="l1">L1</MenuItem></TextField></Grid>
            </>}
            {specs.length > 0 && (
              <Grid size={{ xs: 12 }}>
                <Typography variant="h3" component="h3">
                  {t('form.params')}
                </Typography>
              </Grid>
            )}
            {specs.map((spec) => {
              const current = recipe?.model.parameters?.[spec.name] ?? spec.default
              const issues = paramIssues(spec.name)
              const help = issues.map((e) => (e.hint ? `${message(e.message)} (${message(e.hint)})` : message(e.message))).join(' ') || message(spec.description)
              return (
                <Grid key={spec.name} size={{ xs: 6, md: 4 }}>
                  {spec.type === 'bool' || spec.choices ? (
                    <TextField select fullWidth label={spec.name} value={form.params[spec.name] ?? String(current)} onChange={setParam(spec.name)} error={issues.length > 0} helperText={help}>
                      {(spec.choices ?? [true, false]).map((c) => (
                        <MenuItem key={String(c)} value={String(c)}>
                          {String(c)}
                        </MenuItem>
                      ))}
                    </TextField>
                  ) : (
                    <TextField fullWidth type={spec.type === 'str' ? 'text' : 'number'} label={spec.name} value={form.params[spec.name] ?? ''} onChange={setParam(spec.name)} placeholder={String(current)} error={issues.length > 0} helperText={help} slotProps={{ inputLabel: { shrink: true }, htmlInput: { step: spec.type === 'int' ? 1 : 'any', min: spec.minimum ?? undefined, max: spec.maximum ?? undefined } }} />
                  )}
                </Grid>
              )
            })}
          </Grid>
      </Paper>}
      {!imported && testing && source.data && <Paper sx={{ p: 2.5 }}>
        <Typography variant="h2" gutterBottom>{t('testing.settings')}</Typography>
        <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>{t('testing.frozen')}</Typography>
        <Grid container spacing={2}>
          <Grid size={{ xs: 12, md: 6 }}><TextField select fullWidth label={t('testing.execution')} value={form.variantKey || source.data.model.key} onChange={set('variantKey')} helperText={t('testing.execution.help')}>
            <MenuItem value={source.data.model.key}>{t('testing.sameExecution')}</MenuItem>
            {models.data.filter((entry) => entry.weights_from === source.data!.model.key && entry.execution_semantics === 'streaming_stateful').map((entry) => <MenuItem key={entry.key} value={entry.key}>{message(entry.display_name)}</MenuItem>)}
          </TextField></Grid>
          {streaming && <Grid size={{ xs: 12, md: 6 }}><TextField fullWidth type="number" label={t('testing.chunk')} value={form.chunkSamples} onChange={set('chunkSamples')} helperText={errorText('chunkSamples') || t('testing.chunk.help')} error={issuesFor('chunkSamples').length > 0} /></Grid>}
        </Grid>
        <Typography variant="body2" sx={{ mt: 1.5 }}>{t('tasks.model')}: {source.data.model.key} · {t('form.frameLength')}: {source.data.training?.frame_length ?? t('common.na')}</Typography>
      </Paper>}
      {!imported && <Accordion disableGutters sx={{ mt: 1.5 }}>
        <AccordionSummary expandIcon={<ExpandMoreIcon />} aria-controls="advanced-panel" id="advanced-header"><Typography>{t('form.advanced')}</Typography></AccordionSummary>
        <AccordionDetails id="advanced-panel">
          <Grid container spacing={2}>
            <Grid size={{ xs: 12, md: 6 }}>
              <TextField select fullWidth label={t('form.profile')} value={form.profileId || offeredProfiles(metricProfiles.data)[0]?.profile_id || ''} onChange={set('profileId')} error={issuesFor('profileId').length > 0} helperText={errorText('profileId') || t('form.profile.help')}>
                {offeredProfiles(metricProfiles.data).map((p) => (
                  <MenuItem key={p.profile_id} value={p.profile_id}>
                    {p.profile_id} v{p.version}
                    {p.frozen ? ` · ${t('results.detail.frozen')}` : ''}
                  </MenuItem>
                ))}
              </TextField>
            </Grid>
            <Grid size={{ xs: 12, md: 6 }}><TextField fullWidth type="number" label={t('tasks.threads')} value={form.numThreads} onChange={set('numThreads')} helperText={errorText('numThreads') || t('tasks.threads.help')} error={issuesFor('numThreads').length > 0} /></Grid>
            {!testing && !leastSquares && <>
              <Grid size={{ xs: 12, md: 6 }}>
                <TextField select fullWidth label={t('form.preview')} value={form.previewMode} onChange={set('previewMode')} helperText={t('form.preview.epochHelp')}>
                  <MenuItem value="epoch">{t('form.preview.epoch')}</MenuItem>
                  <MenuItem value="batch" sx={{ color: 'error.main' }}>{t('form.preview.batch')}</MenuItem>
                </TextField>
              </Grid>
              {form.previewMode === 'batch' && <>
                <Grid size={{ xs: 12, md: 6 }}><TextField fullWidth type="number" color="error" label={t('form.preview.batches')} value={form.previewBatches} onChange={set('previewBatches')} placeholder="100" sx={{ '& .MuiInputLabel-root': { color: 'error.main' }, '& .MuiOutlinedInput-notchedOutline': { borderColor: 'error.main' } }} slotProps={{ inputLabel: { shrink: true }, htmlInput: { min: 1, max: 1000000, step: 1 } }} helperText={errorText('previewBatches') || ' '} error={issuesFor('previewBatches').length > 0} /></Grid>
                <Grid size={{ xs: 12 }}><Alert severity="error">{t('form.preview.warning')}</Alert></Grid>
              </>}
            </>}
          </Grid>
        </AccordionDetails>
      </Accordion>}
      </Box>
      {step === 2 && <Paper role="tabpanel" id="workflow-panel-2" aria-labelledby="workflow-step-2" sx={{ p: 2.5 }}>
        <Stack direction="row" sx={{ justifyContent: 'space-between', mb: 2 }}><Typography variant="h2">{t('workflow.reviewTitle')}</Typography>{report?.ok && !checking && <Chip size="small" color="success" variant="outlined" label={t('form.valid')} />}</Stack>
        <Box component="dl" sx={{ display: 'grid', gridTemplateColumns: 'max-content 1fr', columnGap: 3, rowGap: 1, m: 0, '& dt': { color: 'text.secondary' }, '& dd': { m: 0, overflowWrap: 'anywhere' } }}>
          <dt>{t('form.dataset')}</dt><dd>{config?.dataset.id} · {config?.dataset.preprocessing_version || 'raw-v1'}</dd>
          <dt>{t('tasks.task')}</dt><dd>{taskLabel(config?.task ?? task)}</dd>
          <dt>{t('tasks.model')}</dt><dd>{message(models.data.find((entry) => entry.key === config?.model.key)?.display_name ?? config?.model.key)}</dd>
          <dt>{t('form.device')}</dt><dd>{config?.execution?.device}</dd>
          {(config?.task === 'train_pa' || config?.task === 'train_dpd') && <>
            <dt>{t('form.epochs')}</dt><dd>{config?.training?.epochs}</dd>
            <dt>{t('form.batchSize')}</dt><dd>{config?.training?.batch_size}</dd>
            <dt>{t('form.learningRate')}</dt><dd>{config?.training?.learning_rate}</dd>
            <dt>{t('form.preview')}</dt><dd>{config?.execution?.preview_every_batches ? t('live.cadence.batch', { batches: config.execution.preview_every_batches }) : t('form.preview.epoch')}</dd>
          </>}
          {(config?.task === 'evaluate_pa' || config?.task === 'run_dpd') && <><dt>{t('testing.model')}</dt><dd>{config?.pa_reference && config.task === 'evaluate_pa' ? config.pa_reference.run_id : config?.dpd_reference?.run_id}</dd><dt>{t('testing.partition')}</dt><dd>{phaseLabel('test')}</dd></>}
        </Box>
      </Paper>}
      {validationError != null && <ErrorState error={validationError} onRetry={() => {
        setValidated(null)
        setValidationAttempt((attempt) => attempt + 1)
      }} />}
      {report && !report.ok && (
        <Alert severity="error" role="alert">
          <AlertTitle>{t('form.errors')}</AlertTitle>
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            {report.errors.map((e, i) => (
              <li key={`${e.field}-${i}`}>
                <code>{e.field}</code>: {message(e.message)}
                {e.hint ? ` — ${message(e.hint)}` : ''}
              </li>
            ))}
          </ul>
        </Alert>
      )}
      {report?.ok && report.warnings.length > 0 && (
        <Alert severity="warning">
          <AlertTitle>{t('form.warnings')}</AlertTitle>
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            {report.warnings.map((w, i) => (
              <li key={`${w.field}-${i}`}>
                <code>{w.field}</code>: {message(w.message)}
              </li>
            ))}
          </ul>
        </Alert>
      )}
      {submit.isError && <ErrorState error={submit.error} />}
      <Stack sx={{ alignItems: 'center' }} direction="row" spacing={1}>
        {step > 0 && <Button startIcon={<ArrowBackIcon />} onClick={() => setStep(step - 1)}>{t('workflow.back')}</Button>}
        {step < 2 && <Button variant="contained" endIcon={<ArrowForwardIcon />} onClick={next} disabled={!canNext}>{t('workflow.next')}</Button>}
        <Button type="submit" variant="contained" disabled={!canSubmit} sx={{ display: step === 2 ? 'inline-flex' : 'none' }}>
          {submit.isPending ? t('form.submitting') : t('form.submit')}
        </Button>
        <Button component={RouterLink} to="/experiments" variant="text">
          {t('form.cancel')}
        </Button>
        <Typography variant="body2" color="text.secondary" role="status">
          {checking ? t('form.validating') : report?.ok ? t('form.valid') : ''}
        </Typography>
      </Stack>
    </Stack>
  )
}
