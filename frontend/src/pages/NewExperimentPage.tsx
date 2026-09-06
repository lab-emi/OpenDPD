import Accordion from '@mui/material/Accordion'
import AccordionDetails from '@mui/material/AccordionDetails'
import AccordionSummary from '@mui/material/AccordionSummary'
import ExpandMoreIcon from '@mui/icons-material/ExpandMore'
import Alert from '@mui/material/Alert'
import AlertTitle from '@mui/material/AlertTitle'
import Button from '@mui/material/Button'
import Grid from '@mui/material/Grid'
import MenuItem from '@mui/material/MenuItem'
import Paper from '@mui/material/Paper'
import Stack from '@mui/material/Stack'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { useEffect, useMemo, useRef, useState, type FormEvent } from 'react'
import { Link as RouterLink, useNavigate, useSearchParams } from 'react-router'
import { versionNames } from '@/api/datasets'
import { useCapabilities, useDatasets, useMetricProfiles, useModels, useRecipes, useRunConfig, useRuns, useSubmitRun, validateConfig } from '@/api/hooks'
import { offeredProfiles } from '@/api/profiles'
import type { ConfigIssue, Device, ExperimentConfigInput, ModelInfo, RecipeInfo, ValidationReport } from '@/api/types'
import { t } from '@/i18n'
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
  return Number.isFinite(v) ? (spec.type === 'int' ? Math.trunc(v) : v) : undefined
}

/** Removes the server-side resolution block so an exported (resolved) configuration can be resubmitted. */
function importableConfig(data: unknown): ExperimentConfigInput {
  if (data === null || typeof data !== 'object' || Array.isArray(data)) throw new Error('expected a JSON object')
  const { resolution: _resolution, ...rest } = data as Record<string, unknown>
  const cfg = rest as ExperimentConfigInput
  if (typeof cfg.task !== 'string' || typeof cfg.dataset?.id !== 'string' || typeof cfg.model?.key !== 'string') throw new Error('task, dataset.id and model.key are required')
  return cfg
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
    execution: { device: f.device as Device },
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
}

export function NewExperimentPage() {
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
  const [importError, setImportError] = useState<string | null>(null)
  const idempotencyKey = useRef(crypto.randomUUID())
  const [edits, setEdits] = useState<FormState>({ recipeId: '', datasetId: '', dataVersion: '', paRunId: '', device: 'cpu', seed: '', name: '', epochs: '', batchSize: '', learningRate: '', frameLength: '', frameStride: '', params: {}, profileId: '' })
  // The report is stored with the config it validated, so "checking" is derived, not duplicated state.
  const [validated, setValidated] = useState<{ configJson: string; report: ValidationReport | null } | null>(null)

  // Defaults (first recipe / first dataset) are derived during render, never copied into state.
  const datasetId = edits.datasetId || datasets.data?.[0]?.dataset_id || ''
  const dataset = datasets.data?.find((d) => d.dataset_id === datasetId)
  const versions = dataset ? versionNames(dataset) : ['raw-v1']
  // '' means "server default (raw-v1)"; a version that no longer exists for the chosen dataset falls back too.
  const form: FormState = { ...edits, recipeId: edits.recipeId || recipes.data?.[0]?.recipe_id || '', datasetId, dataVersion: versions.includes(edits.dataVersion) ? edits.dataVersion : '' }
  const recipe = recipes.data?.find((r) => r.recipe_id === form.recipeId) ?? null

  const paRuns = useMemo(() => (succeeded.data ?? []).filter((r) => r.task === 'train_pa' && r.dataset_id === form.datasetId), [succeeded.data, form.datasetId])
  const model = models.data?.find((m) => m.key === recipe?.model.key)
  const specs = model?.params ?? []
  // An imported configuration (file or `?from=<run>`) is derived during render, never copied into form state.
  const imported = importedFile ?? (fromRunId && fromRun.data ? { config: importableConfig(fromRun.data), source: fromRunId } : null)
  const config = imported ? { ...imported.config, name: form.name.trim() || imported.config.name || null } : recipe && form.datasetId ? buildConfig(recipe, form, specs) : null
  const configJson = config ? JSON.stringify(config) : ''
  const report = validated && validated.configJson === configJson ? validated.report : null
  const checking = !!configJson && (validated === null || validated.configJson !== configJson)

  // Server-side validation, debounced; the server is the only source of rules.
  useEffect(() => {
    if (!configJson) return
    let active = true
    const timer = window.setTimeout(() => {
      validateConfig(JSON.parse(configJson))
        .then((r) => active && setValidated({ configJson, report: r }))
        .catch(() => active && setValidated({ configJson, report: null }))
    }, 300)
    return () => {
      active = false
      window.clearTimeout(timer)
    }
  }, [configJson])

  if (recipes.isPending || datasets.isPending || models.isPending) return <LoadingState />
  if (recipes.isError) return <ErrorState error={recipes.error} onRetry={() => void recipes.refetch()} />
  if (datasets.isError) return <ErrorState error={datasets.error} onRetry={() => void datasets.refetch()} />

  const issuesFor = (key: keyof FormState): ConfigIssue[] => (report?.errors ?? []).filter((e) => FIELD_MAP[e.field] === key)
  const errorText = (key: keyof FormState) => issuesFor(key).map((e) => (e.hint ? `${e.message} (${e.hint})` : e.message)).join(' ')
  const paramIssues = (name: string) => (report?.errors ?? []).filter((e) => e.field === `model.parameters.${name}`)
  const set = (key: keyof FormState) => (e: { target: { value: string } }) => setEdits((f) => ({ ...f, [key]: e.target.value }))
  const setParam = (name: string) => (e: { target: { value: string } }) => setEdits((f) => ({ ...f, params: { ...f.params, [name]: e.target.value } }))
  const untested = model && !model.devices_tested.includes(form.device)
  const onImportFile = (file: File) => {
    file
      .text()
      .then((text) => {
        setImportedFile({ config: importableConfig(JSON.parse(text) as unknown), source: file.name })
        setImportError(null)
      })
      .catch((err: unknown) => setImportError(err instanceof Error ? err.message : String(err)))
  }
  const discardImport = () => {
    setImportedFile(null)
    setImportError(null)
    if (fromRunId) setParams({})
  }
  const canSubmit = !!config && !!report?.ok && !checking && !submit.isPending

  const onSubmit = (e: FormEvent) => {
    e.preventDefault()
    if (!config || !canSubmit) return
    submit.mutate({ config, name: form.name.trim() || undefined, idempotency_key: idempotencyKey.current }, { onSuccess: (run) => navigate(`/runs/${encodeURIComponent(run.run_id)}`) })
  }

  return (
    <Stack component="form" spacing={2} onSubmit={onSubmit} noValidate aria-labelledby="form-title" sx={{ maxWidth: 880 }}>
      <Typography variant="h1" id="form-title">
        {t('form.title')}
      </Typography>
      {recipe?.purpose === 'smoke' && !imported && <Alert severity="info">{t('form.smokeBanner')}</Alert>}
      <Stack direction="row" spacing={1} sx={{ alignItems: 'center' }}>
        <Button component="label" variant="outlined" size="small">
          {t('form.import')}
          <input
            hidden
            type="file"
            accept="application/json,.json"
            data-testid="import-config"
            onChange={(e) => {
              const f = e.target.files?.[0]
              if (f) onImportFile(f)
              e.target.value = ''
            }}
          />
        </Button>
        <Typography variant="caption" color="text.secondary">
          {t('form.import.help')}
        </Typography>
      </Stack>
      {importError && <Alert severity="error">{t('form.import.invalid', { error: importError })}</Alert>}
      {imported && (
        <Alert severity="info" action={<Button color="inherit" size="small" onClick={discardImport}>{t('form.import.discard')}</Button>} data-testid="imported-banner">
          {t('form.imported', { source: imported.source })}
          <pre style={{ margin: '8px 0 0', maxHeight: 240, overflow: 'auto', fontSize: 12 }}>{JSON.stringify(imported.config, null, 2)}</pre>
        </Alert>
      )}
      <Paper sx={{ p: 2 }}>
        <Grid container spacing={2}>
          <Grid size={{ xs: 12, md: 6 }}>
            <TextField select fullWidth required label={t('form.recipe')} value={form.recipeId} onChange={set('recipeId')} helperText={recipe ? `${recipe.description} ${recipe.limits} (${recipe.expected_duration})` : t('form.recipe.help')} error={issuesFor('recipeId').length > 0}>
              {recipes.data.map((r) => (
                <MenuItem key={r.recipe_id} value={r.recipe_id}>
                  {r.title}
                </MenuItem>
              ))}
            </TextField>
          </Grid>
          <Grid size={{ xs: 12, md: 6 }}>
            <TextField select fullWidth required label={t('form.dataset')} value={form.datasetId} onChange={set('datasetId')} error={issuesFor('datasetId').length > 0} helperText={errorText('datasetId') || (datasets.data.length === 0 ? t('form.dataset.none') : ' ')}>
              {datasets.data.map((d) => (
                <MenuItem key={d.dataset_id} value={d.dataset_id}>
                  {d.display_name}
                </MenuItem>
              ))}
            </TextField>
          </Grid>
          {versions.length > 1 && (
            <Grid size={{ xs: 12, md: 6 }}>
              <TextField select fullWidth label={t('form.dataVersion')} value={form.dataVersion || 'raw-v1'} onChange={(e) => setEdits((f) => ({ ...f, dataVersion: e.target.value === 'raw-v1' ? '' : e.target.value }))} error={issuesFor('dataVersion').length > 0} helperText={errorText('dataVersion') || ' '}>
                {versions.map((v) => (
                  <MenuItem key={v} value={v}>
                    {v}
                  </MenuItem>
                ))}
              </TextField>
            </Grid>
          )}
          {recipe?.task === 'train_dpd' && (
            <Grid size={{ xs: 12, md: 6 }}>
              <TextField select fullWidth required label={t('form.paRun')} value={form.paRunId} onChange={set('paRunId')} error={issuesFor('paRunId').length > 0} helperText={errorText('paRunId') || (paRuns.length === 0 ? t('form.paRun.none') : t('form.paRun.help'))}>
                {paRuns.map((r) => (
                  <MenuItem key={r.run_id} value={r.run_id}>
                    {r.name || r.run_id}
                  </MenuItem>
                ))}
              </TextField>
            </Grid>
          )}
          <Grid size={{ xs: 12, md: 3 }}>
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
          <Grid size={{ xs: 6, md: 3 }}>
            <TextField fullWidth type="number" label={t('form.seed')} value={form.seed} onChange={set('seed')} placeholder={String(recipe?.training.seed ?? 0)} error={issuesFor('seed').length > 0} helperText={errorText('seed') || ' '} />
          </Grid>
          <Grid size={{ xs: 12, md: 6 }}>
            <TextField fullWidth label={t('form.name')} value={form.name} onChange={set('name')} helperText=" " />
          </Grid>
        </Grid>
      </Paper>
      <Accordion disableGutters>
        <AccordionSummary expandIcon={<ExpandMoreIcon />} aria-controls="advanced-panel" id="advanced-header">
          <Typography>{t('form.advanced')}</Typography>
        </AccordionSummary>
        <AccordionDetails id="advanced-panel">
          <Grid container spacing={2}>
            {(
              [
                ['epochs', 'form.epochs', recipe?.training.epochs],
                ['batchSize', 'form.batchSize', recipe?.training.batch_size],
                ['learningRate', 'form.learningRate', recipe?.training.learning_rate],
                ['frameLength', 'form.frameLength', recipe?.training.frame_length],
                ['frameStride', 'form.frameStride', recipe?.training.frame_stride],
              ] as const
            ).map(([key, label, placeholder]) => (
              <Grid key={key} size={{ xs: 6, md: 4 }}>
                <TextField fullWidth type="number" label={t(label)} value={form[key]} onChange={set(key)} placeholder={placeholder === undefined ? '' : String(placeholder)} error={issuesFor(key).length > 0} helperText={errorText(key) || ' '} slotProps={{ htmlInput: { step: key === 'learningRate' ? 'any' : 1 } }} />
              </Grid>
            ))}
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
              const help = issues.map((e) => (e.hint ? `${e.message} (${e.hint})` : e.message)).join(' ') || spec.description
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
                    <TextField fullWidth type={spec.type === 'str' ? 'text' : 'number'} label={spec.name} value={form.params[spec.name] ?? ''} onChange={setParam(spec.name)} placeholder={String(current)} error={issues.length > 0} helperText={help} slotProps={{ htmlInput: { step: spec.type === 'int' ? 1 : 'any', min: spec.minimum ?? undefined, max: spec.maximum ?? undefined } }} />
                  )}
                </Grid>
              )
            })}
            <Grid size={{ xs: 12, md: 4 }}>
              <TextField select fullWidth label={t('form.profile')} value={form.profileId || offeredProfiles(metricProfiles.data)[0]?.profile_id || ''} onChange={set('profileId')} error={issuesFor('profileId').length > 0} helperText={errorText('profileId') || t('form.profile.help')}>
                {offeredProfiles(metricProfiles.data).map((p) => (
                  <MenuItem key={p.profile_id} value={p.profile_id}>
                    {p.profile_id} v{p.version}
                    {p.frozen ? ` · ${t('results.detail.frozen')}` : ''}
                  </MenuItem>
                ))}
              </TextField>
            </Grid>
          </Grid>
        </AccordionDetails>
      </Accordion>
      {report && !report.ok && (
        <Alert severity="error" role="alert">
          <AlertTitle>{t('form.errors')}</AlertTitle>
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            {report.errors.map((e, i) => (
              <li key={`${e.field}-${i}`}>
                <code>{e.field}</code>: {e.message}
                {e.hint ? ` — ${e.hint}` : ''}
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
                <code>{w.field}</code>: {w.message}
              </li>
            ))}
          </ul>
        </Alert>
      )}
      {submit.isError && <ErrorState error={submit.error} />}
      <Stack sx={{ alignItems: 'center' }} direction="row" spacing={1}>
        <Button type="submit" variant="contained" disabled={!canSubmit}>
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
