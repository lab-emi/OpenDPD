/**
 * In-browser fake of the Studio API for L1 journeys: routes are intercepted
 * with page.route(); state lives in this object for one test. The packaged,
 * real-server journey is the S06 L2 test.
 */
import type { Page, Route } from '@playwright/test'
import { readFileSync } from 'node:fs'
import { fileURLToPath } from 'node:url'

function mock<T>(name: string): T {
  const path = fileURLToPath(new URL(`../mocks/${name}.json`, import.meta.url))
  return (JSON.parse(readFileSync(path, 'utf-8')) as { data: T }).data
}

type Json = Record<string, unknown>

export interface FakeState {
  datasets: Json[]
  runs: Json[]
  submitted: Json[]
  diagnostics: Record<string, Json>
}

const source = {
  kind: 'csv_import',
  path: 'capture.csv',
  columns: ['tx_i', 'tx_q', 'rx_q', 'rx_i'],
  n_rows: 20000,
  arrays: {},
  legacy_files: [],
  preview: [{ tx_i: 0.1, tx_q: -0.2, rx_q: 0.3, rx_i: 0.4 }],
  problems: [],
  suggested_mapping: { I_in: 'tx_i', Q_in: 'tx_q', I_out: 'rx_i', Q_out: 'rx_q' },
}
const report: Json = {
  report_id: 'doc-20260906-abcd1234',
  dataset_id: 'capture',
  doctor_version: 'dataset-doctor-v1',
  generated_at: '2026-09-06T08:00:00Z',
  evaluation_blocked: false,
  schema_version: 1,
  dataset_raw_sha256: null,
  items: [
    { code: 'time_misalignment', severity: 'warning', title: 'Time misalignment', message: 'Output lags input by 6.0 samples', evidence: { delay_samples: 6, confidence: 0.98 }, suggestion: 'Apply delay correction', confidence: 0.98, blocking: false },
    { code: 'linear_gain_phase', severity: 'info', title: 'Linear gain and phase', message: 'gain 2.9 dB, phase 25.1 deg', evidence: { gain_db: 2.9, phase_deg: 25.1 }, blocking: false },
  ],
}
const aligned = { code: 'alignment_ok', severity: 'info', title: 'Aligned', message: 'residual 0.01 samples', evidence: { delay_samples: 0.01 }, blocking: false }

export async function installFakeApi(page: Page): Promise<FakeState> {
  const dataset = mock<Json>('dataset_builtin')
  const running = mock<Json>('run_running')
  const events = mock<Json[]>('events_running')
  const result = mock<Json>('result_pa_modeling_mock')
  const resolved = mock<Json>('resolved_train_pa_smoke')
  const state: FakeState = { datasets: [], runs: [], submitted: [], diagnostics: {} }
  const rawVersion: Json = { version: 'raw-v1', base_version: null, created_at: '2026-09-06T08:00:00Z', params: null, code_version: null, fit_range: null, record: {}, n_samples: 20000, split: dataset['split'], files: [], sha256: null }
  const recipes = [
    {
      recipe_id: 'pa-gru-smoke-v1',
      title: 'PA model · GRU · smoke',
      purpose: 'smoke',
      task: 'train_pa',
      model: { key: 'gru', parameters: { hidden_size: 23, num_layers: 1 } },
      training: { epochs: 3, batch_size: 64, batch_size_eval: 256, learning_rate: 0.005, lr_end: 0.00005, lr_schedule: true, decay_factor: 0.5, patience: 5, optimizer: 'adamw', loss: 'l2', grad_clip: 200, frame_length: 50, frame_stride: 16, seed: 0, reproducibility: 'soft', eval_val: true, eval_test: true },
      description: 'Proves the PA pipeline end to end.',
      limits: 'Not a benchmark.',
      expected_duration: '~1 min on CPU',
    },
  ]
  const json = (route: Route, body: unknown, status = 200) => route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) })

  await page.route('**/api/v1/**', async (route) => {
    const req = route.request()
    const url = new URL(req.url())
    const path = url.pathname.replace('/api/v1', '')
    const method = req.method()
    if (path === '/session') return json(route, { authenticated: true, csrf_token: 'e2e-csrf', version: '2.2.0.dev0' })
    if (path === '/system/capabilities') {
      return json(route, { version: '2.2.0.dev0', workspace: '/home/user/opendpd workspace', note: 'detected does not imply tested', devices: [{ device: 'cpu', detected: true, count: 1, tested_models: ['gru'] }, { device: 'cuda', detected: false, count: 0, tested_models: ['gru', 'tres_deltagru'] }, { device: 'mps', detected: false, count: 0, tested_models: [] }] })
    }
    if (path === '/models') return json(route, [{ key: 'gru', display_name: 'GRU', family: 'recurrent', legacy_backbone: 'gru', training_method: 'gradient', roles: ['pa', 'dpd'], params: [], status: 'supported', devices_tested: ['cpu', 'cuda'], lookahead_samples: 0, lookahead_note: '', execution_semantics: 'offline_segmented', export_formats: [] }])
    if (path === '/recipes') return json(route, recipes)
    if (path === '/datasets' && method === 'GET') return json(route, state.datasets)
    if (path === '/datasets/import-builtin') {
      if (req.headers()['x-opendpd-csrf'] !== 'e2e-csrf') return json(route, { error: { code: 'csrf_required', message: 'missing', details: [], hint: null } }, 403)
      state.datasets = [dataset]
      return json(route, dataset, 201)
    }
    if (path === '/datasets/import-roots') return json(route, [{ root_id: 'imports', path: '/home/user/opendpd workspace/imports', exists: true }])
    if (path.startsWith('/datasets/import-roots/')) return json(route, [{ path: 'capture.csv', kind: 'file', size_bytes: 1_280_000 }])
    if (path === '/datasets/inspect') return json(route, source)
    if (path === '/datasets/import') {
      const body = req.postDataJSON() as { dataset_id?: string | null; display_name?: string | null; signal: Json; mapping: Json; origin: string }
      const id = body.dataset_id ?? 'capture'
      const imported: Json = { ...dataset, dataset_id: id, display_name: body.display_name || id, origin: body.origin, columns: body.mapping, signal: body.signal, n_samples: 20000, source: { kind: 'csv_import', name: 'capture.csv', original_path: null, imported_at: '2026-09-06T08:00:00Z' }, versions: [rawVersion] }
      state.datasets = [...state.datasets, imported]
      return json(route, imported, 201)
    }
    const dm = path.match(/^\/datasets\/([^/]+)(?:\/(.*))?$/)
    if (dm) {
      const id = decodeURIComponent(dm[1])
      const found = state.datasets.find((d) => d['dataset_id'] === id) ?? dataset
      const sub = dm[2] ?? ''
      if (sub === '') return json(route, found)
      if (sub === 'diagnostics' && method === 'GET') return json(route, state.diagnostics[id] ?? null)
      if (sub === 'diagnostics') {
        state.diagnostics[id] = { ...report, dataset_id: id }
        return json(route, state.diagnostics[id])
      }
      if (sub === 'preprocess/preview') return json(route, { n_samples_before: 20000, n_samples_after: 19994, record: {}, report_after: { ...report, items: [aligned] } })
      if (sub === 'preprocess') {
        const body = req.postDataJSON() as { version: string; params: Json; base_version: string }
        const version: Json = { ...rawVersion, version: body.version, base_version: body.base_version, params: body.params, n_samples: 19994, fit_range: [0, 11996] }
        found['versions'] = [...((found['versions'] as Json[] | undefined) ?? []), version]
        return json(route, version, 201)
      }
      if (sub === 'manifest') {
        Object.assign(found, req.postDataJSON() as Json)
        return json(route, found)
      }
    }
    if (path === '/experiments/validate') return json(route, { ok: true, errors: [], warnings: [{ field: 'training.epochs', message: 'smoke length', hint: null }], resolved })
    if (path === '/runs' && method === 'GET') {
      const status = url.searchParams.get('status')
      return json(route, status ? state.runs.filter((r) => r['status'] === status) : state.runs)
    }
    if (path === '/runs' && method === 'POST') {
      state.submitted.push(req.postDataJSON() as Json)
      const run = { ...running, run_id: 'run-e2e-0001', name: 'e2e smoke', status: 'running', result_id: null }
      state.runs = [run, ...state.runs]
      return json(route, run, 201)
    }
    const m = path.match(/^\/runs\/([^/]+)(?:\/(.*))?$/)
    if (m) {
      const run = state.runs.find((r) => r['run_id'] === m[1])
      if (!run) return json(route, { error: { code: 'run_not_found', message: 'no such run', details: [], hint: null } }, 404)
      const sub = m[2] ?? ''
      if (sub === 'events') {
        const lines = events.map((e): Json => ({ ...e, run_id: run['run_id'] })).map((e) => `id: ${e['seq']}\nevent: ${e['type']}\ndata: ${JSON.stringify(e)}\n\n`)
        const finished = { ...run, status: 'succeeded', result_id: 'res-e2e', progress_epoch: 2, progress_total_epochs: 3 }
        Object.assign(run, finished)
        return route.fulfill({ status: 200, contentType: 'text/event-stream', body: lines.join('') + `event: end\ndata: {"status":"succeeded","last_seq":${events.length}}\n\n` })
      }
      if (sub === 'logs') return json(route, { lines: ['::: Number of PA Model Parameters: 1911', 'Training Completed...'], next_offset: 64, eof: true, size: 64 })
      if (sub === 'artifacts') return json(route, mock<Json>('artifact_manifest_complete'))
      if (sub === 'config') return json(route, resolved)
      if (sub === '') return json(route, run)
    }
    if (path.startsWith('/results/')) return json(route, { ...result, run_id: path.split('/')[2], is_mock: true })
    return json(route, { error: { code: 'not_found', message: `unmocked ${method} ${path}`, details: [], hint: null } }, 404)
  })
  return state
}
