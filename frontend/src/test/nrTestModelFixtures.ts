/**
 * Stand-in for the NR test-model backend (catalog + resolve) used by the vitest files.
 *
 * Nothing in here is computed by the frontend: every number is a real answer of the backend.
 *  - `fixtureCatalog()` is `frontend/mocks/nr_test_model_catalog.json` (`python -m opendpd.schemas export-mocks --out frontend/mocks`).
 *  - `fixtureResolve(recipe)` looks the recipe up in `nrTmFixtureResponses.json`, which `scripts/capture_nr_tm_fixtures.py` fills by sending the
 *    recipes of `nrTmFixtureRecipes.json` to the real local API (resolve answers and the 422 envelopes of refused recipes, verbatim).
 *  - `fixtureSignal(name)` is a generated signal (resolve + batches + GET signal) from the same file, long plot arrays decimated.
 * A test that needs another recipe adds it to `nrTmFixtureRecipes.json` and runs the script; an unknown recipe fails with that instruction.
 */
import catalogMock from '@mocks/nr_test_model_catalog.json'
import type { NrTmRecipe } from '@/api/nrTestModels'
import type { GeneratedSignal } from '@/api/signalGenerator'
import type { Handler } from '@/test/utils'
import type { Schemas } from '@/api/types'
import { recipeKey } from '@/utils/nrTestModel'
import captured from './nrTmFixtureResponses.json'

type Catalog = Schemas['NrTmCatalog']
type Resolution = Schemas['NrTmResolution']
export interface Envelope { error: { code: string; message: string; details: Array<{ field: string; message: string }>; hint: string | null } }
interface Answer { request: NrTmRecipe; status: number; body: Resolution | Envelope }

/** Thrown by `fixtureResolve` for a recipe the real server refused; `status` and `body` are its answer. */
export class FixtureInvalid extends Error {
  readonly status: number
  readonly body: Envelope
  constructor(status: number, body: Envelope) {
    super(body.error.details[0]?.message ?? body.error.message)
    this.status = status
    this.body = body
  }
}
/** Thrown for a recipe nobody captured from the real backend yet. */
export class FixtureMissing extends Error {}

const answers = new Map((captured.resolutions as unknown as Answer[]).map(answer => [recipeKey(answer.request), answer]))

/** The real answer to a recipe: status 200 with a resolution or the refusal envelope. */
export function fixtureAnswer(recipe: NrTmRecipe): { status: number; body: Resolution | Envelope } {
  const answer = answers.get(recipeKey(recipe))
  if (!answer) throw new FixtureMissing(`No captured backend answer for the recipe ${JSON.stringify(recipe)}. Add it to frontend/src/test/nrTmFixtureRecipes.json and run `
    + '`PYTHONPATH=. python scripts/capture_nr_tm_fixtures.py`.')
  return answer
}

export function fixtureResolve(recipe: NrTmRecipe): Resolution {
  const { status, body } = fixtureAnswer(recipe)
  if (status !== 200) throw new FixtureInvalid(status, body as Envelope)
  return body as Resolution
}

export const fixtureCatalog = (): Catalog => catalogMock.data as unknown as Catalog

/** A generated signal captured from the real API; `recipe` is the one it was generated from. */
export function fixtureSignal(name: string): { recipe: NrTmRecipe; signal: GeneratedSignal } {
  const entry = (captured.signals as unknown as Record<string, { recipe: NrTmRecipe; signal: GeneratedSignal }>)[name]
  if (!entry) throw new FixtureMissing(`No captured signal "${name}"; add it to "generate" in frontend/src/test/nrTmFixtureRecipes.json and run scripts/capture_nr_tm_fixtures.py.`)
  return entry
}

/** `mockApi` routes for the two endpoints; refused recipes answer exactly as the real server did when the fixture was captured. */
export const nrTmRoutes = (): Record<string, Handler> => ({
  'GET /api/v1/signal-generator/nr-test-models': () => fixtureCatalog(),
  'POST /api/v1/signal-generator/nr-test-models/resolve': (_url, init) => {
    try { return fixtureAnswer(JSON.parse(String(init.body)) as NrTmRecipe) } catch (error) {
      if (error instanceof FixtureMissing) return { status: 500, body: { error: { code: 'fixture_missing', message: error.message, details: [], hint: null } } }
      throw error
    }
  },
})
