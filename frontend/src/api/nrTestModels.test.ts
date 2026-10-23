import { expect, test } from 'vitest'
import { ApiError } from '@/api/client'
import { fixtureCatalog } from '@/test/nrTestModelFixtures'
import { catalogUnavailable, CatalogShapeError, parseCatalog } from './nrTestModels'

const catalog = () => structuredClone(fixtureCatalog()) as unknown as Record<string, unknown>

test('the catalogue of the backend passes the shape check and is returned as it is', () => {
  const raw = catalog()
  expect(parseCatalog(raw)).toBe(raw)
})

test.each([
  ['null', () => null],
  ['a list', () => []],
  ['a string', () => '<html>proxy</html>'],
  ['no version', () => ({ ...catalog(), version: undefined })],
  ['another document type', () => ({ ...catalog(), version: 'something-else' })],
  ['an empty model list', () => ({ ...catalog(), models: [] })],
  ['models that is not a list', () => ({ ...catalog(), models: {} })],
  ['a model without its lists', () => ({ ...catalog(), models: [{ model: 'NR-FR1-TM1.1', title: 'x' }] })],
  ['no limits', () => ({ ...catalog(), limits: undefined })],
  ['a limit that is not a number', () => ({ ...catalog(), limits: { ...(catalog().limits as object), max_carriers: '16' } })],
  ['no resource-block table', () => ({ ...catalog(), resource_blocks: undefined })],
  ['no TDD patterns', () => ({ ...catalog(), tdd_patterns: undefined })],
  ['no NB-IoT tables', () => ({ ...catalog(), nb_iot: {} })],
  ['no default recipe', () => ({ ...catalog(), default_recipe: undefined })],
  ['no default configuration', () => ({ ...catalog(), default_config: {} })],
])('a reply with %s is refused, so the page treats it as a failure instead of crashing in render', (_name, build) => {
  expect(() => parseCatalog(build())).toThrow(CatalogShapeError)
})

test('only 403 and 404 mean "this server does not offer the test models"; every other failure deserves a notice', () => {
  expect(catalogUnavailable(new ApiError(403, 'forbidden', 'no'))).toBe(true)
  expect(catalogUnavailable(new ApiError(404, 'not_found', 'no'))).toBe(true)
  expect(catalogUnavailable(new ApiError(500, 'internal_error', 'no'))).toBe(false)
  expect(catalogUnavailable(new ApiError(502, 'bad_gateway', 'no'))).toBe(false)
  expect(catalogUnavailable(new CatalogShapeError('no models'))).toBe(false)
  expect(catalogUnavailable(new TypeError('Failed to fetch'))).toBe(false)
  expect(catalogUnavailable(null)).toBe(false)
})
