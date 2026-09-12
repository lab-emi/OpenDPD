import { ApiError, api, setCsrfToken } from './client'
import { mockApi } from '@/test/utils'

test('POST carries the CSRF header and GET does not', async () => {
  const { fetchMock } = mockApi({ 'GET /api/v1/runs': () => [], 'POST /api/v1/runs': () => ({ run_id: 'r' }) })
  setCsrfToken('tok')
  await api.get('/runs')
  await api.post('/runs', { config: {} })
  const [getInit, postInit] = fetchMock.mock.calls.map((c) => c[1] as RequestInit)
  expect((getInit!.headers as Record<string, string>)['X-OpenDPD-CSRF']).toBeUndefined()
  expect((postInit!.headers as Record<string, string>)['X-OpenDPD-CSRF']).toBe('tok')
  expect(postInit!.credentials).toBe('same-origin')
})

test('error envelope becomes ApiError with code, details and hint', async () => {
  mockApi({
    'POST /api/v1/runs': () => ({
      status: 422,
      body: { error: { code: 'invalid_config', message: 'bad', details: [{ field: 'model.key', message: 'unknown', hint: 'try gru' }], hint: 'fix it' } },
    }),
  })
  const err = await api.post('/runs', {}).catch((e: unknown) => e)
  expect(err).toBeInstanceOf(ApiError)
  const e = err as ApiError
  expect(e.status).toBe(422)
  expect(e.code).toBe('invalid_config')
  expect(e.details[0]?.field).toBe('model.key')
  expect(e.hint).toBe('fix it')
})
