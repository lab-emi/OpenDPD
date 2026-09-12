import { afterEach, beforeEach, expect, test, vi } from 'vitest'

beforeEach(() => {
  vi.resetModules()
  vi.stubEnv('VITE_STUDIO_MODE', 'web')
  vi.stubEnv('VITE_API_ORIGIN', 'https://api.opendpd.com')
  sessionStorage.clear()
})
afterEach(() => { vi.unstubAllEnvs(); vi.unstubAllGlobals(); sessionStorage.clear() })

test('web requests use a tab session bearer, no cookies, CSRF or URL credentials', async () => {
  const fetcher = vi.fn().mockResolvedValueOnce(new Response(JSON.stringify({ authenticated: true, access_token: 'private-capability', expires_at: '2026-09-12T23:55:00Z' }), { status: 201 }))
    .mockResolvedValueOnce(new Response('[]'))
  vi.stubGlobal('fetch', fetcher)
  const { api, createWebSession, setCsrfToken } = await import('./client')
  const session = await createWebSession()
  expect(session).not.toHaveProperty('access_token')
  setCsrfToken('local-only')
  await api.post('/runs', { config: {} })
  expect(fetcher.mock.calls[1]![0]).toBe('https://api.opendpd.com/api/v1/runs')
  expect(fetcher.mock.calls[1]![1]).toMatchObject({ credentials: 'omit', headers: { Authorization: 'Bearer private-capability' } })
  expect(fetcher.mock.calls[1]![1].headers).not.toHaveProperty('X-OpenDPD-CSRF')
})

test('expired capabilities are removed and cannot follow a user into a new session', async () => {
  sessionStorage.setItem('opendpd-web-session:https://api.opendpd.com', 'expired')
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{"error":{"code":"session_expired","message":"expired"}}', { status: 401 })))
  const { loadSession } = await import('./client')
  expect(await loadSession()).toMatchObject({ authenticated: false })
  expect(sessionStorage.length).toBe(0)
})

test('download requests cannot leak a capability to another origin or non-API path', async () => {
  const fetcher = vi.fn()
  vi.stubGlobal('fetch', fetcher)
  const { downloadFile } = await import('./client')
  await expect(downloadFile('https://evil.example/api/v1/artifact')).rejects.toThrow('Invalid artifact URL')
  await expect(downloadFile('https://api.opendpd.com/redirect')).rejects.toThrow('Invalid artifact URL')
  expect(fetcher).not.toHaveBeenCalled()
})

test('web uploads are refused before reading or transmitting a file', async () => {
  const fetcher = vi.fn()
  vi.stubGlobal('fetch', fetcher)
  const { api } = await import('./client')
  await expect(api.upload('/datasets/upload', new FormData())).rejects.toMatchObject({ status: 403 })
  expect(fetcher).not.toHaveBeenCalled()
})

test('a DNS or transport failure is recoverable without creating a session or retrying a POST automatically', async () => {
  const fetcher = vi.fn().mockRejectedValueOnce(new TypeError('Failed to fetch'))
    .mockResolvedValueOnce(new Response(JSON.stringify({ authenticated: true, access_token: 'recovered' }), { status: 201 }))
  vi.stubGlobal('fetch', fetcher)
  const { createWebSession, ApiConnectionError } = await import('./client')
  await expect(createWebSession()).rejects.toBeInstanceOf(ApiConnectionError)
  expect(fetcher).toHaveBeenCalledTimes(1)
  expect(sessionStorage.length).toBe(0)
  expect(await createWebSession()).toMatchObject({ authenticated: true })
  expect(sessionStorage.getItem('opendpd-web-session:https://api.opendpd.com')).toBe('recovered')
})

test('HTTP quota responses remain distinguishable from a connection failure', async () => {
  vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{"error":{"code":"rate_limited","message":"Try later"}}', { status: 429 })))
  const { createWebSession } = await import('./client')
  await expect(createWebSession()).rejects.toMatchObject({ status: 429, code: 'rate_limited' })
})
