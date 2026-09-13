/**
 * Thin fetch wrapper over /api/v1: same-origin cookies, CSRF header on
 * writes, and the uniform error envelope turned into ApiError.
 */
import type { SessionInfo } from './types'
import { boundedRead } from './read-queue'

export const WEB_MODE = import.meta.env.VITE_STUDIO_MODE === 'web'
export const API_ORIGIN = WEB_MODE ? String(import.meta.env.VITE_API_ORIGIN).replace(/\/$/, '') : ''
export const API = `${API_ORIGIN}/api/v1`
const SESSION_KEY = `opendpd-web-session:${API_ORIGIN}`
export type WebSessionInfo = SessionInfo & { mode?: 'web'; expires_at?: string; access_token?: string }
const CSRF_HEADER = 'X-OpenDPD-CSRF'

export interface ApiErrorDetail {
  field?: string
  message: string
  hint?: string | null
}

export class ApiError extends Error {
  readonly status: number
  readonly code: string
  readonly details: ApiErrorDetail[]
  readonly hint: string | null
  retryAfterMs = 0

  constructor(status: number, code: string, message: string, details: ApiErrorDetail[] = [], hint: string | null = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
    this.details = details
    this.hint = hint
  }
}

/** Fetch cannot distinguish DNS, offline, TLS and blocked cross-origin requests. */
export class ApiConnectionError extends Error {
  readonly origin: string

  constructor(origin: string) {
    super(`Cannot connect to the compute server at ${origin}`)
    this.name = 'ApiConnectionError'
    this.origin = origin
  }
}

async function fetchApi(input: string, init: RequestInit): Promise<Response> {
  try {
    return await fetch(input, init)
  } catch (error) {
    if (WEB_MODE && error instanceof TypeError) throw new ApiConnectionError(API_ORIGIN)
    throw error
  }
}

let csrfToken: string | null = null

function bearerToken(): string | null {
  return WEB_MODE ? sessionStorage.getItem(SESSION_KEY) : null
}

export function clearWebSession(): void {
  sessionStorage.removeItem(SESSION_KEY)
}

function authorizationHeaders(): Record<string, string> {
  const token = bearerToken()
  return token ? { Authorization: `Bearer ${token}` } : {}
}

function expired(response: Response): void {
  if (WEB_MODE && response.status === 401) {
    clearWebSession()
    window.dispatchEvent(new Event('opendpd-session-expired'))
  }
}

export function setCsrfToken(token: string | null): void {
  csrfToken = token
}

async function parseError(response: Response): Promise<ApiError> {
  let code = 'http_error'
  let message = `${response.status} ${response.statusText}`
  let details: ApiErrorDetail[] = []
  let hint: string | null = null
  try {
    const body = (await response.json()) as { error?: { code: string; message: string; details?: ApiErrorDetail[]; hint?: string | null } }
    if (body.error) {
      code = body.error.code
      message = body.error.message
      details = body.error.details ?? []
      hint = body.error.hint ?? null
    }
  } catch {
    // non-JSON body: keep the HTTP status text
  }
  const error = new ApiError(response.status, code, message, details, hint)
  const retry = response.headers.get('retry-after')
  if (retry) error.retryAfterMs = Math.max(0, Math.min(60_000, /^\d+$/.test(retry) ? Number(retry) * 1000 : Date.parse(retry) - Date.now())) || 0
  return error
}

async function request<T>(method: 'GET' | 'POST' | 'PUT', path: string, body?: unknown, signal?: AbortSignal): Promise<T> {
  const session = bearerToken()
  const headers: Record<string, string> = { Accept: 'application/json', ...authorizationHeaders() }
  if (body !== undefined || method !== 'GET') headers['Content-Type'] = 'application/json'
  if (!WEB_MODE && method !== 'GET' && csrfToken) headers[CSRF_HEADER] = csrfToken
  const response = await fetchApi(`${API}${path}`, {
    method,
    headers,
    redirect: 'error',
    credentials: WEB_MODE ? 'omit' : 'same-origin',
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
  })
  if (WEB_MODE && bearerToken() !== session) throw new DOMException('Session changed', 'AbortError')
  if (!response.ok) { expired(response); throw await parseError(response) }
  if (response.status === 204) return undefined as T
  return (await response.json()) as T
}

/** Multipart POST (file upload): same cookie/CSRF rules, no JSON body. */
async function upload<T>(path: string, form: FormData): Promise<T> {
  if (WEB_MODE) {
    const file = form.get('file')
    if (path !== '/datasets/upload' || !(file instanceof File) || !/\.csv$/i.test(file.name)) throw new ApiError(415, 'csv_required', 'Only CSV dataset uploads are accepted')
    if (file.size > 25 * 1024 * 1024) throw new ApiError(413, 'payload_too_large', 'CSV files must be at most 25 MiB')
    const session = bearerToken()
    const response = await fetchApi(`${API}${path}?filename=${encodeURIComponent(file.name)}`, {
      method: 'POST', headers: { Accept: 'application/json', 'Content-Type': 'text/csv', ...authorizationHeaders() },
      credentials: 'omit', redirect: 'error', body: file, signal: AbortSignal.timeout(60_000),
    })
    if (bearerToken() !== session) throw new DOMException('Session changed', 'AbortError')
    if (!response.ok) { expired(response); throw await parseError(response) }
    return await response.json() as T
  }
  const headers: Record<string, string> = { Accept: 'application/json' }
  if (csrfToken) headers[CSRF_HEADER] = csrfToken
  const response = await fetch(`${API}${path}`, { method: 'POST', headers, credentials: 'same-origin', body: form })
  if (!response.ok) throw await parseError(response)
  return (await response.json()) as T
}

export const api = {
  get: <T>(path: string, signal?: AbortSignal) => {
    const session = bearerToken()
    return WEB_MODE ? boundedRead((readSignal) => {
      if (bearerToken() !== session) throw new DOMException('Session changed', 'AbortError')
      return request<T>('GET', path, undefined, readSignal)
    }, signal) : request<T>('GET', path, undefined, signal)
  },
  post: <T>(path: string, body?: unknown) => request<T>('POST', path, body),
  put: <T>(path: string, body?: unknown) => request<T>('PUT', path, body),
  upload,
}

/** Loads the session and remembers the CSRF token for later writes. */
export async function loadSession(): Promise<WebSessionInfo> {
  if (WEB_MODE && !bearerToken()) return { authenticated: false, mode: 'web', version: '' }
  let info: WebSessionInfo
  try {
    info = await api.get<WebSessionInfo>('/session')
  } catch (error) {
    if (WEB_MODE && error instanceof ApiError && error.status === 401) return { authenticated: false, mode: 'web', version: '' }
    throw error
  }
  setCsrfToken(info.csrf_token ?? null)
  return info
}

export async function createWebSession(): Promise<WebSessionInfo> {
  const info = await api.post<WebSessionInfo>('/web/sessions', {})
  if (info.access_token) sessionStorage.setItem(SESSION_KEY, info.access_token)
  // Keep the capability out of React Query caches and browser-visible diagnostics.
  const { access_token: _token, ...publicInfo } = info
  return publicInfo
}

/** Authenticated downloads never put bearer capabilities into URLs or referrers. */
export async function downloadFile(href: string, filename?: string): Promise<void> {
  const url = new URL(href, WEB_MODE ? API_ORIGIN : window.location.origin)
  const expected = WEB_MODE ? API_ORIGIN : window.location.origin
  if (url.origin !== expected || !url.pathname.startsWith('/api/v1/')) throw new Error('Invalid artifact URL')
  const session = bearerToken()
  const download = async (signal?: AbortSignal) => {
    if (WEB_MODE && bearerToken() !== session) throw new DOMException('Session changed', 'AbortError')
    const response = await fetchApi(url.href, { headers: authorizationHeaders(), credentials: WEB_MODE ? 'omit' : 'same-origin', redirect: 'error', signal })
    if (WEB_MODE && bearerToken() !== session) throw new DOMException('Session changed', 'AbortError')
    if (!response.ok) { expired(response); throw await parseError(response) }
    return { data: await response.blob(), disposition: response.headers.get('content-disposition') ?? '' }
  }
  const { data, disposition } = WEB_MODE ? await boundedRead(download, undefined, 120_000) : await download()
  const suggested = disposition.match(/filename="([^"\r\n]+)"/)?.[1]
  const blob = URL.createObjectURL(data)
  const link = document.createElement('a')
  link.href = blob
  link.download = (filename || suggested || url.pathname.split('/').pop() || 'download').replace(/[/\\]/g, '_')
  link.click()
  window.setTimeout(() => URL.revokeObjectURL(blob), 1000)
}

export async function bootstrapSession(token: string): Promise<SessionInfo> {
  const info = await api.post<SessionInfo>('/session/bootstrap', { token })
  setCsrfToken(info.csrf_token ?? null)
  return info
}

export function artifactUrl(runId: string, artifactId: string): string {
  return `${API}/artifacts/${encodeURIComponent(runId)}/${encodeURIComponent(artifactId)}`
}
