/**
 * Thin fetch wrapper over /api/v1: same-origin cookies, CSRF header on
 * writes, and the uniform error envelope turned into ApiError.
 */
import type { SessionInfo } from './types'

export const API = '/api/v1'
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

  constructor(status: number, code: string, message: string, details: ApiErrorDetail[] = [], hint: string | null = null) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.code = code
    this.details = details
    this.hint = hint
  }
}

let csrfToken: string | null = null

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
  return new ApiError(response.status, code, message, details, hint)
}

async function request<T>(method: 'GET' | 'POST', path: string, body?: unknown): Promise<T> {
  const headers: Record<string, string> = { Accept: 'application/json' }
  if (body !== undefined) headers['Content-Type'] = 'application/json'
  if (method !== 'GET' && csrfToken) headers[CSRF_HEADER] = csrfToken
  const response = await fetch(`${API}${path}`, {
    method,
    headers,
    credentials: 'same-origin',
    body: body === undefined ? undefined : JSON.stringify(body),
  })
  if (!response.ok) throw await parseError(response)
  if (response.status === 204) return undefined as T
  return (await response.json()) as T
}

export const api = {
  get: <T>(path: string) => request<T>('GET', path),
  post: <T>(path: string, body?: unknown) => request<T>('POST', path, body),
}

/** Loads the session and remembers the CSRF token for later writes. */
export async function loadSession(): Promise<SessionInfo> {
  const info = await api.get<SessionInfo>('/session')
  setCsrfToken(info.csrf_token ?? null)
  return info
}

export async function bootstrapSession(token: string): Promise<SessionInfo> {
  const info = await api.post<SessionInfo>('/session/bootstrap', { token })
  setCsrfToken(info.csrf_token ?? null)
  return info
}

export function artifactUrl(runId: string, artifactId: string): string {
  return `${API}/artifacts/${encodeURIComponent(runId)}/${encodeURIComponent(artifactId)}`
}
