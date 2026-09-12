import { ThemeProvider } from '@mui/material/styles'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, type RenderOptions } from '@testing-library/react'
import type { ReactElement, ReactNode } from 'react'
import { MemoryRouter, Route, Routes } from 'react-router'
import { vi } from 'vitest'
import { setCsrfToken } from '@/api/client'
import { theme } from '@/theme'

export function renderWithProviders(ui: ReactElement, { route = '/', path = '*', ...options }: RenderOptions & { route?: string; path?: string } = {}) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false, gcTime: 0 }, mutations: { retry: false } } })
  const Wrapper = ({ children }: { children: ReactNode }) => (
    <ThemeProvider theme={theme}>
      <QueryClientProvider client={client}>
        <MemoryRouter initialEntries={[route]}>
          <Routes>
            <Route path={path} element={children} />
          </Routes>
        </MemoryRouter>
      </QueryClientProvider>
    </ThemeProvider>
  )
  return { client, ...render(ui, { wrapper: Wrapper, ...options }) }
}

export type Handler = (url: URL, init: RequestInit) => unknown | { status: number; body: unknown }

/** Installs a fetch mock routed by `METHOD path` (query string ignored) and records calls. */
export function mockApi(routes: Record<string, Handler>) {
  const calls: Array<{ method: string; path: string; body: unknown }> = []
  setCsrfToken('test-csrf')
  const fetchMock = vi.fn(async (input: RequestInfo | URL, init: RequestInit = {}) => {
    const url = new URL(typeof input === 'string' ? input : input instanceof URL ? input.href : input.url, 'http://127.0.0.1')
    const method = (init.method ?? 'GET').toUpperCase()
    const key = `${method} ${url.pathname}`
    const body = typeof init.body === 'string' ? (JSON.parse(init.body) as unknown) : undefined
    calls.push({ method, path: url.pathname, body })
    const handler = routes[key]
    if (!handler) {
      return new Response(JSON.stringify({ error: { code: 'not_found', message: `no mock for ${key}`, details: [], hint: null } }), { status: 404, headers: { 'Content-Type': 'application/json' } })
    }
    const result = handler(url, init) as { status?: number; body?: unknown }
    const isEnvelope = result !== null && typeof result === 'object' && 'status' in result && 'body' in result
    const status = isEnvelope ? (result.status as number) : 200
    const payload = isEnvelope ? result.body : result
    return new Response(JSON.stringify(payload), { status, headers: { 'Content-Type': 'application/json' } })
  })
  vi.stubGlobal('fetch', fetchMock)
  return { calls, fetchMock }
}

/** Minimal EventSource double: tests push events with `emit`. */
export class FakeEventSource {
  static instances: FakeEventSource[] = []
  url: string
  listeners = new Map<string, Array<(e: MessageEvent<string>) => void>>()
  onopen: (() => void) | null = null
  onerror: (() => void) | null = null
  closed = false
  constructor(url: string) {
    this.url = url
    FakeEventSource.instances.push(this)
  }
  addEventListener(type: string, cb: (e: MessageEvent<string>) => void) {
    this.listeners.set(type, [...(this.listeners.get(type) ?? []), cb])
  }
  close() {
    this.closed = true
  }
  open() {
    this.onopen?.()
  }
  fail() {
    this.onerror?.()
  }
  emit(type: string, data: unknown) {
    for (const cb of this.listeners.get(type) ?? []) cb(new MessageEvent(type, { data: JSON.stringify(data) }))
  }
}

export function installFakeEventSource() {
  FakeEventSource.instances = []
  vi.stubGlobal('EventSource', FakeEventSource)
  return FakeEventSource
}
