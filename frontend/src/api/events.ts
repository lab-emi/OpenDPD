/**
 * Live run events over SSE. The browser's EventSource reconnects on its own
 * and sends Last-Event-ID, which the server honours; the hook only tracks
 * the connection state and folds events into a compact view model.
 */
import { useCallback, useEffect, useReducer, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { API, WEB_MODE, ApiError, api } from './client'
import { keys } from './hooks'
import type { RunEvent } from './types'

export type Connection = 'idle' | 'connecting' | 'live' | 'disconnected' | 'ended'

export interface MetricPoint {
  epoch: number
  split: string
  values: Record<string, number>
}

export interface StreamState {
  connection: Connection
  lastSeq: number
  lastUpdate: Date | null
  progress: { epoch: number; total: number } | null
  batchProgress?: { phase: string; epoch?: number; total_epochs?: number; batch?: number; total_batches?: number; sequences?: number; sequence_samples?: number; sample_rate_hz?: number | null; padded_samples?: number }
  metrics: MetricPoint[]
  statusEvents: RunEvent[]
  heartbeats: number
  lastError: RunEvent | null
}

const initial: StreamState = {
  connection: 'idle',
  lastSeq: 0,
  lastUpdate: null,
  progress: null,
  metrics: [],
  statusEvents: [],
  heartbeats: 0,
  lastError: null,
}

type Action = { kind: 'connection'; connection: Connection } | { kind: 'events'; events: RunEvent[] } | { kind: 'reset' }

const MAX_METRIC_POINTS = 20_000

function asNumber(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null
}

export function reduceEvent(state: StreamState, event: RunEvent): StreamState {
  if (event.seq <= state.lastSeq) return state
  const next: StreamState = { ...state, lastSeq: event.seq, lastUpdate: new Date(event.ts) }
  const p = event.payload ?? {}
  switch (event.type) {
    case 'progress': {
      const epoch = asNumber(p['epoch'])
      const total = asNumber(p['total_epochs'])
      if (epoch !== null && total !== null && p['scope'] !== 'live') next.progress = { epoch, total }
      if (typeof p['phase'] === 'string' && p['phase'] !== 'epoch_end') {
        next.batchProgress = { phase: p['phase'], ...Object.fromEntries(['epoch', 'total_epochs', 'batch', 'total_batches', 'sequences', 'sequence_samples', 'sample_rate_hz', 'padded_samples'].flatMap((key) => asNumber(p[key]) !== null ? [[key, p[key]]] : [])) }
      }
      break
    }
    case 'metric': {
      const epoch = asNumber(p['epoch'])
      const values = p['values']
      if (epoch !== null && values && typeof values === 'object') {
        const clean: Record<string, number> = {}
        for (const [k, v] of Object.entries(values as Record<string, unknown>)) {
          const n = asNumber(v)
          if (n !== null) clean[k] = n
        }
        const point: MetricPoint = { epoch, split: String(p['split'] ?? 'val'), values: clean }
        next.metrics = state.metrics.length >= MAX_METRIC_POINTS ? [...state.metrics.slice(1), point] : [...state.metrics, point]
      }
      break
    }
    case 'status':
      next.statusEvents = [...state.statusEvents, event]
      break
    case 'heartbeat':
      next.heartbeats = state.heartbeats + 1
      break
    case 'error':
      next.lastError = event
      break
    default:
      break
  }
  return next
}

function reducer(state: StreamState, action: Action): StreamState {
  switch (action.kind) {
    case 'connection':
      return state.connection === action.connection ? state : { ...state, connection: action.connection }
    case 'events':
      return action.events.reduce(reduceEvent, state)
    case 'reset':
      return initial
  }
}

const EVENT_TYPES = ['status', 'progress', 'metric', 'log', 'artifact', 'checkpoint', 'heartbeat', 'error'] as const

/** Subscribes to /runs/{id}/events while `enabled`; replays from seq 0 on mount. */
export function useRunStream(runId: string, enabled: boolean): StreamState & { reconnect: () => void } {
  const [state, dispatch] = useReducer(reducer, initial)
  const qc = useQueryClient()
  const sourceRef = useRef<EventSource | null>(null)
  const cursorRef = useRef(0)
  const [revision, setRevision] = useState(0)
  const reconnect = useCallback(() => {
    setRevision((value) => value + 1)
    void qc.invalidateQueries({ queryKey: keys.run(runId) })
    void qc.invalidateQueries({ queryKey: keys.runHistory(runId) })
  }, [qc, runId])

  useEffect(() => {
    dispatch({ kind: 'reset' })
    cursorRef.current = 0
    // The run ID deliberately resets replay, even before a transport is enabled.
    // eslint-disable-next-line react/exhaustive-effect-dependencies
  }, [runId])

  useEffect(() => {
    if (enabled && WEB_MODE) {
      // Short, authenticated polls work across github.io and the Tunnel without
      // third-party cookies or bearer tokens in EventSource query strings.
      let stopped = false
      let polling = false
      let ended = false
      let timer: number | undefined
      let retryAt = 0
      const controller = new AbortController()
      const poll = async () => {
        if (stopped || polling || ended) return
        window.clearTimeout(timer)
        if (document.visibilityState === 'hidden' || Date.now() < retryAt) {
          timer = window.setTimeout(() => void poll(), Math.max(3000, retryAt - Date.now()))
          return
        }
        polling = true
        let delay = 3000
        try {
          const page = await api.get<{ events: RunEvent[]; last_seq: number; terminal: boolean }>(`/runs/${encodeURIComponent(runId)}/events/list?after=${cursorRef.current}&limit=500`, controller.signal)
          if (stopped) return
          cursorRef.current = page.last_seq
          dispatch({ kind: 'events', events: page.events })
          dispatch({ kind: 'connection', connection: 'live' })
          // Replay quickly, without restarting the snapshot queries on every heartbeat.
          if (page.events.length === 500) delay = 100
          else if (page.events.some((event) => ['status', 'error', 'artifact'].includes(event.type))) {
            void qc.invalidateQueries({ queryKey: keys.run(runId), exact: true })
          }
          if (page.events.length < 500 && page.events.some((event) => event.type === 'progress' && event.payload?.['preview_revision'])) {
            void qc.invalidateQueries({ queryKey: ['run', runId, 'live'], exact: true })
          }
          if (page.terminal && page.events.length < 500) {
            ended = true
            dispatch({ kind: 'connection', connection: 'ended' })
            for (const key of [keys.run(runId), keys.result(runId), ['runs']]) void qc.invalidateQueries({ queryKey: key })
          }
        } catch (error) {
          if (stopped) return
          dispatch({ kind: 'connection', connection: 'disconnected' })
          if (error instanceof ApiError) {
            if (error.status === 401) ended = true
            if (error.code === 'cursor_out_of_range') {
              cursorRef.current = 0
              dispatch({ kind: 'reset' })
            }
            delay = Math.max(delay, error.retryAfterMs)
            retryAt = Date.now() + delay
          }
        } finally {
          polling = false
          if (!stopped && !ended) timer = window.setTimeout(() => void poll(), delay)
        }
      }
      const resume = () => { if (document.visibilityState !== 'hidden') void poll() }
      window.addEventListener('online', resume)
      document.addEventListener('visibilitychange', resume)
      dispatch({ kind: 'connection', connection: 'connecting' })
      void poll()
      return () => {
        stopped = true
        controller.abort()
        window.clearTimeout(timer)
        window.removeEventListener('online', resume)
        document.removeEventListener('visibilitychange', resume)
      }
    }
    if (!enabled || typeof EventSource === 'undefined') return
    const source = new EventSource(`${API}/runs/${encodeURIComponent(runId)}/events?after=${cursorRef.current}`)
    sourceRef.current = source
    dispatch({ kind: 'connection', connection: 'connecting' })
    source.onopen = () => dispatch({ kind: 'connection', connection: 'live' })
    source.onerror = () => dispatch({ kind: 'connection', connection: 'disconnected' })
    let pending: RunEvent[] = []
    let timer: number | undefined
    const flush = () => {
      window.clearTimeout(timer)
      timer = undefined
      if (pending.length) {
        cursorRef.current = pending[pending.length - 1]!.seq
        dispatch({ kind: 'events', events: pending })
      }
      pending = []
    }
    const onEvent = (raw: MessageEvent<string>) => {
      let event: RunEvent
      try {
        event = JSON.parse(raw.data) as RunEvent
      } catch {
        return
      }
      pending.push(event)
      if (timer === undefined) timer = window.setTimeout(flush, 500)
      if (event.type === 'status' || event.type === 'error' || event.type === 'artifact') {
        flush()
        void qc.invalidateQueries({ queryKey: keys.run(runId) })
      }
      if (event.type === 'progress' && event.payload?.['preview_revision']) void qc.invalidateQueries({ queryKey: ['run', runId, 'live'] })
    }
    for (const type of EVENT_TYPES) source.addEventListener(type, onEvent as EventListener)
    source.addEventListener('end', () => {
      flush()
      dispatch({ kind: 'connection', connection: 'ended' })
      source.close()
      void qc.invalidateQueries({ queryKey: keys.run(runId) })
      void qc.invalidateQueries({ queryKey: keys.result(runId) })
      void qc.invalidateQueries({ queryKey: keys.runArtifacts(runId) })
      void qc.invalidateQueries({ queryKey: ['run', runId, 'live'] })
      void qc.invalidateQueries({ queryKey: ['runs'] })
    })
    return () => {
      window.clearTimeout(timer)
      source.close()
      sourceRef.current = null
    }
    // A manual reconnect deliberately replaces the transport, preserving its cursor.
    // eslint-disable-next-line react/exhaustive-effect-dependencies
  }, [runId, enabled, qc, revision])

  return { ...state, reconnect }
}
