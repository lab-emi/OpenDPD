import type { RunEvent } from './types'
import events from '@mocks/events_running.json'
import { reduceEvent, type StreamState } from './events'

const initial: StreamState = { connection: 'idle', lastSeq: 0, lastUpdate: null, progress: null, metrics: [], statusEvents: [], heartbeats: 0, lastError: null }

test('folds the contract example events into progress, metrics and status', () => {
  const state = (events.data as unknown as RunEvent[]).reduce(reduceEvent, initial)
  expect(state.lastSeq).toBe(events.data.length)
  expect(state.progress).not.toBeNull()
  expect(state.metrics.length).toBeGreaterThan(0)
  expect(state.statusEvents.map((e) => e.payload?.['to'])).toContain('running')
})

test('duplicate or older seq numbers are ignored (safe replay)', () => {
  const list = events.data as unknown as RunEvent[]
  const once = list.reduce(reduceEvent, initial)
  const twice = list.reduce(reduceEvent, once)
  expect(twice).toEqual(once)
})

test('non-finite metric values are dropped instead of plotted as zero', () => {
  const bad: RunEvent = { seq: 1, run_id: 'r', ts: '2026-01-01T00:00:00Z', type: 'metric', payload: { epoch: 0, split: 'val', values: { NMSE: null, EVM: -20 } } }
  const s = reduceEvent(initial, bad)
  expect(s.metrics[0]?.values).toEqual({ EVM: -20 })
})
