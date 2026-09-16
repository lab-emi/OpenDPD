/// <reference types="node" />
import { createHash } from 'node:crypto'
import { readdirSync, readFileSync } from 'node:fs'
import { join } from 'node:path'
import { artifactDownloadUrl, safePullRequestUrl } from './api/client'
import { isRunEvent } from './api/events'

function files(dir: string): string[] {
  return readdirSync(dir, { withFileTypes: true }).flatMap(entry => entry.isDirectory() ? files(join(dir, entry.name)) : [join(dir, entry.name)])
}

test('HTML injection is confined to the audited equation renderer', () => {
  const sinks = files('src').filter(file => /\.tsx?$/.test(file) && !file.includes('.test.'))
    .filter(file => /dangerouslySetInnerHTML|\.innerHTML\s*=|insertAdjacentHTML\s*\(/.test(readFileSync(file, 'utf8')))
  expect(sinks).toEqual(['src/components/MathFormula.tsx'])
})

test('vendored plot bundle matches its reviewed digest', () => {
  expect(createHash('sha256').update(readFileSync('src/vendor/plotly-scatter-strict.cjs')).digest('hex'))
    .toBe('d4f966bf3a7f2acaecc7a21734311ea984389a0ca97c62b457892f70f9a53112')
})

test('artifact and pull request links reject external or active URLs', () => {
  for (const value of ['javascript:alert(1)', 'https://evil.example/api/v1/export', '//evil.example/api/v1/export', '/unrelated']) {
    expect(() => artifactDownloadUrl(value)).toThrow()
    expect(safePullRequestUrl(value)).toBeUndefined()
  }
  expect(artifactDownloadUrl('/api/v1/exports/test')).toContain('/api/v1/exports/test')
  expect(safePullRequestUrl('https://github.com/lab-emi/OpenDPD/pull/42')).toBeDefined()
})

test('event transport rejects malformed cursors, dates and payloads', () => {
  const valid = { seq: 1, run_id: 'run-1', type: 'status', ts: '2026-09-16T00:00:00Z', payload: {} }
  expect(isRunEvent(valid)).toBe(true)
  for (const patch of [{ seq: NaN }, { seq: -1 }, { ts: 'bad' }, { type: 'unknown' }, { payload: [] }]) expect(isRunEvent({ ...valid, ...patch })).toBe(false)
})
