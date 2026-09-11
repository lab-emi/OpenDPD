import { describe, expect, test } from 'vitest'
import type { Catalogue } from './index'
import de from './de.json'
import en from './en.json'
import es from './es.json'
import fr from './fr.json'
import ja from './ja.json'
import ko from './ko.json'
import zh from './zh.json'

// Compile-time completeness: a catalogue missing a key does not typecheck.
const typed: Catalogue[] = [fr, de, es, zh, ja, ko]
const all: Record<string, Record<string, string>> = { fr, de, es, zh, ja, ko }
const placeholders = (s: string) => [...s.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort()
const code = (s: string) => [...s.matchAll(/`[^`]+`/g)].map((m) => m[0]).sort()

describe.each(Object.entries(all))('catalogue %s', (_code, catalogue) => {
  test('has exactly the English keys', () => {
    expect(typed.length).toBe(6)
    expect(Object.keys(catalogue).sort()).toEqual(Object.keys(en).sort())
  })
  test('keeps every placeholder and inline code, and has no empty string', () => {
    for (const [key, value] of Object.entries(en)) {
      const translated = catalogue[key] ?? ''
      expect(placeholders(translated), key).toEqual(placeholders(value))
      expect(code(translated), key).toEqual(code(value))
      expect(translated.trim().length, key).toBeGreaterThan(0)
    }
  })
  test('keeps the markers that must never be translated', () => {
    expect(catalogue['evidence.mock']).toBe('MOCK')
    expect(catalogue['app.title']).toBe('OpenDPD Studio')
  })
})
