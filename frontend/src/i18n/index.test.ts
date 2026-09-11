import { afterEach, expect, test } from 'vitest'
import { LANGUAGES, formatDateTime, formatNumber, getLanguage, resetLanguage, resolveLanguage, setLanguage, t } from './index'

afterEach(() => resetLanguage())

test('seven languages with tags, native names and flags', () => {
  expect(LANGUAGES.map((l) => l.code)).toEqual(['en', 'fr', 'de', 'es', 'zh', 'ja', 'ko'])
  expect(LANGUAGES.map((l) => l.name)).toEqual(['English', 'Français', 'Deutsch', 'Español', '中文', '日本語', '한국어'])
  expect(LANGUAGES.find((l) => l.code === 'zh')?.tag).toBe('zh-CN')
  for (const l of LANGUAGES) expect(l.flag, l.code).toMatch(/\.svg$|^data:image\/svg\+xml/)
})

test('resolution: stored value, then the browser languages, then English', () => {
  expect(resolveLanguage('ja', ['fr-CA'])).toBe('ja')
  expect(resolveLanguage(null, ['fr-CA', 'en-US'])).toBe('fr')
  expect(resolveLanguage(null, ['pt-BR', 'zh-TW'])).toBe('zh')
  expect(resolveLanguage(null, ['pt-BR'])).toBe('en')
  expect(resolveLanguage('xx', [])).toBe('en')
})

test('switching the language translates t() and updates <html lang>', async () => {
  expect(t('nav.datasets')).toBe('Datasets')
  await setLanguage('de')
  expect(getLanguage()).toBe('de')
  expect(t('nav.datasets')).toBe('Datensätze')
  expect(document.documentElement.lang).toBe('de')
  expect(t('topbar.nowRunning', { count: 2 })).toContain('2')
})

test('numbers and dates follow the UI language; English is the default', async () => {
  expect(formatNumber(1234567)).toBe('1,234,567')
  await setLanguage('de')
  expect(formatNumber(1234567)).toBe('1.234.567')
  expect(formatDateTime('2026-09-11T10:30:00Z')).toMatch(/2026/)
})
