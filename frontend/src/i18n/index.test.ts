import { afterEach, expect, test } from 'vitest'
import { LANGUAGES, formatDateTime, formatNumber, getLanguage, resetLanguage, resolveLanguage, setLanguage, t } from './index'

afterEach(() => resetLanguage())

test('nine languages with tags, native names and flags', () => {
  expect(LANGUAGES.map((l) => l.code)).toEqual(['en', 'nl', 'zh', 'fr', 'de', 'it', 'ja', 'ko', 'es'])
  expect(LANGUAGES.map((l) => l.name)).toEqual(['English', 'Nederlands', '中文', 'Français', 'Deutsch', 'Italiano', '日本語', '한국어', 'Español'])
  expect(LANGUAGES.find((l) => l.code === 'zh')?.tag).toBe('zh-CN')
  for (const l of LANGUAGES) expect(l.flag, l.code).toMatch(/\.svg$|^data:image\/svg\+xml/)
})

test('resolution: an explicit stored choice, otherwise English', () => {
  expect(resolveLanguage('ja')).toBe('ja')
  expect(resolveLanguage(null)).toBe('en')
  expect(resolveLanguage(undefined)).toBe('en')
  expect(resolveLanguage('xx')).toBe('en')
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
