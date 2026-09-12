import { afterEach, expect, test } from 'vitest'
import { LANGUAGES, datasetLabel, formatNumber, message, phaseLabel, resetLanguage, setLanguage, t } from './index'

const catalogues = import.meta.glob<Record<string, string>>('../../../opendpd/studio/locales/*.json', { eager: true, import: 'default' })
const placeholders = (text: string) => [...text.matchAll(/\{(\w+)\}/g)].map((m) => m[1]).sort()
afterEach(resetLanguage)

test('every shared presentation catalogue preserves all keys and interpolation variables', () => {
  const en = catalogues['../../../opendpd/studio/locales/en.json']!
  expect(Object.keys(catalogues)).toHaveLength(LANGUAGES.length)
  for (const [path, catalogue] of Object.entries(catalogues)) {
    expect(Object.keys(catalogue).sort(), path).toEqual(Object.keys(en).sort())
    for (const [key, source] of Object.entries(en)) {
      expect(catalogue[key]?.trim(), `${path}: ${key}`).toBeTruthy()
      expect(placeholders(catalogue[key]!), `${path}: ${key}`).toEqual(placeholders(source))
    }
  }
})

test.each(LANGUAGES.map((l) => l.code))('%s translates generated diagnostics, plot labels and progress without changing captured values', async (language) => {
  await setLanguage(language)
  const title = message('Input and output are time aligned')
  const diagnostic = message('cross-correlation peaks at +1.25 samples (correlation 0.98)')
  expect(diagnostic).toContain('+1.25')
  expect(diagnostic).toContain('0.98')
  expect(diagnostic).not.toMatch(/\{\w+\}/)
  if (language !== 'en') {
    expect(title).not.toBe('Input and output are time aligned')
    expect(message('PA model output')).not.toBe('PA model output')
    expect(message('Hidden state size of the backbone')).not.toBe('Hidden state size of the backbone')
    expect(message('3 epochs is a smoke/demo run, not a benchmark result')).not.toContain('not a benchmark result')
    expect(message('ACLR left')).not.toContain('left')
    const method = message('dataset-doctor-v1: small-signal complex gain is +4.88 dB at +0.6 degrees. Gain/phase fit uses the lowest 40% of input amplitudes after integer alignment; this includes the capture/attenuator scale, not calibrated PA gain.')
    expect(method).toContain('+4.88')
    expect(method).not.toContain('Gain/phase fit uses')
    expect(datasetLabel({ display_name: 'Example (built-in, measured)', source: { kind: 'builtin', name: 'Example' } })).not.toContain('built-in')
  }
  expect(datasetLabel({ display_name: 'My custom experiment', source: { kind: 'imported' } })).toBe('My custom experiment')
  expect(phaseLabel('validation_probe')).toBe(message('Validation preview'))
  expect(phaseLabel('test_probe')).toBe(message('Test preview'))
  expect(phaseLabel('complete')).toBe(message('Completed'))
  expect(t('live.geometry', { samples: formatNumber(1024), rate: '122.88 MSa/s' })).not.toMatch(/\{\w+\}/)
  expect(message('my_custom_capture_01')).toBe('my_custom_capture_01')
  expect(message('opendpd run --config my.json')).toBe('opendpd run --config my.json')
})

test('a newer selection wins even when multiple language loads overlap', async () => {
  await Promise.all([setLanguage('nl'), setLanguage('it'), setLanguage('zh')])
  expect(document.documentElement.lang).toBe('zh-CN')
  expect(t('language.label')).toBe('语言')
})
