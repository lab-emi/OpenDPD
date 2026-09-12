import { getContrastRatio } from '@mui/material/styles'
import { expect, test } from 'vitest'
import { colorsFor } from './index'

test.each(['light', 'dark'] as const)('%s semantic text and controls meet contrast targets on their surfaces', mode => {
  const c = colorsFor(mode)
  for (const surface of [c.background, c.surface, c.surfaceMuted, c.selected]) {
    for (const text of [c.textPrimary, c.textSecondary, c.primary, ...Object.values(c.status), ...Object.values(c.evidence)]) {
      expect(getContrastRatio(text, surface), `${text} on ${surface}`).toBeGreaterThanOrEqual(4.5)
    }
  }
  expect(getContrastRatio(c.primaryContrast, c.primary)).toBeGreaterThanOrEqual(4.5)
  expect(getContrastRatio(c.inputBorder, c.surface)).toBeGreaterThanOrEqual(3)
  for (const trace of c.chart) expect(getContrastRatio(trace, c.surface)).toBeGreaterThanOrEqual(3)
})
