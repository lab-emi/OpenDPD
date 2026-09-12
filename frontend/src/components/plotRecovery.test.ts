import { describe, expect, it } from 'vitest'
import { shouldRecoverPlot, type RecoveryTrace } from './plotRecovery'

const view = { x: [0, 100], y: [-10, 10] }
const line: RecoveryTrace = { x: [0, 100], y: [-10, 10], mode: 'lines' }

describe('data-aware plot recovery', () => {
  it('preserves a normal sparse curve and a zoom into its connecting segment', () => {
    expect(shouldRecoverPlot(view, [line])).toBe(false)
    const detail = { x: [49, 51], y: [-.2, .2] }
    expect(shouldRecoverPlot(detail, [line])).toBe(false)
    expect(shouldRecoverPlot(detail, [{ ...line, mode: 'markers' }])).toBe(true)
  })

  it('fits a completely missed dataset or an extreme zoom out', () => {
    expect(shouldRecoverPlot({ x: [200, 300], y: [-10, 10] }, [line])).toBe(true)
    expect(shouldRecoverPlot({ x: [-500, 500], y: [-100, 100] }, [line])).toBe(true)
  })

  it('uses the chosen empty-area threshold, including on reversed axes', () => {
    const flat = { x: [0, 100], y: [0, 0], mode: 'lines' }
    const shifted = { x: [80, 180], y: [-1, 1] }
    expect(shouldRecoverPlot(shifted, [flat], .85)).toBe(false)
    expect(shouldRecoverPlot(shifted, [flat], .75)).toBe(true)
    expect(shouldRecoverPlot({ x: [180, 80], y: [1, -1] }, [flat], .75)).toBe(true)
  })

  it('does not mistake flat lines, one-point histories, or their normal margins for empty views', () => {
    expect(shouldRecoverPlot(view, [{ x: [0, 100], y: [3, 3], mode: 'lines' }])).toBe(false)
    expect(shouldRecoverPlot(view, [{ x: [50], y: [0], mode: 'markers' }])).toBe(false)
    expect(shouldRecoverPlot(view, [{ x: [50], y: [30], mode: 'markers' }])).toBe(true)
  })

  it('ignores legend-hidden traces and never fits an empty or entirely hidden plot', () => {
    const zoomedOut = { x: [0, 1000], y: [-10, 10] }
    const remote = { x: [0, 1000], y: [-10, 10], visible: 'legendonly' as const }
    expect(shouldRecoverPlot(zoomedOut, [line, remote])).toBe(true)
    expect(shouldRecoverPlot(zoomedOut, [remote, { ...line, visible: false }])).toBe(false)
    expect(shouldRecoverPlot(view, [])).toBe(false)
    expect(shouldRecoverPlot(view, [{ x: [NaN, Infinity], y: [0, 1] }])).toBe(false)
  })

  it('respects data gaps and actual segment intersection instead of just bounding boxes', () => {
    const detail = { x: [49, 51], y: [-.2, .2] }
    const gap = { x: [0, NaN, 100], y: [-10, NaN, 10], mode: 'lines' }
    expect(shouldRecoverPlot(detail, [gap])).toBe(true)
    expect(shouldRecoverPlot(detail, [{ ...gap, connectgaps: true }])).toBe(false)
    expect(shouldRecoverPlot({ x: [40, 50], y: [8, 9] }, [line])).toBe(true)
    expect(gap.x).toEqual([0, NaN, 100])
  })

  it('keeps a visible sample at the end of a dense trace without sampling it away', () => {
    const x = Float64Array.from({ length: 100_000 }, (_, i) => i + 1000)
    const y = x.slice()
    x[x.length - 1] = 0; y[y.length - 1] = 0
    expect(shouldRecoverPlot({ x: [-1, 1], y: [-1, 1] }, [{ x, y, mode: 'markers' }])).toBe(false)
  })

  it('skips unresolved or invalid axes', () => {
    expect(shouldRecoverPlot({ x: [0, Infinity], y: [-10, 10] }, [line])).toBe(false)
    expect(shouldRecoverPlot({ x: [0, 0], y: [-10, 10] }, [line])).toBe(false)
  })
})
