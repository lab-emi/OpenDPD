/** Geometry for viewport recovery, not a signal metric or a data filter. */
export interface RecoveryTrace {
  x: ArrayLike<number>
  y: ArrayLike<number>
  mode?: string
  visible?: boolean | 'legendonly'
  connectgaps?: boolean
}

export interface PlotRecoverySettings { enabled: boolean; emptyThreshold: number }
export interface PlotRecoveryState { pendingSince?: number }
export const DEFAULT_PLOT_RECOVERY: PlotRecoverySettings = { enabled: true, emptyThreshold: .85 }
export const PLOT_RECOVERY_IDLE_MS = 800

interface Rectangle { left: number; right: number; bottom: number; top: number }

/** Liang–Barsky clipping: a line can cross the view even with both samples outside. */
function crosses(rect: Rectangle, x0: number, y0: number, x1: number, y1: number): boolean {
  let start = 0, end = 1
  const dx = x1 - x0, dy = y1 - y0
  const p = [-dx, dx, -dy, dy], q = [x0 - rect.left, rect.right - x0, y0 - rect.bottom, rect.top - y0]
  for (let i = 0; i < 4; i++) {
    if (p[i] === 0) { if (q[i]! < 0) return false; continue }
    const t = q[i]! / p[i]!
    if (p[i]! < 0) start = Math.max(start, t)
    else end = Math.min(end, t)
    if (start > end) return false
  }
  return true
}

/**
 * Inspect all finite, visible sample pairs once, after input settles. Background
 * between a thin curve or sparse markers is NOT empty area: only the viewport
 * outside the data's bounding rectangle counts. A completely missed trace also
 * recovers. Constant axes are handled by visibility, not a zero-area rectangle.
 */
export function shouldRecoverPlot(
  view: { x: readonly number[]; y: readonly number[] },
  traces: readonly RecoveryTrace[],
  emptyThreshold = DEFAULT_PLOT_RECOVERY.emptyThreshold,
): boolean {
  if (view.x.length !== 2 || view.y.length !== 2 || ![...view.x, ...view.y].every(Number.isFinite)) return false
  const rect = { left: Math.min(...view.x), right: Math.max(...view.x), bottom: Math.min(...view.y), top: Math.max(...view.y) }
  if (rect.right <= rect.left || rect.top <= rect.bottom) return false
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity
  let visible = false, hasData = false
  for (const trace of traces) {
    if (trace.visible === false || trace.visible === 'legendonly' || trace.mode === 'none') continue
    const lines = trace.mode === undefined || trace.mode.includes('lines')
    let previousX = NaN, previousY = NaN
    for (let i = 0, count = Math.min(trace.x.length, trace.y.length); i < count; i++) {
      const x = trace.x[i]!, y = trace.y[i]!
      if (!Number.isFinite(x) || !Number.isFinite(y)) {
        if (!trace.connectgaps) { previousX = NaN; previousY = NaN }
        continue
      }
      hasData = true
      minX = Math.min(minX, x); maxX = Math.max(maxX, x)
      minY = Math.min(minY, y); maxY = Math.max(maxY, y)
      if (!visible) {
        visible = (x >= rect.left && x <= rect.right && y >= rect.bottom && y <= rect.top)
          || (lines && Number.isFinite(previousX) && crosses(rect, previousX, previousY, x, y))
      }
      previousX = x; previousY = y
    }
  }
  // Empty, unavailable, or deliberately hidden data must not cause a fit loop.
  if (!hasData) return false
  if (!visible) return true
  const coverage = (low: number, high: number, min: number, max: number) => min === max
    ? 1 : Math.max(0, Math.min(high, max) - Math.max(low, min)) / (high - low)
  const empty = 1 - coverage(rect.left, rect.right, minX, maxX) * coverage(rect.bottom, rect.top, minY, maxY)
  const threshold = Number.isFinite(emptyThreshold) && emptyThreshold > 0 && emptyThreshold < 1 ? emptyThreshold : DEFAULT_PLOT_RECOVERY.emptyThreshold
  return empty > threshold
}
