import { PLOT_RECOVERY_IDLE_MS, shouldRecoverPlot, type PlotRecoverySettings, type PlotRecoveryState, type RecoveryTrace } from './plotRecovery'

/** Shared input handling for the app's numeric, two-dimensional Plotly charts. */
export type Range = [number, number]
export interface PlotViewport {
  x: Range
  y: Range
  autoX: boolean
  autoY: boolean
  dragmode: 'pan' | 'zoom'
}

interface Axis {
  range: Range
  autorange: boolean
}

export interface PlotElement extends HTMLDivElement {
  // Plotly's resolved axes include autoranges and scale-anchor adjustments.
  _fullLayout?: { xaxis?: Axis; yaxis?: Axis; dragmode?: string }
  _fullData?: RecoveryTrace[]
  on?: (event: string, listener: () => void) => void
  removeListener?: (event: string, listener: () => void) => void
}

export interface PlotInteractionApi {
  relayout: (element: HTMLElement, update: object) => Promise<unknown>
  Plots: { resize: (element: HTMLElement) => Promise<unknown> }
}

interface Bounds { left: number; top: number; width: number; height: number }
interface Gesture extends Event { scale: number; clientX?: number; clientY?: number }
interface TouchPinch {
  ids: number[]
  view: PlotViewport
  rect: Bounds
  x: number
  y: number
  distance: number
}

export function readViewport(element: PlotElement): PlotViewport | undefined {
  const layout = element['_fullLayout']
  if (!layout?.xaxis?.range || !layout.yaxis?.range) return
  return {
    x: [...layout.xaxis.range], y: [...layout.yaxis.range],
    autoX: !!layout.xaxis.autorange, autoY: !!layout.yaxis.autorange,
    dragmode: layout.dragmode === 'zoom' ? 'zoom' : 'pan',
  }
}

/** DOM_DELTA_LINE and DOM_DELTA_PAGE occur with mouse wheels and accessibility settings. */
export function wheelPixels(event: Pick<WheelEvent, 'deltaMode' | 'deltaX' | 'deltaY' | 'shiftKey'>, width: number, height: number) {
  const unitX = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? width : 1
  const unitY = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? height : 1
  const x = event.deltaX * unitX, y = event.deltaY * unitY
  return event.shiftKey && x === 0 ? { x: y, y: 0 } : { x, y }
}

/** Transform ranges in data coordinates; a zoom keeps the point under the cursor fixed. */
export function transformViewport(view: PlotViewport, dx: number, dy: number, factor = 1, anchorX = .5, anchorY = .5): PlotViewport {
  const transform = (range: Range, delta: number, anchor: number): Range => {
    const span = range[1] - range[0]
    const pivot = range[0] + span * anchor
    return range.map((value) => pivot + (value - pivot) * factor + span * delta) as Range
  }
  const x = transform(view.x, dx, anchorX), y = transform(view.y, dy, anchorY)
  // Stop at floating-point resolution, without imposing data-dependent zoom limits.
  const valid = (range: Range) => range.every(Number.isFinite) && Math.abs(range[1] - range[0]) > Number.EPSILON * Math.max(...range.map(Math.abs), Number.MIN_VALUE) * 8
  return valid(x) && valid(y) ? { ...view, x, y, autoX: false, autoY: false } : view
}

export function viewportUpdate(view: PlotViewport) {
  return {
    'xaxis.range': [...view.x], 'yaxis.range': [...view.y],
    'xaxis.autorange': view.autoX, 'yaxis.autorange': view.autoY,
    dragmode: view.dragmode,
  }
}

/**
 * One pending target and one Plotly operation at a time. New input composes onto
 * the latest target, so a slow scatter redraw cannot create an event backlog.
 * Ordinary wheel/two-finger scrolling pans; pinch or Ctrl/Cmd+wheel zooms.
 * Selecting Plotly's zoom tool also makes an unmodified mouse wheel zoom.
 */
export function attachPlotInteractions(
  element: PlotElement,
  plotly: PlotInteractionApi,
  onView: (view: PlotViewport) => void,
  onError: (error: unknown) => void,
  recovery?: { settings: PlotRecoverySettings; state: PlotRecoveryState; traces: () => readonly RecoveryTrace[] },
) {
  let disposed = false, drawing = false, resizing = false
  let frame = 0, version = 0
  let inFlight: Promise<void> = Promise.resolve()
  let target: PlotViewport | undefined
  let gestureScale: number | undefined
  let gesturePoint: { x: number; y: number } | undefined
  let pointer: { x: number; y: number } | undefined
  let pointerDown = false
  let touchActive = false
  let touchPinch: TouchPinch | undefined
  let recoveryTimer: ReturnType<typeof setTimeout> | undefined
  let observed = readViewport(element)

  const cancelRecoveryTimer = () => { clearTimeout(recoveryTimer); recoveryTimer = undefined }
  const clearRecovery = () => { cancelRecoveryTimer(); if (recovery) recovery.state.pendingSince = undefined }
  const scheduleRecovery = () => {
    cancelRecoveryTimer()
    if (disposed || !recovery?.settings.enabled || recovery.state.pendingSince === undefined || pointerDown || touchActive || gestureScale !== undefined || drawing || target || resizing) return
    recoveryTimer = setTimeout(() => {
      recoveryTimer = undefined
      if (disposed || drawing || target || resizing || pointerDown || touchActive || gestureScale !== undefined) return
      const view = readViewport(element)
      clearRecovery()
      if (view && !(view.autoX && view.autoY) && shouldRecoverPlot(view, recovery.traces(), recovery.settings.emptyThreshold)) {
        update({ ...view, autoX: true, autoY: true })
      }
    }, Math.max(0, PLOT_RECOVERY_IDLE_MS - (Date.now() - recovery.state.pendingSince)))
  }
  const interacted = () => {
    if (recovery) recovery.state.pendingSince = Date.now()
    scheduleRecovery()
  }

  const bounds = (): Bounds | undefined => {
    const rect = element.querySelector('.nsewdrag')?.getBoundingClientRect()
    return rect && rect.width > 0 && rect.height > 0 ? rect : undefined
  }
  const inside = (x: number, y: number, rect: Bounds) => x >= rect.left && x <= rect.left + rect.width && y >= rect.top && y <= rect.top + rect.height
  const save = () => {
    const view = target ?? readViewport(element)
    if (view && !disposed) { observed = view; onView(view) }
  }
  const schedule = () => {
    if (!disposed && !drawing && !frame) frame = requestAnimationFrame(flush)
  }
  const flush = () => {
    frame = 0
    if (disposed || drawing || (!target && !resizing)) return
    drawing = true
    const currentVersion = version, next = target, resize = resizing
    resizing = false
    const draw = async () => {
      if (resize) await plotly.Plots.resize(element)
      if (!disposed && next) await plotly.relayout(element, viewportUpdate(next))
    }
    inFlight = draw().then(() => {
      if (disposed) return
      if (currentVersion === version) target = undefined
      save()
    }).catch((error: unknown) => {
      if (!disposed) { target = undefined; onError(error) }
    }).finally(() => {
      drawing = false
      if (target || resizing) schedule()
      else scheduleRecovery()
    })
  }
  const update = (view: PlotViewport) => {
    target = view
    version++
    save()
    schedule()
  }
  const transform = (dx: number, dy: number, factor = 1, x = .5, y = .5) => {
    const view = target ?? readViewport(element)
    if (view) { update(transformViewport(view, dx, dy, factor, x, y)); interacted() }
  }
  const wheel = (event: WheelEvent) => {
    const rect = bounds()
    if (event.altKey || !rect || !inside(event.clientX, event.clientY, rect)) return
    const pixels = wheelPixels(event, rect.width, rect.height)
    if (!Number.isFinite(pixels.x) || !Number.isFinite(pixels.y) || (!pixels.x && !pixels.y)) return
    event.preventDefault()
    event.stopPropagation()
    // WebKit can emit both gesture events and wheel events for one pinch.
    if (gestureScale !== undefined || touchActive) return
    const zoom = event.ctrlKey || event.metaKey || (target ?? readViewport(element))?.dragmode === 'zoom'
    if (zoom) {
      const delta = pixels.y || pixels.x
      transform(0, 0, Math.exp(Math.max(-.8, Math.min(.8, delta * .004))),
        (event.clientX - rect.left) / rect.width, 1 - (event.clientY - rect.top) / rect.height)
    } else transform(pixels.x / rect.width, -pixels.y / rect.height)
  }
  const move = (event: PointerEvent) => { pointer = { x: event.clientX, y: event.clientY } }
  const leave = () => { pointer = undefined }
  const focus = (event: PointerEvent) => {
    if (event.pointerType === 'touch') return
    const rect = bounds()
    if (rect && inside(event.clientX, event.clientY, rect)) {
      pointerDown = true
      cancelRecoveryTimer()
      element.focus({ preventScroll: true })
    }
  }
  const release = () => { if (pointerDown) { pointerDown = false; interacted() } }
  const touch = (event: TouchEvent) => {
    const points = Array.from(event.touches)
    const rect = bounds()
    // Own only touches starting on the plotting area. Controls, legends and
    // page pinch-zoom outside it retain their normal browser behavior.
    if (!touchActive) {
      if (event.type !== 'touchstart' || !rect || !(event.target instanceof Element)
          || !event.target.closest('.nsewdrag') || !points.every((p) => inside(p.clientX, p.clientY, rect))) return
      touchActive = true
    }
    // Do not let Plotly's touch drag and Safari's GestureEvent zoom both act on
    // the same fingers. Single-finger movement remains native page scrolling.
    event.stopPropagation()
    cancelRecoveryTimer()
    if (points.length !== 2 || event.type === 'touchcancel') {
      touchPinch = undefined
      if (points.length > 1) event.preventDefault()
      if (!points.length || event.type === 'touchcancel') { touchActive = false; interacted() }
      return
    }
    // A second finger that started on another chart or outside the axes belongs
    // to page zoom. Once claimed, fingers may move beyond the original bounds.
    if (!touchPinch && !points.every((p) => p.target instanceof Element
      && element.contains(p.target) && p.target.closest('.nsewdrag'))) return
    event.preventDefault()
    gestureScale = undefined; gesturePoint = undefined
    const [a, b] = points as [Touch, Touch]
    const x = (a.clientX + b.clientX) / 2, y = (a.clientY + b.clientY) / 2
    const distance = Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY)
    if (distance < 8 || !rect) return
    if (!touchPinch || !points.every((p) => touchPinch!.ids.includes(p.identifier))) {
      const view = target ?? readViewport(element)
      if (view) touchPinch = { ids: points.map((p) => p.identifier), view, rect, x, y, distance }
      return
    }
    const start = touchPinch
    const factor = start.distance / distance
    // Always solve from the gesture's initial geometry, avoiding accumulated
    // rounding/redraw drift. The data under the midpoint follows both fingers.
    update(transformViewport(start.view,
      (start.x - x) / start.rect.width * factor,
      (y - start.y) / start.rect.height * factor, factor,
      (start.x - start.rect.left) / start.rect.width,
      1 - (start.y - start.rect.top) / start.rect.height))
    interacted()
  }
  const gestureStart = (event: Event) => {
    if (touchActive) { if (touchPinch) event.preventDefault(); event.stopPropagation(); return }
    const gesture = event as Gesture, rect = bounds()
    if (!rect) return
    // Safari's trackpad gesture coordinates may be absent; use the last pointer
    // location, with the plot centre only when the event itself targets the plot.
    const x = gesture.clientX || pointer?.x || rect.left + rect.width / 2
    const y = gesture.clientY || pointer?.y || rect.top + rect.height / 2
    if (!inside(x, y, rect)) return
    event.preventDefault(); event.stopPropagation()
    gestureScale = Number.isFinite(gesture.scale) && gesture.scale > 0 ? gesture.scale : 1
    gesturePoint = { x: (x - rect.left) / rect.width, y: 1 - (y - rect.top) / rect.height }
    cancelRecoveryTimer()
  }
  const gestureChange = (event: Event) => {
    if (touchActive) { if (touchPinch) event.preventDefault(); event.stopPropagation(); return }
    if (gestureScale === undefined || !gesturePoint) return
    event.preventDefault(); event.stopPropagation()
    const scale = (event as Gesture).scale
    if (!Number.isFinite(scale) || scale <= 0) return
    transform(0, 0, gestureScale / scale, gesturePoint.x, gesturePoint.y)
    gestureScale = scale
  }
  const gestureEnd = (event: Event) => {
    if (gestureScale === undefined) return
    gestureChange(event)
    gestureScale = undefined; gesturePoint = undefined
    scheduleRecovery()
  }
  const keyboard = (event: KeyboardEvent) => {
    if (event.target !== element || event.ctrlKey || event.metaKey || event.altKey) return
    const view = target ?? readViewport(element)
    if (!view) return
    const step = event.shiftKey ? .2 : .05
    switch (event.key) {
      case 'ArrowLeft': transform(-step, 0); break
      case 'ArrowRight': transform(step, 0); break
      case 'ArrowUp': transform(0, step); break
      case 'ArrowDown': transform(0, -step); break
      case '+': case '=': transform(0, 0, 1 / 1.2); break
      case '-': case '_': transform(0, 0, 1.2); break
      case 'Home': case '0': clearRecovery(); update({ ...view, autoX: true, autoY: true }); break
      case 'p': case 'P': update({ ...view, dragmode: 'pan' }); break
      case 'z': case 'Z': update({ ...view, dragmode: 'zoom' }); break
      default: return
    }
    event.preventDefault(); event.stopPropagation()
  }
  const relayout = () => {
    if (drawing) return
    target = undefined
    const view = readViewport(element)
    if (view?.autoX && view.autoY) clearRecovery()
    else if (view && (!observed || view.x.some((v, i) => v !== observed!.x[i]) || view.y.some((v, i) => v !== observed!.y[i]))) interacted()
    save()
  }
  const restyle = () => { if (!drawing) interacted() }
  element.addEventListener('wheel', wheel, { passive: false, capture: true })
  const touchEvents = ['touchstart', 'touchmove', 'touchend', 'touchcancel'] as const
  for (const name of touchEvents) element.addEventListener(name, touch, { passive: false, capture: true })
  element.addEventListener('gesturestart', gestureStart, { passive: false, capture: true })
  element.addEventListener('gesturechange', gestureChange, { passive: false, capture: true })
  element.addEventListener('gestureend', gestureEnd, { passive: false, capture: true })
  element.addEventListener('pointermove', move)
  element.addEventListener('pointerleave', leave)
  element.addEventListener('pointerdown', focus)
  document.addEventListener('pointerup', release)
  document.addEventListener('pointercancel', release)
  element.addEventListener('keydown', keyboard)
  element.on?.('plotly_relayout', relayout)
  element.on?.('plotly_restyle', restyle)
  scheduleRecovery()

  return {
    resize() { resizing = true; schedule() },
    dispose() {
      save()
      disposed = true
      cancelRecoveryTimer()
      cancelAnimationFrame(frame)
      element.removeEventListener('wheel', wheel, true)
      for (const name of touchEvents) element.removeEventListener(name, touch, true)
      element.removeEventListener('gesturestart', gestureStart, true)
      element.removeEventListener('gesturechange', gestureChange, true)
      element.removeEventListener('gestureend', gestureEnd, true)
      element.removeEventListener('pointermove', move)
      element.removeEventListener('pointerleave', leave)
      element.removeEventListener('pointerdown', focus)
      document.removeEventListener('pointerup', release)
      document.removeEventListener('pointercancel', release)
      element.removeEventListener('keydown', keyboard)
      element.removeListener?.('plotly_relayout', relayout)
      element.removeListener?.('plotly_restyle', restyle)
      return inFlight
    },
  }
}
