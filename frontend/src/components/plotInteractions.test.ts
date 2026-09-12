import { afterEach, describe, expect, it, vi } from 'vitest'
import { attachPlotInteractions, readViewport, transformViewport, wheelPixels, type PlotElement, type PlotViewport } from './plotInteractions'
import { DEFAULT_PLOT_RECOVERY, PLOT_RECOVERY_IDLE_MS, type PlotRecoveryState } from './plotRecovery'

const initial = (): PlotViewport => ({ x: [0, 100], y: [-10, 10], autoX: true, autoY: true, dragmode: 'pan' })
const microtasks = async () => { for (let i = 0; i < 8; i++) await Promise.resolve() }

function setup(recovery?: Parameters<typeof attachPlotInteractions>[4]) {
  let serial = 0
  const frames = new Map<number, FrameRequestCallback>()
  vi.stubGlobal('requestAnimationFrame', (callback: FrameRequestCallback) => { frames.set(++serial, callback); return serial })
  vi.stubGlobal('cancelAnimationFrame', (id: number) => frames.delete(id))
  const frame = async () => {
    const callbacks = [...frames.values()]; frames.clear()
    callbacks.forEach((callback) => callback(performance.now()))
    await microtasks()
  }
  const element = document.createElement('div') as PlotElement
  element.tabIndex = 0
  const area = document.createElement('div')
  area.className = 'nsewdrag'
  area.getBoundingClientRect = () => ({ left: 100, top: 50, width: 400, height: 200, right: 500, bottom: 250, x: 100, y: 50, toJSON: () => ({}) })
  element.append(area); document.body.append(element)
  element['_fullLayout'] = { xaxis: { range: [0, 100], autorange: true }, yaxis: { range: [-10, 10], autorange: true }, dragmode: 'pan' }
  const listeners = new Map<string, () => void>()
  element.on = (event, listener) => { listeners.set(event, listener) }
  element.removeListener = (event) => { listeners.delete(event) }
  const onView = vi.fn(), onError = vi.fn()
  const api = {
    relayout: vi.fn(async (_element: HTMLElement, update: object) => {
      const values = update as Record<string, unknown>
      element['_fullLayout'] = {
        xaxis: { range: values['xaxis.range'] as [number, number], autorange: !!values['xaxis.autorange'] },
        yaxis: { range: values['yaxis.range'] as [number, number], autorange: !!values['yaxis.autorange'] },
        dragmode: values.dragmode as string,
      }
    }),
    Plots: { resize: vi.fn(async () => {}) },
  }
  const controls = attachPlotInteractions(element, api, onView, onError, recovery)
  const wheel = (options: WheelEventInit = {}) => {
    const event = new WheelEvent('wheel', { bubbles: true, cancelable: true, clientX: 200, clientY: 100, ...options })
    area.dispatchEvent(event)
    return event
  }
  const gesture = (type: string, scale: number) => {
    const event = new Event(type, { bubbles: true, cancelable: true })
    Object.assign(event, { scale, clientX: 200, clientY: 100 }); area.dispatchEvent(event)
  }
  const key = (value: string, options: KeyboardEventInit = {}) => element.dispatchEvent(new KeyboardEvent('keydown', { key: value, bubbles: true, cancelable: true, ...options }))
  return { element, area, frames, frame, api, controls, wheel, gesture, key, onView, onError, emit: (event: string) => listeners.get(event)?.() }
}

afterEach(() => { document.body.replaceChildren(); vi.unstubAllGlobals(); vi.useRealTimers() })

describe('plot viewport input', () => {
  it('keeps the cursor anchor fixed during zoom and supports reversed axes', () => {
    const zoomed = transformViewport(initial(), 0, 0, .5, .25, .75)
    expect(zoomed.x).toEqual([12.5, 62.5])
    expect(zoomed.y).toEqual([-2.5, 7.5])
    const reversed = transformViewport({ ...initial(), x: [100, 0] }, .1, 0)
    expect(reversed.x).toEqual([90, -10])
    expect(transformViewport(initial(), 0, 0, Infinity)).toEqual(initial())
    expect(transformViewport(initial(), 0, 0, 0)).toEqual(initial())
  })

  it('normalizes pixel, line, page and Shift-wheel input', () => {
    expect(wheelPixels({ deltaMode: 0, deltaX: .5, deltaY: .25, shiftKey: false }, 400, 200)).toEqual({ x: .5, y: .25 })
    expect(wheelPixels({ deltaMode: 1, deltaX: 0, deltaY: 3, shiftKey: true }, 400, 200)).toEqual({ x: 48, y: 0 })
    expect(wheelPixels({ deltaMode: 2, deltaX: 1, deltaY: -1, shiftKey: false }, 400, 200)).toEqual({ x: 400, y: -200 })
  })

  it('pans diagonally without changing the scale', async () => {
    const h = setup()
    expect(h.wheel({ deltaX: 40, deltaY: 20 }).defaultPrevented).toBe(true)
    await h.frame()
    expect(readViewport(h.element)).toMatchObject({ x: [10, 110], y: [-12, 8], autoX: false, autoY: false })
    await h.controls.dispose()
  })

  it('coalesces high-rate fractional input without losing displacement', async () => {
    const h = setup()
    for (let i = 0; i < 100; i++) h.wheel({ deltaX: .25 })
    expect(h.frames.size).toBe(1)
    await h.frame()
    expect(h.api.relayout).toHaveBeenCalledTimes(1)
    expect(readViewport(h.element)?.x).toEqual([6.25, 106.25])
    await h.controls.dispose()
  })

  it('bounds the redraw queue while an earlier draw is still in progress', async () => {
    const h = setup()
    let finish!: () => void
    h.api.relayout.mockImplementationOnce(() => new Promise<void>((resolve) => { finish = resolve }))
    h.wheel({ deltaX: 4 }); await h.frame()
    for (let i = 0; i < 50; i++) h.wheel({ deltaX: 4 })
    await h.frame()
    expect(h.api.relayout).toHaveBeenCalledTimes(1)
    finish(); await microtasks(); await h.frame()
    expect(h.api.relayout).toHaveBeenCalledTimes(2)
    expect(readViewport(h.element)?.x).toEqual([51, 151])
    await h.controls.dispose()
  })

  it('zooms at the pointer for Ctrl, Cmd and the mouse zoom tool', async () => {
    for (const options of [{ ctrlKey: true }, { metaKey: true }, {}]) {
      const h = setup()
      if (!('ctrlKey' in options) && !('metaKey' in options)) h.element['_fullLayout']!.dragmode = 'zoom'
      h.wheel({ ...options, deltaY: Math.log(.5) / .004 }); await h.frame()
      expect(readViewport(h.element)?.x).toEqual([12.5, 62.5])
      expect(readViewport(h.element)?.y).toEqual([-2.5, 7.5])
      await h.controls.dispose()
    }
  })

  it('leaves page scrolling and browser zoom available outside the axes and with Alt', async () => {
    const h = setup()
    expect(h.wheel({ deltaY: 40, altKey: true }).defaultPrevented).toBe(false)
    expect(h.wheel({ deltaY: 40, ctrlKey: true, clientX: 20 }).defaultPrevented).toBe(false)
    expect(h.wheel().defaultPrevented).toBe(false)
    await h.frame()
    expect(h.api.relayout).not.toHaveBeenCalled()
    await h.controls.dispose()
  })

  it('handles WebKit relative-scale gestures once even if wheel events accompany them', async () => {
    const h = setup()
    h.gesture('gesturestart', 1)
    h.gesture('gesturechange', 2)
    h.wheel({ deltaY: -50, ctrlKey: true })
    h.gesture('gestureend', 2)
    await h.frame()
    expect(readViewport(h.element)?.x).toEqual([12.5, 62.5])
    await h.controls.dispose()
  })

  it('supports keyboard pan, coarse pan, zoom, tool selection and fit', async () => {
    const h = setup()
    h.key('ArrowRight'); h.key('ArrowUp', { shiftKey: true }); h.key('+')
    await h.frame()
    expect(readViewport(h.element)?.x[0]).toBeCloseTo(13.333333333)
    expect(readViewport(h.element)?.y[0]).toBeCloseTo(-4.333333333)
    h.key('z'); await h.frame()
    expect(readViewport(h.element)?.dragmode).toBe('zoom')
    h.key('p'); h.key('Home'); await h.frame()
    expect(readViewport(h.element)).toMatchObject({ autoX: true, autoY: true, dragmode: 'pan' })
    const calls = h.api.relayout.mock.calls.length
    h.key('ArrowRight', { ctrlKey: true })
    h.area.dispatchEvent(new KeyboardEvent('keydown', { key: '+', bubbles: true }))
    await h.frame()
    expect(h.api.relayout).toHaveBeenCalledTimes(calls)
    await h.controls.dispose()
  })

  it('waits for an in-flight draw on disposal and cancels queued work and listeners', async () => {
    const h = setup()
    let finish!: () => void
    h.api.relayout.mockImplementationOnce(() => new Promise<void>((resolve) => { finish = resolve }))
    h.wheel({ deltaX: 4 }); await h.frame()
    h.wheel({ deltaX: 4 }); h.controls.resize()
    let stopped = false
    const stopping = h.controls.dispose().then(() => { stopped = true })
    await microtasks()
    expect(stopped).toBe(false)
    finish(); await stopping; await h.frame()
    expect(h.api.relayout).toHaveBeenCalledTimes(1)
    expect(h.api.Plots.resize).not.toHaveBeenCalled()
    expect(h.wheel({ deltaX: 10 }).defaultPrevented).toBe(false)
    expect(h.onView.mock.lastCall?.[0].x).toEqual([2, 102])
  })

  it('serializes resizing with queued input and reports rendering failures', async () => {
    const h = setup()
    h.controls.resize(); h.controls.resize(); h.wheel({ deltaX: 4 })
    await h.frame()
    expect(h.api.Plots.resize).toHaveBeenCalledTimes(1)
    expect(h.api.relayout).toHaveBeenCalledTimes(1)
    h.api.relayout.mockRejectedValueOnce(new Error('draw failed'))
    h.wheel({ deltaX: 4 }); await h.frame()
    expect(h.onError).toHaveBeenCalledWith(expect.objectContaining({ message: 'draw failed' }))
    await h.controls.dispose()
  })
})

function protection(enabled = true) {
  return { settings: { ...DEFAULT_PLOT_RECOVERY, enabled }, state: {} as PlotRecoveryState, traces: () => [{ x: [0, 100], y: [-10, 10], mode: 'lines' }] }
}

describe('elastic recovery scheduling', () => {
  it('fits an off-data view only after input settles, then stays idle', async () => {
    vi.useFakeTimers()
    const h = setup(protection())
    h.wheel({ deltaX: 800 }); await h.frame()
    await vi.advanceTimersByTimeAsync(PLOT_RECOVERY_IDLE_MS - 1)
    expect(h.api.relayout).toHaveBeenCalledTimes(1)
    await vi.advanceTimersByTimeAsync(1); await h.frame()
    expect(readViewport(h.element)).toMatchObject({ autoX: true, autoY: true })
    await vi.advanceTimersByTimeAsync(5000); await h.frame()
    expect(h.api.relayout).toHaveBeenCalledTimes(2)
    await h.controls.dispose()
  })

  it('debounces continued scrolling instead of fighting the gesture', async () => {
    vi.useFakeTimers()
    const h = setup(protection())
    h.wheel({ deltaX: 800 }); await h.frame()
    await vi.advanceTimersByTimeAsync(600)
    h.wheel({ deltaX: 4 }); await h.frame()
    await vi.advanceTimersByTimeAsync(PLOT_RECOVERY_IDLE_MS - 1)
    expect(readViewport(h.element)?.autoX).toBe(false)
    await vi.advanceTimersByTimeAsync(1); await h.frame()
    expect(readViewport(h.element)?.autoX).toBe(true)
    await h.controls.dispose()
  })

  it('waits for a held pointer to be released, even outside the plot', async () => {
    vi.useFakeTimers()
    const h = setup(protection())
    h.wheel({ deltaX: 800 }); await h.frame()
    h.area.dispatchEvent(new MouseEvent('pointerdown', { bubbles: true, clientX: 200, clientY: 100 }))
    await vi.advanceTimersByTimeAsync(2000); await h.frame()
    expect(readViewport(h.element)?.autoX).toBe(false)
    document.dispatchEvent(new Event('pointerup'))
    await vi.advanceTimersByTimeAsync(PLOT_RECOVERY_IDLE_MS); await h.frame()
    expect(readViewport(h.element)?.autoX).toBe(true)
    await h.controls.dispose()
  })

  it('waits until a pinch ends', async () => {
    vi.useFakeTimers()
    const h = setup(protection())
    h.gesture('gesturestart', 1); h.gesture('gesturechange', .01); await h.frame()
    await vi.advanceTimersByTimeAsync(2000); await h.frame()
    expect(readViewport(h.element)?.autoX).toBe(false)
    h.gesture('gestureend', .01); await h.frame()
    await vi.advanceTimersByTimeAsync(PLOT_RECOVERY_IDLE_MS); await h.frame()
    expect(readViewport(h.element)?.autoX).toBe(true)
    await h.controls.dispose()
  })

  it('honors disabling and carries pending recovery across controller replacement', async () => {
    vi.useFakeTimers()
    const config = protection(false), h = setup(config)
    h.wheel({ deltaX: 800 }); await h.frame()
    await vi.advanceTimersByTimeAsync(2000); await h.frame()
    expect(readViewport(h.element)?.autoX).toBe(false)
    await h.controls.dispose()
    const resumed = attachPlotInteractions(h.element, h.api, h.onView, h.onError, { ...config, settings: { ...config.settings, enabled: true } })
    await vi.advanceTimersByTimeAsync(1); await h.frame()
    expect(readViewport(h.element)?.autoX).toBe(true)
    await resumed.dispose()
  })

  it('handles native Plotly pan/zoom and removes its timer on disposal', async () => {
    vi.useFakeTimers()
    const h = setup(protection())
    h.element['_fullLayout']!.xaxis = { range: [1000, 1100], autorange: false }
    h.emit('plotly_relayout')
    await vi.advanceTimersByTimeAsync(PLOT_RECOVERY_IDLE_MS); await h.frame()
    expect(readViewport(h.element)?.autoX).toBe(true)
    h.wheel({ deltaX: 800 }); await h.frame()
    const calls = h.api.relayout.mock.calls.length
    await h.controls.dispose()
    await vi.advanceTimersByTimeAsync(2000); await h.frame()
    expect(h.api.relayout).toHaveBeenCalledTimes(calls)
  })

  it('waits for an in-flight draw before checking and fitting', async () => {
    vi.useFakeTimers()
    const h = setup(protection()), normal = h.api.relayout.getMockImplementation()!
    let finish!: () => void
    h.api.relayout.mockImplementationOnce(async (element, update) => {
      await new Promise<void>((resolve) => { finish = resolve })
      await normal(element, update)
    })
    h.wheel({ deltaX: 800 }); await h.frame()
    await vi.advanceTimersByTimeAsync(2000)
    expect(h.api.relayout).toHaveBeenCalledTimes(1)
    finish(); await microtasks()
    await vi.advanceTimersByTimeAsync(1); await h.frame()
    expect(readViewport(h.element)?.autoX).toBe(true)
    await h.controls.dispose()
  })
})
