import '@testing-library/jest-dom/vitest'
import { cleanup } from '@testing-library/react'
import { afterEach, vi } from 'vitest'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

// jsdom lacks these browser APIs that MUI/Plotly touch.
if (!window.matchMedia) {
  window.matchMedia = (query: string) =>
    ({ matches: false, media: query, onchange: null, addListener: () => undefined, removeListener: () => undefined, addEventListener: () => undefined, removeEventListener: () => undefined, dispatchEvent: () => false }) as MediaQueryList
}
if (!('ResizeObserver' in window)) {
  class RO {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
  Object.defineProperty(window, 'ResizeObserver', { value: RO })
}
if (!('randomUUID' in crypto)) {
  Object.defineProperty(crypto, 'randomUUID', { value: () => `uuid-${Math.random().toString(16).slice(2)}` })
}
