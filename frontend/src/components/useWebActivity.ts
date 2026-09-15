import { useEffect } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { ApiError, reportWebActivity, WEB_MODE } from '@/api/client'

const INTERVAL = 60_000
const EVENTS = ['pointerdown', 'pointermove', 'keydown', 'wheel', 'touchstart'] as const

/** Coalesce actual foreground interactions, with no periodic keep-alive loop. */
export function useWebActivity(authenticated: boolean) {
  const qc = useQueryClient()
  useEffect(() => {
    if (!WEB_MODE || !authenticated) return
    const controller = new AbortController()
    let timer: number | undefined
    let stopped = false, busy = false, dirty = false, nextAllowed = 0
    const visible = () => document.visibilityState !== 'hidden'
    const schedule = () => {
      if (!stopped && !busy && dirty && visible() && timer === undefined) {
        timer = window.setTimeout(() => void flush(), Math.max(0, nextAllowed - Date.now()))
      }
    }
    const flush = async () => {
      timer = undefined
      if (stopped || !visible() || !dirty) return
      dirty = false; busy = true
      nextAllowed = Date.now() + INTERVAL
      try {
        const info = await reportWebActivity(controller.signal)
        if (!stopped && info.authenticated) qc.setQueryData(['session'], info)
      } catch (error) {
        // A failed heartbeat retries only after another user interaction. A
        // disconnected, untouched tab must never become a keep-alive loop.
        if (error instanceof ApiError) nextAllowed = Math.max(nextAllowed, Date.now() + error.retryAfterMs)
      } finally {
        busy = false
        schedule()
      }
    }
    const mark = () => { if (visible()) { dirty = true; schedule() } }
    for (const event of EVENTS) window.addEventListener(event, mark, { passive: true })
    window.addEventListener('focus', mark)
    document.addEventListener('visibilitychange', mark)
    mark()
    return () => {
      stopped = true; controller.abort(); window.clearTimeout(timer)
      for (const event of EVENTS) window.removeEventListener(event, mark)
      window.removeEventListener('focus', mark)
      document.removeEventListener('visibilitychange', mark)
    }
  }, [authenticated, qc])
}
