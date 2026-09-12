/** Reserve two web read slots; a write (especially Stop) can use the third API slot. */
let active = 0
const waiting: Array<() => void> = []

export async function boundedRead<T>(operation: (signal: AbortSignal) => Promise<T>, signal?: AbortSignal, timeoutMs = 30_000): Promise<T> {
  signal?.throwIfAborted()
  await new Promise<void>((resolve, reject) => {
    const start = () => {
      signal?.removeEventListener('abort', abort)
      active++
      resolve()
    }
    const abort = () => {
      const index = waiting.indexOf(start)
      if (index >= 0) waiting.splice(index, 1)
      reject(signal?.reason)
    }
    if (active < 2) start()
    else {
      waiting.push(start)
      signal?.addEventListener('abort', abort, { once: true })
    }
  })
  const controller = new AbortController()
  const abort = () => controller.abort(signal?.reason)
  signal?.addEventListener('abort', abort, { once: true })
  const timer = window.setTimeout(() => controller.abort(new DOMException('Compute server request timed out', 'TimeoutError')), timeoutMs)
  try {
    signal?.throwIfAborted()
    return await operation(controller.signal)
  } finally {
    window.clearTimeout(timer)
    signal?.removeEventListener('abort', abort)
    active--
    waiting.shift()?.()
  }
}
