import { boundedRead } from './read-queue'

test('two slow reads leave room for Stop; aborted queued reads never start', async () => {
  const releases: Array<() => void> = []
  const operation = vi.fn(() => new Promise<void>((resolve) => releases.push(resolve)))
  const first = boundedRead(operation)
  const second = boundedRead(operation)
  const controller = new AbortController()
  const third = boundedRead(operation, controller.signal)
  const rejected = expect(third).rejects.toMatchObject({ name: 'AbortError' })
  await Promise.resolve()
  expect(operation).toHaveBeenCalledTimes(2)
  controller.abort()
  await rejected
  releases.forEach((resolve) => resolve())
  await Promise.all([first, second])
  await boundedRead(async () => 'recovered')
  expect(operation).toHaveBeenCalledTimes(2)
})

test('a stalled network read times out and releases its slot for recovery', async () => {
  vi.useFakeTimers()
  try {
    const pending = boundedRead((signal) => new Promise((_, reject) => signal.addEventListener('abort', () => reject(signal.reason))))
    const rejected = expect(pending).rejects.toMatchObject({ name: 'TimeoutError' })
    await vi.advanceTimersByTimeAsync(30_000)
    await rejected
    await expect(boundedRead(async () => 'reconnected')).resolves.toBe('reconnected')
  } finally { vi.useRealTimers() }
})
