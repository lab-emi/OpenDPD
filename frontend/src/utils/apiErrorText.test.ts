import { ApiConnectionError, ApiError } from '@/api/client'
import { BATCH_SAMPLE_LIMIT, describeApiError, explainGenerateError, plainMessage } from './apiErrorText'

// Envelopes as the local API answered POST /signal-generator/batches for 5 x 983,040 samples and for the same configuration twice.
const tooMany = new ApiError(422, 'invalid_request', 'request validation failed', [{ field: '', message: 'Value error, Select at most 4,000,000 samples across all presets.' }])
const twice = new ApiError(422, 'invalid_request', 'request validation failed', [{ field: '', message: 'Value error, Select each preset only once per dataset.' }])
// The hosted Studio's mutation budget (opendpd/web/mutation_policy.py) answers with these two.
const storage = new ApiError(413, 'workspace_limit', 'This operation and its exports would exceed temporary storage. Use fewer samples.')
const hostedGeneric = new ApiError(422, 'invalid_request', 'Check the signal parameters, sample count and required dataset metadata.')

test('the pydantic class prefix is dropped and only that', () => {
  expect(plainMessage('Value error, Carrier 1: x')).toBe('Carrier 1: x')
  expect(plainMessage('Assertion error, y')).toBe('y')
  expect(plainMessage('A value error, is not stripped')).toBe('A value error, is not stripped')
})

test('describeApiError prefers the details over the generic summary and keeps the hint', () => {
  expect(describeApiError(tooMany)).toEqual(['Select at most 4,000,000 samples across all presets.'])
  expect(describeApiError(new ApiError(422, 'invalid_signal', 'Value error, plain'))).toEqual(['plain'])
  expect(describeApiError(new ApiError(409, 'busy', 'Workspace busy', [], 'Wait a moment.'))).toEqual(['Workspace busy', 'Wait a moment.'])
  expect(describeApiError(new ApiConnectionError('https://api.example.test'))).toEqual(['The browser could not reach https://api.example.test.'])
})

test('the batch refusals are explained with the numbers of the selection', () => {
  const selection = { signals: 5, samples: 4_915_200 }
  expect(BATCH_SAMPLE_LIMIT).toBe(4_000_000)
  const sized = 'The 5 selected signals hold 4,915,200 I/Q samples together, but one dataset is limited to 4,000,000.'
  for (const error of [tooMany, hostedGeneric]) expect((explainGenerateError(error, selection) as Error).message).toContain(sized)
  expect(explainGenerateError(tooMany, selection)).not.toBeInstanceOf(ApiError)
  expect((explainGenerateError(storage, { signals: 2, samples: 1_966_080 }) as Error).message).toMatch(/^This server’s temporary workspace cannot hold 1,966,080 I\/Q samples together with their exports\./)
  expect((explainGenerateError(twice, selection) as Error).message).toBe('Two selected signals have identical settings, but a dataset holds each signal once. Change a setting of one of them or remove it.')
})

test('other errors, and the hosted generic 422 of a selection within the limit, are left as the server sent them', () => {
  expect(explainGenerateError(hostedGeneric, { signals: 2, samples: 1_000_000 })).toBe(hostedGeneric)
  const other = new ApiError(500, 'internal_error', 'Server error')
  expect(explainGenerateError(other, { signals: 1, samples: 1 })).toBe(other)
  const plain = new Error('boom')
  expect(explainGenerateError(plain, { signals: 1, samples: 1 })).toBe(plain)
  expect(explainGenerateError(null, { signals: 1, samples: 1 })).toBeNull()
})
