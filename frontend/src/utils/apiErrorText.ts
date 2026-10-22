import { ApiConnectionError, ApiError } from '@/api/client'
import { formatNumber, message as localize, t } from '@/i18n'

/** pydantic prefixes the text of a ValueError raised in a validator ("Value error, Carrier 1: …"); the user does not need the class name. */
const PYDANTIC_PREFIX = /^(?:Value|Assertion) error, /

export const plainMessage = (text: string) => text.trim().replace(PYDANTIC_PREFIX, '')

/**
 * A server refusal as readable lines. Two envelopes occur: pydantic validation errors (`request validation failed` plus `details[]`, whose
 * `field` is empty for model-level validators and whose text starts with "Value error, ") and plain refusals (`{ code: 'invalid_signal', message }`
 * without details). When details exist they carry the information, so the generic summary line is left out.
 */
export function describeApiError(error: unknown): string[] {
  if (error instanceof ApiError) {
    const details = error.details.flatMap(d => d.message ? [`${d.field ? d.field + ': ' : ''}${localize(plainMessage(d.message))}`] : [])
    const lines = details.length ? details : [localize(plainMessage(error.message))]
    return error.hint ? [...lines, localize(error.hint)] : lines
  }
  if (error instanceof ApiConnectionError) return [t('web.connection.body', { origin: error.origin })]
  return [error instanceof Error ? localize(plainMessage(error.message)) : String(error)]
}

/** The batch endpoint refuses more than this many I/Q samples in total (GeneratorBatchRequest). */
export const BATCH_SAMPLE_LIMIT = 4_000_000

/**
 * What the user should read when generating fails. The server's own words for the three batch refusals are replaced by messages that name the
 * numbers of the selection: the 422 "Select at most 4,000,000 samples across all presets." (the hosted Studio answers a generic 422 for the same
 * check), "Select each preset only once per dataset." and the hosted 413 `workspace_limit`. Everything else is returned unchanged.
 */
export function explainGenerateError(error: unknown, selection: { signals: number; samples: number }): unknown {
  if (!(error instanceof ApiError)) return error
  const texts = [error.message, ...error.details.map(d => d.message)].map(plainMessage)
  const named = (text: string) => texts.some(item => item.startsWith(text))
  if (error.status === 413 && error.code === 'workspace_limit') return new Error(t('generator.workspaceLimit', { count: formatNumber(selection.samples) }))
  const tooLarge = () => new Error(t('generator.batchTooLarge', { signals: selection.signals, count: formatNumber(selection.samples), limit: formatNumber(BATCH_SAMPLE_LIMIT) }))
  if (error.status === 422 && named('Select at most 4,000,000 samples')) return tooLarge()
  if (error.status === 422 && named('Select each preset only once')) return new Error(t('generator.duplicateSignal'))
  // The hosted Studio does not say which check failed; a selection above the limit can only have failed that one.
  if (error.status === 422 && error.code === 'invalid_request' && selection.samples > BATCH_SAMPLE_LIMIT) return tooLarge()
  return error
}
