import { screen } from '@testing-library/react'
import { afterEach, expect, test, vi } from 'vitest'
import { getLanguage, resetLanguage } from '@/i18n'
import { LanguageGate } from '@/i18n/LanguageGate'
import { mockApi, renderWithProviders } from '@/test/utils'

afterEach(() => resetLanguage())

test('applies the stored language before rendering the children', async () => {
  mockApi({ 'GET /api/v1/settings': () => ({ language: 'ja' }) })
  renderWithProviders(<LanguageGate><p>ready</p></LanguageGate>)
  expect(await screen.findByText('ready')).toBeInTheDocument()
  expect(getLanguage()).toBe('ja')
  expect(document.documentElement.lang).toBe('ja')
})

test('a new workspace starts in English even in a French browser', async () => {
  vi.spyOn(navigator, 'languages', 'get').mockReturnValue(['fr-CA', 'en'])
  mockApi({ 'GET /api/v1/settings': () => ({ language: null }) })
  renderWithProviders(<LanguageGate><p>ready</p></LanguageGate>)
  expect(await screen.findByText('ready')).toBeInTheDocument()
  expect(getLanguage()).toBe('en')
})

test('a settings failure shows the error state with retry, not a blank page', async () => {
  mockApi({ 'GET /api/v1/settings': () => ({ status: 500, body: { error: { code: 'workspace_error', message: 'cannot read settings.json', details: [], hint: null } } }) })
  renderWithProviders(<LanguageGate><p>ready</p></LanguageGate>)
  expect(await screen.findByText(/cannot read settings.json/)).toBeInTheDocument()
  expect(screen.getByRole('button', { name: 'Retry' })).toBeInTheDocument()
})
