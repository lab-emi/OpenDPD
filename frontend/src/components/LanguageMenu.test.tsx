import { act, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { afterEach, expect, test } from 'vitest'
import { resetLanguage, t } from '@/i18n'
import { mockApi, renderWithProviders } from '@/test/utils'
import { LanguageMenu } from './LanguageMenu'

afterEach(() => resetLanguage())

test('lists nine languages with flags and names, switches at once and stores the choice', async () => {
  const { calls } = mockApi({ 'PUT /api/v1/settings': (_url, init) => JSON.parse(String(init.body)) })
  renderWithProviders(<LanguageMenu />)
  await userEvent.click(screen.getByRole('button', { name: 'Language' }))
  const items = screen.getAllByRole('menuitem')
  expect(items.map((i) => i.textContent)).toEqual(['English', 'Nederlands', '中文', 'Français', 'Deutsch', 'Italiano', '日本語', '한국어', 'Español'])
  expect(items[2]?.querySelector('img')).toHaveAttribute('alt', '')
  await userEvent.click(screen.getByRole('menuitem', { name: 'Deutsch' }))
  await waitFor(() => expect(t('nav.datasets')).toBe('Datensätze'))
  expect(screen.getByRole('button', { name: 'Sprache' })).toHaveTextContent('Deutsch')
  await waitFor(() => expect(calls.filter((c) => c.method === 'PUT')).toEqual([{ method: 'PUT', path: '/api/v1/settings', body: { language: 'de' } }]))
})

test('a failed save keeps the language on screen and says so', async () => {
  mockApi({ 'PUT /api/v1/settings': () => ({ status: 500, body: { error: { code: 'workspace_error', message: 'disk full', details: [], hint: null } } }) })
  renderWithProviders(<LanguageMenu />)
  await userEvent.click(screen.getByRole('button', { name: 'Language' }))
  await userEvent.click(screen.getByRole('menuitem', { name: '日本語' }))
  await waitFor(() => expect(t('nav.datasets')).toBe('データセット'))
  expect(await screen.findByRole('alert')).toHaveTextContent('disk full')
})

test('fast selections keep updating the UI while settings writes finish in selection order', async () => {
  const { fetchMock } = mockApi({ 'PUT /api/v1/settings': (_url, init) => JSON.parse(String(init.body)) })
  let release!: (response: Response) => void
  fetchMock.mockImplementationOnce(() => new Promise<Response>((resolve) => { release = resolve }))
  const { client } = renderWithProviders(<LanguageMenu />)
  // The production shell observes settings; this isolated menu has no query observer.
  client.setQueryDefaults(['settings'], { gcTime: Infinity })
  await userEvent.click(screen.getByTestId('language-menu'))
  await userEvent.click(screen.getByRole('menuitem', { name: 'Nederlands' }))
  await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(1))
  await userEvent.click(screen.getByTestId('language-menu'))
  await userEvent.click(screen.getByRole('menuitem', { name: 'Italiano' }))
  await waitFor(() => expect(document.documentElement.lang).toBe('it'))
  expect(fetchMock).toHaveBeenCalledTimes(1)
  await act(async () => release(new Response(JSON.stringify({ language: 'nl' }), { headers: { 'Content-Type': 'application/json' } })))
  await waitFor(() => expect(fetchMock).toHaveBeenCalledTimes(2))
  await waitFor(() => expect(client.getQueryData(['settings'])).toEqual({ language: 'it' }))
  expect(document.documentElement.lang).toBe('it')
})
