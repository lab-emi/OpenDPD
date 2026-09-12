import { act, fireEvent, render, screen } from '@testing-library/react'
import { useTheme } from '@mui/material/styles'
import { useQueryClient } from '@tanstack/react-query'
import { useState, type ReactNode } from 'react'
import { Outlet } from 'react-router'
import { expect, test, vi } from 'vitest'
import App from './App'
import { StudioLogo } from './components/StudioLogo'

vi.mock('@/pages/SessionGate', () => ({ SessionGate: ({ children }: { children: ReactNode }) => children }))
vi.mock('@/i18n/LanguageGate', () => ({ LanguageGate: ({ children }: { children: ReactNode }) => children }))
vi.mock('@/layout/AppShell', () => ({ AppShell: () => <Outlet /> }))
vi.mock('@/pages/HomePage', () => ({ HomePage: function Probe() {
  const theme = useTheme()
  const client = useQueryClient()
  const [draft, setDraft] = useState('')
  return <>
    <output>{theme.palette.mode}</output>
    <input aria-label="Experiment name" value={draft} onChange={e => { setDraft(e.target.value); client.setQueryData(['draft-check'], e.target.value) }} />
    <span data-testid="cache">{client.getQueryData<string>(['draft-check'])}</span>
    <StudioLogo /><StudioLogo compact />
  </>
} }))

test('live system appearance changes keep the form, query cache and route mounted while switching both SVG sizes', () => {
  window.history.replaceState({}, '', '/')
  let dark = false
  const events = new EventTarget()
  vi.spyOn(window, 'matchMedia').mockImplementation(query => ({
    get matches() { return query === '(prefers-color-scheme: dark)' && dark },
    media: query, onchange: null,
    addEventListener: events.addEventListener.bind(events), removeEventListener: events.removeEventListener.bind(events),
    addListener: vi.fn(), removeListener: vi.fn(), dispatchEvent: events.dispatchEvent.bind(events),
  }))
  render(<App />)
  const input = screen.getByRole('textbox', { name: 'Experiment name' })
  fireEvent.change(input, { target: { value: 'PA draft' } })
  const images = screen.getAllByRole('img', { name: 'OpenDPD Studio' })
  const light = images.map(img => img.getAttribute('src'))
  expect(screen.getByText('light')).toBeVisible()
  act(() => { dark = true; events.dispatchEvent(new Event('change')) })
  expect(screen.getByText('dark')).toBeVisible()
  expect(screen.getByRole('textbox')).toBe(input)
  expect(input).toHaveValue('PA draft')
  expect(screen.getByTestId('cache')).toHaveTextContent('PA draft')
  images.forEach((img, i) => expect(img.getAttribute('src')).not.toBe(light[i]))
  act(() => { dark = false; events.dispatchEvent(new Event('change')) })
  expect(screen.getByTestId('cache')).toHaveTextContent('PA draft')
  images.forEach((img, i) => expect(img.getAttribute('src')).toBe(light[i]))
})
