import { fireEvent, screen } from '@testing-library/react'
import { vi } from 'vitest'
import fixture from '@mocks/virtual_pa_models.json'
import { renderWithProviders } from '@/test/utils'
import { MathFormula } from './MathFormula'

test('every catalog equation renders and its coefficients remain interactive after a parent update', () => {
  const select = vi.fn()
  for (const model of fixture.data) {
    const variables = model.parameters.map(p => ({ key: p.key, symbol: p.symbol_latex, label: p.label.en }))
    const view = renderWithProviders(<>{model.equations_latex.map((latex, i) => <MathFormula key={i} latex={latex} variables={variables} onSelect={select} />)}</>)
    expect(view.container.querySelector('.katex-error')).toBeNull()
    const first = view.container.querySelector<HTMLElement>('[data-variable]')!
    fireEvent.keyDown(first, { key: 'Enter' })
    expect(select).toHaveBeenLastCalledWith(first.dataset.variable)
    view.unmount()
  }
  const variables = [{ key: 'gain', symbol: 'G', label: 'Gain' }]
  const view = renderWithProviders(<MathFormula latex="y={{gain}}x" variables={variables} onSelect={select} />)
  view.rerender(<MathFormula latex="y={{gain}}x" variables={variables} active="gain" onSelect={select} />)
  expect(screen.getByRole('button', { name: 'Gain' })).toHaveAttribute('aria-pressed', 'true')
  fireEvent.click(screen.getByRole('button', { name: 'Gain' }))
  expect(select).toHaveBeenLastCalledWith('gain')
})

test.each([
  String.raw`\href{javascript:alert(1)}{click}`,
  String.raw`\url{https://example.com}`,
  String.raw`\includegraphics{https://example.com/pixel.png}`,
  String.raw`\htmlStyle{background:url(https://example.com/pixel.png)}{x}`,
  String.raw`\htmlClass{untrusted}{x}`,
  '<img src=x onerror=alert(1)><script>alert(1)</script>',
])('untrusted math cannot load resources, inject markup or create links: %s', latex => {
  const { container } = renderWithProviders(<MathFormula latex={latex} />)
  expect(container.querySelector('a, img, script, iframe, object, .untrusted, [onerror], [onclick]')).toBeNull()
  expect(container.querySelector('[style*="url("]')).toBeNull()
})
