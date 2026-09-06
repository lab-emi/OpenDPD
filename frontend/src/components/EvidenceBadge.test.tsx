import { render } from '@testing-library/react'
import { EvidenceBadge } from './EvidenceBadge'

test('mock results are visibly marked as MOCK', () => {
  const { container } = render(<EvidenceBadge evidence="pa_modeling" mock />)
  const chip = container.querySelector('[data-mock="true"]')
  expect(chip).not.toBeNull()
  expect(chip!.textContent).toContain('MOCK')
  expect(chip!.textContent).toContain('PA model')
})

test('real evidence has no mock marker', () => {
  const { container } = render(<EvidenceBadge evidence="dpd_measured" />)
  expect(container.querySelector('[data-mock]')).toBeNull()
  expect(container.textContent).toContain('DPD · measured')
})
