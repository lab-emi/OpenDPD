import { render, screen } from '@testing-library/react'
import type { DiagnosticReport } from '@/api/types'
import doctor from '@mocks/diagnostics_missing_metadata.json'
import { DiagnosticItem } from './DiagnosticItem'

const report = doctor.data as unknown as DiagnosticReport

test('renders severity, evidence, suggestion and the blocking marker', () => {
  const blocking = (report.items ?? []).find((i) => i.blocking)!
  render(<DiagnosticItem item={blocking} />)
  expect(screen.getByRole('article', { name: blocking.title })).toHaveAttribute('data-severity', blocking.severity)
  expect(screen.getByText('blocks evaluation')).toBeInTheDocument()
  expect(screen.getByText(blocking.message)).toBeInTheDocument()
  if (blocking.suggestion) expect(screen.getByText(blocking.suggestion)).toBeInTheDocument()
})
