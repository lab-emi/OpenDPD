import { render, screen } from '@testing-library/react'
import type { MetricValue } from '@/api/types'
import { MetricCard, formatMetric } from './MetricCard'

const ok: MetricValue = { name: 'NMSE', value: -31.234567, unit: 'dB', better: 'lower', status: 'ok', reason: null }
const na: MetricValue = { name: 'EVM', value: null, unit: 'dB', better: 'lower', status: 'not_applicable', reason: 'no linear reference for run_dpd' }

test('shows value with unit and better-direction', () => {
  render(<MetricCard metric={ok} />)
  expect(screen.getByText('-31.23 dB')).toBeInTheDocument()
  expect(screen.getByText('lower is better')).toBeInTheDocument()
})

test('not-applicable metric shows reason and never a number', () => {
  render(<MetricCard metric={na} />)
  expect(screen.getByText('not applicable')).toBeInTheDocument()
  expect(screen.getByText('no linear reference for run_dpd')).toBeInTheDocument()
  expect(screen.queryByText(/^0/)).not.toBeInTheDocument()
  expect(formatMetric({ ...na, status: 'failed' })).toBe('failed')
})
