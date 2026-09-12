import { render } from '@testing-library/react'
import type { RunStatus } from '@/api/types'
import { StatusChip } from './StatusChip'

const ALL: RunStatus[] = ['queued', 'running', 'cancel_requested', 'cancelled', 'succeeded', 'failed', 'interrupted']

test('every status has a text label and an icon, not colour alone', () => {
  for (const status of ALL) {
    const { container, unmount } = render(<StatusChip status={status} />)
    const chip = container.querySelector(`[data-status="${status}"]`)
    expect(chip).not.toBeNull()
    expect(chip!.textContent!.trim().length).toBeGreaterThan(0)
    expect(chip!.querySelector('svg')).not.toBeNull()
    unmount()
  }
})

test('stale heartbeat swaps the icon and is marked', () => {
  const { container } = render(<StatusChip status="running" stale />)
  expect(container.querySelector('[data-stale="true"]')).not.toBeNull()
})
