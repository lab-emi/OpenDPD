import { screen } from '@testing-library/react'
import { vi } from 'vitest'
import runningMock from '@mocks/run_running.json'
import type { RunView } from '@/api/types'
import type { StreamState } from '@/api/events'
import { mockApi, renderWithProviders } from '@/test/utils'
import { LiveRunDashboard } from './LiveRunDashboard'

vi.mock('plotly.js-basic-dist-min', () => ({ default: { react: vi.fn(() => Promise.resolve()), purge: vi.fn() } }))
const run = runningMock.data as unknown as RunView
const stream = { connection: 'live', metrics: [], batchProgress: { phase: 'train', batch: 25, total_batches: 100, sequences: 8, sequence_samples: 50, sample_rate_hz: 800e6 } } as unknown as StreamState

test('displays actual last-batch geometry and preserves the worker metric units', async () => {
  mockApi({ [`GET /api/v1/runs/${run.run_id}/live`]: () => ({
    policy: { min_batches: 25, min_seconds: 2, overhead_target: .05 },
    geometry: { batch_size: 64, sequence_samples: 50, sample_rate_hz: 800e6, frame_stride: 16 },
    preview: { source: 'final_test', samples: 2560, metrics: { NMSE: -30, EVM: -24 }, units: { NMSE: 'dB', EVM: 'dB' }, plots: {}, updated_at: '2026-09-12T11:00:00Z' },
  }) })
  renderWithProviders(<LiveRunDashboard run={run} stream={stream} metrics={[]} />)
  await screen.findByText('-24.00')
  expect(screen.getAllByText('dB')).toHaveLength(2)
  expect(screen.getByText(/400 complex I\/Q samples per batch @ 800 MSa\/s/)).toBeInTheDocument()
  expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '25')
})

test('a spectral-profile DPD test shows its own metrics without an epoch curve', async () => {
  mockApi({ [`GET /api/v1/runs/${run.run_id}/live`]: () => ({
    policy: { min_batches: 25, min_seconds: 2, overhead_target: .05 }, geometry: null,
    preview: { source: 'final_test', metric_profile: 'general-spectral-v1', samples: 2560,
      metrics: { NMSE: -30, IBE: -28, ACPR_L: -40, ACPR_R: -39 }, units: { NMSE: 'dB', IBE: 'dB', ACPR_L: 'dBc', ACPR_R: 'dBc' }, plots: {}, updated_at: '2026-09-12T11:00:00Z' },
  }) })
  renderWithProviders(<LiveRunDashboard run={{ ...run, task: 'run_dpd', status: 'succeeded' }} stream={stream} metrics={[{ epoch: 1, split: 'test_probe', values: { ACLR_AVG: -37 } }]} />)
  expect(await screen.findByText('ACPR_L')).toBeInTheDocument()
  expect(screen.getByText('IBE')).toBeInTheDocument()
  expect(screen.queryByText(/per epoch/)).not.toBeInTheDocument()
  expect(screen.queryByText(/batches and/)).not.toBeInTheDocument()
})
