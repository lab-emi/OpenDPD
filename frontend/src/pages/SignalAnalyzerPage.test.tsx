import { fireEvent, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { vi } from 'vitest'
import { mockApi, renderWithProviders } from '@/test/utils'
import { analyzerDefaults } from '@/api/signalAnalyzer'
import { SignalAnalyzerPage } from './SignalAnalyzerPage'

vi.mock('@/components/PlotlyChart', () => ({ PlotlyChart: ({ title }: { title: string }) => <div>{title}</div> }))
vi.mock('@/components/SignalSpectrogram', () => ({ SignalSpectrogram: () => <div>Spectrogram view</div> }))
const generated = { source: { kind: 'generated', source_id: `sg-${'a'.repeat(64)}`, role: 'input', version: 'raw-v1' }, label: 'Custom signal · PA input', sample_count: 8192, sample_rate_hz: 20e6, bandwidth_hz: 5e6, columns: ['I', 'Q'], complex_columns: [], origin: 'synthetic' }
const uploaded = { source: { kind: 'upload', source_id: `sa-${'b'.repeat(64)}`, role: 'input', version: 'raw-v1' }, label: 'CSV signal', sample_count: 1024, sample_rate_hz: null, bandwidth_hz: null, columns: ['Voltage'], complex_columns: [], origin: 'uploaded' }
const result = { source: generated, config: analyzerDefaults, sample_count: 8192, sample_range: [0, 8192], source_sha256: 'a', real_signal: false,
  frequency_hz: [0], psd_dbfs_hz: [-60], measurements: [{ key: 'rms', label: 'RMS amplitude', value: .2, unit: '' }], notes: [],
  time_s: [0], time_i: [.2], time_q: [0], envelope: [.2], instantaneous_frequency_hz: [null], scatter_i: [.2], scatter_q: [0],
  ccdf_db: [0], ccdf_probability: [0], histogram_amplitude: [.2], histogram_probability: [1], eye_i: [[.2]], eye_q: [[0]] }

test('explicit analysis, exact source handoff, range feedback and stale results', async () => {
  const { calls } = mockApi({
    'GET /api/v1/signal-analyzer/sources': () => [generated],
    'POST /api/v1/signal-analyzer/analyze': (_url, init) => ({ ...result, config: JSON.parse(String(init.body)).config }),
  })
  renderWithProviders(<SignalAnalyzerPage />, { route: `/signal-analyzer?kind=generated&source=${generated.source.source_id}&role=input` })
  const button = await screen.findByRole('button', { name: 'Analyze signal' })
  expect(calls.filter(c => c.method === 'POST')).toHaveLength(0)
  expect(screen.getByTestId('analyzer-count')).toHaveTextContent('8,192 samples')
  expect(screen.getByRole('link', { name: 'Choose Virtual PA →' })).toHaveAttribute('href', `/pa-library?input=${generated.source.source_id}`)
  await userEvent.click(button)
  await screen.findByText('Spectrogram view')
  expect(calls.find(c => c.method === 'POST')?.body).toMatchObject({ source: generated.source, config: { sample_rate_hz: 20e6, bandwidth_hz: 5e6 }, reference: null })
  expect(screen.queryByText(/Settings changed/)).not.toBeInTheDocument()
  fireEvent.change(screen.getByLabelText('Start sample (zero-based)'), { target: { value: '1000' } })
  expect(screen.getByTestId('analyzer-count')).toHaveTextContent('7,192 samples')
  expect(screen.getByText(/Settings changed/)).toBeVisible()
})

test('real CSV upload is selected without creating a paired dataset or auto-analyzing', async () => {
  const { calls } = mockApi({
    'GET /api/v1/signal-analyzer/sources': () => [],
    'POST /api/v1/signal-analyzer/upload': () => uploaded,
    'POST /api/v1/signal-analyzer/analyze': () => ({ ...result, source: uploaded, real_signal: true }),
  })
  renderWithProviders(<SignalAnalyzerPage />)
  const file = new File(['Voltage\n'+'.1\n'.repeat(1024)], 'real.csv', { type: 'text/csv' })
  await userEvent.upload(await screen.findByTestId('analyzer-upload'), file)
  await waitFor(() => expect(screen.getByTestId('analyzer-count')).toHaveTextContent('1,024 samples'))
  expect(screen.getByText(/CSV contains no sample-rate metadata/)).toBeVisible()
  expect(screen.queryByLabelText('Q column')).not.toBeInTheDocument()
  expect(calls.filter(c => c.method === 'POST').map(c => c.path)).toEqual(['/api/v1/signal-analyzer/upload'])
  expect(screen.getByRole('button', { name: 'Analyze signal' })).toBeEnabled()
})

test('missing linked source never silently selects another capture', async () => {
  const { calls } = mockApi({ 'GET /api/v1/signal-analyzer/sources': () => [generated] })
  renderWithProviders(<SignalAnalyzerPage />, { route: '/signal-analyzer?kind=upload&source=missing' })
  await screen.findByText(/This signal is unavailable/)
  expect(screen.getByRole('button', { name: 'Analyze signal' })).toBeDisabled()
  expect(calls.some(c => c.method === 'POST')).toBe(false)
})
