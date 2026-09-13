import { fireEvent, screen } from '@testing-library/react'
import { vi } from 'vitest'
import { mockApi, renderWithProviders } from '@/test/utils'
import type { FigureSpec } from '@/api/types'
import { PublicationFigureDialog } from './PublicationFigureDialog'

vi.mock('./PlotlyChart', () => ({ seriesDash: () => 'solid', seriesSymbol: () => 'circle', PlotlyChart: () => <output>Panel preview</output> }))
const initial: FigureSpec = { version: 'figure-v2', title: 'Research figure', reference_run_id: 'run-a', profiles: { 'run-a': 'general-spectral-v1' }, width: 'double_column', mode: 'same_condition', panels: [{ kind: 'spectrum', traces: [{ run_id: 'run-a', trace_name: 'output', visible: true, color: '#2563EB', dash: 'solid' }], show_bands: false }] }

test('multi-panel authoring needs a valid preview, retains the exact spec and offers private full reproduction', async () => {
  const { calls } = mockApi({
    'POST /api/v1/figure-sources': () => ({ sources: ['spectrum', 'error_distribution'].map(kind => ({ run_id: 'run-a', kind, trace_name: 'output', role: 'primary', source: 'synthetic fixture' })), missing: ['Normalized IQ has no dBm'] }),
    'GET /api/v1/figures': () => [],
    'POST /api/v1/figure-preview': (_url, init) => ({ figure: { figure_id: 'fig-preview', spec: JSON.parse(String(init.body)), bindings: [] }, plots: { 'run-a/spectrum': { frequency: [1, 2], axis: 'hz', traces: [{ name: 'output', psd_db: [-40, -50] }] }, 'run-a/error_distribution': { x: [0, 1], x_label: 'Residual', x_unit: 'ratio', y_label: 'Cumulative samples', y_unit: '%', traces: [{ name: 'output', y: [50, 100] }] } } }),
    'POST /api/v1/figures': (_url, init) => ({ figure_id: 'fig-test', spec: JSON.parse(String(init.body)), bindings: [] }),
  })
  renderWithProviders(<PublicationFigureDialog initial={initial} onClose={() => {}} />)
  expect(screen.getByRole('button', { name: 'Save view' })).toBeDisabled()
  await screen.findByText('Unavailable plot sources')
  fireEvent.click(screen.getByRole('button', { name: 'Add panel' }))
  expect(screen.getByText('2. Residual distribution (not EVM)')).toBeInTheDocument()
  fireEvent.click(screen.getByRole('button', { name: 'Preview panels' }))
  await screen.findAllByText('Panel preview')
  fireEvent.click(screen.getByRole('button', { name: 'Save view' }))
  expect(await screen.findByRole('link', { name: 'Download full metric + figure reproduction' })).toHaveAttribute('href', '/api/v1/figures/fig-test/reproduction')
  expect(calls.find(c => c.path === '/api/v1/figures' && c.method === 'POST')?.body).toMatchObject({ version: 'figure-v2', panels: [{ kind: 'spectrum' }, { kind: 'error_distribution' }] })
  fireEvent.change(screen.getByLabelText('Figure title'), { target: { value: 'Changed figure' } })
  expect(screen.getByRole('button', { name: 'Save view' })).toBeDisabled()
  expect(screen.queryByRole('link', { name: 'Download full metric + figure reproduction' })).not.toBeInTheDocument()
})
