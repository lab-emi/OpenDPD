import { fireEvent, screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useLocation } from 'react-router'
import { mockApi, renderWithProviders } from '@/test/utils'
import { SignalImport } from './SignalImport'

const uploaded = { source: { kind: 'upload', source_id: 'sa-' + 'a'.repeat(64) }, label: 'custom.csv', sample_count: 32768,
  columns: ['I', 'Q'], complex_columns: [], origin: 'uploaded' }
const imported = { signal_id: 'sg-' + 'b'.repeat(64), dataset_id: 'sds-' + 'c'.repeat(64), dataset_name: 'usr_pa_in_custom_n1' }
function Probe() { const loc = useLocation(); return <output data-testid="location">{loc.pathname}{loc.search}</output> }

test('custom waveform upload collects metadata before sending it to the PA Library', async () => {
  const { calls } = mockApi({
    'POST /api/v1/signal-analyzer/upload': (_url, init) => {
      expect(init.body).toBeInstanceOf(FormData)
      return uploaded
    },
    'POST /api/v1/signal-generator/import': () => imported,
  })
  renderWithProviders(<><SignalImport /><Probe /></>)
  await userEvent.click(screen.getByRole('button', { name: /Import custom signal/ }))
  await userEvent.upload(screen.getByTestId('signal-import-upload'), new File(['I,Q\n0.2,0.3'], 'custom.csv', { type: 'text/csv' }))
  const name = await screen.findByRole('textbox', { name: 'PA input dataset name' })
  expect(name).toHaveValue('usr_pa_in_custom_n1')
  expect(calls.map(c => c.path)).toEqual(['/api/v1/signal-analyzer/upload'])
  const button = screen.getByRole('button', { name: 'Import & use in Virtual PA' })
  const rate = screen.getByRole('spinbutton', { name: 'Sample rate (MHz)' })
  fireEvent.change(rate, { target: { value: '10' } })
  expect(button).toBeDisabled()
  fireEvent.change(rate, { target: { value: '160' } })
  fireEvent.change(screen.getByRole('spinbutton', { name: 'Baseband bandwidth (MHz)' }), { target: { value: '40' } })
  await userEvent.click(button)
  expect(calls.find(c => c.path.endsWith('/import'))?.body).toEqual({
    upload_id: uploaded.source.source_id, dataset_name: 'usr_pa_in_custom_n1', sample_rate_hz: 160e6,
    bandwidth_hz: 40e6, carrier_frequency_hz: 0, sample_format: 'iq', i_column: 0, q_column: 1,
  })
  await waitFor(() => expect(screen.getByTestId('location')).toHaveTextContent('/pa-library?input=' + imported.signal_id + '&dataset=' + imported.dataset_id))
})

test('a complex CSV selects the complex column and a failed replacement clears the old selection', async () => {
  let attempt = 0
  const { calls } = mockApi({ 'POST /api/v1/signal-analyzer/upload': () => ++attempt === 1
    ? { ...uploaded, columns: ['time', 'signal'], complex_columns: [1], sample_count: 512 }
    : { status: 422, body: { error: { code: 'csv_rejected', message: 'Invalid CSV samples' } } } })
  renderWithProviders(<SignalImport />)
  await userEvent.click(screen.getByRole('button', { name: /Import custom signal/ }))
  const file = screen.getByTestId('signal-import-upload')
  await userEvent.upload(file, new File(['signal\n0.2+0.3j'], 'complex.csv', { type: 'text/csv' }))
  expect(await screen.findByRole('combobox', { name: 'Sample format' })).toHaveTextContent('Complex samples')
  expect(screen.getByRole('combobox', { name: 'Signal / I column' })).toHaveTextContent('signal')
  expect(screen.queryByRole('combobox', { name: 'Q column' })).not.toBeInTheDocument()
  expect(screen.getByText(/at least 8,192 samples/)).toBeVisible()
  await userEvent.upload(file, new File(['nan'], 'broken.csv', { type: 'text/csv' }))
  await screen.findByText(/Invalid CSV samples/)
  expect(screen.queryByRole('button', { name: 'Import & use in Virtual PA' })).not.toBeInTheDocument()
  expect(calls.every(c => c.path.endsWith('/upload'))).toBe(true)
})
