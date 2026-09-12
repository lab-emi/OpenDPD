import { render, screen } from '@testing-library/react'
import resolved from '@mocks/resolved_train_pa_smoke.json'
import { ConfigDiff, diffConfigs } from './ConfigDiff'

test('diff lists only changed leaf fields and ignores resolution metadata', () => {
  const a = resolved.data
  const b = { ...a, training: { ...a.training, epochs: 300 }, resolution: { ...a.resolution, resolved_at: 'later' } }
  const rows = diffConfigs(a, b)
  expect(rows).toEqual([{ field: 'training.epochs', left: '3', right: '300' }])
})

test('identical configs render the "identical" message', () => {
  render(<ConfigDiff left={resolved.data} right={resolved.data} />)
  expect(screen.getByText('The two configurations are identical.')).toBeInTheDocument()
})
