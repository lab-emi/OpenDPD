import { screen, waitFor, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { renderWithProviders } from '@/test/utils'
import { ResetButton } from './ResetButton'

test('cancel and Escape keep progress; continue resets once and announces completion', async () => {
  const onReset = vi.fn()
  renderWithProviders(<ResetButton onReset={onReset} />)
  const user = userEvent.setup()
  const trigger = screen.getByRole('button', { name: 'Reset page' })
  await user.click(trigger)
  let dialog = screen.getByRole('dialog', { name: 'Reset this page?' })
  expect(dialog).toHaveTextContent('This cannot be undone.')
  expect(within(dialog).getByRole('button', { name: 'Cancel' })).toHaveFocus()
  await user.keyboard('{Escape}')
  await waitFor(() => expect(dialog).not.toBeInTheDocument())
  expect(trigger).toHaveFocus()
  expect(onReset).not.toHaveBeenCalled()
  await user.click(trigger)
  dialog = screen.getByRole('dialog', { name: 'Reset this page?' })
  await user.click(within(dialog).getByRole('button', { name: 'Cancel' }))
  await waitFor(() => expect(dialog).not.toBeInTheDocument())
  expect(onReset).not.toHaveBeenCalled()
  await user.click(trigger)
  await user.click(within(screen.getByRole('dialog')).getByRole('button', { name: 'Continue' }))
  expect(onReset).toHaveBeenCalledTimes(1)
  await waitFor(() => expect(screen.queryByRole('dialog')).not.toBeInTheDocument())
  expect(screen.getByRole('alert')).toHaveTextContent('Reset complete.')
})

test('a pending save explains the delay instead of leaving an unresponsive reset button', async () => {
  const onReset = vi.fn()
  const { rerender } = renderWithProviders(<ResetButton disabled onReset={onReset} />)
  await userEvent.click(screen.getByRole('button', { name: 'Reset page' }))
  const dialog = screen.getByRole('dialog')
  expect(dialog).toHaveTextContent('A save or submission is in progress.')
  expect(within(dialog).getByRole('button', { name: 'Continue' })).toBeDisabled()
  expect(onReset).not.toHaveBeenCalled()
  rerender(<ResetButton onReset={onReset} />)
  await userEvent.click(within(dialog).getByRole('button', { name: 'Continue' }))
  expect(onReset).toHaveBeenCalledTimes(1)
})
