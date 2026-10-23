import TextField from '@mui/material/TextField'
import { useEffect, useRef, useState } from 'react'

const text = (v: number | null) => v === null || !Number.isFinite(v) ? '' : String(v)

/**
 * A number field that keeps what is being typed ("0.", "-") while it commits only finished numbers.
 * An empty field commits `null` when `allowEmpty` is set (automatic value) and is otherwise ignored until blur restores it.
 */
export function NumberInput({ label, value, onCommit, allowEmpty = false, placeholder, helperText, min, max, step, disabled, error }: {
  label: string; value: number | null; onCommit: (value: number | null) => void; allowEmpty?: boolean; placeholder?: string; helperText?: string
  min?: number; max?: number; step?: number | 'any'; disabled?: boolean; error?: boolean
}) {
  const [raw, setRaw] = useState(text(value))
  const focused = useRef(false)
  useEffect(() => { if (!focused.current) setRaw(text(value)) }, [value])
  const empty = raw.trim() === '' && !allowEmpty
  return <TextField type="number" label={label} value={raw} disabled={disabled} placeholder={placeholder} helperText={helperText} error={error || empty}
    onFocus={() => { focused.current = true }} onBlur={() => { focused.current = false; setRaw(text(value)) }}
    onChange={e => {
      setRaw(e.target.value)
      if (e.target.value.trim() === '') { if (allowEmpty) onCommit(null); return }
      const next = Number(e.target.value)
      if (Number.isFinite(next)) onCommit(next)
    }}
    slotProps={{ inputLabel: { shrink: true }, htmlInput: { min, max, step: step ?? 1 } }} />
}
