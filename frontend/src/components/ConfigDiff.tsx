import Table from '@mui/material/Table'
import TableBody from '@mui/material/TableBody'
import TableCell from '@mui/material/TableCell'
import TableHead from '@mui/material/TableHead'
import TableRow from '@mui/material/TableRow'
import Typography from '@mui/material/Typography'
import { useMemo } from 'react'
import { t } from '@/i18n'

type Json = string | number | boolean | null | Json[] | { [k: string]: Json }

function flatten(value: unknown, prefix = '', out: Map<string, string> = new Map()): Map<string, string> {
  if (value !== null && typeof value === 'object' && !Array.isArray(value)) {
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) flatten(v, prefix ? `${prefix}.${k}` : k, out)
  } else {
    out.set(prefix, JSON.stringify(value as Json))
  }
  return out
}

interface DiffRow {
  field: string
  left: string | undefined
  right: string | undefined
}

export function diffConfigs(a: unknown, b: unknown, ignore: string[] = ['resolution']): DiffRow[] {
  const fa = flatten(a)
  const fb = flatten(b)
  const fields = [...new Set([...fa.keys(), ...fb.keys()])].filter((f) => !ignore.some((p) => f === p || f.startsWith(`${p}.`))).sort()
  return fields.filter((f) => fa.get(f) !== fb.get(f)).map((f) => ({ field: f, left: fa.get(f), right: fb.get(f) }))
}

/** Field-level difference between two (resolved) configurations; identical fields are hidden. */
export function ConfigDiff({ left, right, leftLabel = t('diff.left'), rightLabel = t('diff.right') }: { left: unknown; right: unknown; leftLabel?: string; rightLabel?: string }) {
  const rows = useMemo(() => diffConfigs(left, right), [left, right])
  if (rows.length === 0) return <Typography color="text.secondary">{t('diff.same')}</Typography>
  return (
    <Table size="small" aria-label={t('diff.title')}>
      <TableHead>
        <TableRow>
          <TableCell>{t('diff.field')}</TableCell>
          <TableCell>{leftLabel}</TableCell>
          <TableCell>{rightLabel}</TableCell>
        </TableRow>
      </TableHead>
      <TableBody>
        {rows.map((r) => (
          <TableRow key={r.field}>
            <TableCell>
              <code>{r.field}</code>
            </TableCell>
            <TableCell>
              <code>{r.left ?? '—'}</code>
            </TableCell>
            <TableCell>
              <code>{r.right ?? '—'}</code>
            </TableCell>
          </TableRow>
        ))}
      </TableBody>
    </Table>
  )
}
