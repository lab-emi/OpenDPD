import Box from '@mui/material/Box'
import TableContainer from '@mui/material/TableContainer'
import Typography from '@mui/material/Typography'
import type { ReactNode } from 'react'

/** A definition list of name / value pairs in a responsive grid (two columns on a phone, four on a desktop). */
export function Facts({ items }: { items: Array<[string, string]> }) {
  return <Box component="dl" sx={{ display: 'grid', gridTemplateColumns: { xs: 'repeat(2,minmax(0,1fr))', md: 'repeat(4,minmax(0,1fr))' }, gap: .75, m: 0 }}>
    {items.map(([name, value]) => <Box key={name} sx={{ minWidth: 0 }}><Typography component="dt" variant="caption">{name}</Typography><Typography component="dd" variant="body2" sx={{ m: 0, overflowWrap: 'anywhere' }}>{value}</Typography></Box>)}
  </Box>
}

/** A table that scrolls inside its own region (keyboard focusable) instead of widening the page. */
export function Scroll({ label, children }: { label: string; children: ReactNode }) {
  return <TableContainer role="region" tabIndex={0} aria-label={label} sx={{ overflowX: 'auto' }}>{children}</TableContainer>
}
