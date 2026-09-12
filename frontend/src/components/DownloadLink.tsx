import Alert from '@mui/material/Alert'
import Button from '@mui/material/Button'
import Link from '@mui/material/Link'
import type { SxProps, Theme } from '@mui/material/styles'
import { useState, type ReactNode } from 'react'
import { WEB_MODE, downloadFile } from '@/api/client'

/** Same-origin desktop links, authenticated fetch downloads in the public app. */
export function DownloadLink({ href, download, children, button = false, variant = 'text', size = 'small', sx }: {
  href: string; download?: string | boolean; children: ReactNode; button?: boolean;
  variant?: 'text' | 'outlined' | 'contained'; size?: 'small' | 'medium' | 'large'; sx?: SxProps<Theme>;
}) {
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const props = { href, download: download ?? true, sx, onClick: (event: React.MouseEvent<HTMLAnchorElement>) => {
    if (!WEB_MODE) return
    event.preventDefault()
    if (busy) return
    setBusy(true); setError(null)
    void downloadFile(href, typeof download === 'string' ? download : undefined)
      .catch((reason: unknown) => setError(reason instanceof Error ? reason.message : String(reason)))
      .finally(() => setBusy(false))
  } }
  return <>
    {button ? <Button component="a" {...props} variant={variant} size={size} disabled={busy}>{children}</Button>
      : <Link {...props} aria-disabled={busy}>{children}</Link>}
    {error && <Alert severity="error">{error}</Alert>}
  </>
}
