import Alert from '@mui/material/Alert'
import Button from '@mui/material/Button'
import Link from '@mui/material/Link'
import type { SxProps, Theme } from '@mui/material/styles'
import { useState, type ReactNode } from 'react'
import { artifactDownloadUrl, downloadFile } from '@/api/client'

/** Same-origin desktop links, authenticated fetch downloads in the public app. */
export function DownloadLink({ href, download, children, button = false, variant = 'text', size = 'small', sx }: {
  href: string; download?: string | boolean; children: ReactNode; button?: boolean;
  variant?: 'text' | 'outlined' | 'contained'; size?: 'small' | 'medium' | 'large'; sx?: SxProps<Theme>;
}) {
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  let safeHref: string | undefined
  try { artifactDownloadUrl(href); safeHref = href } catch { /* Invalid links stay inert. */ }
  const props = { href: safeHref, download: download ?? true, sx, onClick: (event: React.MouseEvent<HTMLAnchorElement>) => {
    event.preventDefault()
    if (!safeHref) { setError('Invalid artifact URL'); return }
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
