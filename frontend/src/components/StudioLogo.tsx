import Box from '@mui/material/Box'
import { useTheme } from '@mui/material/styles'
import logo from '@/assets/opendpd-studio-logo.svg'
import inverseLogo from '@/assets/opendpd-studio-logo-inverse.svg'
import mark from '@/assets/opendpd-studio-mark.svg'
import lightMark from '@/assets/opendpd-studio-mark-light.svg'

/** Transparent vector artwork follows its surface, including the compact rail. */
export function StudioLogo({ compact = false }: { compact?: boolean }) {
  const dark = useTheme().palette.mode === 'dark'
  const wordmark = dark ? inverseLogo : logo
  const emblem = dark ? mark : lightMark
  return <Box component="img" src={compact ? emblem : wordmark} alt="OpenDPD Studio" width={compact ? 160 : 600} height={compact ? 160 : 168} sx={{
    display: 'block', width: '100%', height: '100%', minWidth: 0,
    objectFit: 'contain', objectPosition: 'left center',
  }} />
}
