import ArrowDropDownIcon from '@mui/icons-material/ArrowDropDown'
import Alert from '@mui/material/Alert'
import Button from '@mui/material/Button'
import ListItemIcon from '@mui/material/ListItemIcon'
import ListItemText from '@mui/material/ListItemText'
import Menu from '@mui/material/Menu'
import MenuItem from '@mui/material/MenuItem'
import Snackbar from '@mui/material/Snackbar'
import { useState, type MouseEvent } from 'react'
import { useUpdateSettings } from '@/api/hooks'
import { LANGUAGES, languageInfo, setLanguage, t, useLanguage, type LanguageCode } from '@/i18n'

function Flag({ src }: { src: string }) {
  return <img src={src} alt="" width={20} height={15} style={{ display: 'block', borderRadius: 2, boxShadow: '0 0 0 1px rgba(0,0,0,0.15)' }} />
}

/** Flag-and-name language switcher: applies the choice at once, then stores it in the workspace. */
export function LanguageMenu({ variant = 'toolbar' }: { variant?: 'toolbar' | 'settings' }) {
  const code = useLanguage()
  const current = languageInfo(code)
  const update = useUpdateSettings()
  const [anchor, setAnchor] = useState<HTMLElement | null>(null)
  const [failure, setFailure] = useState<string | null>(null)

  const choose = async (next: LanguageCode) => {
    setAnchor(null)
    if (next === code) return
    await setLanguage(next)
    try {
      await update.mutateAsync({ language: next })
    } catch (err) {
      setFailure(err instanceof Error ? err.message : String(err))
    }
  }

  return (
    <>
      <Button
        onClick={(e: MouseEvent<HTMLElement>) => setAnchor(e.currentTarget)}
        color="inherit"
        variant={variant === 'settings' ? 'outlined' : 'text'}
        size="small"
        aria-label={t('language.label')}
        aria-haspopup="menu"
        aria-expanded={anchor ? 'true' : undefined}
        startIcon={<Flag src={current.flag} />}
        endIcon={<ArrowDropDownIcon />}
        data-testid="language-menu"
        sx={{ textTransform: 'none', flexShrink: 0 }}
      >
        <span lang={current.tag}>{current.name}</span>
      </Button>
      <Menu anchorEl={anchor} open={Boolean(anchor)} onClose={() => setAnchor(null)} slotProps={{ list: { 'aria-label': t('language.label') } }}>
        {LANGUAGES.map((l) => (
          <MenuItem key={l.code} selected={l.code === code} onClick={() => void choose(l.code)} lang={l.tag}>
            <ListItemIcon>
              <Flag src={l.flag} />
            </ListItemIcon>
            <ListItemText>{l.name}</ListItemText>
          </MenuItem>
        ))}
      </Menu>
      <Snackbar open={failure !== null} autoHideDuration={8000} onClose={() => setFailure(null)}>
        <Alert severity="error" onClose={() => setFailure(null)}>
          {t('language.saveFailed', { error: failure ?? '' })}
        </Alert>
      </Snackbar>
    </>
  )
}
