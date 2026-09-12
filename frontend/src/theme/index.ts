import { createTheme } from '@mui/material/styles'
import tokens from './tokens.json'

export { tokens }

export type StatusTone = keyof typeof tokens.color.status

export const theme = createTheme({
  palette: {
    mode: 'light',
    primary: { main: tokens.color.primary, contrastText: tokens.color.primaryContrast },
    secondary: { main: tokens.color.secondary },
    background: { default: tokens.color.background, paper: tokens.color.surface },
    text: { primary: tokens.color.textPrimary, secondary: tokens.color.textSecondary },
    divider: tokens.color.border,
    success: { main: tokens.color.status.success },
    warning: { main: tokens.color.status.warning },
    error: { main: tokens.color.status.error },
    info: { main: tokens.color.status.info },
    // icons and unselected toggle buttons: MUI's default 54 % black is 4.49:1 on the page background
    action: { active: tokens.color.iconActive },
  },
  spacing: tokens.spacing.unit,
  shape: { borderRadius: tokens.radius.md },
  typography: {
    fontFamily: tokens.typography.fontFamily,
    fontSize: tokens.typography.baseSize,
    h1: { fontSize: '1.6rem', fontWeight: 600 },
    h2: { fontSize: '1.25rem', fontWeight: 600 },
    h3: { fontSize: '1.05rem', fontWeight: 600 },
  },
  components: {
    MuiButtonBase: { defaultProps: { disableRipple: true } },
    MuiCard: { defaultProps: { variant: 'outlined' } },
    MuiPaper: { defaultProps: { variant: 'outlined' } },
    MuiTextField: { defaultProps: { size: 'small' } },
    MuiCssBaseline: {
      styleOverrides: {
        ':focus-visible': { outline: `3px solid ${tokens.color.primary}`, outlineOffset: 2 },
        code: { fontFamily: tokens.typography.monoFamily },
      },
    },
  },
})

/** Plotly layout defaults derived from the same tokens. */
export const plotLayoutBase = {
  font: { family: tokens.typography.fontFamily, size: tokens.typography.baseSize - 1, color: tokens.color.textPrimary },
  paper_bgcolor: tokens.color.surface,
  plot_bgcolor: tokens.color.surface,
  colorway: tokens.color.chart,
  margin: { l: 56, r: 16, t: 36, b: 44 },
  legend: { orientation: 'h' as const, y: -0.2 },
}
