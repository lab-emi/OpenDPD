import { deDE, enUS, esES, frFR, itIT, nlNL, jaJP, koKR, zhCN, type Localization } from '@mui/material/locale'
import { createTheme, useTheme } from '@mui/material/styles'
import type { PaletteMode } from '@mui/material'
import type { LanguageCode } from '@/i18n'
import tokens from './tokens.json'

export { tokens }

export type StatusTone = keyof typeof tokens.color.status

export const colorsFor = (mode: PaletteMode) => mode === 'dark' ? tokens.darkColor : tokens.color
export const useStudioColors = () => colorsFor(useTheme().palette.mode)

function createStudioTheme(mode: PaletteMode) {
  const color = colorsFor(mode)
  return createTheme({
    palette: {
      mode,
      primary: { main: color.primary, contrastText: color.primaryContrast },
      secondary: { main: color.secondary },
      background: { default: color.background, paper: color.surface },
      text: { primary: color.textPrimary, secondary: color.textSecondary },
      divider: color.border,
      success: { main: color.status.success },
      warning: { main: color.status.warning },
      error: { main: color.status.error },
      info: { main: color.status.info },
      // icons and unselected toggle buttons: MUI's default 54 % black is 4.49:1 on the page background
      action: { active: color.iconActive },
    },
    spacing: tokens.spacing.unit,
    shape: { borderRadius: tokens.radius.md },
    typography: {
      fontFamily: tokens.typography.fontFamily,
      fontSize: tokens.typography.baseSize,
      h1: { fontSize: '1.5rem', fontWeight: 700, letterSpacing: '-0.035em' },
      h2: { fontSize: '1.25rem', fontWeight: 600 },
      h3: { fontSize: '1.05rem', fontWeight: 600 },
    },
    components: {
      MuiButtonBase: { defaultProps: { disableRipple: true } },
      MuiCard: { defaultProps: { variant: 'outlined' } },
      MuiPaper: { defaultProps: { variant: 'outlined' }, styleOverrides: { root: { backgroundImage: 'none' } } },
      MuiTextField: { defaultProps: { size: 'small' } },
      MuiOutlinedInput: { styleOverrides: { notchedOutline: { borderColor: color.inputBorder } } },
      MuiButton: { styleOverrides: { root: { textTransform: 'none', fontWeight: 600, boxShadow: 'none', borderRadius: 7 } } },
      MuiTab: { styleOverrides: { root: { textTransform: 'none', minHeight: 44, fontWeight: 600 } } },
      MuiTabs: { styleOverrides: { root: { minHeight: 44 }, indicator: { height: 3, borderRadius: 3 } } },
      MuiTableCell: { styleOverrides: { head: { color: color.textSecondary, backgroundColor: color.surfaceMuted, fontWeight: 600 } } },
      MuiCssBaseline: {
        styleOverrides: {
          ':focus-visible': { outline: `3px solid ${color.primary}`, outlineOffset: 2 },
          code: { fontFamily: tokens.typography.monoFamily },
          dt: { color: color.textSecondary },
          '::selection': { backgroundColor: color.selectedHover, color: color.textPrimary },
          body: { fontVariantNumeric: 'tabular-nums', margin: 0, minWidth: 320 },
        },
      },
    },
  })
}

export const theme = createStudioTheme('light')
const themes = { light: theme, dark: createStudioTheme('dark') }

const MUI_LOCALES: Record<LanguageCode, Localization> = { en: enUS, nl: nlNL, it: itIT, fr: frFR, de: deDE, es: esES, zh: zhCN, ja: jaJP, ko: koKR }

/** The design tokens theme merged with MUI's own strings (pagination, …) for a language. */
export function themeFor(code: LanguageCode, mode: PaletteMode = 'light') {
  return createTheme(themes[mode], MUI_LOCALES[code])
}

/** Plotly layout defaults derived from the same tokens. */
export function plotLayoutFor(color: ReturnType<typeof colorsFor>) {
  return {
    font: { family: tokens.typography.fontFamily, size: tokens.typography.baseSize - 1, color: color.textPrimary },
    paper_bgcolor: color.surface,
    plot_bgcolor: color.surface,
    colorway: color.chart,
    hoverlabel: { bgcolor: color.surfaceMuted, bordercolor: color.border, font: { color: color.textPrimary } },
    modebar: { bgcolor: color.surface, color: color.iconActive, activecolor: color.primary },
    margin: { l: 56, r: 16, t: 38, b: 42 },
    legend: { orientation: 'h' as const, x: 0, xanchor: 'left', y: 1.04, yanchor: 'bottom', font: { size: 12 } },
    xaxis: { automargin: true, gridcolor: color.grid, zerolinecolor: color.zeroLine, title: { standoff: 8 } },
    yaxis: { automargin: true, gridcolor: color.grid, zerolinecolor: color.zeroLine, title: { standoff: 8 } },
  }
}
