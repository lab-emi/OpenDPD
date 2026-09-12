import CssBaseline from '@mui/material/CssBaseline'
import useMediaQuery from '@mui/material/useMediaQuery'
import { ThemeProvider } from '@mui/material/styles'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useMemo, useState } from 'react'
import { BrowserRouter, Route, Routes } from 'react-router'
import { useLanguage } from '@/i18n'
import { LanguageGate } from '@/i18n/LanguageGate'
import { AppShell } from '@/layout/AppShell'
import { DatasetDetailPage } from '@/pages/DatasetDetailPage'
import { DatasetsPage } from '@/pages/DatasetsPage'
import { ExperimentsPage } from '@/pages/ExperimentsPage'
import { GalleryPage } from '@/pages/GalleryPage'
import { HomePage } from '@/pages/HomePage'
import { NewExperimentPage } from '@/pages/NewExperimentPage'
import { ResultDetailPage } from '@/pages/ResultDetailPage'
import { ComparePage } from '@/pages/ComparePage'
import { ResultsPage } from '@/pages/ResultsPage'
import { RunDetailPage } from '@/pages/RunDetailPage'
import { SessionGate } from '@/pages/SessionGate'
import { RobustnessPage } from '@/pages/RobustnessPage'
import { SettingsPage } from '@/pages/SettingsPage'
import { AboutPage } from '@/pages/AboutPage'
import { themeFor } from '@/theme'

function createQueryClient(): QueryClient {
  return new QueryClient({ defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: true } } })
}

function AppRoutes() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<HomePage />} />
        <Route path="datasets" element={<DatasetsPage />} />
        <Route path="datasets/:datasetId" element={<DatasetDetailPage />} />
        <Route path="experiments" element={<ExperimentsPage />} />
        <Route path="experiments/new" element={<NewExperimentPage />} />
        <Route path="runs/:runId" element={<RunDetailPage />} />
        <Route path="results" element={<ResultsPage />} />
        <Route path="results/compare" element={<ComparePage />} />
        <Route path="results/:runId" element={<ResultDetailPage />} />
        <Route path="robustness" element={<RobustnessPage />} />
        <Route path="robustness/:planSha" element={<RobustnessPage />} />
        <Route path="settings" element={<SettingsPage />} />
        <Route path="gallery" element={<GalleryPage />} />
        <Route path="about" element={<AboutPage />} />
        <Route path="*" element={<HomePage />} />
      </Route>
    </Routes>
  )
}

export default function App({ queryClient }: { queryClient?: QueryClient }) {
  // Theme/locale changes must keep the query cache, subscriptions and active forms.
  const [defaultClient] = useState(createQueryClient)
  // A language change re-renders from here: new route elements for every page (state kept) and the matching MUI locale.
  const language = useLanguage()
  const dark = useMediaQuery('(prefers-color-scheme: dark)')
  const muiTheme = useMemo(() => themeFor(language, dark ? 'dark' : 'light'), [language, dark])
  return (
    <ThemeProvider theme={muiTheme}>
      <CssBaseline enableColorScheme />
      <QueryClientProvider client={queryClient ?? defaultClient}>
        <BrowserRouter>
          <SessionGate>
            <LanguageGate>
              <AppRoutes />
            </LanguageGate>
          </SessionGate>
        </BrowserRouter>
      </QueryClientProvider>
    </ThemeProvider>
  )
}
