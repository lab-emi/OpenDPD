import { RouteError } from '@/components/RouteContent'
import CssBaseline from '@mui/material/CssBaseline'
import useMediaQuery from '@mui/material/useMediaQuery'
import { ThemeProvider } from '@mui/material/styles'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { lazy, useMemo, useState } from 'react'
import { BrowserRouter, HashRouter, Route, Routes } from 'react-router'
import { WEB_MODE } from '@/api/client'
import { useLanguage } from '@/i18n'
import { LanguageGate } from '@/i18n/LanguageGate'
import { AppShell } from '@/layout/AppShell'
import { HomePage } from '@/pages/HomePage'
import { StudioWorkflowProvider } from '@/workflow/StudioWorkflow'
import { SessionGate } from '@/pages/SessionGate'
import { themeFor } from '@/theme'

const DatasetDetailPage = lazy(() => import('@/pages/DatasetDetailPage').then(m => ({ default: m.DatasetDetailPage })))
const DatasetsPage = lazy(() => import('@/pages/DatasetsPage').then(m => ({ default: m.DatasetsPage })))
const ExperimentsPage = lazy(() => import('@/pages/ExperimentsPage').then(m => ({ default: m.ExperimentsPage })))
// Explicit browser-test builds retain fixtures; shipped builds exclude their module.
const GalleryPage = import.meta.env.DEV || import.meta.env.MODE === 'test-ui'
  ? lazy(() => import('@/pages/GalleryPage').then(m => ({ default: m.GalleryPage }))) : null
const SignalGeneratorPage = lazy(() => import('@/pages/SignalGeneratorPage').then(m => ({ default: m.SignalGeneratorPage })))
const SignalAnalyzerPage = lazy(() => import('@/pages/SignalAnalyzerPage').then(m => ({ default: m.SignalAnalyzerPage })))
const PALibraryPage = lazy(() => import('@/pages/PALibraryPage').then(m => ({ default: m.PALibraryPage })))
const NewExperimentPage = lazy(() => import('@/pages/NewExperimentPage').then(m => ({ default: m.NewExperimentPage })))
const ResultDetailPage = lazy(() => import('@/pages/ResultDetailPage').then(m => ({ default: m.ResultDetailPage })))
const ComparePage = lazy(() => import('@/pages/ComparePage').then(m => ({ default: m.ComparePage })))
const ResultsPage = lazy(() => import('@/pages/ResultsPage').then(m => ({ default: m.ResultsPage })))
const RunDetailPage = lazy(() => import('@/pages/RunDetailPage').then(m => ({ default: m.RunDetailPage })))
const RobustnessPage = lazy(() => import('@/pages/RobustnessPage').then(m => ({ default: m.RobustnessPage })))
const SweepBoardPage = lazy(() => import('@/pages/SweepBoardPage').then(m => ({ default: m.SweepBoardPage })))
const HardwareCostsPage = lazy(() => import('@/pages/HardwareCostsPage').then(m => ({ default: m.HardwareCostsPage })))
const SettingsPage = lazy(() => import('@/pages/SettingsPage').then(m => ({ default: m.SettingsPage })))
const AboutPage = lazy(() => import('@/pages/AboutPage').then(m => ({ default: m.AboutPage })))
const ServerStatusPage = lazy(() => import('@/pages/ServerStatusPage').then(m => ({ default: m.ServerStatusPage })))

function createQueryClient(): QueryClient {
  return new QueryClient({ defaultOptions: { queries: { retry: 1, staleTime: 10_000, refetchOnWindowFocus: true } } })
}

function AppRoutes() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route index element={<HomePage />} />
        <Route path="signal-generator/*" element={<SignalGeneratorPage />} />
        <Route path="signal-analyzer" element={<SignalAnalyzerPage />} />
        <Route path="pa-library" element={<PALibraryPage />} />
        <Route path="datasets" element={<DatasetsPage />} />
        <Route path="datasets/:datasetId" element={<DatasetDetailPage />} />
        <Route path="experiments" element={<ExperimentsPage />} />
        <Route path="experiments/new" element={<NewExperimentPage />} />
        <Route path="runs/:runId" element={<RunDetailPage />} />
        <Route path="results" element={<ResultsPage />} />
        <Route path="results/compare" element={<ComparePage />} />
        <Route path="results/:runId" element={<ResultDetailPage />} />
        <Route path="robustness" element={<RobustnessPage />} />
        <Route path="sweeps" element={<SweepBoardPage />} />
        <Route path="sweeps/:sweepId" element={<SweepBoardPage />} />
        <Route path="hardware" element={<HardwareCostsPage />} />
        <Route path="robustness/:planSha" element={<RobustnessPage />} />
        <Route path="settings" element={<SettingsPage />} />
        {GalleryPage && <Route path="gallery" element={<GalleryPage />} />}
        <Route path="about" element={<AboutPage />} />
        <Route path="server" element={<ServerStatusPage />} />
        <Route path="*" element={<HomePage />} />
      </Route>
    </Routes>
  )
}

export default function App({ queryClient }: { queryClient?: QueryClient }) {
  const Router = WEB_MODE ? HashRouter : BrowserRouter
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
        <Router>
          <SessionGate>
            <LanguageGate>
              <RouteError><StudioWorkflowProvider><AppRoutes /></StudioWorkflowProvider></RouteError>
            </LanguageGate>
          </SessionGate>
        </Router>
      </QueryClientProvider>
    </ThemeProvider>
  )
}
