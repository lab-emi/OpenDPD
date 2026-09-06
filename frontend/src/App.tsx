import CssBaseline from '@mui/material/CssBaseline'
import { ThemeProvider } from '@mui/material/styles'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { BrowserRouter, Route, Routes } from 'react-router'
import { AppShell } from '@/layout/AppShell'
import { DatasetDetailPage } from '@/pages/DatasetDetailPage'
import { DatasetsPage } from '@/pages/DatasetsPage'
import { ExperimentsPage } from '@/pages/ExperimentsPage'
import { GalleryPage } from '@/pages/GalleryPage'
import { HomePage } from '@/pages/HomePage'
import { NewExperimentPage } from '@/pages/NewExperimentPage'
import { ResultDetailPage } from '@/pages/ResultDetailPage'
import { ResultsPage } from '@/pages/ResultsPage'
import { RunDetailPage } from '@/pages/RunDetailPage'
import { SessionGate } from '@/pages/SessionGate'
import { SettingsPage } from '@/pages/SettingsPage'
import { theme } from '@/theme'

export function createQueryClient(): QueryClient {
  return new QueryClient({ defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: true } } })
}

export function AppRoutes() {
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
        <Route path="results/:runId" element={<ResultDetailPage />} />
        <Route path="settings" element={<SettingsPage />} />
        <Route path="gallery" element={<GalleryPage />} />
        <Route path="*" element={<HomePage />} />
      </Route>
    </Routes>
  )
}

export default function App({ queryClient = createQueryClient() }: { queryClient?: QueryClient }) {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <SessionGate>
            <AppRoutes />
          </SessionGate>
        </BrowserRouter>
      </QueryClientProvider>
    </ThemeProvider>
  )
}
