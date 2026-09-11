import { useQuery } from '@tanstack/react-query'
import type { ReactNode } from 'react'
import { useSettings } from '@/api/hooks'
import { ErrorState, LoadingState } from '@/components/StateBlock'
import { resolveLanguage, setLanguage } from '@/i18n'

/**
 * Applies the workspace's language (or the browser's) before the workbench
 * renders. Keyed by the stored value: a refetch returning the same value never
 * re-applies it, so a choice whose save failed is not silently reverted.
 */
export function LanguageGate({ children }: { children: ReactNode }) {
  const settings = useSettings()
  const stored = settings.data?.language ?? null
  const applied = useQuery({
    queryKey: ['language', stored],
    queryFn: async () => {
      const code = resolveLanguage(stored)
      await setLanguage(code)
      return code
    },
    enabled: settings.isSuccess,
    staleTime: Infinity,
    gcTime: Infinity,
  })
  if (settings.isError) return <ErrorState error={settings.error} onRetry={() => void settings.refetch()} />
  if (applied.isError) return <ErrorState error={applied.error} onRetry={() => void applied.refetch()} />
  if (!applied.isSuccess) return <LoadingState />
  return <>{children}</>
}
