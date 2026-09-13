import Button from '@mui/material/Button'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import { useQuery } from '@tanstack/react-query'
import { api } from '@/api/client'
import { isTerminal, type RunView } from '@/api/types'
import type { components } from '@/api/schema'
import { t } from '@/i18n'
import { DownloadLink } from './DownloadLink'

export function ModelDownloadButton({ run }: { run: RunView }) {
  const model = useQuery({
    queryKey: ['run', run.run_id, 'checkpoint', run.status],
    queryFn: () => api.get<components['schemas']['ModelDownloadInfo']>(`/runs/${encodeURIComponent(run.run_id)}/checkpoint`),
    refetchInterval: isTerminal(run.status) ? false : 5000,
  })
  const available = model.data?.available && model.data.download_url
  return <Stack spacing={.75} sx={{ mt: 2 }} data-testid="model-download">
    {available ? <DownloadLink button variant="outlined" size="medium" href={model.data!.download_url!} sx={{ alignSelf: 'flex-start' }}>
      {t(model.data!.final ? 'model.download.final' : 'model.download.checkpoint')}
    </DownloadLink> : <Button variant="outlined" disabled sx={{ alignSelf: 'flex-start' }}>{t('model.download.checkpoint')}</Button>}
    <Typography variant="caption" color={model.isError ? 'error' : 'text.secondary'}>{t(model.isError ? 'model.download.error' : available ? 'model.download.help' : 'model.download.waiting')}</Typography>
  </Stack>
}
