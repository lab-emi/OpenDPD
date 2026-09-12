import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import FormControlLabel from '@mui/material/FormControlLabel'
import Stack from '@mui/material/Stack'
import Switch from '@mui/material/Switch'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { fetchLogPage } from '@/api/hooks'
import { ApiError, WEB_MODE } from '@/api/client'
import { message, t } from '@/i18n'
import { tokens, useStudioColors } from '@/theme'

const ROW_HEIGHT = 20
const PAGE = 500
const MAX_LINES = 50_000

/** Paged (byte offset) and windowed log view; only visible rows are in the DOM. */
export function LogViewer(props: { runId: string; live: boolean; height?: number; tail?: boolean }) {
  return <LogContent key={props.runId} {...props} />
}

function LogContent({ runId, live, height = 420, tail = false }: { runId: string; live: boolean; height?: number; tail?: boolean }) {
  const colors = useStudioColors()
  const [lines, setLines] = useState<string[]>([])
  const [offset, setOffset] = useState(0)
  const [eof, setEof] = useState(true)
  const [filter, setFilter] = useState('')
  const [follow, setFollow] = useState(true)
  const [scrollTop, setScrollTop] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const boxRef = useRef<HTMLDivElement>(null)
  const loadingRef = useRef(false)
  const finalLoadedRef = useRef(false)
  const retryAtRef = useRef(0)

  const loadMore = useCallback(async () => {
    if (loadingRef.current || Date.now() < retryAtRef.current) return
    loadingRef.current = true
    try {
      const page = await fetchLogPage(runId, offset, PAGE, tail && offset === 0)
      setOffset(page.next_offset)
      setEof(page.eof)
      if (page.lines.length > 0) setLines((prev) => [...prev, ...page.lines].slice(-MAX_LINES))
      setError(null)
      finalLoadedRef.current = !live && page.eof
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
      if (err instanceof ApiError) retryAtRef.current = Date.now() + err.retryAfterMs
    } finally {
      loadingRef.current = false
    }
  }, [runId, offset, tail, live])

  /** Fetch every remaining page (2000 lines each) so a long log can be searched; the view keeps MAX_LINES. */
  const loadAll = useCallback(async () => {
    if (loadingRef.current) return
    loadingRef.current = true
    try {
      let next = offset
      let done = eof
      const batch: string[] = []
      while (!done) {
        const page = await fetchLogPage(runId, next, 2000)
        if (page.lines.length === 0 && page.next_offset === next) break
        next = page.next_offset
        done = page.eof
        batch.push(...page.lines)
      }
      setOffset(next)
      setEof(done)
      if (batch.length > 0) setLines((prev) => [...prev, ...batch].slice(-MAX_LINES))
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      loadingRef.current = false
    }
  }, [runId, offset, eof])

  useEffect(() => {
    void loadMore()
    // initial page only; later pages are explicit or timer driven
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId, live])

  useEffect(() => {
    const refresh = () => {
      if (document.visibilityState !== 'hidden' && (live || !finalLoadedRef.current)) void loadMore()
    }
    const timer = window.setInterval(refresh, WEB_MODE ? 3000 : 2000)
    window.addEventListener('online', refresh)
    document.addEventListener('visibilitychange', refresh)
    return () => {
      window.clearInterval(timer)
      window.removeEventListener('online', refresh)
      document.removeEventListener('visibilitychange', refresh)
    }
  }, [live, loadMore])

  const visible = useMemo(() => (filter ? lines.filter((l) => l.toLowerCase().includes(filter.toLowerCase())) : lines), [lines, filter])

  useEffect(() => {
    if (follow && boxRef.current) boxRef.current.scrollTop = boxRef.current.scrollHeight
  }, [visible.length, follow])

  const first = Math.max(0, Math.floor(scrollTop / ROW_HEIGHT) - 5)
  const count = Math.ceil(height / ROW_HEIGHT) + 10
  const slice = visible.slice(first, first + count)

  return (
    <Stack spacing={1} data-testid="log-viewer">
      <Stack sx={{ alignItems: 'center', flexWrap: 'wrap' }} direction="row" spacing={2} useFlexGap>
        <TextField label={t('logs.search')} value={filter} onChange={(e) => setFilter(e.target.value)} sx={{ width: 240, maxWidth: '100%' }} />
        <FormControlLabel control={<Switch checked={follow} onChange={(e) => setFollow(e.target.checked)} />} label={t('logs.follow')} />
        <Typography variant="caption" color="text.secondary">
          {t('logs.lines', { shown: visible.length, total: lines.length })}
        </Typography>
        <Button size="small" onClick={() => void loadMore()}>{t('logs.refresh')}</Button>
        {!eof && (
          <Button size="small" onClick={() => void loadMore()}>
            {t('logs.loadMore')}
          </Button>
        )}
        {!eof && (
          <Button size="small" onClick={() => void loadAll()}>
            {t('logs.loadAll')}
          </Button>
        )}
        {lines.length >= MAX_LINES && (
          <Typography variant="caption" color="text.secondary">
            {t('logs.capped', { max: MAX_LINES })}
          </Typography>
        )}
      </Stack>
      {error && <Typography color="error">{message(error)}</Typography>}
      <Typography variant="caption" color="text.secondary">{t('logs.original')}</Typography>
      <Box
        ref={boxRef}
        role="log"
        aria-live={live ? 'polite' : 'off'}
        tabIndex={0}
        onScroll={(e) => setScrollTop((e.target as HTMLDivElement).scrollTop)}
        sx={{ height, overflow: 'auto', bgcolor: colors.terminal.background, color: colors.terminal.text, colorScheme: 'inherit', borderRadius: 1, fontFamily: tokens.typography.monoFamily, fontSize: 12 }}
      >
        {visible.length === 0 ? (
          <Typography sx={{ p: 2, color: colors.terminal.muted }}>{!live && eof && lines.length === 0 ? t('logs.none.stored') : t('logs.empty')}</Typography>
        ) : (
          <div style={{ height: visible.length * ROW_HEIGHT, position: 'relative' }}>
            {slice.map((line, i) => (
              <div key={first + i} style={{ position: 'absolute', top: (first + i) * ROW_HEIGHT, height: ROW_HEIGHT, left: 0, right: 0, whiteSpace: 'pre', paddingLeft: 8, lineHeight: `${ROW_HEIGHT}px` }}>
                {line}
              </div>
            ))}
          </div>
        )}
      </Box>
    </Stack>
  )
}
