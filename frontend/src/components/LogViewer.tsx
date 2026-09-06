import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import FormControlLabel from '@mui/material/FormControlLabel'
import Stack from '@mui/material/Stack'
import Switch from '@mui/material/Switch'
import TextField from '@mui/material/TextField'
import Typography from '@mui/material/Typography'
import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { fetchLogPage } from '@/api/hooks'
import { t } from '@/i18n'
import { tokens } from '@/theme'

const ROW_HEIGHT = 20
const PAGE = 500
const MAX_LINES = 50_000

/** Paged (byte offset) and windowed log view; only visible rows are in the DOM. */
export function LogViewer({ runId, live, height = 420 }: { runId: string; live: boolean; height?: number }) {
  const [lines, setLines] = useState<string[]>([])
  const [offset, setOffset] = useState(0)
  const [eof, setEof] = useState(true)
  const [filter, setFilter] = useState('')
  const [follow, setFollow] = useState(true)
  const [scrollTop, setScrollTop] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const boxRef = useRef<HTMLDivElement>(null)
  const loadingRef = useRef(false)

  const loadMore = useCallback(async () => {
    if (loadingRef.current) return
    loadingRef.current = true
    try {
      const page = await fetchLogPage(runId, offset, PAGE)
      setOffset(page.next_offset)
      setEof(page.eof)
      if (page.lines.length > 0) setLines((prev) => [...prev, ...page.lines].slice(-MAX_LINES))
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      loadingRef.current = false
    }
  }, [runId, offset])

  useEffect(() => {
    void loadMore()
    // initial page only; later pages are explicit or timer driven
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runId])

  useEffect(() => {
    if (!live) return
    const timer = window.setInterval(() => void loadMore(), 2000)
    return () => window.clearInterval(timer)
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
        <TextField label={t('logs.search')} value={filter} onChange={(e) => setFilter(e.target.value)} sx={{ minWidth: 240 }} />
        <FormControlLabel control={<Switch checked={follow} onChange={(e) => setFollow(e.target.checked)} />} label={t('logs.follow')} />
        <Typography variant="caption" color="text.secondary">
          {t('logs.lines', { shown: visible.length, total: lines.length })}
        </Typography>
        {!eof && (
          <Button size="small" onClick={() => void loadMore()}>
            {t('logs.loadMore')}
          </Button>
        )}
      </Stack>
      {error && <Typography color="error">{error}</Typography>}
      <Box
        ref={boxRef}
        role="log"
        aria-live={live ? 'polite' : 'off'}
        tabIndex={0}
        onScroll={(e) => setScrollTop((e.target as HTMLDivElement).scrollTop)}
        sx={{ height, overflow: 'auto', bgcolor: '#0f172a', color: '#e2e8f0', borderRadius: 1, fontFamily: tokens.typography.monoFamily, fontSize: 12 }}
      >
        {visible.length === 0 ? (
          <Typography sx={{ p: 2, color: '#94a3b8' }}>{!live && eof && lines.length === 0 ? t('logs.none.stored') : t('logs.empty')}</Typography>
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
