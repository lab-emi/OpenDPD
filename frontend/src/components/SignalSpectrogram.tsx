import FullscreenIcon from '@mui/icons-material/Fullscreen'
import CloseIcon from '@mui/icons-material/Close'
import Box from '@mui/material/Box'
import Button from '@mui/material/Button'
import Dialog from '@mui/material/Dialog'
import DialogContent from '@mui/material/DialogContent'
import DialogTitle from '@mui/material/DialogTitle'
import IconButton from '@mui/material/IconButton'
import Slider from '@mui/material/Slider'
import Stack from '@mui/material/Stack'
import Typography from '@mui/material/Typography'
import { useEffect, useRef, useState } from 'react'
import type { SignalAnalysis } from '@/api/signalAnalyzer'
import { t } from '@/i18n'

const color = (v: number) => [Math.round(15 + 238 * v ** 2), Math.round(25 + 200 * Math.sin(v * Math.PI / 2)), Math.round(60 + 95 * (1 - v) * Math.sin(v * Math.PI))]
const gradient = 'linear-gradient(90deg,' + Array.from({ length: 33 }, (_, i) => `rgb(${color(i / 32).join(',')}) ${i / 32 * 100}%`).join(',') + ')'

function SpectrogramView({ result }: { result: SignalAnalysis }) {
  const ref = useRef<HTMLCanvasElement>(null)
  const matrix = result.spectrogram_dbfs_hz
  const max = Math.max(...matrix.map(row => Math.max(...row)))
  const [dynamicRange, setDynamicRange] = useState(70)
  const [point, setPoint] = useState<string>('')
  const times = result.spectrogram_time_s
  const frequencies = result.spectrogram_frequency_hz
  useEffect(() => {
    const canvas = ref.current
    if (!canvas || !matrix.length) return
    canvas.width = frequencies.length; canvas.height = times.length
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    const pixels = ctx.createImageData(canvas.width, canvas.height)
    for (let row = 0; row < matrix.length; row++) for (let col = 0; col < matrix[row]!.length; col++) {
      const v = Math.max(0, Math.min(1, (matrix[row]![col]! - max + dynamicRange) / dynamicRange))
      const index = 4 * ((matrix.length - 1 - row) * canvas.width + col)
      const rgb = color(v)
      pixels.data[index] = rgb[0]!
      pixels.data[index + 1] = rgb[1]!
      pixels.data[index + 2] = rgb[2]!
      pixels.data[index + 3] = 255
    }
    ctx.putImageData(pixels, 0, 0)
  }, [matrix, frequencies, times, max, dynamicRange])
  return <Stack spacing={1}>
    <Box sx={{ display: 'grid', gridTemplateColumns: '42px minmax(0, 1fr)', gap: 1 }}>
      <Stack sx={{ justifyContent: 'space-between', alignItems: 'center' }}><Typography variant="caption">{((times.at(-1) ?? 0) * 1e6).toPrecision(4)}</Typography><Typography variant="caption" sx={{ writingMode: 'vertical-rl', transform: 'rotate(180deg)' }}>{t('analyzer.timeAxis')}</Typography><Typography variant="caption">{((times[0] ?? 0) * 1e6).toPrecision(3)}</Typography></Stack>
      <canvas ref={ref} role="img" aria-label={t('analyzer.spectrogramHelp')} style={{ width: '100%', height: 290, imageRendering: 'pixelated', borderRadius: 4 }} onPointerMove={e => {
        const bounds = e.currentTarget.getBoundingClientRect()
        const col = Math.max(0, Math.min(frequencies.length - 1, Math.floor((e.clientX - bounds.left) / bounds.width * frequencies.length)))
        const row = Math.max(0, Math.min(times.length - 1, times.length - 1 - Math.floor((e.clientY - bounds.top) / bounds.height * times.length)))
        setPoint(`${((frequencies[col] ?? 0) / 1e6).toPrecision(6)} MHz · ${((times[row] ?? 0) * 1e6).toPrecision(6)} µs · ${(matrix[row]?.[col] ?? 0).toFixed(2)} dBFS/Hz`)
      }} />
    </Box>
    <Stack direction="row" sx={{ justifyContent: 'space-between', pl: '50px' }}><Typography variant="caption">{((frequencies[0] ?? 0) / 1e6).toPrecision(4)}</Typography><Typography variant="caption">{((frequencies.at(-1) ?? 0) / 1e6).toPrecision(4)}</Typography></Stack>
    <Typography variant="caption" sx={{ textAlign: 'center', pl: '50px' }}>{t('generator.frequencyAxis')}</Typography>
    <Typography variant="caption" sx={{ minHeight: 20 }}>{point || t('analyzer.spectrogramHelp')}</Typography>
    <Stack direction="row" spacing={2} sx={{ alignItems: 'center', px: 1 }}><Typography variant="caption" sx={{ flexShrink: 0 }}>{t('analyzer.dynamicRange')} · {dynamicRange} dB</Typography><Slider min={20} max={120} value={dynamicRange} onChange={(_, value) => setDynamicRange(value as number)} aria-label={t('analyzer.dynamicRange')} /></Stack>
    <Box sx={{ height: 10, borderRadius: .5, background: gradient }} /><Stack direction="row" sx={{ justifyContent: 'space-between' }}><Typography variant="caption">{(max - dynamicRange).toFixed(1)} dBFS/Hz</Typography><Typography variant="caption">{max.toFixed(1)} dBFS/Hz</Typography></Stack>
  </Stack>
}

export function SignalSpectrogram({ result }: { result: SignalAnalysis }) {
  const [expanded, setExpanded] = useState(false)
  return <><Stack direction="row" sx={{ alignItems: 'center', justifyContent: 'space-between', mb: 1 }}><Typography variant="h3">{t('analyzer.spectrogram')}</Typography><Button aria-label={t('analyzer.expandSpectrogram')} startIcon={<FullscreenIcon />} size="small" onClick={() => setExpanded(true)}>{t('analyzer.expand')}</Button></Stack><SpectrogramView result={result} />
    <Dialog open={expanded} onClose={() => setExpanded(false)} fullWidth maxWidth="lg"><DialogTitle>{t('analyzer.spectrogram')}<IconButton aria-label={t('common.close')} onClick={() => setExpanded(false)} sx={{ float: 'right' }}><CloseIcon /></IconButton></DialogTitle><DialogContent>{expanded && <SpectrogramView result={result} />}</DialogContent></Dialog>
  </>
}
