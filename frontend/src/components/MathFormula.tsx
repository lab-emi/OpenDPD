import { useLayoutEffect, useMemo, useRef } from 'react'
import Box from '@mui/material/Box'
import { renderToString } from 'katex'
import 'katex/dist/katex.min.css'
import { useStudioColors } from '@/theme'

interface Variable { key: string; symbol: string; label: string }
/** Only catalog parameter classes are trusted; links, HTML styles and resource loading stay disabled. */
export function MathFormula({ latex, variables = [], active, onSelect, display = false }: {
  latex: string; variables?: Variable[]; active?: string; onSelect?: (key: string) => void; display?: boolean
}) {
  const ref = useRef<HTMLSpanElement>(null)
  const colors = useStudioColors()
  const signature = JSON.stringify(variables)
  const html = useMemo(() => {
    const known: Variable[] = JSON.parse(signature)
    const source = latex.replace(/\{\{(\w+)\}\}/g, (_, key: string) => {
      const v = known.find(item => item.key === key)
      return v ? `\\htmlClass{pa-var-${key}}{${v.symbol}}` : '\\text{?}'
    })
    return renderToString(source, { displayMode: display, throwOnError: false, maxExpand: 300, maxSize: 10,
      strict: 'ignore', trust: ctx => ctx.command === '\\htmlClass' && known.some(v => ctx.class === `pa-var-${v.key}`) })
  }, [latex, signature, display])
  useLayoutEffect(() => {
    if (!html || !ref.current) return
    const known: Variable[] = JSON.parse(signature)
    for (const v of known) ref.current?.querySelectorAll<HTMLElement>(`.pa-var-${v.key}`).forEach((el, index) => {
      el.dataset.variable = v.key; el.dataset.active = String(active === v.key)
      // KaTeX's visual tree is aria-hidden. Expose each interactive coefficient separately.
      el.setAttribute('role', 'button'); el.tabIndex = 0; el.setAttribute('aria-label', v.label)
      el.setAttribute('aria-pressed', String(active === v.key))
      if (index === 0) el.dataset.testid = 'equation-' + v.key
    })
    const visual = ref.current?.querySelector('.katex-html')
    if (known.length) visual?.removeAttribute('aria-hidden')
  }) // React may replace innerHTML on a parent render; reattach interaction metadata each commit.
  const select = (target: EventTarget) => {
    if (!(target instanceof Element)) return
    const key = target.closest<HTMLElement>('[data-variable]')?.dataset.variable
    if (key && variables.some(v => v.key === key)) onSelect?.(key)
  }
  return <Box component="span" ref={ref} onClick={e => select(e.target)} onKeyDown={e => {
    if ((e.key === 'Enter' || e.key === ' ') && (e.target as HTMLElement).dataset.variable) { e.preventDefault(); select(e.target) }
  }} sx={{ display: display ? 'block' : 'inline-block', maxWidth: '100%', overflowX: 'auto', verticalAlign: 'middle',
    '& .katex-display': { textAlign: 'left', my: 1.5 }, '& .katex-display > .katex': { textAlign: 'left' },
    '& [data-variable]': { cursor: 'pointer', borderRadius: .5, outlineOffset: 2 },
    '& [data-active="true"]': { color: colors.primary, bgcolor: colors.selected, outline: `1px solid ${colors.primary}` },
    '& [data-variable]:focus-visible': { outline: `2px solid ${colors.primary}` },
  }} dangerouslySetInnerHTML={{ __html: html }} />
}
