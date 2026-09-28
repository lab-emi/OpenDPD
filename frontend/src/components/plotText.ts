/** Plotly interprets text as a small HTML dialect. Treat saved labels as text. */
export function escapePlotText(value: string): string {
  return value.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
}

function safeLabel(value: string): string {
  // Responsive legends use line breaks. Only bare breaks survive escaping.
  return escapePlotText(value).replace(/&lt;br\/?&gt;/g, '<br>')
}

export function safePlotLabels<T>(value: T): T {
  if (!value || typeof value !== 'object' || ArrayBuffer.isView(value)) return value
  if (Array.isArray(value)) return value.map((item) => safePlotLabels(item)) as T
  const result: Record<string, unknown> = {}
  for (const [key, item] of Object.entries(value)) {
    if (['x', 'y', 'z', 'customdata'].includes(key)) result[key] = item
    else if (key === 'hovertemplate' && typeof item === 'string') {
      // Preserve the application's line breaks and Plotly's extra box only.
      result[key] = escapePlotText(item).replace(/&lt;(br\/?|\/?extra)&gt;/g, '<$1>')
    }
    else if (['name', 'text', 'hovertext', 'title'].includes(key)) {
      result[key] = typeof item === 'string' ? safeLabel(item)
        : Array.isArray(item) ? item.map((entry) => typeof entry === 'string' ? safeLabel(entry) : entry)
          : safePlotLabels(item)
    }
    else result[key] = safePlotLabels(item)
  }
  return result as T
}
