import { expect, test } from 'vitest'
import { safePlotLabels } from './plotText'
import { message } from '@/i18n'

test('saved labels cannot create Plotly links or markup', () => {
  const x = new Float32Array([1, 2])
  const value = { x, name: '<a href="https://example.invalid">data</a>', xaxis: { title: { text: '<b>Voltage</b>&' } } }
  const result = safePlotLabels(value)
  expect(result.name).toBe('&lt;a href="https://example.invalid"&gt;data&lt;/a&gt;')
  expect(result.xaxis.title.text).toBe('&lt;b&gt;Voltage&lt;/b&gt;&amp;')
  expect(result.x).toBe(x)
  expect(value.name).toContain('<a')
})

test('long untrusted labels bypass translation template matching', () => {
  const value = 'x '.repeat(8000)
  expect(message(value)).toBe(value)
})

test('hover templates retain line breaks while refusing links', () => {
  expect(safePlotLabels({ hovertemplate: '%{x}<br>%{y}<extra><a href="https://example.invalid">click</a></extra>' }).hovertemplate)
    .toBe('%{x}<br>%{y}<extra>&lt;a href="https://example.invalid"&gt;click&lt;/a&gt;</extra>')
})

test('responsive legend breaks remain usable without accepting attributes or links', () => {
  const result = safePlotLabels({ name: 'With DPD · <br>PA model',
    text: ['Line 1<br/>Line 2', '<br onclick="alert(1)">', '&lt;br&gt;', '<a href="https://example.invalid">link</a>'] })
  expect(result.name).toBe('With DPD · <br>PA model')
  expect(result.text).toEqual(['Line 1<br>Line 2', '&lt;br onclick="alert(1)"&gt;',
    '&amp;lt;br&amp;gt;', '&lt;a href="https://example.invalid"&gt;link&lt;/a&gt;'])
})
