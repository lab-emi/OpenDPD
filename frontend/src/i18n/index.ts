/**
 * User-facing strings and the active UI language.
 *
 * `t()` stays synchronous: it reads the catalogue of the active language and
 * falls back to English per key. Non-English catalogues are lazy chunks. The
 * active language is a tiny external store so `App` can re-render the route
 * tree (and pick the MUI locale pack) when it changes.
 */
import { useSyncExternalStore } from 'react'
import cn from '@/assets/flags/cn.svg'
import de from '@/assets/flags/de.svg'
import es from '@/assets/flags/es.svg'
import fr from '@/assets/flags/fr.svg'
import gb from '@/assets/flags/gb.svg'
import jp from '@/assets/flags/jp.svg'
import kr from '@/assets/flags/kr.svg'
import nl from '@/assets/flags/nl.svg'
import it from '@/assets/flags/it.svg'
import en from './en.json'
import enMessages from '../../../opendpd/studio/locales/en.json'

export type MessageKey = keyof typeof en
export type Catalogue = Record<MessageKey, string>
export type LanguageCode = 'en' | 'nl' | 'zh' | 'fr' | 'de' | 'it' | 'ja' | 'ko' | 'es'

export interface Language {
  code: LanguageCode
  /** BCP-47 tag for `<html lang>` and Intl formatting. */
  tag: string
  /** Native name; never translated. */
  name: string
  /** Flag image URL (bundled SVG). */
  flag: string
}

export const LANGUAGES: readonly Language[] = [
  { code: 'en', tag: 'en', name: 'English', flag: gb },
  { code: 'nl', tag: 'nl', name: 'Nederlands', flag: nl },
  { code: 'zh', tag: 'zh-CN', name: '中文', flag: cn },
  // After the preferred three, sort by English language name (stable across locales).
  { code: 'fr', tag: 'fr', name: 'Français', flag: fr },
  { code: 'de', tag: 'de', name: 'Deutsch', flag: de },
  { code: 'it', tag: 'it', name: 'Italiano', flag: it },
  { code: 'ja', tag: 'ja', name: '日本語', flag: jp },
  { code: 'ko', tag: 'ko', name: '한국어', flag: kr },
  { code: 'es', tag: 'es', name: 'Español', flag: es },
]
export const DEFAULT_LANGUAGE: LanguageCode = 'en'

// A catalogue missing a key of en.json fails to typecheck here.
const loaders: Record<Exclude<LanguageCode, 'en'>, () => Promise<{ default: Catalogue }>> = {
  nl: () => import('./nl.json'),
  it: () => import('./it.json'),
  fr: () => import('./fr.json'),
  de: () => import('./de.json'),
  es: () => import('./es.json'),
  zh: () => import('./zh.json'),
  ja: () => import('./ja.json'),
  ko: () => import('./ko.json'),
}

const catalogues: Partial<Record<LanguageCode, Catalogue>> = { en }
type Messages = Record<string, string>
const messages: Partial<Record<LanguageCode, Messages>> = { en: enMessages }
const messageLoaders = import.meta.glob<{ default: Messages }>(['../../../opendpd/studio/locales/*.json', '!../../../opendpd/studio/locales/en.json'])
let current: LanguageCode = DEFAULT_LANGUAGE
let selection = 0
const listeners = new Set<() => void>()

export function isLanguageCode(value: unknown): value is LanguageCode {
  return LANGUAGES.some((l) => l.code === value)
}

export function getLanguage(): LanguageCode {
  return current
}

export function languageInfo(code: LanguageCode = current): Language {
  return LANGUAGES.find((l) => l.code === code) ?? LANGUAGES[0]!
}

/** Stored choice → first supported browser language → English. */
export function resolveLanguage(stored: string | null | undefined, navigatorLanguages: readonly string[] = navigator.languages ?? [navigator.language]): LanguageCode {
  if (isLanguageCode(stored)) return stored
  for (const tag of navigatorLanguages) {
    const base = tag.toLowerCase().split('-')[0]
    if (isLanguageCode(base)) return base
  }
  return DEFAULT_LANGUAGE
}

export async function loadCatalogue(code: LanguageCode): Promise<Catalogue> {
  const cached = catalogues[code]
  if (cached) return cached
  const [ui, generated] = await Promise.all([
    loaders[code as Exclude<LanguageCode, 'en'>](),
    messageLoaders[`../../../opendpd/studio/locales/${code}.json`]!(),
  ])
  const loaded = ui.default
  messages[code] = generated.default
  catalogues[code] = loaded
  return loaded
}

function notify(): void {
  document.documentElement.lang = languageInfo(current).tag
  for (const listener of listeners) listener()
}

/** Loads the catalogue if needed, then switches every `t()` call and re-renders subscribers. */
export async function setLanguage(code: LanguageCode): Promise<void> {
  const request = ++selection
  await loadCatalogue(code)
  if (request !== selection) return
  if (code === current) return
  current = code
  notify()
}

/** Back to English synchronously (English is always loaded); for tests. */
export function resetLanguage(): void {
  selection++
  if (current === DEFAULT_LANGUAGE) return
  current = DEFAULT_LANGUAGE
  notify()
}

function subscribe(listener: () => void): () => void {
  listeners.add(listener)
  return () => listeners.delete(listener)
}

export function useLanguage(): LanguageCode {
  return useSyncExternalStore(subscribe, getLanguage, getLanguage)
}

export function t(key: MessageKey, vars: Record<string, string | number> = {}): string {
  const template: string = catalogues[current]?.[key] ?? en[key]
  return template.replace(/\{(\w+)\}/g, (_, name: string) => String(vars[name] ?? `{${name}}`))
}

export function formatNumber(value: number, options?: Intl.NumberFormatOptions): string {
  return value.toLocaleString(languageInfo().tag, options)
}

export function formatDateTime(value: string | number | Date): string {
  return new Date(value).toLocaleString(languageInfo().tag)
}

export function formatTime(value: string | number | Date): string {
  return new Date(value).toLocaleTimeString(languageInfo().tag)
}

const escapePattern = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
const templates = Object.keys(enMessages).filter((key) => /\{\w+\}/.test(key)).sort((a, b) => b.length - a.length).map((key) => {
  const names = [...key.matchAll(/\{(\w+)\}/g)].map((match) => match[1]!)
  const pattern = key.split(/\{\w+\}/).map(escapePattern).join('([\\s\\S]+?)')
  return { key, names, pattern: new RegExp(`^${pattern}$`) }
})
const uiText = new Map(Object.entries(en).map(([key, value]) => [value as string, key as MessageKey]))

/** Presentation only: translate registered application prose, preserving unknown source text and numbers. */
export function message(value: string | null | undefined): string {
  if (!value) return ''
  const text = value.trim()
  const table = messages[current] ?? enMessages as Messages
  if (table[text] !== undefined) return table[text]!
  const key = uiText.get(text)
  if (key) return t(key)
  for (const template of templates) {
    const match = template.pattern.exec(text)
    if (!match) continue
    const vars = Object.fromEntries(template.names.map((name, i) => [name, match[i + 1]!]))
    return (table[template.key] ?? template.key).replace(/\{(\w+)\}/g, (_, name: string) => vars[name] ?? `{${name}}`)
  }
  // Inspection methods join a version identifier, a formula/finding and notes.
  // Only these application-owned envelopes are split; arbitrary source text is untouched.
  const method = /^(dataset-doctor-v1|general-spectral-v1|ofdm-lte20-evm-v1): ([\s\S]+)$/.exec(text)
  if (method) {
    const body = method[2]!, boundary = body.indexOf('. ')
    return `${method[1]}: ${boundary < 0 ? message(body) : `${message(body.slice(0, boundary))}. ${message(body.slice(boundary + 2))}`}`
  }
  return value
}

export function phaseLabel(phase: string): string {
  const labels: Record<string, string> = { train: 'Training', val: 'Validation', validation: 'Validation', test: 'Test', evaluate: 'Evaluation', train_probe: 'Training preview', validation_probe: 'Validation preview', val_probe: 'Validation preview', test_probe: 'Test preview', init: 'Initializing', prepare: 'Initializing', finalize: 'Finalizing', complete: 'Completed' }
  return message(labels[phase] ?? phase)
}

/** Only the application's default built-in captions are translated; custom names stay exact. */
export function datasetLabel(dataset: { display_name: string; source: { kind: string; name?: string | null } }): string {
  const { display_name: name, source } = dataset
  return source.kind === 'builtin' && source.name && name.startsWith(`${source.name} (built-in, `) ? message(name) : name
}
