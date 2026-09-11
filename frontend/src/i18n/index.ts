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
import en from './en.json'

export type MessageKey = keyof typeof en
export type Catalogue = Record<MessageKey, string>
export type LanguageCode = 'en' | 'fr' | 'de' | 'es' | 'zh' | 'ja' | 'ko'

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
  { code: 'fr', tag: 'fr', name: 'Français', flag: fr },
  { code: 'de', tag: 'de', name: 'Deutsch', flag: de },
  { code: 'es', tag: 'es', name: 'Español', flag: es },
  { code: 'zh', tag: 'zh-CN', name: '中文', flag: cn },
  { code: 'ja', tag: 'ja', name: '日本語', flag: jp },
  { code: 'ko', tag: 'ko', name: '한국어', flag: kr },
]
export const DEFAULT_LANGUAGE: LanguageCode = 'en'

// A catalogue missing a key of en.json fails to typecheck here.
const loaders: Record<Exclude<LanguageCode, 'en'>, () => Promise<{ default: Catalogue }>> = {
  fr: () => import('./fr.json'),
  de: () => import('./de.json'),
  es: () => import('./es.json'),
  zh: () => import('./zh.json'),
  ja: () => import('./ja.json'),
  ko: () => import('./ko.json'),
}

const catalogues: Partial<Record<LanguageCode, Catalogue>> = { en }
let current: LanguageCode = DEFAULT_LANGUAGE
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
  const loaded = (await loaders[code as Exclude<LanguageCode, 'en'>]()).default
  catalogues[code] = loaded
  return loaded
}

function notify(): void {
  document.documentElement.lang = languageInfo(current).tag
  for (const listener of listeners) listener()
}

/** Loads the catalogue if needed, then switches every `t()` call and re-renders subscribers. */
export async function setLanguage(code: LanguageCode): Promise<void> {
  await loadCatalogue(code)
  if (code === current) return
  current = code
  notify()
}

/** Back to English synchronously (English is always loaded); for tests. */
export function resetLanguage(): void {
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
